#!/usr/bin/env python
"""
Phase-3b: integrate (scVI, batch=sample) + annotate compartments with STRICT
osteoclast identity (separating true OC lineage from CTSK+ stroma/osteoblast),
then export the OC-lineage subset for discovery. Writes integrated.h5ad,
oc_lineage.h5ad, annotation.md, integration_diagnostics.md.
Seeds fixed (numpy/torch/scvi/scanpy).
"""
import os, json, numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
import torch
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
try:
    import scvi; scvi.settings.seed = SEED
except Exception as e:
    scvi = None; print("scVI import fail:", e)

PROJ = os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); OUTD = f"{PROJ}/data_engineering"
sc.settings.n_jobs = 8

# ---- marker panels (human symbols) ----
PANELS = {
 "mature_OC": ["CTSK","ACP5","MMP9","NFATC1","ATP6V0D2","DCSTAMP","OCSTAMP","OSCAR","CALCR","CLCN7","OSTM1","ATP6V1B1","SLC4A2"],
 "OCP_mono":  ["CD14","FCN1","S100A8","S100A9","VCAN","CSF1R","ITGAX","LYZ"],
 "macrophage":["CD68","C1QA","C1QB","C1QC","MRC1","APOE","FCGR3A"],
 "Tcell":     ["CD3D","CD3E","TRAC","IL7R","CD8A"],
 "NK":        ["NKG7","GNLY","KLRD1","NCAM1"],
 "Bplasma":   ["CD79A","MS4A1","MZB1","IGHG1","JCHAIN"],
 "endothelial":["PECAM1","VWF","CLDN5","CDH5"],
 "fibro_stroma":["COL1A1","COL1A2","COL3A1","DCN","LUM","PDGFRA","PDGFRB","THY1"],
 "osteoblast":["RUNX2","SP7","BGLAP","IBSP","SPP1","ALPL"],
 "prolif":    ["MKI67","TOP2A","CENPF"],
 "tumor_epi": ["EPCAM","KRT8","KRT18","KRT19"],
 "erythroid": ["HBB","HBA1","GYPA","ALAS2"],
}
OC_LINEAGE = {"mature_OC","OCP_mono","macrophage"}   # myeloid->OC continuum kept for discovery

def sym2ens(adata):
    return {s: e for e, s in zip(adata.var_names, adata.var["symbol"].astype(str))}

def score_panels(A):
    s2e = sym2ens(A)
    for name, syms in PANELS.items():
        ens = [s2e[s] for s in syms if s in s2e]
        if len(ens) >= 2:
            sc.tl.score_genes(A, ens, score_name=f"sig_{name}", use_raw=False)
        else:
            A.obs[f"sig_{name}"] = 0.0

def main():
    A = sc.read_h5ad(f"{OUTD}/master_raw.h5ad")
    A.X = A.layers["counts"].copy()
    print("loaded", A.shape, flush=True)
    # normalize/log for scoring + HVG
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    A.layers["lognorm"] = A.X.copy()
    try:
        sc.pp.highly_variable_genes(A, n_top_genes=2500, flavor="seurat_v3",
                                    layer="counts", batch_key="sample", subset=False)
    except (ImportError, ModuleNotFoundError) as e:
        print("seurat_v3 HVG unavailable (%s) -> seurat flavor on lognorm" % e, flush=True)
        sc.pp.highly_variable_genes(A, n_top_genes=2500, flavor="seurat",
                                    batch_key="sample", subset=False)
    # ---- scVI on counts (batch=sample) ----
    used = "scVI"
    try:
        Ah = A[:, A.var.highly_variable].copy()
        scvi.model.SCVI.setup_anndata(Ah, layer="counts", batch_key="sample")
        m = scvi.model.SCVI(Ah, n_latent=30, n_layers=2, gene_likelihood="nb")
        m.train(max_epochs=120, early_stopping=True, early_stopping_patience=12,
                batch_size=1024, accelerator="gpu", devices=1)
        A.obsm["X_scVI"] = m.get_latent_representation()
        m.save(f"{OUTD}/scvi_model", overwrite=True)
    except Exception as e:
        print("scVI failed -> Harmony fallback:", e, flush=True)
        used = "Harmony"
        sc.pp.scale(A, max_value=10, zero_center=False); sc.tl.pca(A, n_comps=50)
        import scanpy.external as sce; sce.pp.harmony_integrate(A, "sample")
        A.obsm["X_scVI"] = A.obsm["X_pca_harmony"]
    sc.pp.neighbors(A, use_rep="X_scVI", n_neighbors=15, random_state=SEED)
    sc.tl.leiden(A, resolution=2.0, key_added="leiden", random_state=SEED, flavor="igraph", n_iterations=2)
    sc.tl.umap(A, random_state=SEED)
    A.X = A.layers["lognorm"]
    score_panels(A)

    # ---- per-cluster compartment assignment ----
    sigcols = [f"sig_{k}" for k in PANELS]
    cl_mean = A.obs.groupby("leiden")[sigcols].mean()
    comp = {}
    for cl, row in cl_mean.iterrows():
        best = row.idxmax().replace("sig_", "")
        # strict OC guard: a cluster is only "mature_OC" if OC>stroma & OC>osteoblast
        if best == "mature_OC":
            if row["sig_mature_OC"] <= row["sig_fibro_stroma"] or row["sig_mature_OC"] <= row["sig_osteoblast"]:
                best = "fibro_stroma" if row["sig_fibro_stroma"] >= row["sig_osteoblast"] else "osteoblast"
        comp[cl] = best
    A.obs["compartment"] = A.obs["leiden"].map(comp).astype("category")
    A.obs["is_oc_lineage"] = A.obs["compartment"].isin(OC_LINEAGE)

    # OC subtype within lineage (mature vs precursor/mono vs macrophage by max sig)
    occols = ["sig_mature_OC","sig_OCP_mono","sig_macrophage"]
    sub = A.obs[occols].idxmax(axis=1).str.replace("sig_","")
    A.obs["oc_subtype"] = np.where(A.obs["is_oc_lineage"], sub, "non_OC")

    # ---- diagnostics: batch mixing + bio conservation (manual, robust) ----
    diag = {"integration_method": used, "n_cells": int(A.n_obs), "n_clusters": int(A.obs.leiden.nunique())}
    try:
        from sklearn.metrics import silhouette_score
        idx = np.random.choice(A.n_obs, min(20000, A.n_obs), replace=False)
        Z = A.obsm["X_scVI"][idx]
        diag["silhouette_compartment"] = float(silhouette_score(Z, A.obs["compartment"].values[idx]))
        # batch mixing: 1 - silhouette(sample) ; higher = better mixed
        diag["silhouette_sample(lower=better mixed)"] = float(silhouette_score(Z, A.obs["sample"].values[idx]))
    except Exception as e:
        diag["diag_err"] = str(e)
    # per-compartment dataset diversity (Shannon over accession) — mixing of biology across cohorts
    div = {}
    for c, g in A.obs.groupby("compartment"):
        p = g["accession"].value_counts(normalize=True).values
        div[c] = float(-(p*np.log(p+1e-12)).sum())
    diag["compartment_accession_entropy"] = div

    # ---- save ----
    A.write(f"{OUTD}/integrated.h5ad")
    OC = A[A.obs["is_oc_lineage"]].copy(); OC.write(f"{OUTD}/oc_lineage.h5ad")

    # ---- reports ----
    comp_counts = A.obs["compartment"].value_counts().to_dict()
    oc_by_disease = A.obs[A.obs.is_oc_lineage].groupby("disease", observed=True)["oc_subtype"].value_counts().unstack(fill_value=0)
    json.dump({"compartments": comp_counts, "diag": diag,
               "oc_lineage_cells": int(OC.n_obs),
               "mature_OC_cells": int((A.obs.oc_subtype=="mature_OC").sum())},
              open(f"{OUTD}/annotation_summary.json","w"), indent=2, default=str)
    L = [f"# Phase-3b — integration ({used}) + annotation", "",
         f"- cells: {A.n_obs:,} | clusters(leiden res2): {A.obs.leiden.nunique()} | OC-lineage: **{OC.n_obs:,}** | mature-OC: **{(A.obs.oc_subtype=='mature_OC').sum():,}**",
         "", "## Compartments", "| compartment | cells |", "|---|---|"]
    for k,v in comp_counts.items(): L.append(f"| {k} | {v:,} |")
    L += ["", "## OC-lineage subtype × disease", "", oc_by_disease.to_markdown()]
    open(f"{OUTD}/annotation.md","w").write("\n".join(L)+"\n")
    D = [f"# Phase-3b — integration diagnostics ({used})","",
         f"- silhouette by **compartment** (bio conservation, higher=better): {diag.get('silhouette_compartment')}",
         f"- silhouette by **sample** (batch; LOWER = better mixed): {diag.get('silhouette_sample(lower=better mixed)')}","",
         "## Cross-cohort entropy per compartment (higher = biology shared across datasets, not batch-private)",
         "| compartment | accession-entropy |","|---|---|"]
    for k,v in diag["compartment_accession_entropy"].items(): D.append(f"| {k} | {v:.3f} |")
    open(f"{OUTD}/integration_diagnostics.md","w").write("\n".join(D)+"\n")
    print(f"[DONE] integrated cells={A.n_obs:,} OC-lineage={OC.n_obs:,} mature-OC={(A.obs.oc_subtype=='mature_OC').sum():,}")

if __name__ == "__main__":
    main()
