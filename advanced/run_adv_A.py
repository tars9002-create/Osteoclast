#!/usr/bin/env python
"""Advanced downstream analyses on the integrated atlas (compute node):
LIANA cell-cell communication by compartment; decoupler DoRothEA TF-activity and PROGENy
pathway-activity per compartment; gseapy enrichment of the 74-gene program; embedding density.
Outputs -> advanced/. Seeds fixed."""
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); OUT=f"{PROJ}/advanced"; os.makedirs(OUT,exist_ok=True)
log=lambda *a: print(*a,flush=True)
status={}

A=sc.read_h5ad(f"{PROJ}/data_engineering/integrated.h5ad")
A.X=A.layers["lognorm"]
A.var["ensembl"]=A.var_names.astype(str); A.var_names=A.var["symbol"].astype(str); A.var_names_make_unique()
comp=A.obs["compartment"].astype(str)
log("atlas",A.shape,"compartments",sorted(comp.unique()))

# balanced subsample for LIANA + decoupler (per-compartment means are stable on a subsample)
rng=np.random.default_rng(0); idx=[]
for c in comp.unique():
    ci=np.where(comp.values==c)[0]; idx+=list(rng.choice(ci,min(5000,len(ci)),replace=False))
idx=np.array(sorted(idx)); S=A[idx].copy()
log("subsample",S.shape)

# ---------- LIANA cell-cell communication ----------
try:
    import liana as li
    li.mt.rank_aggregate(S, groupby="compartment", use_raw=False, expr_prop=0.1, n_perms=100, verbose=False)
    lr=S.uns["liana_res"].copy(); lr.to_csv(f"{OUT}/liana_res.csv",index=False)
    # OC-incoming (source->OC) and OC-outgoing (OC->source), top by magnitude
    inc=lr[lr.target=="mature_OC"].sort_values("magnitude_rank").head(40); inc.to_csv(f"{OUT}/liana_OC_incoming.csv",index=False)
    out=lr[lr.source=="mature_OC"].sort_values("magnitude_rank").head(40); out.to_csv(f"{OUT}/liana_OC_outgoing.csv",index=False)
    # aggregated interaction strength matrix (count of specific interactions source x target)
    sig=lr[lr.specificity_rank<0.05]
    M=sig.groupby(["source","target"]).size().unstack(fill_value=0); M.to_csv(f"{OUT}/liana_interaction_matrix.csv")
    status["liana"]=f"ok rows={len(lr)} sig={len(sig)}"
except Exception as e:
    import traceback; traceback.print_exc(); status["liana"]=f"FAIL {e}"
log("LIANA:",status.get("liana"))

# ---------- decoupler TF activity (DoRothEA) + PROGENy pathways ----------
try:
    import decoupler as dc
    tfnet=dc.op.dorothea(organism="human", levels=["A","B","C"])
    dc.mt.ulm(data=S, net=tfnet, verbose=False)
    tfkey=[k for k in S.obsm if "ulm" in k.lower() and "padj" not in k.lower()]
    acts=S.obsm[tfkey[0]] if tfkey else None
    if acts is not None:
        acts=pd.DataFrame(acts) if not isinstance(acts,pd.DataFrame) else acts
        acts.index=S.obs_names
        tf_by=acts.groupby(comp.values[idx]).mean()
        tf_by.to_csv(f"{OUT}/tf_activity_by_compartment.csv")
        status["tf"]=f"ok TFs={acts.shape[1]} key={tfkey}"
    else: status["tf"]="FAIL no obsm key"
except Exception as e:
    import traceback; traceback.print_exc(); status["tf"]=f"FAIL {e}"
log("TF:",status.get("tf"))
try:
    import decoupler as dc
    pw=dc.op.progeny(organism="human", top=500)
    dc.mt.mlm(data=S, net=pw, verbose=False)
    pkey=[k for k in S.obsm if "mlm" in k.lower() and "padj" not in k.lower()]
    pa=S.obsm[pkey[0]] if pkey else None
    if pa is not None:
        pa=pd.DataFrame(pa) if not isinstance(pa,pd.DataFrame) else pa; pa.index=S.obs_names
        pa.groupby(comp.values[idx]).mean().to_csv(f"{OUT}/progeny_by_compartment.csv")
        status["progeny"]=f"ok pathways={pa.shape[1]} key={pkey}"
    else: status["progeny"]="FAIL no obsm key"
except Exception as e:
    import traceback; traceback.print_exc(); status["progeny"]=f"FAIL {e}"
log("PROGENy:",status.get("progeny"))

# ---------- gseapy enrichment of the 74-gene program ----------
try:
    import gseapy
    sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
    genes=[str(x) for x in sig.symbol]
    enr=gseapy.enrichr(gene_list=genes, gene_sets=["GO_Biological_Process_2021","Reactome_2022","MSigDB_Hallmark_2020"], organism="human", outdir=None)
    enr.results.sort_values("Adjusted P-value").head(40).to_csv(f"{OUT}/gsea_enrichr.csv",index=False)
    status["gsea"]=f"ok terms={len(enr.results)}"
except Exception as e:
    import traceback; traceback.print_exc(); status["gsea"]=f"FAIL {e}"
log("GSEA:",status.get("gsea"))

# ---------- embedding density (program + mature-OC) on full atlas ----------
try:
    sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
    se=[g for g in sig.symbol.astype(str) if g in A.var_names]
    sc.tl.score_genes(A,se,score_name="psig",use_raw=False)
    sc.tl.embedding_density(A, basis="umap", groupby="compartment")
    # save per-cell umap + psig + density + compartment for figure
    df=pd.DataFrame({"umap1":A.obsm["X_umap"][:,0],"umap2":A.obsm["X_umap"][:,1],
                     "psig":A.obs["psig"].values,"compartment":comp.values})
    if "umap_density_compartment" in A.obs: df["dens"]=A.obs["umap_density_compartment"].values
    df.sample(min(120000,len(df)),random_state=0).to_csv(f"{OUT}/umap_cells.csv.gz",index=False,compression="gzip")
    status["density"]="ok"
except Exception as e:
    import traceback; traceback.print_exc(); status["density"]=f"FAIL {e}"
log("density:",status.get("density"))

json.dump(status,open(f"{OUT}/run_adv_A_status.json","w"),indent=2)
log("[DONE] status:",status)
