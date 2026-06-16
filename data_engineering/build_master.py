#!/usr/bin/env python
"""
Phase-3a: build the human osteoclast-discovery MASTER object.
Loads all GEO tumor cohorts (Ensembl-keyed) + the CELLxGENE normal-OC reference,
per-sample QC, harmonizes on Ensembl gene IDs, concatenates, keeps raw counts.
Writes data_engineering/master_raw.h5ad + qc_report.{md,json}.

Seeds fixed. Run on a compute node.
"""
import os, sys, json, gzip, glob, re
import numpy as np, pandas as pd
import scipy.io, scipy.sparse as sp
import anndata as ad, scanpy as sc

np.random.seed(0)
PROJ = os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL   = os.environ.get("OC_DOWNLOADS", os.path.join(PROJ, "downloads"))
OUTD = f"{PROJ}/data_engineering"
MAN  = json.load(open(f"{OUTD}/datasets_manifest.json"))

# QC thresholds
MIN_GENES, MIN_COUNTS, MAX_MT_PCT, MIN_CELLS = 200, 500, 25.0, 3

def _open(f): return gzip.open(f, "rt") if f.endswith(".gz") else open(f)

def find_groups(d):
    groups = {}
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        b = os.path.basename(f)
        m = re.match(r"(?:(.+?)[._-])?(matrix\.mtx|features\.tsv|genes\.tsv|barcodes\.tsv)(\.gz)?$", b, re.I)
        if not m: continue
        pre, kind = m.group(1), m.group(2).lower()
        key = pre if pre else os.path.dirname(f)
        g = groups.setdefault(key, {})
        if "matrix" in kind: g["mtx"] = f
        elif "barcodes" in kind: g["bc"] = f
        else: g["feat"] = f
    return {k: v for k, v in groups.items() if {"mtx","feat","bc"} <= set(v)}

def load_geo_sample(g, sample, meta):
    """Load one 10x sample keyed by ENSEMBL id (features col1), symbol in var['symbol']."""
    M = scipy.io.mmread(g["mtx"]).tocsr()
    feats = [l.rstrip("\n").split("\t") for l in _open(g["feat"])]
    ens  = [c[0] for c in feats]
    sym  = [c[1] if len(c) > 1 else c[0] for c in feats]
    bcs  = [l.strip() for l in _open(g["bc"])]
    if M.shape == (len(ens), len(bcs)):   X = M.T.tocsr()
    elif M.shape == (len(bcs), len(ens)): X = M.tocsr()
    else: raise ValueError(f"shape {M.shape} vs g{len(ens)} c{len(bcs)}")
    A = ad.AnnData(X=X)
    A.var_names = pd.Index(ens); A.var["symbol"] = sym
    A.var_names_make_unique()
    A.obs_names = [f"{sample}|{b}" for b in bcs]
    for k, v in meta.items(): A.obs[k] = v
    A.obs["sample"] = sample
    return A

def qc(A):
    A.var["mt"] = A.var["symbol"].astype(str).str.startswith("MT-")
    sc.pp.calculate_qc_metrics(A, qc_vars=["mt"], inplace=True, percent_top=None)
    n0 = A.n_obs
    A = A[(A.obs.n_genes_by_counts >= MIN_GENES) &
          (A.obs.total_counts >= MIN_COUNTS) &
          (A.obs.pct_counts_mt <= MAX_MT_PCT)].copy()
    return A, n0, A.n_obs

def main():
    parts, qclog = [], []
    # ---- GEO cohorts ----
    for d in MAN["geo"]:
        acc = d["accession"]; ext = f"{DL}/{acc}/extracted"
        groups = find_groups(ext)
        print(f"[{acc}] {len(groups)} samples", flush=True)
        for sample, g in sorted(groups.items()):
            try:
                A = load_geo_sample(g, f"{acc}:{sample}", {k: d[k] for k in ("accession","context","disease","role") if k in d} | {"accession": acc, "source": "GEO"})
                A, n0, n1 = qc(A)
                if n1 < 30:
                    qclog.append({"sample": f"{acc}:{sample}", "raw": n0, "kept": n1, "note": "dropped<30"}); continue
                parts.append(A); qclog.append({"sample": f"{acc}:{sample}", "raw": int(n0), "kept": int(n1)})
                print(f"   {sample}: {n0}->{n1}", flush=True)
            except Exception as e:
                qclog.append({"sample": f"{acc}:{sample}", "error": f"{type(e).__name__}: {e}"})
                print(f"   {sample} ERROR {e}", flush=True)
    # ---- census reference ----
    c = MAN["census"]
    try:
        R = ad.read_h5ad(c["path"])
        R.var["symbol"] = R.var["feature_name"].astype(str).values
        R.var_names = pd.Index(R.var["feature_id"].astype(str).values); R.var_names_make_unique()
        R.obs["sample"] = "census:" + R.obs["dataset_id"].astype(str)
        R.obs["accession"] = "census"; R.obs["context"] = c["context"]
        R.obs["disease"] = R.obs.get("disease", c["disease"]); R.obs["role"] = c["role"]; R.obs["source"] = "census"
        R.obs_names = ["census|" + x for x in R.obs_names]
        R.X = sp.csr_matrix(R.X)
        keep = ["sample","accession","context","disease","role","source"]
        R.obs = R.obs[keep + [x for x in ["assay","development_stage","sex"] if x in R.obs.columns]]
        R2 = ad.AnnData(X=R.X, obs=R.obs.copy()); R2.var_names = R.var_names; R2.var["symbol"] = R.var["symbol"].values
        R2, n0, n1 = qc(R2)
        parts.append(R2); qclog.append({"sample": "census(all)", "raw": int(n0), "kept": int(n1)})
        print(f"[census] {n0}->{n1}", flush=True)
    except Exception as e:
        qclog.append({"sample": "census", "error": str(e)}); print("census ERROR", e, flush=True)

    # ---- harmonize on Ensembl intersection + concat ----
    common = set(parts[0].var_names)
    for A in parts[1:]: common &= set(A.var_names)
    common = sorted(common)
    print(f"[concat] {len(parts)} parts, {len(common)} shared Ensembl genes", flush=True)
    sym_map = {}
    for A in parts:
        for e, s in zip(A.var_names, A.var["symbol"].astype(str)): sym_map.setdefault(e, s)
    parts = [A[:, common].copy() for A in parts]
    M = ad.concat(parts, join="outer", merge="same", index_unique=None)
    M.var["symbol"] = [sym_map.get(e, e) for e in M.var_names]
    M.layers["counts"] = M.X.copy()
    for col in ["accession","context","disease","role","sample","source"]:
        if col in M.obs: M.obs[col] = M.obs[col].astype("category")
    M.uns["build"] = {"min_genes": MIN_GENES, "min_counts": MIN_COUNTS, "max_mt_pct": MAX_MT_PCT,
                      "n_shared_genes": len(common), "seed": 0}
    out = f"{OUTD}/master_raw.h5ad"; M.write(out)
    # ---- report ----
    rep = {"n_cells": int(M.n_obs), "n_genes": int(M.n_vars),
           "by_context": M.obs["context"].value_counts().to_dict(),
           "by_disease": M.obs["disease"].value_counts().to_dict(),
           "by_role": M.obs["role"].value_counts().to_dict(),
           "n_samples": int(M.obs["sample"].nunique()), "qc": qclog}
    json.dump(rep, open(f"{OUTD}/qc_report.json", "w"), indent=2, default=str)
    L = [f"# Phase-3a master build — QC report", "",
         f"- **cells**: {M.n_obs:,}  | **genes (shared Ensembl)**: {M.n_vars:,} | **samples**: {rep['n_samples']}",
         f"- output: `master_raw.h5ad` (raw counts in `layers['counts']`)", "",
         "## By context", "| context | cells |", "|---|---|"]
    for k, v in rep["by_context"].items(): L.append(f"| {k} | {v:,} |")
    L += ["", "## By role (discovery/replication/clinical/reference)", "| role | cells |", "|---|---|"]
    for k, v in rep["by_role"].items(): L.append(f"| {k} | {v:,} |")
    open(f"{OUTD}/qc_report.md", "w").write("\n".join(L) + "\n")
    print(f"[DONE] master_raw.h5ad cells={M.n_obs:,} genes={M.n_vars:,} samples={rep['n_samples']}")

if __name__ == "__main__":
    main()
