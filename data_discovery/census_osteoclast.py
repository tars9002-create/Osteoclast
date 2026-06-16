#!/usr/bin/env python
"""Pull every HUMAN cell annotated cell_type=='osteoclast' from the CELLxGENE
census -> a clean multi-source osteoclast reference (raw counts) for label
transfer / signature anchoring. Writes human_osteoclast_census.h5ad + a report."""
import sys, json, os
import cellxgene_census
import scanpy as sc

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

cols = ["assay","cell_type","tissue","tissue_general","disease",
        "dataset_id","donor_id","development_stage","sex","self_reported_ethnicity"]
print("[census] opening stable census ...", flush=True)
with cellxgene_census.open_soma(census_version="stable") as census:
    ver = cellxgene_census.get_census_version_description("stable").get("release_build","?") \
        if hasattr(cellxgene_census,"get_census_version_description") else "stable"
    adata = cellxgene_census.get_anndata(
        census, organism="Homo sapiens",
        obs_value_filter="cell_type == 'osteoclast'",
        column_names={"obs": cols},
    )
print(f"[census] osteoclast cells: {adata.n_obs:,} x {adata.n_vars:,} genes", flush=True)
adata.write(os.path.join(OUT, "human_osteoclast_census.h5ad"))

rep = {"census_version": str(ver), "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)}
for c in ["dataset_id","tissue_general","disease","assay","development_stage"]:
    rep[c] = adata.obs[c].value_counts().head(20).to_dict()
json.dump(rep, open(os.path.join(OUT, "census_osteoclast_report.json"), "w"), indent=2, default=str)
print("[census] dataset_id distribution:")
for k, v in list(rep["dataset_id"].items())[:15]:
    print(f"   {v:>7,}  {k}")
print("[census] tissue_general:", rep["tissue_general"])
print("[census] disease:", rep["disease"])
print("[census] DONE")
