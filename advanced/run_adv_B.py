#!/usr/bin/env python
"""Trajectory analyses on the OC-lineage object (compute node):
PAGA abstraction (OCP/monocyte -> macrophage -> mature OC), diffusion pseudotime (dpt),
and a pseudotime-ordered gene-expression matrix for a heatmap, plus program/module trends.
Outputs -> advanced/. Seeds fixed."""
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd, scanpy as sc
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); OUT=f"{PROJ}/advanced"; os.makedirs(OUT,exist_ok=True)
log=lambda *a: print(*a,flush=True)

A=sc.read_h5ad(f"{PROJ}/data_engineering/oc_lineage_umap.h5ad")
A.X=A.layers["lognorm"]
A.var["ensembl"]=A.var_names.astype(str); A.var_names=A.var["symbol"].astype(str); A.var_names_make_unique()
comp=A.obs["compartment"].astype(str)
log("oc-lineage",A.shape,"compartments",sorted(comp.unique()))

# neighbors on the integrated latent, PAGA on compartment
sc.pp.neighbors(A, use_rep="X_scVI", n_neighbors=15)
sc.tl.paga(A, groups="compartment")
paga=A.uns["paga"]["connectivities"].toarray()
groups=list(A.obs["compartment"].cat.categories) if hasattr(A.obs["compartment"],"cat") else sorted(comp.unique())
pd.DataFrame(paga,index=groups,columns=groups).to_csv(f"{OUT}/paga_connectivities.csv")

# diffusion pseudotime, root in OCP/monocyte progenitor
sc.tl.diffmap(A)
prog=np.where(comp.values=="OCP_mono")[0]
if len(prog)==0: prog=np.where(comp.values=="macrophage")[0]
# root = progenitor cell at the extreme of DC1
dc1=A.obsm["X_diffmap"][:,1]
root=prog[np.argmin(dc1[prog])] if np.mean(dc1[prog])<np.mean(dc1) else prog[np.argmax(dc1[prog])]
A.uns["iroot"]=int(root); sc.tl.dpt(A)
dpt=A.obs["dpt_pseudotime"].values
log("dpt range",float(np.nanmin(dpt)),float(np.nanmax(dpt)))

# per-cell export (umap + dpt + compartment + psig)
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
se=[g for g in sig.symbol.astype(str) if g in A.var_names]
sc.tl.score_genes(A,se,score_name="psig",use_raw=False)
MOD={"Podosome":["SH3PXD2A","MYO1E","MYO1D","DNM3","TIAM1","CAMSAP2"],
     "Protease":["MMP19","CEMIP2","COL27A1","PAM","BAMBI"],
     "Sulfation":["PAPSS2","UST","EXT1","FAM20C","SLC16A10"],
     "TF":["JDP2","RUNX3","RFX8","SOX4","MSI2","CDK6"]}
for m,gs in MOD.items():
    g=[x for x in gs if x in A.var_names]
    if len(g)>=2: sc.tl.score_genes(A,g,score_name=f"mod_{m}",use_raw=False)
df=pd.DataFrame({"umap1":A.obsm["X_umap"][:,0],"umap2":A.obsm["X_umap"][:,1],"dpt":dpt,
                "compartment":comp.values,"psig":A.obs["psig"].values})
for m in MOD:
    if f"mod_{m}" in A.obs: df[f"mod_{m}"]=A.obs[f"mod_{m}"].values
df.sample(min(80000,len(df)),random_state=0).to_csv(f"{OUT}/trajectory_cells.csv.gz",index=False,compression="gzip")

# pseudotime-ordered gene heatmap matrix: bin cells by dpt, mean lognorm expr per bin
GENES=["CD14","CSF1R","LYZ","FCGR3A",  # progenitor/myeloid
       "CTSK","ACP5","NFATC1","ATP6V0D2","DCSTAMP",  # OC maturation
       "SH3PXD2A","MYO1E","DNM3","TIAM1",  # podosome
       "PAPSS2","UST","EXT1","FAM20C","SLC16A10",  # sulfation
       "MMP19","CEMIP2",  # protease
       "JDP2","RUNX3","RFX8","SOX4"]  # TF
genes=[g for g in GENES if g in A.var_names]
nb=50; ok=np.isfinite(dpt); order=np.argsort(dpt[ok]); cells=np.where(ok)[0][order]
bins=np.array_split(cells, nb)
import scipy.sparse as sp
def col(g):
    x=A[:,g].X; return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
mat=np.zeros((len(genes),nb))
for gi,g in enumerate(genes):
    c=col(g)
    for bi,b in enumerate(bins): mat[gi,bi]=c[b].mean()
# row z-score for the heatmap
matz=(mat-mat.mean(1,keepdims=True))/(mat.std(1,keepdims=True)+1e-9)
pd.DataFrame(matz,index=genes).to_csv(f"{OUT}/pseudotime_gene_heatmap.csv")
# program + modules per bin (mean + sem)
def binstat(v):
    return [ (float(v[b].mean()), float(v[b].std()/max(np.sqrt(len(b)),1))) for b in bins ]
trends={"psig":binstat(A.obs["psig"].values)}
for m in MOD:
    if f"mod_{m}" in A.obs: trends[m]=binstat(A.obs[f"mod_{m}"].values)
# compartment composition per bin (for the band under the heatmap)
compbin=[ {c:float((comp.values[b]==c).mean()) for c in sorted(comp.unique())} for b in bins ]
json.dump({"trends":trends,"compbin":compbin,"n_bins":nb,"genes":genes},open(f"{OUT}/pseudotime_trends.json","w"),indent=2)
log("[DONE] paga+dpt+pseudotime heatmap; genes",len(genes),"bins",nb)
