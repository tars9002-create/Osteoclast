#!/usr/bin/env python
"""Hardening re-analysis (responds to adversarial review). On the integrated atlas:
(1) bootstrap 95% CI for the OC-vs-macrophage compartment Cohen's d;
(2) doublet confound: program-score vs doublet-score correlation in osteoclasts, per-gene
    doublet loading (flag doublet-driven genes), and whether the sulfation module tracks doublets;
(3) sulfation-module specificity with bootstrap CI (OC vs next-highest compartment).
Seeds fixed."""
import os, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); V=f"{PROJ}/validation"
def cohend(x,y):
    x=np.asarray(x); y=np.asarray(y); nx,ny=len(x),len(y)
    if nx<3 or ny<3: return float("nan")
    s=np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/max(nx+ny-2,1))
    return float((x.mean()-y.mean())/s) if s>0 else 0.0
def boot_d(x,y,n=1000):
    rng=np.random.default_rng(0); x=np.asarray(x); y=np.asarray(y); ds=[]
    for _ in range(n):
        ds.append(cohend(rng.choice(x,len(x),replace=True),rng.choice(y,len(y),replace=True)))
    return {"d":cohend(x,y),"ci":[float(np.percentile(ds,2.5)),float(np.percentile(ds,97.5))]}
def dense1(A,e):
    col=A[:,e].X; return np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()

A=sc.read_h5ad(f"{PROJ}/data_engineering/integrated.h5ad")
if "lognorm" in A.layers: A.X=A.layers["lognorm"]
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
sig_ens=[g for g in sig.ensembl if g in A.var_names]
sc.tl.score_genes(A,sig_ens,score_name="psig",use_raw=False)
comp=A.obs["compartment"].astype(str).values; S=A.obs["psig"].values
out={"obs_cols":[c for c in A.obs.columns]}

# (1) OC vs macrophage compartment d + CI
out["d_OC_vs_macrophage"]=boot_d(S[comp=="mature_OC"],S[comp=="macrophage"])

# (2) doublet confound
dubcol=None
for c in ["doublet_score","scrublet_score","predicted_doublet_score","doublet","scrublet","dub_score"]:
    if c in A.obs.columns: dubcol=c; break
ocm=comp=="mature_OC"
if dubcol:
    dub=pd.to_numeric(A.obs[dubcol],errors="coerce").values
    msk=ocm&np.isfinite(dub)
    out["doublet"]={"col":dubcol,
        "pearson_psig_vs_dub_inOC":float(pearsonr(S[msk],dub[msk])[0]),
        "spearman_psig_vs_dub_inOC":float(spearmanr(S[msk],dub[msk])[0]),
        "mean_dub_topdecile_psig":float(dub[msk][S[msk]>=np.percentile(S[msk],90)].mean()),
        "mean_dub_rest":float(dub[msk][S[msk]<np.percentile(S[msk],90)].mean())}
    # per-gene doublet loading within OC
    load=[]
    for e,s_ in zip(sig.ensembl,sig.symbol):
        if e in A.var_names:
            x=dense1(A,e)[msk]
            r=float(spearmanr(x,dub[msk])[0]) if np.std(x)>0 else 0.0
            load.append((str(s_),round(r,3)))
    load.sort(key=lambda t:-t[1])
    out["doublet"]["top10_doublet_loaded_genes"]=load[:10]
    SULF=["PAPSS2","UST","EXT1","FAM20C","SLC16A10","ST3GAL6","SELENOI","GBE1"]
    out["doublet"]["sulfation_gene_loading"]=[(s,r) for s,r in load if s in SULF]
    # does program survive after dropping top-doublet-decile OCs? recompute OC vs macrophage among low-doublet OC
    keep=msk&(dub<np.percentile(dub[msk],90))
    out["d_OC_vs_macrophage_lowDoublet"]=boot_d(S[keep],S[comp=="macrophage"])
else:
    out["doublet"]={"note":"no doublet column in obs; cols="+",".join(map(str,A.obs.columns))}

# (3) sulfation module specificity with CI (OC vs next-highest compartment)
sym2ens={s:e for e,s in zip(A.var_names,A.var["symbol"].astype(str))}
SULF=["PAPSS2","UST","EXT1","FAM20C","SLC16A10","ST3GAL6","SELENOI","GBE1"]
se=[sym2ens[g] for g in SULF if g in sym2ens]
sc.tl.score_genes(A,se,score_name="sulf",use_raw=False); SU=A.obs["sulf"].values
COMPS=[c for c in pd.unique(comp)]
mod_by={c:float(SU[comp==c].mean()) for c in COMPS}
ranked=sorted(mod_by.items(),key=lambda t:-t[1])
out["sulfation_module_by_compartment"]=dict(ranked)
nexttop=[c for c,_ in ranked if c!="mature_OC"][0]
out["sulfation_OC_vs_nexttop"]={"next":nexttop, **boot_d(SU[comp=="mature_OC"],SU[comp==nexttop])}
# tumour-epithelial = osteoblast-like in OS: report explicitly
if "tumor_epi" in COMPS: out["sulfation_OC_vs_tumor_epi"]=boot_d(SU[comp=="mature_OC"],SU[comp=="tumor_epi"])

json.dump(out,open(f"{V}/reanalysis_harden.json","w"),indent=2,default=str)
print("[DONE] d_OC_vs_mac=",round(out['d_OC_vs_macrophage']['d'],2),out['d_OC_vs_macrophage']['ci'],
      "| sulf OC vs",out['sulfation_OC_vs_nexttop']['next'],"=",round(out['sulfation_OC_vs_nexttop']['d'],2),
      "| doublet col:",out['doublet'].get('col','NONE'))
