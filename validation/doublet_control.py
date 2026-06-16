#!/usr/bin/env python
"""Doublet control (F3): run scrublet on the OC-lineage object, then test whether the program
and its sulfation module survive doublet removal, and which genes load on doublet score.
Also report a 'specific-core' program score with non-specific/implausible genes removed."""
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
    rng=np.random.default_rng(0); x=np.asarray(x); y=np.asarray(y); ds=[cohend(rng.choice(x,len(x),True),rng.choice(y,len(y),True)) for _ in range(n)]
    return {"d":cohend(x,y),"ci":[float(np.percentile(ds,2.5)),float(np.percentile(ds,97.5))]}
def dense1(A,e):
    col=A[:,e].X; return np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()

A=sc.read_h5ad(f"{PROJ}/data_engineering/oc_lineage.h5ad")
print("loaded",A.shape,"layers",list(A.layers.keys()),flush=True)
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
sym2ens={s:e for e,s in zip(A.var_names,A.var["symbol"].astype(str))} if "symbol" in A.var else {}
sig_ens=[g for g in sig.ensembl if g in A.var_names]
# program score on lognorm
if "lognorm" in A.layers: A.X=A.layers["lognorm"]
sc.tl.score_genes(A,sig_ens,score_name="psig",use_raw=False)
comp=A.obs["compartment"].astype(str).values; S=A.obs["psig"].values
out={}
# scrublet on counts, per sample
try:
    if "counts" in A.layers: A.layers["_X"]=A.X.copy(); A.X=A.layers["counts"].copy()
    sc.pp.scrublet(A, n_prin_comps=20)  # global (per-sample batches too small for arpack PCA)
    if "_X" in A.layers: A.X=A.layers["_X"]
    dub=pd.to_numeric(A.obs["doublet_score"],errors="coerce").values
    out["scrublet_ok"]=True
except Exception as e:
    import traceback; traceback.print_exc(); out["scrublet_ok"]=False; dub=None

ocm=comp=="mature_OC"
if dub is not None:
    msk=ocm&np.isfinite(dub)
    out["doublet"]={"pearson_psig_dub_inOC":float(pearsonr(S[msk],dub[msk])[0]),
                    "spearman_psig_dub_inOC":float(spearmanr(S[msk],dub[msk])[0]),
                    "mean_dub_topdecile_psig":float(dub[msk][S[msk]>=np.percentile(S[msk],90)].mean()),
                    "mean_dub_rest":float(dub[msk][S[msk]<np.percentile(S[msk],90)].mean())}
    load=[]
    for e,s_ in zip(sig.ensembl,sig.symbol):
        if e in A.var_names:
            x=dense1(A,e)[msk]; r=float(spearmanr(x,dub[msk])[0]) if np.std(x)>0 else 0.0
            load.append((str(s_),round(r,3)))
    load.sort(key=lambda t:-t[1])
    out["doublet"]["top10_loaded"]=load[:10]
    SULF=["PAPSS2","UST","EXT1","FAM20C","SLC16A10","ST3GAL6","SELENOI","GBE1"]
    out["doublet"]["sulfation_loading"]=[(s,r) for s,r in load if s in SULF]
    # re-derive OC vs macrophage among low-doublet OC (drop top decile)
    thr=np.percentile(dub[msk],90); keepOC=ocm&np.isfinite(dub)&(dub<thr)
    out["d_OC_vs_mac_lowDoublet"]=boot_d(S[keepOC],S[comp=="macrophage"])
    out["d_OC_vs_mac_all"]=boot_d(S[ocm],S[comp=="macrophage"])
# specific-core: drop MMP19/ACTN2/COL27A1 (non-specific/implausible)
drop={"MMP19","ACTN2","COL27A1"}
core_ens=[e for e,s_ in zip(sig.ensembl,sig.symbol) if s_ not in drop and e in A.var_names]
sc.tl.score_genes(A,core_ens,score_name="pcore",use_raw=False); C=A.obs["pcore"].values
out["specific_core_OC_vs_mac"]=boot_d(C[ocm],C[comp=="macrophage"])
out["n_core_genes"]=len(core_ens)
json.dump(out,open(f"{V}/doublet_control.json","w"),indent=2,default=str)
print("[DONE] scrublet_ok:",out.get("scrublet_ok"),
      "| OC-vs-mac all:",round(out.get('d_OC_vs_mac_all',{}).get('d',float('nan')),2) if 'd_OC_vs_mac_all' in out else 'NA',
      "| lowDoublet:",round(out.get('d_OC_vs_mac_lowDoublet',{}).get('d',float('nan')),2) if 'd_OC_vs_mac_lowDoublet' in out else 'NA',
      "| core:",round(out['specific_core_OC_vs_mac']['d'],2))
