#!/usr/bin/env python
"""Supplementary analyses on the existing 379k-cell integrated atlas (responds to Codex review):
A1 program score per compartment (OC vs TAM/macrophage vs ...);
A2 MMP19 + key program genes compartment-specificity (+ TAM markers) -> OC vs macrophage effect;
A3 four-module scores per compartment;
A4 GSE266330 program across cancer origins (convergence across cancers).
Seeds fixed. Run on a big-memory compute node."""
import os, json, re, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); V=f"{PROJ}/validation"
def dense(X): return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
def cohend(x,y):
    if len(x)<3 or len(y)<3: return float("nan")
    s=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/max(len(x)+len(y)-2,1)); return float((x.mean()-y.mean())/s) if s>0 else 0.0

A=sc.read_h5ad(f"{PROJ}/data_engineering/integrated.h5ad")
if "lognorm" in A.layers: A.X=A.layers["lognorm"]
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
sym2ens={s:e for e,s in zip(A.var_names,A.var["symbol"].astype(str))}; ens2sym={e:s for s,e in sym2ens.items()}
sig_ens=[g for g in sig.ensembl if g in A.var_names]
sc.tl.score_genes(A,sig_ens,score_name="psig",use_raw=False)
comp=A.obs["compartment"].astype(str).values; S=A.obs["psig"].values
COMPS=["mature_OC","macrophage","OCP_mono","fibro_stroma","osteoblast","tumor_epi","Tcell","NK","Bplasma","endothelial","erythroid","prolif"]
COMPS=[c for c in COMPS if (comp==c).any()]
print("compartments:",[(c,int((comp==c).sum())) for c in COMPS],flush=True)

# A1 program score per compartment (with quartiles for boxplots)
def qd(v): return {"mean":float(v.mean()),"median":float(np.median(v)),"q05":float(np.percentile(v,5)),
                   "q25":float(np.percentile(v,25)),"q75":float(np.percentile(v,75)),"q95":float(np.percentile(v,95)),"n":int(len(v))}
res={"program_score_by_compartment":{c:qd(S[comp==c]) for c in COMPS}}
res["program_OC_vs_macrophage_cohensd"]=cohend(S[comp=="mature_OC"],S[comp=="macrophage"])

# A2 MMP19 + key genes + markers compartment expression
PANEL=["MMP19","SH3PXD2A","FAM20C","PAPSS2","MYO1E","DNM3","RUNX3","RFX8","CTSK","ACP5","NFATC1",
       "CD68","C1QA","C1QB","CD163","MRC1","LYZ","FCGR3A","APOE"]
panel=[g for g in PANEL if g in sym2ens]
expr={}
for g in panel:
    e=sym2ens[g]; col=dense(A[:,e].X).ravel()
    expr[g]={c:{"mean_in_pos":float(col[(comp==c)&(col>0)].mean()) if ((comp==c)&(col>0)).any() else 0.0,
                "frac_pos":float((col[comp==c]>0).mean())} for c in COMPS}
res["panel_by_compartment"]=expr
# MMP19 specifically: OC vs macrophage
mmp=dense(A[:,sym2ens["MMP19"]].X).ravel() if "MMP19" in sym2ens else np.zeros(A.n_obs)
res["MMP19"]={"OC_mean":float(mmp[comp=="mature_OC"].mean()),"macrophage_mean":float(mmp[comp=="macrophage"].mean()),
              "OC_fracpos":float((mmp[comp=="mature_OC"]>0).mean()),"macrophage_fracpos":float((mmp[comp=="macrophage"]>0).mean()),
              "cohensd_OC_vs_macrophage":cohend(mmp[comp=="mature_OC"],mmp[comp=="macrophage"]),
              "log2FC_OC_vs_macrophage":float(mmp[comp=="mature_OC"].mean()-mmp[comp=="macrophage"].mean())}

# A3 four-module scores per compartment
MOD={"Podosome":["SH3PXD2A","MYO1E","MYO1D","DNM3","TIAM1","PTPRM","CAMSAP2","NAV2"],
     "Protease":["MMP19","CEMIP2","COL27A1","PAM","BAMBI"],
     "Sulfation":["PAPSS2","UST","EXT1","FAM20C","SLC16A10"],
     "TF":["JDP2","RUNX3","KLF6","RFX8","SOX4","NR4A3"]}
modres={}
for m,gs in MOD.items():
    e=[sym2ens[g] for g in gs if g in sym2ens]
    if len(e)>=2:
        sc.tl.score_genes(A,e,score_name="mm",use_raw=False); v=A.obs["mm"].values
        modres[m]={c:float(v[comp==c].mean()) for c in COMPS}
res["module_by_compartment"]=modres

# A4 GSE266330 program across cancer origins (mature-OC only)
acc=A.obs["accession"].astype(str).values; samp=A.obs["sample"].astype(str).values
CODE={"BC":"Breast","KC":"Renal","LC":"Lung","CC":"Colon","EC":"Esophagus","TC":"Thyroid","BDC":"BileDuct","PC":"Prostate","ctrl":"Control(BM)"}
def origin(s):
    m=re.search(r"_(BC|KC|LC|CC|EC|TC|BDC|PC|ctrl)_",s) or re.search(r"_(BC|KC|LC|CC|EC|TC|BDC|PC|ctrl)\b",s)
    return CODE.get(m.group(1),"Other") if m else "Other"
oc266=(comp=="mature_OC")&(acc=="GSE266330")
org=np.array([origin(s) for s in samp])
canc={}
for o in sorted(set(org[oc266])):
    m=oc266&(org==o)
    if m.sum()>=15: canc[o]=qd(S[m])
res["GSE266330_by_cancer_origin"]=canc

json.dump(res,open(f"{V}/supp_compartment.json","w"),indent=2,default=str)
# tidy csv for the panel dotplot
rows=[]
for g in panel:
    for c in COMPS: rows.append({"gene":g,"compartment":c,"mean_in_pos":expr[g][c]["mean_in_pos"],"frac_pos":expr[g][c]["frac_pos"]})
pd.DataFrame(rows).to_csv(f"{V}/supp_panel_dotplot.csv",index=False)
print("[DONE] MMP19 OC vs macrophage d=",round(res['MMP19']['cohensd_OC_vs_macrophage'],2),
      "| program OC vs mac d=",round(res['program_OC_vs_macrophage_cohensd'],2),
      "| cancer origins:",list(canc.keys()))
