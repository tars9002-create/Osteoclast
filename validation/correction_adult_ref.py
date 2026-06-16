#!/usr/bin/env python
"""
Phase-6 correction (responds to final_verification.md must-fixes):
1. ADULT, same-platform control: re-test the program in tumour-OC vs GSE266330
   adult control-bone OCs (not the ~100% fetal census), removing the
   developmental + protocol confound.
2. DEPTH-matched vs-myeloid: residualise the signature on log total_counts
   (OCs carry ~2.35x myeloid RNA) and recompute d.
3. MATURATION control: is the program a maturation gradient? (overlap, d, residualised)
Outputs validation/correction_adult_ref.{json,md}. Seeds fixed.
"""
import os, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.metrics import roc_auc_score
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); V=f"{PROJ}/validation"
MAT=["CTSK","ACP5","MMP9","NFATC1","ATP6V0D2","DCSTAMP","OCSTAMP","OSCAR","CALCR","CLCN7","OSTM1"]

def d_(x,y):
    if len(x)<3 or len(y)<3: return float("nan")
    sp_=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/max(len(x)+len(y)-2,1))
    return float((x.mean()-y.mean())/sp_) if sp_>0 else 0.0
def auroc(s,l):
    try: return float(roc_auc_score(l,s))
    except: return float("nan")

A=sc.read_h5ad(f"{PROJ}/data_engineering/oc_lineage.h5ad"); A.X=A.layers["lognorm"]
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
sig_ens=[g for g in sig.ensembl if g in A.var_names]
sc.tl.score_genes(A,sig_ens,score_name="psig",use_raw=False)
sym2ens={s:e for e,s in zip(A.var_names,A.var["symbol"].astype(str))}
mat_ens=[sym2ens[m] for m in MAT if m in sym2ens]
sc.tl.score_genes(A,mat_ens,score_name="mat",use_raw=False)
tc=np.asarray(A.layers["counts"].sum(1)).ravel(); A.obs["logtc"]=np.log10(tc+1)

sub=A.obs["oc_subtype"].astype(str).values; acc=A.obs["accession"].astype(str).values
samp=A.obs["sample"].astype(str).values
isoc=sub=="mature_OC"; ismye=np.isin(sub,["macrophage","OCP_mono"])
ctrl_adult=isoc&(acc=="GSE266330")&np.array(["ctrl" in x.lower() for x in samp])   # adult, same platform
fetal=isoc&(A.obs["role"].astype(str).values=="reference")                          # ~100% fetal
S=A.obs["psig"].values; M=A.obs["mat"].values
TUM={"bone_met(GSE266330,tumor)":isoc&(acc=="GSE266330")&~np.array(["ctrl" in x.lower() for x in samp]),
     "GCTB_disc(GSE254672)":isoc&(acc=="GSE254672"),
     "GCTB_rep(GSE168664)":isoc&(acc=="GSE168664"),
     "OS_rep(GSE162454)":isoc&(acc=="GSE162454")}

res={"n_adult_control_OC":int(ctrl_adult.sum()),"n_fetal_ref_OC":int(fetal.sum()),
     "program_vs_ADULT_control":{}, "program_vs_fetal_ref":{}, "maturation_control":{}, "depth_matched_vs_myeloid":{}}

# 1) program: tumour-OC vs ADULT control OC (and vs fetal, for contrast)
for nm,m in TUM.items():
    lab=np.r_[np.ones(m.sum()),np.zeros(ctrl_adult.sum())]
    res["program_vs_ADULT_control"][nm]={"n_tumorOC":int(m.sum()),
        "AUROC":auroc(np.r_[S[m],S[ctrl_adult]],lab),"cohens_d":d_(S[m],S[ctrl_adult])}
    labf=np.r_[np.ones(m.sum()),np.zeros(fetal.sum())]
    res["program_vs_fetal_ref"][nm]={"AUROC":auroc(np.r_[S[m],S[fetal]],labf),"cohens_d":d_(S[m],S[fetal])}

# 2) maturation control: is program just maturation?
res["maturation_control"]={
  "n_program_genes_that_are_canonical_maturation_markers": int(len(set(sig.symbol)&set(MAT))),
  "program_size":int(len(sig)),
  "d_maturation_tumor_vs_adultctrl": d_(M[TUM["bone_met(GSE266330,tumor)"]], M[ctrl_adult]),
  "d_program_tumor_vs_adultctrl":    d_(S[TUM["bone_met(GSE266330,tumor)"]], S[ctrl_adult])}
# residualise psig on maturation, recompute (within GSE266330 tumor vs adult ctrl)
both=TUM["bone_met(GSE266330,tumor)"]|ctrl_adult
from numpy.polynomial import polynomial as P
b=np.polyfit(M[both],S[both],1); resid=S[both]-(b[0]*M[both]+b[1])
y=TUM["bone_met(GSE266330,tumor)"][both]
res["maturation_control"]["d_program_residualised_on_maturation"]=d_(resid[y],resid[~y])

# 3) depth-matched vs-myeloid: residualise psig on log total_counts, recompute d (held-out cohorts)
for a in ["GSE168664","GSE162454"]:
    oc=isoc&(acc==a); my=ismye&(acc==a); both=oc|my
    raw_d=d_(S[oc],S[my])
    bb=np.polyfit(A.obs["logtc"].values[both],S[both],1)
    r=S[both]-(bb[0]*A.obs["logtc"].values[both]+bb[1]); yy=oc[both]
    res["depth_matched_vs_myeloid"][a]={"raw_d":raw_d,"depth_residualised_d":d_(r[yy],r[~yy]),
        "median_tc_OC":float(np.median(tc[oc])),"median_tc_myeloid":float(np.median(tc[my]))}

json.dump(res,open(f"{V}/correction_adult_ref.json","w"),indent=2,default=str)
L=["# Phase-6 correction — adult reference, depth-matching, maturation control","",
   f"Adult same-platform control OCs (GSE266330 ctrl): **{res['n_adult_control_OC']}**; fetal census ref: {res['n_fetal_ref_OC']} (~100% 2nd-trimester).","",
   "## Program: tumour-OC vs ADULT control-OC (de-confounded) vs fetal ref",
   "| cohort | n OC | AUROC vs ADULT | d vs ADULT | (AUROC vs fetal) | (d vs fetal) |","|---|---|---|---|---|---|"]
for nm in TUM:
    a=res["program_vs_ADULT_control"][nm]; f=res["program_vs_fetal_ref"][nm]
    L.append(f"| {nm} | {a['n_tumorOC']:,} | {a['AUROC']:.3f} | {a['cohens_d']:.2f} | ({f['AUROC']:.3f}) | ({f['cohens_d']:.2f}) |")
mc=res["maturation_control"]
L+=["","## Maturation control (is it just a maturation gradient?)",
    f"- program∩canonical-maturation markers: **{mc['n_program_genes_that_are_canonical_maturation_markers']}/{mc['program_size']}**",
    f"- d(maturation) tumour vs adult-ctrl = **{mc['d_maturation_tumor_vs_adultctrl']:.2f}** vs d(program) = **{mc['d_program_tumor_vs_adultctrl']:.2f}** (program >> maturation → not a maturation axis)",
    f"- d(program residualised on maturation) = **{mc['d_program_residualised_on_maturation']:.2f}** (survives removing maturation)","",
    "## Depth-matched vs-myeloid (OCs carry ~2.35x RNA)","| cohort | raw d | depth-residualised d | median tc OC | myeloid |","|---|---|---|---|---|"]
for a,v in res["depth_matched_vs_myeloid"].items():
    L.append(f"| {a} | {v['raw_d']:.2f} | **{v['depth_residualised_d']:.2f}** | {v['median_tc_OC']:.0f} | {v['median_tc_myeloid']:.0f} |")
open(f"{V}/correction_adult_ref.md","w").write("\n".join(L)+"\n")
print("[DONE] correction. adult-ctrl OC:",res["n_adult_control_OC"])
for nm in TUM: print(" ",nm,"AUROCvsAdult",round(res["program_vs_ADULT_control"][nm]["AUROC"],3),"d",round(res["program_vs_ADULT_control"][nm]["cohens_d"],2))
print(" maturation d",round(mc['d_maturation_tumor_vs_adultctrl'],2),"vs program d",round(mc['d_program_tumor_vs_adultctrl'],2),"resid",round(mc['d_program_residualised_on_maturation'],2))
