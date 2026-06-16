#!/usr/bin/env python
"""
Phase-5 v2: AMBIENT-ROBUST convergent pathological osteoclast program.
A gene enters the program only if, in EACH discovery tumor, it is BOTH
  (i) UP in mature-OC vs the SAME tumor's myeloid (macrophage+OCP)  [ambient control]
  (ii) UP in tumor mature-OC vs normal-reference OC                 [disease effect]
minus foreign-lineage markers and technical-artifact gene families.
Convergence = intersection across discovery tumors (bone-met + GCTB).
Held-out test (one peek): independent GCTB (GSE168664) + held-out DISEASE
osteosarcoma (GSE162454) — and crucially the OC-vs-same-tumor-myeloid effect,
which a pure ambient signal cannot reproduce.
Seeds fixed.
"""
import os, json, numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D5=f"{PROJ}/discovery"

FOREIGN = set("""CD3D CD3E CD3G TRAC TRBC1 TRBC2 IL7R CD8A CD8B CD4 CCL5 LCK CD2 CD28 CD27 CD7
NKG7 GNLY KLRD1 KLRF1 NCAM1 NCR1 KLRB1 GZMB GZMK PRF1 KLRC1
CD79A CD79B MS4A1 CD19 MZB1 IGHG1 IGHG3 IGHM IGKC IGLC1 JCHAIN DERL3 XBP1 TCL1A
PECAM1 VWF CLDN5 CDH5 KDR FLT1 EGFL7 RAMP2 PLVAP AQP1
COL1A1 COL1A2 COL3A1 COL6A1 COL6A2 COL6A3 DCN LUM PDGFRA PDGFRB THY1 FN1 SPARC MGP POSTN
CTHRC1 ACTA2 TAGLN MYH11 PDPN GSN APOD CFD
RUNX2 SP7 BGLAP IBSP ALPL SPP1 CADM1 SATB2
EPCAM KRT8 KRT18 KRT19 KRT5 KRT17 CDH1 ELF3 KRT14 KRT7
HBB HBA1 HBA2 HBD GYPA ALAS2 AHSP PPBP PF4 GP9 ITGA2B TUBB1
CD34 GATA2 KIT MPO ELANE PRTN3""".split())

def is_artifact(s):
    s=str(s)
    if s.startswith(("MT-","RPL","RPS","MRPL","MRPS","HBA","HBB","HBD","HSP","DNAJ","MTRNR")): return True
    ieg={"FOS","FOSB","JUN","JUNB","JUND","EGR1","ATF3","DUSP1","IER2","MALAT1","NEAT1","XIST","TSIX"}
    cc={"MKI67","TOP2A","CENPF","PCNA","MCM2","MCM3","MCM6","CCNB1","CCNB2","CDK1","UBE2C","BIRC5","TYMS","CENPA","SMC4","HMGB2","STMN1"}
    return s in ieg or s in cc

DISC={"bone_met":"GSE266330","gctb_disc":"GSE254672"}
REPL={"gctb_rep":"GSE168664","os_rep":"GSE162454"}

def de_up(adata, a_mask, b_mask, lfc, pct):
    A=adata[a_mask|b_mask].copy()
    A.obs["g"]=np.where(a_mask[a_mask|b_mask],"A","B"); A.obs["g"]=A.obs["g"].astype("category")
    sc.tl.rank_genes_groups(A,"g",groups=["A"],reference="B",method="wilcoxon",pts=True)
    r=sc.get.rank_genes_groups_df(A,group="A")
    r["symbol"]=adata.var.loc[r["names"],"symbol"].values
    r=r[(r.logfoldchanges>lfc)&(r.pvals_adj<0.05)&(r.get("pct_nz_group",1)>pct)]
    r=r[~r["symbol"].map(is_artifact)]; r=r[~r["symbol"].isin(FOREIGN)]
    return r.set_index("names")

def cohens_d(x,y):
    nx,ny=len(x),len(y);
    if nx<3 or ny<3: return float("nan")
    sp=np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/max(nx+ny-2,1))
    return float((x.mean()-y.mean())/sp) if sp>0 else 0.0

def auroc(s,l):
    from sklearn.metrics import roc_auc_score
    try: return float(roc_auc_score(l,s))
    except Exception: return float("nan")

def main():
    A=sc.read_h5ad(f"{PROJ}/data_engineering/oc_lineage.h5ad"); A.X=A.layers["lognorm"]
    acc=A.obs["accession"].astype(str).values
    sub=A.obs["oc_subtype"].astype(str).values
    is_oc=sub=="mature_OC"; is_mye=np.isin(sub,["macrophage","OCP_mono"])
    normalOC=is_oc & (A.obs["role"].astype(str).values=="reference")
    print("matureOC",is_oc.sum(),"myeloid",is_mye.sum(),"normalOC",normalOC.sum(),flush=True)

    drop={}; cand={}
    for k,a in DISC.items():
        ocm=is_oc&(acc==a); mym=is_mye&(acc==a)
        e_i=de_up(A, ocm, mym, lfc=0.5, pct=0.25)      # OC vs same-tumor myeloid (ambient control)
        e_ii=de_up(A, ocm, normalOC, lfc=0.25, pct=0.10) # tumor-OC vs normal-OC (disease)
        c=set(e_i.index)&set(e_ii.index)
        cand[k]=(c,e_i,e_ii)
        drop[k]={"ocVSmyeloid_up":len(e_i),"tumorVSnormal_up":len(e_ii),"intersection":len(c),
                 "n_OC":int(ocm.sum()),"n_myeloid":int(mym.sum())}
        print(f"[{k} {a}] OCvsMye={len(e_i)} TvsN={len(e_ii)} cand={len(c)}",flush=True)
    conv=sorted(set.intersection(*[cand[k][0] for k in DISC]))
    rows=[]
    for g in conv:
        sym=A.var.loc[g,"symbol"]
        lfc_i=np.mean([cand[k][1].loc[g,"logfoldchanges"] for k in DISC])
        lfc_ii=np.mean([cand[k][2].loc[g,"logfoldchanges"] for k in DISC])
        rows.append({"ensembl":g,"symbol":sym,"mean_lfc_OCvsMyeloid":float(lfc_i),"mean_lfc_TumorVsNormal":float(lfc_ii)})
    sig=pd.DataFrame(rows).sort_values("mean_lfc_TumorVsNormal",ascending=False)
    sig.to_csv(f"{D5}/convergent_signature_v2.csv",index=False)
    print(f"[CONVERGENT v2] {len(sig)} ambient-robust genes",flush=True)
    print("top30:",list(sig.symbol.head(30)),flush=True)

    if len(sig)>=5: sc.tl.score_genes(A,list(sig.ensembl),score_name="psig",use_raw=False)
    else: A.obs["psig"]=0.0
    S=A.obs["psig"].values

    def evalc(a):
        ocm=is_oc&(acc==a); mym=is_mye&(acc==a)
        lab=np.r_[np.ones(ocm.sum()),np.zeros(normalOC.sum())]
        out={"accession":a,"n_OC":int(ocm.sum()),
             "AUROC_OC_vs_normalOC":auroc(np.r_[S[ocm],S[normalOC]],lab),
             "d_OC_vs_normalOC":cohens_d(S[ocm],S[normalOC]),
             "d_OC_vs_sameTumorMyeloid":cohens_d(S[ocm],S[mym]),  # ambient-robust replication
             "mean_OC":float(S[ocm].mean()),"mean_myeloid":float(S[mym].mean()),"mean_normalOC":float(S[normalOC].mean())}
        return out
    res={"n_convergent_genes":int(len(sig)),"filter_dropoff":drop,
         "top_genes":list(sig.symbol.head(50)),
         "discovery":{k:evalc(a) for k,a in DISC.items()},
         "replication_heldout":{k:evalc(a) for k,a in REPL.items()}}
    # DE recovery in held-out
    for k,a in REPL.items():
        ocm=is_oc&(acc==a); mym=is_mye&(acc==a)
        try:
            ei=de_up(A,ocm,mym,0.5,0.25)
            res["replication_heldout"][k]["frac_recovered_OCvsMyeloid"]=float(len(set(ei.index)&set(sig.ensembl))/max(len(sig),1))
        except Exception as ex: res["replication_heldout"][k]["err"]=str(ex)
    json.dump(res,open(f"{D5}/convergence_result_v2.json","w"),indent=2,default=str)

    # figure
    try:
        cats=[("normalOC",normalOC)]
        for k,a in {**DISC,**REPL}.items():
            cats.append((f"{k}-OC",is_oc&(acc==a))); cats.append((f"{k}-Mye",is_mye&(acc==a)))
        plt.figure(figsize=(11,4)); plt.boxplot([S[m] for _,m in cats],showfliers=False)
        plt.xticks(range(1,len(cats)+1),[c for c,_ in cats],rotation=45,ha="right")
        plt.ylabel("ambient-robust pathological-OC signature (v2)")
        plt.title("v2 signature: OC vs same-tumor myeloid vs normal-OC")
        plt.tight_layout(); plt.savefig(f"{D5}/signature_v2_by_cohort.png",dpi=130)
    except Exception as ex: print("fig",ex)

    # report
    L=[f"# Phase-5 v2 — ambient-robust convergent pathological-OC program","",
       f"- **{len(sig)} genes** (down from 1,703 naive) after interaction + foreign-marker + artifact filters.",
       f"- top: {', '.join(sig.symbol.head(30))}","",
       "## Filter drop-off (per discovery tumor)","| tumor | OC-vs-myeloid up | tumor-vs-normal up | intersection |","|---|---|---|---|"]
    for k,v in drop.items(): L.append(f"| {k} | {v['ocVSmyeloid_up']} | {v['tumorVSnormal_up']} | {v['intersection']} |")
    L+=["","## Held-out replication — KEY = d(OC vs same-tumor myeloid) (ambient cannot reproduce this)",
        "| cohort | n OC | AUROC vs normalOC | d vs normalOC | **d vs same-tumor myeloid** | %recovered |","|---|---|---|---|---|---|"]
    for k,a in REPL.items():
        v=res["replication_heldout"][k]
        L.append(f"| {k} ({a}) | {v['n_OC']:,} | {v['AUROC_OC_vs_normalOC']:.3f} | {v['d_OC_vs_normalOC']:.2f} | **{v['d_OC_vs_sameTumorMyeloid']:.2f}** | {v.get('frac_recovered_OCvsMyeloid',float('nan')):.2f} |")
    L+=["","Gate: real convergent program if d(OC vs same-tumor myeloid) > 0.5 in held-out OS AND independent GCTB, with non-trivial recovery. The vs-myeloid contrast is the ambient-proof test."]
    open(f"{D5}/candidates_v2.md","w").write("\n".join(L)+"\n")
    print("[DONE v2] wrote candidates_v2.md")

if __name__=="__main__": main()
