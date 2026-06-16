#!/usr/bin/env python
"""
Phase-6a robustness + Phase-4 driver nomination for the 74-gene convergent
pathological-OC program.
Robustness: per-sample consistency, bootstrap CI on OC-vs-myeloid effect,
cell-cycle correlation, scrublet doublet enrichment.
Driver: expression-correlation of candidate TFs with the program + GRNBoost2
regulatory centrality (how many program genes each TF targets).
Seeds fixed.
"""
import os, json, numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D5=f"{PROJ}/discovery"; D4=f"{PROJ}/modeling"
os.makedirs(D4,exist_ok=True)

TF_PANEL=("NFATC1 FOS FOSB JUN JUNB JDP2 MITF TFEB TFE3 NFE2L2 PRDM1 BHLHE40 BHLHE41 EOMES IRF8 "
 "MAFB MAF NR4A1 NR4A2 NR4A3 KLF2 KLF4 KLF6 CEBPA CEBPB CEBPD ETV6 RUNX1 RUNX2 RUNX3 ATF3 ATF4 "
 "CREB1 XBP1 SREBF1 SREBF2 PPARG STAT1 STAT3 RELB NFKB1 SPI1 RFX8 RFX2 RFX3 FOSL1 FOSL2 BACH1 MYC "
 "EGR2 RXRA NR1H3 IKZF1 TCF4 MEF2C MEF2A NFATC2 CIITA REL").split()
S_GENES="MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 EXO1 TIPIN DSCC1 CDC6 CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8".split()
G2M_GENES="HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANLN TUBB4B GTSE1 KIF20B HJURP CDCA3 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANP32E TUBB4A GAS2L3 CENPA".split()

def boot_d(x,y,n=1000):
    rng=np.random.default_rng(0); ds=[]
    for _ in range(n):
        xi=rng.choice(x,len(x)); yi=rng.choice(y,len(y))
        sp=np.sqrt(((len(xi)-1)*xi.var(ddof=1)+(len(yi)-1)*yi.var(ddof=1))/max(len(xi)+len(yi)-2,1))
        ds.append((xi.mean()-yi.mean())/sp if sp>0 else 0)
    return float(np.percentile(ds,2.5)),float(np.percentile(ds,97.5))

def main():
    A=sc.read_h5ad(f"{PROJ}/data_engineering/oc_lineage.h5ad"); A.X=A.layers["lognorm"]
    sig=pd.read_csv(f"{D5}/convergent_signature_v2.csv")
    sig_ens=[g for g in sig.ensembl if g in A.var_names]
    sym2ens={s:e for e,s in zip(A.var_names,A.var["symbol"].astype(str))}
    sc.tl.score_genes(A,sig_ens,score_name="psig",use_raw=False)
    acc=A.obs["accession"].astype(str).values; samp=A.obs["sample"].astype(str).values
    sub=A.obs["oc_subtype"].astype(str).values
    is_oc=sub=="mature_OC"; is_mye=np.isin(sub,["macrophage","OCP_mono"])
    normal=is_oc&(A.obs["role"].astype(str).values=="reference")
    S=A.obs["psig"].values
    rep={}

    # 1) per-sample consistency (tumor samples)
    tumor_acc=["GSE266330","GSE254672","GSE168664","GSE162454","GSE268835","GSE212341"]
    persample=[]
    for s in np.unique(samp):
        m=is_oc&(samp==s);
        if m.sum()<20: continue
        a=acc[m][0]
        if a not in tumor_acc: continue
        mm=is_mye&(samp==s)
        persample.append({"sample":s,"acc":a,"n_OC":int(m.sum()),"mean_psig":float(S[m].mean()),
                          "d_vs_own_myeloid": float((S[m].mean()-S[mm].mean())/ (np.sqrt(0.5*(S[m].var()+S[mm].var()))+1e-9)) if mm.sum()>20 else None})
    ps=pd.DataFrame(persample)
    rep["per_sample"]={"n_tumor_samples":int(len(ps)),
        "frac_above_normal":float((ps.mean_psig>S[normal].mean()).mean()),
        "median_mean_psig":float(ps.mean_psig.median()),
        "frac_d_vs_myeloid_gt0.5":float((ps.d_vs_own_myeloid.dropna()>0.5).mean())}
    ps.to_csv(f"{D5}/per_sample_psig.csv",index=False)

    # 2) bootstrap CI on held-out OC vs same-tumor myeloid
    for a in ["GSE168664","GSE162454"]:
        m=is_oc&(acc==a); mm=is_mye&(acc==a)
        lo,hi=boot_d(S[m],S[mm])
        rep.setdefault("bootstrap_d_vs_myeloid",{})[a]={"ci95":[lo,hi]}

    # 3) cell cycle correlation
    sgi=[sym2ens[g] for g in S_GENES if g in sym2ens]; g2i=[sym2ens[g] for g in G2M_GENES if g in sym2ens]
    sc.tl.score_genes(A,sgi,score_name="S_sc"); sc.tl.score_genes(A,g2i,score_name="G2M_sc")
    ocm=is_oc
    rep["cellcycle"]={"corr_psig_S":float(np.corrcoef(S[ocm],A.obs["S_sc"].values[ocm])[0,1]),
                      "corr_psig_G2M":float(np.corrcoef(S[ocm],A.obs["G2M_sc"].values[ocm])[0,1])}

    # 4) scrublet doublet enrichment (per tumor accession on its lineage cells)
    try:
        import scrublet as scr
        dz=np.full(A.n_obs,np.nan)
        for a in tumor_acc:
            idx=np.where((acc==a)&(is_oc|is_mye))[0]
            if len(idx)<200: continue
            C=A.layers["counts"][idx]
            sObj=scr.Scrublet(C, random_state=0)
            try: sc_score,_=sObj.scrub_doublets(min_counts=2,min_cells=3,n_prin_comps=20); dz[idx]=sc_score
            except Exception as e: print("scrublet",a,e)
        A.obs["dub"]=dz
        oc_d=dz[is_oc]; psig_oc=S[is_oc]; valid=~np.isnan(oc_d)
        top=psig_oc>=np.nanpercentile(psig_oc[valid],90)
        rep["doublet"]={"mean_dub_topdecile_psig":float(np.nanmean(oc_d[valid&top])),
                        "mean_dub_rest":float(np.nanmean(oc_d[valid&~top]))}
    except Exception as e:
        rep["doublet"]={"err":str(e)}

    # 5) DRIVER — expression correlation + DE
    drv=[]
    ocmask=is_oc
    for tf in TF_PANEL:
        e=sym2ens.get(tf)
        if e is None or e not in A.var_names: continue
        expr=np.asarray(A[ocmask,e].X.todense()).ravel() if hasattr(A.X,'todense') else np.asarray(A[ocmask,e].X).ravel()
        if expr.std()==0: continue
        corr=float(np.corrcoef(expr,S[ocmask])[0,1])
        t=is_oc&(np.isin(acc,["GSE266330","GSE254672"])); n=normal
        et=np.asarray(A[t,e].X.todense()).ravel() if hasattr(A.X,'todense') else np.asarray(A[t,e].X).ravel()
        en=np.asarray(A[n,e].X.todense()).ravel() if hasattr(A.X,'todense') else np.asarray(A[n,e].X).ravel()
        lfc=float(et.mean()-en.mean())
        drv.append({"TF":tf,"in_program":tf in set(sig.symbol),"corr_with_psig":corr,"logdiff_tumorOC_vs_normalOC":lfc})
    drv=pd.DataFrame(drv).sort_values("corr_with_psig",ascending=False)

    # 6) GRNBoost2 regulatory centrality (best-effort)
    grn_targets={}
    try:
        from arboreto.algo import grnboost2
        hv=A.var["highly_variable"].values if "highly_variable" in A.var else np.ones(A.n_vars,bool)
        keep=sorted(set(np.where(hv)[0]) | {A.var_names.get_loc(e) for e in sig_ens} |
                    {A.var_names.get_loc(sym2ens[t]) for t in TF_PANEL if t in sym2ens})
        cellidx=np.where(is_oc|is_mye)[0]
        rng=np.random.default_rng(0)
        if len(cellidx)>12000: cellidx=rng.choice(cellidx,12000,replace=False)
        X=A.layers["lognorm"][cellidx][:,keep]
        ex=pd.DataFrame(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X),
                        columns=[A.var["symbol"].iloc[k] for k in keep])
        ex=ex.loc[:,~ex.columns.duplicated()]
        tfs=[t for t in TF_PANEL if t in ex.columns]
        net=grnboost2(expression_data=ex, tf_names=tfs, seed=0, verbose=False)
        prog=set(sig.symbol)
        net_prog=net[net.target.isin(prog)]
        grn_targets=net_prog.groupby("TF").size().sort_values(ascending=False).head(20).to_dict()
        net.to_csv(f"{D4}/grnboost2_network.csv",index=False)
    except Exception as e:
        grn_targets={"err":str(e)}
    rep["grn_program_target_count"]=grn_targets
    drv["grn_program_targets"]=drv["TF"].map(lambda t: grn_targets.get(t,0) if isinstance(grn_targets,dict) and "err" not in grn_targets else np.nan)
    drv.to_csv(f"{D4}/driver_nomination.csv",index=False)
    json.dump(rep,open(f"{D5}/robustness.json","w"),indent=2,default=str)

    # report
    L=[f"# Phase-6a robustness + Phase-4 driver nomination","",
       "## Robustness of the 74-gene program",
       f"- per-sample: {rep['per_sample']['n_tumor_samples']} tumor samples; "
       f"**{100*rep['per_sample']['frac_above_normal']:.0f}%** have mean signature > normal-OC; "
       f"**{100*rep['per_sample']['frac_d_vs_myeloid_gt0.5']:.0f}%** have d(OC vs own myeloid)>0.5 (not driven by few samples).",
       f"- bootstrap d(OC vs same-tumor myeloid) 95% CI: " + "; ".join(f"{k} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]" for k,v in rep.get("bootstrap_d_vs_myeloid",{}).items()),
       f"- cell-cycle correlation: psig~S r={rep['cellcycle']['corr_psig_S']:.2f}, psig~G2M r={rep['cellcycle']['corr_psig_G2M']:.2f} (want |r| small → not a proliferation artifact).",
       f"- doublet (scrublet) mean score top-decile-psig vs rest: {rep.get('doublet')}","",
       "## Candidate driver TFs (top by correlation with program)","| TF | in program | corr w/ psig | tumorOC−normalOC | GRN program targets |","|---|---|---|---|---|"]
    for _,r in drv.head(15).iterrows():
        L.append(f"| {r['TF']} | {'Y' if r['in_program'] else ''} | {r['corr_with_psig']:.3f} | {r['logdiff_tumorOC_vs_normalOC']:.2f} | {r['grn_program_targets']} |")
    L+=["","Top GRN regulators of the program: "+", ".join(f"{k}({v})" for k,v in list(grn_targets.items())[:10]) if isinstance(grn_targets,dict) and "err" not in grn_targets else "GRNBoost2: "+str(grn_targets)]
    open(f"{D5}/robustness_and_driver.md","w").write("\n".join(L)+"\n")
    print("[DONE] robustness + driver. top drivers:", list(drv.TF.head(8)))

if __name__=="__main__": main()
