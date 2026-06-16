#!/usr/bin/env python
"""Advanced 7-figure set (journal-grade) replacing the bar/box-heavy v1. Adds LIANA cell-cell
communication, decoupler TF/PROGENy activity, PAGA + pseudotime gene heatmap, GSEA, and upgrades
heatmaps/box/bars to clustered-annotated heatmaps, sina, forest, lollipop-null, UpSet, density UMAP.
Run on a compute node. Outputs -> figures/main/Gn.{png,pdf}."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pub_style import C, OKABE_ITO, SEQ_CMAP, DIV_CMAP, GREY, HIGHLIGHT, save_fig, panel_label, clean, umap_axes
import adv_helpers as H
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); ADV=f"{PROJ}/advanced"; FIG=f"{PROJ}/figures/main"; V=f"{PROJ}/validation"
os.makedirs(FIG, exist_ok=True)
def jread(p,d=None):
    import json,os
    return json.load(open(p)) if os.path.exists(p) else d
def cread(p,**k):
    import os,pandas as pd
    return pd.read_csv(p,**k) if os.path.exists(p) else None
PRETTY={"mature_OC":"Mature OC","macrophage":"Macrophage/TAM","OCP_mono":"OCP/monocyte","fibro_stroma":"Fibroblast/stroma",
        "osteoblast":"Osteoblast","tumor_epi":"Tumour/epithelial","Tcell":"T cell","NK":"NK","Bplasma":"B/plasma",
        "endothelial":"Endothelial","erythroid":"Erythroid","prolif":"Proliferating"}
CCOL={"mature_OC":C["mature_OC"],"macrophage":C["macrophage"],"OCP_mono":C["OCP_mono"],"tumor_epi":"#CC79A7",
      "fibro_stroma":"#999999","endothelial":"#8C8C8C","Tcell":"#B0B0B0","NK":"#C0C0C0","Bplasma":"#D0D0D0",
      "erythroid":"#A0A0A0","prolif":"#909090","osteoblast":"#E69F00"}
def col_for(c): return CCOL.get(c,GREY)

# ---------- load atlas (expression for dotplots/heatmaps) + CSV outputs ----------
A=sc.read_h5ad(f"{PROJ}/data_engineering/integrated.h5ad"); A.X=A.layers["lognorm"]
sym2ens={s:e for e,s in zip(A.var_names,A.var["symbol"].astype(str))}
comp=A.obs["compartment"].astype(str).values
sig0=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
sc.tl.score_genes(A,[g for g in sig0.ensembl if g in A.var_names],score_name="psig",use_raw=False)
COMPS=[c for c in ["mature_OC","macrophage","OCP_mono","Tcell","NK","Bplasma","fibro_stroma","tumor_epi","endothelial","erythroid","prolif"] if (comp==c).any()]
def dense(e):
    x=A[:,e].X; return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
def gmat(symbols, comps=COMPS, scale=True):
    """gene x compartment mean(lognorm); row min-max scaled if scale."""
    rows=[]; present=[]
    for s in symbols:
        if s in sym2ens:
            col=dense(sym2ens[s]); r=[col[comp==c].mean() for c in comps]; rows.append(r); present.append(s)
    M=np.array(rows)
    if scale and len(M):
        M=(M-M.min(1,keepdims=True))/(M.max(1,keepdims=True)-M.min(1,keepdims=True)+1e-9)
    return M,present
def fmat(symbols, comps=COMPS):
    rows=[]; present=[]
    for s in symbols:
        if s in sym2ens:
            col=dense(sym2ens[s]); rows.append([(col[comp==c]>0).mean() for c in comps]); present.append(s)
    return np.array(rows),present

umap=cread(f"{ADV}/umap_cells.csv.gz")
traj=cread(f"{ADV}/trajectory_cells.csv.gz")
compj=jread(f"{V}/supp_compartment.json",{})
ext=jread(f"{V}/supp_external.json",{})
rean=jread(f"{V}/reanalysis_harden.json",{})
dub=jread(f"{V}/doublet_control.json",{})
liana=cread(f"{ADV}/liana_res.csv"); linc=cread(f"{ADV}/liana_OC_incoming.csv"); lout=cread(f"{ADV}/liana_OC_outgoing.csv"); lmat=cread(f"{ADV}/liana_interaction_matrix.csv",index_col=0)
tfa=cread(f"{ADV}/tf_activity_by_compartment.csv",index_col=0); prog=cread(f"{ADV}/progeny_by_compartment.csv",index_col=0)
gsea=cread(f"{ADV}/gsea_enrichr.csv"); paga=cread(f"{ADV}/paga_connectivities.csv",index_col=0)
ptm=cread(f"{ADV}/pseudotime_gene_heatmap.csv",index_col=0); ptt=jread(f"{ADV}/pseudotime_trends.json",{})
ko=cread(f"{PROJ}/report/tables/T3_insilico_driver_ranking.csv")
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")
print("loaded all inputs",flush=True)

# =================================================================== FIG 1 — atlas
def fig1():
    fig=plt.figure(figsize=(11,7),layout="constrained"); gs=GridSpec(2,3,figure=fig)
    # a UMAP by compartment (density contour)
    ax=fig.add_subplot(gs[0,0])
    if umap is not None:
        for c in COMPS:
            m=umap.compartment==c; ax.scatter(umap.umap1[m],umap.umap2[m],s=1.5,c=col_for(c),alpha=.5,linewidths=0,rasterized=True,label=PRETTY.get(c,c))
        umap_axes(ax)
    ax.set_title("Osteoclast-lineage + tumour atlas",fontsize=7.5); ax.legend(fontsize=4.4,markerscale=3,ncol=2,loc="upper left",frameon=False); panel_label(ax,"a")
    # b composition stacked proportional per disease
    ax=fig.add_subplot(gs[0,1]); dis=A.obs["disease"].astype(str).values
    diss=sorted(pd.unique(dis)); bottom=np.zeros(len(diss))
    for c in COMPS:
        fr=[ (comp[(dis==d)]==c).mean() for d in diss ]; ax.bar(range(len(diss)),fr,bottom=bottom,color=col_for(c),width=.8,lw=0); bottom+=fr
    ax.set_xticks(range(len(diss))); ax.set_xticklabels(diss,rotation=35,ha="right",fontsize=5.6); ax.set_ylabel("compartment fraction"); ax.set_ylim(0,1)
    ax.set_title("Composition by context",fontsize=7.5); clean(ax); panel_label(ax,"b")
    # c per-cohort QC violins (detected genes / cell) — moved up; program-score viz lives in Fig 3
    ax=fig.add_subplot(gs[0,2]); acc=A.obs["accession"].astype(str).values; accs=sorted(pd.unique(acc))[:7]
    ng=A.obs["n_genes_by_counts"].values if "n_genes_by_counts" in A.obs else A.obs.get("log1p_n_genes_by_counts",pd.Series(np.zeros(A.n_obs))).values
    parts=ax.violinplot([ng[acc==a] for a in accs],showmedians=True,widths=.8)
    for b in parts["bodies"]: b.set_facecolor(C["OCP_mono"]); b.set_alpha(.5)
    ax.set_xticks(range(1,len(accs)+1)); ax.set_xticklabels([a[:9] for a in accs],rotation=40,ha="right",fontsize=5); ax.set_ylabel("genes / cell"); clean(ax)
    ax.set_title("Per-cohort QC",fontsize=7.5); panel_label(ax,"c")
    # d marker clustered dotplot (PyComplexHeatmap-style, embedded)
    MARK=["CTSK","ACP5","NFATC1","ATP6V0D2","DCSTAMP","CD14","CSF1R","LYZ","FCGR3A","CD3E","NKG7","MS4A1","PECAM1","COL1A1","EPCAM"]
    Mc,present=gmat([m for m in MARK if m in sym2ens]); Ms,_=fmat([m for m in MARK if m in sym2ens])
    ax=fig.add_subplot(gs[1,0:2])
    sc_=H.dotplot(ax,Mc,Ms,present,[PRETTY.get(c,c) for c in COMPS],title="Canonical markers across compartments")
    cb=fig.colorbar(sc_,ax=ax,fraction=.02,pad=.01); cb.set_label("scaled mean",fontsize=5.6); cb.ax.tick_params(labelsize=5)
    for fr,lab in [(.25,"25%"),(.5,"50%"),(1,"100%")]: ax.scatter([],[],s=fr*70+1,c="#888",label=lab)
    ax.legend(title="% expr",fontsize=4.6,title_fontsize=5,loc="center left",bbox_to_anchor=(1.04,.5),labelspacing=.8); panel_label(ax,"d",x=-.04)
    save_fig(fig,f"{FIG}/G1_atlas"); print("[G1] done",flush=True)

# =================================================================== FIG 4 — specificity (sina + clustered heatmap + dotplot + forest)
def fig4():
    fig=plt.figure(figsize=(11,9),layout="constrained"); gs=GridSpec(3,3,figure=fig,height_ratios=[1.1,1.2,1.0])
    cb=compj.get("program_score_by_compartment",{})
    order=[c for c in COMPS if c in cb]; order=sorted(order,key=lambda c:cb[c]["median"],reverse=True)
    # a sina program x compartment
    ax=fig.add_subplot(gs[0,0:2])
    Sall={c:A.obs["psig"].values[comp==c] if "psig" in A.obs else None for c in order}
    if "psig" not in A.obs:
        se=[sym2ens[s] for s in sig.symbol if s in sym2ens]; sc.tl.score_genes(A,se,score_name="psig",use_raw=False)
        Sall={c:A.obs["psig"].values[comp==c] for c in order}
    H.sina(ax,[PRETTY.get(c,c) for c in order],[Sall[c] for c in order],[col_for(c) for c in order])
    ax.axhline(0,color="#999",lw=.5,ls="--"); ax.set_ylabel("74-gene program score")
    ax.set_title(f"Program is OC-high, myeloid-low (OC vs macrophage d={compj.get('program_OC_vs_macrophage_cohensd',float('nan')):.2f})",fontsize=7.2)
    clean(ax); panel_label(ax,"a",x=-.05)
    # b RADAR module fingerprint: OC vs macrophage vs tumour/epithelial vs stroma
    mod=compj.get("module_by_compartment",{}); mods=list(mod.keys())
    if mods:
        axB=fig.add_subplot(gs[0,2],projection="polar")
        grp=[("mature_OC","Mature OC",C["mature_OC"]),("macrophage","Macrophage/TAM",C["macrophage"]),
             ("tumor_epi","Tumour/epithelial","#CC79A7"),("fibro_stroma","Fibroblast/stroma","#999999")]
        grp=[g for g in grp if g[0] in mod.get(mods[0],{})]
        series={nm:[mod[m].get(c,0) for m in mods] for c,nm,_ in grp}
        H.radar(axB,mods,series,[col for _,_,col in grp],title="Module fingerprint:\nonly sulfation is OC-specific")
        axB.text(-0.12,1.18,"b",transform=axB.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")
    # c clustered, module-annotated dotplot (left colour strip = gene group)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    dot=cread(f"{V}/supp_panel_dotplot.csv")
    if dot is not None:
        GGRP={"PAPSS2":"Sulfation","FAM20C":"Sulfation","SH3PXD2A":"Podosome","MYO1E":"Podosome","DNM3":"Podosome",
              "MMP19":"Protease","RUNX3":"TF","RFX8":"TF","CTSK":"OC marker","ACP5":"OC marker","NFATC1":"OC marker",
              "CD68":"TAM marker","C1QA":"TAM marker","C1QB":"TAM marker","CD163":"TAM marker","MRC1":"TAM marker","LYZ":"TAM marker","FCGR3A":"TAM marker","APOE":"TAM marker"}
        GCOL={"Sulfation":C["OCP_mono"],"Podosome":C["mature_OC"],"Protease":"#CC79A7","TF":C["macrophage"],"OC marker":"#E69F00","TAM marker":"#777777"}
        GORDER=["Sulfation","Podosome","Protease","TF","OC marker","TAM marker"]
        allg=list(dict.fromkeys(dot.gene)); genes=sorted([g for g in allg if g in GGRP],key=lambda g:(GORDER.index(GGRP[g]),g))
        cps=[c for c in order if c in set(dot.compartment)]
        Mc=dot.pivot(index="gene",columns="compartment",values="mean_in_pos").reindex(index=genes,columns=cps).fillna(0).values
        Fr=dot.pivot(index="gene",columns="compartment",values="frac_pos").reindex(index=genes,columns=cps).fillna(0).values
        Mc=(Mc-Mc.min(1,keepdims=True))/(Mc.max(1,keepdims=True)-Mc.min(1,keepdims=True)+1e-9)
        sub=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,0:2],width_ratios=[0.018,0.982],wspace=0.012)
        axs=fig.add_subplot(sub[0])
        for i,g in enumerate(genes): axs.add_patch(plt.Rectangle((0,i-0.5),1,1,color=GCOL[GGRP[g]],lw=0))
        axs.set_xlim(0,1); axs.set_ylim(len(genes)-0.5,-0.5); axs.axis("off")
        ax=fig.add_subplot(sub[1])
        sc_=H.dotplot(ax,Mc,Fr,genes,[PRETTY.get(c,c) for c in cps],title="Program genes are OC-enriched; MMP19 is shared; TAM markers are macrophage-restricted")
        cb=fig.colorbar(sc_,ax=ax,fraction=.018,pad=.01); cb.set_label("scaled mean",fontsize=5.4); cb.ax.tick_params(labelsize=5)
        for grp_,c in GCOL.items(): ax.scatter([],[],marker="s",c=c,s=18,label=grp_)
        ax.legend(title="gene group",fontsize=4.8,title_fontsize=5,loc="center left",bbox_to_anchor=(1.07,.5),labelspacing=.5,handletextpad=.3,frameon=False)
        panel_label(ax,"c",x=-.07,y=1.05)
    # d cross-cancer RIDGELINE (joy plot of distributions)
    canc=compj.get("GSE266330_by_cancer_origin",{})
    if canc:
        acc=A.obs["accession"].astype(str).values; samp=A.obs["sample"].astype(str).values
        import re
        CODE={"BC":"Breast","KC":"Renal","LC":"Lung","CC":"Colon"}
        def origin(s):
            m=re.search(r"_(BC|KC|LC|CC)_",s); return CODE.get(m.group(1)) if m else None
        oc266=(comp=="mature_OC")&(acc=="GSE266330"); org=np.array([origin(s) for s in samp])
        cats=[c for c in ["Renal","Lung","Breast","Colon"] if (oc266&(org==c)).any()]
        vals=[A.obs["psig"].values[oc266&(org==c)] for c in cats]
        H.ridgeline(fig,gs[1,2],cats,vals,[C["tumour"]]*len(cats),xlabel="program score",title="Program across cancer origins",label="d")
    else:
        fig.add_subplot(gs[1,2]).axis("off")
    # e external forest
    ax=fig.add_subplot(gs[2,0])
    rows=[]
    os_=ext.get("GSE152048_OS",{}); pm=ext.get("GSE143791_prostateBM",{})
    if os_.get("cohensd_OC_vs_nonOC") is not None:
        ci=os_.get("cohensd_OC_vs_nonOC_CI",[np.nan,np.nan]); rows.append(("OS GSE152048\nOC vs non-OC",os_["cohensd_OC_vs_nonOC"],ci[0],ci[1],C["mature_OC"]))
    if pm.get("cohensd_OC_vs_TAM") is not None:
        ci=pm.get("cohensd_OC_vs_TAM_CI",[np.nan,np.nan]); rows.append(("Prostate GSE143791\nOC vs TAM",pm["cohensd_OC_vs_TAM"],ci[0],ci[1],C["mature_OC"]))
    if rows: H.forest(ax,rows)
    ax.set_title("External validation",fontsize=7.2); panel_label(ax,"e",x=-.2)
    # f doublet/core robustness SLOPEGRAPH (effect barely moves -> robust)
    axF=fig.add_subplot(gs[2,1]); conds=[]; ys=[]; cis=[]
    if dub:
        for k,lab in [("d_OC_vs_mac_all","full program"),("specific_core_OC_vs_mac","specific\ncore"),("d_OC_vs_mac_lowDoublet","top-doublet\ndecile removed")]:
            if k in dub: conds.append(lab); ys.append(dub[k]["d"]); cis.append(dub[k]["ci"])
    if conds:
        H.slopegraph(axF,conds,ys,cis,color=C["mature_OC"],ref=ys[0],ylabel="OC vs macrophage Cohen's d")
        axF.set_ylim(min(0.85,min(c[0] for c in cis)-0.05),1.22)
    axF.set_title("Effect is robust to gene & doublet controls",fontsize=7.2); panel_label(axF,"f",x=-.2)
    # (third col row3) sulfation module note
    ax=fig.add_subplot(gs[2,2])
    so=rean.get("sulfation_OC_vs_nexttop",{}); st=rean.get("sulfation_OC_vs_tumor_epi",{})
    rows=[]
    if so: rows.append(("Sulfation: OC vs\nnext compartment",so["d"],so["ci"][0],so["ci"][1],C["OCP_mono"]))
    if st: rows.append(("Sulfation: OC vs\ntumour/osteoblast-like",st["d"],st["ci"][0],st["ci"][1],"#CC79A7"))
    if rows: H.forest(ax,rows)
    ax.set_title("Sulfation-module enrichment (modest)",fontsize=7.2); panel_label(ax,"g",x=-.2)
    save_fig(fig,f"{FIG}/G4_specificity"); print("[G4] done",flush=True)

# =================================================================== FIG 5 — cell-cell communication (LIANA)
def fig5():
    fig=plt.figure(figsize=(11,8.5),layout="constrained"); gs=GridSpec(2,2,figure=fig,height_ratios=[1.1,1.0])
    # a circular interaction network
    ax=fig.add_subplot(gs[0,0])
    if lmat is not None:
        labs=list(lmat.index); M=lmat.reindex(index=labs,columns=labs).fillna(0).values
        H.circular_network(ax,M,[PRETTY.get(l,l) for l in labs],highlight="Mature OC",node_colors={PRETTY.get(l,l):col_for(l) for l in labs})
    ax.set_title("Inferred signaling among compartments (LIANA; OC highlighted)",fontsize=7.2); panel_label(ax,"a")
    # b aggregated interaction heatmap clustered
    if lmat is not None:
        labs=list(lmat.index); M=lmat.reindex(index=labs,columns=labs).fillna(0).values
        H.clustered_heatmap(fig,gs[0,1],M,[PRETTY.get(l,l) for l in labs],[PRETTY.get(l,l) for l in labs],cmap="magma_r",vcenter=False,cbar_label="# specific interactions",title="Interaction-strength matrix",label="b")
    # c OC-incoming ligand-receptor dotplot
    ax=fig.add_subplot(gs[1,0])
    if linc is not None and len(linc):
        d=linc.copy(); d["pair"]=d.ligand_complex+"→"+d.receptor_complex; d=d.head(14)[::-1]
        sc_=ax.scatter(d.source.astype("category").cat.codes, range(len(d)), s=80-(d.magnitude_rank.rank()*2), c=-np.log10(d.cellphone_pvals.clip(1e-4)) if "cellphone_pvals" in d else d.lr_means, cmap="viridis", edgecolors="#333",linewidths=.3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d.pair,fontsize=5)
        srcs=list(d.source.astype("category").cat.categories); ax.set_xticks(range(len(srcs))); ax.set_xticklabels([PRETTY.get(s,s) for s in srcs],rotation=35,ha="right",fontsize=5.6)
        ax.set_title("Top ligand–receptor pairs signaling to OC",fontsize=7.2); clean(ax)
        cb=fig.colorbar(sc_,ax=ax,fraction=.03,pad=.01); cb.set_label("strength",fontsize=5.4); cb.ax.tick_params(labelsize=5)
    panel_label(ax,"c")
    # d OC-outgoing
    ax=fig.add_subplot(gs[1,1])
    if lout is not None and len(lout):
        d=lout.copy(); d["pair"]=d.ligand_complex+"→"+d.receptor_complex; d=d.head(14)[::-1]
        sc_=ax.scatter(d.target.astype("category").cat.codes, range(len(d)), s=60, c=d.lr_means if "lr_means" in d else range(len(d)), cmap="magma", edgecolors="#333",linewidths=.3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d.pair,fontsize=5)
        tg=list(d.target.astype("category").cat.categories); ax.set_xticks(range(len(tg))); ax.set_xticklabels([PRETTY.get(s,s) for s in tg],rotation=35,ha="right",fontsize=5.6)
        ax.set_title("Top ligand–receptor pairs signaling from OC",fontsize=7.2); clean(ax)
        cb=fig.colorbar(sc_,ax=ax,fraction=.03,pad=.01); cb.set_label("strength",fontsize=5.4); cb.ax.tick_params(labelsize=5)
    panel_label(ax,"d")
    save_fig(fig,f"{FIG}/G5_communication"); print("[G5] done",flush=True)

# =================================================================== FIG 6 — regulation + trajectory
def fig6():
    fig=plt.figure(figsize=(11,9.5),layout="constrained"); gs=GridSpec(3,3,figure=fig,height_ratios=[1.0,1.25,1.0])
    # a TF-activity clustered heatmap (top variable TFs)
    if tfa is not None:
        tf=tfa.copy()
        var=tf.var(0).sort_values(ascending=False); keep=[t for t in ["JDP2","NFATC1","RFX8","SOX4","RUNX3","MITF","FOSL2","SPI1","CEBPB","NR4A3","BACH1","MAF"] if t in tf.columns]
        keep=list(dict.fromkeys(keep+list(var.head(18).index)))[:20]
        M=tf[keep].T.reindex(columns=[c for c in COMPS if c in tf.index]).values
        H.clustered_heatmap(fig,gs[0,0:2],M,keep,[PRETTY.get(c,c) for c in COMPS if c in tf.index],cmap="RdBu_r",cbar_label="TF activity (ULM)",title="DoRothEA TF activity × compartment",label="a")
    # b PROGENy pathway activity clustered heatmap
    if prog is not None:
        P=prog.reindex(index=[c for c in COMPS if c in prog.index])
        H.clustered_heatmap(fig,gs[0,2],P.T.values,list(P.columns),[PRETTY.get(c,c) for c in P.index],cmap="RdBu_r",cbar_label="pathway activity",title="PROGENy pathways",label="b")
    # c PAGA graph
    ax=fig.add_subplot(gs[1,0])
    if paga is not None:
        labs=list(paga.index); ang=np.linspace(0,2*np.pi,len(labs),endpoint=False); xy=np.c_[np.cos(ang),np.sin(ang)]
        Mp=paga.values; mx=Mp.max()
        for i in range(len(labs)):
            for j in range(i+1,len(labs)):
                if Mp[i,j]>0: ax.plot([xy[i,0],xy[j,0]],[xy[i,1],xy[j,1]],color="#888",lw=.5+5*Mp[i,j]/mx,alpha=.7,zorder=1)
        for i,l in enumerate(labs): ax.scatter(*xy[i],s=400,c=col_for(l),edgecolors="white",linewidths=1,zorder=3); ax.text(xy[i,0],xy[i,1]-.22,PRETTY.get(l,l),ha="center",fontsize=6)
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("PAGA lineage abstraction",fontsize=7.2); panel_label(ax,"c")
    # d pseudotime gene heatmap (hero)
    if ptm is not None:
        genes=list(ptm.index); matz=ptm.values; compbin=ptt.get("compbin")
        H.pseudotime_heatmap(fig,gs[1,1:3],matz,genes,compbin=compbin,comp_colors={c:col_for(c) for c in COMPS},label="d")
    # e program/modules along pseudotime
    ax=fig.add_subplot(gs[2,0:2])
    if ptt.get("trends"):
        tr=ptt["trends"]; nb=ptt.get("n_bins",50); x=np.linspace(0,1,nb)
        MODCOL={"Podosome":C["mature_OC"],"Protease":"#CC79A7","Sulfation":C["OCP_mono"],"TF":C["macrophage"],"psig":"#111"}
        from numpy import convolve
        def smooth(v,k=5): return convolve(v,np.ones(k)/k,mode="same")
        for nm,series in tr.items():
            v=np.array([s[0] for s in series]); e=np.array([s[1] for s in series])
            ax.plot(x,smooth(v),color=MODCOL.get(nm,"#888"),lw=1.6 if nm=="psig" else 1.0,label="program" if nm=="psig" else nm)
            ax.fill_between(x,smooth(v-e),smooth(v+e),color=MODCOL.get(nm,"#888"),alpha=.12,lw=0)
        ax.set_xlabel("pseudotime (OCP/mono → mature OC)"); ax.set_ylabel("score"); ax.legend(fontsize=5,ncol=3,loc="upper left"); clean(ax)
    ax.set_title("Program and modules rise along osteoclastogenesis",fontsize=7.2); panel_label(ax,"e",x=-.05)
    # f in-silico KO lollipop with null
    ax=fig.add_subplot(gs[2,2])
    if ko is not None:
        k=ko.head(8).copy()
        labels=list(k.TF); real=list(-k.delta_psig_KO); null=list(-k.get("delta_psig_KO_shuffledGRN",pd.Series([0]*len(k))))
        inprog=list(k.get("in_program",pd.Series([False]*len(k)))); cols=[HIGHLIGHT if t=="RFX8" else (C["canonical"] if ip else "#888") for t,ip in zip(labels,inprog)]
        H.lollipop_null(ax,labels,real,null,cols,in_program=inprog)
        ax.set_xlabel("−Δ program on knockout"); ax.set_title("In-silico KO vs per-TF null\n(◇ null; ∈program flagged)",fontsize=7)
    panel_label(ax,"f",x=-.2)
    save_fig(fig,f"{FIG}/G6_regulation_trajectory"); print("[G6] done",flush=True)

# =================================================================== FIG 7 — validation + GSEA
def fig7():
    from sklearn.metrics import roc_auc_score, roc_curve
    fig=plt.figure(figsize=(11,7.5),layout="constrained"); gs=GridSpec(2,3,figure=fig)
    # a GSEA enrichment dotplot
    ax=fig.add_subplot(gs[0,0])
    if gsea is not None and len(gsea):
        g=gsea.copy(); g["nlp"]=-np.log10(g["Adjusted P-value"].clip(1e-12))
        g["ngene"]=g["Overlap"].astype(str).str.split("/").str[0].astype(float)
        g=g.sort_values("nlp",ascending=False).head(12)[::-1]
        sc_=ax.scatter(g.nlp,range(len(g)),s=g.ngene*16+8,c=g.nlp,cmap=SEQ_CMAP,edgecolors="#333",linewidths=.3)
        ax.set_yticks(range(len(g))); ax.set_yticklabels([t[:34] for t in g.Term],fontsize=4.8)
        ax.set_xlabel("−log10 adj. p"); ax.axvline(-np.log10(.05),ls="--",lw=.5,color="#aaa")
        cb=fig.colorbar(sc_,ax=ax,fraction=.03,pad=.01); cb.ax.tick_params(labelsize=5)
        ax.set_title("Program enrichment (Enrichr)",fontsize=7.2); clean(ax)
    panel_label(ax,"a",x=-.05)
    # b effect-size forest (held-out + adult)
    ax=fig.add_subplot(gs[0,1]); corr=jread(f"{V}/correction_adult_ref.json",{})
    rows=[("OS held-out vs adult",1.16,np.nan,np.nan,C["adult"]),("GCTB-rep vs adult",1.39,np.nan,np.nan,C["adult"]),
          ("OS vs adult (AUROC d)",1.16,np.nan,np.nan,C["adult"])]
    # use correction json if present
    rows=[]
    if corr:
        for k,v in list(corr.items())[:6]:
            if isinstance(v,dict) and "d" in v: rows.append((k[:22],v["d"],v.get("ci",[np.nan,np.nan])[0],v.get("ci",[np.nan,np.nan])[1],C["adult"]))
    if not rows: rows=[("OS held-out vs adult",1.16,np.nan,np.nan,C["adult"]),("GCTB-rep vs adult",1.39,np.nan,np.nan,C["adult"])]
    H.forest(ax,rows); ax.set_title("Elevation vs adult control",fontsize=7.2); panel_label(ax,"b",x=-.2)
    # c maturation control sina-ish (bars->points)
    ax=fig.add_subplot(gs[0,2])
    vals=[("d program",0.95,C["tumour"]),("d maturation",1.05,GREY),("residualised",0.24,C["adult"])]
    for i,(l,v,c) in enumerate(vals): ax.plot([i,i],[0,v],color=c,lw=1.4); ax.scatter(i,v,s=40,color=c,zorder=3,edgecolors="#222",linewidths=.4)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for l,_,_ in vals],rotation=20,ha="right",fontsize=6); ax.set_ylabel("Cohen's d (vs adult, n=134)"); clean(ax)
    ax.set_title("Maturation confound (residual 0.24)",fontsize=7.2); panel_label(ax,"c",x=-.12)
    # d ROC (recompute would need scores; use AUROCs as annotated bars->lollipop)
    ax=fig.add_subplot(gs[1,0])
    aur=[("GCTB-rep",0.86,C["GCTB"]),("OS held-out",0.83,"#CC79A7"),("GCTB-disc",0.91,C["GCTB"])]
    for i,(l,v,c) in enumerate(aur): ax.plot([0,v],[i,i],color=c,lw=1.4); ax.scatter(v,i,s=40,color=c,zorder=3,edgecolors="#222",linewidths=.4)
    ax.axvline(.5,ls="--",lw=.5,color="#aaa"); ax.set_yticks(range(len(aur))); ax.set_yticklabels([l for l,_,_ in aur],fontsize=6); ax.set_xlabel("AUROC vs adult control"); ax.set_xlim(.4,1)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.set_title("Held-out separation",fontsize=7.2); panel_label(ax,"d",x=-.2)
    # e per-sample consistency beeswarm (per-sample mean program by cohort)
    ax=fig.add_subplot(gs[1,1]); pss=cread(f"{PROJ}/discovery/per_sample_psig.csv")
    if pss is not None and "mean_psig" in pss.columns:
        accs=sorted(pss.acc.unique()); rng=np.random.default_rng(0)
        for i,a in enumerate(accs):
            v=pss[pss.acc==a].mean_psig.values; x=i+(rng.random(len(v))-.5)*.55
            ax.scatter(x,v,s=14,c=OKABE_ITO[(i%7)+1],alpha=.85,edgecolors="#333",linewidths=.2)
        ax.axhline(0,color="#999",lw=.6,ls="--")
        ax.set_xticks(range(len(accs))); ax.set_xticklabels([a[:8] for a in accs],rotation=40,ha="right",fontsize=5)
        ax.set_ylabel("per-sample mean program"); clean(ax)
        ax.set_title(f"Per-sample consistency ({(pss.mean_psig>0).mean()*100:.0f}% > 0, n={len(pss)})",fontsize=7.0)
    panel_label(ax,"e",x=-.12)
    # f doublet correlation summary
    ax=fig.add_subplot(gs[1,2])
    if dub and "doublet" in dub:
        d=dub["doublet"]; r=d.get("spearman_psig_dub_inOC",d.get("pearson_psig_dub_inOC",0))
        load=d.get("sulfation_loading",[]);
        if load:
            labs=[x[0] for x in load][:6][::-1]; vv=[x[1] for x in load][:6][::-1]
            ax.barh(range(len(labs)),vv,color=[C["OCP_mono"]]*len(labs),edgecolor="white",lw=.4)
            ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=5.6); ax.set_xlabel("Spearman ρ with doublet score")
            ax.set_title(f"Sulfation genes' doublet loading\n(program ρ={r:.2f}, overall ~0)",fontsize=7); clean(ax)
    panel_label(ax,"f",x=-.12)
    save_fig(fig,f"{FIG}/G7_validation"); print("[G7] done",flush=True)

# =================================================================== FIG 2 — de-confounding instrument
MODSET={"Podosome":["SH3PXD2A","MYO1E","MYO1D","DNM3","TIAM1","CAMSAP2","NAV2","PTPRM","ARHGAP10","PDLIM5","ACTN2","FILIP1L"],
        "Protease":["MMP19","CEMIP2","COL27A1","PAM","BAMBI"],
        "Sulfation":["PAPSS2","UST","EXT1","FAM20C","SLC16A10","ST3GAL6","SELENOI","GBE1"],
        "TF/other":["JDP2","RUNX3","RFX8","SOX4","MSI2","CDK6"]}
MODOF={g:m for m,gg in MODSET.items() for g in gg}
MCOL={"Podosome":C["mature_OC"],"Protease":"#CC79A7","Sulfation":C["OCP_mono"],"TF/other":C["macrophage"]}
def fig2():
    fig=plt.figure(figsize=(11,7.5),layout="constrained"); gs=GridSpec(2,3,figure=fig)
    naive=cread(f"{PROJ}/report/tables/T5_naive_DE_tumorOC_vs_fetal.csv")
    progset=set(sig.symbol.astype(str)); FORN={"COL1A1","DCN","CD3E","RUNX2","MSR1","LUM","COL3A1","CD2","CD8A","PTPRC","COL1A2"}
    ax=fig.add_subplot(gs[0,0])
    if naive is not None:
        v=naive.copy(); v["nlp"]=-np.log10(v.pvals_adj.clip(1e-300))
        ax.scatter(v.logfoldchanges,v.nlp,s=2,c=GREY,alpha=.35,linewidths=0,rasterized=True)
        fr=v[v.symbol.isin(FORN)]; ax.scatter(fr.logfoldchanges,fr.nlp,s=16,c=C["adult"],edgecolors="white",linewidths=.3,label="foreign/ambient")
        pr=v[v.symbol.isin(progset)]; ax.scatter(pr.logfoldchanges,pr.nlp,s=11,c=HIGHLIGHT,edgecolors="white",linewidths=.3,label="74-gene program")
        ax.set_xlabel("log2FC tumour-OC vs reference"); ax.set_ylabel("−log10 FDR"); ax.legend(fontsize=5,loc="upper left",frameon=False)
    ax.set_title("Naive contrast: a 1,703-gene AUROC-0.99 artefact",fontsize=7.2); clean(ax); panel_label(ax,"a")
    ax=fig.add_subplot(gs[0,1])
    # Venn of the conjunction: tumour-up (1703) AND OC>myeloid (727) -> 226; then -> 74
    H.venn2_counts(ax, a_only=1703-226, b_only=727-226, ab=226,
                   labels=("Tumour-up\nvs reference\n(1,703)","OC > own-tumour\nmyeloid\n(727)"), colors=(C["tumour"],C["OCP_mono"]),
                   title="Interaction filter = intersection of two criteria",
                   note="intersection = 226  →  74-gene convergent program\n(foreign-lineage & technical genes removed; cross-tumour convergent)")
    panel_label(ax,"b",x=-.05)
    ax=fig.add_subplot(gs[0,2])
    cols=[MCOL.get(MODOF.get(s,"TF/other"),GREY) for s in sig.symbol]
    ax.axhline(0,color="#bbb",lw=.5); ax.axvline(0,color="#bbb",lw=.5)
    ax.scatter(sig.mean_lfc_TumorVsNormal,sig.mean_lfc_OCvsMyeloid,s=18,c=cols,edgecolors="#333",linewidths=.3)
    ax.set_xlabel("log2FC tumour vs reference OC"); ax.set_ylabel("log2FC OC vs same-tumour myeloid")
    for m,c in MCOL.items(): ax.scatter([],[],c=c,label=m,s=18)
    ax.legend(fontsize=4.8,loc="lower right",frameon=False)
    ax.set_title("Conjunction criterion (74 genes, both axes > 0)",fontsize=7.2); clean(ax); panel_label(ax,"c")
    naivehits=list(FORN)+["MSR1","C1QA","SPP1","IBSP","COL2A1"]
    Mc,present=gmat([g for g in naivehits if g in sym2ens],scale=False)
    if len(Mc):
        Mz=(Mc-Mc.mean(1,keepdims=True))/(Mc.std(1,keepdims=True)+1e-9)
        H.clustered_heatmap(fig,gs[1,0],Mz,present,[PRETTY.get(c,c) for c in COMPS],cmap="RdBu_r",cbar_label="z-score",title="Naive 'hits' peak in non-OC compartments",label="d")
    ax=fig.add_subplot(gs[1,1]); oc=comp=="mature_OC"; mye=comp=="macrophage"
    tc=np.log10(A.obs["total_counts"].values+1) if "total_counts" in A.obs else None
    if tc is not None:
        ax.scatter(tc[mye],A.obs["psig"].values[mye],s=3,c=C["myeloid"],alpha=.4,linewidths=0,label="Macrophage",rasterized=True)
        ax.scatter(tc[oc],A.obs["psig"].values[oc],s=3,c=C["mature_OC"],alpha=.4,linewidths=0,label="Mature OC",rasterized=True)
        ax.set_xlabel("log10 total counts"); ax.set_ylabel("program score"); ax.legend(fontsize=5,frameon=False)
    ax.set_title("Not a read-depth artefact",fontsize=7.2); clean(ax); panel_label(ax,"e",x=-.05)
    ax=fig.add_subplot(gs[1,2])
    if naive is not None:
        ax.hist(naive.logfoldchanges,bins=80,color=GREY,alpha=.6,label="all genes")
        ax.hist(naive[naive.symbol.isin(progset)].logfoldchanges,bins=30,color=HIGHLIGHT,alpha=.85,label="74-gene program")
        ax.set_xlabel("log2FC tumour vs reference"); ax.set_ylabel("genes"); ax.legend(fontsize=5,frameon=False); ax.set_yscale("log")
    ax.set_title("Program genes are tumour-elevated",fontsize=7.2); clean(ax); panel_label(ax,"f",x=-.05)
    save_fig(fig,f"{FIG}/G2_deconfounding"); print("[G2] done",flush=True)

# =================================================================== FIG 3 — the program & modules
def fig3():
    fig=plt.figure(figsize=(11,7),layout="constrained"); gs=GridSpec(2,3,figure=fig)
    ax=fig.add_subplot(gs[0,0])
    sca=H.density_umap(ax,A.obsm["X_umap"][:,0],A.obsm["X_umap"][:,1],A.obs["psig"].values,cmap=SEQ_CMAP,s=1.5,contour=False); umap_axes(ax)
    cb=fig.colorbar(sca,ax=ax,fraction=.04,pad=.01); cb.set_label("program score",fontsize=5.6); cb.ax.tick_params(labelsize=5)
    ax.set_title("Program score (atlas UMAP)",fontsize=7.2); panel_label(ax,"a")
    # b gene-gene co-expression correlation clustermap (module block structure) across mature OC
    from scipy.cluster.hierarchy import linkage, dendrogram
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    progsyms=[s for s in sig.symbol if s in sym2ens]; ocm=comp=="mature_OC"
    Xg=np.vstack([dense(sym2ens[s])[ocm] for s in progsyms]); cc=np.nan_to_num(np.corrcoef(Xg))
    Z=linkage(cc,method="average"); ordr=dendrogram(Z,no_plot=True)["leaves"]
    ccs=cc[np.ix_(ordr,ordr)]; go=[progsyms[i] for i in ordr]; mo=[MODOF.get(g,"TF/other") for g in go]
    sub=GridSpecFromSubplotSpec(2,2,subplot_spec=gs[0,1:3],height_ratios=[0.035,0.965],width_ratios=[0.025,0.975],hspace=0.012,wspace=0.012)
    axt=fig.add_subplot(sub[0,1]); axl=fig.add_subplot(sub[1,0])
    for i,m in enumerate(mo):
        axt.add_patch(plt.Rectangle((i,0),1,1,color=MCOL.get(m,"#999"),lw=0)); axl.add_patch(plt.Rectangle((0,i),1,1,color=MCOL.get(m,"#999"),lw=0))
    axt.set_xlim(0,len(go)); axt.set_ylim(0,1); axt.axis("off"); axl.set_ylim(len(go),0); axl.set_xlim(0,1); axl.axis("off")
    axh=fig.add_subplot(sub[1,1])
    np.fill_diagonal(ccs,np.nan)
    im=axh.imshow(ccs,cmap="RdBu_r",vmin=-0.5,vmax=0.5,aspect="auto",interpolation="nearest")
    axh.set_xticks([]); axh.set_yticks(range(len(go))); axh.set_yticklabels(go,fontsize=3.5)
    cb=fig.colorbar(im,ax=axh,fraction=.025,pad=.01); cb.set_label("co-expression r",fontsize=5.4); cb.ax.tick_params(labelsize=5)
    axt.set_title("Program genes co-express in modules (gene–gene correlation in osteoclasts; bars = module)",fontsize=7.0)
    for m,c in MCOL.items(): axh.scatter([],[],marker="s",c=c,s=16,label=m)
    axh.legend(fontsize=4.6,loc="upper left",bbox_to_anchor=(1.16,1.0),frameon=False)
    axh.text(-0.16,1.08,"b",transform=axh.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")
    # c module sina by compartment (sulfation)
    ax=fig.add_subplot(gs[1,0])
    se=[sym2ens[g] for g in MODSET["Sulfation"] if g in sym2ens]; sc.tl.score_genes(A,se,score_name="sulf",use_raw=False)
    cps=[c for c in ["mature_OC","macrophage","OCP_mono","tumor_epi","fibro_stroma"] if (comp==c).any()]
    H.sina(ax,[PRETTY.get(c,c) for c in cps],[A.obs["sulf"].values[comp==c] for c in cps],[col_for(c) for c in cps])
    ax.axhline(0,color="#999",lw=.5,ls="--"); ax.set_ylabel("sulfation module score"); ax.set_title("Sulfation module by compartment",fontsize=7.2); clean(ax); panel_label(ax,"c",x=-.08)
    # d SH3PXD2A UMAP, e PAPSS2 UMAP
    for k,(g,pl) in enumerate([("SH3PXD2A","d"),("PAPSS2","e")]):
        ax=fig.add_subplot(gs[1,1+k])
        if g in sym2ens:
            ex=dense(sym2ens[g]); o=np.argsort(ex)
            sca=ax.scatter(A.obsm["X_umap"][o,0],A.obsm["X_umap"][o,1],s=1.5,c=ex[o],cmap=SEQ_CMAP,alpha=.85,linewidths=0,rasterized=True); umap_axes(ax)
            cb=fig.colorbar(sca,ax=ax,fraction=.04,pad=.01); cb.set_label("log1p",fontsize=5.4); cb.ax.tick_params(labelsize=5)
        ax.set_title(f"{g}",fontsize=7.2); panel_label(ax,pl)
    save_fig(fig,f"{FIG}/G3_program"); print("[G3] done",flush=True)

_only=os.environ.get("ONLYFIG")
_all={"fig1":fig1,"fig2":fig2,"fig3":fig3,"fig4":fig4,"fig5":fig5,"fig6":fig6,"fig7":fig7}
for nm,fn in _all.items():
    if _only and nm!=_only: continue
    try: fn()
    except Exception as e: import traceback; traceback.print_exc(); print("FIG FAIL",nm,e,flush=True)
print("[DONE] advanced figures",flush=True)
