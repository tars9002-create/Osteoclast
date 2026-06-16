"""Advanced publication plot helpers (journal-grade): sina, forest, lollipop-with-null,
clustered+annotated heatmap with dendrogram, dotplot, circular interaction network,
density UMAP, pseudotime gene heatmap. Embeddable into a parent GridSpec."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import gaussian_kde

def _violin_outline(ax, pos, vals, color, width=0.36, horiz=False):
    vals=np.asarray(vals); vals=vals[np.isfinite(vals)]
    if len(vals)<5: return
    try:
        k=gaussian_kde(vals); xs=np.linspace(vals.min(),vals.max(),120); d=k(xs); d=d/d.max()*width
    except Exception: return
    if horiz:
        ax.fill_between(xs,pos-d,pos+d,color=color,alpha=0.18,lw=0); ax.plot(xs,pos+d,color=color,lw=0.5); ax.plot(xs,pos-d,color=color,lw=0.5)
    else:
        ax.fill_betweenx(xs,pos-d,pos+d,color=color,alpha=0.18,lw=0); ax.plot(pos+d,xs,color=color,lw=0.5); ax.plot(pos-d,xs,color=color,lw=0.5)

def sina(ax, labels, values, colors, horiz=False, maxpts=400, point_s=2.2):
    """strip(jittered points)+violin outline+median bar. values: list of arrays aligned to labels."""
    rng=np.random.default_rng(0)
    for i,(lab,v,c) in enumerate(zip(labels,values,colors)):
        v=np.asarray(v); v=v[np.isfinite(v)]
        if len(v)==0: continue
        _violin_outline(ax,i,v,c,horiz=horiz)
        vv=v if len(v)<=maxpts else rng.choice(v,maxpts,replace=False)
        jit=(rng.random(len(vv))-0.5)*0.5
        if horiz: ax.scatter(vv,i+jit,s=point_s,c=c,alpha=0.5,linewidths=0,rasterized=True)
        else: ax.scatter(i+jit,vv,s=point_s,c=c,alpha=0.5,linewidths=0,rasterized=True)
        med=np.median(v)
        if horiz: ax.plot([med,med],[i-0.34,i+0.34],color="#111",lw=1.3,zorder=5)
        else: ax.plot([i-0.34,i+0.34],[med,med],color="#111",lw=1.3,zorder=5)
    if horiz:
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    else:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=35,ha="right")

def forest(ax, rows, xlabel="Cohen's d (95% CI)"):
    """rows: list of (label, est, lo, hi, color). Horizontal forest."""
    n=len(rows)
    for i,(lab,est,lo,hi,c) in enumerate(rows):
        y=n-1-i
        if np.isfinite(lo) and np.isfinite(hi): ax.plot([lo,hi],[y,y],color=c,lw=1.6,zorder=2,solid_capstyle="round")
        ax.scatter(est,y,s=34,color=c,zorder=3,edgecolors="#222",linewidths=0.4)
    ax.axvline(0,color="#999",lw=0.6,ls="--")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1],fontsize=6.2)
    ax.set_ylim(-0.6,n-0.4); ax.set_xlabel(xlabel)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)

def lollipop_null(ax, labels, real, null, colors, in_program=None):
    """KO: stem from 0 to real Δ; real filled dot; null = open diamond. labels on y."""
    n=len(labels)
    for i,(lab,r,nu,c) in enumerate(zip(labels,real,null,colors)):
        y=n-1-i
        ax.plot([0,r],[y,y],color=c,lw=1.4,zorder=2)
        ax.scatter(r,y,s=42,color=c,zorder=4,edgecolors="#222",linewidths=0.4)
        ax.scatter(nu,y,s=34,facecolors="none",edgecolors="#555",linewidths=0.9,marker="D",zorder=3)
        if in_program is not None and in_program[i]:
            ax.annotate("∈program",(r,y),textcoords="offset points",xytext=(-4,0),ha="right",va="center",fontsize=4.6,color="#7A2020")
    ax.axvline(0,color="#999",lw=0.6)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1],fontsize=6.4)
    ax.set_ylim(-0.6,n-0.4)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)

def clustered_heatmap(fig, ss, M, rows, cols, cmap="RdBu_r", vcenter=True, row_anno=None,
                      anno_colors=None, title="", cbar_label="", annotate=False, cluster_cols=True, cluster_rows=False, label=None):
    """Embed an annotated clustered heatmap (col dendrogram + optional row-annotation track) into subplotspec ss.
    M: rows x cols array. row_anno: list aligned to rows (categorical) drawn as a left colour strip."""
    M=np.asarray(M,float)
    co=np.arange(M.shape[1]); ro=np.arange(M.shape[0])
    if cluster_cols and M.shape[1]>2:
        Z=linkage(M.T,method="average"); co=dendrogram(Z,no_plot=True)["leaves"]
    if cluster_rows and M.shape[0]>2:
        Zr=linkage(M,method="average"); ro=dendrogram(Zr,no_plot=True)["leaves"]
    Mo=M[np.ix_(ro,co)]; rows=[rows[i] for i in ro]; cols=[cols[i] for i in co]
    has_anno=row_anno is not None
    width_ratios=[0.05,0.92] if has_anno else [0.001,0.97]
    sub=GridSpecFromSubplotSpec(2,2,subplot_spec=ss,height_ratios=[0.16,0.84],width_ratios=width_ratios,hspace=0.04,wspace=0.04)
    # col dendrogram
    axd=fig.add_subplot(sub[0,1]); axd.axis("off")
    if cluster_cols and M.shape[1]>2:
        dendrogram(Z,ax=axd,color_threshold=0,above_threshold_color="#888",no_labels=True)
        axd.set_xticks([])
    # row annotation strip
    if has_anno:
        axa=fig.add_subplot(sub[1,0]); ra=[row_anno[i] for i in ro]
        cats=list(dict.fromkeys(row_anno)); cmap_a=anno_colors or {c:plt.cm.tab10(i%10) for i,c in enumerate(cats)}
        for j,cat in enumerate(ra): axa.add_patch(plt.Rectangle((0,j),1,1,color=cmap_a.get(cat,"#ccc"),lw=0))
        axa.set_xlim(0,1); axa.set_ylim(len(ra),0); axa.axis("off")
    # heatmap
    axh=fig.add_subplot(sub[1,1])
    vm=np.nanmax(np.abs(Mo)) if vcenter else None
    im=axh.imshow(Mo,aspect="auto",cmap=cmap,vmin=(-vm if vcenter else None),vmax=(vm if vcenter else None),interpolation="nearest")
    axh.set_xticks(range(len(cols))); axh.set_xticklabels(cols,rotation=35,ha="right",fontsize=5.8)
    axh.set_yticks(range(len(rows))); axh.set_yticklabels(rows,fontsize=5.6)
    if annotate:
        for i in range(Mo.shape[0]):
            for j in range(Mo.shape[1]):
                axh.text(j,i,f"{Mo[i,j]:.2f}",ha="center",va="center",fontsize=4.2,color="#222" if abs(Mo[i,j])<(vm or 1)*0.6 else "white")
    cb=fig.colorbar(im,ax=axh,fraction=0.025,pad=0.01); cb.set_label(cbar_label,fontsize=5.6); cb.ax.tick_params(labelsize=5)
    if title: axd.set_title(title,fontsize=7.2,pad=2)
    if label: axh.text(-0.12,1.30,label,transform=axh.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")
    return axh

def dotplot(ax, Mcolor, Msize, rows, cols, cmap="viridis", smax=70, clabel="scaled mean", title=""):
    """clustered dotplot: dot color=Mcolor (0-1), size=Msize (0-1)."""
    Mc=np.asarray(Mcolor); Ms=np.asarray(Msize)
    gx,gy=np.meshgrid(range(len(cols)),range(len(rows)))
    sc=ax.scatter(gx.ravel(),gy.ravel(),s=(Ms.ravel()*smax)+1,c=Mc.ravel(),cmap=cmap,vmin=0,vmax=1,edgecolors="#444",linewidths=0.25)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=35,ha="right",fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=5.8)
    ax.set_xlim(-0.6,len(cols)-0.4); ax.set_ylim(-0.7,len(rows)-0.3); ax.invert_yaxis()
    if title: ax.set_title(title,fontsize=7.2)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    return sc

def circular_network(ax, M, labels, highlight=None, hl_color="#D55E00", base="#BDBDBD", node_colors=None):
    """compartments on a circle; edge width ~ M[i,j]; chord-like arcs. M square (labels x labels)."""
    n=len(labels); ang=np.linspace(0,2*np.pi,n,endpoint=False); R=1.0
    xy=np.c_[R*np.cos(ang),R*np.sin(ang)]
    mx=np.nanmax(M) if np.nanmax(M)>0 else 1
    for i in range(n):
        for j in range(n):
            if i>=j: continue
            w=(M[i,j]+M[j,i])/2
            if w<=0: continue
            lw=0.3+3.2*(w/mx)
            col=hl_color if (highlight is not None and (labels[i]==highlight or labels[j]==highlight)) else base
            # quadratic bezier through center for chord look
            mid=(xy[i]+xy[j])*0.18
            t=np.linspace(0,1,40); bz=(1-t)[:,None]**2*xy[i]+2*(1-t)[:,None]*t[:,None]*mid+t[:,None]**2*xy[j]
            ax.plot(bz[:,0],bz[:,1],color=col,lw=lw,alpha=0.55,zorder=1,solid_capstyle="round")
    for i,(p,lab) in enumerate(zip(xy,labels)):
        nc=(node_colors or {}).get(lab, hl_color if lab==highlight else "#444")
        ax.scatter(*p,s=70,c=nc,zorder=3,edgecolors="white",linewidths=0.6)
        ha="left" if p[0]>0.05 else ("right" if p[0]<-0.05 else "center")
        ax.text(p[0]*1.18,p[1]*1.18,lab,fontsize=5.6,ha=ha,va="center")
    ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.4,1.4); ax.set_aspect("equal"); ax.axis("off")

def density_umap(ax, x, y, val, cmap="viridis", s=2, contour=True, label=""):
    o=np.argsort(val); sc=ax.scatter(np.asarray(x)[o],np.asarray(y)[o],s=s,c=np.asarray(val)[o],cmap=cmap,alpha=0.85,linewidths=0,rasterized=True)
    if contour:
        try:
            from scipy.stats import gaussian_kde
            idx=np.random.default_rng(0).choice(len(x),min(6000,len(x)),replace=False)
            k=gaussian_kde(np.vstack([np.asarray(x)[idx],np.asarray(y)[idx]]))
            xi,yi=np.mgrid[min(x):max(x):80j,min(y):max(y):80j]
            zi=k(np.vstack([xi.ravel(),yi.ravel()])).reshape(xi.shape)
            ax.contour(xi,yi,zi,levels=5,colors="#333",linewidths=0.35,alpha=0.5)
        except Exception: pass
    return sc

def venn2_counts(ax, a_only, b_only, ab, labels, colors, title="", note="", label=None):
    """Area-proportional 2-set Venn from counts (the conjunction criterion), paper layout.
    a_only/b_only = exclusive counts, ab = intersection. labels=(A,B) drawn above each circle.
    Axis limits are derived from the true circle extents (+margin) so nothing clips; subset
    counts sit at their region centroids and the two set labels are horizontally separated."""
    import matplotlib.patches as mp
    A=a_only+ab; B=b_only+ab
    rA=np.sqrt(A/np.pi); rB=np.sqrt(B/np.pi); s=1.0/max(rA,rB); rA*=s; rB*=s   # larger circle r=1
    d=0.66*(rA+rB); cA=np.array([-d/2,0.0]); cB=np.array([d/2,0.0])
    ax.add_patch(mp.Circle(cA,rA,fc=colors[0],ec="white",lw=1.4,alpha=0.55,zorder=1))
    ax.add_patch(mp.Circle(cB,rB,fc=colors[1],ec="white",lw=1.4,alpha=0.55,zorder=1))
    # subset counts at region centroids (left-only / lens / right-only)
    xL=cA[0]-rA*0.50; xR=cB[0]+rB*0.52; xM=((cA[0]+rA)+(cB[0]-rB))/2
    ax.text(xL,0,f"{a_only:,}",ha="center",va="center",fontsize=8.5,zorder=4)
    ax.text(xR,0,f"{b_only:,}",ha="center",va="center",fontsize=8.5,zorder=4)
    ax.text(xM,0,f"{ab:,}",ha="center",va="center",fontsize=10,fontweight="bold",zorder=4)
    # set labels above each circle, nudged apart so they never collide
    ax.text(cA[0]-0.06,rA+0.07,labels[0],ha="center",va="bottom",fontsize=6.6,color=colors[0],fontweight="bold",linespacing=0.95)
    ax.text(cB[0]+0.06,rB+0.07,labels[1],ha="center",va="bottom",fontsize=6.6,color=colors[1],fontweight="bold",linespacing=0.95)
    # limits from true extents + margin -> no clipping; equal aspect keeps circles round
    xmin=cA[0]-rA; xmax=cB[0]+rB; m=0.13*(xmax-xmin); R=max(rA,rB)
    ax.set_xlim(xmin-m,xmax+m); ax.set_ylim(-R-0.58,R+0.58)
    ax.set_aspect("equal"); ax.axis("off")
    if note: ax.text((cA[0]+cB[0])/2,-R-0.16,note,ha="center",va="top",fontsize=6.2,color="#333",linespacing=1.2)
    if title: ax.set_title(title,fontsize=7.2,pad=6)
    if label: ax.text(0.0,1.0,label,transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")

def coexpr_network(ax, corr, genes, gene_module, module_colors, thresh=0.30, topk=3, label_genes=None, title="", label=None):
    """Co-expression network: nodes=genes colored by module, edges = strongest positive correlations.
    corr: genes x genes Pearson matrix. Shows the program's module-community structure."""
    import networkx as nx
    n=len(genes); G=nx.Graph(); [G.add_node(g) for g in genes]
    for i in range(n):
        order=np.argsort(-corr[i]); cnt=0
        for j in order:
            if j==i: continue
            if corr[i,j]<thresh: break
            G.add_edge(genes[i],genes[j],w=float(corr[i,j])); cnt+=1
            if cnt>=topk: break
    pos=nx.spring_layout(G,seed=0,k=1.1/np.sqrt(max(n,1)),iterations=200)
    for u,v,d in G.edges(data=True):
        ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],color="#bbb",lw=0.25+2.2*max(d["w"]-thresh,0),alpha=0.5,zorder=1,solid_capstyle="round")
    deg=dict(G.degree())
    for g in genes:
        ax.scatter(*pos[g],s=24+deg.get(g,0)*16,c=module_colors.get(gene_module.get(g,"TF/other"),"#999"),edgecolors="white",linewidths=0.4,zorder=3)
    lab=set(label_genes or [g for g in genes if deg.get(g,0)>=4])
    for g in genes:
        if g in lab: ax.annotate(g,pos[g],fontsize=4.4,ha="center",va="center",zorder=4,fontweight="bold")
    ax.axis("off"); ax.set_aspect("equal")
    for m,c in module_colors.items(): ax.scatter([],[],c=c,label=m,s=22)
    ax.legend(fontsize=5.2,loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=len(module_colors),frameon=False,columnspacing=0.9,handletextpad=0.3)
    if title: ax.set_title(title,fontsize=7.2)
    if label: ax.text(-0.04,1.04,label,transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")

def radar(ax, axis_labels, series, colors, title="", per_axis_scale=True):
    """Radar/spider fingerprint on a POLAR ax. series: dict name->[value per axis]; per-axis min-max scaled."""
    labels=list(axis_labels); N=len(labels)
    ang=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); ang+=ang[:1]
    M=np.array([series[k] for k in series])  # names x axes
    if per_axis_scale:
        lo=M.min(0,keepdims=True); hi=M.max(0,keepdims=True); Msc=(M-lo)/(hi-lo+1e-9)
    else: Msc=M
    for i,name in enumerate(series):
        v=Msc[i].tolist(); v+=v[:1]
        ax.plot(ang,v,color=colors[i],lw=1.6,label=name,zorder=3)
        ax.fill(ang,v,color=colors[i],alpha=0.12,zorder=1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels,fontsize=6.2)
    ax.set_yticks([0,0.5,1]); ax.set_yticklabels(["low","","high"],fontsize=5)
    ax.set_ylim(0,1.05); ax.tick_params(pad=1)
    ax.legend(fontsize=5.2,loc="upper center",bbox_to_anchor=(0.5,-0.10),ncol=2,frameon=False,columnspacing=1.0,handletextpad=0.4)
    if title: ax.set_title(title,fontsize=7.2,pad=14)

def ridgeline(fig, ss, labels, datasets, colors, xlabel="", overlap=0.62, title="", label=None):
    """Joy/ridgeline: stacked overlapping KDE densities, one per group."""
    ax=fig.add_subplot(ss); n=len(labels)
    allv=np.concatenate([np.asarray(d)[np.isfinite(d)] for d in datasets if len(d)])
    lo,hi=np.percentile(allv,1),np.percentile(allv,99); xs=np.linspace(lo,hi,200)
    step=1.0
    for i in range(n-1,-1,-1):
        d=np.asarray(datasets[i]); d=d[np.isfinite(d)]; base=i*step*(1-overlap)
        if len(d)>5:
            try:
                k=gaussian_kde(d); dens=k(xs); dens=dens/dens.max()*step
            except Exception: dens=np.zeros_like(xs)
            ax.fill_between(xs,base,base+dens,color=colors[i],alpha=0.8,lw=0.6,edgecolor="white",zorder=n-i)
            med=np.median(d); ax.plot([med,med],[base,base+0.18],color="#111",lw=1.0,zorder=n-i+0.5)
        ax.text(lo,base+0.05,labels[i],fontsize=6,ha="right",va="bottom")
    ax.set_yticks([]); ax.set_xlabel(xlabel); ax.set_xlim(lo,hi)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    if title: ax.set_title(title,fontsize=7.2)
    if label: ax.text(-0.12,1.05,label,transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")
    return ax

def slopegraph(ax, xlabels, yvals, cis, color="#0072B2", ref=None, ylabel="Cohen's d"):
    """Robustness slopegraph: connected points across conditions with CI ribbon; flat = robust."""
    x=np.arange(len(xlabels)); y=np.asarray(yvals)
    lo=np.array([c[0] for c in cis]); hi=np.array([c[1] for c in cis])
    ax.fill_between(x,lo,hi,color=color,alpha=0.15,lw=0,zorder=1)
    ax.plot(x,y,color=color,lw=1.8,zorder=2,marker="o",ms=7,mfc=color,mec="#222",mew=0.5)
    for xi,yi in zip(x,y): ax.annotate(f"{yi:.2f}",(xi,yi),textcoords="offset points",xytext=(0,9),ha="center",fontsize=6.2)
    if ref is not None: ax.axhline(ref,color="#999",lw=0.6,ls="--")
    ax.set_xticks(x); ax.set_xticklabels(xlabels,fontsize=6,rotation=18,ha="right")
    ax.set_ylabel(ylabel); ax.set_xlim(-0.4,len(xlabels)-0.6)
    for s in ["top","right"]: ax.spines[s].set_visible(False)

def pseudotime_heatmap(fig, ss, matz, genes, compbin=None, comp_colors=None, label=None):
    """genes x pseudotime-bin heatmap (row z-score) with an optional compartment-composition band on top."""
    nb=matz.shape[1]
    if compbin is not None:
        sub=GridSpecFromSubplotSpec(2,1,subplot_spec=ss,height_ratios=[0.08,0.92],hspace=0.03)
        axb=fig.add_subplot(sub[0]); cats=sorted({k for d in compbin for k in d})
        bottom=np.zeros(nb)
        for c in cats:
            vals=np.array([d.get(c,0) for d in compbin]); axb.bar(range(nb),vals,bottom=bottom,width=1.0,color=(comp_colors or {}).get(c,"#ccc"),lw=0)
            bottom+=vals
        axb.set_xlim(-0.5,nb-0.5); axb.set_ylim(0,1); axb.axis("off")
        axh=fig.add_subplot(sub[1])
    else:
        axh=fig.add_subplot(ss)
    vm=np.nanmax(np.abs(matz))
    im=axh.imshow(matz,aspect="auto",cmap="RdBu_r",vmin=-vm,vmax=vm,interpolation="nearest")
    axh.set_yticks(range(len(genes))); axh.set_yticklabels(genes,fontsize=5.0)
    axh.set_xticks([0,nb-1]); axh.set_xticklabels(["OCP/mono","mature OC"],fontsize=5.6)
    axh.set_xlabel("pseudotime →",fontsize=6)
    cb=fig.colorbar(im,ax=axh,fraction=0.025,pad=0.01); cb.set_label("row z-score",fontsize=5.4); cb.ax.tick_params(labelsize=5)
    if label: axh.text(-0.10,1.18,label,transform=axh.transAxes,fontsize=10,fontweight="bold",va="top",ha="right")
    return axh
