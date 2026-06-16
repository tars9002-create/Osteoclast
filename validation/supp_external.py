#!/usr/bin/env python
"""External validation (responds to Codex): score the 74-gene program in
B1 GSE152048 (independent osteosarcoma, 11 tumours) and
B2 GSE143791 (prostate bone-metastasis: Tumor/Involved/Distal/Benign marrow) as a
   myeloid/marrow-background control (OC vs TAM vs marrow).
Programs scored by gene SYMBOL. Seeds fixed."""
import os, sys, json, glob, gzip, re, numpy as np, pandas as pd, scanpy as sc, scipy.io, scipy.sparse as sp
import anndata as ad
np.random.seed(0)
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); DL=os.environ.get("OC_DOWNLOADS", os.path.join(PROJ, "downloads")); V=f"{PROJ}/validation"
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv"); SIG=set(sig.symbol.astype(str))
OCm=["CTSK","ACP5","MMP9","NFATC1","ATP6V0D2","DCSTAMP","OCSTAMP","OSCAR","CALCR"]
TAMm=["CD68","C1QA","C1QB","C1QC","CD163","MRC1","LYZ","APOE","FCGR3A"]
def _open(f): return gzip.open(f,"rt") if f.endswith(".gz") else open(f)
def cohend(x,y):
    x=np.asarray(x); y=np.asarray(y)
    if len(x)<3 or len(y)<3: return float("nan")
    s=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/max(len(x)+len(y)-2,1)); return float((x.mean()-y.mean())/s) if s>0 else 0.0
def boot_ci(x,y,n=1000):
    rng=np.random.default_rng(0); x=np.asarray(x); y=np.asarray(y)
    if len(x)<3 or len(y)<3: return [float("nan"),float("nan")]
    ds=[cohend(rng.choice(x,len(x),replace=True),rng.choice(y,len(y),replace=True)) for _ in range(n)]
    return [float(np.percentile(ds,2.5)),float(np.percentile(ds,97.5))]
def score(A):
    A.layers["counts"]=A.X.copy(); sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
    def sg(genes,nm):
        g=[x for x in genes if x in A.var_names]
        if len(g)>=2: sc.tl.score_genes(A,g,score_name=nm,use_raw=False)
        else: A.obs[nm]=0.0
    sg(list(SIG),"psig"); sg(OCm,"ocs"); sg(TAMm,"tams"); return A
def oc_mask(A):
    raw=A.layers["counts"]
    def pos(g): return (raw[:,A.var_names.get_loc(g)]>0).toarray().ravel() if (g in A.var_names and sp.issparse(raw)) else (np.asarray(raw[:,A.var_names.get_loc(g)]).ravel()>0 if g in A.var_names else np.zeros(A.n_obs,bool))
    return pos("CTSK")&pos("ACP5")

# ---------- B1 GSE152048 (osteosarcoma) ----------
def gse152048():
    acc="GSE152048"; out=f"{DL}/{acc}"; os.makedirs(f"{out}/ext",exist_ok=True)
    base="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152048/suppl"
    # list tumour tarballs
    import urllib.request
    html=urllib.request.urlopen(base+"/",timeout=60).read().decode("utf-8","ignore")
    tars=sorted(set(re.findall(r'(GSE152048_[A-Za-z0-9]+\.matrix\.tar\.gz)',html)))
    print(f"[152048] {len(tars)} tumour tarballs",flush=True)
    parts=[]
    for t in tars:
        dst=f"{out}/{t}"
        if not os.path.exists(dst): os.system(f'wget -q -O "{dst}" "{base}/{t}"')
        sub=f"{out}/ext/{t.replace('.tar.gz','')}"; os.makedirs(sub,exist_ok=True); os.system(f'tar xzf "{dst}" -C "{sub}" 2>/dev/null')
        # find 10x triplet
        mtx=glob.glob(f"{sub}/**/*matrix.mtx*",recursive=True);
        if not mtx: continue
        d=os.path.dirname(mtx[0])
        feat=glob.glob(f"{d}/*genes.tsv*")+glob.glob(f"{d}/*features.tsv*"); bc=glob.glob(f"{d}/*barcodes.tsv*")
        if not(feat and bc): continue
        M=scipy.io.mmread(mtx[0]).tocsr(); syms=[l.rstrip().split("\t")[1] if len(l.split("\t"))>1 else l.strip() for l in _open(feat[0])]
        bcs=[l.strip() for l in _open(bc[0])]
        X=M.T.tocsr() if M.shape==(len(syms),len(bcs)) else M.tocsr()
        Ai=ad.AnnData(X=X); Ai.var_names=pd.Index(syms); Ai.var_names_make_unique(); Ai.obs["tumour"]=t.split(".")[0].replace("GSE152048_",""); parts.append(Ai)
    if not parts: return {"error":"no tumours loaded"}
    common=set(parts[0].var_names)
    for p in parts[1:]: common&=set(p.var_names)
    common=sorted(common); A=ad.concat([p[:,common] for p in parts],merge="same")
    sc.pp.filter_cells(A,min_genes=200); A=score(A); oc=oc_mask(A)
    p_oc=A.obs["psig"].values[oc]; p_rest=A.obs["psig"].values[~oc]
    return {"n_cells":int(A.n_obs),"n_tumours":len(parts),"n_OC":int(oc.sum()),
            "program_OC_mean":float(p_oc.mean()) if oc.sum() else None,"program_nonOC_mean":float(p_rest.mean()),
            "cohensd_OC_vs_nonOC":cohend(p_oc,p_rest),"cohensd_OC_vs_nonOC_CI":boot_ci(p_oc,p_rest),
            "sig_genes_matched":int(len(SIG&set(A.var_names)))}

# ---------- B2 GSE143791 (prostate bone-met; marrow background control) ----------
def gse143791():
    acc="GSE143791"; out=f"{DL}/{acc}"; os.makedirs(f"{out}/ext",exist_ok=True)
    tar=f"{out}/{acc}_RAW.tar"
    if not os.path.exists(tar): os.system(f'wget -q -O "{tar}" "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE143nnn/GSE143791/suppl/GSE143791_RAW.tar"')
    os.system(f'tar xf "{tar}" -C "{out}/ext" 2>/dev/null')
    csvs=sorted(glob.glob(f"{out}/ext/*count.csv.gz"))
    print(f"[143791] {len(csvs)} count csvs",flush=True)
    parts=[]
    for f in csvs:
        nm=os.path.basename(f); m=re.search(r"_([A-Za-z0-9]+)-(Tumor|Involved|Distal|Benign)",nm)
        if m is None: continue   # RAW.tar bundles unrelated mouse-KO / melanoma samples w/ different gene sets -> skip
        site=m.group(2); samp=m.group(1)
        try:
            df=pd.read_csv(f,index_col=0)
            ref=SIG|set(OCm)|set(TAMm)|{"ACTB","GAPDH","PTPRC","COL1A1","VIM","B2M","MALAT1","CD3E"}
            idx_hit=len(ref&set(map(str,df.index))); col_hit=len(ref&set(map(str,df.columns)))
            if idx_hit>=col_hit:   # genes are ROWS (GSE143791 format) -> transpose to cells x genes
                X=sp.csr_matrix(df.values.T.astype(np.float32)); var=df.index.astype(str)
            else:                  # genes are columns -> already cells x genes
                X=sp.csr_matrix(df.values.astype(np.float32)); var=df.columns.astype(str)
            Ai=ad.AnnData(X=X); Ai.var_names=pd.Index(var); Ai.var_names_make_unique(); Ai.obs["site"]=site; Ai.obs["sample"]=samp; parts.append(Ai)
            if len(parts)<=2: print(f"  loaded {nm}: raw{df.shape} idx_hit={idx_hit} col_hit={col_hit} -> {Ai.n_obs}cells x {Ai.n_vars}genes",flush=True)
        except Exception as e: print("  csv fail",nm,e)
    if not parts: return {"error":"no csv loaded"}
    common=set(parts[0].var_names)
    for p in parts[1:]: common&=set(p.var_names)
    common=sorted(common); print(f"  [143791] {len(parts)} parts, {len(common)} common genes",flush=True)
    A=ad.concat([p[:,common] for p in parts],merge="same")
    sc.pp.filter_cells(A,min_genes=200); A=score(A); oc=oc_mask(A)
    tam=(A.obs["tams"].values>np.percentile(A.obs["tams"].values,80))&~oc
    site=A.obs["site"].astype(str).values; P=A.obs["psig"].values
    by_site={s:{"n":int((site==s).sum()),"program_mean":float(P[site==s].mean())} for s in sorted(set(site))}
    return {"n_cells":int(A.n_obs),"n_OC":int(oc.sum()),"n_TAM":int(tam.sum()),
            "program_OC_mean":float(P[oc].mean()) if oc.sum() else None,
            "program_TAM_mean":float(P[tam].mean()) if tam.sum() else None,
            "cohensd_OC_vs_TAM":cohend(P[oc],P[tam]) if oc.sum() and tam.sum() else None,
            "cohensd_OC_vs_TAM_CI":boot_ci(P[oc],P[tam]) if oc.sum() and tam.sum() else None,
            "by_site":by_site,"sig_genes_matched":int(len(SIG&set(A.var_names)))}

res={}
for nm,fn in [("GSE152048_OS",gse152048),("GSE143791_prostateBM",gse143791)]:
    try: res[nm]=fn(); print(nm,"->",res[nm] if "error" in res[nm] else {k:res[nm][k] for k in list(res[nm])[:6]},flush=True)
    except Exception as e: import traceback; res[nm]={"error":str(e)}; traceback.print_exc()
json.dump(res,open(f"{V}/supp_external.json","w"),indent=2,default=str)
print("[DONE] external validation written")
