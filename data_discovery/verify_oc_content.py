#!/usr/bin/env python
"""
verify_oc_content.py — Gate-1 dataset verification + osteoclast-content probe.

For a GEO RAW.tar extracted into a directory of flat 10x triplets
(GSM..._*_barcodes.tsv.gz / _features.tsv.gz / _matrix.mtx.gz), this:
  1. groups files by GSM/sample prefix and loads each as AnnData,
  2. confirms RAW INTEGER counts (not normalized/log),
  3. reports n_cells / n_genes / median counts,
  4. scores OSTEOCLAST-LINEAGE content: conservative co-expression count
     (CTSK>0 & ACP5>0), a mature-OC signature score, and per-marker positivity.
Writes <verif_dir>/<accession>.md and <accession>.json.

Usage: verify_oc_content.py <accession> <extracted_dir> <verif_dir> [sha256_of_tar]
"""
import sys, os, gzip, json, glob, re
import numpy as np
import scipy.io, scipy.sparse as sp
import anndata as ad
import scanpy as sc

# Curated HUMAN osteoclast-lineage marker panels (symbols + common aliases).
OC_MATURE = ["CTSK","ACP5","MMP9","ATP6V0D2","NFATC1","DCSTAMP","TM7SF4",
             "OCSTAMP","OSCAR","CALCR","SLC4A2","ATP6V1B1","CLCN7","OSTM1","SLC9B2"]
OC_CORE2  = ["CTSK","ACP5"]                      # both highly OC-specific -> strict co-expression
MONO_MAC  = ["CD14","FCGR3A","CSF1R","ITGAX","CX3CR1","CD68","LYZ","MRC1","C1QA","C1QB"]

def find_groups(d):
    """Group 10x files into samples. Handles both flat GEO files with GSM prefixes
    (GSM..._barcodes.tsv.gz) AND nested per-sample dirs with bare names
    (L109/barcodes.tsv.gz). Sample key = filename prefix if present, else parent dir."""
    groups = {}
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        b = os.path.basename(f)
        m = re.match(r"(?:(.+?)[._-])?(matrix\.mtx|features\.tsv|genes\.tsv|barcodes\.tsv)(\.gz)?$", b, re.I)
        if not m:
            continue
        pre, kind = m.group(1), m.group(2).lower()
        key = pre if pre else os.path.dirname(f)   # bare names -> key by directory
        g = groups.setdefault(key, {})
        if "matrix" in kind: g["mtx"] = f
        elif "barcodes" in kind: g["bc"] = f
        else: g["feat"] = f
    return {k: v for k, v in groups.items() if {"mtx","feat","bc"} <= set(v)}

def _open(f): return gzip.open(f, "rt") if f.endswith(".gz") else open(f)

def load_sample(g):
    M = scipy.io.mmread(g["mtx"]).tocsr()          # features x cells (10x convention)
    feats = [l.rstrip("\n").split("\t") for l in _open(g["feat"])]
    syms = [c[1] if len(c) > 1 else c[0] for c in feats]   # col2 = symbol
    bcs  = [l.strip() for l in _open(g["bc"])]
    # orient to cells x genes
    if M.shape[0] == len(syms) and M.shape[1] == len(bcs):
        X = M.T.tocsr()
    elif M.shape[0] == len(bcs) and M.shape[1] == len(syms):
        X = M.tocsr()
    else:
        raise ValueError(f"shape mismatch {M.shape} vs genes {len(syms)} cells {len(bcs)}")
    A = ad.AnnData(X=X)
    A.var_names = sc.AnnData(np.empty((0,len(syms)))).var_names if False else syms
    A.var_names_make_unique(); A.obs_names = bcs
    return A

def is_raw(A):
    sub = A.X[:200] if A.n_obs > 200 else A.X
    data = sub.data if sp.issparse(sub) else np.asarray(sub).ravel()
    if data.size == 0: return False, "empty"
    frac_int = float(np.mean(np.isclose(data, np.round(data))))
    return frac_int > 0.999, f"int_frac={frac_int:.4f} max={float(data.max()):.2f}"

def oc_score(A):
    A = A.copy()
    sc.pp.filter_cells(A, min_genes=200)
    if A.n_obs < 30:
        return {"n_cells_qc": int(A.n_obs), "note": "too few cells after QC"}
    raw = A.copy()
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    present_mat = [g for g in OC_MATURE if g in A.var_names]
    present_mac = [g for g in MONO_MAC if g in A.var_names]
    out = {"n_cells_qc": int(A.n_obs), "oc_markers_present": present_mat,
           "mac_markers_present": present_mac}
    # strict co-expression: CTSK>0 & ACP5>0 (raw)
    co = None
    if all(g in raw.var_names for g in OC_CORE2):
        idx = [raw.var_names.get_loc(g) for g in OC_CORE2]
        sub = raw.X[:, idx]
        pos = (sub > 0).toarray() if sp.issparse(sub) else (np.asarray(sub) > 0)
        co = int(np.sum(pos.all(axis=1)))
    out["ctsk_acp5_coexpr_cells"] = co
    # per-marker positivity (raw)
    posmap = {}
    for g in OC_CORE2 + ["MMP9","ATP6V0D2","NFATC1"]:
        if g in raw.var_names:
            col = raw[:, g].X
            posmap[g] = int((col > 0).sum())
    out["marker_pos_cells"] = posmap
    # signature score
    if len(present_mat) >= 3:
        sc.tl.score_genes(A, present_mat, score_name="oc_sig")
        s = A.obs["oc_sig"].values
        thr = np.percentile(s, 99)
        out["oc_sig_top1pct_thr"] = float(thr)
        out["oc_sig_high_cells"] = int(np.sum(s > max(thr, 0.1)))
        out["oc_sig_mean"] = float(np.mean(s))
    return out

def main():
    acc, ddir, vdir = sys.argv[1], sys.argv[2], sys.argv[3]
    sha = sys.argv[4] if len(sys.argv) > 4 else ""
    os.makedirs(vdir, exist_ok=True)
    groups = find_groups(ddir)
    rep = {"accession": acc, "extracted_dir": ddir, "tar_sha256": sha,
           "n_samples": len(groups), "samples": {}}
    tot_cells = tot_co = 0
    for pre, g in sorted(groups.items()):
        try:
            A = load_sample(g)
            raw_ok, raw_msg = is_raw(A)
            s = oc_score(A)
            s.update({"n_cells_raw": int(A.n_obs), "n_genes": int(A.n_vars),
                      "raw_counts": raw_ok, "raw_check": raw_msg})
            rep["samples"][pre] = s
            tot_cells += s.get("n_cells_qc", 0)
            tot_co += (s.get("ctsk_acp5_coexpr_cells") or 0)
        except Exception as e:
            rep["samples"][pre] = {"error": f"{type(e).__name__}: {e}"}
    rep["total_cells_qc"] = tot_cells
    rep["total_ctsk_acp5_coexpr"] = tot_co
    rep["oc_coexpr_fraction"] = (tot_co / tot_cells) if tot_cells else None
    json.dump(rep, open(os.path.join(vdir, f"{acc}.json"), "w"), indent=2)
    # markdown
    L = [f"# Verification — {acc}", "",
         f"- extracted_dir: `{ddir}`", f"- tar sha256: `{sha}`",
         f"- samples (GSM): **{rep['n_samples']}**",
         f"- total cells (post min_genes=200 QC): **{tot_cells:,}**",
         f"- **CTSK+ACP5 co-expressing cells (strict mature-OC proxy): {tot_co:,}** "
         f"({100*rep['oc_coexpr_fraction']:.2f}% of QC cells)" if tot_cells else "",
         "", "| sample | cells | genes | raw? | CTSK&ACP5+ | OC-sig-high | OC markers present |",
         "|---|---|---|---|---|---|---|"]
    for pre, s in sorted(rep["samples"].items()):
        if "error" in s:
            L.append(f"| {pre} | ERROR | | | | | {s['error'][:40]} |"); continue
        L.append(f"| {pre} | {s.get('n_cells_qc',0):,} | {s.get('n_genes','?')} | "
                 f"{'Y' if s.get('raw_counts') else 'N'} | {s.get('ctsk_acp5_coexpr_cells','?')} | "
                 f"{s.get('oc_sig_high_cells','?')} | {len(s.get('oc_markers_present',[]))}/{len(OC_MATURE)} |")
    open(os.path.join(vdir, f"{acc}.md"), "w").write("\n".join(L) + "\n")
    print(f"[{acc}] samples={rep['n_samples']} cells={tot_cells:,} CTSK&ACP5+={tot_co:,} "
          f"frac={rep['oc_coexpr_fraction']}")

if __name__ == "__main__":
    main()
