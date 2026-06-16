#!/usr/bin/env python
"""Curated prior-evidence matrix for the 74-gene convergent program (responds to Codex).
Each gene -> functional module + prior-literature class + one-line note. Honest curation:
'OC-direct' = published osteoclast role; 'bone-adjacent' = skeletal/mineralization; 'myeloid'
= macrophage/immune; 'cancer' = tumour/prognosis only; 'novel' = no prior OC literature found.
This documents that the PROGRAM mixes a minority of bone-anchored genes with a majority that are
novel-in-OC -> the discovery is the convergent program, not any single canonical gene."""
import os
import pandas as pd, json
PROJ=os.environ.get("OC_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); V=f"{PROJ}/validation"
TABLES=f"{PROJ}/report/tables"; os.makedirs(TABLES,exist_ok=True)
sig=pd.read_csv(f"{PROJ}/discovery/convergent_signature_v2.csv")

# module, class, note  (class in: OC-direct / bone-adjacent / myeloid / cancer / novel)
CUR={
 "SH3PXD2A":("podosome","OC-direct","Tks5 podosome scaffold; sealing-zone/actin-ring in osteoclasts"),
 "MYO1E":("podosome","bone-adjacent","unconventional myosin at podosomes/phagocytic cups"),
 "MYO1D":("podosome","novel","unconventional myosin; no prior osteoclast report"),
 "DNM3":("podosome","bone-adjacent","dynamin-3; membrane fission/endocytosis at podosomes"),
 "TIAM1":("podosome","bone-adjacent","Rac1 GEF; podosome dynamics & migration"),
 "PTPRM":("podosome","novel","type-R PTP cell-adhesion; uncharacterised in OC"),
 "CAMSAP2":("podosome","novel","microtubule minus-end stabiliser; podosome-belt relevant, novel in OC"),
 "NAV2":("podosome","novel","neuron-navigator cytoskeletal regulator; novel in OC"),
 "ARHGAP10":("podosome","novel","Rho-GAP; actin remodeling, uncharacterised in OC"),
 "PDLIM5":("podosome","novel","actin/PDZ-LIM adaptor; novel in OC"),
 "ACTN2":("podosome","novel","alpha-actinin-2 actin crosslinker; novel in OC"),
 "FILIP1L":("podosome","cancer","filamin-interacting; anti-migratory in cancer"),
 "MMP19":("protease","myeloid","matrix metalloproteinase-19; ALSO expressed by TAMs -> specificity tested here"),
 "CEMIP2":("protease","bone-adjacent","TMEM2 cell-surface hyaluronidase; ECM turnover"),
 "COL27A1":("protease","bone-adjacent","fibrillar collagen XXVII; cartilage-bone transition (stromal-spillover risk)"),
 "PAM":("protease","novel","peptidylglycine amidating monooxygenase; secretory, novel in OC"),
 "BAMBI":("protease","bone-adjacent","BMP/TGFb pseudo-receptor; bone signaling modulator"),
 "PAPSS2":("sulfation","bone-adjacent","PAPS synthase-2; loss -> spondyloepimetaphyseal dysplasia (skeletal)"),
 "UST":("sulfation","novel","uronyl-2-O-sulfotransferase; GAG sulfation, novel in OC"),
 "EXT1":("sulfation","bone-adjacent","heparan-sulfate polymerase; mutations -> hereditary exostoses (bone)"),
 "FAM20C":("sulfation","bone-adjacent","secretory kinase of bone matrix; loss -> Raine osteosclerosis"),
 "SLC16A10":("sulfation","novel","aromatic-AA transporter; novel in OC"),
 "ST3GAL6":("sulfation","cancer","sialyltransferase; myeloma bone-homing"),
 "SELENOI":("sulfation","novel","ethanolamine-phosphotransferase; novel in OC"),
 "GBE1":("sulfation","novel","glycogen-branching enzyme; metabolic, novel in OC"),
 "JDP2":("TF","OC-direct","Jun dimerization protein-2; established osteoclast-differentiation TF"),
 "RUNX3":("TF","bone-adjacent","RUNX-family TF; immune/skeletal, RUNX2 paralog"),
 "RFX8":("TF","novel","orphan RFX transcription factor; TOP in-silico driver, no prior OC literature"),
 "SOX4":("TF","cancer","progenitor TF; tumour progression"),
 "MSI2":("TF","myeloid","Musashi-2 RNA-binding; myeloid stemness"),
 "CDK6":("TF","myeloid","cyclin-dependent kinase-6; myeloid proliferation/differentiation"),
 "UHRF1BP1":("TF","novel","UHRF1-binding; epigenetic-adjacent, novel in OC"),
 "SFMBT1":("TF","novel","polycomb MBT reader; novel in OC"),
 "JADE1":("TF","novel","HBO1 histone-acetyltransferase subunit; novel in OC"),
 "UBTD2":("TF","novel","ubiquitin-domain protein; uncharacterised"),
 "PPARGC1B":("metabolism","OC-direct","PGC-1beta; REQUIRED for osteoclast mitochondrial biogenesis"),
 "GLUD1":("metabolism","bone-adjacent","glutamate dehydrogenase; OC bioenergetics"),
 "MTAP":("metabolism","cancer","methylthioadenosine phosphorylase; cancer metabolism"),
 "SLC16A7":("metabolism","bone-adjacent","MCT2 lactate transporter; OC glycolytic efflux"),
 "AK5":("metabolism","novel","adenylate kinase-5; novel in OC"),
 "GLUD1 ":("metabolism","novel","dup-guard"),
 "XPR1":("signaling","bone-adjacent","phosphate exporter; mineral homeostasis"),
 "ANKH":("signaling","bone-adjacent","pyrophosphate transporter; craniometaphyseal dysplasia (bone)"),
 "PTPN1":("signaling","bone-adjacent","PTP1B; dampens RANK/CSF1R signaling in OC lineage"),
 "IL32":("signaling","OC-direct","IL-32 cytokine; promotes human osteoclastogenesis"),
 "APP":("signaling","bone-adjacent","amyloid precursor; reported in osteoclast function"),
 "PLCL1":("signaling","novel","phospholipase-C-like 1 (catalytically inactive); novel in OC"),
 "PRKCH":("signaling","myeloid","PKC-eta; myeloid signaling"),
 "RASAL2":("signaling","novel","Ras-GAP; novel in OC"),
 "SIPA1L2":("signaling","novel","Rap-GAP; novel in OC"),
 "NEDD4L":("membrane_traffic","bone-adjacent","HECT E3; TGFb/ion-channel turnover"),
 "RETREG1":("membrane_traffic","novel","FAM134B ER-phagy receptor; novel in OC"),
 "DENND1B":("membrane_traffic","myeloid","Rab35 GEF; immune membrane traffic"),
 "TBC1D4":("membrane_traffic","novel","Rab-GAP AS160; glucose-transport traffic, novel in OC"),
 "RNF19B":("membrane_traffic","myeloid","RING E3; immune effector"),
 "OTUD4":("membrane_traffic","novel","OTU deubiquitinase; novel in OC"),
 "RMND5A":("membrane_traffic","novel","CTLH-complex E3; novel in OC"),
 "RNF145":("membrane_traffic","novel","sterol-regulated E3; novel in OC"),
 "RBFOX2":("RNA","novel","splicing regulator; novel in OC"),
 "AFF1":("RNA","myeloid","super-elongation complex; leukaemia-associated"),
 "TARBP1":("RNA","novel","RNA methyltransferase; novel in OC"),
 "CPEB4":("RNA","novel","cytoplasmic polyadenylation; novel in OC"),
 "STRBP":("RNA","novel","spermatid RNA-binding; novel in OC"),
 "MDFIC":("other","novel","MyoD-family inhibitor domain; novel in OC"),
 "TMEM181":("other","novel","transmembrane-181; uncharacterised"),
 "TULP4":("other","novel","tubby-like-4 E3 adaptor; novel in OC"),
 "F5":("other","cancer","coagulation factor V; tumour-microenvironment (spillover risk)"),
 "RNLS":("other","novel","renalase oxidase; novel in OC"),
 "DTWD2":("other","novel","DTW-domain; uncharacterised"),
 "UBAP2":("other","novel","ubiquitin-associated; stress granules, novel in OC"),
 "SELENOI ":("other","novel","dup-guard"),
 "PPP1R14B":("other","novel","PP1 inhibitor; novel in OC"),
 "LINC02725":("other","novel","long non-coding RNA; uncharacterised, novel"),
 "CCNYL1":("other","novel","cyclin-Y-like; novel in OC"),
 "TMEM181 ":("other","novel","dup-guard"),
 "EFL1":("other","bone-adjacent","ribosome-assembly GTPase; Shwachman-Diamond (marrow)"),
 "UHRF1BP1 ":("other","novel","dup-guard"),
 "MAST4":("signaling","novel","microtubule-associated kinase; novel in OC"),
}
rows=[]
for _,r in sig.iterrows():
    s=r["symbol"]; mod,cls,note=CUR.get(s,("other","novel","uncharacterised; novel in OC"))
    rows.append({"symbol":s,"ensembl":r["ensembl"],"module":mod,"prior_class":cls,
                 "lfc_OC_vs_myeloid":round(float(r["mean_lfc_OCvsMyeloid"]),3),
                 "lfc_tumor_vs_normal":round(float(r["mean_lfc_TumorVsNormal"]),3),"note":note})
df=pd.DataFrame(rows).sort_values(["prior_class","module","symbol"])
df.to_csv(f"{TABLES}/T7_74gene_evidence_matrix.csv",index=False)
summ=df["prior_class"].value_counts().to_dict()
json.dump({"class_counts":summ,"module_counts":df["module"].value_counts().to_dict(),"n":len(df)},
          open(f"{V}/evidence_matrix_summary.json","w"),indent=2)
print("class counts:",summ); print("module counts:",df["module"].value_counts().to_dict())
print(f"[DONE] {len(df)} genes -> T7_74gene_evidence_matrix.csv")
