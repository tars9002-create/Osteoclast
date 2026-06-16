# Code Availability

This repository contains the code used for the manuscript-associated osteoclast
single-cell analysis. It is intended to be the public GitHub code record for review
and release.

The repository includes:

- Analysis scripts organized by pipeline stage.
- Environment and dependency notes.
- Main and supplementary figure exports.
- Lightweight repository checks for syntax and packaging consistency.

The repository excludes:

- Raw sequencing data downloaded from public archives.
- Large intermediate `.h5ad`, `.h5`, matrix, model, and cache files.
- Manuscript documents, local notebooks, cluster logs, and regenerated CSV/JSON outputs.

Large generated artifacts should be regenerated from the public data or archived separately
with a DOI-bearing service such as Zenodo, Figshare, or OSF if the journal requires direct
artifact access.

Before publication, update the repository release tag, manuscript citation, and archival DOI
if available.
