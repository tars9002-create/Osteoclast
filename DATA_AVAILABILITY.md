# Data Availability

No raw or controlled-access data are distributed in this repository.

The analysis scripts reference publicly available datasets from GEO and CELLxGENE Census,
including:

- `GSE266330`
- `GSE254672`
- `GSE168664`
- `GSE162454`
- `GSE152048`
- `GSE268835`
- `GSE212341`
- `GSE143791`
- CELLxGENE Census human osteoclast reference cells

Runtime downloads should be placed under `$OC_DOWNLOADS` or the default `$OC_PROJ/downloads`
layout expected by the scripts. Generated atlas objects, integrated `.h5ad` files, model
outputs, and stage-level CSV/JSON summaries are intentionally ignored by Git because they are
large or reproducible intermediates.

For manuscript review or publication, any non-regenerable intermediate artifact should be
archived outside GitHub with checksums and a stable DOI.
