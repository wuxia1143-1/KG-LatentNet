# Data Availability

The patient-level dataset used for the KG-LatentNet experiments is not public
and is not included in this repository.

This repository contains source code only. It excludes:

- raw clinical workbooks;
- processed feature matrices and fold files;
- train/validation/test split files;
- patient-level prediction CSV files;
- checkpoints and model weights;
- logs and intermediate result folders.

Researchers who have appropriate institutional approval and data-use permission
can place their own de-identified dataset under `data/raw/` and adapt
`configs/data.yaml` and `configs/columns.yaml` to their local schema.

Optional patient-specific endpoint-month corrections should be stored only in
`configs/local_endpoint_month_overrides.csv`. This file is ignored by Git
because it contains private cohort identifiers.
