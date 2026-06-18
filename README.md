# KG-LatentNet

Knowledge-guided latent state modeling of post-treatment vascular responses in
lung cancer patients.

This repository contains the public source code for KG-LatentNet, including
model definitions, data preprocessing utilities, training/evaluation scripts,
baseline adapters, and paper-result generation scripts.

## Data Availability

The patient-level cohort used in the study is not included in this public
repository because it contains non-public clinical data. The repository is
configured to keep raw data, processed folds, checkpoints, prediction files,
logs, and result tables out of version control.

To run the pipeline, place an authorized local copy of a de-identified dataset
under `data/raw/` and update:

- `configs/data.yaml`
- `configs/columns.yaml`

The current public config files are examples/placeholders and must be adapted
to the local data dictionary before preprocessing.

## Repository Structure

```text
configs/   Example configuration files
scripts/   End-to-end training, evaluation, and paper-result scripts
src/       KG-LatentNet model, preprocessing, baselines, and training modules
```

Generated artifacts are intentionally excluded:

```text
data/
results/
baseline_results/
logs/
checkpoints/
*.pkl
*.pt
*.pth
*.csv prediction/result outputs
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Typical Workflow

After preparing an authorized local dataset and updating the config files:

```bash
python scripts/00_check_data.py
python scripts/01_prepare_data.py
bash scripts/09_run_all_full_5fold.sh
python scripts/honest_real_final_outputs_validation_top.py
python scripts/regenerate_component_ablation_table.py
python scripts/regenerate_paper_style_figures.py
python scripts/supplement_best_model_missingness.py
```

The paper-ready scripts read model outputs from local `results/` directories.
Those outputs are not tracked in git.

## Privacy Guardrails

- Do not commit raw patient workbooks.
- Do not commit processed fold files or split files.
- Do not commit patient-level prediction CSVs.
- Do not commit checkpoints, logs, or intermediate experiment outputs.
- Before pushing, run `git status --short` and verify only source/config/docs
  files are staged.

## License

No open-source license has been selected yet. Please contact the authors before
redistributing or reusing this code beyond review/reproducibility purposes.
