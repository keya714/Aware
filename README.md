# Aware

Repository for the AWARe biometric authentication project.

## Environment setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Clarkson II data access and placement

Clarkson II is not redistributed in this repository. Request the dataset from
Clarkson CITeR by contacting `dhou@clarkson.edu` and identifying the release;
access requires an agreement with the dataset authors. After approval, place
the extracted raw user files under:

```text
filtered_dataset/
```

The preprocessing notebooks also accept a sibling `clarkson2/` directory;
`filtered_dataset/` is the default path used by the current checkout.

Do not commit the raw archive, extracted event logs, or generated model data.
The `.gitignore` rules exclude local data and generated outputs.

## Repository map and reproducibility workflow

- `clarkson2_preprocessing/`: Jupyter notebooks for schema validation,
  anomaly investigation, threshold sensitivity, event features, cleaning,
  windowing, chronological splitting, and normalization.
- `clarkson2_attention_model/`: sequence preparation and the attention-model
  training baseline.
- `clarkson2_30day_model/`: 30-day dataset construction, baseline evaluation,
  longitudinal stage evaluation, drift analysis, causal tracing, ROME-style
  editing, and report-figure generation.
- `scripts/make_rubric_figures.py`: creates the report correlation matrix,
  univariate hold-time figure, class/window-balance figure, and threshold plot.

Run commands from the repository root unless noted otherwise:

```bash
python clarkson2_30day_model/build_30day_dataset.py
python clarkson2_attention_model/prepare_sequences.py
python clarkson2_attention_model/train_attention.py --seed 228
python clarkson2_30day_model/evaluate_remaining_stages.py
python clarkson2_30day_model/analyze_user_drift.py
python clarkson2_30day_model/causal_trace_stage1.py
python clarkson2_30day_model/model_editing.py --v-steps 20 --adaptation-size 64 --replay-size 64
python scripts/make_rubric_figures.py
```

The main saved artifacts are under
`clarkson2_30day_model/results/`, including `attention_metrics.json`,
`remaining_stage_metrics.csv`, `user_drift_summary.json`, causal-tracing
outputs, and model-editing metrics. Report figures are written to
`clarkson2_30day_model/figures/`.

## Clarkson II preprocessing

The [`clarkson2_preprocessing`](clarkson2_preprocessing) folder contains the completed Clarkson II data-preparation notebooks:

- validation and timestamp normalization
- anomaly investigation
- event-level feature extraction
- hold-time threshold sensitivity analysis
- cleaning, windowing, chronological splitting, normalization, and verification-trial generation

The raw Clarkson II dataset and generated output artifacts are intentionally not stored in this repository. See `clarkson2_preprocessing/Progress.txt` for the current processing results and output locations.
### Longitudinal drift figure

After the saved longitudinal metrics artifact is available, run:

```bash
python clarkson2_30day_model/make_drift_metrics_figure.py
```

This writes `clarkson2_30day_model/figures/longitudinal_drift_metrics.png`, which plots EER, FAR, and FRR for the 30-day test and Stages 1--3. The script fails if the metrics artifact does not contain measured `stage`, `eer`, `far`, and `frr` fields; it does not infer EER from fixed-threshold FAR/FRR.
