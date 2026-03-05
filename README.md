# krisp_ai_engineer_role_task

## Workflow

The pipeline loads a CSV dataset, deduplicates questions (exact then semantic), discovers clusters via an LLM (intent/action-style names), classifies every question into those clusters

**Steps:** load data → exact dedupe → semantic dedupe → LLM clustering → LLM classification → evaluation.

## Results

- **Clusters with counts:** `data/output.json` — list of clusters (id, name, description, count).
- **Evaluation metrics:** `data/evaluation.json` — ARI, NMI, homogeneity, completeness, v_measure

## Install dependencies (from pyproject.toml)
From the project root:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -e .
```

Then run the pipeline:

```bash
run-pipeline
# or
python -m src.main
```