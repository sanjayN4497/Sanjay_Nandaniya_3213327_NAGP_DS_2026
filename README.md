# Customer Churn Prediction

An end-to-end Python solution using the supplied IBM Telco Customer Churn data. The executed notebook contains data checks, six EDA visualizations with business insights, two engineered features, decision-tree comparisons, five-fold hyperparameter tuning and class-weight evaluation, held-out evaluation, and model interpretation.

## Setup

Use Python 3.11. Open a terminal in `DS`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## View or rerun the analysis

```powershell
.\.venv\Scripts\python.exe -m notebook notebook/churn_analysis.ipynb
```

Select the virtual environment's Python kernel and run all cells in order. The notebook uses a stratified 70:30 train/test split and `random_state=42`. Model selection and all learned preprocessing use training-only cross-validation. Running all cells recreates `model/churn_model.pkl`, `sample_request.json`, and `sample_response.json`. Outputs are already included in the submitted notebook.

## Run the REST API

```powershell
.\.venv\Scripts\python.exe app.py
```

In a second terminal in the project directory:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary "@sample_request.json"
```

The exact response for this request is saved in `sample_response.json`. The response contains `prediction` (`Yes` or `No`) and `churn_probability` (0–1, rounded to six decimals).

Supply all 19 predictor fields shown in `sample_request.json`; `customerID` is optional and ignored. Use the categorical values in the supplied data dictionary, numeric 0/1 for `SeniorCitizen`, a nonnegative integer for `tenure`, and nonnegative numbers for charges. `TotalCharges` may be null or an empty string and is imputed by the saved pipeline. Other missing fields, invalid values, inconsistent service combinations, unknown fields, and malformed JSON return HTTP 400 with an `error` message. Do not include `Churn` in a request.

## Submission files

- `data/`: original dataset and data dictionary.
- `notebook/churn_analysis.ipynb`: executed analysis, model comparison, evaluation, interpretation, and inference checks.
- `preprocessing.py`: reusable feature engineering shared by training and API inference.
- `app.py`: Flask `POST /predict` endpoint, loading the saved pipeline at startup.
- `model/churn_model.pkl`: fitted feature engineering, imputers, encoder, and decision tree.
- `requirements.txt`: Python dependencies.
- `sample_request.json` and `sample_response.json`: verified request and actual API response.

The notebook checks that saving and loading preserves predictions and probabilities and that the API handles valid and invalid inputs. Recall is prioritized for retention, with F1 used for model selection to balance churn detection and outreach volume. Probabilities are decision-tree estimates rather than calibrated guarantees; the notebook reports the limits of the random-split evaluation.
