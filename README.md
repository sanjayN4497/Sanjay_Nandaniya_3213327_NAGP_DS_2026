# Customer Churn Prediction

An end-to-end Python solution using the supplied IBM Telco Customer Churn data. The executed notebook contains data checks, six EDA visualizations with business insights, two engineered features, decision-tree comparisons, five-fold hyperparameter tuning and class-weight evaluation, held-out evaluation, and model interpretation.

## Project structure

The repository root is the project directory (equivalent to `customer_churn_project/` in the suggested submission structure).

```text
Sanjay_Nandaniya_3213327_NAGP_DS_2026/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   └── churn_model.pkl
├── app.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── sample_request.json
└── sample_response.json
```

## Demo recording

**Demo recording link:** TODO — add your recording URL here before submission.

## Setup

Install Git and Python 3.11, then open PowerShell. Clone the repository and enter the project directory:

```powershell
git clone https://github.com/sanjayN4497/Sanjay_Nandaniya_3213327_NAGP_DS_2026.git
cd Sanjay_Nandaniya_3213327_NAGP_DS_2026
```

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the commands below from this project directory.

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

Sample request (`POST /predict`, `Content-Type: application/json`):

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 18,
  "MonthlyCharges": 96.05,
  "TotalCharges": 1740.7
}
```

Sample response (`HTTP 200`) from the submitted model:

```json
{
  "churn_probability": 0.750078,
  "prediction": "Yes"
}
```

Supply all 19 predictor fields shown in `sample_request.json`; `customerID` is optional and ignored. Use the categorical values in the supplied data dictionary, numeric 0/1 for `SeniorCitizen`, a nonnegative integer for `tenure`, and nonnegative numbers for charges. `TotalCharges` may be null or an empty string and is imputed by the saved pipeline. Other missing fields, invalid values, inconsistent service combinations, unknown fields, and malformed JSON return HTTP 400 with an `error` message. Do not include `Churn` in a request.

## Submission files

- `data/`: original dataset and data dictionary.
- `notebook/churn_analysis.ipynb`: executed analysis, model comparison, evaluation, interpretation, and inference checks.
- `preprocessing.py`: reusable feature engineering shared by training and API inference.
- `app.py`: Flask `POST /predict` endpoint, loading the saved pipeline at startup.
- `model/churn_model.pkl`: fitted feature engineering, imputers, encoder, and decision tree.
- `requirements.txt`: Python dependencies.
- `README.md`: project structure, setup, notebook and API execution instructions, and sample API usage.
- `sample_request.json` and `sample_response.json`: verified request and actual API response.

## Workflow coverage

| Required stage | Implementation |
| --- | --- |
| Business Problem | Notebook introduction: identify likely churners for retention outreach. |
| Data | Supplied CSV and data dictionary in `data/`; loaded in the notebook. |
| Preparation | Notebook section 1: quality checks, target encoding, stratified split, and training-fitted imputation and encoding. |
| EDA | Notebook section 2: six visualization groups with business insights using training data. |
| Feature Engineering | Notebook section 3 and `preprocessing.py`: `SupportServicesCount` and `IsNewCustomer`. |
| Model | Notebook section 4: unrestricted, regularized, and tuned decision-tree comparisons. |
| Evaluation | Notebook section 5: held-out accuracy, precision, recall, F1, confusion matrix, and business trade-offs. |
| Interpretation | Notebook section 6: feature importance, tree visualization, and split rules. |
| Saved Model | Notebook section 7 and `model/churn_model.pkl`: complete fitted pipeline with reload checks. |
| API | `app.py`: validated `POST /predict`, demonstrated by the sample JSON files and notebook inference checks. |

Additional activities already implemented include five-fold stratified cross-validation, hyperparameter tuning, evaluation of balanced class weights, comparison of decision-tree configurations, and save/reload and API validation checks.

The notebook checks that saving and loading preserves predictions and probabilities and that the API handles valid and invalid inputs. Recall is prioritized for retention, with F1 used for model selection to balance churn detection and outreach volume. Probabilities are decision-tree estimates rather than calibrated guarantees; the notebook reports the limits of the random-split evaluation.
