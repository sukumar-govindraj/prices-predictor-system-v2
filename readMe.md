# 🏠 Prices Predictor System

**PyPI – Python Version**

---

## Problem Statement

For a given set of historical housing records, we are tasked to predict the sale price of a home before it’s listed. We leverage features such as lot area, year built, neighborhood, and more from a standard housing dataset to train a regression model. Accurate price prediction helps real-estate professionals and homeowners set competitive listings and make data-driven pricing decisions.

## Purpose

This repository demonstrates how to build, track, and deploy a production-ready ML pipeline using **ZenML** for orchestration and **MLflow** for experiment tracking & model serving.

---

🐍 **Python Requirements**
Within your Python environment of choice, run:

```bash
git clone https://github.com/your-org/prices-predictor-system.git
cd prices-predictor-system
pip install -r requirements.txt
```

**`requirements.txt`** includes:

```text
click==8.1.3
matplotlib==3.7.5
mlflow==2.15.1
mlflow_skinny==2.15.1
numpy==1.24.4
pandas==2.0.3
scikit_learn==1.3.2
seaborn==0.13.2
statsmodels==0.14.1
zenml==0.64.0
```

---

⚙️ **Configuration**
All pipeline & stack settings live in `config.yaml`:

```yaml
enable_cache: False
settings:
  docker:
    required_integrations:
      - mlflow
model:
  name: prices_predictor
  license: Apache 2.0
  description: Predictor of housing sale prices
  tags:
    - regression
    - housing
    - price_prediction
```

Customize file paths, caching, or metadata before your first run.

---

## 👍 The Solution

We supply three ZenML pipelines:

1. **Training Pipeline** (`run_pipeline.py`)

   * Ingest raw CSV from ZIP
   * Clean & impute missing values
   * Engineer features (log, scaling, one-hot)
   * Remove outliers (Z-score)
   * Split into train/test sets
   * Train a LinearRegression pipeline (with ColumnTransformer)
   * Evaluate metrics (MSE, R²) and log via MLflow

2. **Continuous Deployment Pipeline** (`run_deployment.py`)

   * Retrain & reevaluate the model
   * If performance meets threshold, redeploy as an MLflow service (3 workers)
   * Trigger a batch inference run to verify the live endpoint

3. **Batch Inference Pipeline** (invoked by deployment)

   * `dynamic_importer()`: Load new batch data from an API or file
   * `prediction_service_loader()`: Connect to the live MLflow endpoint
   * `predictor()`: Send batch payload, return predictions

---

## 📓 Diving into the Code

### 1. Run the Training Pipeline

```bash
python run_pipeline.py
```

This will log experiments & artifacts to your local MLflow backend.
Launch the MLflow UI to inspect results:

```bash
mlflow ui --backend-store-uri "$(python -c "from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri; print(get_tracking_uri())")"
```

### 2. Run Continuous Deployment & Batch Inference

```bash
python run_deployment.py
```

* Retrains & reevaluates
* Auto-redeploys as MLflow service if criteria met
* Executes a batch inference pipeline to confirm predictions

To stop the running MLflow service:

```bash
python run_deployment.py --stop-service
```

### 3. Single‐Record Prediction Client

With the service running at `http://127.0.0.1:8000/invocations`, run:

```bash
python sample_predict.py
```

Modify feature values in `sample_predict.py` as needed.

---

## 📈 Pipeline Details

### Training Pipeline (`ml_pipeline`)

* **data\_ingestion\_step**:

  ```python
  ```

def data\_ingestion\_step(file\_path: str) -> pd.DataFrame

````
  Reads a ZIP of CSVs into a DataFrame via `DataIngestorFactory`.

- **handle_missing_values_step**:
  ```python
def handle_missing_values_step(df: pd.DataFrame, strategy: str="mean") -> pd.DataFrame
````

Imputes or drops nulls (`mean`, `median`, `mode`, `constant`, or `drop`).

* **feature\_engineering\_step**:

  ```python
  ```

def feature\_engineering\_step(df: pd.DataFrame, strategy: str="log", features: list=None) -> pd.DataFrame

````
  Applies log transform, scaling, or one-hot encoding.

- **outlier_detection_step**:
  ```python
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame
````

Removes outliers using a Z-score threshold of 3.

* **data\_splitter\_step**:

  ```python
  ```

def data\_splitter\_step(df: pd.DataFrame, target\_column: str) -> Tuple\[X\_train, X\_test, y\_train, y\_test]

````
  Performs a train/test split.

- **model_building_step**:
  ```python
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> sklearn.pipeline.Pipeline
````

Constructs a `ColumnTransformer + LinearRegression` pipeline, autologs to MLflow.

* **model\_evaluator\_step**:

  ```python
  ```

def model\_evaluator\_step(trained\_model: Pipeline, X\_test: pd.DataFrame, y\_test: pd.Series) -> Tuple\[dict, float]

````
  Computes MSE, R², and other metrics via `ModelEvaluator`.

### Continuous Deployment Pipeline (`continuous_deployment_pipeline`)

- **ml_pipeline()**: Runs the training pipeline and returns the fitted model.  
- **mlflow_model_deployer_step**:
  ```python
def mlflow_model_deployer_step(model, deploy_decision: bool=True, workers: int=3)
````

Deploys (or updates) the model as an MLflow service if `deploy_decision` is `True`.

### Batch Inference Pipeline (`inference_pipeline`)

* **dynamic\_importer**:

  ```python
  ```

def dynamic\_importer() -> Any

````
  Loads new batch data from an external source.  
- **prediction_service_loader**:
  ```python
def prediction_service_loader(pipeline_name: str, pipeline_step_name: str, running=False) -> PredictionService
````

Connects to the active MLflow deployment.

* **predictor**:

  ```python
  ```

def predictor(service: PredictionService, input\_data: Any) -> np.ndarray

````
  Sends data to the MLflow service and returns predictions.

---

## 🔍 Exploratory Data Analysis
All initial EDA artifacts live in `analysis/`:

- **basic_data_inspection.py** – data types, ranges & missingness  
- **missing_values_analysis.py** – null‐value patterns  
- **univariate_analysis.py** – single‐feature distributions  
- **bivariate_analysis.py** – feature‐vs‐target scatterplots  
- **multivariate_analysis.py** – correlation heatmaps & PCA diagnostics  
- **EDA.ipynb** – combined notebook with narrative and plots  

Run them locally or open `analysis/EDA.ipynb` in Jupyter.

---

## 🧪 Testing
_No automated tests are included yet._  
**Next Steps**:

1. Add pytest-style tests under `tests/`.  
2. Test each `step/` and `src/` function.  
3. Run:
   ```bash
   pytest
````

Or manually verify:

* `python run_pipeline.py` completes successfully.
* `python run_deployment.py` spins up the MLflow service.
* `python sample_predict.py` returns valid predictions.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -am 'Add feature foo'`)
4. Push to your branch (`git push origin feature/foo`)
5. Open a Pull Request

---

## 📙 Resources & References

* ZenML docs: [https://docs.zenml.io](https://docs.zenml.io)
* MLflow docs: [https://mlflow.org/docs](https://mlflow.org/docs)

---

## 📄 License

**Apache 2.0** — see [LICENSE](./LICENSE) for details.
