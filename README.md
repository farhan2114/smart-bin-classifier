# Smart Bin Classifier

An image-based product quantity prediction and order validation prototype using computer vision and deep learning.

## 1. Project Overview

Smart Bin Classifier predicts the quantities of selected catalogue products from a photograph of a storage bin. It compares the predicted quantities against an order entered by the user and reports whether the order matches the model's predictions.

The prototype supports a fixed catalogue of 10 Amazon Standard Identification Numbers (ASINs). It is an experimental system intended to demonstrate image-based inventory recognition and order validation, not a production-grade guarantee of inventory accuracy.

## 2. Features

- **Image-based quantity prediction:** Predicts quantities for the 10 supported catalogue products from a bin photograph.
- **Order validation:** Compares entered quantities against model predictions and reports per-product PASS/FAIL results.
- **Blind prediction:** Generates predictions from an uploaded photograph and a bin identifier without using blind ground-truth labels as model inputs.
- **Demo image selection:** Supports selecting example photographs from `data/demo_images/`.
- **CSV logging:** Records validation events, model predictions, results, timestamps, and processing time.
- **Downloadable results:** Allows users to download validation and prediction results as CSV files.
- **Multiple modelling approaches:** Explores frequency-based, image-feature-based, presence-classification, and quantity-classification methods.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and model development |
| PyTorch | Deep learning and model training |
| Torchvision | ResNet-18 architecture and image transformations |
| ResNet-18 | Image-based product prediction |
| Pandas | Tabular data processing |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| Scikit-learn | Baseline models and evaluation utilities |
| Streamlit | Interactive web application |
| Matplotlib and Seaborn | Data analysis and visualisation |
| Plotly | Interactive visualisations |
| Joblib | Serialisation of supported baseline models |

The exact dependencies are listed in `requirements.txt`.

## 4. Dataset

**Dataset:** [Amazon Bin Image Dataset: 536,434 Images at 224 × 224](https://www.kaggle.com/datasets/williamhyun/amazon-bin-image-dataset-536434-images-224x224)

The project uses a selected subset of 2,000 bin images and a catalogue of 10 products.

### Dataset splits

| Split | Number of images | Purpose |
|---|---:|---|
| Training | 1,400 | Model training |
| Validation | 200 | Model selection and threshold tuning |
| Test | 200 | Held-out evaluation |
| Blind | 200 | Prediction demonstration without using ground-truth labels |
| **Total** | **2,000** | |

The blind split contains photographs and associated ordering information where applicable. Its ground-truth metadata must remain separate from model inputs and must not be used for model selection, threshold tuning, or validation decisions.

The dataset is distributed under the **CC BY-NC-SA 3.0 US** licence, as stated by the dataset source. Users should consult the original dataset page and follow its licence terms, attribution requirements, and restrictions.

The dataset and processed image splits are not intended to be committed to the Git repository. They must be obtained and prepared locally according to the project scripts.

## 5. Project Structure

The repository includes the application, configuration, dataset preparation scripts, model code, reports, and supporting files. The exact contents may change as development progresses.

```text
smart-bin-classifier/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── analyze_metadata.py
├── inspect_metadata.py
├── prepare_dataset.py
├── run_eda.py
├── select_bins.py
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── artifacts/
│   └── model checkpoints
├── data/
│   ├── demo_images/
│   └── local dataset and processed splits
├── reports/
│   └── metrics/
└── logs/
```

Some directories and files are generated during dataset preparation, training, evaluation, or application use. The local dataset, model checkpoints, and logs may be excluded from version control.

## 6. Installation and Setup

### Prerequisites

- Python installed on your system. Python 3.11 is a reasonable starting point, but compatibility with the dependencies should be verified.
- Git, if cloning the repository.
- The required dataset and prepared files.
- The trained model checkpoint and catalogue CSV expected by the application.

### Step 1: Clone the repository

```powershell
git clone https://github.com/farhan2114/smart-bin-classifier.git
cd smart-bin-classifier
```

If you already have the repository locally, open PowerShell in the existing project directory instead.

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, follow your organisation's security policy or use another supported method of activating a Python virtual environment.

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation requirements can vary by operating system and hardware. If installation fails, consult the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Step 4: Prepare the required files

Before launching the application, verify that the required files exist at the paths expected by `app.py` and the model-loading code.

These include:

- The catalogue CSV, such as `data/processed/catalogue.csv`.
- The trained quantity-model checkpoint, such as `artifacts/resnet18_quantity.pth`.
- Any additional metadata or processed files required by the selected application workflow.
- Demo photographs, if you want to use the demo image selection feature.

The dataset preparation and training scripts can be used to recreate supported files when the necessary source data is available. Exact commands depend on the configuration and requirements of the individual scripts.

## 7. Running the Application

From the activated virtual environment and the project root, run:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address in the terminal, usually:

[http://localhost:8501](http://localhost:8501)

Open that address in your browser.

### 7.1 Order Validation

1. Enter the expected quantity for each supported catalogue product.
2. Upload a bin photograph or select a demo photograph.
3. Submit the order for validation.
4. Review the predicted quantities and per-product PASS/FAIL results.
5. Download the available validation results as a CSV file.

The result reflects the model's predictions and the application's comparison logic. It should not be treated as a guarantee that the physical inventory has been counted correctly.

### 7.2 Blind Prediction

1. Enter the bin identifier.
2. Upload a bin photograph.
3. Generate quantity predictions.
4. Review the predicted quantities.
5. Download the available prediction results.

Blind prediction must use the photograph as model input without accessing its ground-truth labels to produce or adjust predictions.

## 8. Model Approaches

The project explores several approaches to product recognition and quantity estimation.

### 8.1 Frequency-Based Baseline

A baseline based on product frequency provides a simple reference for comparison with image-based approaches.

### 8.2 RGB Histogram Baseline

An RGB histogram feature representation combined with logistic regression is used to explore product-presence prediction.

### 8.3 ResNet-18 Product-Presence Model

A pretrained ResNet-18 model is fine-tuned to predict whether supported catalogue products are present in an image.

### 8.4 Threshold-Tuned Presence Model

Per-product decision thresholds are evaluated to adjust the balance between precision and recall.

The project recorded a validation micro F1 score of approximately **0.587** for a threshold-tuned presence-model evaluation. Because the thresholds were tuned using the validation set, the reported result may be optimistic and should not be interpreted as an independent test score.

### 8.5 ResNet-18 Quantity Classifier

A ResNet-18-based quantity classifier predicts one of six quantity classes, from 0 through 5, for each of the 10 catalogue products.

The output represents 60 product-quantity class scores in total.

The model's performance is evaluated separately from the order-validation experiment because quantity estimation and order PASS/FAIL classification measure different things.

## 9. Quantity Model Test Results

The final recorded quantity-model evaluation used **200 held-out test images**. Predictions were evaluated using the argmax quantity class for each product.

| Metric | Result |
|---|---:|
| Test images evaluated | 200 |
| Catalogue products | 10 |
| Whole-vector exact match | 60.00% |
| Mean absolute error across all product entries | 0.0565 |
| Actual nonzero product entries | 103 |
| Exact quantity accuracy on nonzero entries | 27.18% |
| Mean absolute error on nonzero entries | 1.00 |

### Interpretation

- **Whole-vector exact match** measures the proportion of images for which all 10 predicted quantities exactly match the ground truth.
- **Overall mean absolute error (MAE)** averages absolute quantity errors across all image-product entries, including absent products.
- **Exact quantity accuracy on nonzero entries** measures exact quantity predictions for products that are actually present.
- **MAE on nonzero entries** measures the average absolute quantity error for those present products.

The overall MAE is influenced by the large number of zero-quantity entries. Performance on products that are actually present is substantially more challenging, as reflected in the exact quantity accuracy and nonzero-entry MAE.

These figures describe the recorded test evaluation only. They do not establish production-level inventory accuracy.

## 10. Synthetic Order Validation Evaluation

A separate evaluation examined **400 synthetic order cases**: 200 valid orders and 200 deliberately invalid orders. The order-validation logic compared the model's predictions against the quantities specified by each order.

| Metric | Result |
|---|---:|
| Overall PASS/FAIL accuracy | 79.25% |
| Invalid orders incorrectly passed | 0 of 200 |
| Valid orders incorrectly rejected | 83 of 200 |
| Valid-order false rejection rate | 41.50% |

### Interpretation

The evaluation produced no false passes among the 200 deliberately invalid orders, but it incorrectly rejected 83 of the 200 valid orders.

This indicates that the evaluated system was conservative in this sample and still had a substantial false-rejection problem. Zero false passes in a limited experiment does not guarantee zero false passes on unseen data or in real-world use.

The synthetic orders were constructed and scored using test ground-truth quantities. Ground-truth labels were used for evaluation and order construction, not as inputs to the prediction model. The test results must not be used to tune model parameters or decision thresholds.

These order-validation metrics are separate from the quantity-model metrics in Section 9.

## 11. Limitations and Failure Analysis

The current prototype has several limitations:

- **Limited catalogue:** Only 10 selected products are supported.
- **Zero-versus-one confusion:** Distinguishing an absent product from a product present in a small quantity remains challenging.
- **False positives:** The model may predict that an absent product is present.
- **False negatives:** The model may fail to detect a product that is actually present.
- **Class imbalance:** Higher quantity classes have fewer training examples, which can limit model performance.
- **Visual variability:** Image quality, lighting, occlusion, product orientation, and overlapping objects can affect predictions.
- **Prototype-level validation:** The system requires more extensive testing before being used for operational inventory decisions.
- **Latency measurements:** Recorded processing time may not represent complete end-to-end application latency, including upload, network, and hosting overhead.
- **Limited generalisation evidence:** Results from one selected dataset and catalogue may not generalise to different products, storage conditions, or camera setups.

Example false-rejection cases have been recorded in `reports/metrics/five_failure_cases.csv`, if that report is present in the local project.

The evaluation reports and diagnostic scripts can help identify failure patterns and guide further experiments.

## 12. Logging

The application is configured to write validation events to:

```text
logs/validation_log.csv
```

Depending on the active workflow, log entries may include:

- Timestamp.
- Order information.
- Photograph or bin identifier.
- Model predictions.
- Validation result.
- Processing-time measurement.

Check the actual CSV schema and the application logging implementation for the exact fields recorded.

**Hosted-storage limitation:** Files written to a hosted application's local filesystem may not persist across restarts, redeployments, or storage resets. Download or export logs when they need to be retained. Durable production logging would require a suitable persistent storage service.

## 13. Deployment

**Hosting platform:** Streamlit Community Cloud

**Configured live application URL:**  
[https://smart-bin-classifier-farhan21.streamlit.app/](https://smart-bin-classifier-farhan21.streamlit.app/)

The application requires compatible dependencies and access to the catalogue CSV and trained model checkpoint at the paths expected by the application code. Demo-image selection also requires the relevant photographs to be available.

For deployment, verify that:

1. The configured repository and branch are correct.
2. The selected application entry point is `app.py`.
3. Dependencies are installed successfully.
4. Required model and catalogue files are accessible to the hosted application.
5. Order validation and blind prediction work.
6. CSV downloads function as expected.
7. Logs and other generated files are not assumed to be persistent.

The URL above is the configured application address. Its current availability and functionality should be checked directly before submission.

## 14. Production Monitoring and Retraining Plan

Before considering production use, the following improvements are recommended:

1. Monitor false-pass and false-rejection rates on representative data.
2. Review errors by product and quantity class.
3. Collect representative, correctly labelled examples with appropriate permissions.
4. Investigate changes in image quality, product appearance, and data distributions.
5. Retrain through a reproducible and controlled training pipeline.
6. Keep training, validation, and test data separated.
7. Evaluate candidate models on a held-out test set before release.
8. Track model versions, training configurations, and evaluation results.
9. Retain the ability to roll back to a previously validated model.
10. Add persistent storage and monitoring before relying on the system for operational inventory decisions.

## 15. AI Tools Used

AI assistance was used during development for code drafting, debugging, experiment planning, and documentation. Generated code and recommendations were reviewed and tested as part of the development process.

One documented debugging example involved an indentation error in `app.py` that caused Python compilation to fail. The error was identified from the application traceback, corrected, and checked locally before proceeding.

AI assistance does not replace software testing or model evaluation. The project documents its evaluation results and known limitations so that the prototype's capabilities are not overstated.

## 16. Git and Submission

The repository is intended to contain the source code, configuration, dependency list, README, relevant evaluation scripts, selected reports, presentation, and Git history.

Repository:

[https://github.com/farhan2114/smart-bin-classifier](https://github.com/farhan2114/smart-bin-classifier)

Presentation directory:

[https://github.com/farhan2114/smart-bin-classifier/tree/main/presentation](https://github.com/farhan2114/smart-bin-classifier/tree/main/presentation)

### Recommended submission contents

- Source code and required configuration.
- `requirements.txt`.
- Updated `README.md`.
- Project presentation, within the required maximum of 12 slides.
- Relevant model evaluation reports.
- Required Git history, including the `.git` directory, if explicitly requested.
- Instructions for obtaining or recreating any required model artifacts.

### Files generally excluded from a submission ZIP

Unless specifically required by the evaluator, avoid including:

- The `.venv/` virtual environment.
- The full raw dataset.
- Large processed image splits.
- Local logs containing test or runtime data.
- Secrets, API keys, credentials, and local environment files.
- Unnecessary temporary files, debug outputs, and backups.

**Important:** The application needs its trained model checkpoint and catalogue CSV to run. If the checkpoint is excluded from the submission ZIP, provide a clear and reproducible method for obtaining or recreating it, subject to the applicable dataset and model-distribution terms.

Before submitting, test the setup instructions in a clean environment, verify that the required model and data files are available, open the deployed application, and confirm that the final presentation and source files are included.

---

**Project status:** Experimental prototype for image-based product quantity prediction and order validation. Reported metrics apply to the documented evaluation samples and should not be interpreted as guarantees of real-world inventory accuracy.