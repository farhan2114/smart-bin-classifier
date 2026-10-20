**Smart Bin Classifier**



An image-based product quantity prediction and order validation prototype using computer vision and deep learning.



**1. Project Overview**



Smart Bin Classifier predicts the quantities of selected catalogue products from a photograph of a storage bin. It compares the model's predictions with an entered order and reports whether the quantities match.



The prototype covers a fixed catalogue of 10 ASINs. It is an experimental system, not a production-grade guarantee of inventory accuracy.



2. Features

Image-based quantity prediction for the selected catalogue products.

Order validation with per-product PASS/FAIL results.

Blind prediction using a bin identifier and uploaded photograph.

Demo photograph selection from the dedicated data/demo_images/ folder.

CSV logging of requests, model predictions, results, timestamps, and processing time.

Downloadable validation and blind-prediction results.


**3. Technology Stack**



\- Python

\- PyTorch and Torchvision

\- ResNet-18

\- Pandas, NumPy, Pillow

\- Scikit-learn

\- Streamlit



**4. Dataset**



Dataset: \[Amazon Bin Image Dataset: 536,434 Images at 224 × 224](https://www.kaggle.com/datasets/williamhyun/amazon-bin-image-dataset-536434-images-224x224?resource=download)



The project uses a selected subset of 2,000 bins and a catalogue of 10 products. The prepared splits contain 1,400 training images, 200 validation images, 200 test images, and 200 blind images.



The blind split contains photographs only. Its ground-truth metadata must not be used for predictions or validation.



The dataset is provided under the CC BY-NC-SA 3.0 US license. Use it according to its terms and cite the original dataset.



**5. Project Structure**



```text

smart-bin-classifier/

├── app.py

├── config.py

├── requirements.txt

├── README.md

├── notebooks/

├── src/

│   ├── data/

│   ├── models/

│   ├── evaluation/

│   └── utils/

├── data/

│   └── (local dataset files, excluded from Git)

├── artifacts/

│   └── (local model checkpoints, excluded from Git)

├── reports/

│   └── metrics/

├── logs/

└── tests/

```



**6. Setup**



Use a compatible Python environment. Python 3.11 is a reasonable starting point if recreating the environment, but package compatibility should be verified before installation.



Create and activate a virtual environment on Windows:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install -r requirements.txt

```



The dataset, processed splits, catalogue CSV, and trained model checkpoint must be available in the expected local paths before running the application.



7. Run the Application

Install dependencies and ensure the required catalogue CSV and trained model checkpoint are available at the paths expected by app.py.

python -m streamlit run app.py

Open the local URL printed in the terminal, usually http://localhost:8501.

Validate an order

Enter a quantity for each catalogue product.

Upload a bin photograph or select a demo photograph.

Submit the order for validation.

Review the predicted quantities and per-product PASS/FAIL results.

Download the validation results as a CSV file.

Blind prediction

Enter a bin identifier.

Upload a bin photograph.

Generate the model's quantity predictions.

Review or download the predictions.

Blind prediction uses the photograph as model input. Blind ground-truth metadata must not be used to generate predictions or make validation decisions.**8. Model Approaches**



The project explores:



\- A frequency-based baseline.

\- An RGB histogram feature model with logistic regression for product presence.

\- A pretrained ResNet-18 model for product presence.

\- A fine-tuned ResNet-18 model with tuned per-product thresholds.

\- A ResNet-18 quantity classifier that predicts quantity classes from 0 to 5 for each catalogue product.



The fine-tuned presence model achieved a validation micro F1 score of approximately 0.587 after threshold tuning. Since thresholds were tuned on the same validation set used for scoring, this result may be optimistic.



**9. Evaluation Results**



The held-out test evaluation generated 400 synthetic orders: 200 valid orders and 200 deliberately invalid orders.



| Metric | Result |

|---|---:|

| Overall PASS/FAIL accuracy | 79.25% |

| Invalid orders incorrectly passed | 0 of 200 |

| Valid orders incorrectly rejected | 83 of 200 |

| Valid-order false rejection rate | 41.50% |



These results apply only to the evaluated sample and the selected 10-product catalogue. Zero false passes in this experiment does not guarantee zero false passes in real-world use.



The valid and invalid orders were constructed using test ground-truth labels for evaluation. Ground-truth labels were used for scoring and order construction, not as model inputs.



\## 10. Limitations and Failure Analysis



\- The catalogue is limited to 10 selected products.

\- Low quantities, particularly the distinction between zero and one, remain challenging.

\- False positives can occur when an absent product is predicted as present.

\- False negatives can occur when a product in the bin is missed.

\- Rare higher quantities have limited training examples.

\- The app is a prototype and requires further testing before real inventory decisions.

\- Processing-time measurements should not be interpreted as complete end-to-end application latency.



Five example false-rejection cases are saved in `reports/metrics/five\_failure\_cases.csv`.



**11. Logging**



The application writes validation events to `logs/validation\_log.csv`. Entries include a timestamp, order, photo ID, model predictions, result, and processing time.



Local logs may be lost when a hosted instance restarts or its storage is reset. Download or export logs regularly if they must be retained.



12. Deployment

Platform: Streamlit Community Cloud

Live application: https://smart-bin-classifier-farhan21.streamlit.app/

The application requires compatible Python dependencies, the trained model checkpoint, and the catalogue CSV. The demo-photo feature requires images in data/demo_images/.

The application has been tested for order validation, blind prediction, and CSV downloads. Hosted filesystem logs may not persist across restarts, so they should not be treated as durable production storage.


**13. Production Monitoring and Retraining Plan**



Before production use:



1\. Monitor false-pass and false-rejection rates.

2\. Review prediction errors by product and quantity class.

3\. Collect representative, correctly labelled examples with appropriate permissions.

4\. Investigate data drift and image-quality problems.

5\. Retrain using a controlled training pipeline and separate validation data.

6\. Evaluate the updated model on a held-out test set before release.

7\. Track model versions and retain rollback capability.



14. AI Tools Used

AI assistance was used during development for code drafting, debugging, experiment planning, and documentation. Generated code and recommendations were reviewed and tested before use.

Example of an error caught during development: An indentation error in app.py caused Streamlit to fail during Python compilation. The error was identified from the deployment traceback, and the file was corrected and checked locally before redeployment.

AI assistance does not replace testing. Model metrics were evaluated using the project's evaluation scripts, and limitations are documented in this README.

**15. Git and Submission**



The repository should contain the source code, evaluation scripts, requirements, README, tests, and Git history.



Do not include the virtual environment, raw dataset, processed image splits, local logs, secrets, or large model checkpoints in the final ZIP unless explicitly required. Include clear instructions for obtaining or recreating the required model artifact.



Before submission, verify that the final ZIP contains the presentation (maximum 12 slides), source code, `.git` directory, and all other required deliverables.



