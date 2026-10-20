\# Final Quantity Model Test Evaluation



\## Evaluation Setup



\- Dataset split: Held-out test split

\- Test images evaluated: 200

\- Catalogue products: 10

\- Prediction method: Argmax over quantity classes 0 to 5

\- Model checkpoint: `artifacts/resnet18\_quantity.pth`



\## Overall Results



| Metric | Result |

|---|---:|

| Whole-vector exact match | 60.00% |

| Quantity MAE, all product entries | 0.0565 |

| Actual present product entries | 103 |

| MAE on actually present products | 1.0000 |

| Exact quantity accuracy on actually present products | 27.18% |



\## Interpretation



The model predicted all ten catalogue-product quantities exactly for 60% of the evaluated test images.



The overall quantity MAE is influenced by the large number of zero-quantity labels. Therefore, it should not be interpreted independently of the metrics for products that were actually present.



For the 103 product entries with nonzero ground-truth quantities, exact quantity accuracy was 27.18%, and mean absolute quantity error was 1.0.



\## Limitations



\- The catalogue contains only ten selected products.

\- Product presence and exact quantity prediction remain challenging.

\- The dataset contains substantially more zero-quantity labels than positive quantities.

\- These results apply to the evaluated test split and prediction method.

\- The evaluation uses argmax predictions. The Streamlit application uses a separate threshold-based prediction rule, so these metrics do not directly describe the application's current threshold-based outputs.

\- The test results were collected for reporting and must not be used to tune the model or prediction threshold.



\## Conclusion



The quantity classifier is an experimental prototype. Further improvements and independent evaluation are required before using it for operational inventory decisions.
