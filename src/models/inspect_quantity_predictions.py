
import runpy
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Reuse the validation dataset and model setup from the evaluator.
scope = runpy.run_path(
    str(Path(__file__).resolve().parent / "evaluate_quantity.py")
)

QuantityDataset = scope["QuantityDataset"]
models = scope["models"]
torch = scope["torch"]
DataLoader = scope["DataLoader"]
DEVICE = scope["DEVICE"]
MODEL_PATH = scope["MODEL_PATH"]
BATCH_SIZE = scope["BATCH_SIZE"]

dataset = QuantityDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

checkpoint = torch.load(
    MODEL_PATH, map_location=DEVICE, weights_only=False
)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.asins) * 6)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

actual_batches = []
predicted_batches = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images).view(images.size(0), len(dataset.asins), 6)
        predictions = logits.argmax(dim=2)
        actual_batches.append(labels.numpy())
        predicted_batches.append(predictions.cpu().numpy())

y_true = np.concatenate(actual_batches).ravel()
y_pred = np.concatenate(predicted_batches).ravel()

print("\nActual quantity counts (0 through 5):")
print(np.bincount(y_true, minlength=6))

print("\nPredicted quantity counts (0 through 5):")
print(np.bincount(y_pred, minlength=6))

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5]))

print("\nActual present entries predicted as absent:")
print(f"{np.sum((y_true > 0) & (y_pred == 0))} / {np.sum(y_true > 0)}")
