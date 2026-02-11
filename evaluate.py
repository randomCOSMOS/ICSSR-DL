import torch
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.dataloader import get_loaders
from models.baseline import BaselineCNN
from models.improved import ImprovedCNN
import seaborn as sns
import matplotlib.pyplot as plt

device = "cpu"

_, _, test_loader = get_loaders(64)

model_dict = {
    "baseline": BaselineCNN(),
    "improved": ImprovedCNN()
}

os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

for model_name, model in model_dict.items():

    print(f"\nEvaluating {model_name}")

    model.load_state_dict(torch.load(f"outputs/models/{model_name}.pt"))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            preds = output.argmax(1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    metrics_path = f"outputs/metrics/{model_name}_metrics.txt"

    with open(metrics_path, "w") as f:
        f.write(f"MODEL : {model_name}\n")
        f.write(f"ACCURACY : {acc}\n\n")
        f.write(report)

    print(f"Saved metrics -> {metrics_path}")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm)
    plt.title(f"{model_name} confusion matrix")
    plt.savefig(f"outputs/plots/{model_name}_cm.png")
    plt.close()
