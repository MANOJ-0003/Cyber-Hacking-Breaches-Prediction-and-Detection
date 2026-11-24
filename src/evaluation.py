"""Create a confusion matrix plot from the trained model metadata."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def create_confusion_matrix(metadata_path=None, output_path=None):
    metadata_path = Path(metadata_path or ROOT / "models" / "feature_metadata.json")
    output_path = Path(output_path or ROOT / "reports" / "confusion_matrix.png")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrix = data["evaluation"]["confusion_matrix"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"], xlabel="Predicted", ylabel="Actual", title="Synthetic holdout confusion matrix")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(create_confusion_matrix())
