import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, f1_score,
    cohen_kappa_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from medipredict.config import FIGURES_DIR, RANDOM_SEED

# Set clean matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    auc = 0.5
    fpr_list, tpr_list = [], []
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fpr_list, tpr_list = fpr.tolist(), tpr.tolist()
        except Exception:
            pass

    return {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "cohen_kappa": round(float(kappa), 4),
        "roc_auc": round(float(auc), 4),
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list}
    }

def plot_learning_curves(estimator, X, y, filename="learning_curve.png"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring="recall",
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=RANDOM_SEED
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_sizes, train_mean, "o-", color="#0f766e", label="Training Score (Recall)")
    ax.plot(train_sizes, val_mean, "o--", color="#dc2626", label="Cross-Validation Score (Recall)")
    ax.set_title("Learning Curve (Bias-Variance Analysis)", fontsize=12)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Recall")
    ax.legend(loc="best")
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

def plot_validation_curve(estimator, X, y, param_name, param_range, filename="validation_curve.png"):
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring="recall"
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(param_range, train_mean, "o-", color="#0f766e", label="Training Score")
    ax.plot(param_range, val_mean, "o--", color="#2563eb", label="Validation Score")
    ax.set_title(f"Validation Curve for {param_name}", fontsize=12)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Recall Score")
    ax.legend(loc="best")
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

def plot_svm_kernel_trick(X_2d, y, filename="svm_kernel_trick.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Linear SVM
    clf_lin = SVC(kernel="linear", C=1.0, random_state=RANDOM_SEED)
    clf_lin.fit(X_2d, y)
    
    # RBF SVM
    clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED)
    clf_rbf.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    for ax, clf, title in zip(axes, [clf_lin, clf_rbf], ["Linear Kernel", "RBF Kernel"]):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=35)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Feature 1 (Standardized)")
        ax.set_ylabel("Feature 2 (Standardized)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

def plot_scree_plot(pca, filename="scree_plot.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    ax.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.7, color="#0f766e", label="Individual Variance")
    ax.step(range(1, len(cum_var) + 1), cum_var, where="mid", color="#2563eb", label="Cumulative Variance")
    ax.set_title("PCA Scree Plot (Explained Variance)", fontsize=12)
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

def plot_clustering_pca(X_pca, labels_dict, filename="patient_segmentation.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (algo_name, labels) in enumerate(labels_dict.items()):
        ax = axes[idx]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=30, alpha=0.8)
        ax.set_title(f"Clustering: {algo_name}", fontsize=11)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

def plot_roc_curves_overlay(roc_dict, filename="roc_overlay.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, roc_data in roc_dict.items():
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auc = roc_data.get("auc", 0.5)
        if len(fpr) > 0:
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title("ROC Curves Overlay", fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
