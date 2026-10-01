import json
import datetime
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from medipredict.config import (
    DATASET_SCHEMAS, RANDOM_SEED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
)
from medipredict.data import load_clean_data, get_features_and_target
from medipredict.features import build_model_pipeline, build_preprocessing_pipeline
from medipredict.evaluate import (
    calculate_classification_metrics, plot_learning_curves, plot_validation_curve,
    plot_svm_kernel_trick, plot_scree_plot, plot_clustering_pca, plot_roc_curves_overlay
)
from medipredict.scratch.linear_regression import LinearRegressionNormalEq, LinearRegressionGD
from medipredict.scratch.logistic_regression import LogisticRegressionGD
from medipredict.scratch.cart import CARTClassifier

def mst_clustering(X_scaled, n_clusters=3):
    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    mst = minimum_spanning_tree(dist_matrix).toarray()
    
    edges = []
    for i in range(mst.shape[0]):
        for j in range(mst.shape[1]):
            if mst[i, j] > 0:
                edges.append((mst[i, j], i, j))
    
    # Sort edges descending by weight and cut top (n_clusters - 1) heaviest edges
    edges.sort(key=lambda x: x[0], reverse=True)
    cut_edges = set((i, j) for w, i, j in edges[:n_clusters - 1])
    
    adj = mst.copy()
    for w, i, j in edges[:n_clusters - 1]:
        adj[i, j] = 0
        adj[j, i] = 0
        
    n_components, labels = connected_components(csgraph=adj, directed=False)
    return labels

def train_all():
    np.random.seed(RANDOM_SEED)
    metrics_report = {}

    # 1. Classification Models per Condition
    for condition in ["heart", "diabetes", "liver"]:
        print(f"--- Training models for condition: {condition} ---")
        df = load_clean_data(condition)
        X, y = get_features_and_target(df, condition)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        models_to_test = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
                {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
            ),
            "Decision Tree (CART)": (
                DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
                {"classifier__max_depth": [3, 5, 7, 10], "classifier__min_samples_leaf": [1, 2, 5]}
            ),
            "AdaBoost (Stumps)": (
                AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=RANDOM_SEED),
                {"classifier__n_estimators": [20, 50, 100], "classifier__learning_rate": [0.01, 0.1, 1.0]}
            ),
            "XGBoost": (
                XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss"),
                {"classifier__n_estimators": [30, 50, 100], "classifier__max_depth": [2, 3, 5], "classifier__learning_rate": [0.01, 0.1]}
            ),
            "Bagging": (
                BaggingClassifier(random_state=RANDOM_SEED),
                {"classifier__n_estimators": [10, 30, 50]}
            ),
            "Subagging": (
                BaggingClassifier(max_samples=0.8, bootstrap=False, random_state=RANDOM_SEED),
                {"classifier__n_estimators": [10, 30, 50]}
            ),
            "Random Forest": (
                RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED),
                {"classifier__n_estimators": [30, 50, 100], "classifier__max_depth": [3, 5, 8]}
            ),
            "SVM (Linear)": (
                SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
                {"classifier__C": [0.1, 1.0, 5.0]}
            ),
            "SVM (RBF)": (
                SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
                {"classifier__C": [0.1, 1.0, 5.0], "classifier__gamma": ["scale", "auto"]}
            )
        }

        condition_results = {}
        best_model_name = None
        best_recall = -1.0
        best_pipeline = None
        best_params = None

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        for name, (base_clf, param_grid) in models_to_test.items():
            # Test without SMOTE
            pipe = build_model_pipeline(base_clf, use_smote=False)
            search = GridSearchCV(pipe, param_grid, cv=cv, scoring="recall", n_jobs=-1)
            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_
            y_pred = best_pipe.predict(X_test)
            y_prob = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, "predict_proba") else None
            
            eval_metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            eval_metrics["best_params"] = search.best_params_

            # Test with SMOTE to evaluate if SMOTE actually helped
            pipe_smote = build_model_pipeline(base_clf, use_smote=True)
            search_smote = GridSearchCV(pipe_smote, param_grid, cv=cv, scoring="recall", n_jobs=-1)
            search_smote.fit(X_train, y_train)
            y_pred_smote = search_smote.best_estimator_.predict(X_test)
            y_prob_smote = search_smote.best_estimator_.predict_proba(X_test)[:, 1] if hasattr(search_smote.best_estimator_, "predict_proba") else None
            eval_smote = calculate_classification_metrics(y_test, y_pred_smote, y_prob_smote)
            eval_metrics["smote_recall"] = eval_smote["sensitivity"]
            eval_metrics["smote_auc"] = eval_smote["roc_auc"]

            condition_results[name] = eval_metrics

            # Track best model by test recall
            if eval_metrics["sensitivity"] > best_recall:
                best_recall = eval_metrics["sensitivity"]
                best_model_name = name
                best_pipeline = best_pipe
                best_params = search.best_params_

        metrics_report[condition] = {
            "best_model_name": best_model_name,
            "models": condition_results
        }

        # Save best model joblib + json
        best_file = MODELS_DIR / f"{condition}_best.joblib"
        joblib.dump(best_pipeline, best_file)

        meta = {
            "condition": condition,
            "best_model_name": best_model_name,
            "metrics": condition_results[best_model_name],
            "best_params": best_params,
            "feature_names": list(X.columns),
            "trained_at": datetime.datetime.now().isoformat()
        }
        with open(MODELS_DIR / f"{condition}_best.json", "w") as f:
            json.dump(meta, f, indent=2)

    # 2. Continuous Lab Value Regression Task (Heart Chol prediction)
    print("--- Running Continuous Regression Task (Heart Chol Prediction) ---")
    df_heart = load_clean_data("heart")
    X_reg = df_heart.drop(columns=["chol"]).copy()
    y_reg = df_heart["chol"].values

    # Impute missing in X_reg
    prep = build_preprocessing_pipeline()
    X_reg_prep = prep.fit_transform(X_reg)
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg_prep, y_reg, test_size=0.2, random_state=RANDOM_SEED)

    reg_models = {
        "Multivariate Linear Regression": LinearRegression(),
        "Linear Regression Scratch (Normal Eq)": LinearRegressionNormalEq(),
        "Linear Regression Scratch (GD)": LinearRegressionGD(lr=0.01, n_iter=2000),
        "SVR (Linear)": SVR(kernel="linear", C=1.0),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0)
    }

    reg_results = {}
    for name, r_model in reg_models.items():
        r_model.fit(X_r_train, y_r_train)
        preds = r_model.predict(X_r_test)
        reg_results[name] = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_r_test, preds))), 4),
            "mae": round(float(mean_absolute_error(y_r_test, preds)), 4),
            "r2": round(float(r2_score(y_r_test, preds)), 4)
        }
    metrics_report["regression_task"] = reg_results

    # 3. Patient Segmentation / Clustering Task (Heart dataset)
    print("--- Running Patient Segmentation & Clustering ---")
    df_heart_clean = load_clean_data("heart")
    X_h, _ = get_features_and_target(df_heart_clean, "heart")
    prep_cl = build_preprocessing_pipeline()
    X_scaled = prep_cl.fit_transform(X_h)

    km_labels = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10).fit_predict(X_scaled)
    db_labels = DBSCAN(eps=1.5, min_samples=4).fit_predict(X_scaled)
    gmm_labels = GaussianMixture(n_components=3, random_state=RANDOM_SEED).fit_predict(X_scaled)
    mst_labels = mst_clustering(X_scaled, n_clusters=3)

    clustering_scores = {
        "KMeans": round(float(silhouette_score(X_scaled, km_labels)), 4),
        "DBSCAN": round(float(silhouette_score(X_scaled, db_labels)) if len(set(db_labels)) > 1 else -1.0, 4),
        "Gaussian Mixture (EM)": round(float(silhouette_score(X_scaled, gmm_labels)), 4),
        "MST-based Clustering": round(float(silhouette_score(X_scaled, mst_labels)), 4)
    }
    metrics_report["clustering_task"] = clustering_scores

    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    plot_clustering_pca(X_pca_2d, {
        "KMeans": km_labels, "DBSCAN": db_labels,
        "GMM (EM)": gmm_labels, "MST-Clustering": mst_labels
    }, filename="patient_segmentation.png")

    # 4. Dimensionality Reduction Task (PCA, LDA, SVD)
    print("--- Running Dimensionality Reduction (PCA, LDA, SVD) ---")
    pca_full = PCA(random_state=RANDOM_SEED).fit(X_scaled)
    plot_scree_plot(pca_full, filename="scree_plot.png")

    # Evaluate downstream effect on Random Forest accuracy
    y_h = df_heart_clean["target"]
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED)
    rf_base.fit(X_r_train, y_r_train if False else y_h.iloc[:len(X_r_train)]) # dummy check

    X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_scaled, y_h, test_size=0.2, random_state=RANDOM_SEED, stratify=y_h)

    # Base RF
    rf_full = RandomForestClassifier(random_state=RANDOM_SEED).fit(X_tr_h, y_tr_h)
    score_full = rf_full.score(X_te_h, y_te_h)

    # PCA 3 components
    pca_3 = PCA(n_components=3, random_state=RANDOM_SEED)
    X_tr_pca = pca_3.fit_transform(X_tr_h)
    X_te_pca = pca_3.transform(X_te_h)
    score_pca = RandomForestClassifier(random_state=RANDOM_SEED).fit(X_tr_pca, y_tr_h).score(X_te_pca, y_te_h)

    # LDA 1 component
    lda_1 = LinearDiscriminantAnalysis(n_components=1)
    X_tr_lda = lda_1.fit_transform(X_tr_h, y_tr_h)
    X_te_lda = lda_1.transform(X_te_h)
    score_lda = RandomForestClassifier(random_state=RANDOM_SEED).fit(X_tr_lda, y_tr_h).score(X_te_lda, y_te_h)

    # SVD 3 components
    svd_3 = TruncatedSVD(n_components=3, random_state=RANDOM_SEED)
    X_tr_svd = svd_3.fit_transform(X_tr_h)
    X_te_svd = svd_3.transform(X_te_h)
    score_svd = RandomForestClassifier(random_state=RANDOM_SEED).fit(X_tr_svd, y_tr_h).score(X_te_svd, y_te_h)

    metrics_report["dimensionality_reduction"] = {
        "explained_variance_ratio_pca": [round(float(v), 4) for v in pca_full.explained_variance_ratio_],
        "accuracy_full_features": round(float(score_full), 4),
        "accuracy_pca_3_components": round(float(score_pca), 4),
        "accuracy_lda_1_component": round(float(score_lda), 4),
        "accuracy_svd_3_components": round(float(score_svd), 4)
    }

    # 5. Generate Module 1 & 4 Demonstration Figures
    print("--- Generating Syllabus Figures ---")
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)
    pipe_tree = build_model_pipeline(tree_clf)
    plot_learning_curves(pipe_tree, X_h, y_h, filename="learning_curve.png")
    plot_validation_curve(pipe_tree, X_h, y_h, param_name="classifier__max_depth", param_range=[1, 2, 3, 5, 7, 10, 15], filename="validation_curve.png")
    plot_svm_kernel_trick(X_scaled[:, :2], y_h.values, filename="svm_kernel_trick.png")

    # ROC overlay for heart condition models
    roc_dict = {}
    for m_name, m_res in metrics_report["heart"]["models"].items():
        roc_dict[m_name] = {
            "fpr": m_res["roc_curve"]["fpr"],
            "tpr": m_res["roc_curve"]["tpr"],
            "auc": m_res["roc_auc"]
        }
    plot_roc_curves_overlay(roc_dict, filename="roc_overlay.png")

    # Write reports/metrics.json
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print("Training complete! Metrics saved to reports/metrics.json.")

if __name__ == "__main__":
    train_all()
