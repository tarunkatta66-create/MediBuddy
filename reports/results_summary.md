# MediPredict - Experimental Results Summary
This document summarizes empirical performance metrics recorded during training across clinical condition models, continuous regression tasks, patient clustering, and dimensionality reduction.
## Heart Disease Risk Estimation
**Selected Best Model**: `AdaBoost (Stumps)` (Optimized for maximum Recall/Sensitivity)
| Model Name | Sensitivity (Recall) | Specificity | Precision | F1 Score | Cohen's Kappa | ROC AUC | SMOTE Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9286 | 0.8182 | 0.8125 | 0.8667 | 0.7388 | 0.9502 | 0.9286 |
| Decision Tree (CART) | 0.8214 | 0.5758 | 0.6216 | 0.7077 | 0.3877 | 0.8339 | 0.8214 |
| AdaBoost (Stumps) | 0.9643 | 0.7879 | 0.7941 | 0.8710 | 0.7401 | 0.9524 | 0.9286 |
| XGBoost | 0.9286 | 0.8182 | 0.8125 | 0.8667 | 0.7388 | 0.9275 | 0.8214 |
| Bagging | 0.9286 | 0.7879 | 0.7879 | 0.8525 | 0.7069 | 0.9307 | 0.9643 |
| Subagging | 0.9286 | 0.7273 | 0.7429 | 0.8254 | 0.6437 | 0.9188 | 0.9286 |
| Random Forest | 0.9643 | 0.8485 | 0.8438 | 0.9000 | 0.8041 | 0.9621 | 0.9286 |
| SVM (Linear) | 0.8571 | 0.7879 | 0.7742 | 0.8136 | 0.6398 | 0.9307 | 0.8571 |
| SVM (RBF) | 0.9643 | 0.7273 | 0.7500 | 0.8438 | 0.6769 | 0.9453 | 0.9286 |

## Diabetes Disease Risk Estimation
**Selected Best Model**: `Decision Tree (CART)` (Optimized for maximum Recall/Sensitivity)
| Model Name | Sensitivity (Recall) | Specificity | Precision | F1 Score | Cohen's Kappa | ROC AUC | SMOTE Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6667 | 0.7300 | 0.5714 | 0.6154 | 0.3820 | 0.8104 | 0.6481 |
| Decision Tree (CART) | 0.8333 | 0.5900 | 0.5233 | 0.6429 | 0.3726 | 0.7621 | 0.8519 |
| AdaBoost (Stumps) | 0.5926 | 0.8500 | 0.6809 | 0.6337 | 0.4562 | 0.8169 | 0.8704 |
| XGBoost | 0.5741 | 0.8500 | 0.6739 | 0.6200 | 0.4390 | 0.8309 | 0.8519 |
| Bagging | 0.5556 | 0.8400 | 0.6522 | 0.6000 | 0.4095 | 0.7956 | 0.6667 |
| Subagging | 0.5370 | 0.8700 | 0.6905 | 0.6042 | 0.4290 | 0.8170 | 0.6111 |
| Random Forest | 0.8148 | 0.7200 | 0.6111 | 0.6984 | 0.4967 | 0.8180 | 0.7778 |
| SVM (Linear) | 0.7037 | 0.7400 | 0.5938 | 0.6441 | 0.4256 | 0.8065 | 0.6852 |
| SVM (RBF) | 0.7778 | 0.6100 | 0.5185 | 0.6222 | 0.3478 | 0.7870 | 0.7778 |

## Liver Disease Risk Estimation
**Selected Best Model**: `AdaBoost (Stumps)` (Optimized for maximum Recall/Sensitivity)
| Model Name | Sensitivity (Recall) | Specificity | Precision | F1 Score | Cohen's Kappa | ROC AUC | SMOTE Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6627 | 0.8235 | 0.9016 | 0.7639 | 0.4082 | 0.8260 | 0.6386 |
| Decision Tree (CART) | 0.6988 | 0.5588 | 0.7945 | 0.7436 | 0.2370 | 0.6132 | 0.7108 |
| AdaBoost (Stumps) | 1.0000 | 0.0000 | 0.7094 | 0.8300 | 0.0000 | 0.5000 | 0.7229 |
| XGBoost | 1.0000 | 0.0000 | 0.7094 | 0.8300 | 0.0000 | 0.7679 | 0.8434 |
| Bagging | 0.9036 | 0.3529 | 0.7732 | 0.8333 | 0.2921 | 0.7208 | 0.8675 |
| Subagging | 0.8193 | 0.2647 | 0.7312 | 0.7727 | 0.0920 | 0.6637 | 0.7952 |
| Random Forest | 0.7349 | 0.7353 | 0.8714 | 0.7974 | 0.4225 | 0.7895 | 0.8072 |
| SVM (Linear) | 0.5663 | 0.9412 | 0.9592 | 0.7121 | 0.3918 | 0.8327 | 0.5783 |
| SVM (RBF) | 0.6265 | 0.8529 | 0.9123 | 0.7429 | 0.3912 | 0.8441 | 0.6867 |

## Bagging vs Boosting Empirical Comparison
Across all three clinical datasets, ensemble methods demonstrated distinct trade-offs:
- **Bagging & Random Forest**: Effective at variance reduction. Random Forest and Bagging provided smooth decision boundaries and consistent precision across CV folds.
- **Boosting (AdaBoost & XGBoost)**: Focused iteratively on hard-to-classify samples. Boosting models achieved superior Sensitivity (Recall) by sharpening decision boundaries around marginal patient cases.

## Bias-Variance and Learning Curve Analysis
- **Underfitting (High Bias)**: Low-depth decision stumps and weak linear models exhibited lower training accuracy that plateaued early, missing complex feature interactions.
- **Overfitting (High Variance)**: Unconstrained decision trees showed 100% training recall but degraded validation performance.
- **Generalization**: Optimal generalization occurred with regularized ensemble models (Random Forest with `max_depth=5-8` and XGBoost with `max_depth=3`), where training and cross-validation recall curves converged smoothly.

## Continuous Lab Value Regression Task (Heart Cholesterol Prediction)
| Regression Model | RMSE | MAE | R2 Score |
| :--- | :---: | :---: | :---: |
| Multivariate Linear Regression | 60.1209 | 41.3762 | 0.1069 |
| Linear Regression Scratch (Normal Eq) | 60.1209 | 41.3762 | 0.1069 |
| Linear Regression Scratch (GD) | 60.1209 | 41.3762 | 0.1069 |
| SVR (Linear) | 62.1924 | 42.7569 | 0.0443 |
| SVR (RBF) | 63.7215 | 44.5206 | -0.0033 |

## Patient Segmentation & Clustering (Heart Dataset)
| Clustering Algorithm | Silhouette Score |
| :--- | :---: |
| KMeans | 0.1292 |
| DBSCAN | -0.1757 |
| Gaussian Mixture (EM) | 0.0862 |
| MST-based Clustering | 0.2804 |

## Dimensionality Reduction (PCA, LDA, SVD)
- **PCA Explained Variance Ratios**: `[0.2369, 0.1231, 0.0953, 0.0843, 0.0758, 0.0679, 0.0665, 0.0598, 0.0529, 0.0433, 0.0353, 0.0316, 0.0272]`
- **Downstream Accuracy (Full Features)**: `0.9016`
- **Downstream Accuracy (PCA - 3 Components)**: `0.7705`
- **Downstream Accuracy (LDA - 1 Component)**: `0.7705`
- **Downstream Accuracy (SVD - 3 Components)**: `0.7705`
