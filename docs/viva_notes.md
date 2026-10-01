# MediPredict - Examiner Viva Notes (Sem VII CSC701 Machine Learning)

This document provides quick theoretical reference, mathematical formulas, project application context, and sample viva oral questions for all algorithms evaluated in MediPredict.

---

## 1. Logistic Regression
- **Intuition**: Models the probability of a binary outcome using a linear combination of input features mapped through a sigmoid logistic function into the range `[0, 1]`.
- **Key Formula**:
  $$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-(w^T x + b)}}, \quad \mathcal{L}(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$
- **Where Used in Project**: Evaluated as a baseline classifier across Heart, Diabetes, and Liver datasets, and implemented from scratch via gradient descent in `src/medipredict/scratch/logistic_regression.py`.
- **Likely Examiner Question**: *Why do we use the Sigmoid function instead of a step function in Logistic Regression?*
- **Good Viva Answer**: The step function has zero derivative everywhere except at zero where it is undefined, making gradient descent optimization impossible. The Sigmoid function is smooth, continuous, and everywhere differentiable, allowing exact gradient updates via backpropagation.

---

## 2. Decision Tree (CART / Gini Index)
- **Intuition**: A greedy binary decision tree that repeatedly splits the feature space to maximize node purity, evaluated using the Gini Impurity metric.
- **Key Formula**:
  $$\text{Gini}(D) = 1 - \sum_{k=1}^K p_k^2, \quad \Delta \text{Gini}(D, A) = \text{Gini}(D) - \frac{|D_L|}{|D|} \text{Gini}(D_L) - \frac{|D_R|}{|D|} \text{Gini}(D_R)$$
- **Where Used in Project**: Used as a standalone interpretable model across all conditions, as base estimators in AdaBoost/Bagging, and implemented from scratch in `src/medipredict/scratch/cart.py`.
- **Likely Examiner Question**: *How does Gini impurity differ from Information Gain (Entropy)?*
- **Good Viva Answer**: Gini impurity measures the probability of misclassifying a randomly chosen element from the set, whereas Information Gain uses Shannon Entropy ($-\sum p \log_2 p$). Gini is computationally faster because it avoids logarithmic operations while yielding nearly identical split locations.

---

## 3. Random Forest
- **Intuition**: An ensemble of decision trees trained on bootstrap samples of data (bagging) with random feature selection at each split to decorrelate individual trees and reduce overall model variance.
- **Key Formula**:
  $$\hat{f}_{rf}^B(x) = \frac{1}{B} \sum_{b=1}^B T_b(x), \quad \text{Feature sampling per split: } m = \lfloor \sqrt{p} \rfloor$$
- **Where Used in Project**: Applied as a robust classifier across Heart, Diabetes, and Liver conditions; also used to assess downstream feature reduction accuracy in Module 6.
- **Likely Examiner Question**: *Why does Random Forest select a random subset of features at each node split instead of using all features?*
- **Good Viva Answer**: If one or two features are dominant predictors, every tree in a simple bagged ensemble would choose them for top splits, creating strongly correlated trees. Random feature selection decorrelates the decision trees, ensuring true variance reduction when averaging their predictions.

---

## 4. AdaBoost (Decision Stumps)
- **Intuition**: A sequential boosting ensemble where weak learners (single-split decision stumps, `max_depth=1`) are trained iteratively, re-weighting misclassified instances after each round to focus on hard samples.
- **Key Formula**:
  $$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right), \quad w_{i}^{(m+1)} = w_{i}^{(m)} \exp \left( \alpha_m \cdot \mathbb{I}(y_i \neq h_m(x_i)) \right)$$
- **Where Used in Project**: Implemented for all three datasets under Module 3 syllabus coverage, achieving top recall performance on the Heart dataset.
- **Likely Examiner Question**: *What is a decision stump and why is it used as a weak learner in AdaBoost?*
- **Good Viva Answer**: A decision stump is a 1-level decision tree that splits data on a single attribute threshold. It has high bias and low variance, satisfying the theoretical definition of a weak learner (performing slightly better than random guessing), allowing AdaBoost to build a strong classifier by iteratively reducing bias.

---

## 5. XGBoost (Extreme Gradient Boosting)
- **Intuition**: An optimized gradient boosting framework that fits new trees to the negative gradient (pseudo-residuals) of the loss function, incorporating L1/L2 regularization on leaf weights to prevent overfitting.
- **Key Formula**:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
- **Where Used in Project**: Evaluated across all conditions with tree-based SHAP explanations (`TreeExplainer`).
- **Likely Examiner Question**: *How does XGBoost differ from standard Gradient Boosting (GBDT)?*
- **Good Viva Answer**: XGBoost uses second-order Taylor expansion (gradients $g_i$ and Hessians $h_i$) for accurate loss approximation, includes explicit L1/L2 tree complexity regularization, supports column subsampling, and handles missing values natively via split-direction learning.

---

## 6. Support Vector Machine (SVM) & Kernel Trick
- **Intuition**: Finds the optimal maximum-margin hyperplane separating classes in feature space. The Kernel Trick computes dot products in a high-dimensional implicit feature space without performing explicit feature transformation.
- **Key Formula**:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad K_{RBF}(x, z) = \exp(-\gamma \|x - z\|^2)$$
- **Where Used in Project**: Linear and RBF SVM models evaluated for risk classification; 2D kernel trick boundary visualization generated in `reports/figures/svm_kernel_trick.png`. SVR evaluated on continuous cholesterol regression.
- **Likely Examiner Question**: *What is the role of parameter C and gamma in RBF SVM?*
- **Good Viva Answer**: `C` controls the trade-off between margin width and misclassification penalty (large `C` penalizes errors heavily, risking overfitting; small `C` allows wider margins). `gamma` defines the sphere of influence for single training points (high `gamma` creates tight, complex boundaries; low `gamma` smooths the boundary).

---

## 7. PCA, LDA, and SVD (Dimensionality Reduction)
- **Intuition**:
  - **PCA**: Unsupervised projection maximizing variance along orthogonal principal axes.
  - **LDA**: Supervised projection maximizing between-class scatter while minimizing within-class scatter.
  - **SVD**: Matrix factorization decomposing feature matrices into singular vectors and values.
- **Key Formula**:
  $$\text{PCA: } \Sigma v = \lambda v, \quad \text{LDA: } J(w) = \frac{w^T S_B w}{w^T S_W w}, \quad \text{SVD: } X = U \Sigma V^T$$
- **Where Used in Project**: Evaluated under Module 6. Scree plot produced in `reports/figures/scree_plot.png`, and downstream RF accuracy compared across full, PCA, LDA, and SVD features.
- **Likely Examiner Question**: *Why is LDA supervised while PCA is unsupervised?*
- **Good Viva Answer**: PCA ignores target class labels and only looks at overall feature covariance to find directions of maximum variance. LDA uses class labels to maximize the ratio of between-class variance to within-class variance, specifically optimizing for class separability.

---

## 8. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Intuition**: Groups points that are closely packed together (density reachable within radius $\epsilon$ with at least `min_samples`), marking points in low-density regions as noise (-1).
- **Key Formula**:
  $$N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \epsilon\}, \quad |N_\epsilon(p)| \ge \text{MinPts}$$
- **Where Used in Project**: Patient segmentation on standardized Heart parameters evaluated with Silhouette score in Module 5.
- **Likely Examiner Question**: *What is the advantage of DBSCAN over K-Means?*
- **Good Viva Answer**: DBSCAN does not require specifying the number of clusters $k$ in advance, can discover clusters of arbitrary non-spherical shapes, and is robust to outliers/noise.

---

## 9. Gaussian Mixture Model (EM Algorithm)
- **Intuition**: Assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters, optimized iteratively using Expectation-Maximization.
- **Key Formula**:
  $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
- **Where Used in Project**: Soft patient clustering under Module 5.
- **Likely Examiner Question**: *How does EM differ from hard K-Means assignment?*
- **Good Viva Answer**: K-Means performs hard assignment, placing each point into exactly one cluster. GMM via EM performs soft assignment, assigning a posterior probability $\gamma_{ik}$ that a data point belongs to each Gaussian component.

---

## 10. Minimum Spanning Tree (MST) Based Clustering
- **Intuition**: Constructs a Minimum Spanning Tree over the Euclidean distance graph of data points and cuts the $(k - 1)$ heaviest edges to isolate $k$ connected cluster components.
- **Key Formula**:
  $$\text{MST} = \arg\min_{T \subset G} \sum_{e \in T} w(e), \quad \text{Cut edges: } \{e_1, e_2, \dots, e_{k-1}\} \text{ with max } w(e)$$
- **Where Used in Project**: Implemented from scratch using `scipy.sparse.csgraph.minimum_spanning_tree` in `src/medipredict/train.py` under Module 5 syllabus requirements.
- **Likely Examiner Question**: *Why does cutting the heaviest edges in an MST yield meaningful clusters?*
- **Good Viva Answer**: Edges with maximum weight in an MST correspond to the largest spatial gaps between point groupings. Removing these heaviest bridges naturally breaks the tree into disconnected sub-graphs representing dense clusters separated by low-density space.
