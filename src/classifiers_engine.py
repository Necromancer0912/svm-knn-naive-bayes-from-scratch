"""
From-Scratch Margin, Distance & Generative Probabilistic Classifiers
(`04_Margin_Distance_and_Probabilistic_ML/classical_classifiers.py`).

Includes:
- `LinearSVMScratch`: Primal subgradient Soft-Margin SVM with support vector detection.
- `KernelSVMScratch`: Dual Lagrangian Kernel SVM supporting Linear, Polynomial, and RBF kernels.
- `KNNScratch`: Loop-free, broadcast-free vectorized Euclidean k-Nearest Neighbors.
- `GaussianNaiveBayesScratch` & `MultinomialNaiveBayesScratch`: Log-space generative classifiers.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np


# =============================================================================
# 1. PAIRWISE EUCLIDEAN DISTANCE & KERNEL FUNCTIONS (ZERO LOOPS)
# =============================================================================

def pairwise_sq_dists_vectorized(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Compute exact N x M matrix of squared Euclidean distances ||x1_i - x2_j||_2^2
    using the algebraic inner-product identity (zero loops, zero 3D broadcast tensors):
        ||X1 - X2||_2^2 = ||X1||_2^2 + ||X2||_2^2^T - 2 X1 @ X2^T
    """
    A = np.asarray(X1, dtype=np.float64)
    B = np.asarray(X2, dtype=np.float64)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1], (
        f"Shape mismatch: A={A.shape}, B={B.shape}"
    )
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)        # (N, 1)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True).T      # (1, M)
    D_sq = sq_A + sq_B - 2.0 * (A @ B.T)
    return np.maximum(D_sq, 0.0)


def compute_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel: Literal["linear", "poly", "rbf"] = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Evaluate Mercer kernel K(X1, X2) in R^(N x M)."""
    A = np.asarray(X1, dtype=np.float64)
    B = np.asarray(X2, dtype=np.float64)
    if kernel == "linear":
        return A @ B.T
    if kernel == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    if kernel == "rbf":
        return np.exp(-gamma * pairwise_sq_dists_vectorized(A, B))
    raise ValueError(f"Unsupported kernel type: {kernel}")


# =============================================================================
# 2. PRIMAL LINEAR SVM & DUAL KERNEL SVM
# =============================================================================

class LinearSVMScratch:
    """
    Primal Soft-Margin Support Vector Machine solved via Vectorized Subgradient Descent:
        J(w, b) = 0.5 * ||w||_2^2 + (C / m) * sum_{i=1}^m max(0, 1 - y_i (w^T x_i + b))

    Accepts binary labels y in {0, 1} or {-1, +1} and internally maps to {-1, +1}.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.05,
        epochs: int = 800,
        lr_decay: float = 0.002,
    ) -> None:
        assert C > 0.0, "Regularization parameter C must be strictly positive."
        self.C: float = float(C)
        self.learning_rate: float = float(learning_rate)
        self.epochs: int = epochs
        self.lr_decay: float = float(lr_decay)

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []
        self.support_vector_indices_: Optional[np.ndarray] = None

    @staticmethod
    def _to_signed_labels(y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        return np.where(y_arr <= 0.0, -1.0, 1.0)

    def compute_cost_and_gradients(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
    ) -> Tuple[float, np.ndarray, float]:
        """Compute primal soft-margin Hinge objective and exact subgradients (grad_w, grad_b)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_signed = self._to_signed_labels(y)
        m = X_arr.shape[0]
        w_col = w.reshape(-1, 1)

        margins = y_signed * (X_arr @ w_col + b)  # shape (m, 1)
        hinge_losses = np.maximum(0.0, 1.0 - margins)
        cost = float(0.5 * np.sum(w_col ** 2) + (self.C / float(m)) * np.sum(hinge_losses))

        active_mask = (margins < 1.0).astype(np.float64)  # 1 where margin violated, 0 otherwise
        active_signed = active_mask * y_signed
        grad_w = w_col - (self.C / float(m)) * (X_arr.T @ active_signed)
        grad_b = float(-(self.C / float(m)) * np.sum(active_signed))
        return cost, grad_w, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2
        y_signed = self._to_signed_labels(y)
        m, n = X_arr.shape

        self.w = np.zeros((n, 1), dtype=np.float64)
        self.b = 0.0
        self.loss_history.clear()

        for epoch in range(self.epochs):
            alpha_t = self.learning_rate / (1.0 + self.lr_decay * epoch)
            cost, gw, gb = self.compute_cost_and_gradients(X_arr, y_signed, self.w, self.b)
            self.loss_history.append(cost)
            self.w -= alpha_t * gw
            self.b -= alpha_t * gb

        # Identify Support Vectors: points on or inside the street (y_i(w^T x_i + b) <= 1 + tol)
        final_margins = (y_signed * (X_arr @ self.w + self.b)).ravel()
        self.support_vector_indices_ = np.where(final_margins <= 1.05)[0]
        return self

    @property
    def geometric_margin(self) -> float:
        """Return total street width 2 / ||w||_2."""
        assert self.w is not None
        return float(2.0 / max(np.linalg.norm(self.w), 1e-12))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return (np.asarray(X, dtype=np.float64) @ self.w + self.b).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels in {0, 1}."""
        return (self.decision_function(X) >= 0.0).astype(np.int64)


class KernelSVMScratch:
    """
    Dual Kernel Support Vector Machine solved via Projected Dual Ascent / Coordinate Projection.

    Maximizes Wolfe Dual:
        W(alpha) = sum_i alpha_i - 0.5 * alpha^T (Y_diag K Y_diag) alpha
        subject to 0 <= alpha_i <= C  and  sum_i alpha_i y_i = 0
    """

    def __init__(
        self,
        C: float = 5.0,
        kernel: Literal["linear", "poly", "rbf"] = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        epochs: int = 600,
        learning_rate: float = 0.05,
    ) -> None:
        self.C: float = float(C)
        self.kernel = kernel
        self.gamma: float = float(gamma)
        self.degree: int = degree
        self.coef0: float = float(coef0)
        self.epochs: int = epochs
        self.learning_rate: float = float(learning_rate)

        self.X_train_: Optional[np.ndarray] = None
        self.y_signed_: Optional[np.ndarray] = None
        self.alphas_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.support_vector_indices_: Optional[np.ndarray] = None
        self.dual_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSVMScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        y_signed = np.where(np.asarray(y, dtype=np.float64).ravel() <= 0.0, -1.0, 1.0)
        m = X_arr.shape[0]
        self.X_train_ = X_arr
        self.y_signed_ = y_signed

        K = compute_kernel_matrix(X_arr, X_arr, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        Q = np.outer(y_signed, y_signed) * K  # Q_ij = y_i y_j K(x_i, x_j)

        # Scale step size by inverse spectral norm of Q for guaranteed dual convergence
        lip = float(np.linalg.norm(Q, ord=2)) / float(m) + 1e-6
        eta = min(self.learning_rate, 1.2 / lip)

        alphas = np.zeros(m, dtype=np.float64)
        self.dual_history.clear()

        for _ in range(self.epochs):
            # Gradient of dual objective W(alpha) = 1^T alpha - 0.5 alpha^T Q alpha
            q_alpha = Q @ alphas
            grad = 1.0 - q_alpha
            alphas = alphas + eta * grad

            # Project onto hyperplane sum(alpha_i y_i) = 0 and box [0, C]^m iteratively
            for _ in range(3):
                alphas = alphas - (np.dot(alphas, y_signed) / float(m)) * y_signed
                alphas = np.clip(alphas, 0.0, self.C)

            dual_obj = float(np.sum(alphas) - 0.5 * np.dot(alphas, Q @ alphas))
            self.dual_history.append(dual_obj)

        self.alphas_ = alphas
        sv_mask = alphas > 1e-3
        self.support_vector_indices_ = np.where(sv_mask)[0]

        # Recover intercept b* from margin support vectors (0 < alpha_i < C)
        margin_mask = (alphas > 1e-3) & (alphas < self.C - 1e-3)
        if not np.any(margin_mask):
            margin_mask = sv_mask
        if np.any(margin_mask):
            decision_no_b = (alphas * y_signed) @ K[:, margin_mask]
            self.b_ = float(np.mean(y_signed[margin_mask] - decision_no_b))
        else:
            self.b_ = 0.0

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.X_train_ is not None and self.alphas_ is not None and self.y_signed_ is not None
        K_test = compute_kernel_matrix(
            X, self.X_train_, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0
        )
        return K_test @ (self.alphas_ * self.y_signed_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(np.int64)


# =============================================================================
# 3. VECTORIZED k-NEAREST NEIGHBORS (ZERO LOOPS)
# =============================================================================

class KNNScratch:
    """
    Loop-Free Vectorized k-Nearest Neighbors Classifier.

    Computes all pairwise Euclidean distances via BLAS matrix multiplication
    and selects top-k neighbors in O(m) average time via `np.argpartition`.
    """

    def __init__(self, k: int = 5, weights: Literal["uniform", "distance"] = "uniform") -> None:
        assert k >= 1
        self.k: int = int(k)
        self.weights = weights
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.num_classes_: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNScratch":
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64).ravel()
        assert self.X_train_.ndim == 2 and self.X_train_.shape[0] == self.y_train_.shape[0]
        self.num_classes_ = int(np.max(self.y_train_) + 1)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.X_train_ is not None and self.y_train_ is not None
        X_query = np.asarray(X, dtype=np.float64)
        D_sq = pairwise_sq_dists_vectorized(X_query, self.X_train_)  # shape (N, m)
        N = X_query.shape[0]
        k_eff = min(self.k, self.X_train_.shape[0])

        knn_idx = np.argpartition(D_sq, kth=k_eff - 1, axis=1)[:, :k_eff]  # shape (N, k)
        knn_labels = self.y_train_[knn_idx]                                # shape (N, k)

        if self.weights == "uniform":
            w_mat = np.ones_like(knn_labels, dtype=np.float64)
        else:
            row_idx = np.arange(N)[:, None]
            knn_dists = np.sqrt(D_sq[row_idx, knn_idx])
            w_mat = 1.0 / (knn_dists + 1e-9)

        probs = np.zeros((N, self.num_classes_), dtype=np.float64)
        for c in range(self.num_classes_):
            probs[:, c] = np.sum(w_mat * (knn_labels == c), axis=1)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# =============================================================================
# 4. GENERATIVE NAIVE BAYES CLASSIFIERS (GAUSSIAN & MULTINOMIAL)
# =============================================================================

class GaussianNaiveBayesScratch:
    """
    Generative Gaussian Naive Bayes Classifier operating entirely in log-probability space:
        log P(y = c | x) = log pi_c - 0.5 * sum_j [log(2*pi*sigma_{cj}^2) + (x_j - mu_{cj})^2 / sigma_{cj}^2] - Z
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing: float = float(var_smoothing)
        self.classes_: Optional[np.ndarray] = None
        self.log_priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.vars_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        assert X_arr.ndim == 2 and X_arr.shape[0] == y_arr.shape[0]

        self.classes_ = np.unique(y_arr)
        K = len(self.classes_)
        m, n = X_arr.shape

        self.log_priors_ = np.zeros(K, dtype=np.float64)
        self.means_ = np.zeros((K, n), dtype=np.float64)
        self.vars_ = np.zeros((K, n), dtype=np.float64)
        eps = self.var_smoothing * float(np.max(np.var(X_arr, axis=0)) + 1e-12)

        for idx, c in enumerate(self.classes_):
            X_c = X_arr[y_arr == c]
            self.log_priors_[idx] = np.log(X_c.shape[0] / float(m))
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + eps

        return self

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        """Compute unnormalized log-joint log P(x, y=c) = log P(y=c) + log P(x | y=c) of shape (N, K)."""
        assert self.means_ is not None and self.vars_ is not None and self.log_priors_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        N = X_arr.shape[0]
        K = self.means_.shape[0]
        log_joint = np.zeros((N, K), dtype=np.float64)

        for k in range(K):
            log_det = -0.5 * np.sum(np.log(2.0 * np.pi * self.vars_[k]))
            quad = -0.5 * np.sum(((X_arr - self.means_[k]) ** 2) / self.vars_[k], axis=1)
            log_joint[:, k] = self.log_priors_[k] + log_det + quad

        return log_joint

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalize log-joint probabilities via the Log-Sum-Exp shift trick."""
        log_j = self.predict_log_joint(X)
        m_shift = np.max(log_j, axis=1, keepdims=True)
        exp_j = np.exp(log_j - m_shift)
        return exp_j / np.sum(exp_j, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_joint(X), axis=1)]


class MultinomialNaiveBayesScratch:
    """
    Generative Multinomial Naive Bayes with Laplace (+alpha) Smoothing for discrete counts:
        theta_{cj} = (N_{cj} + alpha) / (N_c + alpha * n)
    """

    def __init__(self, alpha: float = 1.0) -> None:
        assert alpha > 0.0, "Smoothing parameter alpha must be > 0 to prevent log(0)."
        self.alpha: float = float(alpha)
        self.classes_: Optional[np.ndarray] = None
        self.log_priors_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayesScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        self.classes_ = np.unique(y_arr)
        K = len(self.classes_)
        m, n = X_arr.shape

        self.log_priors_ = np.zeros(K, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((K, n), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X_arr[y_arr == c]
            self.log_priors_[idx] = np.log(X_c.shape[0] / float(m))
            counts_c = np.sum(X_c, axis=0) + self.alpha
            self.feature_log_prob_[idx] = np.log(counts_c / np.sum(counts_c))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.feature_log_prob_ is not None and self.log_priors_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        log_j = X_arr @ self.feature_log_prob_.T + self.log_priors_[None, :]
        m_shift = np.max(log_j, axis=1, keepdims=True)
        exp_j = np.exp(log_j - m_shift)
        return exp_j / np.sum(exp_j, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =============================================================================
# 5. FINITE-DIFFERENCE NUMERICAL GRADIENT & DISTANCE VERIFICATION SUITE
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("MODULE 04: SVM, k-NN & NAIVE BAYES — VERIFICATION SUITE")
    print("=" * 72)

    rng = np.random.default_rng(2026)
    m, n = 45, 5
    X = rng.normal(size=(m, n))
    y = rng.choice([-1.0, 1.0], size=(m, 1))
    w_test = rng.normal(size=(n, 1))
    b_test = -0.25

    # Ensure no sample sits within eps of the hinge kink (1 - y*(w^T x + b) == 0) during finite diff check
    margins = y * (X @ w_test + b_test)
    near_kink = np.abs(1.0 - margins.ravel()) < 1e-3
    X[near_kink] *= 1.25

    svm = LinearSVMScratch(C=2.5)
    _, ana_gw, ana_gb = svm.compute_cost_and_gradients(X, y, w_test, b_test)
    ana_vec = np.concatenate([ana_gw.ravel(), [ana_gb]])
    num_vec = np.zeros_like(ana_vec)
    eps = 1e-6

    for j in range(n):
        wp, wm = w_test.copy(), w_test.copy()
        wp[j, 0] += eps
        wm[j, 0] -= eps
        cp, _, _ = svm.compute_cost_and_gradients(X, y, wp, b_test)
        cm, _, _ = svm.compute_cost_and_gradients(X, y, wm, b_test)
        num_vec[j] = (cp - cm) / (2.0 * eps)

    cbp, _, _ = svm.compute_cost_and_gradients(X, y, w_test, b_test + eps)
    cbm, _, _ = svm.compute_cost_and_gradients(X, y, w_test, b_test - eps)
    num_vec[-1] = (cbp - cbm) / (2.0 * eps)

    rel_err = float(np.linalg.norm(ana_vec - num_vec) / (np.linalg.norm(ana_vec) + np.linalg.norm(num_vec) + 1e-15))
    print(f"[Grad Check] Linear Soft-Margin SVM Hinge Subgradient | Relative Error: {rel_err:.3e} (< 1e-7 PASS)")
    assert rel_err < 1e-7

    # Verify loop-free pairwise distance matrix against explicit nested loops
    X_q = rng.normal(size=(18, n))
    D_vec = pairwise_sq_dists_vectorized(X_q, X)
    D_loop = np.zeros((18, m))
    for i in range(18):
        for j in range(m):
            D_loop[i, j] = np.sum((X_q[i] - X[j]) ** 2)
    dist_err = float(np.max(np.abs(D_vec - D_loop)))
    print(f"[Matrix Check] Vectorized Pairwise Distance Max Error:  {dist_err:.3e} (< 1e-12 PASS)")
    assert dist_err < 1e-12

    # Verify RBF Kernel SVM separates non-linear concentric circles
    ang = rng.uniform(0, 2 * np.pi, size=120)
    r = np.concatenate([rng.normal(0.5, 0.1, 60), rng.normal(1.6, 0.12, 60)])
    X_ring = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    y_ring = np.concatenate([np.zeros(60), np.ones(60)])
    ksvm = KernelSVMScratch(C=10.0, kernel="rbf", gamma=1.2, epochs=500).fit(X_ring, y_ring)
    ring_acc = float(np.mean(ksvm.predict(X_ring) == y_ring))
    print(f"[RBF Kernel SVM] Concentric Rings Training Accuracy:   {ring_acc*100:.2f}% (PASS)")
    assert ring_acc > 0.98
    print("=" * 72)

