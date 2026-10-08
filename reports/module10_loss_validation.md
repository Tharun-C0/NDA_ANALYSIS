# Module 10: Loss Function & Mathematical Validation Report

## Executive Summary

This report documents the mathematical definitions, implementation details, and validation checks for all three loss functions supported in the NDA multi-label clause classification pipeline.

---

## 1. Supported Loss Functions & Mathematical Formulations

### 1. Standard Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
Used for baseline multi-label learning without class re-weighting or focal modulation.

$$\mathcal{L}_{\text{BCE}}(\mathbf{z}, \mathbf{y}) = -\frac{1}{C} \sum_{c=1}^{C} \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$

where:
- $\mathbf{z} \in \mathbb{R}^C$ are the unnormalized model logits for $C=14$ categories.
- $\mathbf{y} \in \{0, 1\}^C$ is the ground-truth binary label vector.
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function.

---

### 2. Multi-Label Focal Loss (`FocalLoss`)
Modulates the standard BCE loss to down-weight easy examples (high probability correct predictions) and focus training on hard minority examples.

$$\mathcal{L}_{\text{Focal}}(\mathbf{z}, \mathbf{y}) = -\frac{1}{C} \sum_{c=1}^{C} \alpha_c (1 - p_{t,c})^\gamma \log(p_{t,c})$$

where:
- $p_{t,c} = \sigma(z_c)$ if $y_c = 1$, and $p_{t,c} = 1 - \sigma(z_c)$ if $y_c = 0$.
- Focusing hyperparameter $\gamma = 2.0$ reduces loss contribution from easy examples ($p_{t,c} \to 1$).
- Weighting factor $\alpha_c = 0.25$ if $y_c = 1$, and $\alpha_c = 0.75$ ($1 - \alpha$) if $y_c = 0$.

#### PyTorch Implementation Snippet (`scripts/train_multilabel_classifier.py`):
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_factor * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()
```

---

### 3. Class-Weighted BCE (`Class-Weighted BCE`)
Addresses positive class imbalance across categories by assigning positive weights $w_c > 1$ to minority categories.

$$\mathcal{L}_{\text{WeightedBCE}}(\mathbf{z}, \mathbf{y}) = -\frac{1}{C} \sum_{c=1}^{C} \left[ w_c y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$

where positive class weight $w_c$ is calculated as:

$$w_c = \frac{N_{\text{neg}, c}}{N_{\text{pos}, c}} = \frac{N_{\text{train}} - N_{\text{pos}, c}}{\max(N_{\text{pos}, c}, 1)}$$

---

## 2. Zero-Leakage Safeguard Verification

- **Class Weight Calculation Boundary**:
  Weights $w_c$ are calculated **EXCLUSIVELY from `df_train`** inside `compute_pos_weights_from_train_only(df_train)`.
- **Validation & Test Isolation**: Zero label counts or statistics from validation (`df_val`) or test (`df_test`) are passed to `compute_pos_weights_from_train_only`.
- **Numerical Stability**: Positive count is clamped with $\max(N_{\text{pos}, c}, 1.0)$ to prevent division by zero for rare categories.

---

## 3. Loss Function Validation Matrix

| Loss ID | Loss Function Name | Parameters | Weight Source | Leakage Risk | Unit Test Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bce` | `BCEWithLogitsLoss` | Standard | None | **ZERO** | **PASS** |
| `focal` | `Multi-Label Focal Loss` | $\gamma=2.0, \alpha=0.25$ | Constant Hyperparameters | **ZERO** | **PASS** |
| `class_weighted_bce` | `Class-Weighted BCE` | Dynamic $w_c$ vector | Train Split Only (`df_train`) | **ZERO** | **PASS** |

---

## 4. Verification Verdict

All loss functions are correctly specified, mathematically rigorous, free of data leakage, and fully verified by unit tests in `scripts/train_multilabel_classifier.py`.
