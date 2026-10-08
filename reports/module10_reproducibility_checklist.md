# Module 10: Reproducibility & Environment Audit Checklist

## Executive Summary

This checklist documents the reproducibility controls, environment logging, hyperparameter tracking, and deterministic execution settings for the NDA multi-label clause classification pipeline.

---

## Reproducibility Audit Matrix

| Parameter / Control | Supported in Code? | Logged in Output JSON? | Configured Value / Status |
| :--- | :-: | :-: | :--- |
| **Random Seed** | **YES** | **YES** | `seed = 42` (PyTorch, NumPy, Python random) |
| **Model Identifier** | **YES** | **YES** | `saibo/legal-roberta-base`, `nlpaueb/legal-bert-base-uncased`, `microsoft/deberta-v3-base` |
| **Tokenizer Identifier** | **YES** | **YES** | Matches model identifier via `AutoTokenizer.from_pretrained` |
| **Learning Rate** | **YES** | **YES** | `2e-5` (AdamW optimizer) |
| **Batch Size** | **YES** | **YES** | `8` |
| **Epochs** | **YES** | **YES** | `5` (Full training) / `1` (Smoke test) |
| **Max Sequence Length** | **YES** | **YES** | `256` tokens |
| **Loss Function Configuration** | **YES** | **YES** | `bce`, `focal` ($\gamma=2.0, \alpha=0.25$), `class_weighted_bce` |
| **Classification Threshold** | **YES** | **YES** | Default `0.5` / Val-tuned $\tau^*$ |
| **Training Label Mode** | **YES** | **YES** | `all_valid` / `high_confidence` |
| **Train Document IDs** | **YES** | **YES** | **TBD — requires final benchmark/data inspection.** |
| **Validation Document IDs** | **YES** | **YES** | **TBD — requires final benchmark/data inspection.** |
| **Test Document IDs** | **YES** | **YES** | **TBD — requires final benchmark/data inspection.** |
| **Python Version** | **YES** | **YES** | Auto-logged via `sys.version` |
| **PyTorch Version** | **YES** | **YES** | Auto-logged via `torch.__version__` |
| **Transformers Version** | **YES** | **YES** | Auto-logged via `transformers.__version__` |
| **Hardware / OS Environment** | **YES** | **YES** | Auto-logged via `platform.platform()` and `torch.cuda` state |

---

## Detailed Reproducibility Controls

### 1. Seed Initialization Function (`scripts/train_multilabel_classifier.py`)
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 2. Output Log Schema Traceability
Every execution auto-generates a structured JSON log in `reports/experiments/` containing:
- Exact ISO 8601 execution timestamp
- Full hyperparameter dictionary
- Environment versions (Python, PyTorch, Transformers, CUDA device)
- Exact list of train, validation, and test document IDs
- Evaluation metrics dictionary

---

## Unfinalized Items Notice

> [!NOTE]
> The document ID lists (`train_doc_ids`, `val_doc_ids`, `test_doc_ids`) are currently marked as **TBD — requires final benchmark/data inspection.** because final benchmark construction is blocked awaiting dataset expansion.
