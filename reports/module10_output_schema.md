# Module 10: Experiment Output Directory & Result JSON Schema

## Executive Summary

This document specifies the standard directory structure and output JSON/CSV schema for all classifier training and evaluation runs.

---

## 1. Directory Structure

The following directories are established in `reports/experiments/`:

- `reports/experiments/results/`: Stores structured JSON summary reports for each completed experiment.
- `reports/experiments/checkpoints/`: Model weight checkpoints (e.g. `pytorch_model.bin`, `config.json`, `tokenizer.json`).
- `reports/experiments/logs/`: Detailed per-step training logs and loss curves.
- `reports/experiments/figures/`: Confusion matrices, per-category F1 bar charts, and PR curve plots.

---

## 2. Standard Experiment Result JSON Schema

Every executed experiment generates a JSON result file in `reports/experiments/results/{experiment_id}_result.json` adhering strictly to this schema:

```json
{
  "experiment_id": "EXP-01",
  "model": "saibo/legal-roberta-base",
  "loss": "bce",
  "label_mode": "all_valid",
  "seed": 42,
  "threshold": 0.5,
  "status": "COMPLETED",
  "disclaimer": "Training labels are LLM-generated pseudo-labels and have not yet been fully human-verified.",
  "timestamp": "2026-10-03T20:00:00.000000",
  "dataset_splits": {
    "train_docs": ["0859334b", "30647c0b", "0b59dfc4"],
    "val_docs": ["32668bcf", "35a40a59"],
    "test_docs": ["9a5cb310", "586c367e"],
    "train_clauses": 184,
    "val_clauses": 36,
    "test_clauses": 75
  },
  "hyperparameters": {
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 2e-05,
    "max_sequence_length": 256,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1
  },
  "metrics": {
    "macro_f1": 0.7452,
    "micro_f1": 0.8120,
    "weighted_f1": 0.7891,
    "hamming_loss": 0.0412,
    "mcc": 0.7104,
    "macro_precision": 0.7610,
    "macro_recall": 0.7315
  },
  "per_category_metrics": [
    {
      "category": "Definition of Confidential Information",
      "precision": 0.8500,
      "recall": 0.8200,
      "f1_score": 0.8346,
      "mcc": 0.8100,
      "actual_positives": 20,
      "predicted_positives": 19,
      "true_positives": 17,
      "false_positives": 2,
      "false_negatives": 3,
      "true_negatives": 53
    }
  ],
  "performance_timing": {
    "training_time_sec": 45.2,
    "inference_time_sec": 1.4
  },
  "reproducibility_environment": {
    "python_version": "3.11.x",
    "pytorch_version": "2.x.x",
    "transformers_version": "4.x.x",
    "device": "cuda:0",
    "platform": "Windows-10-..."
  }
}
```

---

## 3. History CSV Aggregation Schema

In addition to individual JSON files, all experiments append a single row to `data/classification/experiment_results/experiment_history.csv` with the following columns:

`experiment_id`, `timestamp`, `model`, `loss`, `label_mode`, `seed`, `threshold`, `status`, `train_clauses`, `val_clauses`, `test_clauses`, `macro_f1`, `micro_f1`, `weighted_f1`, `hamming_loss`, `mcc`, `training_time_sec`, `inference_time_sec`

---

## 4. Operational Integrity Rule

> [!CAUTION]
> Fake or dummy result files MUST NOT be placed in `reports/experiments/results/`. Files in this directory must only originate from actual verified experiment runs.
