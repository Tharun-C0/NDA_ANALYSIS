"""
scripts/train_exp12_model_comparison.py

EXP-12: Model Comparison Experiment (Legal-RoBERTa vs Legal-BERT vs DeBERTa-v3)
Dataset: Combined Training (344 genuine + 5,000 synthetic = 5,344 clauses)
Loss: Class-Weighted BCE (computed exclusively from combined 5,344 training dataset)
Validation Data: Existing benchmark validation set (137 clauses, untouched)
Test Data: Existing benchmark test set (236 clauses, untouched)
Identical Hyperparameters across Models:
  - Max Epochs: 10
  - Early Stopping Patience: 2 (monitors Validation Macro F1)
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Max Sequence Length: 256
  - Seed: 42
  - Device: GPU/CUDA with Mixed Precision (bfloat16 / float16)
"""

import os
import sys
import json
import time
import random
import platform
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    matthews_corrcoef
)
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)

ROOT = Path(__file__).resolve().parents[1]

APPROVED_CATEGORIES = [
    "Party Identification",
    "Purpose",
    "NDA Type",
    "Definition of Confidential Information",
    "Confidentiality Obligations",
    "Authorized Disclosure",
    "Non-Confidential Information",
    "Liability for Damages",
    "Competition Rights",
    "Term and Termination",
    "Intellectual Property",
    "Employees",
    "Governing Law and Jurisdiction",
    "Additional Information",
]

CAT_TO_LABEL = {c: f"label_{c.lower().replace(' ', '_')}" for c in APPROVED_CATEGORIES}
LABEL_COLS = list(CAT_TO_LABEL.values())

MODELS = [
    {
        "id": "legal_roberta",
        "name": "Legal-RoBERTa",
        "huggingface_id": "saibo/legal-roberta-base"
    },
    {
        "id": "legal_bert",
        "name": "Legal-BERT",
        "huggingface_id": "nlpaueb/legal-bert-base-uncased"
    },
    {
        "id": "deberta_v3",
        "name": "DeBERTa-v3",
        "huggingface_id": "microsoft/deberta-v3-base"
    }
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_gpu_and_cuda() -> Dict[str, Any]:
    print("\n=======================================================")
    print("      STEP 1: GPU & CUDA ENVIRONMENT VERIFICATION     ")
    print("=======================================================")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA availability: {cuda_avail}")
    print(f"PyTorch version:   {torch.__version__}")
    
    if not cuda_avail:
        print("\n[CRITICAL ERROR] CUDA is NOT available! Halting execution as per experiment rules.")
        sys.exit(1)
        
    gpu_name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda
    gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    bf16_support = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else True
    
    print(f"GPU name:          {gpu_name}")
    print(f"CUDA version:      {cuda_ver}")
    print(f"GPU memory:        {gpu_vram} GB")
    print(f"BF16 support:      {bf16_support}")
    print("[CHECK PASS] GPU environment verified successfully.\n")
    
    return {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "cuda_version": cuda_ver,
        "pytorch_version": torch.__version__,
        "gpu_memory_gb": gpu_vram,
        "bf16_support": bf16_support
    }


def perform_data_validation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=======================================================")
    print("        STEP 2: DATASET VALIDATION & PRE-FLIGHT       ")
    print("=======================================================")
    
    synthetic_path = ROOT / "nda_clause_dataset_5000.csv"
    train_path = ROOT / "data/classification/module16_benchmark/train.csv"
    val_path = ROOT / "data/classification/module16_benchmark/validation.csv"
    test_path = ROOT / "data/classification/module16_benchmark/test.csv"

    df_synth_raw = pd.read_csv(synthetic_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    df_synth = df_synth_raw.copy()
    for cat, label_col in CAT_TO_LABEL.items():
        if cat in df_synth.columns:
            df_synth[label_col] = df_synth[cat]

    assert len(df_synth) == 5000, f"Expected 5,000 rows in synthetic dataset, got {len(df_synth)}"
    missing_synth_labels = [c for c in LABEL_COLS if c not in df_synth.columns]
    assert len(missing_synth_labels) == 0, f"Missing label columns in synthetic: {missing_synth_labels}"
    invalid_synth_vals = set(df_synth[LABEL_COLS].values.flatten()) - {0, 1}
    assert len(invalid_synth_vals) == 0, f"Invalid label values found in synthetic: {invalid_synth_vals}"
    empty_synth_text = df_synth[df_synth["clause_text"].isna() | (df_synth["clause_text"].str.strip() == "")]
    assert len(empty_synth_text) == 0, f"Found {len(empty_synth_text)} empty clause text rows in synthetic!"
    dup_cids = df_synth[df_synth.duplicated(subset=["clause_id"])]
    assert len(dup_cids) == 0, f"Found {len(dup_cids)} duplicate clause IDs in synthetic dataset!"
    dup_synth_text = df_synth[df_synth.duplicated(subset=["clause_text"])]
    assert len(dup_synth_text) == 0, f"Found {len(dup_synth_text)} duplicate clause text rows in synthetic dataset!"

    benchmark_label_cols = [c for c in df_train.columns if c.startswith("label_") and c != "label_source"]
    assert sorted(LABEL_COLS) == sorted(benchmark_label_cols), "Label columns mismatch with benchmark"
    assert list(df_train.columns) == list(df_val.columns) == list(df_test.columns), "Training/validation/test schemas mismatch"

    val_texts = set(df_val["clause_text"].str.strip())
    test_texts = set(df_test["clause_text"].str.strip())
    synth_texts = set(df_synth["clause_text"].str.strip())

    synth_in_val = synth_texts.intersection(val_texts)
    synth_in_test = synth_texts.intersection(test_texts)
    assert len(synth_in_val) == 0, f"Found {len(synth_in_val)} synthetic clauses in validation set!"
    assert len(synth_in_test) == 0, f"Found {len(synth_in_test)} synthetic clauses in test set!"

    synth_formatted = df_synth.copy()
    if "document_id" not in synth_formatted.columns:
        synth_formatted["document_id"] = "SYNTHETIC_DOC"
    
    for col in df_train.columns:
        if col not in synth_formatted.columns:
            if col.startswith("label_"):
                synth_formatted[col] = 0
            else:
                synth_formatted[col] = "SYNTHETIC"

    synth_formatted = synth_formatted[df_train.columns]
    df_combined_train = pd.concat([df_train, synth_formatted], ignore_index=True)

    print("\n--- DATASET ROW COUNTS ---")
    print(f"Training = {len(df_combined_train)}")
    print(f"Validation = {len(df_val)}")
    print(f"Test = {len(df_test)}")
    print("=======================================================\n")

    return df_train, df_synth, df_combined_train, df_val, df_test


class NDAClauseDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def calculate_multilabel_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, List[float]]:
    num_classes = y_true.shape[1]
    mccs = []
    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        if len(np.unique(yt)) == 1 and len(np.unique(yp)) == 1:
            mcc = 1.0 if yt[0] == yp[0] else 0.0
        else:
            try:
                mcc = float(matthews_corrcoef(yt, yp))
            except Exception:
                mcc = 0.0
        mccs.append(mcc)
    return float(np.mean(mccs)), mccs


def get_minority_category_indices(df_train_benchmark: pd.DataFrame) -> List[int]:
    counts = df_train_benchmark[LABEL_COLS].sum().values
    med = float(np.median(counts))
    minority_indices = [i for i, cnt in enumerate(counts) if cnt < max(15, med)]
    if not minority_indices:
        minority_indices = list(np.argsort(counts)[:5])
    return minority_indices


def compute_all_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    minority_indices: List[int]
) -> Dict[str, Any]:
    y_pred = (y_probs >= threshold).astype(int)
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred))
    macro_mcc, per_cat_mccs = calculate_multilabel_mcc(y_true, y_pred)

    per_category = []
    per_cat_f1s = []
    for c, cat in enumerate(APPROVED_CATEGORIES):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))
        mcc = per_cat_mccs[c]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        support = int(np.sum(yt == 1))
        
        per_cat_f1s.append(f1)
        per_category.append({
            "category": cat,
            "label_col": LABEL_COLS[c],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

    minority_f1s = [per_cat_f1s[i] for i in minority_indices]
    minority_macro_f1 = float(np.mean(minority_f1s)) if minority_f1s else macro_f1

    return {
        "threshold": round(threshold, 2),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "minority_macro_f1": round(minority_macro_f1, 4),
        "hamming_loss": round(h_loss, 4),
        "mcc": round(macro_mcc, 4),
        "per_category": per_category
    }


def compute_class_weights_from_combined_train(df_combined_train: pd.DataFrame, device: torch.device) -> torch.Tensor:
    train_labels = df_combined_train[LABEL_COLS].values
    pos_counts = torch.tensor(train_labels.sum(axis=0), dtype=torch.float32)
    total_counts = float(len(df_combined_train))
    neg_counts = total_counts - pos_counts
    pos_weights = (neg_counts / torch.clamp(pos_counts, min=1.0)).to(device)
    pos_weights = torch.clamp(pos_weights, max=25.0)
    return pos_weights


def is_model_completed(model_exp_dir: Path) -> bool:
    required_files = [
        "training_config.json",
        "training_log.csv",
        "threshold_sweep_results.csv",
        "test_metrics.json",
        "per_class_metrics.csv",
        "predictions.csv"
    ]
    if not model_exp_dir.exists():
        return False
    if not all((model_exp_dir / f).exists() for f in required_files):
        return False

    checkpoint_dir = model_exp_dir / "best_checkpoint"
    has_checkpoint = checkpoint_dir.exists() and (
        len(list(checkpoint_dir.glob("*.safetensors"))) > 0 or len(list(checkpoint_dir.glob("*.bin"))) > 0
    )
    if not has_checkpoint:
        return False

    try:
        # Check test_metrics.json for valid non-zero metrics
        with open(model_exp_dir / "test_metrics.json", "r", encoding="utf-8") as f:
            test_json = json.load(f)
        test_res = test_json.get("test_metrics", {})
        macro_f1 = test_res.get("macro_f1", 0.0)
        micro_f1 = test_res.get("micro_f1", 0.0)

        if macro_f1 <= 0.0 or micro_f1 <= 0.0:
            return False

        # Check training_log.csv
        df_log = pd.read_csv(model_exp_dir / "training_log.csv")
        if len(df_log) < 1 or (df_log["macro_f1"] == 0.0).all():
            return False

        # Check predictions.csv
        df_preds = pd.read_csv(model_exp_dir / "predictions.csv")
        if len(df_preds) != 236:
            return False

        # Check per_class_metrics.csv
        df_per_class = pd.read_csv(model_exp_dir / "per_class_metrics.csv")
        if len(df_per_class) != 14:
            return False

        # Check threshold_sweep_results.csv
        df_sweep = pd.read_csv(model_exp_dir / "threshold_sweep_results.csv")
        if len(df_sweep) < 1 or (df_sweep["macro_f1"] == 0.0).all():
            return False

        return True
    except Exception:
        return False


def load_completed_model_results(model_meta: Dict[str, str], model_exp_dir: Path) -> Dict[str, Any]:
    model_id = model_meta["id"]
    model_display_name = model_meta["name"]
    
    test_metrics_file = model_exp_dir / "test_metrics.json"
    per_class_file = model_exp_dir / "per_class_metrics.csv"
    
    with open(test_metrics_file, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
        
    df_per_class = pd.read_csv(per_class_file)
    cat_dict = dict(zip(df_per_class["category"], df_per_class["f1_score"]))
    
    test_res = meta_json["test_metrics"]
    best_epoch = meta_json.get("best_epoch", 0)
    best_val_macro_f1 = meta_json.get("best_val_macro_f1", 0.0)
    selected_threshold = meta_json.get("selected_threshold", 0.50)
    
    return {
        "Model": model_display_name,
        "model_id": model_id,
        "Best Epoch": best_epoch,
        "Best Val Macro F1": best_val_macro_f1,
        "Threshold": selected_threshold,
        "Test Macro F1": test_res["macro_f1"],
        "Test Micro F1": test_res["micro_f1"],
        "Test Weighted F1": test_res["weighted_f1"],
        "Minority F1": test_res["minority_macro_f1"],
        "Hamming Loss": test_res["hamming_loss"],
        "MCC": test_res["mcc"],
        "per_category": cat_dict,
        "skipped": True
    }


def train_single_model(
    model_meta: Dict[str, str],
    df_train_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_combined_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    gpu_meta: Dict[str, Any],
    exp_dir: Path
) -> Dict[str, Any]:
    model_id = model_meta["id"]
    model_display_name = model_meta["name"]
    hf_model_id = model_meta["huggingface_id"]
    
    model_exp_dir = exp_dir / model_id
    
    # Requirement C: Check if completed artifacts already exist
    if is_model_completed(model_exp_dir):
        print(f"\n======================================================================")
        print(f"  [SKIPPING] {model_display_name} ({hf_model_id})")
        print(f"  Existing completed artifacts detected in {model_exp_dir}. Preserving existing results!")
        print(f"======================================================================")
        return load_completed_model_results(model_meta, model_exp_dir)
        
    best_checkpoint_dir = model_exp_dir / "best_checkpoint"
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(42)
    device = torch.device("cuda")
    
    # Requirement A: Use bfloat16 mixed precision on RTX GPUs (avoids FP16 unscale issues in DeBERTa-v3)
    amp_dtype = torch.bfloat16 if gpu_meta.get("bf16_support", True) else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    
    max_epochs = 10
    patience = 2
    batch_size = 16
    learning_rate = 2e-5
    max_length = 256
    
    print(f"\n======================================================================")
    print(f"  TRAINING MODEL {model_id.upper()}: {model_display_name} ({hf_model_id})")
    print(f"  Mixed Precision Dtype: {amp_dtype} | Loss Scaler Enabled: {use_scaler}")
    print(f"======================================================================")
    
    pos_weights = compute_class_weights_from_combined_train(df_combined_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    config = AutoConfig.from_pretrained(
        hf_model_id,
        num_labels=14,
        problem_type="multi_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id, config=config, dtype=torch.float32)
    model.to(device)
    
    train_dataset = NDAClauseDataset(df_combined_train["clause_text"].tolist(), df_combined_train[LABEL_COLS].values, tokenizer, max_length)
    val_dataset = NDAClauseDataset(df_val["clause_text"].tolist(), df_val[LABEL_COLS].values, tokenizer, max_length)
    test_dataset = NDAClauseDataset(df_test["clause_text"].tolist(), df_test[LABEL_COLS].values, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    
    minority_indices = get_minority_category_indices(df_train_orig)
    
    train_config = {
        "experiment_id": "EXP-12",
        "model_id": model_id,
        "model_name": model_display_name,
        "huggingface_id": hf_model_id,
        "loss_function": "Class-Weighted BCE",
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_sequence_length": max_length,
        "seed": 42,
        "amp_dtype": str(amp_dtype),
        "gpu_environment": gpu_meta,
        "dataset_rows": {
            "benchmark_train": len(df_train_orig),
            "synthetic_train": len(df_synth),
            "combined_train": len(df_combined_train),
            "validation": len(df_val),
            "test": len(df_test)
        }
    }
    (model_exp_dir / "training_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    
    training_history = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    stopped_epoch = max_epochs
    
    for epoch in range(1, max_epochs + 1):
        start_ep = time.time()
        model.train()
        running_train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
                
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(**kwargs)
                loss = criterion(outputs.logits.float(), labels)
                
            if use_scaler:
                scaler.scale(loss).backward()
                
                # Requirement A & B: Safe unscale and scheduler step
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                    
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                # Only step scheduler if optimizer step was not skipped by GradScaler
                if scale_before <= scale_after:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validation Evaluation
        model.eval()
        running_val_loss = 0.0
        val_logits_list = []
        val_trues_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
                    
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(**kwargs)
                    loss = criterion(outputs.logits.float(), labels)
                    
                running_val_loss += loss.item()
                val_logits_list.append(outputs.logits.float().cpu())
                val_trues_list.append(labels.float().cpu().numpy())
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_logits = torch.cat(val_logits_list, dim=0).float().cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_trues = np.vstack(val_trues_list)
        
        val_metrics = compute_all_metrics(val_trues, val_probs, threshold=0.60, minority_indices=minority_indices)
        ep_duration = round(time.time() - start_ep, 2)
        
        val_macro_f1 = val_metrics["macro_f1"]
        is_best = val_macro_f1 > best_val_macro_f1
        
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)
        else:
            patience_counter += 1
            
        ep_row = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "macro_f1": val_metrics["macro_f1"],
            "micro_f1": val_metrics["micro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "minority_f1": val_metrics["minority_macro_f1"],
            "hamming_loss": val_metrics["hamming_loss"],
            "mcc": val_metrics["mcc"],
            "duration_sec": ep_duration,
            "is_best": is_best,
            "patience_counter": patience_counter
        }
        training_history.append(ep_row)
        
        print(
            f"[{model_id.upper()}] Ep {epoch:02d}/{max_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | Micro F1: {val_metrics['micro_f1']:.4f} | "
            f"Weighted F1: {val_metrics['weighted_f1']:.4f} | Minority F1: {val_metrics['minority_macro_f1']:.4f} | "
            f"Hamming Loss: {val_metrics['hamming_loss']:.4f} | MCC: {val_metrics['mcc']:.4f} "
            f"{'[BEST]' if is_best else f'[Patience {patience_counter}/{patience}]'}"
        )
        
        if patience_counter >= patience:
            stopped_epoch = epoch
            print(f"[{model_id.upper()}] Early stopping triggered at epoch {epoch}. Restoring best checkpoint from Epoch {best_epoch} (Val Macro F1 = {best_val_macro_f1:.4f}).")
            break

    df_train_log = pd.DataFrame(training_history)
    df_train_log.to_csv(model_exp_dir / "training_log.csv", index=False)
    
    # Load BEST model checkpoint for evaluation & threshold sweep
    best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir, dtype=torch.float32)
    best_model.to(device)
    best_model.eval()
    
    # Obtain validation probabilities
    val_logits_list = []
    val_trues_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = best_model(**kwargs)
            val_logits_list.append(outputs.logits.float().cpu())
            val_trues_list.append(labels.float().cpu().numpy())
            
    val_logits = torch.cat(val_logits_list, dim=0).float().cpu().numpy()
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_trues = np.vstack(val_trues_list)
    
    sweep_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_results = []
    best_sweep_thresh = 0.60
    best_sweep_val_macro_f1 = -1.0
    
    for t in sweep_thresholds:
        t_res = compute_all_metrics(val_trues, val_probs, threshold=t, minority_indices=minority_indices)
        sweep_row = {
            "threshold": t,
            "macro_f1": t_res["macro_f1"],
            "minority_f1": t_res["minority_macro_f1"],
            "mcc": t_res["mcc"],
            "micro_f1": t_res["micro_f1"],
            "weighted_f1": t_res["weighted_f1"],
            "hamming_loss": t_res["hamming_loss"]
        }
        sweep_results.append(sweep_row)
        
        if t_res["macro_f1"] > best_sweep_val_macro_f1:
            best_sweep_val_macro_f1 = t_res["macro_f1"]
            best_sweep_thresh = t
            
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(model_exp_dir / "threshold_sweep_results.csv", index=False)
    
    print(f"[{model_id.upper()}] Selected Threshold via Validation Macro F1 Sweep: {best_sweep_thresh:.2f} (Val Macro F1 = {best_sweep_val_macro_f1:.4f})")
    
    # Evaluate ONCE on untouched frozen test set
    test_logits_list = []
    test_trues_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = best_model(**kwargs)
            test_logits_list.append(outputs.logits.float().cpu())
            test_trues_list.append(labels.float().cpu().numpy())
            
    test_logits = torch.cat(test_logits_list, dim=0).float().cpu().numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_trues = np.vstack(test_trues_list)
    
    test_metrics_opt = compute_all_metrics(test_trues, test_probs, threshold=best_sweep_thresh, minority_indices=minority_indices)
    
    # Save test metrics JSON
    test_metrics_json = {
        "model_id": model_id,
        "model_name": model_display_name,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "selected_threshold": best_sweep_thresh,
        "test_metrics": test_metrics_opt
    }
    (model_exp_dir / "test_metrics.json").write_text(json.dumps(test_metrics_json, indent=2), encoding="utf-8")
    
    # Save per-class metrics CSV
    df_per_class = pd.DataFrame(test_metrics_opt["per_category"])
    df_per_class.to_csv(model_exp_dir / "per_class_metrics.csv", index=False)
    
    # Save predictions.csv
    test_preds_binary = (test_probs >= best_sweep_thresh).astype(int)
    df_preds = df_test[["clause_id", "document_id", "clause_text"]].copy()
    
    for c, cat in enumerate(APPROVED_CATEGORIES):
        label_col = LABEL_COLS[c]
        df_preds[f"true_{label_col}"] = test_trues[:, c]
        df_preds[f"prob_{label_col}"] = np.round(test_probs[:, c], 4)
        df_preds[f"pred_{label_col}"] = test_preds_binary[:, c]
        
    df_preds.to_csv(model_exp_dir / "predictions.csv", index=False)
    
    return {
        "Model": model_display_name,
        "model_id": model_id,
        "Best Epoch": best_epoch,
        "Best Val Macro F1": round(best_sweep_val_macro_f1, 4),
        "Threshold": best_sweep_thresh,
        "Test Macro F1": test_metrics_opt["macro_f1"],
        "Test Micro F1": test_metrics_opt["micro_f1"],
        "Test Weighted F1": test_metrics_opt["weighted_f1"],
        "Minority F1": test_metrics_opt["minority_macro_f1"],
        "Hamming Loss": test_metrics_opt["hamming_loss"],
        "MCC": test_metrics_opt["mcc"],
        "per_category": {pc["category"]: pc["f1_score"] for pc in test_metrics_opt["per_category"]},
        "skipped": False
    }


def run_exp12_model_comparison():
    set_seed(42)
    
    # Step 1: Verify GPU & CUDA
    gpu_meta = verify_gpu_and_cuda()
    
    # Step 2: Data Validation & Counts
    df_train_orig, df_synth, df_combined_train, df_val, df_test = perform_data_validation()
    
    exp_dir = ROOT / "experiments/EXP-12_model_comparison"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("=======================================================")
    print("    STEP 3: ARTIFACT VALIDATION PRE-FLIGHT CHECK       ")
    print("=======================================================")
    for m in MODELS:
        m_dir = exp_dir / m["id"]
        completed = is_model_completed(m_dir)
        print(f"{m['name']} completed: {completed}")
    print("=======================================================\n")
    
    model_results = []
    
    for model_meta in MODELS:
        res = train_single_model(
            model_meta=model_meta,
            df_train_orig=df_train_orig,
            df_synth=df_synth,
            df_combined_train=df_combined_train,
            df_val=df_val,
            df_test=df_test,
            gpu_meta=gpu_meta,
            exp_dir=exp_dir
        )
        model_results.append(res)
        
    # Generate overall model_comparison.csv
    summary_rows = []
    for r in model_results:
        summary_rows.append({
            "Model": r["Model"],
            "Best Epoch": r["Best Epoch"],
            "Best Val Macro F1": r["Best Val Macro F1"],
            "Threshold": r["Threshold"],
            "Test Macro F1": r["Test Macro F1"],
            "Test Micro F1": r["Test Micro F1"],
            "Test Weighted F1": r["Test Weighted F1"],
            "Minority F1": r["Minority F1"],
            "Hamming Loss": r["Hamming Loss"],
            "MCC": r["MCC"]
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(exp_dir / "model_comparison.csv", index=False)
    
    # Generate Markdown Report
    best_model_row = df_summary.sort_values(by="Test Macro F1", ascending=False).iloc[0]
    
    md = [
        "# EXP-12: Transformer Architecture Model Comparison Report\n",
        "## Executive Summary\n",
        "This experiment evaluates **3 transformer architectures** (`Legal-RoBERTa`, `Legal-BERT`, and `DeBERTa-v3`) under identical training parameters (Class-Weighted BCE, 5,344 combined clauses, max 10 epochs with Early Stopping patience=2, learning rate 2e-5, seed 42) on the frozen NDA clause benchmark.\n",
        "### Benchmark Dataset Parameters",
        "- **Combined Training Set**: 5,344 clauses (344 genuine + 5,000 synthetic)",
        "- **Validation Set**: 137 clauses (untouched)",
        "- **Test Set**: 236 clauses (untouched)\n",
        "---\n",
        "## Overall Model Comparison Table\n",
        "| Model | Best Epoch | Best Val Macro F1 | Threshold | Test Macro F1 | Test Micro F1 | Test Weighted F1 | Minority F1 | Hamming Loss | MCC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for r in summary_rows:
        md.append(
            f"| **{r['Model']}** | {r['Best Epoch']} | `{r['Best Val Macro F1']:.4f}` | `{r['Threshold']:.2f}` | "
            f"**`{r['Test Macro F1']:.4f}`** | `{r['Test Micro F1']:.4f}` | `{r['Test Weighted F1']:.4f}` | "
            f"`{r['Minority F1']:.4f}` | `{r['Hamming Loss']:.4f}` | `{r['MCC']:.4f}` |"
        )

    md.extend([
        "\n---\n",
        "## Per-Category 14-Label F1 Comparison\n",
        "| Category | Legal-RoBERTa | Legal-BERT | DeBERTa-v3 | Best Architecture |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])
    
    # Category lookup dicts
    cat_f1_roberta = model_results[0]["per_category"]
    cat_f1_bert = model_results[1]["per_category"]
    cat_f1_deberta = model_results[2]["per_category"]
    
    for cat in APPROVED_CATEGORIES:
        f1_r = cat_f1_roberta.get(cat, 0.0)
        f1_b = cat_f1_bert.get(cat, 0.0)
        f1_d = cat_f1_deberta.get(cat, 0.0)
        
        scores = [("Legal-RoBERTa", f1_r), ("Legal-BERT", f1_b), ("DeBERTa-v3", f1_d)]
        best_arch = max(scores, key=lambda x: x[1])[0]
        
        md.append(f"| {cat} | `{f1_r:.4f}` | `{f1_b:.4f}` | `{f1_d:.4f}` | **`{best_arch}`** |")

    md.extend([
        "\n---\n",
        "## Key Findings\n",
        f"- **Winning Architecture**: **`{best_model_row['Model']}`** achieved the highest Test Macro F1 (**`{best_model_row['Test Macro F1']:.4f}`**) at decision threshold `{best_model_row['Threshold']:.2f}`.",
        "- **Generalization & Calibration**: Comparative evaluation across domain-adapted (Legal-RoBERTa, Legal-BERT) vs disentangled attention (DeBERTa-v3) models on multi-label legal text.\n",
        "## Reproducibility Command\n",
        "To reproduce EXP-12 from scratch on your RTX 5060 Ti GPU, run:\n",
        "```powershell",
        "python scripts/train_exp12_model_comparison.py",
        "```\n"
    ])

    (exp_dir / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[SUCCESS] EXP-12 Model Comparison Complete! All artifacts saved to {exp_dir}\n")


if __name__ == "__main__":
    run_exp12_model_comparison()
