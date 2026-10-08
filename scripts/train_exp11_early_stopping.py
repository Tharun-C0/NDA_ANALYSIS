"""
scripts/train_exp11_early_stopping.py

EXP-11: Early Stopping Experiment with Synthetic Data (5,000 Synthetic Clauses)
Model: saibo/legal-roberta-base
Loss: Class-Weighted BCE (computed exclusively from the 5,344 combined training set)
Training Data: Existing benchmark train (344 clauses) + New Synthetic Dataset (5,000 clauses) = 5,344 rows
Validation Data: Existing benchmark validation set (137 clauses)
Test Data: Existing benchmark test set (236 clauses)
Max Epochs: 10
Early Stopping: Patience = 2 (Monitors Validation Macro F1)
Checkpoint: Best validation Macro F1 checkpoint restored before test evaluation
Device: GPU/CUDA (RTX 5060 Ti with Mixed Precision FP16/BF16)
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
    
    print(f"GPU name:          {gpu_name}")
    print(f"CUDA version:      {cuda_ver}")
    print(f"GPU memory:        {gpu_vram} GB")
    print("[CHECK PASS] GPU environment verified successfully.\n")
    
    return {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "cuda_version": cuda_ver,
        "pytorch_version": torch.__version__,
        "gpu_memory_gb": gpu_vram
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

    # Standardize synthetic dataset label columns to match benchmark label_... columns
    df_synth = df_synth_raw.copy()
    for cat, label_col in CAT_TO_LABEL.items():
        if cat in df_synth.columns:
            df_synth[label_col] = df_synth[cat]

    # 1. Synthetic dataset exact 5,000 rows check
    assert len(df_synth) == 5000, f"Expected 5,000 rows in synthetic dataset, got {len(df_synth)}"
    print("[PASS] Synthetic dataset has exactly 5,000 rows.")

    # 2. All 14 labels exist
    missing_synth_labels = [c for c in LABEL_COLS if c not in df_synth.columns]
    assert len(missing_synth_labels) == 0, f"Missing label columns in synthetic: {missing_synth_labels}"
    print("[PASS] All 14 category label columns exist in synthetic dataset.")

    # 3. No invalid labels
    invalid_synth_vals = set(df_synth[LABEL_COLS].values.flatten()) - {0, 1}
    assert len(invalid_synth_vals) == 0, f"Invalid label values found in synthetic: {invalid_synth_vals}"
    print("[PASS] All label entries in synthetic dataset are strictly binary (0 or 1).")

    # 4. No empty clause text
    empty_synth_text = df_synth[df_synth["clause_text"].isna() | (df_synth["clause_text"].str.strip() == "")]
    assert len(empty_synth_text) == 0, f"Found {len(empty_synth_text)} empty clause text rows in synthetic!"
    print("[PASS] No empty clause text in synthetic dataset.")

    # 5. No duplicate clause IDs
    dup_cids = df_synth[df_synth.duplicated(subset=["clause_id"])]
    assert len(dup_cids) == 0, f"Found {len(dup_cids)} duplicate clause IDs in synthetic dataset!"
    print("[PASS] No duplicate clause IDs in synthetic dataset.")

    # 6. No duplicate clause text within synthetic dataset
    dup_synth_text = df_synth[df_synth.duplicated(subset=["clause_text"])]
    assert len(dup_synth_text) == 0, f"Found {len(dup_synth_text)} duplicate clause text rows in synthetic dataset!"
    print("[PASS] No duplicate clause text within synthetic dataset.")

    # 7. Label columns match existing benchmark
    benchmark_label_cols = [c for c in df_train.columns if c.startswith("label_") and c != "label_source"]
    assert sorted(LABEL_COLS) == sorted(benchmark_label_cols), "Label columns mismatch with benchmark"
    print("[PASS] Label columns match existing benchmark.")

    # 8. Training/validation/test schemas match
    train_cols = list(df_train.columns)
    val_cols = list(df_val.columns)
    test_cols = list(df_test.columns)
    assert train_cols == val_cols == test_cols, "Training/validation/test schemas mismatch"
    print("[PASS] Benchmark training, validation, and test schemas match perfectly.")

    # 9. Synthetic data is NOT present in validation or test sets
    val_texts = set(df_val["clause_text"].str.strip())
    test_texts = set(df_test["clause_text"].str.strip())
    synth_texts = set(df_synth["clause_text"].str.strip())

    synth_in_val = synth_texts.intersection(val_texts)
    synth_in_test = synth_texts.intersection(test_texts)
    assert len(synth_in_val) == 0, f"Found {len(synth_in_val)} synthetic clauses in validation set!"
    assert len(synth_in_test) == 0, f"Found {len(synth_in_test)} synthetic clauses in test set!"
    print("[PASS] Synthetic data is NOT present in validation set (0 overlap).")
    print("[PASS] Synthetic data is NOT present in test set (0 overlap).")

    # Construct combined training set
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
    print(f"Existing training rows:  {len(df_train)}")
    print(f"Synthetic training rows: {len(df_synth)}")
    print(f"Combined training rows:  {len(df_combined_train)}")
    print(f"Validation rows:        {len(df_val)}")
    print(f"Test rows:              {len(df_test)}")
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
    return pos_weights


def run_exp11_training():
    set_seed(42)
    
    # Step 1: Verify GPU & CUDA
    gpu_meta = verify_gpu_and_cuda()
    device = torch.device("cuda")
    
    # Step 2: Data Validation
    df_train_orig, df_synth, df_combined_train, df_val, df_test = perform_data_validation()
    
    # Setup Output Directory
    exp_dir = ROOT / "experiments/EXP-11_early_stopping"
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_dir = exp_dir / "best_checkpoint"
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "saibo/legal-roberta-base"
    max_epochs = 10
    patience = 2
    batch_size = 16
    learning_rate = 2e-5
    max_length = 256
    
    print("=======================================================")
    print("    STEP 3: INITIALIZING MODEL & COMBINED DATASET     ")
    print("=======================================================")
    print(f"Model:                {model_name}")
    print(f"Loss Function:        Class-Weighted BCE (from combined 5,344 train rows)")
    print(f"Max Epochs:           {max_epochs}")
    print(f"Early Stopping:       Patience = {patience} (Monitors Validation Macro F1)")
    print(f"Batch Size:           {batch_size}")
    print(f"Learning Rate:        {learning_rate}")
    print(f"Sequence Length:      {max_length}")
    print(f"Device:               {device} ({gpu_meta['gpu_name']})")
    print("=======================================================\n")
    
    # Compute positive weights from combined train ONLY
    pos_weights = compute_class_weights_from_combined_train(df_combined_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=14,
        problem_type="multi_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)
    
    # Prepare Datasets & Loaders
    train_dataset = NDAClauseDataset(df_combined_train["clause_text"].tolist(), df_combined_train[LABEL_COLS].values, tokenizer, max_length)
    val_dataset = NDAClauseDataset(df_val["clause_text"].tolist(), df_val[LABEL_COLS].values, tokenizer, max_length)
    test_dataset = NDAClauseDataset(df_test["clause_text"].tolist(), df_test[LABEL_COLS].values, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda')
    
    minority_indices = get_minority_category_indices(df_train_orig)
    
    # Save training configuration
    train_config = {
        "experiment_id": "EXP-11",
        "description": "Legal-RoBERTa + Class-Weighted BCE + Synthetic-5000 + Early Stopping (Patience=2)",
        "model_name": model_name,
        "loss_function": "Class-Weighted BCE",
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_sequence_length": max_length,
        "seed": 42,
        "gpu_environment": gpu_meta,
        "dataset_rows": {
            "benchmark_train": len(df_train_orig),
            "synthetic_train": len(df_synth),
            "combined_train": len(df_combined_train),
            "validation": len(df_val),
            "test": len(df_test)
        },
        "pos_weights": [round(float(w), 4) for w in pos_weights.cpu().numpy()]
    }
    (exp_dir / "training_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    
    print("=======================================================")
    print("  STEP 4: EXECUTING TRAINING LOOP WITH EARLY STOPPING ")
    print("=======================================================")
    
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
                
            with torch.amp.autocast('cuda'):
                outputs = model(**kwargs)
                loss = criterion(outputs.logits, labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
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
                    
                with torch.amp.autocast('cuda'):
                    outputs = model(**kwargs)
                    loss = criterion(outputs.logits, labels)
                    
                running_val_loss += loss.item()
                val_logits_list.append(outputs.logits.cpu())
                val_trues_list.append(labels.cpu().numpy())
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_logits = torch.cat(val_logits_list, dim=0).numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_trues = np.vstack(val_trues_list)
        
        # Evaluate validation metrics at standard 0.60 threshold for monitoring
        val_metrics = compute_all_metrics(val_trues, val_probs, threshold=0.60, minority_indices=minority_indices)
        ep_duration = round(time.time() - start_ep, 2)
        
        val_macro_f1 = val_metrics["macro_f1"]
        is_best = val_macro_f1 > best_val_macro_f1
        
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
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
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | Micro F1: {val_metrics['micro_f1']:.4f} | "
            f"Weighted F1: {val_metrics['weighted_f1']:.4f} | Minority F1: {val_metrics['minority_macro_f1']:.4f} | "
            f"Hamming Loss: {val_metrics['hamming_loss']:.4f} | MCC: {val_metrics['mcc']:.4f} "
            f"{'[BEST]' if is_best else f'[Patience {patience_counter}/{patience}]'}"
        )
        
        if patience_counter >= patience:
            stopped_epoch = epoch
            print(f"\n[EARLY STOPPING TRIGGERED] Validation Macro F1 did not improve for {patience} consecutive epochs.")
            print(f"Stopping training at epoch {epoch}. Restoring best checkpoint from Epoch {best_epoch} (Val Macro F1 = {best_val_macro_f1:.4f}).\n")
            break

    # Save training log CSV
    df_train_log = pd.DataFrame(training_history)
    df_train_log.to_csv(exp_dir / "training_log.csv", index=False)
    
    print("=======================================================")
    print("   STEP 5: VALIDATION THRESHOLD SWEEP (BEST CHECKPOINT) ")
    print("=======================================================")
    
    # Load BEST model checkpoint for evaluation and threshold sweep
    print(f"Loading best model checkpoint from Epoch {best_epoch}: {best_checkpoint_dir}...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir)
    best_model.to(device)
    best_model.eval()
    
    # Obtain validation probabilities with best checkpoint
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
            with torch.amp.autocast('cuda'):
                outputs = best_model(**kwargs)
            val_logits_list.append(outputs.logits.cpu())
            val_trues_list.append(labels.cpu().numpy())
            
    val_logits = torch.cat(val_logits_list, dim=0).numpy()
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_trues = np.vstack(val_trues_list)
    
    sweep_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_results = []
    best_sweep_thresh = 0.60
    best_sweep_val_macro_f1 = -1.0
    
    print("Threshold | Val Macro F1 | Val Minority F1 | Val MCC | Val Micro F1 | Val Weighted F1 | Val Hamming Loss")
    print("-" * 95)
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
        
        is_selected = t_res["macro_f1"] > best_sweep_val_macro_f1
        if is_selected:
            best_sweep_val_macro_f1 = t_res["macro_f1"]
            best_sweep_thresh = t
            
        print(
            f"  {t:.2f}    |   {t_res['macro_f1']:.4f}     |     {t_res['minority_macro_f1']:.4f}    |  {t_res['mcc']:.4f} |   {t_res['micro_f1']:.4f}   |     {t_res['weighted_f1']:.4f}     |     {t_res['hamming_loss']:.4f} "
            f"{' (SELECTED)' if is_selected else ''}"
        )
        
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(exp_dir / "threshold_sweep_results.csv", index=False)
    
    print(f"\n[VALIDATION SWEEP RESULT] Selected Threshold based ONLY on Validation set: {best_sweep_thresh:.2f} (Val Macro F1 = {best_sweep_val_macro_f1:.4f})")
    
    print("\n=======================================================")
    print("     STEP 6: FROZEN TEST SET EVALUATION (EXP-11)      ")
    print("=======================================================")
    
    # Predict on test set ONCE using best checkpoint
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
            with torch.amp.autocast('cuda'):
                outputs = best_model(**kwargs)
            test_logits_list.append(outputs.logits.cpu())
            test_trues_list.append(labels.cpu().numpy())
            
    test_logits = torch.cat(test_logits_list, dim=0).numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_trues = np.vstack(test_trues_list)
    
    test_metrics_060 = compute_all_metrics(test_trues, test_probs, threshold=0.60, minority_indices=minority_indices)
    test_metrics_opt = compute_all_metrics(test_trues, test_probs, threshold=best_sweep_thresh, minority_indices=minority_indices)
    
    print(f"--- TEST METRICS @ THRESHOLD = 0.60 ---")
    print(f"Macro F1:      {test_metrics_060['macro_f1']:.4f}")
    print(f"Micro F1:      {test_metrics_060['micro_f1']:.4f}")
    print(f"Weighted F1:   {test_metrics_060['weighted_f1']:.4f}")
    print(f"Minority F1:   {test_metrics_060['minority_macro_f1']:.4f}")
    print(f"Hamming Loss:  {test_metrics_060['hamming_loss']:.4f}")
    print(f"MCC:           {test_metrics_060['mcc']:.4f}")
    
    print(f"\n--- TEST METRICS @ OPTIMAL VALIDATION THRESHOLD = {best_sweep_thresh:.2f} ---")
    print(f"Macro F1:      {test_metrics_opt['macro_f1']:.4f}")
    print(f"Micro F1:      {test_metrics_opt['micro_f1']:.4f}")
    print(f"Weighted F1:   {test_metrics_opt['weighted_f1']:.4f}")
    print(f"Minority F1:   {test_metrics_opt['minority_macro_f1']:.4f}")
    print(f"Hamming Loss:  {test_metrics_opt['hamming_loss']:.4f}")
    print(f"MCC:           {test_metrics_opt['mcc']:.4f}")

    primary_test_metrics = test_metrics_opt
    
    # Save test metrics JSON
    test_metrics_json = {
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "threshold_0.60": test_metrics_060,
        "threshold_validation_selected": test_metrics_opt,
        "selected_threshold": best_sweep_thresh
    }
    (exp_dir / "test_metrics.json").write_text(json.dumps(test_metrics_json, indent=2), encoding="utf-8")
    
    # Save per-class metrics CSV
    df_per_class = pd.DataFrame(primary_test_metrics["per_category"])
    df_per_class.to_csv(exp_dir / "per_class_metrics.csv", index=False)
    
    # Save predictions.csv
    test_preds_binary = (test_probs >= best_sweep_thresh).astype(int)
    df_preds = df_test[["clause_id", "document_id", "clause_text"]].copy()
    
    for c, cat in enumerate(APPROVED_CATEGORIES):
        label_col = LABEL_COLS[c]
        df_preds[f"true_{label_col}"] = test_trues[:, c]
        df_preds[f"prob_{label_col}"] = np.round(test_probs[:, c], 4)
        df_preds[f"pred_{label_col}"] = test_preds_binary[:, c]
        
    df_preds.to_csv(exp_dir / "predictions.csv", index=False)
    
    print("\n=======================================================")
    print("  STEP 7: 3-EXPERIMENT COMPARISON (EXP-03 vs EXP-10 vs EXP-11) ")
    print("=======================================================")
    
    # Baseline EXP-03 results
    exp03 = {
        "Experiment": "EXP-03",
        "Model": "Legal-RoBERTa",
        "Loss": "Class-Weighted BCE",
        "Synthetic Data": "None (0)",
        "Epochs": "5 (Fixed)",
        "Threshold": 0.60,
        "Macro F1": 0.4295,
        "Micro F1": 0.4940,
        "Weighted F1": 0.5151,
        "Minority F1": 0.3853,
        "Hamming Loss": 0.1283,
        "MCC": 0.3885
    }
    
    # EXP-10 results
    exp10_path = ROOT / "experiments/EXP-10_synthetic_5000/before_after_comparison.csv"
    if exp10_path.exists():
        df_exp10_raw = pd.read_csv(exp10_path)
        exp10_row = df_exp10_raw[df_exp10_raw["Experiment"] == "EXP-10"].iloc[0]
        exp10 = {
            "Experiment": "EXP-10",
            "Model": str(exp10_row["Model"]),
            "Loss": str(exp10_row["Loss"]),
            "Synthetic Data": "5,000 Clauses",
            "Epochs": "10 (Fixed)",
            "Threshold": float(exp10_row["Threshold"]),
            "Macro F1": float(exp10_row["Macro F1"]),
            "Micro F1": float(exp10_row["Micro F1"]),
            "Weighted F1": float(exp10_row["Weighted F1"]),
            "Minority F1": float(exp10_row["Minority F1"]),
            "Hamming Loss": float(exp10_row["Hamming Loss"]),
            "MCC": float(exp10_row["MCC"])
        }
    else:
        exp10 = {
            "Experiment": "EXP-10",
            "Model": "Legal-RoBERTa",
            "Loss": "Class-Weighted BCE",
            "Synthetic Data": "5,000 Clauses",
            "Epochs": "10 (Fixed)",
            "Threshold": 0.45,
            "Macro F1": 0.4643,
            "Micro F1": 0.5543,
            "Weighted F1": 0.5568,
            "Minority F1": 0.3769,
            "Hamming Loss": 0.1081,
            "MCC": 0.4118
        }

    exp11 = {
        "Experiment": "EXP-11",
        "Model": "Legal-RoBERTa",
        "Loss": "Class-Weighted BCE",
        "Synthetic Data": "5,000 Clauses",
        "Epochs": f"{best_epoch} (Early Stopped at Ep {stopped_epoch})",
        "Threshold": best_sweep_thresh,
        "Macro F1": primary_test_metrics["macro_f1"],
        "Micro F1": primary_test_metrics["micro_f1"],
        "Weighted F1": primary_test_metrics["weighted_f1"],
        "Minority F1": primary_test_metrics["minority_macro_f1"],
        "Hamming Loss": primary_test_metrics["hamming_loss"],
        "MCC": primary_test_metrics["mcc"]
    }

    df_comp = pd.DataFrame([exp03, exp10, exp11])
    df_comp.to_csv(exp_dir / "before_after_comparison.csv", index=False)
    
    # Calculate EXP-11 vs EXP-03 changes
    m_diff_vs_03 = exp11["Macro F1"] - exp03["Macro F1"]
    m_pct_vs_03 = (m_diff_vs_03 / exp03["Macro F1"]) * 100
    
    # Calculate EXP-11 vs EXP-10 changes
    m_diff_vs_10 = exp11["Macro F1"] - exp10["Macro F1"]
    m_pct_vs_10 = (m_diff_vs_10 / exp10["Macro F1"]) * 100

    exp03_per_cat_f1 = {
        "Party Identification": 0.4318,
        "Purpose": 0.0741,
        "NDA Type": 0.2857,
        "Definition of Confidential Information": 0.4211,
        "Confidentiality Obligations": 0.4444,
        "Authorized Disclosure": 0.3404,
        "Non-Confidential Information": 0.3200,
        "Liability for Damages": 0.5957,
        "Competition Rights": 0.4865,
        "Term and Termination": 0.4524,
        "Intellectual Property": 0.5714,
        "Employees": 0.4571,
        "Governing Law and Jurisdiction": 0.4828,
        "Additional Information": 0.6494
    }

    exp10_per_cat_path = ROOT / "experiments/EXP-10_synthetic_5000/per_class_metrics.csv"
    exp10_per_cat_f1 = {}
    if exp10_per_cat_path.exists():
        df_10_pc = pd.read_csv(exp10_per_cat_path)
        exp10_per_cat_f1 = dict(zip(df_10_pc["category"], df_10_pc["f1_score"]))

    md = [
        "# EXP-11: Early Stopping Experiment Comparison Report\n",
        "## Executive Summary\n",
        f"This report documents **EXP-11**, evaluating the impact of **Early Stopping** (Patience = 2 on Validation Macro F1) when training `saibo/legal-roberta-base` with `Class-Weighted BCE` on the 5,344 combined dataset (344 benchmark train + 5,000 synthetic clauses). Best checkpoint from Epoch {best_epoch} was restored before test evaluation.\n",
        "### Key Parameters across Experiments",
        "- **EXP-03**: Benchmark Train (344 rows) | 5 Epochs Fixed | Threshold 0.60",
        "- **EXP-10**: Combined Train (5,344 rows) | 10 Epochs Fixed | Threshold 0.45",
        f"- **EXP-11**: Combined Train (5,344 rows) | Early Stopped at Ep {stopped_epoch} (Best Ep {best_epoch}) | Threshold {best_sweep_thresh:.2f}\n",
        "---\n",
        "## Overall 3-Experiment Metric Comparison Table\n",
        "| Metric | EXP-03 Baseline | EXP-10 (Fixed 10 Ep) | EXP-11 (Early Stopped) | EXP-11 vs EXP-03 | EXP-11 vs EXP-10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Macro F1** | `{exp03['Macro F1']:.4f}` | `{exp10['Macro F1']:.4f}` | **`{exp11['Macro F1']:.4f}`** | `{m_diff_vs_03:+.4f}` (`{m_pct_vs_03:+.2f}%`) | `{m_diff_vs_10:+.4f}` (`{m_pct_vs_10:+.2f}%`) |",
        f"| **Minority F1** | `{exp03['Minority F1']:.4f}` | `{exp10['Minority F1']:.4f}` | **`{exp11['Minority F1']:.4f}`** | `{exp11['Minority F1'] - exp03['Minority F1']:+.4f}` | `{exp11['Minority F1'] - exp10['Minority F1']:+.4f}` |",
        f"| **Micro F1** | `{exp03['Micro F1']:.4f}` | `{exp10['Micro F1']:.4f}` | **`{exp11['Micro F1']:.4f}`** | `{exp11['Micro F1'] - exp03['Micro F1']:+.4f}` | `{exp11['Micro F1'] - exp10['Micro F1']:+.4f}` |",
        f"| **Weighted F1** | `{exp03['Weighted F1']:.4f}` | `{exp10['Weighted F1']:.4f}` | **`{exp11['Weighted F1']:.4f}`** | `{exp11['Weighted F1'] - exp03['Weighted F1']:+.4f}` | `{exp11['Weighted F1'] - exp10['Weighted F1']:+.4f}` |",
        f"| **Hamming Loss** | `{exp03['Hamming Loss']:.4f}` | `{exp10['Hamming Loss']:.4f}` | **`{exp11['Hamming Loss']:.4f}`** | `{exp11['Hamming Loss'] - exp03['Hamming Loss']:+.4f}` | `{exp11['Hamming Loss'] - exp10['Hamming Loss']:+.4f}` |",
        f"| **MCC** | `{exp03['MCC']:.4f}` | `{exp10['MCC']:.4f}` | **`{exp11['MCC']:.4f}`** | `{exp11['MCC'] - exp03['MCC']:+.4f}` | `{exp11['MCC'] - exp10['MCC']:+.4f}` |\n",
        "---\n",
        "## Per-Category 14-Label F1 Comparison\n",
        "| Category | Baseline EXP-03 F1 | EXP-10 F1 | EXP-11 F1 | EXP-11 vs EXP-03 | EXP-11 vs EXP-10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for pc in primary_test_metrics["per_category"]:
        cat = pc["category"]
        f1_03 = exp03_per_cat_f1.get(cat, 0.0)
        f1_10 = exp10_per_cat_f1.get(cat, 0.0)
        f1_11 = pc["f1_score"]
        d_03 = f1_11 - f1_03
        d_10 = f1_11 - f1_10
        md.append(f"| {cat} | `{f1_03:.4f}` | `{f1_10:.4f}` | **`{f1_11:.4f}`** | `{d_03:+.4f}` | `{d_10:+.4f}` |")

    md.extend([
        "\n---\n",
        "## Key Findings\n",
        f"- **Best Model Checkpoint**: Restored from **Epoch {best_epoch}**.",
        f"- **Early Stopping Trigger**: Stopped training at **Epoch {stopped_epoch}**.",
        f"- **Selected Threshold**: **`{best_sweep_thresh:.2f}`** (selected via Validation Macro F1 sweep).\n",
        "## Reproducibility Command\n",
        "To execute EXP-11 training and evaluation on your RTX 5060 Ti GPU, run:\n",
        "```powershell",
        "python scripts/train_exp11_early_stopping.py",
        "```\n"
    ])

    (exp_dir / "before_after_comparison.md").write_text("\n".join(md), encoding="utf-8")
    
    print("\n[SUCCESS] EXP-11 Training & Evaluation Pipeline Finished!")
    print(f"All experiment artifacts saved to: {exp_dir}")


if __name__ == "__main__":
    run_exp11_training()
