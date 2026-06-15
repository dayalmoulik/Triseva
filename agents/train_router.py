import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# Set seed for reproducibility
set_seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-multilingual-cased"
DATA_PATH = "evaluation/results/triseva_qa_dataset.json"
OUTPUT_DIR = "data/models/domain_router"
LOGS_DIR = "data/models/logs"

DOMAIN_TO_LABEL = {
    "health": 0,
    "legal": 1,
    "agriculture": 2
}
LABEL_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_LABEL.items()}


# ── Dataset Wrapper ───────────────────────────────────────────────────────────
class TriSevaRoutingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Metrics Helper ────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# ── Main Training Pipeline ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STARTING DOMAIN ROUTER TRAINING")
    print("=" * 60)

    # 1. Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run curate_dataset.py first.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs.")

    texts = [item["question"] for item in qa_pairs]
    labels = [DOMAIN_TO_LABEL[item["domain"]] for item in qa_pairs]

    # Verify distributions
    from collections import Counter
    dist = Counter(labels)
    print("Domain distribution:")
    for lbl, count in dist.items():
        print(f"  {LABEL_TO_DOMAIN[lbl]:<12} : {count}")

    # 2. Train-Validation Split (Stratified)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"Train samples: {len(train_texts)} | Val samples: {len(val_texts)}")

    # 3. Tokenization
    print(f"Loading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Tokenizing train and validation sets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = TriSevaRoutingDataset(train_encodings, train_labels)
    val_dataset = TriSevaRoutingDataset(val_encodings, val_labels)

    # 4. Model setup
    print(f"Loading classification model: {BASE_MODEL}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(DOMAIN_TO_LABEL)
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGS_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        use_cpu=True,  # enforce CPU training since CUDA was not found
        disable_tqdm=True,  # avoid progress bar cluttering log files
        report_to="none"  # disable wandb/tensorboard integrations
    )

    # 6. Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. Fine-tuning
    print("Training started...")
    trainer.train()
    print("Training finished.")

    # 8. Evaluation
    print("\nRunning evaluation on validation set...")
    val_predictions = trainer.predict(val_dataset)
    preds = np.argmax(val_predictions.predictions, axis=-1)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(val_labels, preds, target_names=list(DOMAIN_TO_LABEL.keys())))

    # 9. Save Best Model
    print(f"Saving fine-tuned model and tokenizer to: {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[SUCCESS] Model successfully saved!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
