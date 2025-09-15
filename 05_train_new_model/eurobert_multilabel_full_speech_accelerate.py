import os
import numpy as np
import torch
import pandas as pd

from tqdm.auto import tqdm

# Transformers / Accelerate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator

# Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Data Handling
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------
# 1) ENV and Basic Setup
# ------------------------------------------------------------------------------------
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # Or whichever GPUs you want
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create an Accelerator instance
accelerator = Accelerator()

# ------------------------------------------------------------------------------------
# 2) Load your dataset & Model
# ------------------------------------------------------------------------------------
ds = load_dataset("SinclairSchneider/trainset_political_party_big")
df = ds['train'].to_pandas()

# Prepare labels and mappings
label_list = ["AfD", "BÜNDNIS 90/DIE GRÜNEN", "CDU/CSU", "DIE LINKE", "FDP", "SPD"]
num_labels = len(label_list)

id2label = {idx: label for idx, label in enumerate(label_list)}
label2id = {label: idx for idx, label in enumerate(label_list)}

# Preprocess label columns for multi-label
df["senti_AfD"] = df["senti_AfD"].apply(lambda n: 1 if n == 1 else 0)
df["senti_BUENDNIS_90_DIE_GRUENEN"] = df["senti_BUENDNIS_90_DIE_GRUENEN"].apply(lambda n: 1 if n == 1 else 0)
df["senti_CDU_CSU"] = df["senti_CDU_CSU"].apply(lambda n: 1 if n == 1 else 0)
df["senti_DIE_LINKE"] = df["senti_DIE_LINKE"].apply(lambda n: 1 if n == 1 else 0)
df["senti_FDP"] = df["senti_FDP"].apply(lambda n: 1 if n == 1 else 0)
df["senti_SPD"] = df["senti_SPD"].apply(lambda n: 1 if n == 1 else 0)

labels_data = list(
    zip(
        df["senti_AfD"].values.astype(float),
        df["senti_BUENDNIS_90_DIE_GRUENEN"].values.astype(float),
        df["senti_CDU_CSU"].values.astype(float),
        df["senti_DIE_LINKE"].values.astype(float),
        df["senti_FDP"].values.astype(float),
        df["senti_SPD"].values.astype(float),
    )
)

# Train/Test Split
sentences = df["text"].values
train_texts, test_texts, train_labels, test_labels = train_test_split(
    sentences, labels_data, test_size=0.2
)

# ------------------------------------------------------------------------------------
# 3) On-the-fly Dataset Definition
# ------------------------------------------------------------------------------------
class PoliticsOnTheFlyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=8192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize here
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

        # Convert to PyTorch tensors
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item

# ------------------------------------------------------------------------------------
# 4) Instantiate Tokenizer, Model, and Datasets
# ------------------------------------------------------------------------------------
#model_name = "EuroBERT/EuroBERT-2.1B"
#model_name = "EuroBERT/EuroBERT-610m"
model_name = "EuroBERT/EuroBERT-210m"
max_length = 8192

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    do_lower_case=False,
    max_length=max_length,
    TOKENIZERS_PARALLELISM=True,
    trust_remote_code=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True
)

train_dataset = PoliticsOnTheFlyDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = PoliticsOnTheFlyDataset(test_texts, test_labels, tokenizer, max_length)

# ------------------------------------------------------------------------------------
# 5) Create Dataloaders
# ------------------------------------------------------------------------------------
#train_batch_size = 1
train_batch_size = 4
eval_batch_size = 8

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

# ------------------------------------------------------------------------------------
# 6) Define Optimizer and (Optional) Scheduler
# ------------------------------------------------------------------------------------
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# You can optionally set up a learning rate scheduler if you like,
# but for simplicity, let's omit that here or define a simple one:
# from transformers import get_linear_schedule_with_warmup
# total_steps = len(train_dataloader) * num_epochs
# scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=0, t_total=total_steps)

# ------------------------------------------------------------------------------------
# 7) Prepare with Accelerate
# ------------------------------------------------------------------------------------
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
# If you have a scheduler, you would also pass it to `.prepare(...)`

# ------------------------------------------------------------------------------------
# 8) Training Loop
# ------------------------------------------------------------------------------------
num_epochs = 4
best_f1 = -1.0
#output_dir = "./politic_EuroBERT-2.1B_multilabel_bundestag_and_wahlomat"
output_dir = "./politic_EuroBERT-210m_multilabel_bundestag_and_wahlomat"

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    y_true, y_pred: (batch_size, num_labels)
      – y_pred are raw logits that must be passed through a sigmoid.
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(y_pred)
    preds = (probs >= threshold).float().cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='micro')
    roc_auc = roc_auc_score(y_true, preds, average='micro')
    accuracy = accuracy_score(y_true, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    # Average training loss across all batches
    train_loss = total_loss / len(train_dataloader)

    # --------------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------------
    model.eval()
    all_labels = []
    all_logits = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        logits = outputs.logits
        labels = batch["labels"]

        all_labels.append(labels)
        all_logits.append(logits)

    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    metrics = compute_metrics(all_labels, all_logits)
    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Valid Metrics: {metrics}")

    current_f1 = metrics['f1']
    # Save best model
    if current_f1 > best_f1:
        best_f1 = current_f1
        accelerator.print(f"New best F1: {best_f1:.4f}, saving model...")
        # Use accelerator to safely save the model (handles multi-GPU, etc.)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

# ------------------------------------------------------------------------------------
# 9) Save tokenizer as well
# ------------------------------------------------------------------------------------
if accelerator.is_main_process:
    tokenizer.save_pretrained(output_dir)