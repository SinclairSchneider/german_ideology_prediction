from transformers import AutoConfig, LlamaForSequenceClassification, LlamaTokenizer, EvalPrediction, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
import numpy as np
import pandas as pd
import os
import wandb

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#run_name = "politic_EuroBERT-210m_multilabel_bundestag_and_wahlomat"
#run_name = "politic_EuroBERT-610m_multilabel_bundestag_and_wahlomat"
#run_name = "politic_EuroBERT-2.1B_multilabel_bundestag_and_wahlomat"
#run_name = "politic_Llama-3.2-1B_multilabel_bundestag_and_wahlomat"
#run_name = "politic_Qwen2.5-1.5B_multilabel_bundestag_and_wahlomat"
run_name = "politic_gemma-3-1b_multilabel_bundestag_and_wahlomat"
per_device_train_batch_size = 2

# 1) Load dataset
ds = load_dataset("SinclairSchneider/trainset_political_party_big")
df = ds['train'].to_pandas()

# 2) Prepare labels and mappings
labels = ["AfD", "BÜNDNIS 90/DIE GRÜNEN", "CDU/CSU", "DIE LINKE", "FDP", "SPD"]
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

# 3) Load model and tokenizer
#model_name = "EuroBERT/EuroBERT-2.1B"
#model_name = "EuroBERT/EuroBERT-610m"
#model_name = "EuroBERT/EuroBERT-210m"
#model_name = "meta-llama/Llama-3.2-1B"
#model_name = "Qwen/Qwen2.5-1.5B"
model_name = "google/gemma-2-2b-it"
max_length = 8192
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    output_attentions=False,
    output_hidden_states=False,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16    
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    do_lower_case=False,
    max_length=max_length,
    TOKENIZERS_PARALLELISM=True,
    trust_remote_code=True,
    add_prefix_space=True
)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.pad_token = "[PAD]"
#tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

# 4) Preprocess label columns for multi-label
df["senti_AfD"] = df["senti_AfD"].apply(lambda n: 1 if n == 1 else 0)
df["senti_BUENDNIS_90_DIE_GRUENEN"] = df["senti_BUENDNIS_90_DIE_GRUENEN"].apply(lambda n: 1 if n == 1 else 0)
df["senti_CDU_CSU"] = df["senti_CDU_CSU"].apply(lambda n: 1 if n == 1 else 0)
df["senti_DIE_LINKE"] = df["senti_DIE_LINKE"].apply(lambda n: 1 if n == 1 else 0)
df["senti_FDP"] = df["senti_FDP"].apply(lambda n: 1 if n == 1 else 0)
df["senti_SPD"] = df["senti_SPD"].apply(lambda n: 1 if n == 1 else 0)

# Convert label columns to list of lists (multi-hot)
labels_data = list(zip(
    df["senti_AfD"].values.astype(float),
    df["senti_BUENDNIS_90_DIE_GRUENEN"].values.astype(float),
    df["senti_CDU_CSU"].values.astype(float),
    df["senti_DIE_LINKE"].values.astype(float),
    df["senti_FDP"].values.astype(float),
    df["senti_SPD"].values.astype(float)
))

# 5) Train-test split
sentences = df["text"].values
train_texts, test_texts, train_labels, test_labels = train_test_split(
    sentences, labels_data, test_size=0.2
)

# 6) Create an on-the-fly dataset
class PoliticsOnTheFlyDataset(torch.utils.data.Dataset):
    """
    This dataset will tokenize each sample on the fly
    in __getitem__, thus avoiding holding all tokenized data in memory.
    """
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

        # Perform tokenization here
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

        # Convert to PyTorch tensors
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float)

        return item

# Instantiate the on-the-fly datasets
train_dataset = PoliticsOnTheFlyDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = PoliticsOnTheFlyDataset(test_texts, test_labels, tokenizer, max_length)

# 7) Define the metrics function
def compute_metrics(pred: EvalPrediction, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': acc,
        'roc_auc': roc_auc
    }

wandb.init(project="politic_alternative_models", entity="unibw", name=run_name) #auskommentiert wegen fehlendem Internt
metric_name = "f1"

# 8) Training arguments
output_dir = "./"+run_name
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    disable_tqdm=False,
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_steps=8,
    fp16=False,
    bf16=True,
    dataloader_num_workers=8,
    metric_for_best_model=metric_name,
    report_to="wandb",
    run_name=run_name
)

# 9) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# 10) Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)