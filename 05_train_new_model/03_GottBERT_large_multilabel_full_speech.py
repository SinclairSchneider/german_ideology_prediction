from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
import wandb #auskommentiert wegen fehlendem Internt
import numpy as np
import pandas as pd

run_name = "politic_GottBERT_large_multilabel_bundestag_and_wahlomat"

#ds = load_dataset("SinclairSchneider/Bundestagsreden_senti_pos_neg")
#df = ds['train'].to_pandas()
#df = df.query("Redner_Partei_oder_Rolle in ('CDU/CSU', 'SPD', 'AfD', 'FDP', 'BÜNDNIS 90/DIE GRÜNEN', 'DIE LINKE')")
#df = pd.read_json("trainset_combined.json")
ds = load_dataset("SinclairSchneider/trainset_political_party_big")
df = ds['train'].to_pandas()

labels = ["AfD", "BÜNDNIS 90/DIE GRÜNEN", "CDU/CSU", "DIE LINKE", "FDP", "SPD"]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

#model_name = "ikim-uk-essen/geberta-base"
#model_name = "ikim-uk-essen/geberta-xlarge"
model_name = "TUM/GottBERT_large"
max_length = 512
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = len(labels), output_attentions = False, output_hidden_states = False, problem_type="multi_label_classification", id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, max_length = max_length, TOKENIZERS_PARALLELISM=True)

sentences = df.text.values

df["senti_AfD"] = df["senti_AfD"].apply(lambda n: 1 if n==1 else 0).values
df["senti_BUENDNIS_90_DIE_GRUENEN"] = df["senti_BUENDNIS_90_DIE_GRUENEN"].apply(lambda n: 1 if n==1 else 0).values
df["senti_CDU_CSU"] = df["senti_CDU_CSU"].apply(lambda n: 1 if n==1 else 0).values
df["senti_DIE_LINKE"] = df["senti_DIE_LINKE"].apply(lambda n: 1 if n==1 else 0).values
df["senti_FDP"] = df["senti_FDP"].apply(lambda n: 1 if n==1 else 0).values
df["senti_SPD"] = df["senti_SPD"].apply(lambda n: 1 if n==1 else 0).values

labels = list(zip(df.senti_AfD.values.astype(float), df.senti_BUENDNIS_90_DIE_GRUENEN.values.astype(float), df.senti_CDU_CSU.values.astype(float), df.senti_DIE_LINKE.values.astype(float), df.senti_FDP.values.astype(float), df.senti_SPD.values.astype(float)))

train_texts, test_texts, train_labels, test_labels = train_test_split(sentences, labels, test_size=.2)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length = max_length)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True,  max_length = max_length)

class PoliticsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PoliticsDataset(train_encodings, train_labels)
test_dataset = PoliticsDataset(test_encodings, test_labels)

def compute_metrics(pred: EvalPrediction, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = pred.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
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

# define the training arguments
output_dir = "./"+run_name
training_args = TrainingArguments(
    #output_dir = './politic_Deberta-base_multilabel_full_speech',
    output_dir = output_dir,
    num_train_epochs=4,
    #num_train_epochs=8,
    #per_device_train_batch_size = 16,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 16,    
    per_device_eval_batch_size= 8,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-5,
    disable_tqdm = False, 
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_steps = 8,
    #fp16 = True,
    fp16 = False,
    dataloader_num_workers = 8,
    metric_for_best_model=metric_name,
    report_to="wandb", #auskommentiert wegen fehlendem Internt
    #report_to="none",
    #run_name="politic_Deberta-base_multilabel_full_speech"
    run_name=run_name
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.save_model()

tokenizer.save_pretrained(output_dir)