import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate
import numpy as np 


bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
print("Loading dataset...")
dataset = load_dataset("csv", data_files=r"C:\Users\Alienware\Documents\Project\RoBERTA-ViAttention\merged_vIErr2_and_ViLexNorm.csv")
print("Tokenizing dataset...")
model_name = "vinai/bartpho-syllable"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    em_score = np.mean([1 if pred == label else 0 for pred, label in zip(decoded_preds, decoded_labels)])

    return {
        "bleu": bleu_score["score"],
        "rougeL": rouge_score["rougeL"],
        "exact_match": em_score * 100
    }

def preprocess_function(examples):
    model_inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=50)
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=50)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

class Bart_LSTM(nn.Module):
    def __init__(self, model_name, hidden_dim=768, num_layers=2, output_dim=768):
        super(Bart_LSTM, self).__init__()
        self.bartpho = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BartPho
            bart_output = self.bartpho(input_ids=input_ids, attention_mask=attention_mask)
        
        lstm_output, _ = self.lstm(bart_output.last_hidden_state)  # (batch, seq_len, hidden_dim*2)
        output = self.fc(lstm_output)
        return output  

model = Bart_LSTM(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("test", None),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics  
)

trainer.train()
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
