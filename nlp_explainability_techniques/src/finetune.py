from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from src.preprocess import get_train_dev_test_data, tokenize


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

train, val, test = get_train_dev_test_data()

train_dataset = tokenize(train, tokenizer)
eval_dataset = tokenize(val, tokenizer)
test_Dataset = tokenize(test, tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=4
)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("pretrained_model/")
