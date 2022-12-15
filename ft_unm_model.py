import os

from datasets import load_dataset
from torch import nn
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, RobertaConfig,
                          RobertaForMaskedLM, Trainer, TrainingArguments)

raw_dataset= load_dataset("imdb")
# raw_dataset= load_dataset("sst")
# model = AutoModelForMaskedLM.from_config(config)
# def dataset_mapping(x):
#         return {
#             "sentence": x["sentence"],
#             "label": 1 if x["label"] > 0.5 else 0,
#         }
# raw_dataset = raw_dataset.map(function=dataset_mapping)
model = AutoModelForMaskedLM.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=raw_dataset["train"].column_names,
)

block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

# tokenizer.add_special_tokens({'eos_token': '[EOS]'})
# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# model = nn.DataParallel(model.cuda(), device_ids=[0,1])
training_args = TrainingArguments(
    
    output_dir="./results/imdb_ro/",
    save_strategy='epoch',
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=20,
    weight_decay=0.01,
    # per_device_train_batch_size=32,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)


trainer.train()


