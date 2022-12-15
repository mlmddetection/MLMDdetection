import importlib
import json
import os
import time

import textattack
import transformers
from datasets import load_dataset
from textattack.models.helpers import (LSTMForClassification,
                                       WordCNNForClassification)

def main():

    tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    # tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    # tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/distilbert-base-cased-snli")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-cased-snli")
    # tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    dataset = textattack.datasets.HuggingFaceDataset("imdb", None, "test")
    # attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    # attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)
    # attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    # attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    # attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)


    dataset_name = 'ag_news'
    attack_name = 'TextFooler'
    victim = 'bert'

    attack_args = textattack.AttackArgs(num_examples=3500,
                                        log_to_txt="./attack_log/" + dataset_name + '/' + attack_name + '_' + victim + '_log.txt', disable_stdout=True,parallel=True)#

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

if __name__ == '__main__':
    main()

