
import textattack
import torch
from textattack.models.helpers import (GloveEmbeddingLayer,
                                       LSTMForClassification,
                                       WordCNNForClassification)
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.models.tokenizers import glove_tokenizer
from textattack.shared import utils
from torch import nn as nn

#load cnn
model1 = WordCNNForClassification.from_pretrained('cnn-sst2')
# model1 = WordCNNForClassification.from_pretrained('cnn-ag-news')
# model1 = WordCNNForClassification.from_pretrained('cnn-imdb')

cnn_model = WordCNNForClassification()
tokenizer = cnn_model.tokenizer
model = textattack.models.wrappers.PyTorchModelWrapper(model1, tokenizer)

