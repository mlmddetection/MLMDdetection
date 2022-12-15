import textattack
import torch
from textattack.models.helpers import (GloveEmbeddingLayer,
                                       LSTMForClassification,
                                       WordCNNForClassification)
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.models.tokenizers import glove_tokenizer
from textattack.shared import utils
from torch import nn as nn


# #load lstm
model1 = LSTMForClassification.from_pretrained('lstm-sst2')
# model1 = LSTMForClassification.from_pretrained('lstm-ag-news')
# model1 = LSTMForClassification.from_pretrained('lstm-imdb')
lstm_model = LSTMForClassification()
tokenizer = lstm_model.tokenizer
model = textattack.models.wrappers.PyTorchModelWrapper(model1, tokenizer)