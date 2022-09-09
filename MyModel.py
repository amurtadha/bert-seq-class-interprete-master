
import numpy as np
import torch
import torch.nn as nn
from transformers import  AutoModel, AutoConfig, AutoModelForSequenceClassification


class FineTuning(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(FineTuning, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')
        # layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
        #           nn.Linear(hidden_size, args.lebel_dim)]

        # layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        # self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return outputs['logits'], None, None

