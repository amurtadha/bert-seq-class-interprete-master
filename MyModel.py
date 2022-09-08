
import numpy as np
import torch
import torch.nn as nn
from transformers import  AutoModel, AutoConfig, AutoModelForSequenceClassification
# from transformers.models.roberta.modeling_roberta import  shift
from transformers.models.bart.modeling_bart import  shift_tokens_right
from torch.autograd import Function

# import torch.nn.functional as F


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class Pure_labse_(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(Pure_labse_, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.enocder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.enocder.to('cuda')
        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):

        input_ids,token_type_ids, attention_mask = inputs[:3]
        outputs = self.enocder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0, :]

        logits = self.classifier(pooled_output)

        return logits

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
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# class Model_ADS(nn.Module):
#
#     def __init__(self, args, hidden_size=256):
#         super(Model_ADS, self).__init__()
#         config = AutoConfig.from_pretrained(args.pretrained_bert_name)
#         config.num_labels = args.lebel_dim
#         self.encoder= AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
#         # self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
#         # self.encoder.to('cuda')
#         layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
#                   nn.Linear(hidden_size, args.lebel_dim)]
#         # layers_rev = [nn.Linear(config.hidden_size, 2)]
#         layers_rev = [nn.Linear(config.hidden_size, args.lebel_dim)]
#         self.classifier = nn.Sequential(*layers)
#         self.discriminator = nn.Sequential(*layers_rev)
#
#     def forward(self, inputs, alpha=0.0):
#         input_ids, token_type_ids, attention_mask = inputs[:3]
#         outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#
#         # logits_rev =
#         # return self.classifier(outputs['pooler_output']), outputs['pooler_output']
#         pooled_output= outputs['pooler_output']
#         reverse_feature = ReverseLayerF.apply(pooled_output, alpha)
#         logits_revers = self.discriminator(reverse_feature)
#         return self.classifier(pooled_output), pooled_output, logits_revers
#         # return self.classifier(pooled_output), pooled_output, None
class Model_GANS(nn.Module):
    def __init__(self, args, hidden_size=256):
        super(Model_GANS, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder= AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        # self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        # self.encoder.to('cuda')
        layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
                  nn.Linear(hidden_size, args.lebel_dim)]
        # layers_rev = [nn.Linear(config.hidden_size, 2)]
        layers_rev = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.classifier = nn.Sequential(*layers)
        self.discriminator = nn.Sequential(*layers_rev)

    def forward(self, inputs, alpha=0.0):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # logits_rev =
        # return self.classifier(outputs['pooler_output']), outputs['pooler_output']
        pooled_output= outputs['pooler_output']
        # reverse_feature = ReverseLayerF.apply(pooled_output, alpha)
        logits_Gan = self.discriminator(pooled_output)
        return self.classifier(pooled_output), pooled_output, logits_Gan
        # return self.classifier(pooled_output), pooled_output, None
class Model_contrastive(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(Model_contrastive, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder= AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        config.output_hidden_states=True
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.config.output_hidden_states=True
        # self.encoder.to('cuda')
        # layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
        #           nn.Linear(hidden_size, args.lebel_dim)]
        # layers_rev = [nn.Linear(config.hidden_size, 2)]
        # layers_rev = [nn.Linear(config.hidden_size, args.lebel_dim)]
        # self.classifier = nn.Sequential(*layers)
        # self.discriminator = nn.Sequential(*layers_rev)

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, inputs, alpha=0.0):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs= self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logit, x= outputs[0],outputs[1]
        pooled_output = x[-1][:, 0, :]
        # x = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # logit = self.out_proj(x)

        # logits_rev =
        # return self.classifier(outputs['pooler_output']), outputs['pooler_output']
        # pooled_output= outputs['pooler_output']
        # reverse_feature = ReverseLayerF.apply(pooled_output, alpha)
        # logits_revers = self.discriminator(reverse_feature)
        # return self.classifier(pooled_output), pooled_output, logits_revers
        # return self.classifier(pooled_output), pooled_output
        return logit, None, pooled_output
class Model_contrastive_ads(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(Model_contrastive_ads, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder= AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        config.output_hidden_states=True
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.config.output_hidden_states=True
        # self.encoder.to('cuda')
        # layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
        #           nn.Linear(hidden_size, args.lebel_dim)]
        layers_rev = [nn.Linear(config.hidden_size, 2)]
        # layers_rev = [nn.Linear(config.hidden_size, args.lebel_dim)]
        # self.classifier = nn.Sequential(*layers)
        self.discriminator = nn.Sequential(*layers_rev)

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, inputs, alpha=0.0):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs= self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logit, x= outputs[0],outputs[1]
        pooled_output = x[-1][:, 0, :]
        # x = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # logit = self.out_proj(x)

        # logits_rev =
        # return self.classifier(outputs['pooler_output']), outputs['pooler_output']
        # pooled_output= outputs['pooler_output']
        reverse_feature = ReverseLayerF.apply(pooled_output, alpha)
        logits_revers = self.discriminator(reverse_feature)
        # return self.classifier(pooled_output), pooled_output, logits_revers
        # return self.classifier(pooled_output), pooled_output
        return logit, logits_revers, pooled_output
class Model_GANS(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(Model_GANS, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder= AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        config.output_hidden_states=True
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.config.output_hidden_states=True
        # self.encoder.to('cuda')
        # layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
        #           nn.Linear(hidden_size, args.lebel_dim)]
        layers_rev = [nn.Linear(config.hidden_size, 2)]
        # layers_rev = [nn.Linear(config.hidden_size, args.lebel_dim)]
        # self.classifier = nn.Sequential(*layers)
        self.discriminator = nn.Sequential(*layers_rev)

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, inputs, alpha=0.0):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs= self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logit, x= outputs[0],outputs[1]
        pooled_output = x[-1][:, 0, :]
        # x = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # logit = self.out_proj(x)

        # logits_rev =
        # return self.classifier(outputs['pooler_output']), outputs['pooler_output']
        # pooled_output= outputs['pooler_output']
        # reverse_feature = ReverseLayerF.apply(pooled_output, alpha)
        logits_revers = self.discriminator(pooled_output)
        # return self.classifier(pooled_output), pooled_output, logits_revers
        # return self.classifier(pooled_output), pooled_output
        return logit, logits_revers, pooled_output

