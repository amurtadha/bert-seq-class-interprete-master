
import argparse
import os
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

from captum.attr import IntegratedGradients
# from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
# from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from collections import OrderedDict
from data_utils import Process_baseline
from tqdm import tqdm
import json



class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
        self.model = BertForSequenceClassification.from_pretrained(opt.pretrained_bert_name)
        self.Label = {1: 'positive', 0: 'negative'}
        self.labels = json.load(open('datasets/{0}/labels.json'.format(self.opt.dataset)))
        self.opt.lebel_dim = len(self.labels)

        state_dict = torch.load('state_dict/{}_fn.bm'.format(opt.dataset))
        self.data = Process_baseline('datasets/{0}/train.json'.format(opt.dataset), self.tokenizer, opt.max_seq_len, opt.dataset)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'discriminator' in k: continue
            name = k[8:] if 'encoder.' in k else k  # remove `encoder.`
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.opt.device)
        self.ig = IntegratedGradients(self.forward_model)
        self.vis_data_records_ig = []

    def forward_model(self, inputs=None):
            outputs = self.compute_bert_outputs(inputs)
            pooled_output = outputs[1]
            pooled_output = self.model.dropout(pooled_output)
            logits = self.model.classifier(pooled_output)
            return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)
    def compute_bert_outputs(self, embedding_output, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.model.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.model.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.model.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.model.config.num_hidden_layers

        encoder_outputs = self.model.bert.encoder(embedding_output,
                                             extended_attention_mask,
                                             head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def interpret_sentence(self, sentence, label=1):
        self.model.eval()
        self.model.zero_grad()

        input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)], ).to(self.opt.device)
        input_embedding = self.model.bert.embeddings(input_ids)

        # predict
        pred= self.forward_model(input_embedding).item()
        pred_ind = round(pred)

        # compute attributions and approximation delta using integrated gradients
        attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)

        # print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        self.add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta)

    def add_attributions_to_visualizer(self, attributions, tokens, pred, pred_ind, label, delta):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        attributions=attributions[1:-1]
        # storing couple samples in an array for visualization purposes
        self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
            attributions,
            pred,
            self.Label[pred_ind],
            self.Label[label],
            "label",
            attributions.sum(),
            tokens[1:len(attributions)],
            delta))

    def run_sentence(self):
        for _, d in enumerate(tqdm(self.data)):
            self.interpret_sentence(sentence=d['text'], label=d['label'])
            if _ >10: break

        if False: # use these sentences as tests
            self.interpret_sentence( 'It was a fantastic performance !', label=1)
            self.interpret_sentence( 'Best film ever', label=1)
            self.interpret_sentence( 'Such a great show!', label=1)
            self.interpret_sentence( 'It was a horrible movie', label=0)
            self.interpret_sentence( 'I\'ve never watched something as bad', label=0)
            self.interpret_sentence( 'That is a terrible movie.', label=0)
        print('Visualize attributions based on Integrated Gradients')
        html = visualization.visualize_text(self.vis_data_records_ig)

        with open("outputs/{}.html".format(self.opt.dataset), "w") as f:
            f.write(html.data)
        f.close()

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MR', type=str, help='Corpus-8,Corpus-26, ')
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--device_group', default='1' , type=str, help='e.g. cuda:0')
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    opt = parser.parse_args()



    opt.max_seq_len = {'TNEWS': 128, 'OCNLI': 128, 'IFLYTEK': 128, 'AFQMC': 128, 'YELP': 156, 'TREC': 20, 'yahoo': 256,
                       'ELEC': 256, 'MPQA': 10, 'AG': 50, 'MR': 30, 'SST-2': 30, 'SST-2-small': 30, 'SST-5': 30,
                       'PC': 30, 'CR': 30,
                       'DBPedia': 160, 'IMDB': 220, 'SUBJ': 30,
                       'semeval': 80, 'R8': 207, 'hsumed': 156, 'FAKE': 885}.get(opt.dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_group

    opt.inputs_cols =   ['input_ids', 'segments_ids', 'input_mask', 'label']
    ins = Instructor(opt)
    ins.run_sentence()


if __name__ == '__main__':
    main()
