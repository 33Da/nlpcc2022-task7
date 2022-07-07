import torch
import torch.nn as nn
from transformers import BertPreTrainedModel,BertModel,BertForSequenceClassification

import torch.nn.functional as F

class BertForSequence(nn.Module):

    def __init__(self,model,config):
        super(BertForSequence, self).__init__()
        self.num_labels = config.num_labels
        model.config.output_hidden_states = True
        self.bert = model

        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input):
        outputs = self.bert(**input)
        cls_hidden = outputs[2][-1][:,0]

        logits1 = outputs[1]
        logits2 = self.classifier(cls_hidden)


        return logits1,logits2



class BertForSequence2(nn.Module):

    def __init__(self,model,config):
        super(BertForSequence2, self).__init__()
        self.num_labels = config.num_labels
        model.config.output_hidden_states = True
        self.bert = model

        self.classifier = nn.Linear(config.hidden_size, 4)

    def forward(self, input):
        outputs = self.bert(**input)
        cls_hidden = outputs[2][-1][:,0]

        logits1 = outputs[1]
        logits2 = self.classifier(cls_hidden)


        return logits1,logits2