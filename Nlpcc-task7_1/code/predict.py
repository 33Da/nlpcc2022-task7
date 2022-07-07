import torch
import torch.nn as nn
import argparse
import os
from transformers import BertForSequenceClassification,BertTokenizer,BertConfig
from models import BertForSequence,BertForSequence2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import warnings
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from utils.data_utils import (PROCESSORS, multi_classification_convert_examples_to_dataset)
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
# Evaluation
def compute_f1(preds, y,task=1):
    preds = F.softmax(preds)
    _, indices = torch.max(preds, 1)

    correct = (indices == y).float()
    acc = correct.sum() / len(correct)  # compute accuracy

    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    # print(preds)
    # 准确率，召回率，f1
    if task == 1:
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                               labels=[0, 1, 2, 3])
    else:
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                                   labels=[0, 1, 2,3])
    f1 = f_class.mean()
    precision = p_class.mean()
    recall = r_class.mean()

    return acc, f1, p_class, recall, indices,f_class


def load_and_cache_examples(args, tokenizer, set_type='dev'):

    processor = PROCESSORS['pair']()
    examples = processor.get_examples('../data/Dataset/', set_type)
    label_list = processor.get_labels()
    if isinstance(label_list, dict):
        label_list = list(label_list.keys())
    dataset = multi_classification_convert_examples_to_dataset(
        examples,
        tokenizer,
        max_length=120,
        label_list=label_list,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        threads=10,
        set_type=set_type
    )
    return dataset


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--model_select", type=str, default="Langboat/mengzi-bert-base", help="ernie or BERT")
    parser.add_argument("--train_mode", type=str, default="joined", help="joined or single")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=2837)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--model_path1", type=str, default="../static_dict/model1.pkl")
    parser.add_argument("--model_path2", type=str, default="../static_dict/model.pkl")

    args = parser.parse_args()
    model_path1 = args.model_path1
    model_path2 = args.model_path2
    model_select = args.model_select


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = BertConfig.from_pretrained(
        model_select,
        num_labels=4
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_select,
        do_lower_case=True,
    )
    pre_model1 = BertForSequenceClassification.from_pretrained(
        model_select,
        config = config
    )
    model1 = BertForSequence(pre_model1, config)

    pre_model2 = BertForSequenceClassification.from_pretrained(
        model_select,
        config=config
    )
    model2 = BertForSequence2(pre_model2,config).to(device)


    print("loading model")

    static_dict1 = torch.load(model_path1)
    new_static_dict = {}
    for k,v in static_dict1.items():
        new_static_dict[k[7:]] = v
    model1.load_state_dict(new_static_dict)

    new_static_dict = {}
    static_dict2 = torch.load(model_path2)
    for k, v in static_dict2.items():
        new_static_dict[k[7:]] = v
    model2.load_state_dict(new_static_dict)
    model1.to(device)
    model2.to(device)

    model1.eval()
    model2.eval()
    print("eval data")
    dev_dataset = load_and_cache_examples(args, tokenizer, set_type='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=16)
    output = torch.tensor([]).to(device)
    label = torch.tensor([]).to(device)
    with torch.no_grad():
        epoch_iterator = tqdm(dev_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3][:, 0]}
            label1 = batch[3][:, 0]

            output1_1, output1_2 = model1(inputs)
            output2_1, output2_2 = model2(inputs)
            output = torch.cat([output,0.6 * output2_1+0.4 * output1_1])
            label = torch.cat([label,label1])



    result = compute_f1(output,label)
    print(result)



if __name__ == '__main__':
    test()