import torch
import torch.nn as nn
import argparse
from data_utils import data_load, get_test_dataloader
import os
from transformers import AutoModel, AutoTokenizer,BertTokenizer,BertModel,AlbertModel,ElectraTokenizer, ElectraModel
from model import Bisa_Model,Bisa_Model_joined
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

# Evaluation
def compute_f1(preds, y):
    _, indices = torch.max(preds, 1)

    correct = (indices == y).float()
    acc = correct.sum() / len(correct)  # compute accuracy

    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    # print(preds)
    # 准确率，召回率，f1
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                               labels=[0, 1, 2, 3])

    f1 = f_class.mean()
    precision = p_class.mean()
    recall = r_class.mean()

    return acc, f1, p_class, recall, indices

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--model_select", type=str, default="roberta", help="ernie or BERT")
    parser.add_argument("--train_mode", type=str, default="joined", help="joined or single")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--model_path", type=str, default="/media/lab1510/新加卷/lab1510/NLPCC_Task7/models_result/all/roberta/epoch_168_f1_0.54194_acc_0.65075_recall_0.56345.pkl")

    args = parser.parse_args()
    model_path = args.model_path
    model_select = args.model_select
    dataset = args.dataset
    batch_size = args.batch_size
    dropout = args.dropout
    max_len = args.max_len

    pretrain_model_name = {"ernie": "nghuyong/ernie-1.0","bert":"bert-base-chinese","roberta":"hfl/chinese-roberta-wwm-ext"
                           ,"albert":"clue/albert_chinese_tiny","electra":"hfl/chinese-electra-180g-small-discriminator"}



    USE_MULTI_GPU = True

    # 检测机器是否有多张显卡
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        MULTI_GPU = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        device_ids = [0, 1]
    else:
        MULTI_GPU = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_select == "ernie":
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name[model_select])
        pretrain_model = AutoModel.from_pretrained(pretrain_model_name[model_select])
    elif model_select == "albert":
        tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
        pretrain_model = AlbertModel.from_pretrained("clue/albert_chinese_tiny")
    elif model_select == "electra":
        tokenizer = ElectraTokenizer.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
        pretrain_model = ElectraModel.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
    else:
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_name[model_select])
        pretrain_model = BertModel.from_pretrained(pretrain_model_name[model_select])


    model = Bisa_Model(pretrain_model, dropout).to(device)
    if MULTI_GPU:
        model = nn.DataParallel(model, device_ids=device_ids)

    # prepare for data
    val_filename = "data/val/" + dataset + ".csv"
    test_data = data_load(val_filename, tokenizer, max_len=max_len)
    test_dataloader = get_test_dataloader(test_data, batch_size)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        acc_list, f1_list, recall_list = [], [], []
        for b_index, data in enumerate(test_dataloader):
            print(b_index)
            pred = model(data)
            label = data[-2]
            acc, f1, precision, recall, indices = compute_f1(pred, label)
            acc_list.append(acc)
            f1_list.append(f1)
            recall_list.append(recall)
        f1_average = sum(f1_list) / len(f1_list)
        acc_average = sum(acc_list) / len(acc_list)
        recall_average = sum(recall_list) / len(recall_list)
    print(f"best epoch test f1:{f1_average},acc:{acc_average},recall:{recall_average}")



if __name__ == '__main__':
    test()