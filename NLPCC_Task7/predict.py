import torch
import torch.nn as nn
import argparse
from data_utils import test_load, get_test_dataloader
import os
from transformers import AutoModel, AutoTokenizer,BertTokenizer,BertModel,AlbertModel,ElectraTokenizer, ElectraModel
from model import Bisa_Model,Bisa_Model_joined,Bisa_Model_joined_task3
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import warnings
from ple import PLEModel
import pandas as pd
warnings.filterwarnings("ignore")

# Evaluation
def compute_f1(preds, y,task):
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

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--model_select", type=str, default="mengzi", help="ernie or BERT")
    parser.add_argument("--train_mode", type=str, default="joined", help="joined or single")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=2837)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--model_path", type=str, default="/run/media/lab1510/新加卷/lab1510/NLPCC_Task7/models_result/all/joined/mengzi/3/epoch_13_f1_0.63082_acc_0.66831_recall_0.59826.pkl")

    args = parser.parse_args()
    model_path = args.model_path
    model_select = args.model_select
    dataset = args.dataset
    batch_size = args.batch_size
    dropout = args.dropout
    max_len = args.max_len
    train_mode = args.train_mode

    pretrain_model_name = {"ernie": "nghuyong/ernie-1.0","bert":"bert-base-chinese","roberta":"hfl/chinese-roberta-wwm-ext"
                           ,"albert":"clue/albert_chinese_tiny","electra":"hfl/chinese-electra-180g-small-discriminator","mengzi":"Langboat/mengzi-bert-base"}



    USE_MULTI_GPU = False

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

    if train_mode == "joined":
        model = Bisa_Model_joined_task3(pretrain_model, dropout).to(device)
    elif train_mode == "ple":
        model = PLEModel(pretrain_model,dropout).to(device)
        train_mode = "joined"
    else:
        model = Bisa_Model(pretrain_model, dropout).to(device)
    if MULTI_GPU:
        model = nn.DataParallel(model, device_ids=device_ids)

    # prepare for data
    val_filename = "data/test/test.csv"
    test_data = test_load(val_filename, tokenizer, max_len=max_len)
    test_dataloader = get_test_dataloader(test_data, batch_size)

    print("loading model:",model_path.split("/")[-1])
    if not USE_MULTI_GPU:
        from collections import OrderedDict
        static_dict = torch.load(model_path)
        new_static_dict = OrderedDict()
        for k,v in static_dict.items():
            name = k[7:]
            new_static_dict[name] = v
        model.load_state_dict(new_static_dict)
    else:
        model.load_state_dict(torch.load(model_path))

    model.eval()
    print("eval data")
    with torch.no_grad():
        pre= []
        for b_index, data in enumerate(test_dataloader):
            if train_mode == "joined":
                pred,pred2,pred3 = model(data)
                preds = F.softmax(pred)
                _, indices = torch.max(preds, 1)
                pre += indices.tolist()
            else:
                pred = model(data)
                preds = F.softmax(pred)
                _, indices = torch.max(preds, 1)
                pre += indices.tolist()


    # write data
    print("writing data")
    source = pd.read_csv(val_filename)
    idx = source['idx'].values.tolist()
    text1 = source['q'].values.tolist()
    text2 = source['a'].values.tolist()

    predict_data = pd.DataFrame({
        "idx":idx,
        "text1":text1,
        "text2":text2,
        "label":pre
    })


    predict_data.to_csv("13_BiasEval.csv",index=False)








if __name__ == '__main__':
    test()