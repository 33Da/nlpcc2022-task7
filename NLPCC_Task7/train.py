from ddparser import DDParser
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
import json
from focal_loss import FocalLoss
from data_utils import data_load, get_dataloader
import os
from transformers import AutoModel, AutoTokenizer,BertTokenizer,BertModel,AlbertModel,ElectraTokenizer, ElectraModel
from model import *
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import shutil
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
import torch.nn.functional as F
from Adversarial import PGD,FGM
from lookahead.lookahead import Lookahead
from ple import PLEModel
from r_dropout import compute_kl_loss

warnings.filterwarnings("ignore")


def get_type(tokenizer,pretrain_model):
    # types = ["性别","地域","职业","少数民族","种族文化"]
    types = ["性别", "地域", "职业", "种族"]
    know_inputs = []
    know_attention = []
    know_type = []
    for t in types:
        know = tokenizer.encode_plus(
            t,  # Target to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            padding='max_length',
            max_length= 6 ,
            return_attention_mask=True,  # Construct attn. masks.
        )
        know_inputs.append(know["input_ids"])
        know_attention.append(know["attention_mask"])
        know_type.append(know["token_type_ids"])

    input_ids = torch.tensor(know_inputs, dtype=torch.long).cuda()
    attention_mask = torch.tensor(know_attention, dtype=torch.long).cuda()
    token_type_ids = torch.tensor(know_type, dtype=torch.long).cuda()
    # pretrain_model = pretrain_model.cuda()
    # t_cls_temp = pretrain_model(input_ids=input_ids,
    #                             attention_mask=attention_mask,
    #                             token_type_ids=token_type_ids
    #                             ).last_hidden_state[:, 0]
    # t_cls_temp[3] = t_cls_temp[3,:] + t_cls_temp[4,:] / 2
    # t_cls = t_cls_temp[:4,:]
    return [input_ids,attention_mask,token_type_ids]

class Graph:
    def __init__(self):
        self.graph = []
        self.concat_graph = []


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
    elif task == 2:
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                                   labels=[0, 1,2,3])
    else:
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                                   labels=[0, 1, 2])

    f1 = f_class.mean()

    precision = p_class.mean()
    recall = r_class.mean()

    return acc, f1, p_class, recall, indices,f_class


def run_classifier(lr,lr_linear,tokenizer,pretrain_model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--model_select", type=str, default="mengzi", help="ernie or BERT")
    parser.add_argument("--train_mode", type=str, default="joined3", help="joined or single or adversarial" )
    parser.add_argument("--lr", type=float, default=lr) # 2e-3
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--test_dev", type=str, default="3:7")
    parser.add_argument("--weight_decay", type=float, default=1e-5)


    args = parser.parse_args()
    alpha = args.alpha
    model_select = args.model_select
    dataset = args.dataset
    train_mode = args.train_mode
    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    dropout = args.dropout
    max_len = args.max_len
    test_dev = args.test_dev
    weight_decay = args.weight_decay

    pretrain_model_name = {"ernie": "nghuyong/ernie-1.0","bert":"bert-base-chinese","roberta":"hfl/chinese-roberta-wwm-ext"
                           ,"albert":"clue/albert_chinese_tiny","electra":"hfl/chinese-electra-180g-small-discriminator","mengzi":"Langboat/mengzi-bert-base"}
    print("train mode:",train_mode)

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

    # if model_select == "ernie":
    #     tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name[model_select])
    #     pretrain_model = AutoModel.from_pretrained(pretrain_model_name[model_select])
    # elif model_select == "albert":
    #     tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
    #     pretrain_model = AlbertModel.from_pretrained("clue/albert_chinese_tiny")
    # elif model_select == "electra":
    #     tokenizer = ElectraTokenizer.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
    #     pretrain_model = ElectraModel.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(pretrain_model_name[model_select])
    #     pretrain_model = BertModel.from_pretrained(pretrain_model_name[model_select])


    # set up the random seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


    # prepare for data
    print("prepare for data")
    train_filename = "data/train/" + dataset + ".csv"
    val_filename = "data/val/" + dataset + ".csv"
    train_data = data_load(train_filename, tokenizer, max_len=max_len)
    test_data = data_load(val_filename, tokenizer, max_len=max_len)
    train_dataloader, test_dataloader, dev_dataloader = get_dataloader(train_data, test_data, batch_size, test_dev)

    # prepare for model
    if train_mode == "joined":
        types = get_type(tokenizer,pretrain_model)
        bisa_model = Bisa_Model_joined_concat(pretrain_model,types,dropout)
        # bisa_model = Bisa_Model_joined(pretrain_model, dropout)
        pgd = PGD(bisa_model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
    elif train_mode == "joined3":
        bisa_model = Bisa_Model_joined_task3(pretrain_model,dropout)
        pgd = PGD(bisa_model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
    elif train_mode == "adversarial":
        bisa_model = Bisa_Model(pretrain_model, dropout)
        # fgm = FGM(bisa_model)
        pgd = PGD(bisa_model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
    elif train_mode == "ple":
        bisa_model = PLEModel(pretrain_model,dropout)
        pgd = PGD(bisa_model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
        train_mode = "joined3"
    elif train_mode == "meta":
        bisa_model = Metamodel(pretrain_model,dropout)
        pgd = PGD(bisa_model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
    else:
        bisa_model = Bisa_Model(pretrain_model, dropout)

    if MULTI_GPU:
        bisa_model = nn.DataParallel(bisa_model, device_ids=device_ids)
    print("load model")
    # bisa_model.load_state_dict(torch.load(
    #     "/run/media/lab1510/新加卷/lab1510/NLPCC_Task7/static_dict/epoch_13_f1_0.63082_acc_0.66831_recall_0.59826.pkl"))
    bisa_model.to(device)
    # ple 2e-4
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.bert.encoder')], 'lr': lr},
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.bert.pooler')], 'lr': lr},
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.linear')], 'lr': 1e-3},
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.expert')], 'lr': 1e-3},
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.out')], 'lr': 1e-3},
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.tower')], 'lr': lr},
    #     {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.gate')], 'lr': lr},
    # ]
    # 2e-3
    optimizer_grouped_parameters = [
       # {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.bert.embeddings')], 'lr': lr},
        {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.bert.encoder')], 'lr': lr},
        {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.bert.pooler')], 'lr': lr},
        {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.linear')], 'lr': 1e-3},
        {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.lstm')], 'lr': 1e-3},
        {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.out')], 'lr': lr_linear}, # 2e-4
        {'params': [p for n, p in bisa_model.named_parameters() if n.startswith('module.gcn')], 'lr': lr},
    ]
    loss_function2 = FocalLoss(num_class=4)
    loss_function = FocalLoss(num_class=4)
    loss_function3 = FocalLoss(num_class=3)


    # loss_function = nn.CrossEntropyLoss(weight = torch.Tensor([1 / 9855, 1 / 354, 1 / 6635, 1 / 5825])).to(device)
    # loss_function2 = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters,weight_decay=weight_decay)
    optimizer = torch.optim.SGD(optimizer_grouped_parameters)

    # schedluer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=32)

    # lookahead = Lookahead(optimizer, k=5, alpha=0.5)  # 初始化Lookahead


    # train
    epoch_loss = []
    all_f1, all_acc, all_recall = [], [], []
    max_f1,model_path = 0,""
    for epoch in range(0, total_epoch):
        print('Epoch:', epoch)
        train_loss, valid_loss = [], []
        bisa_model.train()
        train_batchs = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                            postfix={"current random seed": seed, "epoch": epoch})
        for b_index, data in train_batchs:
            label = data[-2]
            label2 = data[-1]
            label3 = data[-3]

            if train_mode == "joined3":
                optimizer.zero_grad()
                output, output2,output3 = bisa_model(data)
                loss1 = loss_function(output, label)
                loss2 = loss_function2(output2, label2)
                loss3 = loss_function3(output3, label3)

                loss = 4 * loss1 + loss2 + loss3
                loss.backward()
                # nn.utils.clip_grad_norm_(bisa_model.parameters(), 1)
                optimizer.step()

                # adversarial
                # output, output2,output3 = bisa_model(data)
                #
                # loss1 = loss_function(output, label)
                # loss2 = loss_function2(output2, label2)
                # loss3 = loss_function3(output3, label3)
                # loss = 3 * loss1 + loss2 + loss3
                # loss.backward()  # 反向传播，得到正常的grad
                # pgd.backup_grad()
                # K = 3
                # for t in range(K):
                #     pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                #     if t != K - 1:
                #         optimizer.zero_grad()
                #     else:
                #         pgd.restore_grad()
                #     output,output2,output3 = bisa_model(data)
                #     loss_adv1 = loss_function(output, label)
                #     loss_adv2 = loss_function2(output2, label2)
                #     loss_adv3 = loss_function3(output3, label3)
                #     loss_adv = 3 * loss_adv1 + loss_adv2 + loss_adv3
                #     loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # pgd.restore()  # 恢复embedding参数
                # # 梯度下降，更新参数
                # optimizer.step()
                # optimizer.zero_grad()
            elif train_mode == "meta":
                for epoch in range(total_epoch):
                    total_loss = 0.
                    nb_sample = 0
                    # train
                    bisa_model.train()
                    for step, batch in enumerate(train_dataloader):
                        input_ids, seg_ids, attention_masks, _, _, _, _, _ = data

                        mini_batch = input_ids.shape[0] / 2
                        input_ids_1 = input_ids[:mini_batch,:,:]
                        seg_ids_1 = seg_ids[:mini_batch,:,:]
                        attention_masks_1 = attention_masks[:mini_batch,:,:]
                        label_1 = label[:mini_batch,:,:]
                        data1 = [input_ids_1,seg_ids_1,attention_masks_1,label_1]

                        input_ids_2 = input_ids[mini_batch:, :, :]
                        seg_ids_2 = input_ids[mini_batch:, :, :]
                        attention_masks_2 = input_ids[mini_batch:, :, :]
                        label_2 = label[mini_batch:, :, :]
                        data2 = [input_ids_2, seg_ids_2, attention_masks_2, label_2]


                        # if (step + 1) % 2 == 0:
                        loss = bisa_model.global_update(data1,data2)

                        optimizer.zero_grad()
                        loss.backward()

            elif train_mode == "joined":
                optimizer.zero_grad()
                output,output2 = bisa_model(data)
                loss1 = loss_function(output, label)
                loss2 = loss_function2(output2,label2)
                loss = 2 * loss1 + loss2
                loss.backward(retain_graph=True)
                # nn.utils.clip_grad_norm_(bisa_model.parameters(), 1)
                optimizer.step()

                # 正常训练 pgd
                # output,output2 = bisa_model(data)
                #
                # loss1= loss_function(output, label)
                # loss2 = loss_function2(output2, label2)
                # loss = 2 * loss1 + loss2
                # loss.backward()  # 反向传播，得到正常的grad
                # pgd.backup_grad()
                #
                # # 对抗训练
                # K = 3
                # for t in range(K):
                #     pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                #     if t != K - 1:
                #         optimizer.zero_grad()
                #     else:
                #         pgd.restore_grad()
                #     output,output2 = bisa_model(data)
                #     loss_adv1 = loss_function(output, label)
                #     loss_adv2 = loss_function2(output2, label2)
                #     loss_adv = 2 * loss_adv1 + loss_adv2
                #     loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # pgd.restore()  # 恢复embedding参数
                # # 梯度下降，更新参数
                # optimizer.step()
                # optimizer.zero_grad()
            elif train_mode == "adversarial":
                # optimizer.zero_grad()

                # FGM
                # output = bisa_model(data)
                # loss = loss_function(output,label)
                # loss.backward()  # 反向传播，得到正常的grad
                # 对抗训练
                # fgm.attack()  # 在embedding上添加对抗扰动
                # output_adv = bisa_model(data)
                # loss_adv = loss_function(output_adv,label)
                # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # fgm.restore()  # 恢复embedding参数
                # # 梯度下降，更新参数
                # optimizer.step()
                # optimizer.zero_grad()


                # 正常训练 pgd
                output = bisa_model(data)
                loss = loss_function(output, label)
                loss.backward()  # 反向传播，得到正常的grad
                pgd.backup_grad()
                # 对抗训练
                K = 3
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        optimizer.zero_grad()
                    else:
                        pgd.restore_grad()
                    output = bisa_model(data)
                    loss_adv = loss_function(output, label)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数
                # 梯度下降，更新参数
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                output = bisa_model(data)
                loss = loss_function(output, label)
                loss.backward()
                # lookahead.step()
                # nn.utils.clip_grad_norm_(bisa_model.parameters(), 1)
                optimizer.step()


            _, indices = torch.max(output, 1)
            correct = (indices == label).float()
            train_loss.append(loss.item())
            train_batchs.set_postfix(loss=loss.item(), batch=b_index, acc=(correct.sum() / len(correct)).item())
        # schedluer.step()
        epoch_loss.append(sum(train_loss) / len(train_loss))
        print(f"epoch {epoch} sum loss:{epoch_loss[epoch]}")

        # evaluation on dev set
        bisa_model.eval()
        with torch.no_grad():
            pred_all, label_all = torch.Tensor([]), torch.Tensor([])
            pred_all2, label_all2 = torch.Tensor([]), torch.Tensor([])
            pred_all3, label_all3 = torch.Tensor([]), torch.Tensor([])
            for b_index, data in enumerate(dev_dataloader):
                label = data[-2]
                label2 = data[-1]
                label3 = data[-3]
                if train_mode == "joined3":
                    pred,pred2,pred3 = bisa_model(data)
                    pred_all = torch.cat([pred_all.cuda(),pred])
                    label_all = torch.cat([label_all.cuda(),label])
                    pred_all2 = torch.cat([pred_all2.cuda(), pred2])
                    label_all2 = torch.cat([label_all2.cuda(), label2])
                    pred_all3 = torch.cat([pred_all3.cuda(), pred3])
                    label_all3 = torch.cat([label_all3.cuda(), label3])
                elif train_mode == "joined":
                    pred,pred2 = bisa_model(data)
                    pred_all = torch.cat([pred_all.cuda(),pred])
                    label_all = torch.cat([label_all.cuda(),label])
                    pred_all2 = torch.cat([pred_all2.cuda(), pred2])
                    label_all2 = torch.cat([label_all2.cuda(), label2])
                else:
                    pred = bisa_model(data)
                    label_all = torch.cat([label_all.cuda(), label])
                    pred_all = torch.cat([pred_all.cuda(), pred])


        acc_average, f1_average, precision, recall_average, indices,all_f1_c = compute_f1(pred_all, label_all,1)
        print("val loss:",loss_function(pred_all,label_all))
        print(f"第 {epoch} epoch, Val acc:{acc_average} ,Val f1_average:{f1_average},Val recall:{recall_average}")
        if train_mode == "joined":
            acc_average2, f1_average2, precision2, recall_average2, indices, all_f1_c = compute_f1(pred_all2, label_all2, 2)
            print(f"Task2 best epoch test f1:{f1_average2},acc:{acc_average2},recall:{recall_average2}")
        if train_mode == "joined3":
            acc_average2, f1_average2, precision2, recall_average2, indices, all_f1_c = compute_f1(pred_all2, label_all2, 2)
            print(f"Task2 best epoch test f1:{f1_average2},acc:{acc_average2},recall:{recall_average2}")
            acc_average3, f1_average3, precision3, recall_average3, indices, all_f1_c = compute_f1(pred_all3, label_all3, 3)
            print(f"Task3 best epoch test f1:{f1_average3},acc:{acc_average3},recall:{recall_average3}")

        if f1_average >= max_f1:
            max_f1 = f1_average
            if not os.path.exists(f"static_dict/{lr}_{lr_linear}/"):
                os.mkdir(f"static_dict/{lr}_{lr_linear}/")
            else:
                shutil.rmtree(f"static_dict/{lr}_{lr_linear}/")
                os.mkdir(f"static_dict/{lr}_{lr_linear}/")
            model_path = f"static_dict/{lr}_{lr_linear}/epoch_{epoch}_f1_{f1_average:.5f}_acc_{acc_average:.5f}_recall_{recall_average:.5f}.pkl"
            torch.save(bisa_model.state_dict(),model_path)

    return max_f1
    # best_epoch = all_f1.index(max(all_f1))
    # print(
    #     f"best epoch {best_epoch} dev,acc:{all_acc[best_epoch]} ,f1:{all_f1[best_epoch]},recall:{all_recall[best_epoch]}")
    #
    # test
    # print("#"*30)
    # if train_mode == "joined":
    #     model = Bisa_Model_joined(pretrain_model, dropout).to(device)
    # else:
    #     model = Bisa_Model(pretrain_model, dropout).to(device)
    #
    # if MULTI_GPU:
    #     model = nn.DataParallel(model, device_ids=device_ids)
    #
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # with torch.no_grad():
    #     acc_list, f1_list, recall_list = [], [], []
    #     acc_list2,f1_list2,recall_list2 = [],[],[]
    #     for b_index, data in enumerate(test_dataloader):
    #         label = data[-2]
    #         label2 = data[-1]
    #
    #         if train_mode == "joined":
    #             pred,pred2 = model(data)
    #             acc2, f12, precision2, recall2, indices2,f_class = compute_f1(pred2, label2,2)
    #             acc_list2.append(acc2)
    #             f1_list2.append(f12)
    #             recall_list2.append(recall2)
    #         else:
    #             pred = model(data)
    #
    #         acc, f1, precision, recall, indices,f_class = compute_f1(pred, label,1)
    #
    #         acc_list.append(acc)
    #         f1_list.append(f1)
    #         recall_list.append(recall)
    #     f1_average = sum(f1_list) / len(f1_list)
    #     acc_average = sum(acc_list) / len(acc_list)
    #     recall_average = sum(recall_list) / len(recall_list)
    #     if train_mode == "joined":
    #         f1_average2 = sum(f1_list2) / len(f1_list2)
    #         acc_average2 = sum(acc_list2) / len(acc_list2)
    #         recall_average2 = sum(recall_list2) / len(recall_list2)
    # print(f"best epoch test f1:{f1_average},acc:{acc_average},recall:{recall_average}")
    # if train_mode == "joined":
    #     print(f"Task2 best epoch test f1:{f1_average2},acc:{acc_average2},recall:{recall_average2}")




if __name__ == '__main__':
    lr = [1e-3,2e-3,1e-4,2e-4,2e-5,2e-6]
    lr_linear = [1e-3,2e-3,1e-4,2e-4,2e-5,2e-6]
    f1_list = []
    for l in lr:
        for ll in lr_linear:
            print("#"*10)
            tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
            pretrain_model = BertModel.from_pretrained("Langboat/mengzi-bert-base")
            print(f"get {l} and {ll}")
            f1 = run_classifier(l,ll,tokenizer,pretrain_model)
            f1_list.append((l,ll,f1))
            print(f"{l} and {ll} result:{f1}")
    print(f1_list)

