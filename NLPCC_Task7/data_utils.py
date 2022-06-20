import pandas as pd
from transformers import AutoModel,AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm

import pickle



def load_graph(filename):
    with open(filename,"rb") as f:
        data = pickle.load(f)
    graphs = data.concat_graph
    for index in range(len(graphs)):
        for i in range(len(graphs[index][0])):
            graphs[index][i][i] = 1
    return graphs

def test_load(filename,tokenizer,max_len):
    df = pd.read_csv(filename, encoding="utf-8")
    text1 = df['q'].values.tolist()
    text2 = df['a'].values.tolist()

    q_input_ids = []
    q_seg_ids = []
    q_attention_masks = []
    q_sent_len = []


    train_batchs = tqdm(zip(text1, text2), total=len(text1))
    for t1, t2 in train_batchs:
        encoded_dict = tokenizer.encode_plus(
            t1,  # Target to encode
            t2,  # Sentence to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
        )

        # Add the encoded sentence to the list.
        q_input_ids.append(encoded_dict['input_ids'])
        q_seg_ids.append(encoded_dict['token_type_ids'])
        q_attention_masks.append(encoded_dict['attention_mask'])
        q_sent_len.append(sum(encoded_dict['attention_mask']))

    a_input_ids = [0] * len(text1)
    a_seg_ids = [0] * len(text1)
    datatype = [0] * len(text1)
    label = [0] * len(text1)
    topic = [0] * len(text1)
    depend_matrix = [0] * len(text1)



    return [q_input_ids, q_seg_ids, q_attention_masks, a_input_ids, a_seg_ids, datatype, label, topic, depend_matrix]


def data_load(filename,tokenizer,max_len):
    df = pd.read_csv(filename,encoding="utf-8")
    text1 = df['q'].values.tolist()
    text2 = df['a'].values.tolist()
    label = df["attitude"].values.tolist()
    group = df["group"].values.tolist()
    topic = df["topic"].values.tolist()
    have_next = df["context"].values.tolist()
    datatype = df["datatype"].values.tolist()


    for t_index,t in enumerate(topic):
        if topic[t_index] == "性别":
            topic[t_index] = 0
        elif topic[t_index] == "地域":
            topic[t_index] = 1
        elif topic[t_index] == "职业":
            topic[t_index] = 2
        elif topic[t_index] in ["少数民族","种族文化"]:
            topic[t_index] = 3
        else:
            print(t)
            assert 0 == 1

    q_input_ids = []
    q_seg_ids = []
    q_attention_masks = []
    q_sent_len = []

    a_input_ids = []
    a_seg_ids = []
    a_attention_masks = []
    a_sent_len = []
    depend_matrix = []
    train_batchs = tqdm(zip(text1,text2), total=len(text1))
    for t1,t2 in train_batchs:
        encoded_dict = tokenizer.encode_plus(
            t1,  # Target to encode
            t2,  # Sentence to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
        )

        # Add the encoded sentence to the list.
        q_input_ids.append(encoded_dict['input_ids'])
        q_seg_ids.append(encoded_dict['token_type_ids'])
        q_attention_masks.append(encoded_dict['attention_mask'])
        q_sent_len.append(sum(encoded_dict['attention_mask']))

        # depend_matrix.append(get_tree(t1,t2))
        depend_matrix.append([])

    # same_datatype_label = []
    # for i, j in zip(label, datatype):
    #     if i == j:
    #         same_datatype_label.append(1)
    #     else:
    #         same_datatype_label.append(0)
    # datatype = same_datatype_label




    # for t1,g in zip(text1,group):
    #     encoded_dict = tokenizer.encode_plus(
    #         t1,  # Target to encode
    #         g,  # Sentence to encode
    #         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #         max_length=max_len,  # Pad & truncate all sentences.
    #         padding='max_length',
    #         return_attention_mask=True,  # Construct attn. masks.
    #     )
    #     # Add the encoded sentence to the list.
    #     q_input_ids.append(encoded_dict['input_ids'])
    #     q_seg_ids.append(encoded_dict['token_type_ids'])
    #     q_attention_masks.append(encoded_dict['attention_mask'])
    #     q_sent_len.append(sum(encoded_dict['attention_mask']))

    for t2,g in zip(text2,group):
        encoded_dict = tokenizer.encode_plus(
            t2,  # Target to encode
            g,  # Sentence to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
        )
        # Add the encoded sentence to the list.
        a_input_ids.append(encoded_dict['input_ids'])
        a_seg_ids.append(encoded_dict['token_type_ids'])
        a_attention_masks.append(encoded_dict['attention_mask'])
        a_sent_len.append(sum(encoded_dict['attention_mask']))




    return [q_input_ids,q_seg_ids,q_attention_masks,a_input_ids,a_seg_ids,datatype,label,topic,depend_matrix]



def get_dataloader(train_data,test_dev_data,batch_size,test_dev):
    # train
    train_input_ids = torch.tensor(train_data[0], dtype=torch.long).cuda()
    train_seg_ids = torch.tensor(train_data[1], dtype=torch.long).cuda()
    train_atten_masks = torch.tensor(train_data[2], dtype=torch.long).cuda()
    train_input_ids_a = torch.tensor(train_data[3], dtype=torch.long).cuda()
    train_seg_ids_a = torch.tensor(train_data[4], dtype=torch.long).cuda()
    train_atten_masks_a = torch.tensor(train_data[5], dtype=torch.long).cuda()
    train_y = torch.tensor(train_data[6], dtype=torch.long).cuda()
    train_datatype = torch.tensor(train_data[7], dtype=torch.long).cuda()


    depend_matrix = torch.tensor(load_graph("/run/media/lab1510/新加卷/lab1510/NLPCC_Task7/data/graph/train_all_graph.txt"), dtype=torch.float).cuda()

    train_tensor_loader = TensorDataset(train_input_ids,train_seg_ids,train_atten_masks,depend_matrix,train_seg_ids_a
                                        ,train_atten_masks_a,train_y,train_datatype)
    train_dataloader = DataLoader(train_tensor_loader,  batch_size=batch_size)




    # test
    test_dev = list(map(int,test_dev.split(":")))
    test_nums = int(len(test_dev_data[0]) * (test_dev[0] / sum(test_dev)))

    test_input_ids = torch.tensor(test_dev_data[0][:test_nums], dtype=torch.long).cuda()
    test_seg_ids = torch.tensor(test_dev_data[1][:test_nums], dtype=torch.long).cuda()
    test_atten_masks = torch.tensor(test_dev_data[2][:test_nums], dtype=torch.long).cuda()
    test_input_ids_a = torch.tensor(test_dev_data[3][:test_nums], dtype=torch.long).cuda()
    test_seg_ids_a = torch.tensor(test_dev_data[4][:test_nums], dtype=torch.long).cuda()
    test_atten_masks_a = torch.tensor(test_dev_data[5][:test_nums], dtype=torch.long).cuda()
    test_y = torch.tensor(test_dev_data[6][:test_nums], dtype=torch.long).cuda()
    test_datatype = torch.tensor(test_dev_data[7][:test_nums], dtype=torch.float).cuda()

    depend_matrix = torch.tensor(load_graph("/run/media/lab1510/新加卷/lab1510/NLPCC_Task7/data/graph/val_all_graph.txt")[:test_nums], dtype=torch.long).cuda()
    test_tensor_loader = TensorDataset(test_input_ids,test_seg_ids,test_atten_masks,depend_matrix,test_seg_ids_a,
                                       test_atten_masks_a,test_y,test_datatype)
    test_dataloader = DataLoader(test_tensor_loader,  batch_size=3000)

    # dev
    dev_input_ids = torch.tensor(test_dev_data[0], dtype=torch.long).cuda()
    dev_seg_ids = torch.tensor(test_dev_data[1], dtype=torch.long).cuda()
    dev_atten_masks = torch.tensor(test_dev_data[2], dtype=torch.long).cuda()
    dev_input_ids_a = torch.tensor(test_dev_data[3], dtype=torch.long).cuda()
    dev_seg_ids_a = torch.tensor(test_dev_data[4], dtype=torch.long).cuda()
    dev_atten_masks_a = torch.tensor(test_dev_data[5], dtype=torch.long).cuda()
    dev_y = torch.tensor(test_dev_data[6], dtype=torch.long).cuda()
    dev_datatype = torch.tensor(test_dev_data[7], dtype=torch.long).cuda()
    depend_matrix = torch.tensor(
        load_graph("/run/media/lab1510/新加卷/lab1510/NLPCC_Task7/data/graph/val_all_graph.txt"),
        dtype=torch.float).cuda()



    dev_tensor_loader = TensorDataset(dev_input_ids,dev_seg_ids,dev_atten_masks,depend_matrix,
                                      dev_seg_ids_a,dev_atten_masks_a,dev_y,dev_datatype)
    dev_dataloader = DataLoader(dev_tensor_loader,  batch_size=512)

    print(f"train sum: {len(train_data[0])}")
    print(f"test sum: {len(test_dev_data[0][:test_nums])}")
    print(f"dev sum: {len(test_dev_data[0][test_nums:])}")

    return train_dataloader,test_dataloader,dev_dataloader



def get_test_dataloader(test_data,batch_size):


    test_input_ids = torch.tensor(test_data[0], dtype=torch.long).cuda()
    test_seg_ids = torch.tensor(test_data[1], dtype=torch.long).cuda()
    test_atten_masks = torch.tensor(test_data[2], dtype=torch.long).cuda()
    test_input_ids_a = torch.tensor(test_data[3], dtype=torch.long).cuda()
    test_seg_ids_a = torch.tensor(test_data[4], dtype=torch.long).cuda()
    test_atten_masks_a = torch.tensor(test_data[5], dtype=torch.long).cuda()
    test_y = torch.tensor(test_data[6], dtype=torch.long).cuda()
    test_datatype = torch.tensor(test_data[7], dtype=torch.long).cuda()

    test_tensor_loader = TensorDataset(test_input_ids, test_seg_ids, test_atten_masks, test_input_ids_a,
                                       test_seg_ids_a,
                                       test_atten_masks_a, test_y,test_datatype)
    test_dataloader = DataLoader(test_tensor_loader, batch_size=512)

    print(f"test sum: {len(test_data[0])}")


    return test_dataloader




if __name__ == '__main__':
    model = AutoModel.from_pretrained("nghuyong/ernie-1.0")
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    data = data_load("/media/lab1510/新加卷/lab1510/NLPCC_Task7/data/train/gender.csv",tokenizer,100)
    print(data)