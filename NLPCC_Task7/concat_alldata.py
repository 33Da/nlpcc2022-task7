import pandas as pd
from sklearn.utils import shuffle

#
# gender_train_df = pd.read_csv("data/train/gender.csv")
# occ_train_df = pd.read_csv("data/train/occupation.csv")
# race_train_df = pd.read_csv("data/train/race.csv")
# region_train_df = pd.read_csv("data/train/region.csv")
#
# print(gender_train_df.shape) # (6706, 8)
# print(occ_train_df.shape)  # (3245, 8)
# print(race_train_df.shape) # (8695, 8)
# print(region_train_df.shape) # (4023, 8)
#
# all_train_df = pd.concat([gender_train_df,occ_train_df,race_train_df,region_train_df])
# all_train_df = shuffle(all_train_df)
# print(all_train_df.shape)  #   (22669, 8)
#
# print(all_train_df.value_counts("attitude"))
# # attitude
# # 0    9855
# # 2    6635
# # 3    5825
# # 1     354
#
# ############### val
#
# gender_val_df = pd.read_csv("data/val/gender.csv")
# occ_val_df = pd.read_csv("data/val/occupation.csv")
# race_val_df = pd.read_csv("data/val/race.csv")
# region_val_df = pd.read_csv("data/val/region.csv")
#
# print(gender_val_df.shape)  # (839, 8)
# print(occ_val_df.shape) # (406, 8)
# print(race_val_df.shape)  # (1088, 8)
# print(region_val_df.shape)  # (504, 8)
#
# all_val_df = pd.concat([gender_val_df,occ_val_df,race_val_df,region_val_df])
# print(all_val_df.shape)  # (2837, 8)
# print(all_val_df.value_counts("attitude"))
# # attitude
# # 0    1285
# # 2     833
# # 3     668
# # 1      51
#
# all_val_df = shuffle(all_val_df)
#
#
# all_train_df.to_csv("data/train/all.csv")
# all_val_df.to_csv("data/val/all.csv")


# 计算数据长度
# from transformers import AutoTokenizer
# import pandas as pd
# tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
# train_data = pd.read_csv("data/train/all.csv")
# test_data = pd.read_csv("data/val/all.csv")
# data = pd.concat([train_data,test_data])
# print(data.shape)
#
# sen = data
# text1 = data['q'].values.tolist()
# text2 = data['a'].values.tolist()
# sent_len = []
# for t1,t2 in zip(text1,text2):
#     encoded_dict = tokenizer.encode_plus(
#         t1,  # Target to encode
#         t2,  # Sentence to encode
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         max_length=150,  # Pad & truncate all sentences.
#         padding='max_length',
#         return_attention_mask=True,  # Construct attn. masks.
#     )
#
#     a = encoded_dict["input_ids"]
#     sent_len.append(sum(encoded_dict['attention_mask']))
#     break
#
# print(max(sent_len))
#
# print(tokenizer.decode(a))


# 计算all 标签分布
import pandas as pd
data = pd.read_csv("data/train/all.csv")
print(data.value_counts("datatype"))
# 0    9855
# 2    6673
# 1    6141