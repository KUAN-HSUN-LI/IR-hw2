import torch
import pandas as pd
import pickle


def load_pkl(file_name):
    with open(file_name, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_top_k(user_emb, item_emb, train_set, k, device):
    mask = user_emb.new_ones(user_emb.shape[0], item_emb.shape[0])
    for i in range(user_emb.shape[0]):
        mask[i].scatter_(dim=0, index=torch.tensor(list(train_set[i])).to(device), value=torch.tensor(0.0).to(device))
    result = torch.sigmoid(torch.mm(user_emb, item_emb.t()))
    result = torch.mul(mask, result)
    result = result.topk(k=k, dim=1)
    return result[1].cpu().numpy()


def submit(file_name, user_size, result):
    df = pd.DataFrame()
    df["UserId"] = [i for i in range(user_size)]
    df["ItemId"] = [" ".join([str(i) for i in r]) for r in result]
    df.to_csv(file_name, index=False)
