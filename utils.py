import torch
import pandas as pd
import pickle
import torch.nn.functional as F


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


def bce_loss(x_ui, x_uj):
    criteria = torch.nn.BCELoss()
    loss = criteria(torch.sigmoid(x_ui), torch.ones(x_ui.shape[0]).cuda())
    loss += criteria(torch.sigmoid(x_uj), torch.zeros(x_uj.shape[0]).cuda())
    return loss


def bpr_loss(x_ui, x_uj):
    x_uij = x_ui - x_uj
    log_prob = F.logsigmoid(x_uij).sum()
    return -log_prob
