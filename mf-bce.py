import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
dataset = []

torch.manual_seed(42)

with open("data/train.csv", "r") as f:
    f.readline()
    for line in f:
        dataset.append([int(item) for item in line.split(",")[1].split(" ")])

item_amount = max(map(max, dataset)) + 1
user_amount = len(dataset)
target_matrix = np.zeros([user_amount, item_amount])

for userId, items in enumerate(dataset):
    for itemId in items:
        target_matrix[userId, itemId] = 1

K = 50
LAMBDA = 1e-2
LR = 1e-2
criteria = nn.BCELoss()

U = Variable(torch.randn([user_amount, K]), requires_grad=True)
P = Variable(torch.randn([K, item_amount]), requires_grad=True)
target_matrix = torch.FloatTensor(target_matrix)
opt = torch.optim.AdamW([U, P], lr=LR, weight_decay=LAMBDA)
m = nn.Sigmoid()

for i in range(75):
    opt.zero_grad()
    R = torch.mm(U, P)
    loss = criteria(m(R), target_matrix)
    loss.backward()
    opt.step()
    print(i, loss.item(), end='\r')

print()
result = torch.mm(U, P).detach().numpy()
result = np.argsort(result, axis=1)[:, ::-1]

top_50 = []
for idx, items in enumerate(result):
    d = []
    raw = set(dataset[idx])
    for item in items:
        if len(d) == 50:
            break
        if item not in raw:
            d.append(str(item))

    top_50.append(" ".join(d))


df = pd.DataFrame()
df["UserId"] = [i for i in range(user_amount)]
df["ItemId"] = top_50
df.to_csv("output.csv", index=False)
