import pdb
from preprocess import get_data
from dataset import PairDataset
from torch.utils.data import DataLoader
from model import BCE
from metrics import map_score
from utils import load_pkl, get_top_k, submit
from args import get_args
import torch
torch.manual_seed(42)


args = get_args()

dataset = load_pkl("dataset.pkl")

raw_set = dataset['raw_set']
train_set = dataset['train_set']
test_set = dataset['test_set']
train_pair = dataset['train_pair']
user_size = dataset['user_size']
item_size = dataset['item_size']


if args.train:
    dim = 64
    dataset = PairDataset(item_size, train_set, train_pair)
    dataloader = DataLoader(dataset, 1024, num_workers=4)
    model = BCE(user_size, item_size, dim).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
    criteria = torch.nn.BCELoss()

    max_score = 0
    for epoch in range(50):
        for idx, (u, i, j) in enumerate(dataloader):
            opt.zero_grad()
            x_ui, x_uj = model(u, i, j)
            loss = criteria(x_ui, torch.ones(x_ui.shape[0]).cuda())
            loss += criteria(x_uj, torch.zeros(x_uj.shape[0]).cuda())
            loss.backward()
            opt.step()
            print(f"{idx}:{loss:.4f}", end='\r')
        result = get_top_k(model.W.detach(), model.H.detach(), train_set, k=50, device="cuda")
        score = map_score(result, test_set)
        print(f"epoch{epoch}: {score:.4f}")
        if score > max_score:
            max_score = score
            torch.save(model.state_dict(), f"model/bce-{dim}-{epoch}.pkl")

if args.test:
    dim = 64
    num = 47
    model = BCE(user_size, item_size, dim)
    model.load_state_dict(torch.load(f"model/bce-{dim}-{num}.pkl", map_location=lambda storage, loc: storage))

    result = get_top_k(model.W.detach(), model.H.detach(), raw_set, 50, "cpu")
    submit(args.output, user_size, result)
