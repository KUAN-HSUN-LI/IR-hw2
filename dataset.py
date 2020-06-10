from torch.utils.data import IterableDataset
import random
import torch


class PairDataset(IterableDataset):
    def __init__(self, item_size, train_set, train_pair):
        self.item_size = item_size
        self.train_set = train_set
        self.train_pair = train_pair
        random.seed(42)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.example_lst = []
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        index_lst = list(range(len(self.train_pair)))
        random.shuffle(index_lst)
        self.example_lst.extend(index_lst)
        return self

    def __next__(self):
        if len(self.example_lst) == 0:
            raise StopIteration
        return self._sample(self.example_lst.pop())

    def _sample(self, idx):
        u = self.train_pair[idx][0]
        i = self.train_pair[idx][1]
        while True:
            j = random.randint(0, self.item_size - 1)
            if j not in self.train_set[u]:
                break

        return u, i, j
