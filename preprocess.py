import random
import pickle

random.seed(42)


def read_data():
    data = []
    with open("data/train.csv", "r") as f:
        f.readline()
        for line in f:
            data.append([int(item) for item in line.split(",")[1].split(" ")])
    return data


def split_train_test(datas, test_size=0.1):
    train_set = []
    test_set = []
    for data in datas:
        test_data = random.sample(data, int(len(data)*test_size))
        train_data = list(set(data) - set(test_data))
        train_set.append(train_data)
        test_set.append(test_data)

    return train_set, test_set


def create_pair(train_set):
    pair = []
    for user, items in enumerate(train_set):
        pair.extend([(user, item) for item in items])
    return pair


def get_data():
    data = read_data()
    item_size = max(map(max, data)) + 1
    user_size = len(data)
    train_set, test_set = split_train_test(data)
    train_pair = create_pair(train_set)
    dataset = {"raw_set": data, "train_set": train_set, "test_set": test_set, "train_pair": train_pair,
               "user_size": user_size, "item_size": item_size}

    return dataset


if __name__ == "__main__":
    dataset = get_data()
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
