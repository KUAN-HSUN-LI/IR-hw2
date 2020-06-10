import numpy as np


def map_score(predicts, ans):
    user_size = predicts.shape[0]
    score = 0
    for i in range(user_size):
        ap = 0
        cnt = 0
        for idx, pred in enumerate(predicts[i]):
            if pred in ans[i]:
                cnt += 1
                ap += cnt / (idx + 1)
        ap /= len(ans[i])
        score += ap
    return score / user_size


if __name__ == "__main__":
    p = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    ans = [[2], [1, 6]]
    print(map_score(p, ans))
