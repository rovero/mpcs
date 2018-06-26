from sklearn.metrics import cohen_kappa_score
import pandas as pd


def qwk(y1, y2):
    k = len(y1)
    O = [[0 for x in range(k)] for y in range(k)]  # k*k matrix
    E = [[0 for x in range(k)] for y in range(k)]  # k*k matrix
    W = [[0 for x in range(k)] for y in range(k)]  # k*k matrix
    for i in range(k):
        O[y1[i]][y2[i]] += 1 / k
    y1_df = pd.DataFrame(y1)
    y2_df = pd.DataFrame(y2)
    cnt1 = y1_df[0].value_counts()
    cnt1 /= k
    cnt2 = y2_df[0].value_counts()
    cnt2 /= k
    for i in cnt1.index:
        for j in cnt2.index:
            E[i][j] = cnt1[i] * cnt2[j]
    for i in range(k):
        for j in range(k):
            W[i][j] = (i - j) ** 2
    up = 0
    low = 0
    for i in range(k):
        for j in range(k):
            up += W[i][j] * O[i][j]
            low += W[i][j] * E[i][j]
    return 1 - up / low