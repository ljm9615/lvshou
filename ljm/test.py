import pandas as pd
import numpy as np
import sys
import time
sys.path.append("..")
import os
from project.divide import load_data, PATH1, PATH2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def acc(Y, Y_pred, rule, threshold):
    """ Showing some metrics about the training process

    Parameters
    ----------
    Y : list, numpy 1-D array, pandas.Series
        The ground truth on the val dataset.
    Y_pred : list, numpy 1-D array, pandas.Series
        The predict by your model on the val dataset.
    """
    Y = list(Y)
    Y_pred = list(Y_pred)
    print('precision:', precision_score(Y, Y_pred))
    print('accuracy:', accuracy_score(Y, Y_pred))
    print('recall:', recall_score(Y, Y_pred))
    print('micro_F1:', f1_score(Y, Y_pred, average='micro'))
    print('macro_F1:', f1_score(Y, Y_pred, average='macro'))
    print()

    PATH = "../../data/Content"
    with open(os.path.join(PATH, rule, str(threshold) + "_sample_proba_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
        f.write(str(accuracy_score(Y, Y_pred)))
        f.write('\n')
        f.write(str(precision_score(Y, Y_pred)))
        f.write('\n')
        f.write(str(recall_score(Y, Y_pred)))
        f.write('\n')
        f.write(str(f1_score(Y, Y_pred, average='micro')))
        f.write('\n')
        f.write(str(f1_score(Y, Y_pred, average='macro')))
        f.write('\n')
        f.write('\n')


if __name__ == "__main__":
    PATH = "../../data/Content"
    rule = "敏感词"
    threshold = 0.7

    with open(os.path.join(PATH, rule, str(threshold) + "_sample_proba_f_t_pro_f_t.txt"), 'a',
              encoding='utf-8') as f:
        f.write("sample false")
        f.write("\n")
        f.write("\n")

    for i in range(5):
        test_file = "sample" + str(i + 1)
        test_data = pd.read_csv(os.path.join(PATH, rule, test_file + "_pred.csv"), sep=',', encoding='utf-8')
        pred = []
        for i in range(len(test_data)):
            if test_data['pred'][i] > threshold:
                pred.append(1)
            else:
                pred.append(0)
        acc(test_data['result'], pred, rule, threshold)

    with open(os.path.join(PATH, rule, str(threshold) + "_sample_proba_f_t_pro_f_t.txt"), 'a',
              encoding='utf-8') as f:
        f.write("sample true")
        f.write("\n")
        f.write("\n")

    for i in range(5):
        test_file = "sample" + str(i + 1)
        test_data = pd.read_csv(os.path.join(PATH, rule, test_file + "_pred_only.csv"), sep=',', encoding='utf-8')
        pred = []
        for i in range(len(test_data)):
            if test_data['pred'][i] > threshold:
                pred.append(1)
            else:
                pred.append(0)
        acc(test_data['result'], pred, rule, threshold)

    with open(os.path.join(PATH, rule, str(threshold) + "_sample_proba_f_t_pro_f_t.txt"), 'a',
              encoding='utf-8') as f:
        f.write("sample_proportion false")
        f.write("\n")
        f.write("\n")

    for i in range(5):
        test_file = "sample_proportion" + str(i + 1)
        test_data = pd.read_csv(os.path.join(PATH, rule, test_file + "_pred.csv"), sep=',', encoding='utf-8')
        pred = []
        for i in range(len(test_data)):
            if test_data['pred'][i] > threshold:
                pred.append(1)
            else:
                pred.append(0)

        acc(test_data['result'], pred, rule, threshold)

    with open(os.path.join(PATH, rule, str(threshold) + "_sample_proba_f_t_pro_f_t.txt"), 'a',
              encoding='utf-8') as f:
        f.write("sample_proportion true")
        f.write("\n")
        f.write("\n")

    for i in range(5):
        test_file = "sample_proportion" + str(i + 1)
        test_data = pd.read_csv(os.path.join(PATH, rule, test_file + "_pred_only.csv"), sep=',', encoding='utf-8')
        pred = []
        for i in range(len(test_data)):
            if test_data['pred'][i] > threshold:
                pred.append(1)
            else:
                pred.append(0)
        acc(test_data['result'], pred, rule, threshold)
