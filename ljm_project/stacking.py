import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
import numpy as np
import pandas as pd


def lgb_cv(weight, label, k_fold):
    train = weight
    kf = KFold(n_splits=k_fold)
    preds =[]
    clf = lgb.LGBMClassifier(num_leaves=35, max_depth=7, n_estimators=20000, n_jobs=20, learning_rate=0.01,
                             colsample_bytree=0.8, subsample=0.8)
    for train_idx, test_idx in kf.split(train):
        print(train_idx, test_idx)
        X = train[train_idx]
        y = label[train_idx]
        X_val = train[test_idx]
        y_val = label[test_idx]
        lgb_model = clf.fit(
            X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=100, verbose=1)
        test_preds = lgb_model.predict_proba(X_val)[:, 1]

        print("predicting...")
        preds.extend(test_preds)

    with open(r"E:\cike\lvshou\zhijian_data\stacking.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')


def lgb_cv_k_fold():
    # weight = np.load(r"E:\cike\lvshou\zhijian_data\count_weight_jjcw.npy")
    # label = np.load(r"E:\cike\lvshou\zhijian_data\label_jjcw.npy")
    path = r"E:\cike\lvshou\zhijian_data\result" + '\\'
    files = ['lgb_pred_10000.txt', 'lgb_pred_12000.txt', 'xgb_pred_10000.txt', 'xgb_pred_12000.txt',
             'lgb_pred_window_10.txt', 'lgb_pred_window_15.txt', 'lgb_pred_window_20.txt',
             'xgb_pred_window_10.txt', 'xgb_pred_window_15.txt', 'xgb_pred_window_20.txt']
    label = np.load(r"E:\cike\lvshou\zhijian_data\敏感词\label_window.npy")

    weight = pd.DataFrame()
    for file in files:
        pred = pd.read_csv(path + file, header=None)
        # temp = []
        # for item in pred.values:
        #     if item > 0.5:
        #         temp.append(1)
        #     else:
        #         temp.append(0)

        # print(file)
        # print("precision: ", precision_score(label, pred))
        # print("recall: ", recall_score(label, pred))
        # print("micro :", f1_score(label, pred, average="micro"))
        # print("macro: ", f1_score(label, pred, average="macro"))
        # print()
        weight = pd.concat([weight, pred], axis=1)
    print(weight)
    weight = weight.apply(sum, axis=1)
    pred = []
    for item in weight.values:
        if item > 5:
            pred.append(1)
        else:
            pred.append(0)
    #
    # print(weight.shape)
    # print(label.shape)
    # lgb_cv(weight.values, label, 5)

    # pred = []
    # with open(r"E:\cike\lvshou\zhijian_data\stacking.txt", 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         p = float(line.strip())
    #         if p > 0.5:
    #             pred.append(1)
    #         else:
    #             pred.append(0)

    print("precision: ", precision_score(label, pred))
    print("recall: ", recall_score(label, pred))
    print("micro :", f1_score(label, pred, average="micro"))
    print("macro: ", f1_score(label, pred, average="macro"))


if __name__ == "__main__":
    lgb_cv_k_fold()
