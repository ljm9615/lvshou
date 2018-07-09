import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import lightgbm as lgb

random.seed(2018)


def load_data(rule):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\agent_sentences.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList']. \
        apply(eval).apply(lambda x: [word.replace("禁忌部门名称", "部门名称") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    if not rule:
        return data
    else:
        key_words = []
        counter = 0
        index = []
        with open(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + ".txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                key_words.append(line.strip())

        # 对每个数据样本
        for sentences in data['agent_sentences']:

            # 针对该样本统计遍历违规词，计算是否在句子中
            for key_word in key_words:
                # 违规词在句子中
                if key_word in sentences:
                    index.append(counter)
                    break
            counter += 1
        return data.iloc[index].reset_index()


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

    with open(r"E:\cike\lvshou\zhijian_data\lgb_pred.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')


def get_label_index(data, rule):
    index = []
    not_index = []

    # 对每个数据样本，遍历其检测出的违规类型
    for counter in range(len(data)):
        for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
            # 如果违规类型为要统计的类型且检测结果正确，总数量加1
            if rule == item and data['correctInfoData.correctResult'][counter].get("correctResult")[i] == '1':
                index.append(counter)
    for i in range(len(data)):
        if i not in index:
            not_index.append(i)
    return index, not_index


if __name__ == "__main__":
    data = load_data(rule="敏感词")
    weight = np.load(r"E:\cike\lvshou\zhijian_data\count_weight.npy")

    index, not_index = get_label_index(data, "敏感词")
    label = np.zeros(shape=(len(data, )), dtype=int)
    not_index = random.sample(not_index, len(index))
    label[index] = 1
    index.extend(not_index)
    random.shuffle(index)

    weight = weight[index]
    label = label[index]

    lgb_cv(weight, label, 5)

    pred = []
    with open(r"E:\cike\lvshou\zhijian_data\lgb_pred.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p = float(line.strip())
            if p > 0.5:
                pred.append(1)
            else:
                pred.append(0)

    print("precision: ", precision_score(label, pred))
    print("recall: ", recall_score(label, pred))
    print("micro :", f1_score(label, pred, average="micro"))
    print("macro: ", f1_score(label, pred, average="macro"))