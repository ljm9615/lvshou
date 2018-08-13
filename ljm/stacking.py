# encoding=utf-8
import sys
import time
import os
import numpy as np
import lightgbm as lgb
import pickle
sys.path.append("..")
import pandas as pd
from project.interface import SupperModel
from sklearn.model_selection import KFold
from project.divide import load_data, PATH1, PATH2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random

PATH = "../../data/Content"
SAMPLE_PATH = "../../data/Sample"


def load_all_data(alone=False):
    rules = os.listdir(PATH)
    rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
    if not alone:
        suffix = "_agent_tokens.csv"
    else:
        suffix = "_tokens.csv"
    all_data = pd.DataFrame()
    for rule in rules:
        _ = load_data(os.path.join(PATH, rule, rule + suffix))
        all_data = pd.concat([all_data, _], axis=0)

    # 测试集样本空间
    all_data.drop_duplicates(['UUID'], inplace=True)
    all_data.reset_index(inplace=True)
    return all_data


def sample_train_data(train_label, n=5):
    """
    使用训练集中所有正样本，随机选择等数量负样本，生成n份不同的训练集，返回每份对应的下标
    :param n: 训练集份数
    :return: 每份训练集对应下标矩阵
    """
    result = []
    pos_idx = []
    for i, label in enumerate(train_label):
        if label == 1:
            pos_idx.append(i)

    neg_idx = [i for i in range(len(train_label)) if i not in pos_idx]
    print("pos size", len(pos_idx))
    print("neg size", len(neg_idx))
    print()
    for i in range(n):
        result.append(pos_idx)
        seed = (i + 1) * 1000
        random.seed(seed)
        sample_neg_idx = random.sample(neg_idx, len(pos_idx))
        result[i].extend(sample_neg_idx)
    return result


class Feature(object):
    def __init__(self, rule, max_df, min_df, max_features, use_idf=False):
        self.rule = rule
        self.path = os.path.join(PATH, rule)
        self.train_data = None
        self.test_data = None
        self.seed = 2018
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.use_idf = use_idf
        self.Counter = None

    def get_label(self, data):
        label = []
        # 对每个数据样本，遍历其检测出的违规类型
        for counter in range(len(data)):
            if self.rule not in data['analysisData.illegalHitData.ruleNameList'][counter]:
                label.append(0)
            else:
                for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
                    if self.rule == item:
                        label.append(1 if data['correctInfoData.correctResult'][counter].
                                     get("correctResult")[i] == '1' else 0)
        return label

    def get_data(self, test_file, all_data):
        test_uuid = pd.read_csv(os.path.join(SAMPLE_PATH, test_file + ".txt"), header=None)

        # 训练集为去掉测试集的全部数据
        train_data = all_data[~all_data['UUID'].isin(test_uuid.values[:, 0])]
        train_data.reset_index(drop=True, inplace=True)
        self.train_data = train_data.reset_index(drop=True)
        label = self.get_label(self.train_data)
        train_data['label'] = label
        self.train_data = train_data.sample(frac=1, random_state=self.seed)

        # 测试集由test_file指定
        test_data = all_data[all_data['UUID'].isin(test_uuid.values[:, 0])]
        test_data.reset_index(drop=True, inplace=True)
        self.test_data = test_data.reset_index(drop=True)
        label = self.get_label(self.test_data)
        self.test_data['label'] = label
        self.test_data = test_data.sample(frac=1, random_state=self.seed)

        print(len(self.train_data))
        print(len(self.test_data))
        print()

    def get_counter(self):
        if self.use_idf:
            self.Counter = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df,
                                           max_features=self.max_features, use_idf=True)
        else:
            if os.path.exists(os.path.join(PATH, self.rule, "CountVectorizer_total.pkl")):
                print("load counter_vectorizer...")
                self.Counter = pickle.load(open(os.path.join(PATH, self.rule, "CountVectorizer_total.pkl"), 'rb'))
                # print(sorted(self.Counter.vocabulary_.items(), key=lambda x: x[1], reverse=True))
                print(len(self.Counter.vocabulary_.items()))
            else:
                self.Counter = CountVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.max_features)
                pickle.dump(self.Counter, open(os.path.join(PATH, self.rule, "CountVectorizer_total.pkl"), 'wb'))

    def get_feature(self, train=True):
        if train:
            token_counter = self.Counter.transform(self.train_data['transData.sentenceList'].values)
            path = os.path.join(PATH, self.rule, "stacking", "train")
            feature_name = "train_feature.pkl"
            label_name = "train_label.csv"
        else:
            token_counter = self.Counter.transform(self.test_data['transData.sentenceList'].values)
            path = os.path.join(PATH, self.rule, "stacking", "test")
            feature_name = "test_feature.pkl"
            label_name = "test_label.csv"

        weight = token_counter.toarray()
        print("weight shape", weight.shape)
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(token_counter, open(os.path.join(path, feature_name), 'wb'))
        self.train_data[['UUID', 'label']].to_csv(os.path.join(path, label_name), sep=',', encoding="utf-8", index=False)


class LGBM(SupperModel):
    def __init__(self, model, param, **kags):
        super(SupperModel, self).__init__()
        self.params = param
        self.model = model(num_leaves=self.params.get("num_leaves"),
                           max_depth=self.params.get("max_depth"),
                           n_estimators=self.params.get("n_estimators"),
                           n_jobs=self.params.get("n_jobs"),
                           learning_rate=self.params.get("learning_rate"),
                           colsample_bytree=self.params.get("colsample_bytree"),
                           subsample=self.params.get("subsample"))
        self.clf = None
        self.best_iters = []

    def cv(self, X_train, y_train, k_fold, path):
        print("using " + str(k_fold) + " cross validation...")
        kf = KFold(n_splits=k_fold)
        preds = []
        probs = []
        for train_idx, test_idx in kf.split(X_train):
            print(train_idx, test_idx)
            X = X_train[train_idx]
            y = y_train[train_idx]
            X_val = X_train[test_idx]
            y_val = y_train[test_idx]

            lgb_model = self.model.fit(
                X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=self.params.get("early_stopping_rounds"),
                verbose=0)

            print("predicting...")
            test_preds = lgb_model.predict(X_val)
            test_probs = lgb_model.predict_proba(X_val)[:, 1]
            preds.extend(test_preds)
            probs.extend(test_probs)
            self.best_iters.append(lgb_model.best_iteration_)

        labels = pd.DataFrame(y_train)
        preds = pd.DataFrame(preds)
        probs = pd.DataFrame(probs)

        result = pd.concat([labels, preds])
        result = pd.concat([result, probs])
        result.to_csv(path, sep=',', encoding="utf-8", index=False)

        print("validation result...")
        self.acc(y_train, preds)
        print("best iters:")
        print(self.best_iters)

    def train(self, X_train, y_train):
        self.model.n_estimators = (sum(self.best_iters) // len(self.best_iters))
        print("training...")
        print("iters:", str(self.model.n_estimators))
        self.clf = self.model.fit(X_train, y_train, verbose=1)

    def predict(self, X_test, proba=False):
        probs = self.clf.predict_proba(X_test)[:, 1]
        preds = self.clf.predict(X_test)

        if proba:
            return probs
        else:
            return preds


if __name__ == "__main__":
    start_time = time.time()

    param = {
        "num_leaves": 35,
        "max_depth": 7,
        "n_estimators": 20000,
        "n_jobs": 20,
        "learning_rate": 0.01,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "early_stopping_rounds": 100,
    }
    rule = "部门名称"
    test_file = "sample1"
    n = 5
    model = LGBM(lgb.LGBMClassifier, param)

    if not os.path.exists(os.path.join(PATH, rule, "stacking", "train", "train_label.pkl")):
        print("loading data...")
        data = load_all_data(alone=False)

        f = Feature(rule, max_df=0.5, min_df=3, max_features=10000)
        f.get_data(test_file, data)
        f.get_counter()
        f.get_feature(train=True)
        f.get_feature(train=False)

    train_feature = pickle.load(open(os.path.join(PATH, rule, "stacking", "train", "train_feature.pkl"), 'rb'))
    train_label = pd.read_csv(os.path.join(PATH, rule, "stacking", "train", "train_label.csv"),
                              sep=',', encoding="utf-8")
    train_label = train_label.values

    print("all train data size", train_feature.shape)
    print("all train label size", train_label.shape)

    random.seed(2018)
    idx = random.sample(range(len(train_label)), k=int(len(train_label) * 0.2))

    train_data, eval_data = train_feature[idx], train_feature[~idx]
    train_label, eval_label = train_label[idx], train_label[~idx]

    print("train size", train_data.shape, "train label", train_label.shape)
    print("eval size", eval_data.shape, "eval label", eval_label.shape)
    print()

    sample_result = sample_train_data(train_label, n)
    print("sample size", np.array(sample_result).shape)

    # for i in range(n):
    #     _train_data = train_data[sample_result[i]]
    #     _train_label = train_label[sample_result[i]]
    #     model.cv(_train_data, _train_label, k_fold=5, path=os.path.join(PATH, rule, "stacking",
    #                                                                     "train_" + str(i+1) + "_cv.csv"))
    #     model.train(_train_data, _train_label)
    #     preds = model.predict(eval_data, proba=True)
    #
    #     labels = pd.DataFrame(eval_label)
    #     preds = pd.DataFrame(preds)
    #
    #     result = pd.concat([labels, preds])
    #
    #     result.to_csv(path=os.path.join(PATH, rule, "stacking", "train_" + str(i+1) + "_result.csv"), sep=',',
    #                   encoding="utf-8", index=False)

    print('time cost is', time.time() - start_time)


