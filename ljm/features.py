import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import time
sys.path.append("..")
from project.divide import load_data, PATH1, PATH2

PATH = "../../data/Content"
TEST_PATH = "../../data/Content"


class Features(object):
    def __init__(self, rule, alone, max_df, min_df, max_features, use_idf=False):
        self.Counter = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
        if use_idf:
            self.Counter = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, use_idf=True)
        if not alone:
            file_name = rule + "_agent_tokens.csv"
        else:
            file_name = rule + "_tokens.csv"
        self.rule = rule
        self.alone = alone
        self.path = os.path.join(PATH, rule)
        self.data = load_data(path=os.path.join(self.path, file_name))
        self.tokens = "transData.sentenceList"
        self.seed = 2018

    def sample(self, test_uuid, only=True):
        """
        采样训练集数据，首先将出现在测试集中的数据去除
        然后使用剩余数据的所有正样本，采样相同数量的负样本
        将训练集 UUID 保存在 self.path 路径中
        :param test_uuid: 测试集数据 UUID
        :param only: only 为 True，负样本仅从不出现任何违规的数据中提取
        :return: 训练集
        """
        self.data = self.data[~self.data['UUID'].isin(test_uuid.values[:, 0])]
        print("是否只从不出现任何违规的数据中采集负样本: " + str(only))
        # 负样本从不出现任何违规的数据中提取
        if only:
            if not self.alone:
                file_name = os.path.join(PATH, "不违规", "不违规_agent_tokens.csv")
            else:
                file_name = os.path.join(PATH, "不违规", "不违规_tokens.csv")
            neg_data = load_data(file_name)
        else:
            rules = os.listdir(PATH)
            rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
            rules.remove(self.rule)
            if not self.alone:
                suffix = "_agent_tokens.csv"
            else:
                suffix = "_tokens.csv"
            neg_data = pd.DataFrame()
            for rule in rules:
                _ = load_data(os.path.join(PATH, rule, rule + suffix))
                neg_data = pd.concat([neg_data, _], axis=0)

            # 负样本空间
            neg_data.drop_duplicates(['UUID'], inplace=True)

        train_data = pd.concat([self.data, neg_data.sample(n=len(self.data), random_state=self.seed)], axis=0)
        train_data = train_data.sample(frac=1, random_state=self.seed)
        return train_data

    def load_train(self, test_file, only):
        """
        采样训练集数据，首先将出现在测试集中的数据去除
        然后使用剩余数据的所有正样本，采样相同数量的负样本
        将训练集 UUID 保存在 self.path 路径中
        :param test_file: 测试集UUID文件
        :param only: only 为 True，负样本仅从不出现任何违规的数据中提取
        """
        # 已经提取过训练集，直接加载返回
        if only:
            file_name = self.rule + "_train_only_" + test_file + ".csv"
        else:
            file_name = self.rule + "_train_" + test_file + ".csv"
        if os.path.exists(os.path.join(self.path, file_name)):
            self.data = load_data(os.path.join(self.path, file_name))
            return

        print("sample train data...")
        test_uuid = pd.read_csv(os.path.join('../../data/Sample', test_file + ".txt"), header=None)
        self.data = self.sample(test_uuid, only=only)
        self.data.reset_index(drop=True, inplace=True)
        self.data.to_csv(os.path.join(self.path, file_name), sep=',', encoding="utf-8", index=False)

    def load_test(self, test_file):
        test_uuid = pd.read_csv(os.path.join('../../data/Sample', test_file + ".txt"), header=None)
        data1 = load_data(PATH1)
        data2 = load_data(PATH2)
        data = pd.concat([data1, data2])
        del (data2, data1)
        data.drop_duplicates(['UUID'], inplace=True)
        data.reset_index(inplace=True)
        self.data = data[data['UUID'].isin(test_uuid.values[:, 0])]
        self.data.reset_index(drop=True, inplace=True)

    def get_label(self, _file):
        label = []
        # 对每个数据样本，遍历其检测出的违规类型
        for counter in range(len(self.data)):
            if self.rule not in self.data['analysisData.illegalHitData.ruleNameList'][counter]:
                label.append(0)
            else:
                for i, item in enumerate(self.data['analysisData.illegalHitData.ruleNameList'][counter]):
                    if self.rule == item:
                        label.append(1 if self.data['correctInfoData.correctResult'][counter].
                                     get("correctResult")[i] == '1' else 0)
        for i in range(len(label)):
            if label[i] == 1:
                print(self.data['analysisData.illegalHitData.ruleNameList'][i],
                      self.data['correctInfoData.correctResult'][i])
        np.array(label).dump(os.path.join(self.path, _file))

    def get_weight(self, test_file, only, train=True):
        if train:
            print("load train data...")
            self.load_train(test_file, only)
            if only:
                file_name = self.rule + "_train_weight_only_" + test_file + ".npy"
                label_name = self.rule + "_train_label_only_" + test_file + ".npy"
            else:
                file_name = self.rule + "_train_weight_" + test_file + ".npy"
                label_name = self.rule + "_train_label_" + test_file + ".npy"
        else:
            print("load test data...")
            self.load_test(test_file)
            file_name = self.rule + "_test_weight_" + test_file + ".npy"
            label_name = self.rule + "_test_label_" + test_file + ".npy"
        if not os.path.exists(os.path.join(self.path, file_name)):
            print("get label...")
            self.get_label(label_name)
            print("get weight...")
            token_counter = self.Counter.fit_transform(self.data['transData.sentenceList'].values)
            weight = token_counter.toarray()
            print(weight.shape)
            weight.dump(os.path.join(self.path, file_name))


if __name__ == "__main__":
    start_time = time.time()
    f = Features("部门名称", alone=True, max_df=0.5, min_df=3, max_features=10000)
    f.get_weight("sample1", only=False, train=True)
    print('time cost is', time.time() - start_time)
