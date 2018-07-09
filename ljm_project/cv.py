import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random

random.seed(2018)


jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\敏感词.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\部门名称.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\禁忌称谓.txt")


def load_data(rule):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\agent_sentences.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].\
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


def cut_words(sentences):
    # words = [word for word in jieba.cut(sentences) if word not in [' ', '喂', '啊', '吗', '吧', '呢', '你', '他', '这',
    #                                                                '嗯', '了', '的', '对', '是', '哎', '把', '我', '在',
    #                                                                '我们', '哦', '您', "哈", '这', '呃', '了', '才',
    #                                                                '也', '那', '再', '好', '行', '一下', '一', '都', '就',
    #                                                                '会', '呀', '的话', '给', '看', '等', '个', '还']]
    # print(words)
    # print(len(words))
    words = " ".join(list(jieba.cut(sentences)))
    return words


def count_vectorizer(data):
    # counter = {}
    # for line in data['words'].values:
    #     for word in line.split(' '):
    #         counter[word] = counter.get(word, 0) + 1
    #
    # counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    # with open(r"E:\cike\lvshou\zhijian_data\counter.txt", 'w', encoding='utf-8') as f:
    #     for key, value in counter:
    #         f.write(str(key) + ' : ' + str(value) + '\n')

    count_vect = CountVectorizer(max_df=0.5, max_features=1000)
    words_counter = count_vect.fit_transform(data['words'].values)
    weight = words_counter.toarray()

    # vectorizer = TfidfVectorizer(max_df=0.5, max_features=300,
    #                              min_df=5,  # stop_words=list(set(stopwords.words('english'))),
    #                              use_idf=True)
    # tfidf = vectorizer.fit_transform(data['words'].values)
    # weight = tfidf.toarray()
    return weight


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

    data['words'] = data['agent_sentences'].apply(cut_words)
    weight = count_vectorizer(data)
    weight.dump(r"E:\cike\lvshou\zhijian_data\count_weight.npy")
