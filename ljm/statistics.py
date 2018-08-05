import pandas as pd
import csv
import jieba
import os
import re
import numpy as np

jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\敏感词.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\部门名称.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\禁忌称谓.txt")
csv.field_size_limit(500 * 1024 * 1024)


def data_concat():
    data_writer = csv.writer(open(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data.csv", 'w',
                                  encoding='utf-8', newline=''))
    data_writer.writerow(["UUID", "analysisData.illegalHitData.ruleNameList",
                          "correctInfoData.correctResult", "sentences"])
    ids = []
    data1_reader = csv.reader(open(r"E:\cike\lvshou\zhijian_data\zhijian_data.csv", 'r', encoding='utf-8'))
    data2_reader = csv.reader(open(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\zhijian_data_20180709.csv",
                                   'r', encoding='utf-8'))
    for line in data1_reader:
        if line[0] in ids or line[7] == "transData.sentenceList":
            continue
        ids.append(line[0])
        data_writer.writerow([line[0], line[3], line[6], get_sentences(eval(line[7]))])
    for line in data2_reader:
        if line[0] in ids or line[7] == "transData.sentenceList":
            continue
        ids.append(line[0])
        data_writer.writerow([line[0], line[3], line[6], get_sentences(eval(line[7]))])


def statistics(sample=False):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data_cut.csv", sep=',')
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval) \
        .apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
               .replace("过度承诺效果问题", "过度承诺效果")
               .replace("投诉倾向", "投诉")
               .replace("提示客户录音或实物有法律效力", "提示通话有录音")
               .replace("夸大产品功效", "夸大产品效果") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)

    if sample:
        ids = pd.read_csv(r"E:\cike\lvshou\data\Sample\sample_proportion2.txt", header=None).values
        all_ids = data['UUID']
        indices = []
        for i, id in enumerate(all_ids):
            if id in ids:
                indices.append(i)
        data = data.loc[indices].reset_index()
        print(len(ids), len(data))
    all_illegal = []
    correct_illegal = []
    wrong_illegal = []
    counter = 0
    for illegals in data['analysisData.illegalHitData.ruleNameList']:
        result = data['correctInfoData.correctResult'][counter].get("correctResult")
        counter += 1
        for i, l in enumerate(illegals):
            if '1' in result[i]:
                correct_illegal.append(l)
            if '2' in result[i]:
                wrong_illegal.append(l)
            all_illegal.append(l)

    illegal_counter = {}
    correct_counter = {}
    wrong_counter = {}
    for illegal in set(all_illegal):
        illegal_counter[illegal] = all_illegal.count(illegal)
        correct_counter[illegal] = correct_illegal.count(illegal)
        wrong_counter[illegal] = wrong_illegal.count(illegal)
    print(len(illegal_counter), sorted(illegal_counter.items(), key=lambda x: x[1], reverse=True))
    print(len(correct_counter), sorted(correct_counter.items(), key=lambda x: x[1], reverse=True))
    # print(len(wrong_counter), sorted(wrong_counter.items(), key=lambda x: x[1], reverse=True))
    return dict(sorted(illegal_counter.items(), key=lambda x: x[1], reverse=True)), \
           dict(sorted(correct_counter.items(), key=lambda x: x[1], reverse=True))


def get_sentences(sentence_list):
    sentence_content = []
    for sentence in sentence_list:
        if sentence.get("role") == "AGENT":
            sentence_content.append(sentence.get("content"))
    return ' '.join(sentence_content)


def get_stopwords(path=r"E:\cike\lvshou\zhijian_data\stopwords.txt"):
    stop_file = path
    if not os.path.exists(stop_file):
        return []
    with open(stop_file, 'rb+') as fr:
        data = fr.read()
    stopwords = [_ for _ in data.decode("utf-8").strip().split('\n')]
    del data
    return stopwords


def cut_words(sentences, stopwords, pattern_list):
    for _token in stopwords:
        sentences = sentences.replace(_token, '')
    for _token in pattern_list:
        pattern = re.compile(r"%s" % _token)
        sentences = pattern.sub(' ', sentences)
    words = " ".join(list(jieba.cut(sentences)))
    return words


def cut():
    print("load data...")
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList']. \
        apply(eval).apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
                          .replace("过度承诺效果问题", "过度承诺效果") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)

    print("cutting...")
    stopwords = get_stopwords()
    pattern_list = get_stopwords(r"E:\cike\lvshou\zhijian_data\stopre.txt")

    data['sentences'] = data['sentences'].apply(lambda x: cut_words(x, stopwords, pattern_list))
    data.to_csv(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data_cut_stop.csv", sep=',',
                encoding="utf-8", index=False)


if __name__ == "__main__":
    # data_concat()
    all_counter, corrent_counter = statistics()
    all_sample_counter, corrent_sample_counter = statistics(sample=True)
    all_sample_rate = {}
    corrent_sample_rate = {}
    for key, value in all_sample_counter.items():
        all_sample_rate[key] = "%.2f%%" % (value / all_counter.get(key, 0) * 100)
    for key, value in corrent_sample_counter.items():
        if corrent_counter.get(key, 0) == 0:
            corrent_sample_rate[key] = 0
        else:
            corrent_sample_rate[key] = "%.2f%%" % (value / corrent_counter.get(key, 0) * 100)
    print(len(all_sample_rate), all_sample_rate)
    print(len(corrent_sample_rate), corrent_sample_rate)
    # print(len(all_sample_rate), sorted(all_sample_rate.items(), key=lambda x: x[1], reverse=True))
    # print(len(corrent_sample_rate), sorted(corrent_sample_rate.items(), key=lambda x: x[1], reverse=True))
    # cut()
