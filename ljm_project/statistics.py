import pandas as pd
import csv
import jieba
import os
import re

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


def statistics():
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data_cut.csv", sep=',')
    print(len(data))
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval)
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
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
    print(sorted(illegal_counter.items(), key=lambda x: x[1], reverse=True))
    print(sorted(correct_counter.items(), key=lambda x: x[1], reverse=True))
    print(sorted(wrong_counter.items(), key=lambda x: x[1], reverse=True))


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
    # statistics()
    cut()
