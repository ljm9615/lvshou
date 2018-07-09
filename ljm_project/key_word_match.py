import pandas as pd
import csv


def load_data():
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\agent_sentences.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].\
        apply(eval).apply(lambda x: [word.replace("禁忌部门名称", "部门名称") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    return data


def get_precision(data, rule):
    key_words = []
    with open(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + ".txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key_words.append(line.strip())
    # print(set(key_words))
    # key_words = ['权威顾问', '专案组负责人', '资深顾问组负责人', '负责人助手营养师',
    #              '海外留学', '特助', '助手', '首席资深顾问', '高级顾问',
    #              '效果中心总负责人', '老爷子', '资深组办公室负责人', '负责人', '博士',
    #              '秘书', '教练', '元老级', '高级资深顾问', '助理', '资深指导',
    #              '老师', '元老级人物', '效果中心负责人', '首席资深', '效果中心调查员',
    #              '减肥师总监', '指导员', '资深人士', '专家', '资深组长']

    all_word_counter = {}
    right_word_counter = {}
    counter = 0

    all_right_illegal_num = 0  # 样本中所有标记为违反rule且结果正确的个数 TP + FN
    right_illegal_num = 0  # 关键词匹配到的样本且标记结果正确的个数，TP
    all_illegal_num = 0  # 关键词匹配到的所有样本个数，TP + FP

    # 对每个数据样本
    for sentences in data['agent_sentences']:

        # 遍历其检测出的违规类型
        for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
            # 如果违规类型为要统计的类型且检测结果正确，总数量加1
            if rule == item and data['correctInfoData.correctResult'][counter].get("correctResult")[i] == '1':
                all_right_illegal_num += 1

        # 针对该样本统计遍历违规词，计算是否在句子中
        for key_word in key_words:
            # 违规词在句子中
            if key_word in sentences:
                flag = 0
                all_illegal_num += 1
                all_word_counter[key_word] = all_word_counter.get(key_word, 0) + 1
                # 遍历其检测出的违规类型
                for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
                    # 如果违规类型为要统计的类型且检测结果正确，正确结果加1
                    if rule == item and data['correctInfoData.correctResult'][counter].get("correctResult")[i] == '1':
                        right_illegal_num += 1
                        right_word_counter[key_word] = right_word_counter.get(key_word, 0) + 1
                        flag = 1
                    elif rule == item:
                        flag = 1
                if flag == 0:
                    print(data['UUID'][counter], key_word)
                # 检测出一个违规词，证明该样本已违规
                break
        counter += 1

    print(right_illegal_num, all_illegal_num, all_right_illegal_num)
    print(right_illegal_num / all_illegal_num, right_illegal_num / all_right_illegal_num)
    print(sorted(all_word_counter.items(), key=lambda word: word[1], reverse=True))
    print(sorted(right_word_counter.items(), key=lambda word: word[1], reverse=True))


if __name__ == "__main__":
    data = load_data()
    get_precision(data, "敏感词")
