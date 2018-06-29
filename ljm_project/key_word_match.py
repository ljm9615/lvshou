import pandas as pd


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
    counter = 0
    all_illegal_data = 0
    illegal_data = 0
    for sentences in data['agent_sentences']:
        if rule in data['analysisData.illegalHitData.ruleNameList'][counter]:
            all_illegal_data += 1
        for key_word in key_words:
            if key_word in sentences:
                if rule in data['analysisData.illegalHitData.ruleNameList'][counter]:
                    illegal_data += 1
                    break
        counter += 1
    print(illegal_data, all_illegal_data)


if __name__ == "__main__":
    data = load_data()
    get_precision(data, "部门名称")
