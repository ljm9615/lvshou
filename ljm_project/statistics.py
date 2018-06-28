import pandas as pd


def statistics():
    # 10264
    data = pd.read_csv(r"E:\cike\绿瘦\zhijian_data\zhijian_data.csv", sep=',')
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
        single_sentence = {sentence.get("role"): sentence.get("content")}
        sentence_content.append(single_sentence)
    return sentence_content


if __name__ == "__main__":
    # statistics()
    data = pd.read_csv(r"E:\cike\绿瘦\zhijian_data\zhijian_data.csv", sep=',')
    data['conversation'] = data['transData.sentenceList'].apply(eval).apply(get_sentences)
    data[['UUID', 'relateData.sourceCustomerId', 'relateData.workNo', 'analysisData.illegalHitData.ruleNameList',
          'analysisData.isIllegal', 'manualData.isChecked', 'correctInfoData.correctResult', 'conversation']]\
        .to_csv(r"E:\cike\绿瘦\zhijian_data\conversation.csv", sep=',', index=False, encoding='utf-8')

