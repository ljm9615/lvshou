import pandas as pd
import os
PATH1 = './.data/zhijian_data.csv'
PATH2 = './.data/zhijian_data_20180709.csv'

def load_data(path=PATH1):
    try:
        with open(path, 'rb') as fr:
            data = pd.read_csv(fr, sep=',', encoding="utf-8")
    except:
        with open(path, 'rb') as fr1:
            data = pd.read_csv(fr1, sep=',', encoding="gbk")
    finally:
        pass
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval)
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    return data

def divide_data(data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    all_rules = {}

    for i in range(len(data)):
        illegal_name = data['analysisData.illegalHitData.ruleNameList'][i]
        result = data['correctInfoData.correctResult'][i].get("correctResult")
        if '1' not in result:
            all_rules['不违规'] = all_rules.get('不违规', [])
            all_rules['不违规'].append(i)
            continue
        if len([0 for _ in result if _ =='1'])>1:
            all_rules['多类别'] = all_rules.get('多类别',[])
            all_rules['多类别'].append(i)
            continue
        for index, l in enumerate(illegal_name):
            if result[index] == '1':
                all_rules[l] = all_rules.get(l, [])
                all_rules[l].append(i)

    for _key, _value in all_rules.items():
        print(_key, len(_value))
        temp_data = data.iloc[_value]
        with open(path + '{}.csv'.format(_key.replace('/', '-')), 'w') as fw:
            temp_data.to_csv(fw, sep=',', index=False, encoding='utf-8')

if __name__ == "__main__":
    data1 = load_data(PATH1)
    print(data1.shape)
    data2 = load_data(PATH2)
    print(data2.shape)
    data = pd.concat([data1,data2])
    print(data.shape)
    del(data2, data1)
    data.drop_duplicates(['UUID'], inplace=True)
    data.reset_index(inplace=True)
    print(data.shape)
    path = "../../../zhijian_data/"
    divide_data(data, path)
