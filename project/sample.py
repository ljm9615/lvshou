import random
import pandas as pd
import os
from divide import load_data, PATH1, PATH2

random_rate = 0.2
random.seed(2018)
PATH = r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\sample"


def random_sample_data(data, file_name):
    length = data.shape[0]
    index = random.sample(range(length), int(float(length) * random_rate))
    uuid = data['UUID'][index]
    uuid.to_csv(os.path.join(PATH, file_name), sep=',', encoding="utf-8", index=False)


def sample_data_proportion(data, file_name):
    pass


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
    # random_sample_data(data, "sample1.txt")
