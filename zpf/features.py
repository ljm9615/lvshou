import bdc
from divide import load_data
import pandas as pd
import os

PATH1 = '../../zhijian_data/zhijian_data.csv'
PATH2 = '../../zhijian_data/zhijian_data_20180709.csv'

class PreProcess:
    def __init__(self, label=None, role='agent', debug=1):
        '''
        :param label:
        :param role: in ['agent', 'all', 'customer']
        '''
        self.label, self.role, self.debug = label, role, debug
        self.BDC = bdc.Featuers(label=label, role=role, debug=debug)

    def load_data(self, path):
        _labels = os.listdir(path)
        _files = [os.path.join(i,i+'_tokens.csv') for i in _labels]
        _files = [os.path.join(path, i) for i in _files]
        return _files, _labels

    def getBdc(self, path='../../data/Content'):
        if not self.debug and os.path.exists('setting/{}_{}.csv'.format(self.role, self.label)):
            return pd.read_csv('setting/{}_{}.csv'.format(self.role, self.label), index_col=0)
        _files, _labels = self.load_data(path)
        print(_files, _labels)
        data, labels = [], []
        for i, _ in enumerate(_files):
            temp = pd.read_csv(_, index_col=0)['transData.sentenceList']
            print(_labels[i], len(temp))
            data.extend(temp); labels.extend([_labels[i]]*len(temp))
        print('数据总量为', len(labels))
        return self.BDC.calBdc(data, labels)

    def load_all_data(self):
        data1 = load_data(PATH1)
        print(data1.shape)
        data2 = load_data(PATH2)
        print(data2.shape)
        # 合并数据
        data = pd.concat([data1, data2])
        print(data.shape)
        del (data2, data1)
        data.drop_duplicates(['UUID'], inplace=True)
        data.reset_index(inplace=True)
        print(data.shape)
        data[['UUID', '']]


if __name__ == '__main__':
    P = PreProcess(role='agent', label='无中生有')
    P.getBdc()