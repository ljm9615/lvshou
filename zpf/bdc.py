from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import pickle as pk
from collections import Counter
import os

# Bdc其实是一种有监督的词袋模型
class Featuers:
    
    def __init__(self, mindf=3, maxdf=0.9, maxfeature=1500,
                 label=None, Bdc=True, role='agent', debug=1):
        """计算bdc值
        Parameters
        ----------
        mindf : 最小文频
            int or float, float表示所占的比例
        maxdf : 最大文频
            int or float, float表示比例
        maxfeature : 最大特征数
            int
        label : 所要计算的二分类的bdc值的类别
            str(default None, 表示计算的是多分类的bdc值)
        role : 对话的角色, in ['agent', 'all', 'customer']
            str(default agent, 表示只有agent的对话)
        Bdc : 计算Bdc的公式
            bool(default True, 表示采用论文中的公式计算二分类Bdc值)
        """
        self.mindf, self.maxdf, self.maxfeature = mindf, maxdf, maxfeature
        self.label, self.Bdc, self.role, self.debug = label, Bdc, role, debug


    def getDF(self, data, labels, k=80):
        """ 将语料集用pd.DataFrame表示
        Parameters
        ----------
        data : 语料集
            list, like ['This is the first document.','This is the second second document.',\
            'And the third one.','Is this the first document?',]
        labels : 对应的类别信息
            list, numpy.array

        Returns
        -------
        df : token的DataFrame
            pd.DataFrame, index is token_id(int64), columns is label(str)
        vocab : token_id到token的映射
            dict, the format of the vocab like {token_id:token}
        """
        vec = CountVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2))
        # vec = CountVectorizer()
        data = vec.fit_transform(data)
        vocab = {j: i for i, j in vec.vocabulary_.items()}
        
        _label = np.unique(labels)
        labels_token = {} # <dict>{str:[list]}
        for i in tqdm(range(0, len(labels), k)):
            if i+k >= len(labels):
                temp = labels[i:]
                temp_data = data[i:]
            else:
                temp = labels[i:i+k]
                temp_data = data[i:i+k]
            for _ in _label:
                labels_token[_] = labels_token.get(_, 0)
                labels_token[_] += temp_data[np.array(temp) == _].toarray().sum(axis=0)
        del(i, _, data, vec, labels, _label)
        return pd.DataFrame(labels_token), vocab
    

    def calBdc(self, data, labels):
        if not os.path.exists('setting'):
            os.makedirs('setting')
        if not self.debug and os.path.exists('setting/{}data_vocab.pk'.format(self.role)):
            with open('setting/{}data_vocab.pk'.format(self.role),'rb') as fr:
                df, vocab = pk.load(fr)
        else:
            df, vocab = self.getDF(data, labels)
            with open('setting/{}data_vocab.pk'.format(self.role), 'wb') as fw:
                pk.dump([df, vocab], fw)
        labels_counter = Counter(labels)
        label_list = [labels_counter[i] for i in df.columns]

        # 扩展， 如果单纯的计算bdc可以将下列判断部分注释
        if self.label and self.Bdc: # 二分类bdc值
            label_list = [labels_counter[self.label], sum(labels_counter.values())]
            label_list[-1] -= label_list[0]
            df['负类'] = df.sum(axis=1) - df[self.label]
            df = df[[self.label, '负类']]
        
        elif self.label and not self.Bdc: # 使用公式计算二分类bdc值
            x = sum(labels_counter.values()) - labels_counter[self.label]
            label_list = [x if i != df.columns.index(self.label)\
             else j for i, j in enumerate(label_list)]
            
        assert len(label_list) == len(df.columns)

        # 计算Bdc
        temp_df = (df/label_list).apply(lambda x:x/sum(x), axis=1).applymap(lambda x: 0 if x==0 else x*np.log2(x))
        df['TF'] = df.sum(axis=1)
        df['BDC'] = round(temp_df.sum(axis=1)/np.log2(len(label_list)), 4) + 1
        df['Tokens'] = [vocab[i] for i in df.index]
        df.set_index(['Tokens'], inplace=True)
        df.to_csv('setting/{}_{}.csv'.format(self.role, self.label))
        return df


if __name__ == '__main__':
    corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?'
    ]
    labels = [0, 1, 2, 0]
    Bdc = Featuers()
    df = Bdc.calBdc(corpus, labels)
