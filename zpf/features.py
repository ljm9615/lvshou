import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle as pk
import time
D = 20000
SAMPLE = 700
MIN_DF = 30
MAX_DF = 0.5
class Features:
    def __init__(self, BDC=True, debug=True, tfbdc=False, qz='DT'):
        self.tfdbc = tfbdc;self.debug = debug;self.BDC = BDC
        self.count_vec = CountVectorizer(input='filename', max_df=MAX_DF, min_df=MIN_DF, max_features=10000)
        self.tfidf_vec = TfidfVectorizer(max_df=MAX_DF, min_df=MIN_DF)
        self.max_df = MAX_DF; self.min_df = MIN_DF
        self.qz = qz

    def load_token(self, filename):
        tokens = []
        with open(filename, encoding='utf-8') as f:
            for line in f.readlines():
                tokens.extend(line.strip().split())
        return tokens

    def getVocabByDT(self, path):
        _dirs = os.listdir(path)
        _dirs = [_ for _ in _dirs if os.path.isdir(os.path.join(path,_))]
        _labels, _files = [], []
        for _ in _dirs:
            _path = os.path.join(path,_)
            files = os.listdir(_path)
            files = [os.path.join(_path,_) for _ in files if os.path.isfile(os.path.join(_path,_))]
            _files.extend(files)
        assert len(_files) == 43292
        self.count_vec.fit_transform(_files)
        return list(self.count_vec.vocabulary_.keys())

    def get_corpus(self, path):
        label_tokens = {}
        dirs = os.listdir(path)
        label_to_id = {_:i for i, _ in enumerate(dirs) if os.path.isdir(os.path.join(path, _))}
        id_to_label = {_:i for i, _ in label_to_id.items()}
        label_list = [[] for _ in label_to_id.keys()]  # label_list表示类别的文档数
        for _ in label_to_id.keys():
            c_dir = os.path.join(path, _)
            files = os.listdir(c_dir)
            tokens = []; c_num = 0 
            for i in files:
                if os.path.isfile(os.path.join(c_dir, i)):
                    tokens.extend(self.load_token(os.path.join(c_dir, i)))
                    # 处理好一篇文档
                    c_num += 1
            label_tokens[label_to_id[_]] = Counter(tokens)
            label_list[label_to_id[_]] = c_num

        return label_tokens, label_list, id_to_label

    def get_df(self, label_tokens):
        id_s = list(label_tokens.keys())
        Vocab = []
        for i in id_s:
            Vocab.extend(label_tokens[i].keys())
        Vocab = list(set(Vocab))
        label_words = {}
        for i in id_s:
            label_words[i] = [label_tokens[i].get(_, 0) for _ in Vocab]
        
        W_L_DF = pd.DataFrame(data=label_words)

        return W_L_DF, Vocab
    
    def H(self, x):
        return [_ * np.log2(_) if _ != 0 else 0 for _ in x]

    # bdc
    def get_bdc(self, W_L_DF, Vocab, label_list, axis=-1):
        '''
        :param W_L_DF: 词-类别矩阵<pd.DataFrame>
        :param Vocab: 词汇表<list>
        :param label_list: 每个类别的文档数<list>
        :param axis: 类别所在的位置<int:0-#+1 of labels>
        :return:词以及对应的bdc值<dict>
        '''
        # 直接赋值是一种浅拷贝的现象，会出现错误
        new_W_L_DF = W_L_DF[:]; new_tlists = label_list[:]
        temp = W_L_DF.apply(lambda x:len([1 for _ in x if _ > 0]), axis=1)
        if axis != -1:#2分类
            x = sum(label_list) - label_list[axis]
            if self.BDC:
                new_W_L_DF['sum'] = new_W_L_DF.sum(axis=1)
                new_W_L_DF['negative'] = new_W_L_DF['sum'] - new_W_L_DF[axis]
                new_W_L_DF = new_W_L_DF[[axis, 'negative']]
                new_tlists = [label_list[axis], x]
            else:
                new_tlists = [x if _!=axis else __ for _,__ in enumerate(new_tlists)]

        # B(x)
        new_W_L_DF = new_W_L_DF/new_tlists
        # normalize->p
        new_W_L_DF = new_W_L_DF.apply(lambda x: x/sum(x), axis=1)
        # H(p)
        new_W_L_DF = new_W_L_DF.apply(self.H)
        # sum
        W_L_DF['sum'] = W_L_DF.sum(axis=1)
        # W_L_DF['icf'] = round(np.log2((len(new_tlists) if axis==-1 else 2)/temp),4)
        W_L_DF['icf'] = temp
        W_L_DF['bdc'] = new_W_L_DF.sum(axis=1)
        W_L_DF['bdc'] = round(W_L_DF['bdc']/np.log2(len(new_tlists)) + 1,4)
        W_L_DF['word'] = Vocab
        W_L_DF.set_index(['word'], inplace=True)
        if axis != -1:
            W_L_DF['negative'] = W_L_DF['sum'] - W_L_DF[axis]
            W_L_DF['positive'] = W_L_DF[axis]
            return W_L_DF[['positive','negative','sum','bdc','icf']]
        return W_L_DF

    # generate X-data|Y-label_list
    def load_X_Y(self, Vocab_bdc, path, label=''):
        '''
        :param Vocab_bdc: 词及bdc值<pd.DataFrame>
        :param path: 数据集路径<str>
        :param label: 类别标签<str>
        :return:
        '''
        X, Y, UUIDS = [], [], []
        num = 0
        dirs = os.listdir(path)
        dirs = [_ for _ in dirs if os.path.isdir(os.path.join(path, _))]
        if label == '': # 多分类
            labels_id = {_: i for i, _ in enumerate(dirs)}
        else: # 2分类
            labels_id = {label: 1}
        # id_labels = {i: _ for _, i  in labels_id.items()}
        files, labels = [], []
        if label == '':
            for _ in dirs:
                _files = os.listdir(os.path.join(path, _))
                _files = [os.path.join(_,__) for __ in _files]
                np.random.shuffle(_files)
                files.append(_files[:len(_files)//10])
                labels.append(_)
        else:
            temp_files = []
            for _ in dirs:
                _files = os.listdir(os.path.join(path, _))
                _files = [os.path.join(_, __) for __ in _files]
                if _ == label:
                    files.append(_files)
                    labels.append(_)
                else:
                    temp_files.extend(_files)
            for xxx in range(9):
                np.random.shuffle(temp_files)
            files.append(temp_files[:len(files[0])])
            labels.append('')
            del(temp_files)
        del(_files)
        for i,_ in enumerate(files):
            for _file in _:
                _x = []
                if os.path.isfile(os.path.join(path, _file)):
                    _x_token = self.load_token(os.path.join(path, _file))
                    _x_token_counter = Counter(_x_token)
                    if self.tfdbc:
                        _x_token_counter = {i: j for i, j in _x_token_counter.items()}
                    else:
                        _x_token_counter = {i: 1 for i in _x_token_counter.keys()}
                    # _x = [_x_token_counter.get(i, 0) for i in Vocab_bdc.index]
                    _x = [_x_token_counter.get(i, 0)*j for i, j in zip(Vocab_bdc.index, Vocab_bdc['bdc'])]
                    # _x.extend([_x_token_counter.get(i,0) for i in Vocab_bdc.index])
                    # _x.append(len(_x_token))
                    X.append(_x)
                    _x = None;del(_x_token)
                    Y.append(labels_id.get(labels[i], 0))
                    UUIDS.append(_file.split('/')[-1].split('-')[0])

        return X, Y, labels_id, UUIDS
        
    def load_bdc(self, path_train, label=''):
        _file = './setting/all.pk'
        _file1 = './setting/{}_Vocab_bdc_{}.pk'.format(label,self.BDC)
        if not os.path.exists('./setting'):
            os.makedirs('./setting')
        if os.path.exists(_file):
            with open(_file, 'rb') as f:
                label_tokens = pk.load(f)
                label_list = pk.load(f)
                id_to_label = pk.load(f)
                W_L_DF = pk.load(f)
                Vocab = pk.load(f)
        else:
            label_tokens, label_list, id_to_label = self.get_corpus(path_train)
            W_L_DF, Vocab = self.get_df(label_tokens)
            with open(_file, 'wb') as fw:
                pk.dump(label_tokens, fw)
                pk.dump(label_list, fw)
                pk.dump(id_to_label, fw)
                pk.dump(W_L_DF, fw)
                pk.dump(Vocab, fw)
        if os.path.exists(_file1) and not self.debug:
            with open(_file1, 'rb') as f:
                Vocab_bdc = pk.load(f)
        else:
            label_to_id = {_:i for i, _ in id_to_label.items()}
            Vocab_bdc = self.get_bdc(W_L_DF, Vocab, label_list, label_to_id.get(label, -1))
            print('Before filter the shape of Vocab_bdc is', np.shape(Vocab_bdc))
            # 此处有失逻辑
            if self.qz in ('sum', 'icf'):
                Vocab_bdc = Vocab_bdc.sort_values(by=[self.qz])
                V_L = Vocab_bdc.shape[0]
                if type(self.min_df) == type(self.max_df):
                    if type(self.min_df) is float:
                        Vocab_bdc = Vocab_bdc.iloc[int(self.min_df*V_L):int(self.max_df*V_L)]
                    else:
                        Vocab_bdc = Vocab_bdc[Vocab_bdc[self.qz]>=self.min_df]
                        Vocab_bdc=Vocab_bdc[Vocab_bdc[self.qz]<=self.max_df]
                else:
                    if type(self.min_df) is float:
                        Vocab_bdc = Vocab_bdc.iloc[int(self.min_df * V_L):]
                        Vocab_bdc = Vocab_bdc[Vocab_bdc[self.qz] <= self.max_df]
                    else:
                        Vocab_bdc = Vocab_bdc[Vocab_bdc[self.qz] >= self.min_df]
                        Vocab_bdc = Vocab_bdc.iloc[:int(self.max_df * V_L)]
            elif self.qz in ('DT'):
                words = self.getVocabByDT(path_train)
                Vocab_bdc = Vocab_bdc.loc[words]
            print('After filter the shape of Vocab_bdc is', Vocab_bdc.shape)
            Vocab_bdc = Vocab_bdc.sort_values(by=['bdc'])
            Vocab_bdc = Vocab_bdc.tail(D)
            print('After filter the shape of Vocab_bdc is', Vocab_bdc.shape)
            Vocab_bdc.to_csv("./setting/{}_{}_cipin.csv".format(label,self.BDC), sep=',',index=True, encoding='utf-8')
            with open(_file1, 'wb') as fw:
                pk.dump(Vocab_bdc, fw)

        return Vocab_bdc

if __name__ == '__main__':
    start_time = time.time()
    BDC = Features()
    # BDC.load_bdc('../../../zhijian_data/Token')
    # BDC.get_corpus1('../../../zhijian_data/Token')
    print('time cost is',time.time()-start_time)
