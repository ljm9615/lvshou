# encoding=utf-8
import jieba
import os
from divide import load_data
import time
import thulac
import re
import pandas as pd

def save_file(data, path):
    with open(path, 'wb+') as fw:
        fw.write(data.encode('utf-8', 'ignore'))

# 返回停用词列表
def get_stopwords(path='./setting/stopwords.txt'):
    stop_file = path
    if not os.path.exists(stop_file):
        return []
    with open(stop_file, 'rb+') as fr:
        data = fr.read()
    stopwords = [_ for _ in data.decode('utf-8').strip().split('\n')]
    del(data)
    return stopwords

# 单个文件分词
# inFile, outFile为完整路径(unicode)
def fenci_file(inFile, outFile):
    # stopwords = get_stopwords()
    with open(inFile, 'rb') as fin:
        contens = fin.read().decode('utf-8')
    for _token in get_stopwords():
        contens = contens.replace(_token, "")
    for _token in get_stopwords('./setting/stopre.txt'):
        pattern = re.compile(r'%s' % _token)
        contens = pattern.sub('', contens)
    words = list(jieba.cut(contens, cut_all=False))
    words = [word for word in words if len(word) > 1 and word != '\n']
    # words = [word for word in words if word not in stopwords]
    with open(outFile, "wb") as fout:
        fout.write(" ".join(words).encode('utf-8', 'ignore'))

class Tokens:
    def __init__(self, alone=False):
        '''
        :param alone:是否将user和agent的对话严格分开
        '''
        self.path = '../../../zhijian_data'
        self.content = 'Content'
        self.token = 'Token'
        self.alone = alone

    def makeContents(self):
        _files = os.listdir(self.path)
        _files = [_ for _ in _files if '.csv' in _] # 所有的文件
        _labels = [os.path.splitext(_)[0] for _ in _files] # 所有违规标签
        for i, _file in enumerate(_files):
            print(i+1, _labels[i])
            prepath = os.path.join(self.path, self.content, _labels[i])
            if not os.path.exists(prepath):
                os.makedirs(prepath)
            _file_df = load_data(os.path.join(self.path, _file))
            if 'transData.sentenceList' not in _file_df.columns:
                continue
            _file_df['transData.sentenceList'] = _file_df['transData.sentenceList'].apply(eval)
            for _id in range(len(_file_df)):
                uuid = _file_df['UUID'][_id]
                sentenceList = _file_df['transData.sentenceList'][_id]
                if not self.alone:
                    _contents = ['{}:{}'.format(_['role'], _['content']) for _ in sentenceList]
                else:
                    _contents = ['{}'.format(_['content']) for _ in sentenceList if _['role'] == 'AGENT']
                contens = '\n'.join(_contents);_contents=None
                save_file(contens, os.path.join(prepath, '{}-{}.txt'.format(uuid, _labels[i])))
        print('save all contents completed!')
    
    def makeToken(self):
        # thu1 = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注
        # thu1.cut_f("input.txt", "output.txt")  # 对input.txt文件内容进行分词，输出到output.txt
        jieba.load_userdict('./setting/userdict1.txt')
        _files = os.listdir(self.path)
        _files = [_ for _ in _files if '.csv' in _] # 所有的文件
        _labels = [os.path.splitext(_)[0] for _ in _files] # 所有违规标签
        for i, _file in enumerate(_files):
            print(i+1, _labels[i])
            _content_prepath = os.path.join(self.path, self.content, _labels[i])
            _token_prepath = os.path.join(self.path, self.token, _labels[i])
            if not os.path.exists(_token_prepath):
                os.makedirs(_token_prepath)
            _content_files = os.listdir(_content_prepath)
            _content_files = [_ for _ in _content_files if '.txt' in _] # 所有的文件
            for _ in _content_files:
                # print(_)
                # thu1.cut_f(os.path.join(_content_prepath, _), os.path.join(_token_prepath, _))
                fenci_file(os.path.join(_content_prepath, _), os.path.join(_token_prepath, _))
        print('Make Tokens of all files completed')

if __name__ == '__main__':
    start_time = time.time()
    _tokens = Tokens(alone=True)
    # print(get_stopwords())
    _tokens.makeContents()
    _tokens.makeToken()
    print(time.time()-start_time)
