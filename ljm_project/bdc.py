import jieba
import random

random.seed(2018)


jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\敏感词.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\部门名称.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\禁忌称谓.txt")