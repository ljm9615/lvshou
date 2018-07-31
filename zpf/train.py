import numpy as np
np.random.seed(2018)
import pickle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
import lightgbm as lgb
from Stacking import SBBTree
import features
import os

PATH_TN = '../../../zhijian_data/Token'
CV = 5
def acc(clf, Y, Y_pred):
    Y = list(Y); Y_pred = list(Y_pred)
    print(clf + 'accuracy:', accuracy_score(Y, Y_pred))
    print(clf + 'recall:', recall_score(Y, Y_pred))
    print(clf + 'micro_F1:', f1_score(Y, Y_pred, average='micro'))
    print(clf + 'macro_F1:', f1_score(Y, Y_pred, average='macro'))

class Model(object):
    def __init__(self, path_train=PATH_TN,  learning_rate=0.01, n_estimators=1500,
                 max_depth=13, min_child_weight=1, gamma=0.1, subsample=0.7, colsample_bytree=0.6,
                 objective='binary:logistic', seed=2018,
                 label='', tfbdc=False, debug=True, bdc=True, qz='DT', opt='bdc'):
        # qz in ('DT', 'sum', 'icf')
        self.xgboost = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
	        min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
	        objective=objective, seed=seed, n_jobs=60)
        self.gbdt = GradientBoostingClassifier()
        self.rf = RandomForestClassifier()
        self.label = label
        self.model = [lgb, self.xgboost, self.gbdt, self.rf, lgb]
        # self.model = [lgb,lgb]
        self.SBBTree = SBBTree(model=self.model, bagging_num=5)
        self.BDC = bdc
        BDC = features.Features(tfbdc=tfbdc, debug=debug, BDC=self.BDC, qz=qz)
        Vocab_bdc = BDC.load_bdc(PATH_TN, label=self.label)
        self._X, self._Y, label_to_id, self.uuid = BDC.load_X_Y(Vocab_bdc, path=PATH_TN, label=self.label, froms='', opt=opt)
        self._X, self._Y, self.uuid = np.array(self._X), np.array(self._Y), np.array(self.uuid)
        self.kf = KFold(n_splits=CV, random_state=2018)
        self.id_to_lable = {i:_ for _,i in label_to_id.items()}

    def writeFile(self,tests,preds,uuids):
        if not os.path.exists('./setting/model'):
            os.makedirs('./setting/model')
        with open('./setting/model/{}_BDC_{}.txt'.format(self.label,self.BDC),'w+') as f:
            for i, j, k in zip(uuids, tests, preds):
                if j != k:
                    f.write("{},{},{}\n".format(i, self.id_to_lable.get(j, '不违规'),self.id_to_lable.get(k, '不违规')))

    def generalCV(self,index):
        self._X = self._X[index]; self._Y=self._Y[index]; self.uuid = self.uuid[index]
        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=2018)
        preds, tests, uuids = [], [], []
        for k, (train_index, test_index) in enumerate(skf.split(self._X, self._Y)):
            X_train = self._X[train_index]; y_train = self._Y[train_index]
            X_test = self._X[test_index]; y_test = self._Y[test_index]
            gbm = self.xgboost.fit(X_train, y_train)
            pred_y = gbm.predict(X_test)
            print('第{}折的实验结果:'.format(k))
            acc('%%', y_test, pred_y.round())
            preds.extend(pred_y.round()); tests.extend(y_test); uuids.extend(self.uuid[test_index])
        print('汇总之后的实验结果为:')
        acc('%%', tests, preds)
        self.writeFile(tests, preds, uuids)

    def stacking(self, index):
        train_index, test_index = train_test_split(index, test_size=0.4, random_state=2017)
        X_train, X_test = self._X[train_index], self._X[test_index]
        Y_train, Y_test = self._Y[train_index], self._Y[test_index]
        uuid_test = self.uuid[test_index]
        # print(np.shape(X_train), np.shape(Y_train))
        print('avg(auc):', self.SBBTree.fit(X_train, Y_train))
        Y_pred = self.SBBTree.predict(X_test)
        acc('dev', Y_test, Y_pred.round())
        self.writeFile(Y_test,Y_pred.round(),uuid_test)


    def evl(self):
        index = np.array(range(len(self._X)))
        for _ in range(9):
            np.random.shuffle(index)
        # self.stacking(index)
        self.generalCV(index)

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            mdl = self.xgboost
            pickle.dump(mdl, f)

    def load(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.xgboost = pickle.load(f)


if __name__ == "__main__":
    import time
    start = time.time()
    mymodel = Model(label='无中生有', debug=True, bdc=True, qz='', opt='bdc')
    mymodel.evl()
    # mymodel.save('./setting/model/xgboost.pk')
    print('本次训练耗时:{}'.format(time.time()-start))