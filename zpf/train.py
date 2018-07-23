import numpy as np
import pickle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score
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

class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""
    def __init__(self, model, bagging_num, num_boost_round=20000, early_stopping_rounds=50):
        self.lgb_params = {
            'num_leaves': 50,
            'max_depth': 8,
            'subsample': 0.85,
            'subsample_freq': 1,
            'verbosity': -1,
            'colsample_bytree': 0.85,
            'min_child_weight': 50,
            'nthread': 4,
            'seed': 2017,
            'boosting_type': 'rf',
            'objective': 'binary',
            'metric': {'auc'},
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'is_unbalance': True,
            'lambda_l1': 0.5,
            'lambda_l2': 35
        }
        self.bagging_num = bagging_num
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = model
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        if len(self.model) >= 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=len(self.model), shuffle=True, random_state=1)
            for _, model in enumerate(self.model):
                for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                    X_train = X[train_index]
                    y_train = y[train_index]
                    X_test = X[test_index]
                    y_test = y[test_index]
                    if _ == 0 or _ == len(self.model)-1:
                        lgb_train = model.Dataset(X_train, y_train)
                        lgb_eval = model.Dataset(X_test, y_test, reference=lgb_train)
                        gbm = model.train(self.lgb_params,lgb_train,num_boost_round=self.num_boost_round,
                        valid_sets = lgb_eval,early_stopping_rounds=self.early_stopping_rounds,verbose_eval=False)
                        pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                    else:
                        gbm = model.fit(X_train, y_train)
                        pred_y = gbm.predict(X_test)
                    self.stacking_model.append(gbm)
                    layer_train[test_index, 1] = pred_y
                X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass
        cv_auc = []
        self.SK_b = StratifiedKFold(n_splits=self.bagging_num, shuffle=True, random_state=1)
        for _, model in enumerate(self.model):
            for k, (train_index, test_index) in enumerate(self.SK_b.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]
                if _ == 0 or _ == len(self.model)-1:
                    lgb_train = lgb.Dataset(X_train, y_train)
                    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
                    gbm = lgb.train(self.lgb_params,lgb_train,num_boost_round=self.num_boost_round,
                    valid_sets=lgb_eval,early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
                    pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                else:
                    gbm = model.fit(X_train, y_train)
                    pred_y = gbm.predict(X_test)
                self.bagging_model.append(gbm)
                print('第{}次第{}折的实验结果:'.format(_, k))
                acc('%%', y_test, pred_y.round())
                cv_auc.append(roc_auc_score(y_test, pred_y))
        return np.mean(cv_auc)

    def predict(self, X_pred):
        """ predict test data. """
        print(len(self.stacking_model), len(self.model), len(self.bagging_model))
        X_pred = np.array(X_pred)
        if len(self.model) >= 1:
            for _ in range(len(self.model)):
                print(X_pred.shape)
                for sn in range(len(self.model)):
                    test_pred = np.zeros((np.shape(X_pred)[0], len(self.model)))
                    gbm = self.stacking_model[_ * len(self.model) + sn]
                    try:
                        gbm.best_iteration
                        pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                    except:
                        pred = gbm.predict(X_pred)
                    test_pred[:, sn] = pred
                X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))
        else:
            pass
        # 普通cv
        for bn, gbm in enumerate(self.bagging_model):
            try:
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
            except:
                pred = gbm.predict(X_pred)
            pred = np.array(pred)
            pred_out = pred if bn == 0 else pred_out + pred
        return pred_out / len(self.bagging_model)

class Model(object):
    def __init__(self, path_train=PATH_TN,  learning_rate=0.01, n_estimators=1500,
                 max_depth=13, min_child_weight=1, gamma=0.1, subsample=0.7, colsample_bytree=0.6,
                 objective='binary:logistic', seed=27, nthread=24,
                 label='', tfbdc=False, debug=True, bdc=True, qz='DT'):
        self.xgboost = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
	        min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
	        objective=objective, seed=seed,nthread=nthread)
        self.gbdt = GradientBoostingClassifier()
        self.rf = RandomForestClassifier()
        self.label = label
        self.model = [lgb, self.xgboost, self.gbdt, self.rf, lgb]
        # self.model = [lgb,lgb]
        self.SBBTree = SBBTree(model=self.model, bagging_num=5)
        self.BDC = bdc
        BDC = features.Features(tfbdc=tfbdc, debug=debug, BDC=self.BDC, qz=qz)
        Vocab_bdc = BDC.load_bdc(PATH_TN, label=self.label)
        self._X, self._Y, label_to_id, self.uuid = BDC.load_X_Y(Vocab_bdc, path=PATH_TN, label=self.label)
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

    def generalCV(self, index):
        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=1)
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
    mymodel = Model(label='无中生有', debug=True, bdc=True)
    mymodel.evl()
    # mymodel.save('./setting/model/xgboost.pk')
    print('本次训练耗时:{}'.format(time.time()-start))
