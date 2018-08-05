import sys
import time
import os
import numpy as np
import lightgbm as lgb
sys.path.append("..")
from project.interface import SupperModel
from sklearn.model_selection import KFold
PATH = "../../data/Content"


def load_data(rule, test_file, only=True, train=True):
    path = os.path.join(PATH, rule)
    if train:
        if only:
            weight = np.load(os.path.join(path, rule + "_train_weight_only_" + test_file + ".npy"))
            label = np.load(os.path.join(path, rule + "_train_label_only_" + test_file + ".npy"))
        else:
            weight = np.load(os.path.join(path, rule + "_train_weight_" + test_file + ".npy"))
            label = np.load(os.path.join(path, rule + "_train_label_" + test_file + ".npy"))
    else:
        weight = np.load(os.path.join(path, rule + "_test_weight_" + test_file + ".npy"))
        label = np.load(os.path.join(path, rule + "_test_label_" + test_file + ".npy"))
    return weight, label


class LGBM(SupperModel):
    def __init__(self, model, param, **kags):
        super(SupperModel, self).__init__()
        self.params = param
        self.model = model
        self.clf = None
        self.best_iters = []

    def cv(self, X_train, y_train, k_fold):
        print("using " + str(k_fold) + " cross validation...")
        # lgb_train = lgb.Dataset(X_train, y_train)
        # cv_result = self.model.cv(self.params, lgb_train, nfold=k_fold, shuffle=True, metrics="rmse",
        #                           early_stopping_rounds=early_stopping_rounds, seed=2018)
        # print(cv_result)
        kf = KFold(n_splits=k_fold)
        preds = []
        for train_idx, test_idx in kf.split(X_train):
            print(train_idx, test_idx)
            X = X_train[train_idx]
            y = y_train[train_idx]
            X_val = X_train[test_idx]
            y_val = y_train[test_idx]

            # lgb_train = lgb.Dataset(X, y)
            # lgb_eval = lgb.Dataset(X_val, y_val)
            _model = self.model(num_leaves=self.params.get("num_leaves"),
                                max_depth=self.params.get("max_depth"),
                                n_estimators=self.params.get("n_estimators"),
                                n_jobs=self.params.get("n_jobs"),
                                learning_rate=self.params.get("learning_rate"),
                                colsample_bytree=self.params.get("colsample_bytree"),
                                subsample=self.params.get("subsample"))
            lgb_model = _model.fit(
                X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=self.params.get("early_stopping_rounds"),
                verbose=0)

            print("predicting...")
            test_preds = lgb_model.predict(X_val)
            preds.extend(test_preds)
            self.best_iters.append(lgb_model.best_iteration_)
        print("validation result...")
        self.acc(y_train, preds)
        print("best iters:")
        print(self.best_iters)

    def train(self, X_train, y_train):
        _model = self.model(num_leaves=self.params.get("num_leaves"),
                            max_depth=self.params.get("max_depth"),
                            n_estimators=(sum(self.best_iters) // len(self.best_iters)),
                            n_jobs=self.params.get("n_jobs"),
                            learning_rate=self.params.get("learning_rate"),
                            colsample_bytree=self.params.get("colsample_bytree"),
                            subsample=self.params.get("subsample"))
        print("training...")
        print("iters:", str(_model.n_estimators))
        self.clf = _model.fit(X_train, y_train, verbose=1)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def saveModel(self, save_path):
        pass
    def loadModel(self, save_path):
        pass


if __name__ == "__main__":
    start_time = time.time()
    X_train, y_train = load_data("部门名称", test_file="sample1", only=False)
    param = {
        "num_leaves": 35,
        "max_depth": 7,
        "n_estimators": 20000,
        "n_jobs": 20,
        "learning_rate": 0.01,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "early_stopping_rounds": 100,
    }
    model = LGBM(lgb.LGBMClassifier, param)
    model.cv(X_train, y_train, 5)
    model.train(X_train, y_train)
    del X_train, y_train
    X_test, y_test = load_data("部门名称", test_file="sample1")
    preds = model.predict(X_test)
    # with open(r"E:\cike\lvshou\data\result_sample1.txt", 'w', encoding='utf-8') as f:
    #     for p in preds:
    #         f.write(str(p))
    #         f.write('\n')
    model.acc(y_test, preds)
    print('time cost is', time.time() - start_time)
