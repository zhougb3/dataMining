# encoding=utf8

import sys  
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

reload(sys)  
sys.setdefaultencoding('utf-8')  

train = pd.read_csv('../data/train_featureV2.csv')
test = pd.read_csv('../data/test_featureV2.csv')
test_id = pd.DataFrame(test['uid'])
label_train = train['label']
test.fillna(0, inplace=True)
train.fillna(0, inplace=True)

train =train.drop(['uid'],axis=1)
train = train.drop(['label'],axis=1)
test = test.drop(['uid'],axis=1)

'''模型融合中使用到的各个单模型'''
clfs = [
        RandomForestClassifier(n_estimators=590, min_samples_split=18, max_depth=14, min_samples_leaf=10, oob_score=True, random_state=10),
        GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=13,min_samples_split=590,max_features=15,min_samples_leaf=85, random_state=10, subsample=0.9),
        ]

X = np.array(train)
X_predict = np.array(test)
y = np.array(label_train)

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

'''5折stacking'''
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train_index, test_index) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        print("Fold", i)
        X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test_index, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)


#使用xgboost来作为模型
#features = list(train.columns[0:]) 
xgb_model = xgb.XGBClassifier()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.01], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(y, n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(dataset_blend_train, y)

#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))


test_probitily = clf.predict_proba(dataset_blend_test)[:,1]
test_prediction = clf.predict(dataset_blend_test)[0:]
print test_prediction
test_id['label'] = test_prediction
test_id['prob'] = test_probitily
test_id = test_id.sort('prob',ascending=False)
test_id = test_id.drop(['prob'],axis=1)
test_id.to_csv('../result/result.csv',index = False)

