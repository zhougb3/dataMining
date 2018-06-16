# encoding=utf8

import sys  
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
  
reload(sys)  
sys.setdefaultencoding('utf-8')  

#传入通话的开始时间，结束时间，计算每次通话的时长
def getVoiceDura(startTime, endTime):
    #start time
    str1 = str(startTime).zfill(8)
    day1 = int(str1[0:2])
    hour1 = int(str1[2:4])
    minute1 = int(str1[4:6])
    second1 = int(str1[6:8])
    #end time
    str2 = str(endTime).zfill(8)
    day2 = int(str2[0:2])
    hour2 =  int(str2[2:4])
    minute2 = int(str2[4:6])
    second2 = int(str2[6:8])
    if (day2 > day1):
        hour2 += (day2 - day1) * 24
    duraTime = (hour2  - hour1) * 3600 + (minute2 - minute1) * 60 + (second2 - second1)
    return duraTime

#得到缺失的uid
def getLackingUid(hadUid):
    allUid = set(range(1,7000))
    for id in hadUid:
        item = int(id[1:]) #u0123，取出0123即可，转为int
        allUid.remove(item)
    return allUid

# 导入训练集的用户label
user_label = ['uid', 'label']
label_train = pd.read_csv('./uid_train.txt',header=None,encoding='utf-8',names = user_label,index_col = False,low_memory=False,sep='	')

#导入用户的网站/APP访问记录
user_wa = ['uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date']
wa_train = pd.read_csv("./wa_train.txt",header=None,encoding='utf-8',names = user_wa,index_col = False,low_memory=False,sep='	')
wa_test = pd.read_csv("./wa_test_a.txt",header=None,encoding='utf-8',names = user_wa,index_col = False,low_memory=False,sep='	')
wa_train = pd.concat([wa_train,wa_test])
wa_train.fillna(0, inplace=True)
#wa_train.dropna(axis=0,how='any',subset = user_wa,inplace = True)
wa_train['flow'] = wa_train['up_flow'] + wa_train['down_flow']

#得到每个用户的网站/app总访问次数，时长，流量
visit_cnt_by_id = wa_train.groupby('uid')['visit_cnt'].sum().reset_index()
visit_dura_by_id = wa_train.groupby('uid')['visit_dura'].sum().reset_index()
flow_by_id = wa_train.groupby('uid')['flow'].sum().reset_index()

#离散化
visit_cnt_by_id['visit_cnt'] = visit_cnt_by_id['visit_cnt'] // 100;
visit_dura_by_id['visit_dura'] = visit_dura_by_id['visit_dura'] // 1000;
flow_by_id['flow'] = flow_by_id['flow'] // 1000;

#得出每个用户对各个网站的总访问次数，时长，流量，并离散化
id_wa = (wa_train.groupby(['uid','wa_name','wa_type'])['visit_cnt','visit_dura','flow'].sum()).reset_index()
del id_wa['wa_name']
id_wa['visit_cnt'] = id_wa['visit_cnt'] // 100;
id_wa['visit_dura'] = id_wa['visit_dura'] // 1000;
id_wa['flow'] = id_wa['flow'] // 1000;

#得出每个用户访问次数最多的网站的信息，访问时长最多的网站信息，访问流量最多的网站信息
visit_cnt_by_id_wa = id_wa.sort('visit_cnt',ascending=False).groupby('uid',as_index = False).first()
visit_dura_by_id_wa = id_wa.sort('visit_dura',ascending=False).groupby('uid',as_index = False).first()
flow_by_id_wa = id_wa.sort('flow',ascending=False).groupby('uid',as_index = False).first()

# 得到网站信息的训练集
train = visit_cnt_by_id.merge(visit_dura_by_id,how='left',right_on='uid',left_on='uid')
train = train.merge(flow_by_id, how = 'left', right_on = 'uid', left_on = 'uid')
train = train.merge(visit_cnt_by_id_wa, how = 'left', right_on = 'uid', left_on = 'uid')
train = train.merge(visit_dura_by_id_wa, how = 'left', right_on = 'uid', left_on = 'uid')
train = train.merge(flow_by_id_wa, how = 'left', right_on = 'uid', left_on = 'uid')

#修改列名
train.columns = ['uid','visit_cnt', 'visit_dura', 'visit_flow', 'type1', 'visit_cnt1', 'visit_dura1', 'visit_flow1','type2', 'visit_cnt2', 'visit_dura2', 'visit_flow2','type3', 'visit_cnt3', 'visit_dura3', 'visit_flow3']

#读入通话记录
user_voice = ['uid','opp_num','opp_head','opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
voice_train = pd.read_csv("./voice_train.txt",header=None,encoding='utf-8',names = user_voice,index_col = False,low_memory=False,sep='	')
voice_test = pd.read_csv("./voice_test_a.txt",header=None,encoding='utf-8',names = user_voice,index_col = False,low_memory=False,sep='	')
voice_train = pd.concat([voice_train,voice_test])

#得到每个用户的总通话时长，每种类型的通话时长，呼入呼出各自的通话时长
voice_train['voice_time'] = voice_train.apply(lambda t: getVoiceDura(t['start_time'], t['end_time']), axis=1)
voice_time_sum_by_ID = voice_train.groupby(['uid'])['voice_time'].sum()
voice_time_sum_by_ID_type = voice_train.groupby(['uid', 'call_type'])['voice_time'].sum().unstack('call_type')
voice_time_sum_by_ID_type.fillna(0, inplace=True)
voice_time_sum_by_ID_IO = voice_train.groupby(['uid', 'in_out'])['voice_time'].sum().unstack('in_out')
voice_time_sum_by_ID_IO.fillna(0, inplace=True)

#得到每个用户的总通话次数
voice_time_count_by_ID = voice_train.groupby(['uid']).count()
voice_time_count_by_ID = voice_time_count_by_ID.reset_index()[['uid', 'opp_num']].set_index('uid')
voice_time_count_by_ID.rename(columns={'opp_num': 'voice_count'}, inplace=True)

#组合特征值
voice_train = pd.concat([voice_time_sum_by_ID,voice_time_sum_by_ID_type],axis=1)
voice_train.rename(columns={1: 'voice_type1', 2: 'voice_type2', 3: 'voice_type3', 4: 'voice_type4', 5: 'voice_type5'}, inplace=True)
voice_train = pd.concat([voice_train, voice_time_sum_by_ID_IO], axis=1)
voice_train.rename(columns={0: 'voice_out', 1: 'voice_in'}, inplace=True)
voice_train = pd.concat([voice_train, voice_time_count_by_ID], axis=1).reset_index()

# 补充数据，将缺失值的特征全部补充为0
lackUid = getLackingUid(voice_train['uid'])
for i in lackUid:
    i = 'u' + str(i).zfill(4)
    s = pd.Series({'uid': i, 'voice_time': 0, 'voice_out': 0, 'voice_in': 0, 'voice_type1': 0,
                   'voice_type2': 0, 'voice_type3': 0, 'voice_type4': 0, 'voice_type5': 0, 'voice_count': 0})
    voice_train = voice_train.append(s, ignore_index=True)

#读取用户的短信记录
user_sms = ['uid','opp_num','opp_head','opp_len','start_time','in_out']
sms_train = pd.read_csv("./sms_train.txt",header=None,encoding='utf-8',names = user_sms,index_col = False,low_memory=False,sep='	')
sms_test = pd.read_csv("./sms_test_a.txt",header=None,encoding='utf-8',names = user_sms,index_col = False,low_memory=False,sep='	')
sms_train = pd.concat([sms_train,sms_test])

#得到用户总的短信收发次数
sms_count_by_ID = sms_train.groupby(['uid']).count()
sms_count_by_ID = sms_count_by_ID.reset_index()[['uid', 'opp_num']].set_index('uid')
sms_count_by_ID.rename(columns={'opp_num': 'sms_count'}, inplace=True)
sms_count_by_ID.fillna(0, inplace=True)

#用户短信发出，接收的数量
sms_time_count_by_ID_IO = sms_train.groupby(['uid', 'in_out'])['opp_num'].count().unstack('in_out')
sms_time_count_by_ID_IO.fillna(0, inplace=True)
sms_time_count_by_ID_IO.rename(columns={0: 'sms_out', 1: 'sms_in'}, inplace=True)
sms_train = pd.concat([sms_count_by_ID, sms_time_count_by_ID_IO], axis=1).reset_index()

# 补充数据，将缺失值的特征全部补充为0
lackUid = getLackingUid(sms_train['uid'])
for i in lackUid:
    i = 'u' + str(i).zfill(4)
    s = pd.Series({'uid': i, 'sms_count': 0, 'sms_out': 0, 'sms_in': 0})
    sms_train = sms_train.append(s, ignore_index=True)

#合并全部特征值
train = train.merge(voice_train, how = 'left', right_on = 'uid', left_on = 'uid')
train = train.merge(sms_train, how = 'left', right_on = 'uid', left_on = 'uid')

#分开训练集和测试集
test = train[4999:]
train = train[:4999]
test_id = pd.DataFrame(test['uid'])
train = train.merge(label_train, how = 'left', right_on = 'uid', left_on = 'uid')
train =train.drop(['uid'],axis=1)
test = test.drop(['uid'],axis=1)
print train.tail(10)

#使用xgboost来作为模型
features = list(train.columns[0:27]) 
xgb_model = xgb.XGBClassifier()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(train['label'], n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train[features], train['label'])

#trust your CV!
#best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
#print('Raw AUC score:', score)
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))


test_probitily = clf.predict_proba(test)[:,1]
test_prediction = clf.predict(test)[0:]
test_id['label'] = test_prediction
test_id['prob'] = test_probitily
test_id = test_id.sort('prob',ascending=False)
test_id = test_id.drop(['prob'],axis=1)
test_id.to_csv('./result.csv',index = False)

print test_id.head(100)            