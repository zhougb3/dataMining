
# coding: utf-8

import pandas as pd
import numpy as np

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

#传入通话或短信的开始时间，计算该时间是在白天还是晚上
def getTime(time):
    strTime = str(time).zfill(8)
    hour = int(strTime[2:4])
    if hour> 7 and hour < 23 :
        return 0
    return 1

#读取训练集数据
uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

#读取测试集数据
voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

#写入id
uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_b.txt',index=None)

#合并数据
voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)


# 处理通话记录

#每个用户通话总次数和不同通话方的人数
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()
#每个通话对方号码不同前n位的人数
voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()
#每种号码长度的通话次数
voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
#每种通话类型的通话次数
voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)
#每种主被叫类型的通话次数
voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)
#每次通话的通话时长
voice['voice_dura'] = voice.apply(lambda t: getVoiceDura(t['start_time'], t['end_time']), axis=1)
voice_dura = voice.groupby(['uid'])['voice_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_dura_').reset_index()
#每次通话的通话时间
voice['voice_time'] = voice.apply(lambda t: getTime(t['start_time']), axis=1)
voice_time = voice.groupby(['uid','voice_time'])['uid'].count().unstack().add_prefix('voice_time_').reset_index().fillna(0)
#通话次数最多的电话号码的通话次数
voice_max_num = voice.groupby(['uid','opp_num'])['opp_head'].count().reset_index().fillna(0)
voice_max_num = voice_max_num.sort('opp_head',ascending=False).groupby('uid',as_index = False).first()
del voice_max_num['opp_num']
voice_max_num['voice_max_num'] = voice_max_num['opp_head']
del voice_max_num['opp_head']

# 处理短信记录

#每个用户短信总次数和不同短信方的人数
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()
#每个短信对方号码不同前n位的人数
sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()
#每种号码长度的短信次数
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)
#每种类型的短信次数
sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)
#短信接收发送时间
sms['sms_time'] = sms.apply(lambda t: getTime(t['start_time']), axis=1)
sms_time = sms.groupby(['uid','sms_time'])['uid'].count().unstack().add_prefix('sms_time_').reset_index().fillna(0)
#短信收发次数最多的电话号码的短信收发次数
sms_max_num = sms.groupby(['uid','opp_num'])['opp_head'].count().reset_index().fillna(0)
sms_max_num = sms_max_num.sort('opp_head',ascending=False).groupby('uid',as_index = False).first()
del sms_max_num['opp_num']
sms_max_num['sms_max_num'] = sms_max_num['opp_head']
del sms_max_num['opp_head']

# 网站/APP记录

#用户访问网站总天数和不同网站总天数
wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
#用户45天内访问网站次数均值等
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()
#用户45天内访问网站时长均值等
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()
#用户45天内访问网站上行流量均值等
up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()
#用户45天内访问网站下行流量均值等
down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()
#访问次数最多的网站/App的访问次数
wa_max_num = wa.groupby(['uid','wa_name'])['visit_cnt'].sum().reset_index().fillna(0)
wa_max_num = wa_max_num.sort('visit_cnt',ascending=False).groupby('uid',as_index = False).first()
del wa_max_num['wa_name']
wa_max_num['wa_max_num'] = wa_max_num['visit_cnt']
del wa_max_num['visit_cnt']

wa_type = wa.groupby(['uid','wa_type'])['visit_cnt'].sum().unstack().add_prefix('wa_type_').reset_index().fillna(0)


feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_dura,voice_time,voice_max_num,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,sms_time,sms_max_num,wa_name,visit_cnt,visit_dura,up_flow,down_flow,wa_max_num,wa_type]


train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')

train_feature.to_csv('../data/train_featureV2.csv',index=None)
test_feature.to_csv('../data/test_featureV2.csv',index=None)

