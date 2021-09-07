# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.preprocessing import LabelEncoder
from pyspark.ml.fpm import FPGrowth

spark = SparkSession.builder.getOrCreate()

df_org = spark.sql("""
select uuid, rulename as rule from etl.bds_order_rule_hits_split_t where
decisionid not in ('credit_jb','loan_cpm28','loan_cpm37')
and flow_type=1 and hit = true
and categroyname not in ('地址空值校验','淘宝变量空值校验','设备指纹变量空值校验','资信云报告变量空值校验','运营商变量空值校验')
and rulename not in
('BqsCreditReport_rule01','BqsCreditReport_rule02','BqsCreditReport_rule03','BqsCreditReport_rule04','BqsCreditReport_rule05',
'BqsCreditReport_rule06','BqsCreditReport_rule07','BqsCreditReport_rule08','BqsCreditReport_rule09','black_list_rule01',
'black_list_rule05','bqs_time_status_rule01','bqs_time_status_rule02','inside_data_rule01','inside_data_rule02','inside_data_rule03',
'rule','rule01','rule02','rule03','rule04','rule05','rule06','rule_age','rule_bingjian_huomou','rule_end','rule_zmxy_verify',
'rule_zmxy_watchlistii','strategy_addrCheck','strategy_consumingPower','strategy_deviceId','strategy_list01','strategy_list02',
'strategy_list03','strategy_list04','strategy_list05','strategy_list06','strategy_list07','strategy_list08','strategy_list09',
'strategy_list10','strategy_list11','strategy_list12','strategy_list13','strategy_list14','strategy_list15','strategy_list16',
'strategy_list17','strategy_list18','strategy_list19','strategy_list20','strategy_list21','strategy_list22','strategy_list23',
'strategy_list24','strategy_list25','strategy_list26','strategy_list27','strategy_list28','strategy_list29','strategy_list30',
'strategy_list31','strategy_list32','strategy_list33','strategy_list34','strategy_list35','strategy_risk01','strategy_risk02',
'strategy_risk03','strategy_risk04','strategy_risk06','strategy_risk08','strategy_risk09','strategy_risk10','strategy_risk11',
'strategy_risk12','strategy_risk13','strategy_risk15','strategy_risk16','strategy_risk18','strategy_risk20','strategy_risk21',
'strategy_riskCall','strategy_socialNet','yf_black_rule01','yf_black_rule02','yf_black_rule03','yf_black_rule04','yf_black_rule05',
'yf_black_rule06','yf_black_rule07','zmxy_antifraudscore_rule','zmxy_risklist_rule01','zmxy_risklist_rule02','zmxy_risklist_rule03',
'zmxy_risklist_rule04','zmxy_score_rule1','zmxy_score_rule3','zmxy_score_rule4')
""")
df = df_org.toPandas()
df.columns = ['uuid', 'rule']

# convert the string column to numerical number
le_uuid = LabelEncoder().fit(df['uuid'])
le_rule = LabelEncoder().fit(df['rule'])

uuid = le_uuid.transform(df['uuid']).reshape(-1, 1)
rule = le_rule.transform(df['rule']).reshape(-1, 1)

concat = np.concatenate((uuid, rule), axis=1)
# convert to numpy array to pandas DataFrame
data = pd.DataFrame(concat, columns=['uuid', 'rule'])

df_spark = spark.createDataFrame(data)
df_spark.createOrReplaceTempView('t')

# convert the each person with all the rules number for FP-Growth model
train = spark.sql('select collect_set(rule) as item from t group by uuid')

# start to train the model
model = FPGrowth(itemsCol='item', minConfidence=.01, minSupport=.1, numPartitions=3).fit(train)

# get the frequent rule, this is a DataFrame
freq_rule = model.freqItemsets.toPandas()

print('Finished training')
def _conver(x):
    return le_rule.inverse_transform(x)

ch_rule = freq_rule['items'].map(lambda x:_conver(x)).reshape(-1, 1)
ch_rule = pd.DataFrame(ch_rule, columns=['items'])

freq_result = pd.concat((ch_rule, freq_rule['freq']), axis=1)


# get the association rules
association_rule = model.associationRules.toPandas()

antecedent = association_rule['antecedent'].map(lambda x: _conver(x)).reshape(-1, 1)
antecedent = pd.DataFrame(antecedent, columns=['antecedent'])
consequent = association_rule['consequent'].map(lambda x: _conver(x)).reshape(-1, 1)
consequent = pd.DataFrame(consequent, columns=['consequent'])

asso_result = pd.concat((antecedent, consequent, association_rule['confidence']), axis=1)

print('All done!')
freq_result.head()
asso_result.head()