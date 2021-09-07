# -*- coding:utf-8 -*-
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.pipeline import Pipeline

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# read the data from Hive
df_org = spark.sql("""
select uuid,categroyname, rulename as rule from etl.bds_order_rule_hits_split_t where
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

# this is used to convert string to number
string_to_index_uuid = StringIndexer(inputCol='uuid', outputCol='uuid_num').fit(df_org)
df_uuid = string_to_index_uuid.transform(df_org)

string_to_index_rule = StringIndexer(inputCol='rule', outputCol='rule_num').fit(df_uuid)
df_converted = string_to_index_rule.transform(df_uuid).select(['uuid_num', 'rule_num'])

# use the SQL to get the model input data
df_converted.createOrReplaceTempView('t')
df = spark.sql('select uuid_num as uuid, collect_set(rule_num) as rule from t group by uuid').repartition(200)

# start to build the FP-Growth Model
fp = FPGrowth(itemsCol='rule', minSupport=.5, minConfidence=.5)
model = fp.fit(df)

# get the association rule DataFrame
association_relu = model.associationRules

print('This is the result')
association_relu.show()


