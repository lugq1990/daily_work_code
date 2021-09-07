# -*- coding:utf-8 -*-
import pandas as pd
import pandas.io.sql as sql
import MySQLdb

conn = MySQLdb.connect(
    host='10.1.36.18',
    port=3306,
    user='zhanghui',
    passwd='zhanghui',
    db='model_data',
    charset='utf8',
)

try:
    sql_script = '''select mobile, recharge as recharge1, food as food1,
                    commodity as commodity1, clothing as clothing1, tools as tools1, elect as elect1,
                    publish as publish1, entert as entert1, equip as equip1, noness as noness1,
                    medicine as medicine1, other as other1
                    from train_data_X_cnn where day_no=1'''
    dat_X = sql.read_sql(sql_script, con=conn)

    for i in range(2, 54):
        code = tuple([i] * 13)
        dat1 = sql.read_sql('''select recharge as recharge%d, food as food%d,
                commodity as commodity%d, clothing as clothing%d, tools as tools%d, elect as elect%d,
                publish as publish%d, entert as entert%d, equip as equip%d, noness as noness%d,
                medicine as medicine%d, other as other%d
                from train_data_X_cnn where day_no=%d''' % code, con=conn)

        dat_X = pd.concat([dat_X, dat1], axis=1)

    dat_y = sql.read_sql('''select mobile, class_score from train_data_y_cnn''', con=conn)
    train_data = pd.merge(dat_X, dat_y, on="mobile", how="left")
finally:
    conn.commit()

train_data.to_csv('conver_result.csv', index=False)