# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# path = "F:\\"
# data = pd.read_csv(path+"\\words.csv")
#
# a = np.array(["id_card","id_5m_qy_n","id_4m_qy_n","id_2m_order_avg", "id_4m_order_avg","id_6m_order_avg","id_3m_order_avg"
#     ,"id_5m_order_avg","id_1m_order_avg","id_tel_n","id_5m_loanM_avg","relat_id_same","id_1m_uid_avg"
#     ,"id_4m_loanM_avg","id_6m_loanM_avg","id_5m_order_n","id_2m_uid_avg","id_3m_loanM_avg","flow_time","stat_date"])
#
#
# tmp = np.array(data)
# tmp2 = tmp.tolist()
# print(tmp2)
# for i in range(tmp.shape[0]):
#     for j in range(a.shape[0]):
#         if tmp[i][0] == a[j]:
#             print(tmp[i][0])
#             tmp2.remove([tmp[i][0]])
#             print(tmp2)
# print(tmp2)
re = list()
a1 = np.arange(3,11).tolist()
a3 = np.arange(63,91).tolist()
a2 = np.arange(19,24).tolist()
re.extend(a1)
re.extend(a2)
re.extend(a3)
print(re)
a = np.arange(1,91)
out = list()

b1 = np.arange(11,19).tolist()
b2 = np.arange(24,27).tolist()
b3 = np.arange(29,33).tolist()
b4 = np.arange(35,39).tolist()
b5 = np.arange(41,45).tolist()
b6 = np.arange(47,51).tolist()
b7 = np.arange(53,63).tolist()
out.extend(b1)
out.extend(b2)
out.extend(b3)
out.extend(b4)
out.extend(b5)
out.extend(b6)
out.extend(b7)
print(out)