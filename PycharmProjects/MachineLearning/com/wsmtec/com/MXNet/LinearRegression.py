# -*- coding:utf-8 -*-
import numpy as np
import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)

train_data = np.random.uniform(0,1,[100,2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
batch_size = 1
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11,26,16])

train_iter = mx.io.NDArrayIter(train_data,train_label,batch_size=batch_size,shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(eval_data,eval_label,batch_size,shuffle=False)

x = mx.sym.Variable('data')
y = mx.sym.Variable('lin_reg_label')
fc = mx.sym.FullyConnected(data=x,name='fc',num_hidden=1)
lro = mx.sym.LinearRegressionOutput(data=fc, label=y, name="lro")
model = mx.mod.Module(symbol=lro,context=mx.cpu(),data_names=['data'],label_names=['lin_reg_label'])

model.fit(train_iter,eval_data=eval_iter,optimizer_params={'learning_rate':0.005,'momentum':0.9},num_epoch=50,
          eval_metric='mse',batch_end_callback=mx.callback.Speedometer(batch_size,2))
metric = mx.metric.MSE()
mse = model.score(eval_iter,metric)
print("mse=",mse)