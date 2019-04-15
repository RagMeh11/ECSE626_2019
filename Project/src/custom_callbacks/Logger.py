import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn
import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, mylog_path="./log", mylog_name="training.log", myloss_names=["loss"], mymetric_names=["mean_squared_error", "mean_absolute_error"]):
        super(Logger, self).__init__()

        self.log_path = mylog_path
        self.log_name = mylog_name
        self.loss_names = list(myloss_names)
        self.metric_names = list(mymetric_names)

    def to_csv(self, metric_array, epoch):
        if epoch == 0:
            train_c = self.loss_names + self.metric_names
            val_c = ['val_'+t for t in train_c]
            df = pd.DataFrame(columns=train_c+val_c)
            df.loc[0] = metric_array
        else:
            df = pd.read_csv(join(self.log_path,self.log_name), index_col=0)
            df.loc[len(df)] = metric_array
        df.to_csv(join(self.log_path,self.log_name))

