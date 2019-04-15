import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt


class LossPlotter(object):

    def __init__(self, mylog_path="./log", mylog_name="training.log", myloss_names=["loss"], mymetric_names=["mean_squared_error","mean_absolute_error"]):
        super(LossPlotter, self).__init__()
        self.log_path = mylog_path
        self.log_name = mylog_name
        self.loss_names = list(myloss_names)
        self.metric_names = list(mymetric_names)
        os.makedirs(join(self.log_path, "plot"), exist_ok=True)

    def plotter(self):

        dataframe = pd.read_csv(join(self.log_path,self.log_name), skipinitialspace=True)

        for i in range(len(self.loss_names)):
            plt.figure(i)
            plt.plot(dataframe[self.loss_names[i]],label="train_"+self.loss_names[i])
            plt.plot(dataframe["val_"+self.loss_names[i]],label="val_"+self.loss_names[i])
            plt.legend()
            plt.savefig(join(self.log_path,"plot",self.loss_names[i]+".png"))
            plt.close()

        for i in range(len(self.metric_names)):
            plt.figure(i+len(self.loss_names))
            plt.plot(dataframe[self.metric_names[i]],label="train_"+self.metric_names[i])
            plt.plot(dataframe["val_"+self.metric_names[i]],label="val_"+self.metric_names[i])
            plt.legend()
            plt.savefig(join(self.log_path,"plot",self.metric_names[i]+".png"))
            plt.close()
