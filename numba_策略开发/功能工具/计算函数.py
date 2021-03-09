from numba import jit #, int64, float32
import numpy as np
import talib
from datetime import timedelta
import pandas as pd
import traceback
from multiprocessing import Pool, cpu_count  # , Manager
from numba_策略开发.画图工具.echart_可加参数 import draw_charts,only_line
import time, datetime
import os
from numba_策略开发.回测工具.统计函数 import cal_tongji ,cal_per_pos
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data ,get_local_hsi_csv

@jit(nopython=True)
def chanrao_index(df_MA_arrays, cha_zong=np.ndarray([])):

    ma0 = np.sum(df_MA_arrays, axis=1)/df_MA_arrays.shape[1]
    for i in range(df_MA_arrays.shape[1]):
        # print(i)
        cha_zong += np.abs((df_MA_arrays[:,i] - ma0))

    return cha_zong



if __name__ == '__main__':

    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
    df_time_list = [['2015-1-10 09:15:00', '2019-12-20 16:25:00']]
    s_time, e_time = df_time_list[0]
    df = get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
    df = transfer_to_period_data(df, rule_type='1T')
    df['ma1'] = talib.SMA(df['close'],20)
    df['ma2'] = talib.SMA(df['close'], 66)
    df['ma3'] = talib.SMA(df['close'], 120)
    matrix_ = df[['ma1', 'ma2', 'ma3']].values
    df0 = np.zeros(df.shape[0], )
    for i in range(10):
        ts = time.process_time()
        cha_zongv = chanrao_index(df_MA_arrays=matrix_, cha_zong=df0)
        print('runtime2:', (time.process_time() - ts)*1000)
        time.sleep(0.5)
    df['cr0'] = cha_zongv
    print(df.tail(30))

    exit()