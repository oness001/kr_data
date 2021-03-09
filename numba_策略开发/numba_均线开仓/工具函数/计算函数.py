from numba import jit #, int64, float32
import numpy as np
import talib
from datetime import timedelta
import pandas as pd
import traceback
from multiprocessing import Pool, cpu_count  # , Manager
from numba_策略开发.画图工具.echart_可加参数 import draw_charts
import time, datetime
import os

@jit(nopython=True)
def chanrao_index(df_MA_arrays, cha_zong=np.ndarray([])):
    ma0 = np.sum(df_MA_arrays[:], axis=1)/df_MA_arrays.shape[1]
    for i in range(df_MA_arrays.shape[1]-1):
        cha_zong += np.abs((df_MA_arrays[:,i] - ma0))
    # print(cha_zong[-10:],cha_zong.shape)
    return cha_zong

