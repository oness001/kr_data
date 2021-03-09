from numba import jit,int64,float32
import numpy as np
import time
import talib
from datetime import timedelta
import pandas as pd
import traceback
from multiprocessing import Pool, cpu_count  # , Manager
import time, datetime, os
from numba_策略开发.画图工具.echart_可加参数 import draw_charts,only_line,draw_line_charts
from numba_策略开发.回测工具.统计函数 import cal_tongji, cal_per_pos
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data, get_local_hsi_csv,transfer_period_anydata



'''
将止损改为移动止损，方式为：固定移动跟踪止损。
'''
np.seterr(divide='ignore',invalid='ignore')

pd_display_rows = 10
pd_display_cols = 100
pd_display_width = 1000
pd.set_option('display.max_rows', pd_display_rows)
pd.set_option('display.max_columns', pd_display_cols)
pd.set_option('display.width', pd_display_width)
pd.set_option('display.max_colwidth', pd_display_width)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 20000)
pd.set_option('display.float_format', lambda x: '%.8f' % x)


@jit(nopython=True)
def zhyz_func(rsi: np.array,cci: np.array,atr: np.array):
    zhyz = np.where((rsi+cci/10>0),rsi+cci/10+atr,rsi+cci/10-atr)
    zhyz1 = zhyz - np.roll(zhyz, 1)
    zhyz2 = zhyz1 - np.roll(zhyz1, 1)
    # zhyz2[:2] = 0
    return zhyz2
def rsi_cci_atr01(df0,n):
    df =df0.copy()
    df = df.values
    high, low, close = df[:,0],df[:,1],df[:,2]
    # df['rsi'] = talib.RSI(df['close'], n) - 50
    # df['cci'] = talib.CCI(df['high'], df['low'], df['close'], n)
    # df['atr'] = talib.ATR(df['high'], df['low'], df['close'], n)
    # df.loc[(df['rsi'] + df['cci'] / 10) > 0, 'zuheyinzi'] = (df['rsi'] + df['cci'] / 10) + df['atr']
    # df.loc[(df['rsi'] + df['cci'] / 10) <= 0, 'zuheyinzi'] = (df['rsi'] + df['cci'] / 10) - df['atr']
    rsi= talib.RSI(close, n) - 50
    cci = talib.CCI(high,low,close, n)
    atr = talib.ATR(high,low,close, n)
    zhyz00 = zhyz_func(rsi,cci,atr)
    zhyz = talib.SMA(zhyz00, n)
    return zhyz


def rsi_cci_atr01_1(df0,n):
    df =df0.copy()
    high, low, close = df[:,0],df[:,1],df[:,2]
    df = pd.DataFrame()
    df['rsi'] = talib.RSI(close, n) - 50
    df['cci'] = talib.CCI(high, low, close, n)
    df['atr'] = talib.ATR(high, low, close, n)
    df.loc[(df['rsi'] + df['cci'] / 10) > 0, 'zuheyinzi'] = (df['rsi'] + df['cci'] / 10) + df['atr']
    df.loc[(df['rsi'] + df['cci'] / 10) <= 0, 'zuheyinzi'] = (df['rsi'] + df['cci'] / 10) - df['atr']
    df['zuheyinzi'].diff(1).diff(1).rolling(n).mean()
    return df['zuheyinzi']


def rsi_cci_atr02(df0,n):
    df =df0.copy()

    df['rsi'] = talib.RSI(df['close'], n) - 50
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], n)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], n)
    df['zuheyinzi'] = (df['rsi'] +  df['cci'] / 10)/df['atr']
    return df['zuheyinzi']


def rsi_cci_atr03(df0,n):
    df =df0.copy()

    df['rsi'] = talib.RSI(df['close'], n) - 50
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], n)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], n)
    df.loc[ (df['rsi'] +  df['cci'] / 10)>0 , 'zuheyinzi'] = (df['rsi'] +  df['cci'] / 10)+df['atr'].rolling(n).mean()
    df.loc[ (df['rsi'] +  df['cci'] / 10)<=0 , 'zuheyinzi'] = (df['rsi'] +  df['cci'] / 10) - df['atr'].rolling(n).mean()


    return df['zuheyinzi']


def std_zdf_tp(df0,n,m,l):

    df =df0.copy()
    df['tp'] = 0
    df['bd'] = df['close'] - df['open']
    df['std'] = talib.STDDEV(df['bd'], n) * m
    df.loc[df['bd'] > df['std'], 'tp'] = 1
    df['tp'] = talib.SUM(df['tp'],l)

    return df['tp']

def std_zdf_tp2(df0,n,m,l):
    df =df0.copy()
    df['tp'] = 0
    df['bd'] = df['close'] - df['open']
    df['std'] = talib.STDDEV(df['bd'], n) * m
    df['std_mean'] = talib.MA(df['std'], n)
    df.loc[df['bd'] > (df['std_mean'] + df['std']), 'tp'] = 1
    df['tp'] = df['tp'].rolling(l).sum()
    return df["tp"]