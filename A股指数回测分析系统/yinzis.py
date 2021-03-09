
import talib
import numpy as np
import pandas as pd


np.seterr(divide='ignore', invalid='ignore')
pd_display_rows = 10
pd_display_cols = 100
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行




def yinzi01(df_array,m):
    '''

    :param df_array: 仅仅需要close——array
    :param m:
    :return:
    '''

    df0 = df_array.copy()
    yz = ((talib.RSI(np.log(df0["close"]), m)-50) / 100 ) * (talib.STDDEV(np.log(df0["close"]), m))

    # yz = talib.SMA(yz,3)
    return yz

def yinzi02(df_array,m):
    '''
    :param df_array: 仅仅需要close——array
    :param m:
    :return:
    '''

    df0 = df_array.copy()
    yz = (talib.RSI(np.log(df0["close"]), m) / 100 + 1) * (talib.STDDEV(np.log(df0["close"]), m))
    yz = talib.ROCP(np.log(df0["close"]), m)*yz

    return yz

def yinzi03(df_array, m):
    '''

    :param df_array: 仅仅需要close——array
    :param m:
    :return:
    '''
    df0 = df_array.copy()
    yz = talib.ROCP(np.log(df0["close"]), m)

    return yz

def yinzi031(df_array, m):
    '''
    :param df_array: 仅仅需要close——array
    :param m:
    :return:
    '''

    df0 = df_array.copy()
    yz = talib.ROCP(np.log(df0["close"]), m)
    return yz

def yinzi04(df_array, m):
    '''

    :param df_array: 要high,low
    :param m:
    :return:
    '''

    df0 = df_array.copy()

    dn,yz = talib.AROON(np.log(df0["high"]),np.log(df0["low"]) ,m)
    return yz

def yinzi05(df_array, m):
    '''

    :param df_array: 要high,low
    :param m:
    :return:
    '''

    df0 = df_array.copy()
    yz = talib.CCI(np.log(df0["high"]),np.log(df0["low"]),np.log(df0["close"]),m)/100
         # +\
         # +\
         # talib.ROCP(np.log(df0["close"]), m)+talib.RSI(np.log(df0["close"]),m)-50
    return yz

def yinzi06(df_array, m):
    '''

    :param df_array: 要high,low
    :param m:
    :return:
    '''

    df0 = df_array.copy()
    yz = talib.MFI(np.log(df0["high"]),np.log(df0["low"]),np.log(df0["close"]),df0["volume"],m)
         # +\
         # +\
         # talib.ROCP(np.log(df0["close"]), m)+talib.RSI(np.log(df0["close"]),m)-50
    return yz

def yinzi07(df_array, m):
    '''

    :param df_array: 要high,low
    :param m:
    :return:
    '''

    df0 = df_array.copy()
    yz = talib.MFI(np.log(df0["high"]),np.log(df0["low"]),np.log(df0["close"]),df0["volume"],m)-50

    yz += talib.ROCP(np.log(df0["close"]), m)*100
    return yz


if __name__ == '__main__':

    df0 = pd.read_csv(r"F:\vnpy_my_gitee\company\A股票_company\stock_77_201701-201712.csv")
    df0  = df0[df0["code"]=="sz.000922"].copy()
    df1 =yinzi07(df0,20)
    print(df1)
