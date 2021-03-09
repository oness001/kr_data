import pandas as pd
import talib, time, datetime



# 功能函数
def transfer_to_period_data(df, rule_type):
    """
    将数据转换为其他周期的数据

    """
    # ==转换为其他分钟数据
    period_df = df.resample(rule=rule_type, on='candle_begin_time', label='left', closed='left').agg(
        {'open': 'first',
         'high': 'max',
         'low': 'min',
         'close': 'last',
         'volume': 'sum',
         })
    period_df.dropna(subset=['open'], inplace=True)  # 去除一天都没有交易的周期
    period_df = period_df[period_df['volume'] > 0]  # 去除成交量为0的交易周期
    period_df.reset_index(inplace=True)
    df = period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]

    return df

def get_loc_hsicsv(s_time, e_time,datapath):
    print('数据地址：', datapath)
    df = pd.read_csv(filepath_or_buffer=datapath)
    df['candle_begin_time'] = df['datetime']
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    df = df[(df['candle_begin_time'] >= pd.to_datetime(s_time)) & (df['candle_begin_time'] <= pd.to_datetime(e_time))]
    # df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
    df.reset_index(inplace=True, drop=True)
    print('数据大小', df.shape)
    df_day = transfer_to_period_data(df, rule_type='1D')
    df_day['atr_day'] = talib.SMA(talib.ATR(df_day['close'], df_day['high'], df_day['low'], 2), 2) * 0.8
    df_day['candle_begin_time'] = df_day['candle_begin_time'].apply(lambda x: x + datetime.timedelta(hours=9, minutes=15))
    df = pd.merge(df, df_day[['candle_begin_time', 'atr_day']], how='outer', on='candle_begin_time')
    df = df[df['volume'] > 0]
    df.fillna(method='ffill', inplace=True)
    return df