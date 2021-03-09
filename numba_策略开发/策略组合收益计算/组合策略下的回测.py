

from numba_策略开发.numba_突破max01_突破_空策略.max_通道_atr_固定max_ma止盈_大周期择时 import duojincheng_backtesting as bt01
from numba_策略开发.numba_突破max01_突破_空策略.直接突破01_atr_固定max_ma止盈_大周期择时 import duojincheng_backtesting as bt02
from numba_策略开发.numba_突破max01_做多策略.tnd_通道_做多_大周期择时 import duojincheng_backtesting as bt03
from numba_策略开发.numba_突破max01_做多策略.tnd_通道_做多_大周期择时_max_移动止损 import duojincheng_backtesting as bt04
from numba_策略开发.nunba_bollin.bolin_01_long__dazhouqi import duojincheng_backtesting as bt05
from numba_策略开发.nunba_bollin.bolin_02_long__dazhouqi_longma import duojincheng_backtesting as bt06
from numba_策略开发.nunba_ma.all_day_tri_ma_min import duojincheng_backtesting as bt07
from numba_策略开发.nunba_ma.all_day_tri_ma_rsi import duojincheng_backtesting as bt08

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

def transfer_res_data(df, rule_type):
    """
    将数据转换为其他周期的数据
    """
    # ==转换为其他分钟数据
    trans_clos = [i for i in df.keys() if i not in ['candle_begin_time']]
    to_trans_cols =  {k:'last' for k in trans_clos}
    period_df = df.resample(rule=rule_type, on='candle_begin_time', label='left', closed='left').agg(to_trans_cols)

    period_df.sort_values(by = 'candle_begin_time', inplace=True)
    period_df.fillna(method='ffill',inplace=True)
    period_df.reset_index(inplace=True)

    return period_df
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from numba_策略开发.画图工具.echart_可加参数 import draw_charts,draw_line_charts

    import time ,datetime,talib
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H_%M')
    # print(timeer0)
    # exit()
    df_time_list = [['2016-1-01 09:15:00', '2020-12-26 16:25:00']]
    s_time, e_time = df_time_list[0]
    datapath = r'F:\vnpy_my_gitee\new_company\hsi_data_1min\HSI2011-2020-11.csv'
    print('数据地址：', datapath)
    df = pd.read_csv(filepath_or_buffer=datapath)
    df['candle_begin_time'] = df['datetime']
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    df = df[(df['candle_begin_time'] >= pd.to_datetime(s_time))&(df['candle_begin_time'] <= pd.to_datetime(e_time))]
    # df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
    df.reset_index(inplace=True, drop=True)
    print('数据大小',df.shape)
    df_day = transfer_to_period_data(df, rule_type='1D')
    df_day['atr_day'] = talib.SMA(talib.ATR(df_day['close'], df_day['high'], df_day['low'], 10), 10) * 0.8
    df_day['candle_begin_time'] = df_day['candle_begin_time'].apply(lambda x: x + datetime.timedelta(hours=9, minutes=15))
    df = pd.merge(df, df_day[['candle_begin_time', 'atr_day']], how='outer', on='candle_begin_time')
    df = df[df['volume'] > 0]
    df.fillna(method='ffill', inplace=True)

    a = 1
    if a==1:
        df_zong = pd.DataFrame()
        df_res =pd.DataFrame()
        df_zong['candle_begin_time'] = df['candle_begin_time']
        df_zong['close'] = df['close']
        df_zong['整体资金line']=0
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        hcdict = {
        # 'bt01': [205.0, 24.0, 2000.0, 3.0, 1.0]	     ,
        # 'bt02': [205.0, 18.0, 3.0, 4.0, 3000.0, 2.0] ,
        # 'bt03': [260.0,19.0,2000.0,5.0,1.0],
        # 'bt05': [195.0, 13.0, 12.0,1500, 3.50, 30.0],
        # 'bt06': [215.000 ,11.000 ,185.000, 2000.000 , 4.000,  7.000],
        'bt07':[44.0,140.0,30.0,50.0,140.0],
        'bt08': [40, 140, 30, 50.0,140,50]
        }
        for i,cs0 in hcdict.items():
            if i in ['bt05','bt06']:
                strat_time = np.array([1, 9, 15])
                end_time = np.array([27, 16, 20])
            df000 = df.copy()
            resdf0,res0 = eval(i)(df000, zong_can=[cs0], strat_time=strat_time, end_time=end_time, cpu_nums = 1)
            print(res0)
            res0['策略名称'] = i+'_line'
            df_res = df_res.append(res0,ignore_index=True)
            resdf0['per_lr'].fillna(0,inplace=True)
            resdf0[i+'_line'] = resdf0['per_lr'].cumsum()
            df_zong = pd.merge(df_zong,resdf0[['candle_begin_time',i+'_line']],how='outer',on='candle_begin_time')

            df_zong['整体资金line'] +=df_zong[i+'_line']

        df_zong['整体资金line'] = df_zong['整体资金line']/len(hcdict) if len(hcdict) != 0 else len(hcdict)
        mean_res = df_res.mean()
        mean_res['策略名称'] = "平均值"
        sum_res = df_res.sum()
        sum_res['策略名称'] = "总计和"
        df_res = df_res.append(mean_res, ignore_index=True)
        df_res = df_res.append(sum_res, ignore_index=True)
        df_res['开始时间'] = s_time
        df_res['结束时间'] = e_time
        df_res = df_res[["开始时间",'策略名称','最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益',
                         '开仓次数', '胜率', '盈亏比', "结束时间",'参数1', '参数2', '参数3', '参数4', '参数5']]
        df_zong['base_close_line'] = (df_zong['close']-df_zong['close'].shift()).cumsum()
        cslist = df_zong.keys()
        cslist = [ i for i in cslist if i not in ['candle_begin_time','close']]
        df_zong = transfer_res_data(df_zong,rule_type='1H')
        print(df_zong.tail(5))
        draw_line_charts(df_zong,df_res,canshu=cslist,path= r"%s整体组合策略资金曲线_line.html"%s_time[:7])