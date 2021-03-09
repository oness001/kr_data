from 高位突破min_bd_01 import duojincheng_backtesting as bt01
from 高位突破min_bd_02 import duojincheng_backtesting as bt02
from 高位突破min_bd_03 import duojincheng_backtesting as bt03
from 高位突破min_bd_04 import duojincheng_backtesting as bt04
from 高位突破min_bd_05 import duojincheng_backtesting as bt05
from 高位突破min_bd_06_atr import duojincheng_backtesting as bt061
from 高位突破min_bd_06_前高 import duojincheng_backtesting as bt062
from 高位突破min_bd_06_固定 import duojincheng_backtesting as bt063


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
    from echart_可加参数 import draw_charts,draw_line_charts
    import time ,datetime,talib
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H_%M')
    # print(timeer0)
    # exit()
    df_time_list = [['2019-1-01 09:15:00', '2019-12-26 16:25:00']]
    s_time, e_time = df_time_list[0]
    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
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
    if 1:
        df_zong = pd.DataFrame()
        df_res =pd.DataFrame()
        df_zong['candle_begin_time'] = df['candle_begin_time']
        df_zong['close'] = df['close']
        df_zong['整体资金line']=0
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        hcdict = {
        'bt061': [110.0, 290.0, 12.0, 2.0, 2.0]	,
        'bt062': [60.0, 120.0, 290.0, 16.0, 1.0],
        'bt063': [40.0, 130.0, 290.0, 16.0, 1.0]
        }


        for i,cs0 in hcdict.items():
            df000 = df.copy()
            resdf0,res0 = eval(i)(df000, zong_can=[cs0], strat_time=strat_time, end_time=end_time, cpu_nums = 1)
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

        cslist = df_zong.keys()
        cslist = [ i for i in cslist if i not in ['candle_begin_time','close']]
        df_zong = transfer_res_data(df_zong,rule_type='1H')
        print(df_zong.tail(10))
        draw_line_charts(df_zong,df_res,canshu=cslist,path= r"html_gather\%s突破06三个变化_line_base.html"%s_time[:7])