
'''


'''
from numba import jit #, int64, float32
import numpy as np
import talib
from datetime import timedelta
import pandas as pd
import traceback
from multiprocessing import Pool, cpu_count  # , Manager
from numba_策略开发.画图工具.echart_可加参数 import draw_charts,only_line,only_index
import time, datetime
import os
from numba_策略开发.回测工具.统计函数 import cal_tongji ,cal_per_pos
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data ,get_local_hsi_csv
from numba_策略开发.功能工具.计算函数 import chanrao_index
np.seterr(divide='ignore', invalid='ignore')

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
pd.set_option('display.float_format', lambda x: '%.3f' % x)

@jit(nopython=True)
def cal_signal_(df0, df1, df2, strat_time, end_time, cs0):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''
    # 'atr_day', '移动止赢', 'n_atr', '小均线', '中均线', '大均线', '缠绕指标']
    #     6          7         8        9       10     11      12
    # 配置参数:止盈，ma_len（atr），atr_倍数，交易次数
    stop_win_n, ma_xiao_n,ma_zhong_n,ma_da_n,ma_day_da_n,cr_num, atr_sw_n = cs0
    # 配置临时变量
    inner_trade_nums = 1000
    open_pos_size = 1
    max_log_high = np.nan
    min_log_low = np.nan
    log_stop_win = 0
    for i in range(10, df0.shape[0]):  # 类似于vnpy:on_bar
        # 交易时间判断：日：1-28，and （时，分：2：30-16：20）
        trading_time_con = ((df0[i, 6] >= strat_time[0]) and (df0[i, 6] <= end_time[0])) and \
                           ((df0[i, 7] == 9) and (df0[i, 8] >= 15)) or \
                           ((df0[i, 7] == 16) and (df0[i, 8] <= 29)) or \
                           ((df0[i, 7] > 9) and (df0[i, 7] < 16))
        # 我们交易时间
        we_trade_con = ((df0[i, 6] >= strat_time[0]) and (df0[i, 6] <= end_time[0])) and \
                       ((df0[i, 7] == strat_time[1]) and (df0[i, 8] > strat_time[2])) or \
                       ((df0[i, 7] == end_time[1]) and (df0[i, 8] < end_time[2])) or \
                       ((df0[i, 7] > strat_time[1]) and (df0[i, 7] < end_time[1]))
        # 交易所日盘，开放
        if trading_time_con:
            # 快捷变量
            open_bar = df0[i][1]
            high_bar = df0[i][2]
            low_bar = df0[i][3]
            last_close = df0[i - 1][4]
            close_bar = df0[i][4]

            atr_n = df2[i][8]*atr_sw_n
            xiao_ma = df2[i][9]
            zhong_ma = df2[i][10]
            da_ma = df2[i][11]
            chanraoindex = df2[i][12]
            day_ma = df2[i][13]
            last_day_ma = df2[i-1][13]
            last_da = df2[i-1][11]
            last_zhong = df2[i-1][10]
            last_xiao = df2[i-1][9]

            # === 仓位统计。
            df1, now_open_prcie, now_pos = cal_per_pos(i, df1, open_pos_size, open_bar, close_bar, last_close, sxf=1, slip=1)

            # === 当日变量加载
            if ((df0[i, 7] == 9) and (df0[i, 8] == 15)):
                inner_trade_nums = 100
                max_log_high = high_bar
                min_log_low = low_bar
                log_stop_win = 0
            else:
                max_log_high = max(max(open_bar,close_bar), max_log_high)
                min_log_low = min(min(open_bar,close_bar), min_log_low)

            if max_log_high > (now_open_prcie + stop_win_n):
                stop_loss_price = xiao_ma
            elif max_log_high > (now_open_prcie + stop_win_n):
                stop_loss_price = 0.5*(xiao_ma+zhong_ma)
            else:
                stop_loss_price = max(da_ma,(now_open_prcie - atr_n))
            log_stop_win = max(log_stop_win,stop_loss_price)


            if we_trade_con:
                if now_pos == 0:
                    log_stop_win = (da_ma - atr_n)

                    long_condition = (xiao_ma  > zhong_ma)and(last_xiao <= last_zhong)and( close_bar > da_ma) #小中金叉
                    long_condition&= (last_zhong<zhong_ma) #大周期均是向上
                    long_condition &= (chanraoindex < cr_num) #缠绕条件
                    long_condition &= (last_day_ma < day_ma) #大周期条件

                    if long_condition and inner_trade_nums > 0:
                        inner_trade_nums -= 1
                        max_log_high = high_bar
                        df1[i][1] = 1

                elif now_pos > 0:
                    stop_win_con = (close_bar < log_stop_win)
                    stop_loss_con = False
                    close_pos = False
                    if close_pos:
                        df1[i][1] = 0
                    elif stop_loss_con:
                        df1[i][1] = 0
                    elif stop_win_con:
                        df1[i][1] = 0
            else:  # 非交易时间段
                if now_pos != 0:
                    df1[i][1] = 0

            # 记录指标 绘图等等
            # ['开仓线','atr_day','小均线']
            df2[i, 2] = log_stop_win
            df2[i, 3] = min_log_low
            df2[i, 4] = max_log_high
            # df2[i, 7] = log_stop_win
    res0 = cal_tongji(df_input=df1)
    res0 = np.concatenate((res0, cs0))
    return df0, df1, df2, res0

def cal_signal(df, strat_time, end_time, canshu):
    a = time.process_time()
    # ===转化成np.array
    df0cols = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    df1cols = ['candle_begin_time', 'signal', 'pos', 'opne_price', 'per_lr', 'sl']
    df2cols = ['candle_begin_time', '止损价', '止盈价', '日内最低价', '日内最高价', '开仓线',
               'atr_day','移动止赢', 'n_atr','小均线','中均线','大均线','缠绕指标','大周期' ]
                #     6       7         8        9       10     11      12
    # 配置参数:止盈，ma_len（atr），atr_倍数，交易次数
    # stop_win_n, ma_len,big_ma, base_line,extend_n = cs0
    df['atr_day'] = 0

    df['小均线'] = talib.SMA(df['close'], int(canshu[1]))
    df['中均线'] = talib.SMA(df['close'], int(canshu[2]))
    df['大均线'] = talib.SMA(df['close'], int(canshu[3]))
    df['缠绕指标'] = chanrao_index(df[['小均线','中均线','大均线']].values,cha_zong=np.zeros(df['大均线'].shape[0],))
    df['n_atr'] = talib.ATR(df['close'], df['high'], df['low'], int(canshu[1]))
    df['大周期'] = talib.SMA(df['close'], int(canshu[4]))

    df['移动止赢'] = np.nan

    # print(df.tail())
    # exit()
    df0 = df[df0cols].values
    df1 = df[df1cols].values
    df2 = df[df2cols].values

    df0, df1, df2, res = cal_signal_(df0, df1, df2, strat_time, end_time, canshu)
    print('runingtime:', time.process_time() - a, 's')
    return df0, df1, df2, (df0cols, df1cols, df2cols), res

def duojincheng_backtesting(df_input, zong_can, strat_time, end_time, cpu_nums=3,jiexi=False):
    df = df_input.copy()
    if cpu_nums > cpu_count() - 1: cpu_nums = cpu_count() - 1
    huice_df = []

    def tianjia(res):
        huice_df.append(res[-1])
        # print(res[-1])
    def error_get(res):
        # pd.DataFrame(huice_df)
        print(res)

    pass

    #  ===原始数据处理
    df['candle_begin_time'] = (df['candle_begin_time'] - np.datetime64(0, 's')) / timedelta(seconds=1)
    df['days'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().day))
    df['huors'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().hour))
    df['minutes'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().minute))
    #  ===信号仓位统计数据列
    df['signal'] = np.nan
    df['pos'] = 0
    df['opne_price'] = np.nan
    df['per_lr'] = np.nan
    df['sl'] = np.nan
    #  ===指标数据
    df['止损价'] = np.nan
    df['止盈价'] = np.nan
    df['日内最低价'] = np.nan
    df['日内最高价'] = np.nan
    df['开仓线'] = np.nan
    df['小均线'] = np.nan
    df['中均线'] = np.nan
    df['大均线'] = np.nan

    df = df[(df['huors'] >= 2) & (df['huors'] <= 16) & (df['days'] <= end_time[0])]
    pass

    # 多进程回测
    if cpu_nums > 1:
        p = Pool(processes=cpu_nums)
        for j in range(0, len(zong_can), cpu_nums):
            for i in range(cpu_nums):
                if j + i <= len(zong_can) - 1:
                    canshu0 = zong_can[j + i]
                    p.apply_async(cal_signal, args=(df, strat_time, end_time, np.array(canshu0),), callback=tianjia,error_callback= error_get)
                else:
                    break
        p.close()
        p.join()
        print('进程池joined')
        # 整理多进程回测的数据
        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比'] + [f'参数{i}' for i in range(1, len(zong_can[0]) + 1)]
        resdf = pd.DataFrame(huice_df, columns=cols)
        resdf = resdf[cols]
        resdf.sort_values(by='最后收益', inplace=True)
        print(resdf.iloc[:5])
        print(resdf.iloc[-5:])

        print(f'=参数回测结束,谢谢使用.')
        return resdf

    # 单进程测试
    else:
        for cs0 in zong_can:
            a = time.process_time()
            df0, df1, df2, cols, res0 = cal_signal(df, strat_time, end_time, np.array(cs0))
            print('runingtime:', time.process_time() - a, 's')
            tianjia(res=res0)
        if jiexi :
            return

        df0cols, df1cols, df2cols = cols
        # 转换成df数据
        df00 = pd.DataFrame(df0, columns=df0cols)
        df11 = pd.DataFrame(df1, columns=df1cols)
        df22 = pd.DataFrame(df2, columns=df2cols)
        # 合并
        df11_ = pd.merge(df00, df11, on="candle_begin_time", suffixes=('_0', '_1'))
        dfres = pd.merge(df11_, df22, on="candle_begin_time", suffixes=('', '_2'))
        dfres["candle_begin_time"] = pd.to_datetime(dfres["candle_begin_time"], unit='s')
        dfres.sort_values(by='candle_begin_time', inplace=True)
        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比'] +\
               [f'参数{i}' for i in range(1, len(zong_can[0]) + 1)]
        res0 = pd.DataFrame([res0], columns=cols)
        print(dfres.iloc[:5])
        print(dfres.iloc[-5:])

        return dfres, res0
def KAMA(df, n=10, pow1=2, pow2=30):
  ''' kama indicator '''
  ''' accepts pandas dataframe of prices '''


  df['absDiffx'] = abs(df['close'] - df['close'].shift(1) )
  df['ER.num'] = ( df['close'] - df['close'].shift(n) )

  df['ER.den'] = df['absDiffx'].rolling(n).sum()
  df['ER'] = df['ER.num'] / df['ER.den']

  df['SC'] = ( df['ER']*(2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0) ) ** 2.0


  return df['SC']
if __name__ == '__main__':

    filename = os.path.abspath(__file__).split('\\')[-1].split('.')[0]
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H_%M')
    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
    dir_path =r'F:\task\恒生股指期货\numba_策略开发\numba_均线开仓'

    # 1.先测试单进程 cpu == 1
    a = 1
    b = 10
    c = 10
    if True == a:
        info = '金叉'
        to_html_path = r'html_gather\%s_%s' % (filename, info)
        df_time_list = [['2017-10-1 09:20:00', '2017-11-28 16:20:00']]
        s_time, e_time = df_time_list[0]
        df = get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
        df = transfer_to_period_data(df,rule_type='1T')
        df000 = df.copy()
        df["aroondown"],df["aroonup"] = talib.AROON(df['high'],df["low"],14)
        print(df)
        only_index(input_data=df, zhibiaos=["close"],zhibiaos_futu=["aroonup","aroondown"], path='zb.html')

        exit()

        strat_time = np.array([1, 9, 20])
        end_time = np.array([27, 16, 20])
        time0 = time.process_time()
        df000 = df.copy()
        resdf, res0 = duojincheng_backtesting(df000,
                                              zong_can=[[130.0,30.0,59.0,105.0,3000.0,10.0,2.0]],
                                              strat_time=strat_time,
                                              end_time=end_time,
                                              cpu_nums=1,jiexi=False)

        print(res0)
        print(resdf.tail(5))
        mode = 1
        if mode == 0:
            resdf['per_lr'].fillna(0, inplace=True)
            resdf['资金曲线'] =  resdf['per_lr'].cumsum()
            resdf['资金曲线'].fillna(method='ffill', inplace=True)
            resdf['资金曲线2'] =resdf['资金曲线'].rolling(120).mean()
            resdf['资金曲线'].fillna(method='bfill', inplace=True)

            resdf['up'],resdf['ma'],resdf['dn'] = talib.BBANDS(resdf['close'],3000,3,3)
            # , '资金曲线', '资金曲线2'
            only_line(resdf, zhibiaos=['dn','up','ma'],canshu=[ '资金曲线', '资金曲线2'],
                                rule_type='1H', path='资金曲线test.html')
        elif mode == 1:
            resdf_baitian = resdf[(resdf['huors'] >= 2) & (resdf['huors'] <= 16)]
            resdf = resdf_baitian.copy()
            resdf.reset_index(inplace=True)
            resdf.sort_values(by='candle_begin_time', inplace=True)
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       '日内最高价': resdf['日内最高价'],
                                       '小均线': resdf['小均线'],
                                       '中均线': resdf['中均线'],
                                       '大均线': resdf['大均线']},
                        canshu2={'缠绕指标': resdf['缠绕指标']},
                        vol_bar=True, markline_show2=True,
                        path=to_html_path)
        elif mode == 2:
            resdf.loc[resdf['signal'] == 1, 'sig0'] = 1
            resdf.loc[resdf['signal'] == 0, 'sig1'] = 0
            resdf['sig0'] = resdf['sig0'].shift(-20)
            resdf['sig1'] = resdf['sig1'].shift(20)
            resdf['sig00'] = resdf[['sig0', 'sig1']].max(axis=1)
            resdf['sig00'].fillna(method='ffill', inplace=True)
            resdf = resdf[resdf['sig00'] == 1].copy()
            resdf.reset_index(drop=False, inplace=True)

            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                           '止损价': resdf['止损价'],
                           '止盈价': resdf['止盈价'],
                           '日内最低价': resdf['日内最低价'],
                           '日内最高价': resdf['日内最高价'],
                           '小均线': resdf['小均线'],
                           '中均线': resdf['中均线'],
                           '大均线': resdf['大均线'],
                           # '移动止赢' :resdf['移动止赢'],
                           }, vol_bar=True, markline_show2=True,
                        path=to_html_path)
        print(f'总时间：{time.process_time() - time0}  s')
        exit()
    # 多进程策略 :
    # [190, 20, 2, 2]
    if True == b:
        info = '_大周期_1T'
        [150, 18, 60, 100, 3000, 100, 1.8]
        canshu_list = []

        for cs1 in range(100, 150, 10):
            for cs2 in range(8, 13, 2):
                for cs3 in range(40, 51, 3):
                    for cs4 in range(80, 90, 3):
                        for cs5 in range(3000, 3009, 1000):
                            for cs6 in range(25, 35, 3):
                                for cs7 in range(10, 30, 7):
                                    canshu_list.append([cs1, cs2, cs3,cs4, cs5,cs6,cs7/10])
        canshu_list = canshu_list[:]
        df_time_list = [['2016-1-10 09:15:00', '2019-12-20 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
        df = transfer_to_period_data(df,rule_type='1T')
        df000 = df.copy()
        strat_time = np.array([1, 9, 20])
        end_time = np.array([27, 16, 20])
        print('参数列表个数：', len(canshu_list))
        time0 = time.process_time()
        resdf = duojincheng_backtesting(df000, zong_can=canshu_list[:], strat_time=strat_time, end_time=end_time, cpu_nums=3)
        print(resdf.tail())
        resdf.to_csv(r'csv_gather\%s_%s_%s.csv' % (filename,timeer0,info))
        print(f'总时间：{time.process_time() - time0}  s')

    # 多进程策略 :
    if True == c:
        from numba_策略开发.回测工具.多阶段回测 import duojieduan_huice

        [155, 15, 3100, 15, 0.7]
        canshu_list = []
        for cs1 in range(130, 190, 10):
            for cs2 in range(14, 22, 2):
                for cs3 in range(1500,3501,500):
                    for cs4 in range(10, 35, 5):
                        for cs5 in range(5, 8, 1):
                            # for cs6 in range(1, 3, 1):
                            canshu_list.append([cs1, cs2, cs3, cs4, cs5/10])

        canshu_list = canshu_list[:]
        df_time_list = [['2016-1-10 09:15:00', '2019-12-28 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        name = f'{filename}_{s_time[:4]}_{e_time[:4]}_总体回测'

        if True == 1:
            print('参数列表个数：', len(canshu_list))
            time0 = time.process_time()
            func_canshu = {'zong_can': canshu_list[:], 'cpu_nums': 2, "strat_time": strat_time, "end_time": end_time}
            df_zong = duojieduan_huice(df, duojincheng_backtesting, func_canshu, s_time, e_time, jiange='48MS')
            path_csv = dir_path + r'\csv_gather\%s.csv'%name
            df_zong.to_csv(path_csv)
            # 接下来生成html
            df_zong_html = pd.DataFrame()
            for k, v in df_zong.groupby('s_time'):
                print(k)
                dfv = pd.DataFrame(v)
                dfv.sort_values('最后收益', ascending=False, inplace=True)
                df0 = dfv.iloc[:50]
                df_zong_html = df_zong_html.append(df0, ignore_index=True)
            canshu_cols = [i for i in df_zong_html.keys() if '参数' in i]
            df_zong_html['参数_合并'] = df_zong_html[canshu_cols].values.tolist()
            df_zong_html['参数_合并'] = df_zong_html['参数_合并'].astype(str)
            df0 = df_zong_html['参数_合并'].value_counts()  # .to_frame()
            # df0.reset_index(inplace=True)
            df_zong_html['参数出现次数'] = df_zong_html['参数_合并'].apply(lambda x: df0[x])

            df_zong_html.set_index(keys=['s_time', '最后收益'], inplace=True)
            html_path = dir_path+r'\策略介绍html\csv_html\%s.html' % name
            df_zong_html.to_html(html_path)

        # 读取本地数据,单独生成html，
        if True == 0:
            df_zong = pd.read_csv(dir_path + r'\csv_gather\%s.csv'%name)
            df_zong_html = pd.DataFrame()
            for k, v in df_zong.groupby('s_time'):
                print(k)
                dfv = pd.DataFrame(v)
                dfv.sort_values('最后收益', ascending=False, inplace=True)
                df0 = dfv.iloc[:50]
                df_zong_html = df_zong_html.append(df0, ignore_index=True)
            canshu_cols = [i for i in df_zong_html.keys() if '参数' in i]
            df_zong_html['参数_合并'] = df_zong_html[canshu_cols].values.tolist()
            df_zong_html['参数_合并'] = df_zong_html['参数_合并'].astype(str)

            df0 = df_zong_html['参数_合并'].value_counts()  # .to_frame()
            # df0.reset_index(inplace=True)
            df_zong_html['参数出现次数'] = df_zong_html['参数_合并'].apply(lambda x: df0[x])

            df_zong_html.set_index(keys=['s_time', '最后收益'], inplace=True)
            name = 'max_突破06_atr_2015_2019.html'
            html_path = dir_path + r'\策略介绍html\csv_html\%s.html' % name
            df_zong_html.to_html(html_path)
