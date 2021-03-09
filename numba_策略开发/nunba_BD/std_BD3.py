import os
import sys
sys.path.append(r'F:\vnpy_my_gitee\company')
from numba import jit,int64,float32
import numpy as np
import time
import random

import talib
from datetime import timedelta
import pandas as pd
import traceback
from multiprocessing import Pool, cpu_count  # , Manager
import time, datetime, os
from numba_策略开发.画图工具.echart_可加参数 import draw_charts,only_line,draw_line_charts
from numba_策略开发.回测工具.统计函数 import cal_tongji, cal_per_pos
from numba_策略开发.回测工具.多阶段回测 import mp_backtesting
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data_pro, get_local_hsi_csv_pro,transfer_period_anydata
from numba_策略开发.功能工具.因子库 import rsi_cci_atr01 ,std_zdf_tp



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
pd.set_option('display.float_format', lambda x: '%.3f' % x)

@jit(nopython=True)
def cal_signal_(df0,df1,df2,strat_time,end_time,cs0,all_day=0):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''
    # 配置参数
    tp_n,zuhe_limit,max_stop_win = cs0[-3:]
    # 配置临时变量
    open_pos_size = 1
    max_log_high = np.nan
    min_log_low = np.nan
    stop_loss_price = np.nan
    for i in range(2,df0.shape[0]):#类似于vnpy:on_bar
        # 交易时间判断：日：1-28，and （时，分：9：15-16：29）
        trading_time_con = ((df0[i, 7] == 9) and (df0[i, 8] >= 15)) or \
                         ((df0[i, 7] == 16) and (df0[i, 8] <= 29)) or \
                         ((df0[i, 7] > 9) and (df0[i, 7] < 16))
        if all_day>0:
            trading_time_con |=True

        # 交易所日盘，开放
        if trading_time_con :
            # 快捷变量
            open_bar = df0[i][1]
            high_bar = df0[i][2]
            low_bar = df0[i][3]
            last_close = df0[i - 1][4]
            close_bar = df0[i][4]
            # === 仓位统计。
            df1 ,now_open_prcie,now_pos = cal_per_pos(i,df1, open_pos_size, open_bar, close_bar, last_close, sxf=1, slip=1)
            # === 当日变量加载
            if ((df0[i, 7] == 9) and (df0[i, 8] == 15)):
                # inner_trade_nums = 1
                max_log_high = high_bar
                min_log_low = low_bar
            else:

                max_log_high = max(high_bar, max_log_high)
                min_log_low = min(low_bar, min_log_low)

            tp = df2[i][1]
            tp2 =df2[i-1][1]
            zhyinzi = df2[i][2]
            ma_xiao = df2[i][3]
            ma_xiao2 =df2[i-1][3]
            stop_line = df2[i][4]

            # 我们交易时间
            we_trade_con = ((df0[i, 6] >= strat_time[0]) & (df0[i, 6] <= end_time[0]))
            we_trade_con &=(((df0[i, 7] == strat_time[1]) & (df0[i, 8] > strat_time[2])) | \
                           ((df0[i, 7] == end_time[1]) & (df0[i, 8] < end_time[2])) |\
                           ((df0[i, 7] > strat_time[1]) & (df0[i, 7] < end_time[1])))
            if all_day > 0:
                we_trade_con |= True

            # 信号生成
            if we_trade_con:
                if now_pos == 0:
                    long_condition = (tp > tp2)and(tp>=tp_n)and(zhyinzi>zuhe_limit)
                    # long_condition &= (close_bar>ma_xiao)and(last_close<ma_xiao2) #向上方向
                    short_condition = False#(ma_xiao < ma_zhong) and (ma_zhong < ma_da)
                    if long_condition:
                        df1[i][1] = 1 #buy signal
                        max_log_high = high_bar
                    # 空：突破最低线。
                    elif short_condition :
                        min_log_low = low_bar
                        df1[i][1] = -1
                elif now_pos > 0:
                    stop_loss_price = stop_line if (max_log_high-now_open_prcie < 80) else max(now_open_prcie + 20,stop_line)
                    stop_loss_con = (close_bar < stop_loss_price) and (last_close >= stop_loss_price)
                    stop_win_con = ((max_log_high-now_open_prcie > max_stop_win))  and (close_bar <ma_xiao) and (last_close>=ma_xiao2)
                    close_pos = False

                    if close_pos :
                        df1[i][1] = 0
                    elif stop_loss_con:
                        df1[i][1] = 0
                    elif stop_win_con:
                        df1[i][1] = 0

                elif now_pos < 0:
                    stop_loss_price = min_log_low
                    stop_loss_con = (close_bar > stop_loss_price) and (last_close <= stop_loss_price)
                    stop_win_con = False  #((close_bar < stop_win_price)) and (last_close >= stop_win_price)
                    close_pos = False     #(close_bar >me ) and (last_close <= me)
                    if close_pos :
                        df1[i][1] = 0
                    elif stop_loss_con:
                        df1[i][1] = 0
                    elif stop_win_con:
                        df1[i][1] = 0
            else: #非交易时间段
                if now_pos != 0:
                    df1[i][1] = 0
                    
            # 记录指标 绘图等等
            df2[i, 4] = stop_loss_price
            df2[i, 6] = min_log_low
            df2[i, 7] = max_log_high

    res0 = cal_tongji(df_input=df1)
    res0 = np.concatenate((res0,cs0))
    return df0,df1,df2,res0

def cal_signal(df,strat_time,end_time,cs0,ix_zong=100,all_day=0):

    a = time.process_time()
    # std__len,rooling_count,std_bd_n ,tp_n,zuhe_limit,max_stop_win = cs0
    #  ===指标数据
    df['tp'] = std_zdf_tp(df, int(cs0[0]), float(cs0[2]), int(cs0[0]))
    df['zuheyinzi'] = rsi_cci_atr01(df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']],n=int(cs0[0]))
    # df['zuheyinzi'] = (df['zuheyinzi'].diff(1).diff(1)).rolling(int(cs0[0])).mean()
    df['zuheyinzi'] = df['zuheyinzi'] - df['zuheyinzi'].shift(1)
    df['zuheyinzi'] = df['zuheyinzi'] - df['zuheyinzi'].shift(1)
    df['zuheyinzi'] = (talib.MAX(df['zuheyinzi'],int(cs0[0])) +talib.MIN(df['zuheyinzi'],int(cs0[0])))*0.5

    # print(df['zuheyinzi'].tail(20))
    df['xiao'] =talib.MA(df['close'],int(cs0[1]))
    df['stop_line'] =talib.MIN(df['close'],int(cs0[0]))

    df['止损价'] = np.nan
    df['止盈价'] = np.nan
    df['日内最低价'] = np.nan
    df['日内最高价'] = np.nan
    pass
    # ===转化成np.array
    df0cols = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume','days', 'huors', 'minutes']
    df0 = df[df0cols].values
    df1cols = ['candle_begin_time', 'signal', 'pos', 'opne_price', 'per_lr', 'sl']
    df1 = df[df1cols].values
    df2cols = ['candle_begin_time', 'tp',"zuheyinzi", 'xiao',"stop_line", '止损价', '止盈价', '日内最低价', '日内最高价']

    df2 = df[df2cols].values
    df0, df1, df2, res = cal_signal_(df0, df1, df2, strat_time, end_time, cs0,all_day)
    print('runingtime:', time.process_time() - a, f's ,已经完成 ==:{round(ix_zong,2)}%')
    # print(df0.shape, df1.shape, df2.shape,[df0cols,df1cols,df2cols], res)

    # res=[0]
    return df0, df1, df2,[df0cols,df1cols,df2cols],ix_zong, res


if __name__ == '__main__':

    # from empyrical import max_drawdown, alpha_beta,sharpe_ratio,annual_return
    filename = os.path.abspath(__file__).split('\\')[-1].split('.')[0]
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H')
    datapath = r'F:\vnpy_my_gitee\new_company\hsi_data_1min\HSI2011-2020-11.csv'
    dir_path = r'F:\task\恒生股指期货\numba_策略开发\numba_突破max01_突破_空策略'
    strat_time = np.array([1, 9, 20])
    end_time = np.array([27, 16, 20])
    # 1.先测试单进程 cpu == 1
    a = 1
    b = 10
    c = 10

    if True == a:
        # print(r'html_gather\%s_%s' % (filename, timeer0))
        # exit()
        df_time_list = [['2019-10-1 09:15:00', '2019-12-10 16:25:00']]
        s_time, e_time = df_time_list[0]
        df =  get_local_hsi_csv_pro(s_time, e_time,"datetime", datapath)  # 获取本地数据
        df000 = df.copy()
        ALL_DAY = 1
        # strat_time = np.array([1, 9,15])
        # end_time = np.array([27, 16, 20])
        time0 = time.process_time()
        df000 = df.copy()
        df000['ma'] = talib.SMA(df000['close'], 5000)

        #     std__len,rooling_count,std_bd_n ,tp_n,zuhe_limit,max_stop_win = cs0
        # 30.000 50.000  2.500  1.000 60.000 160.000
        # 30.0,50.0,2.5,1.0,60.0,160.0
        res0, resdf = mp_backtesting(df000,hc_func=cal_signal,
                                     zong_can=[[40,  50,  3,   1,  1, 160]],
                                      strat_time=strat_time,end_time=end_time,
        cpu_nums=1,all_day=ALL_DAY)

        print(res0)





        mode = 2
        if mode == 1:
            # , 'zsc_up', 'zsc_dn', 'zsc_ma', 'zsc0'
            resdf_baitian = resdf[(resdf['huors'] >= 2) & (resdf['huors'] <= 16)]
            resdf = resdf_baitian.iloc[-5000:].copy()
            resdf.reset_index(inplace=True)
            resdf.sort_values(by='candle_begin_time', inplace=True)
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       'xiao': resdf['xiao'],
                                       # 'stop_line': resdf['stop_line'],
                                       '日内最高价': resdf['日内最高价'],},
                        canshu2={"tp": resdf['tp'],"zuheyinzi": resdf['zuheyinzi']},
                        vol_bar=False, markline_show1=True,
                        path=f'html_gather\_2123123123' )
        if mode == 2:
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
                                       'xiao': resdf['xiao'],
                                       # 'stop_line': resdf['stop_line'],
                                       '日内最高价': resdf['日内最高价'], },
                        canshu2={"tp": resdf['tp'], "zuheyinzi": resdf['zuheyinzi']},
                        vol_bar=False, markline_show2=True,
                        path=r'html_gather\%s_%s' % (filename, timeer0))
        if mode == 3:
            if True == 0:
                resdf['next_open_price'] = resdf['opne_price'].shift(-1)
                resdf.loc[resdf['signal'].shift() == 1, 'pos_stime'] = resdf.loc[resdf['signal'].shift() == 1, 'candle_begin_time']
                resdf['pos_stime'].fillna(method='ffill', inplace=True)
                resdf.loc[resdf['pos'] == 0, 'pos_stime'] = pd.NaT
                res_trades = []
                print(res0)
                resdf['rsi_cf1'] = resdf['rsi'] - resdf['rsi'].shift(1)
                resdf['atr'] = talib.ATR(resdf['high'],resdf['low'],resdf['close'],18)
                resdf['zuheyinzi'] = resdf['rsi']+resdf['atr'] +resdf['cci']/10
                resdf['zuheyinzi2'] = resdf['close']-resdf['close'].shift(20) +resdf['zuheyinzi']

                yinzis = ['rsi',"cci","rsi_cf1","atr","zuheyinzi","zuheyinzi2"]
                for k, v in resdf.groupby('pos_stime'):
                    result_fenxi = []
                    result_fenxi.append(v.iloc[0]['candle_begin_time'])
                    # result_fenxi.append(v.iloc[-1]['candle_begin_time'])
                    result_fenxi.append((v.iloc[-1]['candle_begin_time'] - v.iloc[0]['candle_begin_time']) / timedelta(minutes=1))
                    result_fenxi.append(v.iloc[-1]['next_open_price'] - v.iloc[0]['opne_price'])
                    result_fenxi.append(v['close'].max() - v.iloc[0]['opne_price'])
                    result_fenxi.append(v['close'].max() - v.iloc[-1]['next_open_price'])
                    result_fenxi.append(v.iloc[0]['opne_price'] - v['close'].min())
                    result_fenxi.append(v.iloc[-1]['next_open_price'] - v['close'].min())
                    for c in yinzis:
                        result_fenxi.append(v.iloc[0][c])

                    res_trades.append(result_fenxi)
                # '开始时间', '结束时间',
                cols = ["开始时间", '持续分钟', '盈亏大小', '最大净值', "结束-最大", '最小净值', "结束-最小"]+yinzis
                df_zb = pd.DataFrame(res_trades, columns=cols)
                df_zb = df_zb.append(df_zb.describe(), ignore_index=False)
                df_zb = df_zb[cols]
                df_zb.to_csv(r'逐笔结果_资金曲线_%s.csv' % (filename))

            resdf['per_lr'].fillna(0, inplace=True)
            resdf['资金曲线'] = resdf['per_lr'].cumsum()
            resdf['资金曲线'].fillna(method='ffill', inplace=True)
            resdf['资金曲线2'] = resdf['资金曲线'].rolling(100).mean()
            resdf['资金曲线'].fillna(method='bfill', inplace=True)
            resdf['close'] = (resdf['close'] - resdf['close'].shift()).cumsum()
            resdf['up'], resdf['ma'], resdf['dn'] = talib.BBANDS(resdf['close'], 3500, 3, 3)

            only_line(resdf, zhibiaos=['dn', 'up', 'ma'], canshu=['资金曲线', '资金曲线2'], rule_type='1H',
                      result_info=res0, path='资金曲线test_%s.html'%(filename))

        print(f'总时间：{time.process_time() - time0}  s')
        exit()
    # 多进程策略 :
    # [190, 20, 2, 2]
    if True == b:
        # std__len,rooling_count ,tp_n,zuhe_limit,max_stop_win = cs0

        li = [20,50,2.5,1,0,160]
        # 10.0  50.0  1.0  1.5  170.0   160.0
        canshu_list = []
        for cs1 in range(15, 55, 5):
            for cs2 in range(20, 81, 10):
                for cs3 in range(5, 41, 5):
                    for cs4 in range(1, 3, 1):
                        for cs5 in range(-10, 11, 5):
                            for cs6 in range(140, 221, 10):
                                canshu_list.append([int(cs1), int(cs2), int(cs3)/10, int(cs4),int(cs5)/10,int(cs6)])

        # canshu_list = canshu_list[int(len(canshu_list)*0.45):int(len(canshu_list)*0.46)]
        df_time_list = [['2016-1-1 09:15:00', '2019-12-20 16:25:00']]
        path=os.getcwd()+r'\csv_gather\%s_%s2.csv' % (filename, timeer0)
        random.shuffle(canshu_list)
        canshu_list = random.sample(canshu_list,int(0.5*len(canshu_list)))
        print('参数列表个数：', len(canshu_list))

        s_time, e_time = df_time_list[0]
        df =  get_local_hsi_csv_pro(s_time, e_time,"datetime", datapath)  # 获取本地数据
        df000 = df.copy()
        ALL_DAY = 1
        time0 = time.process_time()
        resdf = mp_backtesting(df000, cal_signal,zong_can=canshu_list[:],strat_time=strat_time, end_time=end_time,
                                        path=path,cpu_nums=4,all_day=ALL_DAY)
        print(resdf[0].tail(10))
        df = pd.read_csv(path,index_col=0)
        df.reset_index(drop=True,inplace=True)
        print(df.tail(10))
        print(f'总时间：{time.process_time() - time0}  s')

    # 多进程策略 :
    if True == c:
        from numba_策略开发.回测工具.多阶段回测 import duojieduan_huice

        canshu_list = []
        for cs1 in range(180, 280, 10):
            for cs2 in range(10, 25, 5):
                for cs3 in range(2, 5, 1):
                    for cs4 in range(2, 5, 1):
                        for cs5 in range(3000, 5001, 1000):
                            for cs6 in range(1, 3, 1):
                                canshu_list.append([cs1, cs2, cs3, cs4, cs5, cs6])
        canshu_list = canshu_list[:]
        df_time_list = [['2016-1-10 09:15:00', '2019-12-28 16:25:00']]
        s_time, e_time = df_time_list[0]
        df =  get_local_hsi_csv_pro(s_time, e_time,"datetime", datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        name = f'{filename}_{s_time[:4]}_{e_time[:4]}'

        if True == 1:
            print('参数列表个数：', len(canshu_list))
            time0 = time.process_time()
            func_canshu = {'zong_can': canshu_list[:], 'cpu_nums': 2, "strat_time": strat_time, "end_time": end_time}
            df_zong = duojieduan_huice(df, duojincheng_backtesting, func_canshu, s_time, e_time, jiange='12MS')
            path_csv = dir_path + r'\csv_gather\%s.csv' % name
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
            html_path = dir_path + r'\策略介绍html\csv_html\%s.html' % name
            df_zong_html.to_html(html_path)

        # 读取本地数据,单独生成html，
        if True == 0:
            df_zong = pd.read_csv(dir_path + r'\csv_gather\%s.csv' % name)
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