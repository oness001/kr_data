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
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data ,get_loc_hsicsv,get_local_hsi_csv

'''
布林线突破 sma（std——min）+ma金叉+大周期+移动止赢。
'''
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
def cal_signal_(df0, df1, df2, strat_time, end_time, cs0, sxf=1, slip=1):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''
    # 配置参数:止盈，ma_len（atr），atr_倍数，交易次数
    stop_win_n, ma_len, da_ma_n,da_zq,sw_atrn,shift_n = cs0
    trade_inner_nums =1000
    # 配置临时变量
    inner_trade_nums = trade_inner_nums
    open_pos_size = 1
    max_log_high = np.nan
    min_log_low = np.nan
    max_sp = np.nan
    for i in range(10, df0.shape[0]):  # 类似于vnpy:on_bar
        # 交易时间判断：日：1-28，and （时，分：9：30-16：00）
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
            # '小均线', 'n_atr','移动止赢','大均线','max_n','up','dn','大周期','大均线','zsc_up','zsc_dn','zsc_ma','zsc0']
            # 6'小均线',   'n_atr', '移动止赢', '大均线', 'max_n',
            # 11'up',       'dn',     '大周期', '大均线2', 'zsc_up',
            # 16'zsc_dn',   'zsc_ma', 'zsc0'
            open_bar = df0[i][1]
            high_bar = df0[i][2]
            low_bar = df0[i][3]
            last_close = df0[i - 1][4]
            clsoe_bar = df0[i][4]
            swin_atr_n = df2[i-1][7] * sw_atrn
            ma_xiao = df2[i][6]
            da_ma = df2[i][9]

            dazhouqi = df2[i][13]
            da_ma2 = df2[i][14]
            z_up = df2[i][15]
            z_up_1 = df2[i-1][15]
            z_dn = df2[i][16]
            z_me = df2[i][17]
            z_zsc0 = df2[i][18]
            z_zsc0_1 = df2[i-1][18]
            # print(z_zsc0,z_up)
            # === 仓位统计。
            df1, now_open_prcie, now_pos = cal_per_pos(i, df1, open_pos_size, open_bar, clsoe_bar, last_close, sxf=1, slip=1)
            # === 当日变量加载
            if ((df0[i, 7] == 9) and (df0[i, 8] == 15)):
                inner_trade_nums = trade_inner_nums
                max_log_high = high_bar
                min_log_low = low_bar
            else:
                max_log_high = max(max(open_bar,clsoe_bar), max_log_high)
                min_log_low = min(min(open_bar,clsoe_bar), min_log_low)
                inner_low = df2[i - 1][3]
            stop_win_price = now_open_prcie + stop_win_n
            moving_swin_price =  np.nan
            # 信号生成
            if we_trade_con:
                if now_pos == 0:
                    long_condition = (z_zsc0 > z_up_1) and (z_zsc0_1 <= z_up_1) and (clsoe_bar>dazhouqi)and (da_ma2 < da_ma)
                    if long_condition and inner_trade_nums > 0:
                        inner_trade_nums -= 1
                        df1[i][1] = 1
                        max_sp = max_sp-swin_atr_n

                elif now_pos > 0:
                    stop_loss_con = False
                    close_pos = False

                    if max_log_high > stop_win_price:
                        moving_swin_price = ma_xiao
                    else:
                        moving_swin_price = ma_xiao-swin_atr_n
                    stop_win_con = (clsoe_bar < moving_swin_price)
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
            # df2[i, 1] = #stop_loss_price
            df2[i, 2] = stop_win_price
            df2[i, 3] = min_log_low
            df2[i, 4] = max_log_high
            df2[i, 8] = moving_swin_price

    res0 = cal_tongji(df_input=df1)
    res0 = np.concatenate((res0, cs0))
    return df0, df1, df2, res0

def cal_signal(df, strat_time, end_time, canshu):
    a = time.process_time()
    # ===转化成np.array
    # stop_win_n, ma_len, da_ma_n, da_zq, sw_atrn, shift_n
    # [215.0, 15.0, 150.0,2000, 4.5, 10.0]
    # stop_win_n, ma_len, da_ma_n,sw_atrn,trade_inner_nums = cs0
    df['小均线'] = talib.SMA(df['close'], int(canshu[1]))
    df['大周期'] = talib.SMA(df['close'], int(canshu[3]))

    df['xiao'] = talib.SMA(df['close'], canshu[5])
    df['大均线'] = talib.SMA(df['xiao'], int(canshu[2]))
    df['大均线2'] = talib.WMA(df['close'], int(canshu[2]))

    df['std'] = talib.STDDEV(df['xiao'], int(canshu[2]))
    df['zsc'] = (df['xiao']-df['大均线'])/df['std']

    df['zsc_ma']= talib.SMA(df['zsc'], int(canshu[2]))

    df['zsc_up']=df['zsc_ma']+talib.STDDEV(df['zsc'], int(canshu[2]))
    df['zsc_dn']=df['zsc_ma']-talib.STDDEV(df['zsc'], int(canshu[2]))

    df['zsc0'] = talib.SMA(df['zsc'],3)

    df['up']  = talib.MIN(df['大均线'] + df['std']*canshu[4],canshu[5])
    df['up']  = talib.SMA(df['up'],canshu[1])
    df['n_atr'] = talib.ATR(df['close'], df['high'], df['low'], int(canshu[1]))
    df['dn'] = df['大均线'] - df['std']*canshu[4]
    df['max_n'] = talib.MAX(df['close'], int(canshu[1]))

    df['移动止赢'] = np.nan

    df0cols = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    df1cols = ['candle_begin_time', 'signal', 'pos', 'opne_price', 'per_lr', 'sl']
    df2cols = ['candle_begin_time', '止损价', '止盈价', '日内最低价', '日内最高价', '开仓线',
               '小均线', 'n_atr','移动止赢','大均线','max_n','up','dn','大周期','大均线2','zsc_up','zsc_dn','zsc_ma','zsc0']
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
                    p.apply_async(cal_signal, args=(df, strat_time, end_time, np.array(canshu0),), callback=tianjia, )
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
        print(resdf.iloc[-10:])
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
        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比'] + [f'参数{i}' for i in range(1, len(zong_can[0]) + 1)]
        res0 = pd.DataFrame([res0], columns=cols)
        print(dfres.iloc[-10:])

        return dfres, res0

if __name__ == '__main__':

    filename = os.path.abspath(__file__).split('\\')[-1].split('.')[0]
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H')
    datapath = r'F:\vnpy_my_gitee\new_company\hsi_data_1min\HSI2011-2019_12.csv'


    dir_path =r'F:\task\恒生股指期货\numba_策略开发\numba_突破max01_突破_空策略'
    strat_time = np.array([1, 9, 20])
    end_time = np.array([27, 16, 20])
    # 1.先测试单进程 cpu == 1
    a = 1
    b = 10
    c = 10

    if True == a:

        df_time_list = [['2019-8-01 09:15:00', '2019-12-26 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_loc_hsicsv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        time0 = time.process_time()
        df000 = df.copy()
        # [215.000 15.000 150.000 2000.000  4.500 10.000]
        resdf, res0 = duojincheng_backtesting(df000, zong_can=[[215.0, 15.0, 150.0,2000, 4.5, 10.0]],
                                              strat_time=strat_time,
                                              end_time=end_time, cpu_nums=1,jiexi=False)


        print(res0)
        only_index(resdf, zhibiaos=["zsc_up",'zsc_dn' ,"zsc_ma","zsc0"], path='index.html')
        exit()
        mode = 1
        if mode == 1:
            # , 'zsc_up', 'zsc_dn', 'zsc_ma', 'zsc0'
            resdf_baitian = resdf[(resdf['huors'] >= 2) & (resdf['huors'] <= 16)]
            resdf = resdf_baitian.copy()
            resdf.reset_index(inplace=True)
            resdf.sort_values(by='candle_begin_time', inplace=True)
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       # '日内最低价': resdf['日内最低价'],
                                       # '日内最高价': resdf['日内最高价'],
                                       '小均线': resdf['小均线'],
                                       '大均线': resdf['大均线'],
                                       'max_n': resdf['max_n'],
                                       "up": resdf['up'],
                                       '移动止赢' :resdf['移动止赢'],
                                       "dn": resdf['dn'],

                                       },
                        canshu2={
                                        "zsc_up": resdf['zsc_up'],
                                       'zsc_dn' :resdf['zsc_dn'],
                                       "zsc_ma": resdf['zsc_ma'],
                                       "zsc0": resdf['zsc0']},
                        vol_bar=True, markline_show2=True,
                                path=r'html_gather\%s_%s' %(filename, timeer0))
        if mode == 2:
            resdf.loc[resdf['signal'] == 1, 'sig0'] = 1
            resdf.loc[resdf['signal'] == 0, 'sig1'] = 0
            resdf['sig0'] = resdf['sig0'].shift(-20)
            resdf['sig1'] = resdf['sig1'].shift(20)
            resdf['sig00'] = resdf[['sig0', 'sig1']].max(axis=1)
            resdf['sig00'].fillna(method='ffill', inplace=True)
            resdf = resdf[resdf['sig00']==1].copy()
            resdf.reset_index(drop=False,inplace=True)
            print(resdf.tail())
            print(res0)
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       '日内最高价': resdf['日内最高价'],
                                       '小均线': resdf['小均线'],
                                       '大均线': resdf['大均线'],
                                       'max_n': resdf['max_n'],
                                       "up": resdf['up'],
                                       '移动止赢' :resdf['移动止赢'],
                                       "dn": resdf['dn']
                                       }, vol_bar=True, markline_show2=True,
                        path=r'html_gather\%s_%s' %(filename, timeer0))
        if mode == 3:
            resdf['per_lr'].fillna(0, inplace=True)
            resdf['资金曲线'] = resdf['per_lr'].cumsum()
            resdf['资金曲线'].fillna(method='ffill', inplace=True)
            resdf['资金曲线2'] = resdf['资金曲线'].rolling(120).mean()
            resdf['资金曲线'].fillna(method='bfill', inplace=True)
            resdf['up'], resdf['ma'], resdf['dn'] = talib.BBANDS(resdf['close'], 3500, 3, 3)
            # , '资金曲线', '资金曲线2'
            only_line(resdf, zhibiaos=['dn', 'up', 'ma'], canshu=['资金曲线', '资金曲线2'], rule_type='1H', path='资金曲线test.html')

            print(res0)

        print(f'总时间：{time.process_time() - time0}  s')
        exit()
    # 多进程策略 :
    # [190, 20, 2, 2]
    if True == b:
        # [ 220.000 13.000 140.000 2500.000  4.000 10.000]
        canshu_list = []
        for cs1 in range(200, 226, 10):
            for cs2 in range(11, 16, 2):
                for cs3 in [i for i in range(120,201,20)]:
                    for cs4 in [i for i in range(1000, 3001, 1000)]:
                        for cs5 in range(30, 46,5):
                            for cs6 in range(8, 15, 2):
                                canshu_list.append([cs1, cs2, cs3,cs4, cs5/10,cs6])

        print('参数列表个数：', len(canshu_list))
        canshu_list = canshu_list[:]
        to_csv_path = r'csv_gather\%s_%s_粗回测.csv' % (filename,timeer0)
        df_time_list = [['2016-1-10 09:15:00', '2019-12-20 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_loc_hsicsv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()

        time0 = time.process_time()
        resdf = duojincheng_backtesting(df000, zong_can=canshu_list[:], strat_time=strat_time, end_time=end_time, cpu_nums=2)
        print(resdf.tail())
        resdf.to_csv(to_csv_path)
        print(f'总时间：{time.process_time() - time0}  s','\nsave to : ',to_csv_path)

    # 多进程策略 :
    if True == c:
        from numba_策略开发.回测工具.多阶段回测 import duojieduan_huice
        canshu_list = []
        for cs1 in range(190, 215, 5):
            for cs2 in range(16, 31, 2):
                for cs3 in range(2000, 3501, 500):
                    for cs4 in range(3, 6, 1):
                        # for cs5 in range(1, 2, 1):
                            # for cs6 in range(1, 3, 1):
                            canshu_list.append([cs1, cs2, cs3, cs4, 10])

        df_time_list = [['2016-1-10 09:15:00', '2019-12-28 16:25:00']]
        s_time, e_time = df_time_list[0]
        name = f'{filename}_{s_time[:4]}_{e_time[:4]}_3min'

        canshu_list = canshu_list[:10]
        df = get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
        df = transfer_to_period_data(df, rule_type='3T')
        print('数据大小', df.shape)
        df_day = transfer_to_period_data(df, rule_type='1D')
        df_day['atr_day'] = talib.SMA(talib.ATR(df_day['close'], df_day['high'], df_day['low'], 2), 2) * 0.8
        df_day['candle_begin_time'] = df_day['candle_begin_time'].apply(lambda x: x + datetime.timedelta(hours=9, minutes=15))
        df = pd.merge(df, df_day[['candle_begin_time', 'atr_day']], how='outer', on='candle_begin_time')
        df = df[df['volume'] > 0]
        df.fillna(method='ffill', inplace=True)
        print(df.tail(10))
        df000 = df.copy()

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
