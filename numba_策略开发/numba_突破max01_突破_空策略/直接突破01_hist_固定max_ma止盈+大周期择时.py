from numba import jit, int64, float32
import numpy as np
import time
import talib
from datetime import timedelta
import pandas as pd
import traceback
from multiprocessing import Pool, cpu_count  # , Manager
from numba_策略开发.画图工具.echart_可加参数 import draw_charts,only_line,draw_line_charts
from numba_策略开发.回测工具.统计函数 import cal_tongji, cal_per_pos
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data, get_local_hsi_csv,transfer_period_anydata


'''
将止损改为移动止损，方式为：固定移动跟踪止损。
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
pd.set_option('display.max_rows', 60000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


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
#
# @jit(nopython=True)
# def cal_tongji(df_input: np.array):
#     df1 = df_input.copy()
#     # 返回统计数据
#     df1[:, 4][np.isnan(df1[:, 4])] = 0
#     zjline = np.cumsum(df1[:, 4])
#     maxback_line = zjline.copy()
#     max_log = 0
#     for i in range(1, zjline.shape[0]):
#         max_log = max(max(zjline[i], zjline[i - 1]), max_log)
#         maxback_line[i] = max_log - zjline[i]
#
#     end_zj = zjline[-1]
#     max_zj = max(zjline)
#     std_zj = np.std(zjline)
#     mean_zj = np.mean(zjline)
#     max_back = max(maxback_line)
#     y_ = (df1[:, 5][np.where(df1[:, 5] > 0)])
#     k_ = (df1[:, 5][np.where(df1[:, 5] <= 0)])
#     counts = (y_.shape[0] + k_.shape[0])
#     sl = y_.shape[0] / (y_.shape[0] + k_.shape[0]) if counts != 0 else 1
#     yk_rate = 1 / (y_.sum() * -1 / k_.sum()) if k_.sum() != 0 else 1
#     yc_rate = end_zj / max_back if max_back != 0 else 0
#     sharp_rate = mean_zj / std_zj if std_zj != 0 else 0
#
#     res0 = np.array([end_zj, max_zj, (max_back), yc_rate, sharp_rate, mean_zj, (counts), sl, yk_rate])
#     return res0
#
# @jit(nopython=True)
# def cal_per_pos(i, df1: np.array, open_pos_size, open_bar, clsoe_bar, last_close, sxf=1, slip=1):
#     last_signal = df1[i - 1][1]
#     last_bar_pos = df1[i - 1][2]
#     last_open_prcie = df1[i - 1][3]
#
#     # 开多===上一根信号=1 仓位 ==0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点
#     if last_signal >= 1 and last_bar_pos == 0:
#         df1[i][2] = open_pos_size
#         df1[i][3] = open_bar + sxf + slip
#         df1[i][4] = (clsoe_bar - open_bar) * open_pos_size - (sxf + slip)
#
#     # # # 空转多===上一根信号=1 仓位 < 0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点， 结束平仓胜负【5】
#     elif last_signal == 1 and last_bar_pos < 0:
#         df1[i][2] = open_pos_size
#         df1[i][3] = open_bar + sxf + slip
#         df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
#         df1[i][4] += (clsoe_bar - open_bar) * open_pos_size - (sxf + slip)
#         df1[i][5] = open_bar - (sxf + slip) - last_open_prcie
#
#     # # 多转空===上一根信号=-1 仓位 > 0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点， 结束平仓胜负【5】
#     elif (last_signal == -1) and (last_bar_pos > 0):
#         df1[i][2] = -1 * open_pos_size
#         df1[i][3] = open_bar - (sxf + slip)
#         # 先平多，在开空
#         df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
#         df1[i][4] += (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)
#         df1[i][5] = open_bar - (sxf + slip) - last_open_prcie
#
#     # 开空===上一根信号=-1， 【2】仓位==0,   【3】记录持仓价，  【4】当日盈亏-手续费和滑点
#     elif last_signal < 0 and last_bar_pos == 0:  # ===开空
#         df1[i][2] = -1 * open_pos_size
#         df1[i][3] = open_bar - (sxf + slip)
#         df1[i][4] = (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)
#
#     # 平仓===上一根信号=0 【2】仓位 != 0,记录平仓价【3】，当日盈亏【4】-手续费和滑点
#     elif last_signal == 0 and abs(last_bar_pos) > 0:
#         df1[i][2] = 0  # 平仓
#         df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
#         if last_bar_pos > 0:
#             df1[i][3] = open_bar - (sxf + slip)
#             df1[i][5] = open_bar - (sxf + slip) - last_open_prcie
#         elif last_bar_pos < 0:
#             df1[i][3] = open_bar + sxf + slip
#             df1[i][5] = open_bar + sxf + slip - last_open_prcie
#     # 仓位不变===记录开/平仓价【3】，当日盈亏【4】，变化点数乘以仓位
#     else:
#         df1[i][2] = last_bar_pos
#         df1[i][3] = last_open_prcie
#         df1[i][4] = (clsoe_bar - last_close) * df1[i][2]
#
#     return df1, df1[i][3], df1[i][2]

@jit(nopython=True)
def cal_signal_(df0, df1, df2, strat_time, end_time, cs0, sxf=1, slip=1):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''
    # 配置参数:止盈，ma_len（atr），atr_倍数，交易次数
    stop_win_n, ma_len, atrn,sw_atrn,da_ma_n, trade_inner_nums = cs0
    # 配置临时变量
    inner_trade_nums = trade_inner_nums
    open_pos_size = 1
    max_log_high = np.nan
    min_log_low = np.nan
    for i in range(10, df0.shape[0]):  # 类似于vnpy:on_bar
        # 交易时间判断：日：1-28，and （时，分：9：30-16：00）
        trading_time_con = ((df0[i, 6] >= strat_time[0]) and (df0[i, 6] <= end_time[0])) and \
                           ((df0[i, 7] == 9) and (df0[i, 8] >= 15)) or \
                           ((df0[i, 7] == 16) and (df0[i, 8] <= 29)) or \
                           ((df0[i, 7] > 9) and (df0[i, 7] < 16))
        # 我们交易时间
        #日
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
            clsoe_bar = df0[i][4]
            atr_n = df2[i-1][8] * atrn
            swin_atr_n = df2[i-1][8] * sw_atrn
            ma_xiao = df2[i][7]
            da_ma = df2[i][10]

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

            stop_loss_price = (now_open_prcie + atr_n)
            stop_win_price = now_open_prcie - stop_win_n
            moving_swin_price =  np.nan

            # 信号生成
            if we_trade_con:

                if now_pos == 0:
                    short_condition =  (clsoe_bar < inner_low) and (last_close >= inner_low) and (clsoe_bar<da_ma)
                    # 空：突破最低线。
                    if short_condition and inner_trade_nums > 0:
                        inner_trade_nums -= 1
                        df1[i][1] = -1
                elif now_pos < 0:
                    stop_loss_con = (clsoe_bar > stop_loss_price) and\
                                    (last_close <= stop_loss_price)
                    if min_log_low <stop_win_price:
                        stop_win_con = (clsoe_bar > ma_xiao)
                        moving_swin_price = ma_xiao
                    else:
                        stop_win_con = (clsoe_bar > ma_xiao + swin_atr_n)
                        moving_swin_price = ma_xiao+swin_atr_n

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
            df2[i, 1] = stop_loss_price
            df2[i, 2] = stop_win_price
            df2[i, 3] = min_log_low
            df2[i, 4] = max_log_high
            df2[i, 9] = moving_swin_price

    res0 = cal_tongji(df_input=df1)
    res0 = np.concatenate((res0, cs0))
    return df0, df1, df2, res0

def cal_signal(df, strat_time, end_time, canshu):
    a = time.process_time()
    # ===转化成np.array
    df0cols = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    df1cols = ['candle_begin_time', 'signal', 'pos', 'opne_price', 'per_lr', 'sl']
    df2cols = ['candle_begin_time', '止损价', '止盈价', '日内最低价', '日内最高价', '开仓线',
               'atr_day', '小均线', 'n_atr','移动止赢','大均线']
    # 配置参数:止盈，ma_len（atr），atr_倍数，交易次数
    # stop_win_n, ma_len, atrn, trade_inner_nums = cs0
    fast,slow,df['hist_柱子'] = talib.MACD(df['close'],canshu[1],canshu[2], int(canshu[3]))
    df['大均线'] = talib.SMA(df['close'], int(canshu[4]))
    df['n_atr'] = talib.ATR(df['close'], df['high'], df['low'], int(canshu[2]))
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
        print(resdf.iloc[-20:])
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
        print(dfres.iloc[-20:])

        return dfres, res0

if __name__ == '__main__':
    import time, datetime
    import os
    filename = os.path.abspath(__file__).split('\\')[-1].split('.')[0]
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H_%M')
    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
    dir_path =r'F:\task\恒生股指期货\numba_策略开发\numba_突破max01_突破_空策略'

    # 1.先测试单进程 cpu == 1
    a = 10
    b = 1
    c = 0
    if True == a:
        df_time_list = [['2019-10-1 09:15:00', '2019-12-10 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_loc_hsicsv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        time0 = time.process_time()
        df000 = df.copy()
        df000['ma']=talib.SMA(df000['close'],5000)
        # df000['ma2']=talib.MA(df000['close'],5000)
        # print(df000)
        # exit()
        # resdf0,res01 = duojincheng_backtesting(df000.iloc[:1000], zong_can=[[80,180,200,0]], strat_time=strat_time, end_time=end_time, cpu_nums = 1)

        # resdf0, res00 = duojincheng_backtesting(df000.iloc[:3000],
        #                                         zong_can=[[230, 16, 3,1]],
        #                                         strat_time=strat_time,
        #                                         end_time=end_time, cpu_nums=1,jiexi=True)
        # time.sleep(0.2)
        resdf, res0 = duojincheng_backtesting(df000, zong_can=[[210.0, 10.0, 4.0, 4.0, 5000.0, 1.0]],
                                              strat_time=strat_time,
                                              end_time=end_time, cpu_nums=1,jiexi=False)

        print(resdf[resdf['signal']==-1])
        print(resdf[resdf['signal']==-1].shape)

        print(res0)
        exit()
        resdf['atr_day'] = resdf['日内最低价'] + resdf['atr_day']
        mode = 112
        if mode == 1:
            resdf_baitian = resdf[(resdf['huors'] >= 2) & (resdf['huors'] <= 16)]
            resdf = resdf_baitian.copy()
            resdf.reset_index(inplace=True)
            resdf.sort_values(by='candle_begin_time', inplace=True)
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       '日内最高价': resdf['日内最高价'],
                                       '开仓线': resdf['开仓线'],
                                       '小均线': resdf['小均线'],
                                       '大均线': resdf['大均线'],

                                       '移动止赢' :resdf['移动止赢'],
                                       "atr_day": resdf['atr_day']
                                       }, vol_bar=True, markline_show2=True,
                                path=r'html_gather\%s_%s' %(filename, timeer0))
        if mode == 2:
            resdf = resdf[resdf['pos'].notnull()].copy()
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       '日内最高价': resdf['日内最高价'],
                                       '小均线': resdf['小均线'],
                                       '大均线': resdf['大均线'],
                                       '移动止赢' :resdf['移动止赢'],
                                       "atr_day": resdf['atr_day']
                                       }, vol_bar=True, markline_show2=True,
                        path=r'html_gather\%s_%s' %(filename, timeer0))
        print(f'总时间：{time.process_time() - time0}  s')
        exit()
    # 多进程策略 :
    # [190, 20, 2, 2]

    if True == b:
        canshu_list = [[210.0, 10.0, 4.0, 4.0, 5000.0, 1.0]]
        for cs1 in range(200, 280, 10):
            for cs2 in range(10, 20, 3):
                for cs3 in range(2, 5, 1):
                    for cs4 in range(1, 3, 1):
                        canshu_list.append([cs1, cs2, cs3,cs3,5000, cs4])
        canshu_list = canshu_list[:10]
        df_time_list = [['2019-10-10 09:15:00', '2019-12-20 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_loc_hsicsv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        print('参数列表个数：', len(canshu_list))
        time0 = time.process_time()
        resdf = duojincheng_backtesting(df000, zong_can=canshu_list[:], strat_time=strat_time, end_time=end_time, cpu_nums=2)
        print(resdf.tail())
        resdf.to_csv(r'csv_gather\%s_%s.csv' % (filename,timeer0))
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
                                canshu_list.append([cs1, cs2, cs3, cs4,cs5,cs6])
        canshu_list = canshu_list[:]
        df_time_list = [['2016-1-10 09:15:00', '2019-12-28 16:25:00']]
        s_time, e_time = df_time_list[0]
        df = get_loc_hsicsv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 10])
        name = f'{filename}_{s_time[:4]}_{e_time[:4]}'

        if True == 1:
            print('参数列表个数：', len(canshu_list))
            time0 = time.process_time()
            func_canshu = {'zong_can': canshu_list[:], 'cpu_nums': 2, "strat_time": strat_time, "end_time": end_time}
            df_zong = duojieduan_huice(df, duojincheng_backtesting, func_canshu, s_time, e_time, jiange='12MS')
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
