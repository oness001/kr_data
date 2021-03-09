from numba import jit
import numpy as np
import time
from datetime import timedelta,datetime
import pandas as pd
import traceback

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
def cal_signal(df0, df1, df2, strat_time, end_time, cs1=0, cs2=0, cs3=0, cs4=0, sxf=1, slip=1):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''

    # 配置参数
    stop_n = cs1
    max_stop = cs2
    # 配置临时变量
    now_day=0
    inner_trading_nums = 1
    open_pos_size = 1
    max_yk_close = 0
    min_yk_close = 0

    for i in range(10, df0.shape[0]):  # 类似于vnpy:on_bar
        # 交易时间判断：日：1-28，and （时，分：9：30-16：00）
        trading_time_con = ((df0[i, 6] >= strat_time[0]) and (df0[i, 6] <= end_time[0])) and \
                           ((df0[i, 7] == 9) and (df0[i, 8] >= 15)) or \
                           ((df0[i, 7] == 16) and (df0[i, 8] <= 29)) or \
                           ((df0[i, 7] > 9) and (df0[i, 7] < 16))

        we_trade_con = ((df0[i, 6] >= strat_time[0]) and (df0[i, 6] <= end_time[0])) and \
                       ((df0[i, 7] > strat_time[1]) and (df0[i, 7] < end_time[1])) or \
                       ((df0[i, 7] == strat_time[1]) and (df0[i, 8] > strat_time[2])) or \
                       ((df0[i, 7] == end_time[1]) and (df0[i, 8] < end_time[2]))

        # 交易所日盘开放交易时间
        if trading_time_con:
            # 快捷变量
            open_bar = df0[i][1]
            high_bar = df0[i][2]
            low_bar = df0[i][3]
            last_close = df0[i - 1][4]
            clsoe_bar = df0[i][4]
            # inner指标计算
            if ((df0[i, 7] == 9) and (df0[i, 8] == 15)): #strat_time[2]
                max_yk_close = high_bar
                min_yk_close = low_bar
                inner_trading_nums = 1
                low_inner = low_bar
            else:
                max_yk_close = max(high_bar, max_yk_close)
                min_yk_close = min(low_bar, min_yk_close)
                low_inner = df2[i-1, 7]
                # === 仓位统计。
            if 1 == True:
                last_signal = df1[i - 1][1]
                last_bar_pos = df1[i - 1][2]
                last_open_prcie = df1[i - 1][3]

                # 开空===上一根信号=-1， 【2】仓位==0,   【3】记录持仓价，  【4】当日盈亏-手续费和滑点
                if last_signal < 0 and last_bar_pos == 0:  # ===开空
                    df1[i][2] = -1 * open_pos_size
                    df1[i][3] = open_bar - (sxf + slip)
                    df1[i][4] = (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)

                    pass
                # 平仓===上一根信号=0 【2】仓位 != 0,记录平仓价【3】，当日盈亏【4】-手续费和滑点
                elif last_signal == 0 and abs(last_bar_pos) > 0:

                    df1[i][2] = 0  # 平仓
                    df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
                    if last_bar_pos > 0:
                        df1[i][3] = open_bar - (sxf + slip)
                        df1[i][5] = open_bar - (sxf + slip) - last_open_prcie
                    elif last_bar_pos < 0:
                        df1[i][3] = open_bar + sxf + slip
                        df1[i][5] = open_bar + sxf + slip - last_open_prcie
                # 仓位不变===记录开/平仓价【3】，当日盈亏【4】，变化点数乘以仓位
                else:
                    df1[i][2] = last_bar_pos
                    df1[i][3] = last_open_prcie
                    df1[i][4] = (clsoe_bar - last_close) * df1[i][2]
            # 最新仓位
            now_open_prcie = df1[i][3]
            now_pos = df1[i][2]

            stop_loss_price = now_open_prcie + stop_n
            stop_win_price = now_open_prcie - max_stop
            if now_day != df0[i, 6]:
                now_day = df0[i, 6]

            # 信号生成
            if we_trade_con:
                if now_pos == 0 and inner_trading_nums > 0:
                    # 空：突破最低线。
                    short_condition = (clsoe_bar < low_inner) and (last_close >= low_inner)  # and False
                    if short_condition:
                        df1[i][1] = -1
                        inner_trading_nums = 0
                elif now_pos < 0:
                    stop_con = (clsoe_bar > stop_loss_price) and \
                               (last_close <= stop_loss_price)

                    stop_y_con = ((clsoe_bar < stop_win_price)) and \
                                 (last_close >= stop_win_price)
                    close_chort = False

                    if close_chort:
                        df1[i][1] = 0
                    elif stop_con:
                        df1[i][1] = 0
                    elif stop_y_con:
                        df1[i][1] = 0
            else:
                # 有持仓，平仓！
                if now_pos != 0:
                    df1[i][1] = 0

        else:
            min_yk_close = np.nan
            stop_loss_price = np.nan
        # 记录指标
        df2[i, 6] = stop_loss_price
        df2[i, 7] = min_yk_close


    # 返回统计数据
    df1[:, 4][np.isnan(df1[:, 4])] = 0
    zjline = np.cumsum(df1[:, 4])
    end_zj = zjline[-1]
    max_zj = max(zjline)
    std_zj = np.std(zjline)
    mean_zj = np.mean(zjline)
    y_ = (df1[:, 5][np.where(df1[:, 5] > 0)])
    k_ = (df1[:, 5][np.where(df1[:, 5] <= 0)])
    counts = (y_.shape[0] + k_.shape[0])
    sl = y_.shape[0] / (y_.shape[0] + k_.shape[0]) if counts != 0 else 1
    yk_rate = 1 / (y_.sum() * -1 / k_.sum()) if k_.sum() != 0 else 1
    yc_rate = end_zj / (max_zj - mean_zj + std_zj * 0.62) if max_zj != mean_zj + std_zj * 0.62 else 0
    sharp_rate = mean_zj / std_zj if std_zj != 0 else 0
    res0 = np.array([end_zj, max_zj, (max_zj - mean_zj + std_zj),
                     yc_rate, sharp_rate, mean_zj, (counts),
                     sl, yk_rate, cs1, cs2, cs3, cs4])

    return df0, df1, df2, res0


def signal_test(df0, df1, df2, canshu, strat_time=np.array([1, 9, 30]), end_time=np.array([27, 16, 10])):
    a = time.process_time()
    cs1, cs2 = canshu
    df0, df1, df2, res = cal_signal(df0, df1, df2, strat_time, end_time, cs1, cs2, cs3=0, cs4=0)
    # print('runingtime:', time.process_time() - a, 's')
    return df0, df1, df2, res


def duojincheng_backtesting(df_input, zong_can, strat_time, end_time, cpu_nums=3):
    df = df_input.copy()
    from multiprocessing import Pool  # , cpu_count, Manager
    huice_df = []
    def tianjia(res):
        huice_df.append(res[-1])
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
    df['stop_line'] = np.nan
    df['low_inner'] = np.nan
    pass

    # ===转化成np.array
    df0cols = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume',
               'days', 'huors', 'minutes']
    df0 = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume',
              'days', 'huors', 'minutes']].values
    df1cols = ['candle_begin_time', 'signal', 'pos',
               'opne_price', 'per_lr', 'sl']
    df1 = df[['candle_begin_time', 'signal', 'pos',
              'opne_price', 'per_lr', 'sl']].values
    df2cols = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume',
               'stop_line', 'low_inner']
    df2 = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume',
              'stop_line', 'low_inner']].values
    # 多进程回测
    if cpu_nums > 1:
        p = Pool(processes=cpu_nums)
        for j in range(0, len(zong_can), cpu_nums):
            for i in range(cpu_nums):
                if j + i <= len(zong_can) - 1:
                    canshu0 = zong_can[j + i]
                    canshu = np.array(canshu0)
                    p.apply_async(signal_test, args=(df0, df1, df2, canshu, strat_time, end_time,), callback=tianjia, )
                else:
                    break
        p.close()
        p.join()
        # print('进程池joined')
        # 整理多进程回测的数据
        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比', '参数1', '参数2', '参数3', '参数4']
        resdf = pd.DataFrame(huice_df, columns=cols)
        resdf = resdf[cols]
        resdf.sort_values(by='最后收益', inplace=True)
        # print(resdf.iloc[-20:])
        # print(f'一轮回测结束。next。。。')
        return resdf

    # 单进程测试
    else:
        for cs0 in zong_can:
            canshu0 = np.array(cs0)

            df0, df1, df2, res0 = signal_test(df0, df1, df2, canshu=canshu0, strat_time=strat_time, end_time=end_time)
            tianjia(res=res0)
        # 转换成df数据
        df00 = pd.DataFrame(df0, columns=df0cols)
        df11 = pd.DataFrame(df1, columns=df1cols)
        df22 = pd.DataFrame(df2, columns=df2cols)
        # 合并
        df11_ = pd.merge(df00, df11, on="candle_begin_time", suffixes=('_0', '_1'))
        dfres = pd.merge(df11_, df22, on="candle_begin_time", suffixes=('', '_2'))
        dfres["candle_begin_time"] = pd.to_datetime(dfres["candle_begin_time"], unit='s')
        dfres.sort_values(by='candle_begin_time', inplace=True)
        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比', '参数1', '参数2', '参数3', '参数4']

        res = pd.DataFrame([res0],columns=cols)
        print(res)
        return dfres


if __name__ == '__main__':
    from echart_可加参数 import draw_charts

    df_time_list = [['2020-1-01 09:20:00', '2020-12-26 16:20:00']]
    s_time, e_time = df_time_list[0]
    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI202001-202010.csv'
    print('数据地址：', datapath)
    df = pd.read_csv(filepath_or_buffer=datapath)
    df['candle_begin_time'] = df['datetime']
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    df = df[(df['candle_begin_time'] >= pd.to_datetime(s_time)) & (df['candle_begin_time'] <= pd.to_datetime(e_time))]
    # df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
    df.reset_index(inplace=True, drop=True)
    print('数据大小', df.shape)
    df.ffill(inplace=True)
    df000 = df.copy()

    # 1.先测试单进程 cpu == 1
    if 0 == True:
        strat_time = np.array([1, 9, 20])
        end_time = np.array([27, 16, 10])
        time0 = time.process_time()
        resdf = duojincheng_backtesting(df000, zong_can=[[119, 180]], strat_time=strat_time, end_time=end_time, cpu_nums=1)
        draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],

                                   'stop_line': resdf['stop_line'],
                                   'low_inner': resdf['low_inner']
                                   }, vol_bar=True, markline_show2=True, path='1126_test')
        print(resdf.tail())

        print(f'总时间：{time.process_time() - time0}  s')

    # 多进程策略 :
    if 0 == True:
        strat_time = np.array([1, 9, 20])
        end_time = np.array([27, 16, 20])

        canshu_list = []

        # for cs1 in range(450,460, 10):
        for cs2 in range(50, 121, 3):
            for cs3 in range(150, 250, 5):
                # for cs4 in range(10, 20, 10):
                canshu_list.append([cs2, cs3])

        print('参数列表个数：', len(canshu_list))
        time0 = time.process_time()
        resdf = duojincheng_backtesting(df000, zong_can=canshu_list, strat_time=strat_time, end_time=end_time, cpu_nums=3)
        print(resdf.tail())
        resdf.to_csv(r'F:\task\恒生股指期货\numba_突破max01_策略2019-2019_12.csv')
        print(f'总时间：{time.process_time() - time0}  s')

    # 多进程逐年滚动测试策略:
    if 1 == True:
        end_res_df = pd.DataFrame()
        df_res_zong =pd.DataFrame()

        df_time_list = [['2016-1-01', '2019-12-26']]
        s_time, e_time = df_time_list[0]
        canshu_list = []
        # for cs1 in range(450,460, 10):
        for cs2 in range(60, 121, 3):
            for cs3 in range(150, 230, 3):
                # for cs4 in range(10, 20, 10):
                canshu_list.append([cs2, cs3])
        datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
        print('数据地址：', datapath)
        df = pd.read_csv(filepath_or_buffer=datapath)
        df['candle_begin_time'] = df['datetime']
        df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
        df = df[(df['candle_begin_time'] >= pd.to_datetime(s_time)) & (df['candle_begin_time'] <= pd.to_datetime(e_time))]
        # df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
        df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
        df.reset_index(inplace=True, drop=True)
        print('数据大小', df.shape)

        for p in ['1MS','2MS','3MS','4MS','5MS','6MS']: #'1MS','2MS','3MS','4MS','5MS','6MS'
            time0 = time.process_time()
            time_ = pd.date_range(start=s_time, end=e_time,freq=p)
            for i ,t in enumerate(time_):
                ts =time_[i]
                te = time_[i+1] if i+1<= len(time_)-1 else ts

                df0 = df[(df['candle_begin_time'] >= pd.to_datetime(ts)) & (df['candle_begin_time'] < pd.to_datetime(te))].copy()
                if df0.empty:continue
                strat_time = np.array([1, 9, 20])
                end_time = np.array([27, 16, 20])
                # print('参数列表个数：', len(canshu_list))
                resdf = duojincheng_backtesting(df0, zong_can=canshu_list, strat_time=strat_time, end_time=end_time, cpu_nums=3)
                resdf['开始时间'] = ts
                df_res_zong =df_res_zong.append(resdf,ignore_index=True)

            df_res_zong.to_csv(r'F:\task\恒生股指期货\numba_突破max01_策略\numba_突破max01_策略'+s_time+'_'+e_time+'.csv')
            # print(df_res_zong.keys())
            print(df_res_zong.tail())
            print(f'总时间：{time.process_time() - time0}  s')

        # if 1 ==True:
            shaixuandf = pd.DataFrame()
            df_temp = pd.read_csv(filepath_or_buffer=r'F:\task\恒生股指期货\numba_突破max01_策略\numba_突破max01_策略'+s_time+'_'+e_time+'.csv')
            df_temp['参数'] = df_temp[['参数1','参数2','参数3','参数4']].values.tolist()
            df_temp['参数'] = df_temp['参数'].astype(str)
            df_temp['下周期收益'] = 0
            for k ,v in df_temp.groupby('参数'):
                df_temp.loc[df_temp['参数']==k,'下周期收益'] = v['最后收益'].shift(-1)
            print(df_temp.tail())

            for k, v in df_temp.groupby('开始时间'):
                v_=v.sort_values(by = '最后收益',ascending=True)
                shaixuandf = shaixuandf.append(v_.iloc[-1].copy(),ignore_index=True)
            shaixuandf['下周期收益'].fillna(0,inplace=True)
            shaixuandf['最终收益line'] = shaixuandf['下周期收益'].cumsum()
            shaixuandf['滚动周期'] = p
            end_res_df = end_res_df.append(shaixuandf,ignore_index=True)
            print(shaixuandf)
            print(p,':')
            print('最终收益',shaixuandf['下周期收益'].sum())
            print('方差',shaixuandf['下周期收益'].std())
            print('均值',shaixuandf['下周期收益'].mean())

        end_res_df.to_csv(r'F:\task\恒生股指期货\numba_突破max01_策略\滚动回测_'+s_time+'_'+e_time+'.csv')




