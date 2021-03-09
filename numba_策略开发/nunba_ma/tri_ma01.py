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
pd.set_option('display.float_format', lambda x: '%.3f' % x)



@jit(nopython=True)
def cal_signal_(df0,df1,df2,strat_time,end_time,cs0):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''
    # 配置参数
    da , zhong,xiao ,stop_n,max_stop_win = cs0
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

        # 交易所日盘，开放
        if trading_time_con :
            # 快捷变量
            open_bar = df0[i][1]
            high_bar = df0[i][2]
            low_bar = df0[i][3]
            last_close = df0[i - 1][4]
            clsoe_bar = df0[i][4]
            # === 仓位统计。
            df1 ,now_open_prcie,now_pos =cal_per_pos(i,df1, open_pos_size, open_bar, clsoe_bar, last_close, sxf=1, slip=1)
            # === 当日变量加载
            if ((df0[i, 7] == 9) and (df0[i, 8] == 15)):
                # inner_trade_nums = 1
                max_log_high = high_bar
                min_log_low = low_bar
            else:

                max_log_high = max(high_bar, max_log_high)
                min_log_low = min(low_bar, min_log_low)

            ma_da = df2[i][1]
            ma_da2 =df2[i-1][1]
            ma_zhong = df2[i][2]
            ma_zhong2 =df2[i-1][2]
            ma_xiao = df2[i][3]
            ma_xiao2 = df2[i-1][3]

            # 我们交易时间
            we_trade_con = ((df0[i, 6] >= strat_time[0]) & (df0[i, 6] <= end_time[0]))
            we_trade_con &=(((df0[i, 7] == strat_time[1]) & (df0[i, 8] > strat_time[2])) | \
                           ((df0[i, 7] == end_time[1]) & (df0[i, 8] < end_time[2])) |\
                           ((df0[i, 7] > strat_time[1]) & (df0[i, 7] < end_time[1])))

            # 信号生成
            if we_trade_con:
                if now_pos == 0:
                    long_condition = (ma_xiao > ma_zhong)and (ma_zhong > ma_da) #三金叉
                    long_condition&=(ma_zhong>ma_zhong2)and(ma_da>ma_da2) #向上方向
                    long_condition &= (clsoe_bar > ma_zhong)and(last_close <= ma_zhong2) #突破中线
                    short_condition = False#(ma_xiao < ma_zhong) and (ma_zhong < ma_da)
                    if long_condition:
                        df1[i][1] = 1 #buy signal
                        max_log_high = high_bar
                    # 空：突破最低线。
                    elif short_condition :
                        min_log_low = low_bar
                        df1[i][1] = -1
                elif now_pos > 0:
                    stop_loss_price = now_open_prcie-stop_n if (max_log_high-now_open_prcie < 80) else now_open_prcie + 20
                    stop_loss_con = (clsoe_bar < stop_loss_price) and (last_close >= stop_loss_price)
                    stop_win_con = ((max_log_high-now_open_prcie > max_stop_win))  and (clsoe_bar <ma_xiao) and (last_close>=ma_xiao2)
                    close_pos = ((clsoe_bar < ma_da)) and (last_close >= ma_da2)
                    close_pos |= ((ma_xiao < ma_zhong)) and (ma_xiao2 >= ma_zhong2)

                    if close_pos :
                        df1[i][1] = 0
                    elif stop_loss_con:
                        df1[i][1] = 0
                    elif stop_win_con:
                        df1[i][1] = 0
                elif now_pos < 0:
                    stop_loss_price = min_log_low + stop_n
                    stop_loss_con = (clsoe_bar > stop_loss_price) and (last_close <= stop_loss_price)
                    stop_win_con = False#((clsoe_bar < stop_win_price)) and (last_close >= stop_win_price)
                    close_pos = False#(clsoe_bar >me ) and (last_close <= me)
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


def cal_signal(df,strat_time,end_time,cs0,ix_zong=100):
    a = time.process_time()
    #  ===指标数据
    df['ma_da'] = talib.MA(df['close'],cs0[0])
    df['ma_z'] = talib.MA(df['close'],cs0[1])
    df['ma_xiao'] = talib.MA(df['close'],cs0[2])
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
    df2cols = ['candle_begin_time', 'ma_da', 'ma_z', 'ma_xiao', '止损价', '止盈价', '日内最低价', '日内最高价']
    df2 = df[df2cols].values
    df0, df1, df2, res = cal_signal_(df0, df1, df2, strat_time, end_time, cs0)
    print('runingtime:', time.process_time() - a, f's ,已经完成 ==:{round(ix_zong,2)}%')
    # print(df0.shape, df1.shape, df2.shape,[df0cols,df1cols,df2cols], res)

    # res=[0]
    return df0, df1, df2,[df0cols,df1cols,df2cols], res

def cal_test(cs0):
    print('123')
    return cs0

def duojincheng_backtesting(df_input,zong_can,strat_time,end_time, cpu_nums=3):
    df = df_input.copy()
    if cpu_nums >cpu_count()-1:cpu_nums = cpu_count()-1
    huice_df = []

    def tianjia(res):
        huice_df.append(res[-1])

    def error_func(res):
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
    # 多进程回测
    zong_nums = len(zong_can) if len(zong_can)  > 0 else 1
    if cpu_nums > 1:
        p = Pool(processes=cpu_nums)
        for j in range(0, zong_nums, cpu_nums):
            for i in range(cpu_nums):
                if j + i <= len(zong_can) - 1:
                    canshu0 = zong_can[j + i]
                    cs0 = np.array(canshu0)
                    # cal_signal(df, strat_time, end_time, cs0)
                    ix_zong = (j + i)*100/zong_nums

                    p.apply_async(cal_signal, args=(df,strat_time,end_time,cs0,ix_zong,), callback = tianjia,error_callback=error_func,)
                else:
                    break
        p.close()
        p.join()
        print('进程池joined')
        # 整理多进程回测的数据

        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比']+[f'参数{i}' for i  in range(1,len(zong_can[-1])+1)]
        resdf = pd.DataFrame(huice_df, columns=cols)
        resdf = resdf[cols]
        resdf.sort_values(by='最后收益', inplace=True)
        print(resdf.iloc[-20:])
        print(f'=参数回测结束,谢谢使用.')
        return resdf,pd.DataFrame()

    # 单进程测试
    else:
        for cs0 in zong_can:
            a = time.process_time()
            cs0 = np.array(cs0)
            df0, df1, df2, cols, res0 = cal_signal(df, strat_time, end_time, cs0)
            huice_df.append(res0)
            # print('runingtime:', time.process_time() - a, 's')
        df0cols, df1cols, df2cols = cols
        # 转换成df数据
        df00 = pd.DataFrame(df0, columns=df0cols)
        df11 = pd.DataFrame(df1, columns=df1cols)
        df22 = pd.DataFrame(df2, columns=df2cols)
        #合并
        df11_= pd.merge(df00, df11, on="candle_begin_time", suffixes=('_0', '_1'))
        dfres = pd.merge(df11_, df22, on="candle_begin_time", suffixes=('', '_2'))
        dfres["candle_begin_time"] = pd.to_datetime(dfres["candle_begin_time"], unit='s')
        dfres.sort_values(by='candle_begin_time',inplace=True)
        cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', '平均收益', '开仓次数', '胜率', '盈亏比'] + [f'参数{i}' for i in range(1, len(zong_can[-1]) + 1)]
        res0 = pd.DataFrame(huice_df, columns=cols)
        res0 = res0[cols]
        res0.sort_values(by='最后收益', inplace=True)
        print(f'=参数回测结束,谢谢使用.')
        print(dfres.iloc[-20:])

        return res0,dfres

if __name__ == '__main__':


    filename = os.path.abspath(__file__).split('\\')[-1].split('.')[0]
    timeer0 = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d=%H_%M')
    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
    dir_path = r'F:\task\恒生股指期货\numba_策略开发\numba_突破max01_突破_空策略'

    # 1.先测试单进程 cpu == 1
    a = 1
    b = 100
    c = 0
    if True == a:
        df_time_list = [['2019-1-1 09:15:00', '2019-12-10 16:25:00']]
        s_time, e_time = df_time_list[0]
        df =  get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 20])
        time0 = time.process_time()
        df000 = df.copy()
        df000['ma'] = talib.SMA(df000['close'], 5000)
        # df000['ma2']=talib.MA(df000['close'],5000)
        # print(df000)
        # 152.000
        # 57.000
        # 20.000
        # 34.000
        # 120.000
        res0 ,resdf = duojincheng_backtesting(df000, zong_can=[[152, 57, 20, 34.0,120]],
                                              strat_time=strat_time,
                                              end_time=end_time, cpu_nums=1)

        # print(resdf[resdf['signal'] == -1])
        print(resdf[resdf['signal'] == -1].shape)

        print(res0)
        # resdf['atr_day'] = resdf['日内最低价'] + resdf['atr_day']
        df_draw = resdf.iloc[:].copy()
        df_draw.reset_index(drop = False,inplace =True)
        df_draw0 = transfer_period_anydata(df_draw,cols={'close':"last",'per_lr':"sum"},rule_type='15T')
        # df_draw0['per_lr'] = df_draw0['per_lr'].cumsum()
        only_line(df_draw0, path=r'html_gather\资金曲线%s_%s.html' % (filename, timeer0))
        mode = 111
        if mode == 1:

            resdf_baitian = df_draw[(df_draw['huors'] >= 2) & (df_draw['huors'] <= 16)]
            resdf = resdf_baitian.copy()
            resdf.reset_index(inplace=True)
            # resdf.sort_values(by='candle_begin_time', inplace=True)
            # ma_da  # ma_z # ma_xiao # 止损价 # 止盈价# 日内最低价# 日内最高价
            print(resdf.iloc[-10:])
            draw_charts(resdf, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       '日内最高价': resdf['日内最高价'],
                                       'ma_da': resdf['ma_da'],
                                       'ma_z': resdf['ma_z'],
                                       'ma_xiao': resdf['ma_xiao'],
                                       }, vol_bar=True,
                        path=r'html_gather\%s_%s' % (filename, timeer0),markline_show1=False,markline_show2=True)
        if mode == 2:
            df_draw = df_draw[(df_draw['pos'] == 1)&(df_draw['pos'] == -1) ].copy()
            draw_charts(df_draw, canshu={'opne_price': resdf['opne_price'],
                                       '止损价': resdf['止损价'],
                                       '止盈价': resdf['止盈价'],
                                       '日内最低价': resdf['日内最低价'],
                                       '日内最高价': resdf['日内最高价'],
                                       '小均线': resdf['小均线'],
                                       '大均线': resdf['大均线'],
                                       '移动止赢': resdf['移动止赢'],
                                       "atr_day": resdf['atr_day']
                                       }, vol_bar=True, markline_show2=True,
                        path=r'html_gather\%s_%s' % (filename, timeer0))
        print(f'总时间：{time.process_time() - time0}  s')
        exit()
    # 多进程策略 :
    # [190, 20, 2, 2]

    if True == b:
        canshu_list = []
        for cs1 in range(150, 157, 2):
            for cs2 in range(55, 60, 2):
                for cs3 in range(20, 24, 1):
                    for cs4 in range(30, 35, 2):
                        for cs5 in range(110, 121, 5):
                            canshu_list.append([cs1, cs2, cs3, cs4,cs5])

        canshu_list = canshu_list[:]
        df_time_list = [['2017-1-10 09:15:00', '2019-12-20 16:25:00']]
        s_time, e_time = df_time_list[0]
        df =  get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
        df000 = df.copy()
        strat_time = np.array([1, 9, 30])
        end_time = np.array([27, 16, 20])
        print('参数列表个数：', len(canshu_list))
        time0 = time.process_time()
        resdf = duojincheng_backtesting(df000, zong_can=canshu_list[:], strat_time=strat_time, end_time=end_time, cpu_nums=3)
        print(resdf[0].tail(10))
        resdf[0].to_csv(r'csv_gather\%s_%s.csv' % (filename, timeer0))
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
        df =  get_local_hsi_csv(s_time, e_time, datapath)  # 获取本地数据
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