'''
多阶段回测：输入一个总时间段，然后拆分成多个阶段，进行分阶段集中回测。
'''

import numpy as np
import time
from datetime import timedelta
import pandas as pd
import traceback
from functools import partial

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

def duojieduan_huice(df,funcname,func_canshu,s_time,e_time,jiange = '12MS'):
    time_list = list(pd.date_range(s_time,e_time,freq=jiange))
    time_list.append(pd.to_datetime(e_time))
    print(time_list)
    # exit()
    df_jieduan_zong = pd.DataFrame()
    for ix ,i in enumerate(time_list):
        if ix == 0 :continue
        s0 = time_list[ix-1]
        e0 = time_list[ix]
        df000 = df[(df['candle_begin_time'] >= pd.to_datetime(s0)) & (df['candle_begin_time'] <= pd.to_datetime(e0))]

        resdf = mp_backtesting(df000, funcname,zong_can=func_canshu['zong_can'],strat_time=func_canshu['strat_time'],
                         end_time=func_canshu['end_time'], cpu_nums=func_canshu['cpu_nums'],path = func_canshu['path'],all_day=func_canshu['all_day'])
        resdf['s_time'] = s0
        resdf['e_time'] = e0
        resdf.sort_values(by='最后收益',ascending=False,inplace=True)
        df_jieduan_zong = df_jieduan_zong.append(resdf,ignore_index=True)
    # df_jieduan_zong.set_index(keys=['s_time', "最后收益"], inplace=True)
    return df_jieduan_zong

def mp_backtesting(df_input,hc_func,zong_can,strat_time,end_time, path = r'',cpu_nums=3,all_day=0):
    from multiprocessing import Pool, cpu_count  # , Manager
    from functools import partial

    global huice_df ,zong_nums
    huice_df = []
    df = df_input.copy()
    zong_nums = len(zong_can) if len(zong_can) > 0 else 1
    res_cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', "索提诺比率", "盈亏风险率", '平均收益', '开仓次数', '胜率', '盈亏比'] + [f'参数{i}' for i in
                                                                                                          range(1, len(zong_can[-1]) + 1)]

    def tianjia(huice_df0,res):
        # 每10%保存一次
        global zong_nums
        huice_df0.append(res[-1])
        now_ix = int(res[-2]*zong_nums +1)
        # print(now_ix,zong_nums,int(zong_nums/10),now_ix % int(zong_nums/10))
        if (now_ix % int(zong_nums/10) == 0) or (now_ix==zong_nums):
            res_cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率', "索提诺比率", "盈亏风险率", '平均收益', '开仓次数', '胜率', '盈亏比'] +\
                       [f'参数{i}' for i in range(1, len(zong_can[-1]) + 1)]
            resdf0 = pd.DataFrame(huice_df0, columns=res_cols)
            resdf0 = resdf0[res_cols]
            resdf0.sort_values(by='最后收益', inplace=True)
            if now_ix == int(zong_nums/10):
                print(f"{len(resdf0)} 保存一次（{now_ix }%）",path)
                resdf0.to_csv(path, columns=res_cols, mode='a')
            elif now_ix==zong_nums:
                print(f" {now_ix} == {zong_nums}所有保存结束：（{now_ix*100/zong_nums}%）", path)
                resdf0.to_csv(path, columns=res_cols, header=0, mode='a')

            else:
                print(f" {len(resdf0)}保存一次（{now_ix }%）",path)
                resdf0.to_csv(path,columns=res_cols, header=0,mode='a')
            # print(resdf0.iloc[-20:],f"\n当前已经保存：{round(res[-2],4)}%...{res[-2]*len(zong_can)/100}—{len(zong_can)} ，个")

            huice_df0.clear()
            print(f"清空缓存:{len(huice_df0)}")
    def error_func(res):
        print(res)

    #  ===原始数据处理
    df['candle_begin_time'] = (df['candle_begin_time'] - np.datetime64(0, 's')) / timedelta(seconds=1)
    df['days'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().day))
    df['huors'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().hour))
    df['minutes'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().minute))
    #  ===信号仓位统计数据列，标准化接口，匹配  > .统计函数.py
    df['signal'] = np.nan
    df['pos'] = 0
    df['opne_price'] = np.nan
    df['per_lr'] = np.nan
    df['sl'] = np.nan
    # 多进程回测
    if cpu_nums >cpu_count()-1:cpu_nums = cpu_count()-1
    if cpu_nums > 1:
        p = Pool(processes=cpu_nums)
        for j in range(0, zong_nums, cpu_nums):
            for i in range(cpu_nums):
                if j + i <= len(zong_can) - 1:
                    canshu0 = zong_can[j + i]
                    cs0 = np.array(canshu0)
                    # cal_signal(df, strat_time, end_time, cs0)
                    ix_zong = (j + i)/zong_nums

                    p.apply_async(hc_func, args=(df,strat_time,end_time,cs0,ix_zong,all_day,), callback = partial(tianjia, huice_df,),error_callback=error_func,)
                else:
                    break
        p.close()
        p.join()
        print('进程池joined')
        # 整理多进程回测的数据
        try:
            resdf = pd.read_csv(path, index_col=0)
        except Exception:
            resdf = pd.DataFrame(huice_df, columns=res_cols)

        # df.reset_index(drop=True,inplace=True)
        # resdf = resdf[cols]
        # resdf.sort_values(by='最后收益', inplace=True)
        # print(resdf.iloc[-20:])
        print(f'=参数回测结束,谢谢使用.')
        return resdf,pd.DataFrame()
    # 单进程测试
    else:
        for cs0 in zong_can:
            a = time.process_time()
            cs0 = np.array(cs0)
            ix_zong = 100
            df0, df1, df2, cols, ix,res0 = hc_func(df, strat_time, end_time, cs0,ix_zong,all_day=all_day)
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
        # cols = ['最后收益', '最大收益', '模拟最大回撤', '赢撤率', '夏普率',"索提诺比率","盈亏风险率", '平均收益', '开仓次数', '胜率', '盈亏比'] +[f'参数{i}' for i in range(1, len(zong_can[-1]) + 1)]
        res0 = pd.DataFrame(huice_df, columns=res_cols)
        res0 = res0[res_cols]
        res0.sort_values(by='最后收益', inplace=True)
        print(f'=参数回测结束,谢谢使用.')
        print(dfres.iloc[-10:])
        return res0,dfres
