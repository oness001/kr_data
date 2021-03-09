'''
多阶段回测：输入一个总时间段，然后拆分成多个阶段，进行分阶段集中回测。
'''

import numpy as np
import time
from datetime import timedelta
import pandas as pd
import traceback
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

        resdf = funcname(df000, zong_can=func_canshu['zong_can'],strat_time=func_canshu['strat_time'],
                         end_time=func_canshu['end_time'], cpu_nums=func_canshu['cpu_nums'])
        resdf['s_time'] = s0
        resdf['e_time'] = e0
        resdf.sort_values(by='最后收益',ascending=False,inplace=True)
        df_jieduan_zong = df_jieduan_zong.append(resdf,ignore_index=True)
    # df_jieduan_zong.set_index(keys=['s_time', "最后收益"], inplace=True)
    return df_jieduan_zong

