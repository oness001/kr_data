import time,datetime
import talib
# import numpy as np
import pandas as pd
from KRData import CNData ,HKData

# np.seterr(divide='ignore', invalid='ignore')
#
# pd_display_rows = 10
# pd_display_cols = 100
# pd_display_width = 1000
# pd.set_option('display.max_rows', 10000)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 100)
# pd.set_option('display.max_colwidth', 120)
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)
# pd.set_option('expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.4f' % x)


def get_h_stock(s_time="20200125",e_time="20210126"):
    # a = time.time_ns()

    hks = HKData.HKStock(data_type='df', adjust='qfq')

    df0 = hks[s_time:e_time]
    # print(df0.tail())
    # print(df0.shape)
    # print((time.time_ns() - a) / 1000000000, "s")
    df0 = df0.reset_index(drop=True)
    return df0


def get_hk_stock_infos(path=r"Table.csv"):
    df_hk_infos = pd.read_csv(path, encoding="GBK")
    # print(df_hk_infos.keys())
    # ['代码', '名称    ', '恒生行业(一级)']
    df_hk_infos = df_hk_infos[["code"    ,          "name" ,       "industry"]]
    # df_hk_infos.columns = ['code', '名称', '行业名']
    return df_hk_infos


def MA_Sys(df):
    v = df.copy()
    v["MA120"] = talib.MA(v["close"], 120)
    con120 = v["close"] > v["MA120"]
    con120_2 = v["close"] < v["MA120"]
    v.loc[v["MA120"].isnull(), "是否大于120均线"] = 0
    v.loc[con120, "是否大于120均线"] = 1
    v.loc[con120_2, "是否大于120均线"] = -1
    con120_3 = v["是否大于120均线"] != v["是否大于120均线"].shift()
    v.loc[con120_3, "大于120均线开始时间"] = v.loc[con120_3, "date"]
    v["大于120均线开始时间"].fillna(method="ffill", inplace=True)
    g120 = v.groupby(["大于120均线开始时间"]).get_group(v.iloc[-1]["大于120均线开始时间"])
    v.loc[v.index[-1], "是否大于120均线"] = g120["是否大于120均线"].sum()
    v["MA60"] = talib.MA(v["close"], 60)
    con60 = v["close"] > v["MA60"]
    con60_2 = v["close"] < v["MA60"]
    v.loc[v["MA60"].isnull(), "是否大于60均线"] = 0
    v.loc[con60, "是否大于60均线"] = 1
    v.loc[con60_2, "是否大于60均线"] = -1
    con60_3 = v["是否大于60均线"] != v["是否大于60均线"].shift()
    v.loc[con60_3, "大于60均线开始时间"] = v.loc[con60_3, "date"]
    v["大于60均线开始时间"].fillna(method="ffill", inplace=True)

    g60 = v.groupby(["大于60均线开始时间"]).get_group(v.iloc[-1]["大于60均线开始时间"])
    v.loc[v.index[-1], "是否大于60均线"] = g60["是否大于60均线"].sum()

    v["MA20"] = talib.MA(v["close"], 20)
    con20 = v["close"] > v["MA20"]
    con20_2 = v["close"] < v["MA20"]
    v.loc[v["MA20"].isnull(), "是否大于20均线"] = 0
    v.loc[con20, "是否大于20均线"] = 1
    v.loc[con20_2, "是否大于20均线"] = -1
    con20_3 = v["是否大于20均线"] != v["是否大于20均线"].shift()
    v.loc[con20_3, "大于20均线开始时间"] = v.loc[con20_3, "date"]
    v["大于20均线开始时间"].fillna(method="ffill", inplace=True)

    g20 = v.groupby(["大于20均线开始时间"]).get_group(v.iloc[-1]["大于20均线开始时间"])
    v.loc[v.index[-1], "是否大于20均线"] = g20["是否大于20均线"].sum()

    return v

def update_h_stocks():
    df_infos = get_hk_stock_infos(path=r"data/hk_info.csv")

    # print(df_infos.tail())

    s_time = (datetime.datetime.now()-datetime.timedelta(days=360)).strftime("%Y%m%d")
    e_time = datetime.datetime.now().strftime("%Y%m%d")

    df_hk = get_h_stock(s_time=s_time, e_time=e_time)
    # print(df_hk.tail())

    # df_hk = pd.read_csv(r"200_H股_kline.csv",encoding="utf-8")
    df_hk = df_hk[['code', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
    df_hk.columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
    df_hk.reset_index(drop=True, inplace=True)
    # print(df_hk_zong.keys())
    # print(df_hk_zong.tail(300))
    #
    # exit()
    df_hk_zong = pd.DataFrame()
    df_hk.fillna(0, inplace=True)
    for k, v in df_hk.groupby("code"):
        if v.iloc[-1]["close"] <= 0.1:
            continue
        # if int(k) > 100:break
        df0 = MA_Sys(df=v)
        # print("code:", k)
        df0["date0"] = pd.to_datetime(df0["date"])
        df0 = df0[df0["date0"] >= pd.to_datetime(e_time) - datetime.timedelta(days=30)].reset_index()
        df_hk_zong = df_hk_zong.append(df0, ignore_index=True)
        # print(df0)
        # exit()
    df_hk_zong['code'] = df_hk_zong['code'].astype(str)
    df_infos['code'] = df_infos['code'].astype(str)
    df_infos["code"] = df_infos["code"].str.zfill(5)
    # print(df_infos.tail(20))
    df_hk_zong = df_hk_zong.merge(df_infos, how="left", on="code")

    df_hk_zong = df_hk_zong[['date', 'code', 'name', 'industry', 'MA120', 'MA20', 'MA60',
                             'close', '是否大于20均线', '是否大于60均线', '是否大于120均线']]
    df_hk_zong = df_hk_zong.round(3)
    df_hk_zong.to_csv(r"data/H股_show_table.csv", mode="w")

    # print(df_hk_zong.keys())

    print(df_hk_zong.tail(10))
    print(df_hk_zong.shape)
    return "ok"



#
if __name__ =='__main__':
    import pandas as pd

    hks = CNData.CNStock(data_type='df', adjust='qfq')
    # print(hks.all_codes)
    # exit()

    df0 = hks["20050101":"20210301"]
    df0 = pd.DataFrame(df0)
    print(df0.head(100))

    print(df0.tail(100))

    df0.to_pickle("2005-2021-ASTOCKS.pkl")
#     update_h_stocks()
#     if True == 0:
#         # a =time.time_ns()
#         # cns = CNData.CNStock(data_type='df', adjust='qfq')
#         # df = cns["20200126":"20210126"]
#         # print(df.tail())
#         # print(df.shape)
#         # df.to_csv("200_A股_kline.csv",mode="w")
#         # print((time.time_ns()-a)/1000000000,"s")
#         # exit()
#         df_infos = get_hk_stock_infos(path=r"data/hk_info.csv")
#         # df_infos['code'] = df_infos['code'].astype(str)
#         #
#         # df_infos["code"] = df_infos["code"].str.zfill(6)
#         #
#         print(df_infos.tail())
#         # exit()
#         # print(df_infos)
#         # exit()
#         s_time = "20200128"
#         e_time = "20210202"
#         df_hk = get_h_stock(s_time=s_time,e_time=e_time)
#         print(df_hk)
#         # df_hk = pd.read_csv(r"200_H股_kline.csv",encoding="utf-8")
#         df_hk = df_hk[['code', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
#         df_hk.columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
#         df_hk.reset_index(drop=True,inplace=True)
#         # print(df_hk_zong.keys())
#         # print(df_hk_zong.tail(300))
#         #
#         # exit()
#         df_hk_zong = pd.DataFrame()
#         df_hk.fillna(0,inplace=True)
#         for k , v in df_hk.groupby("code"):
#             if v.iloc[-1]["close"] <= 0.1 :
#                 continue
#             # if int(k) > 100:break
#             df0 = MA_Sys(df=v)
#             print("code:",k)
#             df0["date0"] = pd.to_datetime(df0["date"])
#             df0 = df0[df0["date0"] >= pd.to_datetime(e_time)-datetime.timedelta(days=30)].reset_index()
#             df_hk_zong = df_hk_zong.append(df0,ignore_index=True)
#             # print(df0)
#             # exit()
#         df_hk_zong['code'] = df_hk_zong['code'].astype(str)
#         df_infos['code'] = df_infos['code'].astype(str)
#         df_infos["code"] = df_infos["code"].str.zfill(5)
#         # print(df_infos.tail(20))
#         df_hk_zong = df_hk_zong.merge(df_infos,how="left",on="code")
#
#         df_hk_zong = df_hk_zong[[ 'date','code', 'name', 'industry','MA120', 'MA20', 'MA60',
#                                   'close','是否大于20均线', '是否大于60均线', '是否大于120均线']]
#         df_hk_zong = df_hk_zong.round(3)
#         df_hk_zong.to_csv("H股_show_table.csv",mode ="w")
#
#         print(df_hk_zong.tail(10))
#         print(df_hk_zong.keys())
#
#         print(df_hk_zong.shape)
#         exit()
#         # df0.to_csv()
#
#     if True == 0:
#
#         df = pd.read_csv(r"F:\vnpy_my_gitee\company\A股票_company\A股行业强度筛选\H_stocks_info\H股_show_table.csv")
#         df = df[[ 'date','code', '名称', '行业名','MA120', 'MA20', 'MA60', 'close','是否大于20均线', '是否大于60均线', '是否大于120均线',]]
#         df.to_csv(r"F:\vnpy_my_gitee\company\A股票_company\A股行业强度筛选\H_stocks_info\H股_show_table.csv")


