import time,datetime
import baostock as bs
import talib
import numpy as np
import pandas as pd
import asyncio
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
from yinzis import *

np.seterr(divide='ignore', invalid='ignore')

pd_display_rows = 10
pd_display_cols = 100
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行


class BAO_stock:

    def __init__(self):
        import baostock as bs
        lg = bs.login()

        # 登陆系统
        self.lg = lg

    # 获取证券股票的基本资料：除去了指数其他，只保留A股，和科创。
    def get_basic_code_info(self,kechuang=False):

        rs = bs.query_stock_basic()
        print('query_stock_basic respond  error_msg:' + rs.error_msg)
        # 打印结果集
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        # 结果集输出到csv文件
        # result.to_csv("D:/stock_basic.csv", encoding="gbk", index=False)
        result['only_code'] = result['code'].str.lower().str.split('.').str[1]
        result_stock = result[result['type'] == '1']
        if kechuang:
            pass
        else:
            result_stock = result_stock[(result_stock['only_code'].str[:2] != '30')]
        # result_stock[('st' not in  result_stock['code_name'].str.lower() )]
        # result_stock['code_name']=result_stock['code_name'].str.lower()
        # result_stock['st股'] = result_stock['code_name'].apply(lambda x: True if 'st' in str(x) else False )
        # print(result_stock[result_stock['st股'] == False])
        # result[((result['code'].str[:2] == 'sh')&(result['only_code'].str[:2] == '60'))|((result['code'].str[:2] == 'sz')&(result['only_code'].str[:2] == '00'))])
        result_stock.reset_index(drop=True, inplace=True)

        # 获取行业分类数据
        rs = bs.query_stock_industry()
        print('query_stock_industry error_code:' + rs.error_code)
        print('query_stock_industry respond  error_msg:' + rs.error_msg)

        # 打印结果集
        industry_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            industry_list.append(rs.get_row_data())
        result2 = pd.DataFrame(industry_list, columns=rs.fields)

        result = pd.merge(result_stock, result2[['code', 'industry', 'industryClassification']], how='left', on=['code'])
        # print(result.keys())
        # exit()

        return result
    def get_all_index_info(self):

        rs = bs.query_stock_basic()
        print('query_stock_basic respond  error_msg:' + rs.error_msg)
        # 打印结果集
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        # 结果集输出到csv文件
        # result.to_csv("D:/stock_basic.csv", encoding="gbk", index=False)
        # result['type'] = result['type'].str[:]
        # result['only_code'] = result['code'].str.lower().str.split('.').str[1]
        # result_stock = result[result['type'] == '2']

        result = result[result['type'] == '2']
        result.reset_index(drop=True, inplace=True)



        return result
    def get_hangye_info(self):
        # 获取行业分类数据
        rs = bs.query_stock_industry()
        print('query_stock_industry error_code:' + rs.error_code)
        print('query_stock_industry respond  error_msg:' + rs.error_msg)

        # 打印结果集
        industry_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            industry_list.append(rs.get_row_data())
        result2 = pd.DataFrame(industry_list, columns=rs.fields)
        return result2
    #### 获取交易日信息 ####
    def get_trading_time(self,s_time, e_time):
        rs = bs.query_trade_dates(start_date=s_time, end_date=e_time)
        # print('query_trade_dates respond error_code:' + rs.error_code)
        print('query_trade_dates respond  error_msg:' + rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.columns = ['date','is_trading_day']
        # print(result)
        return result

    def get_50_codes(self,date='2017-01-01'):
        # 获取上证50成分股
        rs = bs.query_sz50_stocks(date)
        print('query_sz50 error_code:' + rs.error_code)
        print('query_sz50  error_msg:' + rs.error_msg)

        # 打印结果集
        sz50_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            sz50_stocks.append(rs.get_row_data())
        result = pd.DataFrame(sz50_stocks, columns=rs.fields)
        return result

    def get_300_codes(self,date='2017-01-01'):
        # 获取沪深300成分股
        rs = bs.query_hs300_stocks(date)

        # 打印结果集
        hs300_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            hs300_stocks.append(rs.get_row_data())
        result = pd.DataFrame(hs300_stocks, columns=rs.fields)
        return result

    # 中证500成分股
    def get_500_codes(self,date='2017-01-01'):
        # 获取中证500成分股
        rs = bs.query_zz500_stocks(date)
        print('query_zz500 error_code:'+rs.error_code)
        print('query_zz500  error_msg:'+rs.error_msg)

        # 打印结果集
        zz500_stocks = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            zz500_stocks.append(rs.get_row_data())
        result = pd.DataFrame(zz500_stocks, columns=rs.fields)
        return result

        # 结果集输出到csv文件

    #指数数据
    def get_index(self,index_code="sh.000001",start_date='2017-01-01', end_date='2017-06-30', frequency="d"):

        #frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据
        # ，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线第月最后一个交易日才可以获取。
        # 获取指数(综合指数、规模指数、一级行业指数、二级行业指数、策略指数、成长指数、价值指数、主题指数)K线数据
        # 综合指数，例如：sh.000001 上证指数，sz.399106 深证综指 等；
        # 规模指数，例如：sh.000016 上证50，sh.000300 沪深300，sh.000905 中证500，sz.399001 深证成指等；
        # 一级行业指数，例如：sh.000037 上证医药，sz.399433 国证交运 等；
        # 二级行业指数，例如：sh.000952 300地产，sz.399951 300银行 等；
        # 策略指数，例如：sh.000050 50等权，sh.000982 500等权 等；
        # 成长指数，例如：sz.399376 小盘成长 等；
        # 价值指数，例如：sh.000029 180价值 等；
        # 主题指数，例如：sh.000015 红利指数，sh.000063 上证周期 等；

        # 详细指标参数，参见“历史行情指标参数”章节；“周月线”参数与“日线”参数不同。
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        rs = bs.query_history_k_data_plus(index_code,
                                          "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                          start_date=start_date, end_date=end_date, frequency=frequency)
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        # 打印结果集
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        # 结果集输出到csv文件
        # result.to_csv("D:\\history_Index_k_data.csv", index=False)
        return result

# 登陆系统

# 下载indexs_data 保存本地csv
def get_indexs_data():
    lg = bs.login()
    df = BAO_stock().get_all_index_info()
    df = df[df["status"] == "1"].copy()
    df.to_csv(r"data\indexs_data.csv", header=True, index=True)
    print("保存至：data\indexs_data.csv")
    return df
# 加载本地indexs_data csv
def load_local_indexs_data():
    try:
        df = pd.read_csv(r"data\indexs_data.csv",index_col=0)

    except Exception as e:
        print(e)
        df =pd.DataFrame()
    return df

def get_data(index_code:str,stime:str,etime:str,yinzi_name:str,yinzi_cs0 :int):
    df_index0 = BAO_stock().get_index(index_code=index_code, start_date=stime, end_date=etime, frequency="d")
    df_index0["date"] = pd.to_datetime(df_index0["date"])
    df_index0["open"] = df_index0["open"].astype(float)
    df_index0["high"] = df_index0["high"].astype(float)
    df_index0["low"] = df_index0["low"].astype(float)
    df_index0["close"] = df_index0["close"].astype(float)
    df_index0["preclose"] = df_index0["preclose"].astype(float)
    df_index0["volume"] = df_index0["volume"].astype(float)
    df_index0["pctChg"] = df_index0["pctChg"].astype(float)
    df_index0['weekday'] = pd.to_datetime(df_index0["date"], unit='s').apply(lambda x: float(x.to_pydatetime().weekday()))

    df_index0["yz"] = eval(yinzi_name)(df_index0,m=int(yinzi_cs0))
    
    return df_index0


def huice(df_A,df_B,n,pos_zq=20,sxf=0.001):
    n_day =pos_zq
    i_day = 0
    df_pos = pd.DataFrame()
    df_pos["candle_begin_time"] = df_A["date"]
    df_pos['weekday'] = pd.to_datetime(df_pos["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().weekday()))
    df_pos["signal"] = np.nan
    df_pos["pos"] = 0
    df_pos["open_price"] = np.nan
    df_pos["per_lr"] = 0

    for i in range(n, df_pos.shape[0]):
        if i_day == 0 and df_pos.iloc[i]["weekday"] == 0 and df_pos.iloc[i - 1]["pos"] == 0:
            if (df_A.iloc[i]["yz"] > 0) and (df_B.iloc[i]["yz"] > 0):
                i_day += 1
                if df_A.iloc[i]["yz"] > df_B.iloc[i]["yz"]:
                    df_pos.loc[i, "signal"] = 1
                    df_pos.loc[i, "pos"] = 1
                    df_pos.loc[i, "open_price"] = df_A.iloc[i]["open"] * (1 + sxf)
                    # print((df_A.iloc[i]["close"] / df_pos.iloc[i]["open_price"]) - 1)
                    df_pos.loc[i, "per_lr"] = (df_A.iloc[i]["close"] / df_pos.iloc[i]["open_price"]) - 1
                    # print(df_A.iloc[i]["close"]/df_pos.iloc[i]["open_price"]-1)
                elif df_A.iloc[i]["yz"] <= df_B.iloc[i]["yz"]:
                    df_pos.loc[i, "signal"] = 1
                    df_pos.loc[i, "pos"] = 2
                    df_pos.loc[i, "open_price"] = df_B.iloc[i]["open"] * (1 + sxf)
                    df_pos.loc[i, "per_lr"] = (df_B.iloc[i]["close"] / df_pos.iloc[i]["open_price"]) - 1
                    # print(df_B.iloc[i]["close"]/df_pos.iloc[i]["open_price"]-1)
                # print("开仓",df_pos.iloc[i]["per_lr"])
            else:
                i_day = 0
                df_pos.loc[i, "pos"] = 0

        # 持仓
        elif i_day < n_day - 1 and df_pos.iloc[i - 1]["pos"] != 0:
            i_day += 1
            df_pos.loc[i, "pos"] = df_pos.iloc[i - 1]["pos"]
            if df_pos.iloc[i - 1]["pos"] == 1:
                df_pos.loc[i, "open_price"] = df_pos.iloc[i - 1]["open_price"]
                df_pos.loc[i, "per_lr"] = (df_A.iloc[i]["close"] / df_A.iloc[i - 1]["close"]) - 1
            elif df_pos.iloc[i - 1]["pos"] == 2:
                df_pos.loc[i, "open_price"] = df_pos.iloc[i - 1]["open_price"]
                df_pos.loc[i, "per_lr"] = (df_B.iloc[i]["close"] / df_B.iloc[i - 1]["close"]) - 1
            else:
                df_pos.loc[i, "per_lr"] = 0
        # 卖出
        elif i_day >= n_day - 1 and df_pos.iloc[i - 1]["pos"] != 0:

            i_day = 0
            df_pos.loc[i, "pos"] = 0
            df_pos.loc[i, "signal"] = 0

            if df_pos.iloc[i - 1]["pos"] == 1:
                df_pos.loc[i, "open_price"] = df_A.iloc[i]["close"] * (1 - sxf)
                df_pos.loc[i, "per_lr"] = (df_pos.iloc[i]["open_price"] / df_A.iloc[i - 1]["close"]) - 1
            elif df_pos.iloc[i - 1]["pos"] == 2:
                df_pos.loc[i, "open_price"] = df_B.iloc[i]["close"] * (1 - sxf)
                df_pos.loc[i, "per_lr"] = (df_pos.iloc[i]["open_price"] / df_B.iloc[i - 1]["close"]) - 1
            else:
                df_pos.loc[i, "per_lr"] = 0

        else:
            i_day = 0
            df_pos.loc[i, "pos"] = 0

    df_pos["hc_zj"] = (df_pos["per_lr"] + 1).cumprod()
    df_pos["A_jz"] = (df_A["close"].pct_change() + 1).cumprod()
    df_pos["B_jz"] = (df_B["close"].pct_change() + 1).cumprod()
    return df_pos

def zhubi_fx2(df, tj_type=1):
    '''
    输入正确的df，pos2是处理好的持仓状态的时间分类，起始对应open_price必须正确，假如少了或者不对，整个分析不对
    :param df: df要有'candle_begin_time',  'signal', 'pos','pos',"open_price", 'per_lr'
    'signal'：非连续
    'pos'：连续开始到结束都应包括，
    'pos2':持仓时间
    "open_price"：'pos'的开仓价 ,结束评价仓价格,此处必须正确！
     'per_lr'：cumsum序列
     tj_type：1==”复利净值“   2 ==”点差计算“
    :return:
    '''

    for i in ['candle_begin_time', 'signal', 'pos', 'pos2', "open_price", 'per_lr']:
        if i not in df.keys():
            print("参数不够！", i)
            return [], []

    # 每一笔分析
    df_zbfx = pd.DataFrame(df[['candle_begin_time', 'signal', 'pos', 'pos2', "open_price", 'per_lr']]).copy()

    decrib_per_trade = pd.DataFrame()
    for i, (k, v) in enumerate(df_zbfx.groupby("pos2")):
        if v.iloc[0]["pos"] == 0: continue
        v = pd.DataFrame(v)
        # print(v.iloc[:5])
        # print(v.iloc[-5:])
        pos_fx = 1 if v.iloc[0]["pos"] > 0 else -1
        decrib_per_trade.loc[i, "开始时间"] = (v.iloc[0]["candle_begin_time"])
        decrib_per_trade.loc[i, "结束时间"] = (v.iloc[-1]["candle_begin_time"])
        decrib_per_trade.loc[i, "持续时间"] = (v.iloc[-1]["candle_begin_time"] - v.iloc[0]["candle_begin_time"])
        decrib_per_trade.loc[i, "持仓方向"] = (v.iloc[0]["pos"])
        decrib_per_trade.loc[i, "开始持仓价"] = float(v["open_price"].tolist()[0])
        decrib_per_trade.loc[i, "结束平仓价"] = float(v["open_price"].tolist()[-1])
        decrib_per_trade.loc[i, "持仓点差收益"] = pos_fx * (float(v["open_price"].tolist()[-1]) - float(v["open_price"].tolist()[0]))
        decrib_per_trade.loc[i, "持仓净值收益"] = pos_fx * (float(v["open_price"].tolist()[-1]) / float(v["open_price"].tolist()[0]) - 1)
        decrib_per_trade.loc[i, "最大收益"] = (pos_fx * (v["per_lr"] - v.iloc[0]["per_lr"])).max()
        decrib_per_trade.loc[i, "最小收益"] = (pos_fx * (v["per_lr"] - v.iloc[0]["per_lr"])).min()
        decrib_per_trade.loc[i, "收益波幅"] = decrib_per_trade.at[i, "最大收益"] - decrib_per_trade.at[i, "最小收益"]

    decrib_per_trade = decrib_per_trade.reset_index(drop=True)
    decrib_per_trade["单利持仓净值"] = decrib_per_trade["持仓净值收益"].cumsum()
    decrib_per_trade["复利持仓净值"] = (decrib_per_trade["持仓净值收益"] + 1).cumprod()
    decrib_per_trade["单利持仓总点差"] = decrib_per_trade["持仓点差收益"].cumsum()
    # print(decrib_per_trade)
    decrib_per_trade2 = pd.DataFrame()
    decrib_per_trade2.loc["单利持仓总点差", "tj"] = decrib_per_trade.iloc[-1]["单利持仓总点差"]
    decrib_per_trade2.loc["单利持仓净值", "tj"] = decrib_per_trade.iloc[-1]["单利持仓净值"]
    decrib_per_trade2.loc["复利持仓净值", "tj"] = decrib_per_trade.iloc[-1]["复利持仓净值"]

    if tj_type == 1:  # 复利持仓净值

        decrib_per_trade["累计最大净值"] = (decrib_per_trade["复利持仓净值"]).cummax()
        decrib_per_trade["回撤净值"] = (decrib_per_trade["累计最大净值"] - decrib_per_trade["复利持仓净值"])
        max_huice_index_s = decrib_per_trade[decrib_per_trade["回撤净值"] == decrib_per_trade["回撤净值"].max()].index[0]
        index_mx = decrib_per_trade[decrib_per_trade["累计最大净值"] == decrib_per_trade.iloc[max_huice_index_s]["累计最大净值"]].index
        stime = decrib_per_trade.iloc[int(sorted(index_mx)[1])]["开始时间"]
        etime = decrib_per_trade.iloc[max_huice_index_s]["结束时间"]

        decrib_per_trade2.loc["最大回撤大小", "tj"] = decrib_per_trade["回撤净值"].max()
        decrib_per_trade2.loc["最大回撤率%", "tj"] = (decrib_per_trade["回撤净值"] / decrib_per_trade["累计最大净值"]).max() * 100
        decrib_per_trade2.loc["最大回撤_持仓开始时间", "tj"] = stime
        decrib_per_trade2.loc["最大回撤_持仓结束时间", "tj"] = etime
        decrib_per_trade2.loc["胜率%", "tj"] = decrib_per_trade[decrib_per_trade["持仓净值收益"] > 0].shape[0] * 100 / decrib_per_trade.shape[0]
        decrib_per_trade2.loc["交易次数", "tj"] = decrib_per_trade.shape[0]
        decrib_per_trade2.loc["正收益平均大小%", "tj"] = 100 * decrib_per_trade[decrib_per_trade["持仓净值收益"] > 0]["持仓净值收益"].mean()
        decrib_per_trade2.loc["负收益平均大小%", "tj"] = 100 * decrib_per_trade[decrib_per_trade["持仓净值收益"] < 0]["持仓净值收益"].mean()
        decrib_per_trade2.loc["负正收益比", "tj"] = -1 * decrib_per_trade2.loc["正收益平均大小%", "tj"] / decrib_per_trade2.loc["负收益平均大小%", "tj"]
        decrib_per_trade2.loc["正收益稳定比率%", "tj"] = 100 * \
                                                  decrib_per_trade[decrib_per_trade["持仓净值收益"] > 0]["持仓净值收益"].mean() / decrib_per_trade[
                                                      "持仓净值收益"].std()

        decrib_per_trade2.loc["平均持续时间", "tj"] = str(decrib_per_trade["持续时间"].mean())
        max_ks = decrib_per_trade.loc[decrib_per_trade["持仓净值收益"] == decrib_per_trade["持仓净值收益"].min()].index[0]
        decrib_per_trade2.loc["单笔最大亏损%", "tj"] = 100 * decrib_per_trade.iloc[max_ks]["持仓净值收益"]
        decrib_per_trade2.loc["单笔最大回撤率%", "tj"] = abs(decrib_per_trade["持仓净值收益"].min() * 100 / decrib_per_trade.iloc[-1]["复利持仓净值"])
        decrib_per_trade2.loc["单笔最大亏损开始时间", "tj"] = decrib_per_trade.iloc[max_ks]["开始时间"]
        decrib_per_trade2.loc["单笔最大亏损结束时间", "tj"] = decrib_per_trade.iloc[max_ks]["结束时间"]
    else:

        decrib_per_trade["累计最大净值"] = (decrib_per_trade["单利持仓总点差"]).cummax()
        decrib_per_trade["回撤净值"] = (decrib_per_trade["累计最大净值"] - decrib_per_trade["单利持仓总点差"])
        max_huice_index_s = decrib_per_trade[decrib_per_trade["回撤净值"] == decrib_per_trade["回撤净值"].max()].index[0]
        index_mx = decrib_per_trade[decrib_per_trade["累计最大净值"] == decrib_per_trade.iloc[max_huice_index_s]["累计最大净值"]].index
        stime = decrib_per_trade.iloc[int(sorted(index_mx)[1])]["开始时间"]
        etime = decrib_per_trade.iloc[max_huice_index_s]["结束时间"]

        decrib_per_trade2.loc["最大回撤大小", "tj"] = decrib_per_trade["回撤净值"].max()
        decrib_per_trade2.loc["最大回撤率%", "tj"] = (decrib_per_trade["回撤净值"] * 100 / decrib_per_trade["累计最大净值"]).max()
        decrib_per_trade2.loc["最大回撤_持仓开始时间", "tj"] = stime
        decrib_per_trade2.loc["最大回撤_持仓结束时间", "tj"] = etime
        decrib_per_trade2.loc["胜率%", "tj"] = decrib_per_trade[decrib_per_trade["持仓净值收益"] > 0].shape[0] * 100 / decrib_per_trade.shape[0]
        decrib_per_trade2.loc["交易次数", "tj"] = decrib_per_trade.shape[0]
        decrib_per_trade2.loc["正收益平均大小", "tj"] = decrib_per_trade[decrib_per_trade["持仓点差收益"] > 0]["持仓点差收益"].mean()
        decrib_per_trade2.loc["负收益平均大小", "tj"] = decrib_per_trade[decrib_per_trade["持仓点差收益"] < 0]["持仓点差收益"].mean()
        decrib_per_trade2.loc["正收益大小", "tj"] = decrib_per_trade[decrib_per_trade["持仓点差收益"] > 0]["持仓点差收益"].sum()
        decrib_per_trade2.loc["负收益大小", "tj"] = decrib_per_trade[decrib_per_trade["持仓点差收益"] < 0]["持仓点差收益"].sum()
        decrib_per_trade2.loc["正负平均收益比", "tj"] = -1 * decrib_per_trade2.loc["正收益平均大小", "tj"] / decrib_per_trade2.loc["负收益平均大小", "tj"]
        decrib_per_trade2.loc["正收益稳定率%", "tj"] = 100 * \
                                                 decrib_per_trade[decrib_per_trade["持仓点差收益"] > 0]["持仓点差收益"].mean() / decrib_per_trade[
                                                     "持仓点差收益"].std()

        decrib_per_trade2.loc["平均持续时间", "tj"] = str(decrib_per_trade["持续时间"].mean())
        max_ks = decrib_per_trade.loc[decrib_per_trade["持仓点差收益"] == decrib_per_trade["持仓点差收益"].min()].index[0]
        decrib_per_trade2.loc["单笔最大亏损", "tj"] = 100 * decrib_per_trade.iloc[max_ks]["持仓点差收益"]
        decrib_per_trade2.loc["单笔最大回撤率%", "tj"] = abs(decrib_per_trade["持仓点差收益"].min() * 100 / decrib_per_trade.iloc[-1]["单利持仓总点差"])
        decrib_per_trade2.loc["单笔最大亏损开始时间", "tj"] = decrib_per_trade.iloc[max_ks]["开始时间"]
        decrib_per_trade2.loc["单笔最大亏损结束时间", "tj"] = decrib_per_trade.iloc[max_ks]["结束时间"]


    decrib_per_trade[ "持续时间"] = decrib_per_trade[ "持续时间"].apply(lambda x:str(x).split(":")[0])
    decrib_per_trade[ "开始时间"] = decrib_per_trade[ "开始时间"].apply(lambda x:x.strftime("%Y-%m-%d"))
    decrib_per_trade[ "结束时间"] = decrib_per_trade[ "结束时间"].apply(lambda x:x.strftime("%Y-%m-%d"))

    decrib_per_trade2 = decrib_per_trade2.reset_index(drop=False)
    decrib_per_trade = decrib_per_trade.reset_index(drop=False)
    decrib_per_trade.loc[decrib_per_trade["持仓方向"] == 1 , "持仓方向"] = "A"
    decrib_per_trade.loc[decrib_per_trade["持仓方向"] == 2 , "持仓方向"] = "B"

    return decrib_per_trade, decrib_per_trade2


if __name__ == '__main__':
    # 行业轮动
    if True == 1:

        df = load_local_indexs_data()

        yinzi_canshu_m = 20
        n_day = 20
        sxf = 0.001
        yinzi_name = "yinzi01"
        A ="sh.000300"
        B ="sz.399006"
        info = df[df["code"].isin([A,B])]
        stime = str(max(info["ipoDate"]))
        etime = datetime.datetime.now().strftime("%Y-%m-%d")

        pathname = r"沪深300etf-创业板指etf-轮动持仓明细.html"
        # print(info)
        # print(stime)


        df_A = get_data(index_code=A, stime=stime, etime=etime,yinzi_name=yinzi_name,yinzi_cs0=yinzi_canshu_m)
        df_B = get_data(index_code=B, stime=stime, etime=etime,yinzi_name=yinzi_name,yinzi_cs0=yinzi_canshu_m)

        print(df.tail())
        df_pos = huice(df_A, df_B,yinzi_canshu_m,pos_zq=n_day,sxf=sxf)
        df_pos["per_lr"] = (df_pos["per_lr"] + 1).cumprod()
        df_pos["da_jz"] = (df_A["close"].pct_change() + 1).cumprod()
        df_pos["xiao_jz"] = (df_B["close"].pct_change() + 1).cumprod()

        df_pos2 = df_pos[['candle_begin_time', 'signal', 'pos', "open_price", 'per_lr']].copy()
        df_pos2["pos2"] = df_pos2["pos"].shift(1)
        df_pos2.loc[df_pos2["signal"] == 0, "pos"] = df_pos2.loc[df_pos2["signal"] == 0, "pos2"]

        df_pos2["pos2"] = np.nan
        df_pos2.loc[(df_pos2["pos"] != df_pos2["pos"].shift()), "pos2"] = \
            df_pos2.loc[(df_pos2["pos"] != df_pos2["pos"].shift()), "candle_begin_time"]
        df_pos2["pos2"].fillna(method="ffill", inplace=True)
        decrib_per_trade, decrib_per_trade2 = zhubi_fx2(df=df_pos2)

        df_pos["candle_begin_time"] = pd.to_datetime(df_pos["candle_begin_time"]).\
            apply(lambda x:x.to_pydatetime().strftime("%Y_n_%m_y_%d_r"))

        print(df_pos.tail())
        print(decrib_per_trade, decrib_per_trade2)
