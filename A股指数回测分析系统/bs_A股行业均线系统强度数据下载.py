import time,datetime
import baostock as bs
import talib
# import numpy as np

# import pandas as pd
import asyncio


from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
# np.seterr(divide='ignore', invalid='ignore')
#
#
# pd.set_option('display.max_rows', 10000)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 100)
# pd.set_option('display.max_colwidth', 120)
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)
# pd.set_option('expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.4f' % x)

# #### 获取交易日信息 ####
# def get_trading_time(s_time, e_time):
#     rs = bs.query_trade_dates(start_date=s_time, end_date=e_time)
#     # print('query_trade_dates respond error_code:' + rs.error_code)
#     print('query_trade_dates respond  error_msg:' + rs.error_msg)
#
#     #### 打印结果集 ####
#     data_list = []
#     while (rs.error_code == '0') & rs.next():
#         # 获取一条记录，将记录合并在一起
#         data_list.append(rs.get_row_data())
#     result = pd.DataFrame(data_list, columns=rs.fields)
#     result.columns = ['date','is_trading_day']
#     # print(result)
#     return result

# 获取证券股票的基本资料：除去了指数其他，只保留A股，和科创。

def cal_zhangting_with_st(df):


    # 计算涨停价格
    df['开盘涨停(买不进)'] = 0
    df[ '涨停价'] = df['preclose'] * 1.1

    # 针对st进行修改
    df.loc[df['是否是st股']==1, '涨停价'] = df['preclose'] * 1.05

    # 四舍五入
    df['涨停价'] = df['涨停价'].apply(lambda x: float(Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))


    df.loc[(df['open'] >= df['涨停价']), '开盘涨停(买不进)'] = 1

    return df


def get_basic_code_info(kechuang = False):

    rs = bs.query_stock_basic()
    print('query_stock_basic respond  error_msg:'+rs.error_msg)
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    # 结果集输出到csv文件
    # result.to_csv("D:/stock_basic.csv", encoding="gbk", index=False)
    result['only_code'] = result['code'].str.lower().str.split('.').str[1]
    result_stock = result[result['type']=='1']
    if kechuang:
        pass
    else:
        result_stock = result_stock[(result_stock['only_code'].str[:2] != '30')]
    # result_stock[('st' not in  result_stock['code_name'].str.lower() )]
    # result_stock['code_name']=result_stock['code_name'].str.lower()
    # result_stock['st股'] = result_stock['code_name'].apply(lambda x: True if 'st' in str(x) else False )
    # print(result_stock[result_stock['st股'] == False])
    # result[((result['code'].str[:2] == 'sh')&(result['only_code'].str[:2] == '60'))|((result['code'].str[:2] == 'sz')&(result['only_code'].str[:2] == '00'))])
    result_stock.reset_index(drop=True,inplace=True)

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

    result = pd.merge(result_stock,result2[['code','industry','industryClassification']],how='left',on=['code'])
    # print(result.keys())
    # exit()
    return result

# def get_300_codes():
#     # 获取沪深300成分股
#     rs = bs.query_hs300_stocks()
#
#     # 打印结果集
#     hs300_stocks = []
#     while (rs.error_code == '0') & rs.next():
#         # 获取一条记录，将记录合并在一起
#         hs300_stocks.append(rs.get_row_data())
#     result = pd.DataFrame(hs300_stocks, columns=rs.fields)
#     return result
#
# def get_50_codes():
#     # 获取上证50成分股
#     rs = bs.query_sz50_stocks()
#     print('query_sz50 error_code:' + rs.error_code)
#     print('query_sz50  error_msg:' + rs.error_msg)
#
#     # 打印结果集
#     sz50_stocks = []
#     while (rs.error_code == '0') & rs.next():
#         # 获取一条记录，将记录合并在一起
#         sz50_stocks.append(rs.get_row_data())
#     result = pd.DataFrame(sz50_stocks, columns=rs.fields)
#     return result

#
# async def get_a_caiwu0(kind,code,y,q):
#     # 查询季频估值指标盈利能力
#     print(kind,code,y,q)
#     result0 = eval('bs.' + kind)(code=code, year=y, quarter=q)
#     res0 = result0.get_data()
#     return res0
#
# async def caiwu_data(codes,s_time,e_time,get_infos = {"query_profit_data":True,"query_operation_data":True,
#                                                 "query_growth_data":False,"query_balance_data":False,
#                                                 "query_cash_flow_data":False,"query_dupont_data":True,
#                                                 "query_performance_express_report":False,"query_forecast_report":False }):
#     # 登陆系统
#     lg = bs.login()
#     # 显示登陆返回信息
#     print('login respond error_code:' + lg.error_code)
#     print('login respond  error_msg:' + lg.error_msg)
#
#     #生成执行任务
#     tasks = []
#     time_list = []
#
#     for y in range(s_time,e_time+1,1):
#         for q in range(1,5,1):
#             time_list.append([y,q])
#
#     for c0 in codes:
#         for k, v in get_infos.items():
#             if v:
#                 for y,q in time_list:
#                     task_1 = asyncio.create_task(get_a_caiwu0(kind=k, code=c0, y=y, q=q))
#                     tasks.append(task_1)
#
#     result_all = await asyncio.gather(*tasks, return_exceptions=True)
#     # print(len(res)) # 10.主任务输出res,协程任务结束，事件循环结束
#
#
#     # cols = {
#     #     "code"     :  "code",
#     #     "pubDate"	:  "发布财报日",
#     #     "statDate" :  "统计截至日",
#     #     "roeAvg"	 :  "净资产收益率(平均)(%)",
#     #     "npMargin":	   "销售净利率(%)",
#     #     "gpMargin":	   "销售毛利率(%)",
#     #     "netProfit"	:   "净利润(元)",
#     #     "epsTTM"	 :  "每股收益",
#     #     "MBRevenue"	:   "主营营业收入(元)",
#     #     "totalShare" :  "总股本",
#     #     "liqaShare"	:   "流通股本",
#     #     "NRTurnRatio" :"应收账款周转率(次)",
#     #     "NRTurnDays": "应收账款周转天数(天)",
#     #     "INVTurnRatio": "存货周转率(次)",
#     #     "INVTurnDays": "存货周转天数(天)",
#     #     "CATurnRatio": "流动资产周转率(次)",
#     #     "AssetTurnRatio": "总资产周转率",
#     #     "YOYEquity": "净资产同比增长率",
#     #     "YOYAsset": "总资产同比增长率",
#     #     "YOYNI": "净利润同比增长率",
#     #     "YOYEPSBasic": "基本每股收益同比增长率",
#     #     "YOYPNI": "归属母公司股东净利润同比增长率",
#     #     "CAToAsset": "流动资产除以总资产",
#     #     "NCAToAsset": "非流动资产除以总资产",
#     #     "tangibleAssetToAsset": "有形资产除以总资产",
#     #     "ebitToInterest": "已获利息倍数",
#     #     "CFOToOR": "经营活动产生的现金流量净额除以营业收入",
#     #     "CFOToNP": "经营性现金净流量除以净利润",
#     #     "CFOToGr": "经营性现金净流量除以营业总收入",
#     #     "dupontROE": "净资产收益率",
#     #     "dupontAssetStoEquity": "权益乘数",
#     #     "dupontAssetTurn": "总资产周转率",
#     #     "dupontPnitoni": "归属母公司股东的净利润/净利润",
#     #     "dupontNitogr": "净利润/营业总收入",
#     #     "dupontTaxBurden": "净利润/利润总额",
#     #     "dupontIntburden": "利润总额/息税前利润",
#     #     "dupontEbittogr": "息税前利润/营业总收入",
#     #     #
#     # }
#     # replace_name = {i: cols.get(i) for i in result_all.keys()}
#     # result_all.rename(columns=replace_name,inplace=True)
#
#
#     # 打印输出
#     # 参数名称	   参数描述	              算法说明
#     # {"code"	:       "证券代码"
#     # "pubDate"	 :  "发布财报日"
#     # "statDate" :  "统计截至日"                财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30
#     # "roeAvg"	 :  "净资产收益率(平均)(%)"	  归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
#     # "npMargin":	   "销售净利率(%)"	          净利润/营业收入*100%
#     # "gpMargin":	   "销售毛利率(%)"	          毛利/营业收入*100%=(营业收入-营业成本)/营业收入*100%
#     # "netProfit"	:   "净利润(元)"
#     # "epsTTM"	 :  "每股收益"	               归属母公司股东的净利润TTM/最新总股本
#     # "MBRevenue"	:   "主营营业收入(元)"
#     # "totalShare" :  "总股本"
#     # "liqaShare"	:   "流通股本"
#     # "NRTurnRatio"	:"应收账款周转率(次)"	            营业收入/[(期初应收票据及应收账款净额+期末应收票据及应收账款净额)/2]
#     # "NRTurnDays"	:"应收账款周转天数(天)"	            季报天数/应收账款周转率(一季报：90天，中报：180天，三季报：270天，年报：360天)
#     # "INVTurnRatio"	:"存货周转率(次)"	            营业成本/[(期初存货净额+期末存货净额)/2]
#     # "INVTurnDays"	:"存货周转天数(天)"	            季报天数/存货周转率(一季报：90天，中报：180天，三季报：270天，年报：360天)
#     # "CATurnRatio"	:"流动资产周转率(次)"	            营业总收入/[(期初流动资产+期末流动资产)/2]
#     # "AssetTurnRatio"	:"总资产周转率"
#     # "YOYEquity":	"净资产同比增长率"           # (本期净资产-上年同期净资产)/上年同期净资产的绝对值*100%
#     # "YOYAsset":	"总资产同比增长率"  # (本期总资产-上年同期总资产)/上年同期总资产的绝对值*100%
#     # "YOYNI":	"净利润同比增长率"  # (本期净利润-上年同期净利润)/上年同期净利润的绝对值*100%
#     # "YOYEPSBasic":	"基本每股收益同比增长率"   # (本期基本每股收益-上年同期基本每股收益)/上年同期基本每股收益的绝对值*100%
#     # "YOYPNI":	"归属母公司股东净利润同比增长率"   # (本期归属母公司股东净利润-上年同期归属母公司股东净利润)/上年同期归属母公司股东净利润的绝对值*100%
#     # "CAToAsset":	"流动资产除以总资产"
#     # "NCAToAsset":	"非流动资产除以总资产"
#     # "tangibleAssetToAsset":	"有形资产除以总资产"
#     # "ebitToInterest":	"已获利息倍数"    # 息税前利润/利息费用
#     # "CFOToOR":	"经营活动产生的现金流量净额除以营业收入"
#     # "CFOToNP":	"经营性现金净流量除以净利润"
#     # "CFOToGr":	"经营性现金净流量除以营业总收入"
#     # "dupontROE":	"净资产收益率"    # 归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
#     # "dupontAssetStoEquity":	"权益乘数"  # ，反映企业财务杠杆效应强弱和财务风险	平均总资产/平均归属于母公司的股东权益
#     # "dupontAssetTurn":	"总资产周转率"    # ，反映企业资产管理效率的指标	营业总收入/[(期初资产总额+期末资产总额)/2]
#     # "dupontPnitoni":	"归属母公司股东的净利润/净利润"   # ，反映母公司控股子公司百分比。如果企业追加投资，扩大持股比例，则本指标会增加。
#     # "dupontNitogr":	"净利润/营业总收入"     # ，反映企业销售获利率
#     # "dupontTaxBurden":	"净利润/利润总额"  # ，反映企业税负水平，该比值高则税负较低。净利润/利润总额=1-所得税/利润总额
#     # "dupontIntburden":	"利润总额/息税前利润"    # ，反映企业利息负担，该比值高则税负较低。利润总额/息税前利润=1-利息费用/息税前利润
#     # "dupontEbittogr":	"息税前利润/营业总收入"   # ，反映企业经营利润率，是企业经营获得的可供全体投资人（股东和债权人）分配的盈利占企业全部营收收入的百分比
#     #
#     #
#     #
#     #
#     #
#     # }
#
#
#     return  result_all
#
# def caiwu_data0(codes,s_time,e_time,get_infos = {"query_profit_data":True,"query_operation_data":True,
#                                                 "query_growth_data":False,"query_balance_data":False,
#                                                 "query_cash_flow_data":False,"query_dupont_data":True,
#                                                 "query_performance_express_report":False,"query_forecast_report":False }):
#     # 登陆系统
#     lg = bs.login()
#     # 显示登陆返回信息
#     print('login respond error_code:' + lg.error_code)
#     print('login respond  error_msg:' + lg.error_msg)
#
#     time_list = []
#     for y in range(s_time,e_time+1,1):
#         for q in range(1,5,1):
#             time_list.append([y,q])
#     result_all = pd.DataFrame()
#     for k, v in get_infos.items():
#         if v:
#             # 财务部分指标
#             result_part0 = pd.DataFrame()
#             for y,q in time_list:
#                 print(k,y,q)
#                 for c0 in codes :
#                     # 查询季频估值指标盈利能力
#                     result0 = eval('bs.'+k)(code=c0, year=y,quarter=q)
#                     while (result0.error_code == '0') & result0.next():
#                         result_part0 = result_part0.append(result0.get_data(),ignore_index=True)
#
#
#             # 财务部分指标，汇总
#             if result_all.empty:
#                 result_all = result_part0
#                 # print(result_all)
#                 # exit()
#             else:
#                 result_all = pd.merge(result_all,result_part0,how='outer',on=['code','pubDate','statDate'])
#                 # print(result_all)
#
#     result_all = pd.DataFrame(result_all)
#     cols = {
#         "code"     :  "code",
#         "pubDate"	:  "发布财报日",
#         "statDate" :  "统计截至日",
#         "roeAvg"	 :  "净资产收益率(平均)(%)",
#         "npMargin":	   "销售净利率(%)",
#         "gpMargin":	   "销售毛利率(%)",
#         "netProfit"	:   "净利润(元)",
#         "epsTTM"	 :  "每股收益",
#         "MBRevenue"	:   "主营营业收入(元)",
#         "totalShare" :  "总股本",
#         "liqaShare"	:   "流通股本",
#         "NRTurnRatio" :"应收账款周转率(次)",
#         "NRTurnDays": "应收账款周转天数(天)",
#         "INVTurnRatio": "存货周转率(次)",
#         "INVTurnDays": "存货周转天数(天)",
#         "CATurnRatio": "流动资产周转率(次)",
#         "AssetTurnRatio": "总资产周转率",
#         "YOYEquity": "净资产同比增长率",
#         "YOYAsset": "总资产同比增长率",
#         "YOYNI": "净利润同比增长率",
#         "YOYEPSBasic": "基本每股收益同比增长率",
#         "YOYPNI": "归属母公司股东净利润同比增长率",
#         "CAToAsset": "流动资产除以总资产",
#         "NCAToAsset": "非流动资产除以总资产",
#         "tangibleAssetToAsset": "有形资产除以总资产",
#         "ebitToInterest": "已获利息倍数",
#         "CFOToOR": "经营活动产生的现金流量净额除以营业收入",
#         "CFOToNP": "经营性现金净流量除以净利润",
#         "CFOToGr": "经营性现金净流量除以营业总收入",
#         "dupontROE": "净资产收益率",
#         "dupontAssetStoEquity": "权益乘数",
#         "dupontAssetTurn": "总资产周转率",
#         "dupontPnitoni": "归属母公司股东的净利润/净利润",
#         "dupontNitogr": "净利润/营业总收入",
#         "dupontTaxBurden": "净利润/利润总额",
#         "dupontIntburden": "利润总额/息税前利润",
#         "dupontEbittogr": "息税前利润/营业总收入",
#         #
#     }
#     replace_name = {i: cols.get(i) for i in result_all.keys()}
#     result_all.rename(columns=replace_name,inplace=True)
#
#
#     # 打印输出
#     # 参数名称	   参数描述	              算法说明
#     # {"code"	:       "证券代码"
#     # "pubDate"	 :  "发布财报日"
#     # "statDate" :  "统计截至日"                财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30
#     # "roeAvg"	 :  "净资产收益率(平均)(%)"	  归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
#     # "npMargin":	   "销售净利率(%)"	          净利润/营业收入*100%
#     # "gpMargin":	   "销售毛利率(%)"	          毛利/营业收入*100%=(营业收入-营业成本)/营业收入*100%
#     # "netProfit"	:   "净利润(元)"
#     # "epsTTM"	 :  "每股收益"	               归属母公司股东的净利润TTM/最新总股本
#     # "MBRevenue"	:   "主营营业收入(元)"
#     # "totalShare" :  "总股本"
#     # "liqaShare"	:   "流通股本"
#     # "NRTurnRatio"	:"应收账款周转率(次)"	            营业收入/[(期初应收票据及应收账款净额+期末应收票据及应收账款净额)/2]
#     # "NRTurnDays"	:"应收账款周转天数(天)"	            季报天数/应收账款周转率(一季报：90天，中报：180天，三季报：270天，年报：360天)
#     # "INVTurnRatio"	:"存货周转率(次)"	            营业成本/[(期初存货净额+期末存货净额)/2]
#     # "INVTurnDays"	:"存货周转天数(天)"	            季报天数/存货周转率(一季报：90天，中报：180天，三季报：270天，年报：360天)
#     # "CATurnRatio"	:"流动资产周转率(次)"	            营业总收入/[(期初流动资产+期末流动资产)/2]
#     # "AssetTurnRatio"	:"总资产周转率"
#     # "YOYEquity":	"净资产同比增长率"           # (本期净资产-上年同期净资产)/上年同期净资产的绝对值*100%
#     # "YOYAsset":	"总资产同比增长率"  # (本期总资产-上年同期总资产)/上年同期总资产的绝对值*100%
#     # "YOYNI":	"净利润同比增长率"  # (本期净利润-上年同期净利润)/上年同期净利润的绝对值*100%
#     # "YOYEPSBasic":	"基本每股收益同比增长率"   # (本期基本每股收益-上年同期基本每股收益)/上年同期基本每股收益的绝对值*100%
#     # "YOYPNI":	"归属母公司股东净利润同比增长率"   # (本期归属母公司股东净利润-上年同期归属母公司股东净利润)/上年同期归属母公司股东净利润的绝对值*100%
#     # "CAToAsset":	"流动资产除以总资产"
#     # "NCAToAsset":	"非流动资产除以总资产"
#     # "tangibleAssetToAsset":	"有形资产除以总资产"
#     # "ebitToInterest":	"已获利息倍数"    # 息税前利润/利息费用
#     # "CFOToOR":	"经营活动产生的现金流量净额除以营业收入"
#     # "CFOToNP":	"经营性现金净流量除以净利润"
#     # "CFOToGr":	"经营性现金净流量除以营业总收入"
#     # "dupontROE":	"净资产收益率"    # 归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
#     # "dupontAssetStoEquity":	"权益乘数"  # ，反映企业财务杠杆效应强弱和财务风险	平均总资产/平均归属于母公司的股东权益
#     # "dupontAssetTurn":	"总资产周转率"    # ，反映企业资产管理效率的指标	营业总收入/[(期初资产总额+期末资产总额)/2]
#     # "dupontPnitoni":	"归属母公司股东的净利润/净利润"   # ，反映母公司控股子公司百分比。如果企业追加投资，扩大持股比例，则本指标会增加。
#     # "dupontNitogr":	"净利润/营业总收入"     # ，反映企业销售获利率
#     # "dupontTaxBurden":	"净利润/利润总额"  # ，反映企业税负水平，该比值高则税负较低。净利润/利润总额=1-所得税/利润总额
#     # "dupontIntburden":	"利润总额/息税前利润"    # ，反映企业利息负担，该比值高则税负较低。利润总额/息税前利润=1-利息费用/息税前利润
#     # "dupontEbittogr":	"息税前利润/营业总收入"   # ，反映企业经营利润率，是企业经营获得的可供全体投资人（股东和债权人）分配的盈利占企业全部营收收入的百分比
#     #
#     #
#     #
#     #
#     #
#     # }
#
#
#
#     return  result_all

def get_a_stock(code_name='sh.600004',s_time ="2020-10-23",e_time = "2020-10-31", frequency = 'd'):
    print('正在获取: ', code_name)
    rs = bs.query_history_k_data_plus(code=code_name,
                                      fields="date,code,open,high,low,close,preclose,volume,amount,"
                                             "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                      start_date=s_time, end_date=e_time,
                                      frequency=frequency, adjustflag="2")


    print('获取状态: ' + rs.error_msg)

    #### 打印结果集 ####
    if (rs.error_code == '0') & (len(rs.data) > 0):
        result = pd.DataFrame(rs.data, columns=['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                                                '复权状态', '换手率', '是否交易', '涨跌幅', '滚动市盈率', '市净率',
                                                '滚动市销率', '滚动市现率', '是否是st股'])

    # result = result.fillna(method="ffill")
    result['preclose'] = result['preclose'].apply(lambda x:float(x))
    result['close'] = result['close'].apply(lambda x:float(x))
    result['open'] = result['open'].apply(lambda x:float(x))
    result['涨跌幅'] = result['涨跌幅'].apply(lambda x:float(x) if len(x)!=0 else 0)
    # print(result.iloc[-3]["涨跌幅"],type(result.iloc[-3]["涨跌幅"]))
    # exit()
    result = cal_zhangting_with_st(df=result)
    return result
# 获取单个股票的日线数据,默认前复权。
async def get_a_stock_kline(code_name='sh.600000',s_time ='2010-07-01',e_time = '2017-12-31', frequency = 'd',n = -30):

    x0 = time.time_ns()

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

    # print('正在获取: ',code_name)
    rs = bs.query_history_k_data_plus(code = code_name,
                                      fields="date,code,open,high,low,close,preclose,volume,amount,"
                                             "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                      start_date=s_time, end_date=e_time,
                                      frequency=frequency, adjustflag="2")


    # print('获取状态: ' + rs.error_msg)

    #### 打印结果集 ####
    if (rs.error_code == '0') & (len(rs.data)>0):
        result = pd.DataFrame(rs.data, columns=['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                         '复权状态', '换手率', '是否交易', '涨跌幅', '滚动市盈率', '市净率',
                         '滚动市销率', '滚动市现率', '是否是st股'])


    result['preclose'] = result['preclose'].apply(lambda x: float(x) if len(x) != 0 else 0).astype(float)
    result['close'] = result['close'].apply(lambda x: float(x) if len(x) != 0 else 0).astype(float)
    result['open'] = result['open'].apply(lambda x: float(x) if len(x) != 0 else 0).astype(float)
    result['涨跌幅'] = result['涨跌幅'].apply(lambda x: float(x) if len(x) != 0 else 0).astype(float)
    result = cal_zhangting_with_st(df=result)

    df = MA_Sys(df=result)
    df = zdf_cal(df=df)
    df_to_save = df.iloc[n:].copy()
    # print(df_to_save)
    print("time:",(time.time_ns()-x0)/1000000000,"s")
    return df_to_save

async def main(stock_list,s_time ,e_time ,n=-30):
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(asyncio.wait(*tasks))
    tasks = []
    for s0 in stock_list:
        task_1 = asyncio.create_task(get_a_stock_kline(code_name = s0,s_time =s_time,e_time = e_time, frequency = 'd', n = n))
        tasks.append(task_1)

    res = await asyncio.gather(*tasks, return_exceptions=True)
    # print(len(res)) # 10.主任务输出res,协程任务结束，事件循环结束
    return res

def MA_Sys(df):
    v = df
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

def zdf_cal(df):
    v = df
    v["60日涨跌幅"] = 100*(v["close"]-v["close"].shift(60))/v["close"].shift(60)
    v["20日涨跌幅"] = 100*(v["close"]-v["close"].shift(20))/v["close"].shift(60)
    return v

def update_new_ASTOCKs():
    # 登陆系统
    lg = bs.login()
    hangye_res = get_basic_code_info(kechuang=True)

    hangye_res = hangye_res[hangye_res["outDate"].str.len() == 0]

    codes_list = hangye_res['code'].values.tolist()

    # print(len(codes_list), codes_list)

    a = time.process_time()

    s_time = str(datetime.datetime.now().date() - datetime.timedelta(days=240))
    e_time = str(datetime.datetime.now().date())
    sample = codes_list[:]
    # print(sample)
    index_log = {}
    index_dict = {"sh.000001": "上证指数", "sz.399106": "深圳综指", "sh.000300": "沪深300"}  # ,'sh.000016':"中证50","sh.000300":"沪深300"

    for k, v in index_dict.items():
        resdf0 = get_a_stock(code_name=k, s_time=s_time, e_time=e_time, frequency='d')
        resdf0["close"] = resdf0["close"].astype(float)
        df = MA_Sys(df=resdf0)
        df = zdf_cal(df=df)
        # df_to_save = df.iloc[-1:].copy()
        index_log[f"{v}涨跌幅"] = float(df.iloc[-1:]["涨跌幅"])
        index_log[f"{v}20日涨跌幅"] = float(df.iloc[-1:]["20日涨跌幅"])
        index_log[f"{v}60日涨跌幅"] = float(df.iloc[-1:]["60日涨跌幅"])

    # 携程，更新kline
    x0 = time.time_ns()
    if True == 1:
        res_list = asyncio.run(main(sample, s_time=s_time, e_time=e_time,n=-30))  # 1.程序进入main()函数，事件循环开启

        for ix, res0 in enumerate(res_list):
            if ix == 0:
                pd.DataFrame(res0).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=True, mode="w")
            else:
                pd.DataFrame(res0).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=False, mode="a")
            # time.sleep(0.1)



    # # 单循环
    # a0 = time.time_ns()
    # if True == 0:
    #
    #     # sample = ["sh.600086"]#sample[:]
    #     for ix, s in enumerate(sample[:]):
    #         a = time.time_ns()
    #         resdf0 = get_a_stock(code_name=s, s_time=s_time, e_time=e_time, frequency='d')
    #
    #         df = MA_Sys(df=resdf0)
    #         df = zdf_cal(df=df)
    #         df_to_save = df.iloc[-1:].copy()
    #         # 添加指数数据
    #         # for k0,v0 in index_log.items():
    #         #     df_to_save[k0] = v0
    #         # # 计算相关指标
    #         # for v in index_dict.values():
    #         #     df_to_save[f"涨跌幅1强度_{v}"] = (df_to_save["涨跌幅"]-df_to_save[f"{v}涨跌幅"])*100
    #         #     df_to_save[f"涨跌幅20强度_{v}"] = (df_to_save["20日涨跌幅"]-df_to_save[f"{v}20日涨跌幅"])*100
    #         #     df_to_save[f"涨跌幅60强度_{v}"] =( df_to_save["60日涨跌幅"]-df_to_save[f"{v}60日涨跌幅"])*100
    #
    #         if ix == 0:
    #             pd.DataFrame(df_to_save).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=True, mode="w")
    #         else:
    #             pd.DataFrame(df_to_save).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=False, mode="a")
    #         # time.sleep(0.1)
    #         print((time.time_ns() - a) / 1000000000, "s")
    # # exit()
    # print((time.time_ns() - a0) / 1000000000, "s")



    df_stocks = pd.read_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index_col=0)
    df_stocks = df_stocks.reset_index()
    # print(df_stocks.keys())
    df_stocks.drop_duplicates(subset=["code","date"],keep="first",inplace=True)
    df_stocks["涨跌幅"] = df_stocks["涨跌幅"].astype(float)


    # 添加指数数据
    for k0, v0 in index_log.items():
        if k0 == "上证指数涨跌幅":
            df_stocks.loc[df_stocks["code"].str.contains("sh"), "1指数涨跌幅"] = v0
        elif k0 == "深圳综指涨跌幅":
            df_stocks.loc[~df_stocks["code"].str.contains("sh"), "1指数涨跌幅"] = v0
        elif k0 == "上证指数20日涨跌幅":
            df_stocks.loc[df_stocks["code"].str.contains("sh"), "20指数涨跌幅"] = v0
        elif k0 == "深圳综指20日涨跌幅":
            df_stocks.loc[~df_stocks["code"].str.contains("sh"), "20指数涨跌幅"] = v0
        elif k0 == "上证指数60日涨跌幅":
            df_stocks.loc[df_stocks["code"].str.contains("sh"), "60指数涨跌幅"] = v0
        elif k0 == "深圳综指60日涨跌幅":
            df_stocks.loc[~df_stocks["code"].str.contains("sh"), "60指数涨跌幅"] = v0
        else:
            df_stocks[k0] = v0
        # print(k0)
    # 计算相关指标
    df_stocks[f"1指数涨跌幅强度"] = (df_stocks["涨跌幅"] - df_stocks[f"1指数涨跌幅"])
    df_stocks[f"20指数涨跌幅强度"] = (df_stocks["20日涨跌幅"] - df_stocks[f"20指数涨跌幅"])
    df_stocks[f"60指数涨跌幅强度"] = (df_stocks["60日涨跌幅"] - df_stocks[f"60指数涨跌幅"])
    df_stocks[f"1沪深300涨跌幅强度"] = (df_stocks["涨跌幅"] - df_stocks[f"沪深300涨跌幅"])
    df_stocks[f"20沪深300涨跌幅强度"] = (df_stocks["20日涨跌幅"] - df_stocks[f"沪深30020日涨跌幅"])
    df_stocks[f"60沪深300涨跌幅强度"] = (df_stocks["60日涨跌幅"] - df_stocks[f"沪深30060日涨跌幅"])


    df_stocks = df_stocks.reset_index()
    df_stocks = df_stocks.merge(hangye_res[["code", "code_name", "industry"]], how='left', on='code',suffixes=("", "_y"))
    pd.DataFrame(df_stocks).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=True, mode="w")

    print(df_stocks.shape)
    print("保存成功：", f'data/all_240_new_data_数据.csv')
    return "ok"

def update_all_ASTOCKs(s_time = "2010-01-01"):
    # 登陆系统
    lg = bs.login()
    hangye_res = get_basic_code_info(kechuang=True)

    hangye_res = hangye_res[hangye_res["outDate"].str.len() == 0]

    codes_list = hangye_res['code'].values.tolist()

    # print(len(codes_list), codes_list)

    a = time.process_time()

    s_time =s_time
    e_time = str(datetime.datetime.now().date())
    sample = codes_list[:]
    # print(sample)

    # 携程，更新kline
    x0 = time.time_ns()
    if True == 1:
        res_list = asyncio.run(main(sample, s_time=s_time, e_time=e_time,n=0))  # 1.程序进入main()函数，事件循环开启
        for ix, res0 in enumerate(res_list):
            if ix == 0:
                pd.DataFrame(res0).to_csv(f'data/all_2010_new_data_数据.csv',
                                          encoding='utf-8',
                                          index=False, header=True, mode="w")

            else:
                pd.DataFrame(res0).to_csv(f'data/all_2010_new_data_数据.csv',
                                          encoding='utf-8',
                                          index=False, header=False, mode="a")


    # # 单循环
    # a0 = time.time_ns()
    # if True == 0:
    #
    #     # sample = ["sh.600086"]#sample[:]
    #     for ix, s in enumerate(sample[:]):
    #         a = time.time_ns()
    #         resdf0 = get_a_stock(code_name=s, s_time=s_time, e_time=e_time, frequency='d')
    #
    #         df = MA_Sys(df=resdf0)
    #         df = zdf_cal(df=df)
    #         df_to_save = df.iloc[-1:].copy()
    #         # 添加指数数据
    #         # for k0,v0 in index_log.items():
    #         #     df_to_save[k0] = v0
    #         # # 计算相关指标
    #         # for v in index_dict.values():
    #         #     df_to_save[f"涨跌幅1强度_{v}"] = (df_to_save["涨跌幅"]-df_to_save[f"{v}涨跌幅"])*100
    #         #     df_to_save[f"涨跌幅20强度_{v}"] = (df_to_save["20日涨跌幅"]-df_to_save[f"{v}20日涨跌幅"])*100
    #         #     df_to_save[f"涨跌幅60强度_{v}"] =( df_to_save["60日涨跌幅"]-df_to_save[f"{v}60日涨跌幅"])*100
    #
    #         if ix == 0:
    #             pd.DataFrame(df_to_save).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=True, mode="w")
    #         else:
    #             pd.DataFrame(df_to_save).to_csv(f'data/all_240_new_data_数据.csv', encoding='utf-8', index=False, header=False, mode="a")
    #         # time.sleep(0.1)
    #         print((time.time_ns() - a) / 1000000000, "s")
    # # exit()
    # print((time.time_ns() - a0) / 1000000000, "s")

    df_stocks = pd.read_csv(f'data/all_2010_new_data_数据.csv', encoding='utf-8', index_col=0)
    df_stocks = df_stocks.reset_index()


    # print(df_stocks.keys())
    df_stocks.drop_duplicates(subset=["code","date"],keep="first",inplace=True)

    df_stocks["涨跌幅"] = df_stocks["涨跌幅"].astype(float)

    df_stocks = df_stocks.reset_index()

    df_stocks = df_stocks.merge(hangye_res[["code", "code_name", "industry"]], how='left', on='code',suffixes=("", "_y"))
    pd.DataFrame(df_stocks).to_csv(f'data/all_2010_new_data_数据.csv', encoding='utf-8', index=False, header=True, mode="w")
    pd.DataFrame(df_stocks).to_pickle(f'data/all_2010_new_data_数据.pkl')

    print(df_stocks.shape)
    print("保存成功：", f'data/all_2010_new_data_数据.csv/pkl')
    return "ok"



if __name__ =='__main__':
    #768w
    df = pd.read_pickle(r"F:\repo\A股指数回测分析系统\data\all_2010_new_data_数据.bz2.pkl",compression="bz2")
    # print(df.shape)
    # print(len(df["code"].unique()))

    # update_all_ASTOCKs(s_time = "2010-01-01")
    # dfs = pd.read_csv(r"F:\repo\A股指数回测分析系统\data\all_2010_new_data_数据.gz",chunksize =1000000,)
    # dfz = pd.DataFrame()
    # for i,df in enumerate(dfs):
    #     print(i)
    #     df = pd.DataFrame(df)
    #     dfz = dfz.append(df,ignore_index=True)
    #
    # dfz.to_pickle(r"F:\repo\A股指数回测分析系统\data\all_2010_new_data_数据.bz2.pkl", compression="bz2")

#     # all = pd.read_csv(r'data/all_240_new_data_数据.csv',encoding="utf-8")
#     # print(all.keys())
#     # all.to_csv(r'data/all_240_new_data_数据.csv',"w")
#
#     if True == 0:
#         # 登陆系统
#         lg = bs.login()
#         hangye_res = get_basic_code_info(kechuang=True)
#         hangye_res = hangye_res[hangye_res["outDate"].str.len()== 0]
#
#         codes_list = hangye_res['code'].values.tolist()
#
#
#         print(len(codes_list),codes_list)
#         a = time.process_time()
#
#         s_time = str(datetime.datetime.now().date()-datetime.timedelta(days=240))
#         e_time = str(datetime.datetime.now().date())
#         sample = codes_list[:]
#         print(sample)
#         index_log = {}
#         index_dict ={"sh.000001":"上证指数","sz.399106":"深圳综指","sh.000300":"沪深300"} #,'sh.000016':"中证50","sh.000300":"沪深300"
#
#         for k, v in index_dict.items():
#             resdf0 = get_a_stock(code_name=k, s_time=s_time, e_time=e_time, frequency='d')
#             resdf0["close"] = resdf0["close"].astype(float)
#             df = MA_Sys(df=resdf0)
#             df = zdf_cal(df=df)
#             # df_to_save = df.iloc[-1:].copy()
#             index_log[f"{v}涨跌幅"] = float(df.iloc[-1:]["涨跌幅"])
#             index_log[f"{v}20日涨跌幅"] = float(df.iloc[-1:]["20日涨跌幅"])
#             index_log[f"{v}60日涨跌幅"] = float(df.iloc[-1:]["60日涨跌幅"])
#
#         # 携程，更新kline
#         x0 = time.time_ns()
#         if True==1:
#             res_list = asyncio.run(main(sample, s_time=s_time, e_time=e_time))  # 1.程序进入main()函数，事件循环开启
#             for ix ,res0 in enumerate(res_list):
#                 if ix == 0:
#                     pd.DataFrame(res0).to_csv(f'all_240_{e_time}_数据.csv', encoding='utf-8', index=False, header=True, mode="a")
#                 else:
#                     pd.DataFrame(res0).to_csv(f'all_240_{e_time}_数据.csv', encoding='utf-8', index=False, header=False, mode="a")
#                 # time.sleep(0.1)
#         # 单循环
#         a0 = time.time_ns()
#
#         if True ==0:
#
#             # sample = ["sh.600086"]#sample[:]
#             for ix, s in enumerate(sample[:]):
#                 a = time.time_ns()
#                 resdf0 = get_a_stock(code_name=s, s_time=s_time, e_time=e_time, frequency='d')
#
#                 df = MA_Sys(df=resdf0)
#                 df = zdf_cal(df=df)
#                 df_to_save = df.iloc[-1:].copy()
#                 # 添加指数数据
#                 # for k0,v0 in index_log.items():
#                 #     df_to_save[k0] = v0
#                 # # 计算相关指标
#                 # for v in index_dict.values():
#                 #     df_to_save[f"涨跌幅1强度_{v}"] = (df_to_save["涨跌幅"]-df_to_save[f"{v}涨跌幅"])*100
#                 #     df_to_save[f"涨跌幅20强度_{v}"] = (df_to_save["20日涨跌幅"]-df_to_save[f"{v}20日涨跌幅"])*100
#                 #     df_to_save[f"涨跌幅60强度_{v}"] =( df_to_save["60日涨跌幅"]-df_to_save[f"{v}60日涨跌幅"])*100
#
#                 if ix ==0 :
#                     pd.DataFrame(df_to_save).to_csv(f'all_240_{e_time}_数据.csv', encoding='utf-8', index=False, header=True, mode="a")
#                 else:
#                     pd.DataFrame(df_to_save).to_csv(f'all_240_{e_time}_数据.csv', encoding='utf-8', index=False, header=False, mode="a")
#                 # time.sleep(0.1)
#                 print((time.time_ns() - a) / 1000000000, "s")
#         # exit()
#         print((time.time_ns() - a0) / 1000000000, "s")
#         df_stocks = pd.read_csv(f'all_240_{e_time}_数据.csv', encoding='utf-8',index_col=0)
#         # df_stocks.to_csv(f'all_240_{e_time}_数据.csv',encoding="utf-8")
#         df_stocks["涨跌幅"] = df_stocks["涨跌幅"].astype(float)
#         # print(df_stocks.tail(10))
#         # exit()
#         # 添加指数数据
#         for k0, v0 in index_log.items():
#             if k0=="上证指数涨跌幅":
#                 df_stocks.loc[df_stocks["code"].str.contains("sh"),"1指数涨跌幅"] = v0
#             elif k0=="深圳综指涨跌幅":
#                 df_stocks.loc[~df_stocks["code"].str.contains("sh"),"1指数涨跌幅"] = v0
#             elif k0=="上证指数20日涨跌幅":
#                 df_stocks.loc[df_stocks["code"].str.contains("sh"),"20指数涨跌幅"] = v0
#             elif k0=="深圳综指20日涨跌幅":
#                 df_stocks.loc[~df_stocks["code"].str.contains("sh"),"20指数涨跌幅"] = v0
#             elif k0 == "上证指数60日涨跌幅":
#                 df_stocks.loc[df_stocks["code"].str.contains("sh"), "60指数涨跌幅"] = v0
#             elif k0 == "深圳综指60日涨跌幅":
#                 df_stocks.loc[~df_stocks["code"].str.contains("sh"), "60指数涨跌幅"] = v0
#             else:
#                 df_stocks[k0] = v0
#             print(k0)
#         # 计算相关指标
#         df_stocks[f"1指数涨跌幅强度"] = (df_stocks["涨跌幅"] - df_stocks[f"1指数涨跌幅"])
#         df_stocks[f"20指数涨跌幅强度"] = (df_stocks["20日涨跌幅"] - df_stocks[f"20指数涨跌幅"])
#         df_stocks[f"60指数涨跌幅强度"] = (df_stocks["60日涨跌幅"] - df_stocks[f"60指数涨跌幅"])
#         df_stocks[f"1沪深300涨跌幅强度"] = (df_stocks["涨跌幅"] - df_stocks[f"沪深300涨跌幅"])
#         df_stocks[f"20沪深300涨跌幅强度"] = (df_stocks["20日涨跌幅"] - df_stocks[f"沪深30020日涨跌幅"])
#         df_stocks[f"60沪深300涨跌幅强度"] = (df_stocks["60日涨跌幅"] - df_stocks[f"沪深30060日涨跌幅"])
#
#         print(df_stocks[["code","涨跌幅","1指数涨跌幅","1指数涨跌幅强度","20日涨跌幅", "20指数涨跌幅","沪深30020日涨跌幅","20沪深300涨跌幅强度", "20指数涨跌幅强度",
#                          "60日涨跌幅", "60指数涨跌幅", "60指数涨跌幅强度"]].sample(9))
#
#         df_stocks = df_stocks.reset_index()
#
#         df_stocks = df_stocks.merge(hangye_res[["code", "code_name", "industry"]], how='left', on='code')
#         pd.DataFrame(df_stocks).to_csv(f'all_240_{e_time}_数据.csv', encoding='utf-8', index=False, header=True, mode="w")
#
#         print(df_stocks.shape)
#         print("保存成功：",f'all_240_{e_time}_数据.csv')
#
#         #合并更新数据
#         if True == 0:
#             all = pd.read_csv(f'all_240日数据.csv')
#             new = pd.read_csv(f'all_{s_time}_数据.csv')
#             all.drop_duplicates(subset=['code', 'date'], keep='first', inplace=True)
#             new.drop_duplicates(subset=['code', 'date'], keep='first', inplace=True)
#             all = all.append(new,ignore_index=True)
#             all = all[all['date']!="date"]
#             all.drop_duplicates(subset=['code', 'date'], keep='first', inplace=True)
#             all.sort_values(by=['date','code'],ascending=[True,True],inplace=True)
#             all = all.round(3)
#             all.to_csv('all_100日数据.csv', encoding='utf-8',index=False,header=True, mode="w")
#             print(all.tail(10))
#             exit()
#         # print('--2run_time:(s)====', time.process_time() - a)
#         # 处理生成60MA指标
#         if True == 0:
#             try:
#                 df_stocks = pd.read_csv('all_100日数据.csv', encoding='utf-8')
#                 df_stocks =df_stocks.reset_index()
#
#                 # df_stocks = df_stocks.loc[df_stocks['code'].isin(['sh.600000', 'sh.600004', 'sh.600006', 'sh.600007', 'sh.600008', 'sh.600009', 'sh.600010', 'sh.600011'])]
#
#                 df_stocks = df_stocks.merge(hangye_res[["code","code_name","industry"]], how='left', on = 'code')
#                 df_stocks= df_stocks[['date', 'code_name', "industry",'code', 'open', 'high', 'low', 'close', 'preclose','涨跌幅', '换手率','volume','amount' , '滚动市盈率','市净率', '滚动市现率', '滚动市销率',   '是否交易', '是否是st股']]
#
#                 df_stocks =df_stocks.reset_index()
#
#                 df_stocks["close"] =df_stocks["close"].astype(float)
#                 all_outputdf=pd.DataFrame()
#                 # df_stocks["MA60"] = df_stocks.groupby('code', as_index=True, dropna=False).apply(lambda x: talib.MA(x["close"], 60)).reset_index()[0]
#                 b = time.process_time_ns()
#
#                 for k,v in df_stocks.groupby('code'):
#                     a = time.process_time_ns()
#
#                     v["MA60"]= talib.MA(v["close"], 60)
#                     v["MA20"]= talib.MA(v["close"], 20)
#                     con60 = v["close"] > v["MA60"]
#                     con60_2 = v["close"] < v["MA60"]
#                     con20 = v["close"] > v["MA20"]
#                     con20_2 = v["close"] < v["MA20"]
#
#                     v.loc[v["MA60"].isnull(), "是否大于60均线"] = 0
#                     v.loc[con60, "是否大于60均线"] = 1
#                     v.loc[con60_2, "是否大于60均线"] = -1
#                     v.loc[v["MA20"].isnull(), "是否大于20均线"] = 0
#                     v.loc[con20, "是否大于20均线"] = 1
#                     v.loc[con20_2, "是否大于20均线"] = -1
#
#                     con60_3 = v["是否大于60均线"] != v["是否大于60均线"].shift()
#                     con20_3 = v["是否大于20均线"] != v["是否大于20均线"].shift()
#                     v.loc[con60_3, "大于60均线开始时间"] = v.loc[con60_3, "date"]
#                     v.loc[con20_3, "大于20均线开始时间"] = v.loc[con20_3, "date"]
#                     v.fillna(method="ffill", inplace=True)
#                     # print(v.groupby([ "大于60均线开始时间"]).get_group(v.iloc[-1]["大于60均线开始时间"]))
#                     g60 = v.groupby(["大于60均线开始时间"]).get_group(v.iloc[-1]["大于60均线开始时间"])
#                     g20 = v.groupby(["大于20均线开始时间"]).get_group(v.iloc[-1]["大于20均线开始时间"])
#                     v.loc[v.index[-1],"是否大于60均线"] = g60["是否大于60均线"].sum()
#                     v.loc[v.index[-1],"是否大于20均线"] = g20["是否大于20均线"].sum()
#                     all_outputdf =all_outputdf.append(v.iloc[-1],ignore_index=True)
#                     print( (time.process_time_ns()-a)/1000000000)
#                     # df_stocks.loc[df_stocks["code"]==k,"MA60"] = talib.MA(v["close"], 60)
#                 print(all_outputdf.tail(25))
#                 print(all_outputdf.shape)
#                 all_outputdf.to_csv('all_new10日数据_指标20_60MA.csv', index=True, header=True, encoding='utf-8', mode="w")
#                 print((time.process_time_ns() - b) / 1000000000,"s")
#
#                 exit()
#                 print("cal_ma60,off!")
#                 # df_stocks =df_stocks.sort_values(["code"],[1])
#                 # print(df_stocks)
#                 # exit()
#
#                 # exit()
#
#                 con = df_stocks["close"] > df_stocks["MA60"]
#                 con2 = df_stocks["close"] < df_stocks["MA60"]
#                 # con &= df_stocks["close"].shift()<=df_stocks["MA60"].shift()
#                 # con2 &= df_stocks["close"].shift() >= df_stocks["MA60"].shift()
#                 #标记无数据的row
#                 df_stocks.loc[df_stocks["MA60"].isnull(),"是否大于60均线"] = 0
#                 df_stocks.loc[con,"是否大于60均线"] = 1
#                 df_stocks.loc[con2,"是否大于60均线"] = -1
#
#                 # df_stocks.loc[df_stocks["是否大于60均线"]==1,"大于60均线开始时间"] = df_stocks.loc[df_stocks["是否大于60均线"]==1,"date"]
#                 # df_stocks.loc[df_stocks["是否大于60均线"]==-1, "大于60均线开始时间"] = df_stocks.loc[df_stocks["是否大于60均线"]==-1, "date"]
#                 # 填充信号row
#                 # df_stocks["大于60均线开始时间"].fillna(method="ffill",inplace=True)
#                 # df_stocks["是否大于60均线"].fillna(method="ffill",inplace=True)
#
#                 print("cal_大于60均线天数!")
#                 newdf = pd.DataFrame()
#                 for k,v in df_stocks.groupby(['code']):
#                     print(k)
#                     con3 = v["是否大于60均线"] != v["是否大于60均线"].shift()
#                     v.loc[con3,"大于60均线开始时间"] = v.loc[con3,"date"]
#                     v.fillna(method ="ffill",inplace=True)
#                     newdf =newdf.append(v,ignore_index=True)
#
#                 df_stocks = newdf
#
#                 df_stocks["是否大于60均线"] = df_stocks.groupby(['code',"大于60均线开始时间"], as_index=True,sort=False).apply(lambda x: x["是否大于60均线"].cumsum()).reset_index()["是否大于60均线"]
#                 dfres0 = df_stocks.groupby("code", as_index=True, ).apply(lambda x: x.iloc[-1])
#
#                 dfres0 = dfres0[['date', 'code', 'code_name', 'industry', '滚动市盈率', '市净率', "滚动市现率", "滚动市销率", "涨跌幅", 'close', "MA60", '是否大于60均线']].round(3)
#                 print(dfres0.sample(50))
#                 dfres0.to_csv('all_new10日数据_指标60MA.csv', index=True, header=True, encoding='utf-8', mode="w")
#                 exit()
#                 print("cal_大于60均线天数!,off!")
#
#             except Exception as e:
#                 pass
#                 # print(df_stocks.tail(100))
#             # print(df_stocks.groupby(['code',"大于60均线开始时间"], as_index=True,dropna=False,sort=False).apply(lambda x: x["是否大于60均线"].cumsum()).reset_index())
#             # df_stocks.to_csv('all_100日数据_指标60MA.csv', encoding='utf-8',mode = "w")
#             # time.sleep(1)
#         # 展示数据
#         if True == 0:
#             df = pd.read_csv('all_100日数据_指标60MA.csv',)
#             dfres0 = df.iloc[:].groupby("code",as_index=True,).apply(lambda x:x.iloc[-1])
#
#             dfres0 =dfres0[['date','code', 'code_name', 'industry', '滚动市盈率', '市净率', "滚动市现率", "滚动市销率", "涨跌幅",'close',"MA60",'是否大于60均线']].round(3)
#             print(dfres0.tail(50))
#             dfres0.to_csv('all_new10日数据_指标60MA.csv',index=True,header=True, encoding='utf-8', mode="w")
#             print(dfres0.shape)
#
#             print('run_time:(s)====', time.process_time() - a)
