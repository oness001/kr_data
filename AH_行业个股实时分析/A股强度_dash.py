import base64
import datetime
import io
# import threading

import numpy as np
import time
import dash
from dash.dependencies import Input, Output,State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
# from dash.exceptions import PreventUpdate
from flask_apscheduler import APScheduler
from flask import Flask

from functools import reduce
import plotly.figure_factory as ff
from bs_A股行业均线系统强度数据下载 import update_new_ASTOCKs
from H股_处理 import update_h_stocks


np.seterr(divide='ignore', invalid='ignore')

pd_display_rows = 10
pd_display_cols = 100
pd_display_width = 1000
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def split_filter_part(filter_part):
    operators = [['ge ', '>='],
                 ['le ', '<='],
                 ['lt ', '<'],
                 ['gt ', '>'],
                 ['ne ', '!='],
                 ['eq ', '='],
                 ['contains '],
                 ['datestartswith ']]
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

def split_all_filter(filtering_expressions):
    new_filtering_expressions =[]
    for filter_part in filtering_expressions:
        if "+" in filter_part:
            name = filter_part[filter_part.find('{') + 1: filter_part.rfind('}')]
            value_part = filter_part[filter_part.rfind('}') + 1:]
            value_lists = str(value_part).replace('"', '').split("+")

            for v0 in value_lists:
                new_filtering_expressions.append("{%s}" % name + v0)
        else:
            new_filtering_expressions.append(filter_part)

    return new_filtering_expressions

def data_bars2(df, column,colordn ="CCFF00",color_up1 = "FFAA66",color_up2 = "FF8888"):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [((df[column].max() - df[column].min()) * i) + df[column].min() for i in bounds]
    max_range0 = np.max(ranges)
    min_range0 = np.min(ranges)
    mean_ranges = df[column].mean()
    fws_75 = (max_range0+ mean_ranges)/2
    # print(column,min_range0,mean_ranges,fws_75,max_range0)
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        # max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        # 小于平均
        styles.append({'if': {'filter_query': ('{{{column}}} >= {min_bound}' +
                                               (' && {{{column}}} < {max_bound}'
                                                                               if (i < len(bounds) - 1) else ''))
            .format(column=column, min_bound=min_bound, max_bound=mean_ranges),'column_id': column},
            'background': ("""
                    linear-gradient(90deg,
                    #{color} 0%,
                    #{color} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(color = colordn ,max_bound_percentage=max_bound_percentage)),
            'paddingBottom': 1,
            'paddingTop': 1
        })
        # 大于平均小于75
        styles.append({'if': {'filter_query': ('{{{column}}} >= {min_bound}' + (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else ''))
            .format(column=column, min_bound=mean_ranges, max_bound=fws_75), 'column_id': column},
               'background': ("""
                    linear-gradient(90deg,
                    #{color} 0%,
                    #{color} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(color=color_up1, max_bound_percentage=max_bound_percentage)),
               'paddingBottom': 2,
               'paddingTop': 2
        })
        # 大于75
        styles.append({'if': {'filter_query': ('{{{column}}} >= {min_bound}' + (' && {{{column}}} > {max_bound}' if (i < len(bounds) - 1) else ''))
                      .format(column=column, min_bound=fws_75, max_bound=fws_75), 'column_id': column},
                       'background': ("""
                            linear-gradient(90deg,
                            #{color} 0%,
                            #{color} {max_bound_percentage}%,
                            white {max_bound_percentage}%,
                            white 100%)
                        """.format(color=color_up2, max_bound_percentage=max_bound_percentage)),
                       'paddingBottom': 2,
                       'paddingTop': 2
                       })

        styles.append({'if': {'filter_query': ('{{{column}}} >= {max_bound}')
                      .format(column=column, max_bound=max_range0), 'column_id': column},
                       'background': ("""
                                    linear-gradient(90deg,
                                    #{color} 0%,
                                    #{color} {max_bound_percentage}%,
                                    white {max_bound_percentage}%,
                                    white 100%)
                                """.format(color="FF0033", max_bound_percentage=100)),
                       'paddingBottom': 2,
                       'paddingTop': 2
                       })
        styles.append({'if': {'filter_query': ('{{{column}}} <= {max_bound}')
                      .format(column=column, max_bound=min_range0), 'column_id': column},
                       'background': ("""
                                            linear-gradient(90deg,
                                            #{color} 0%,
                                            #{color} {max_bound_percentage}%,
                                            white {max_bound_percentage}%,
                                            white 100%)
                                        """.format(color="99CCFF", max_bound_percentage=100)),
                       'paddingBottom': 2,
                       'paddingTop': 2
                       })


    return styles

def data_bars(df, column,colordn ="CCFF00",color_up = "FF9999"):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [((df[column].max() - df[column].min()) * i) + df[column].min() for i in bounds]
    mean_ranges = np.mean(ranges)
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound_percentage = bounds[i] * 100
        styles.append({'if': {'filter_query': ('{{{column}}} >= {min_bound}' +(' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else ''))
                      .format(column=column, min_bound=min_bound, max_bound=mean_ranges),'column_id': column},
            'background': ("""
                    linear-gradient(90deg,
                    #{color} 0%,
                    #{color} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(color = colordn ,max_bound_percentage=max_bound_percentage)),
            'paddingBottom': 2,
            'paddingTop': 2
        })
        styles.append({'if': {'filter_query': ('{{{column}}} >= {min_bound}' + (' && {{{column}}} > {max_bound}' if (i < len(bounds) - 1) else ''))
                      .format(column=column, min_bound=min_bound, max_bound=mean_ranges), 'column_id': column},
               'background': ("""
                    linear-gradient(90deg,
                    #{color} 0%,
                    #{color} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(color=color_up, max_bound_percentage=max_bound_percentage)),
               'paddingBottom': 2,
               'paddingTop': 2
        })


    return styles

# 所有股票带自选股标记
def get_all_codes_with_zxg(df0, thszxg_df):
    df0["code_num"] = df0["code"].str[3:]
    df0 = df0.merge(thszxg_df, how="left", on="code_num")
    for i in ['市净率', "滚动市盈率", "滚动市现率", "滚动市销率"]: df0["行业" + i] = df0.groupby('industry', as_index=True)[i].transform('mean')
    df = df0
    df = df.round(3)
    df["industry"].fillna("未选", inplace=True)
    df["自选股分类"].fillna("未选", inplace=True)
    # print(df.keys())
    df = df[['date', 'code', 'code_name', 'industry', "自选股分类", '滚动市盈率', '行业滚动市盈率', '市净率', '行业市净率', '滚动市现率',
             '行业滚动市现率', '滚动市销率', '行业滚动市销率', '涨跌幅', 'close', "1指数涨跌幅",
             '是否大于60均线', "是否大于20均线", "是否大于120均线",
             '1指数涨跌幅强度', '20指数涨跌幅强度','60指数涨跌幅强度',
             '1沪深300涨跌幅强度', '20沪深300涨跌幅强度','60沪深300涨跌幅强度',
             ]]
    df.rename(columns={"industry": "行业名", '是否大于60均线': "大于60线", "是否大于20均线": "大于20线", "是否大于120均线": "大于120线",
                       '1指数涨跌幅强度':"1强度_指数", '20指数涨跌幅强度':"20强度_指数", '60指数涨跌幅强度':"60强度_指数"
                       }, inplace=True)
    return df

# 所有自选股
def get_zxg_df(df):
    zxg_df0 = df[df['自选股分类'] != "未选"]
    zxg_df0 = zxg_df0.drop_duplicates(subset=['code'], keep='first')
    zxg_df0.sort_values("涨跌幅", ascending=False, inplace=True)
    return zxg_df0

# 行业强度
def hy_qiangdu_productor(df):
    # print(df.keys())
    # { "滚动市盈率": ["mean"],"市净率": ["mean"], "滚动市现率": ["mean"],"滚动市销率": ["mean"], "涨跌幅": ["mean"], }
    tj_cols = {i:["mean"]  for i in ['滚动市盈率', '市净率', '涨跌幅', '1强度_指数', '60强度_指数'] if i in df.columns}
    tj_cols["code"] = ["count"]
    # print(df.tail())
    # print(df["行业名"].tail())

    s_df = df.groupby("行业名").agg(tj_cols).reset_index()
    s_df.columns = [a for a,b  in s_df.columns ]
    # print(pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["大于20线"]>0].shape[0])))
    s_df["ge_20ma_n"] = pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["大于20线"]>0].shape[0])).iloc[:,-1]
    s_df["ge_60ma_n"] = pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["大于60线"]>0].shape[0])).iloc[:,-1]
    s_df["ge_120ma_n"] = pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["大于120线"]>0].shape[0])).iloc[:,-1]
    s_df["60强度"] = s_df["ge_20ma_n"]*100 /s_df["code"]
    s_df["20强度"] = s_df["ge_60ma_n"]*100 /s_df["code"]
    s_df["120强度"] = s_df["ge_120ma_n"]*100 /s_df["code"]

    s_df["每日alpha强度_n"] = pd.DataFrame(df.groupby("行业名", as_index=False).apply(lambda x: x[x["1强度_指数"] > 0].shape[0])).iloc[:, -1]
    s_df["60alpha强度_n"] = pd.DataFrame(df.groupby("行业名", as_index=False).apply(lambda x: x[x["60强度_指数"] > 0].shape[0])).iloc[:, -1]
    s_df["每日alpha%"] = s_df["每日alpha强度_n"] * 100 / s_df["code"]
    s_df["60alpha%"] = s_df["60alpha强度_n"] * 100 / s_df["code"]

    s_df.rename(columns = {"code":"股票数",'1强度_指数':"每日alpha_mean", '60强度_指数':'60alpha_mean'},inplace=True)
    s_df = s_df.round(3)
    # print(s_df.keys())
    s_df = s_df[['行业名', '股票数', '滚动市盈率', '市净率',
                    '120强度','60强度', '20强度',
                   '每日alpha%', '60alpha%','每日alpha_mean', '60alpha_mean']]
    s_df.sort_values("60强度", ascending=False, inplace=True)

    return s_df

# 行业强度
def H_hy_qiangdu_productor(df):
    # print(df.keys())
    # { "滚动市盈率": ["mean"],"市净率": ["mean"], "滚动市现率": ["mean"],"滚动市销率": ["mean"], "涨跌幅": ["mean"], }

    tj_cols = {i:["mean"]  for i in ['close',  '是否大于20均线', '是否大于60均线','是否大于120均线',] if i in df.columns}
    tj_cols["code"] = ["count"]
    # print(df.tail())
    # exit()
    s_df = df.groupby("行业名").agg(tj_cols).reset_index()
    s_df.columns = [a for a,b  in s_df.columns ]
    s_df["ge_20ma_n"] = pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["是否大于20均线"]>0].shape[0])).iloc[:,-1]
    s_df["ge_60ma_n"] = pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["是否大于60均线"]>0].shape[0])).iloc[:,-1]
    s_df["ge_120ma_n"] = pd.DataFrame(df.groupby("行业名",as_index=False).apply(lambda x :x[x["是否大于120均线"]>0].shape[0])).iloc[:,-1]
    s_df["60强度"] = s_df["ge_20ma_n"]*100 /s_df["code"]
    s_df["20强度"] = s_df["ge_60ma_n"]*100 /s_df["code"]
    s_df["120强度"] = s_df["ge_120ma_n"]*100 /s_df["code"]

    # s_df["每日alpha强度_n"] = pd.DataFrame(df.groupby("行业名", as_index=False).apply(lambda x: x[x["1强度_指数"] > 0].shape[0])).iloc[:, -1]
    # s_df["60alpha强度_n"] = pd.DataFrame(df.groupby("行业名", as_index=False).apply(lambda x: x[x["60强度_指数"] > 0].shape[0])).iloc[:, -1]
    # s_df["每日alpha%"] = s_df["每日alpha强度_n"] * 100 / s_df["code"]
    # s_df["60alpha%"] = s_df["60alpha强度_n"] * 100 / s_df["code"]
    # s_df.rename(columns = {"code":"股票数",'1强度_指数':"每日alpha_mean", '60强度_指数':'60alpha_mean'},inplace=True)
    s_df.rename(columns = {"code":"股票数",
                           '是否大于20均线':"大于20均线_平均天数", '是否大于60均线':"大于60均线_平均天数", '是否大于120均线':"大于120均线_平均天数",
                           'ge_20ma_n':"大于20均线_个数", 'ge_60ma_n':"大于60均线_个数", 'ge_120ma_n':"大于120均线_个数",
                           },inplace=True)


    s_df = s_df[['行业名','股票数', 'close', '大于20均线_平均天数', '大于60均线_平均天数', '大于120均线_平均天数',
            '大于20均线_个数', '大于60均线_个数', '大于120均线_个数', '60强度', '20强度','120强度']].round(2)
    # s_df = s_df.round(2)
    s_df.sort_values("60强度", ascending=False, inplace=True)

    return s_df

# 市场滚动行业强度
def cal_scqd(df):
    # print(df.tail())
    df_tjl0 = df.copy()
    df_tjl0["date0"] = pd.to_datetime(df_tjl0["date"])
    df_tjl0 = df_tjl0.sort_values(by = "date0",ascending=True)
    sckd = {}
    for col in ["是否大于20均线", "是否大于60均线", "是否大于120均线"]:
        df_tjl = pd.DataFrame()

        for k in df_tjl0["date"].unique():
            v = df_tjl0[df_tjl0["date"]==k]
            df000 = v.groupby("industry").agg({"industry": ["count"], col: ["sum"]})  # ,"是否大于60均线":["sum"],"是否大于120均线":["sum"]
            df000.columns = [a for a, b in df000.columns]
            df000[col] = 100 * df000[col] / df000["industry"]
            df_tjl[k] = df000[col].copy()
        # df_tjl = df_tjl.sort_index(axis=1,ascending=True)
        # print(df_tjl.tail())

        df_tjl = df_tjl.sort_values(by=df_tjl.columns[-1], ascending=True)
        df_tjl["行业股票数"] = df000["industry"]*100/df000["industry"].sum()
        df_tjl = df_tjl.iloc[:, -20:].copy()
        df_tjl.loc["每日强度"] = df_tjl.sum()/len(df000["industry"])

        # df_tjl.loc["每日强度","行业股票数"] = df_tjl["行业股票数"].sum()
        # df_tjl.columns =["行业股票数"] + [i for i in  df_tjl.keys() if i != "行业股票数"]
        df_tjl = df_tjl.reset_index()
        sckd[col] = df_tjl.round(2)
    return sckd

def get_a_scqd_table(df,id=""):
    table0 = dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_data_conditional=reduce(lambda x, y: x + y,[data_bars2(df.iloc[:-2], c)
  for c in df.columns if c not in ["industry","行业股票数"]]),
        style_header={'backgroundColor': 'rgb(240, 200, 200)', 'font-size': 14,
                      'fontWeight': '450', 'border': '2px solid black', "height": 50},
        style_cell={'textAlign': 'right', 'font-size': 17,
                    'minWidth': '60px', 'width': '70px', 'maxWidth': '70px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'border': '2px solid grey',
                    "color": 'black'},
        fixed_columns={"headers": True, "data": 1},
        style_table={'minWidth': '100%', "left": 10, "top": 10, "margin": 10, 'overflowY': 'auto'})
    return table0

def get_a_heat_map(df,id):
    import plotly.io as pio
    # if os.path.exists(f'{id}.png'):
    #     html.Img(src = f'{id}.png')
    #     return html.Img(src = f'{id}.png')
    df["industry"].fillna("未分类",inplace = True)
    df = df.set_index(keys="industry",drop=True)
    fig = ff.create_annotated_heatmap(name='数据',
        x=[str(i)[5:].replace("-","月")for i in list(df.columns) if i !="行业股票数"]+["行业股票数"],
        y=[str(i) for i in df.index],
        z=df.values,
        colorscale=[[0, 'rgb(102, 153, 204)'],
                    [0.7, 'rgb(255, 87, 48)'],
                    [1.0, 'rgb(255, 27, 58)']],
        )
    # print(fig.layout)
    # exit()
    fig.update_layout(title_text=f'市场强度--{id}',height = 1000,)
    # pio.write_image(fig, f'{id}.png')
    graph = dcc.Graph(id=id,figure=fig)

    return graph

def init_all_data():

    global df, df0, df_h0, df_h, zxg_df0, s_df, seleced_table, thszxg_df, zxg_df, H_s_df, cols, choosed_cols,sckd,sckd_h

    # 加载df0，A股数据
    print("加载最新数据——to_web")
    df = pd.read_csv(r'data\all_240_new_data_数据.csv')
    df["date0"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date0", ascending=True)
    df0 = df[df["date0"] == df["date0"].unique()[-1]]
    # print(df0.keys())
    df0 = df0[(df0["是否交易"] == 1) & (df0["是否是st股"] == 0)]
    df0 = df0[~(df0["code_name"].str.contains("退"))]
    df0 = df0[~(df0["code_name"].str.contains("ST"))]

    # 加载df_h0，H股数据
    df_h0 = pd.read_csv(r"data\H股_show_table.csv")
    df_h0["date0"] = pd.to_datetime(df_h0["date"])
    df_h0 = df_h0.sort_values(by="date0", ascending=True)
    df_h0 = df_h0[df_h0["date0"] >= pd.to_datetime("2020-12-20")].reset_index()
    df_h0 = df_h0.round(3)
    df_h0["code"] = df_h0["code"].astype(str)
    df_h0["code"] = df_h0["code"].str.zfill(5)
    df_h = df_h0[df_h0["date0"] == df_h0["date0"].unique()[-1]].reset_index()
    df_h = df_h[['date', 'code', 'name', 'industry', 'MA120', 'MA20', 'MA60', 'close', '是否大于20均线', '是否大于60均线', '是否大于120均线']]

    # 计算A市场强度
    df_tjl0 = df[["date", "code_name", "industry", "是否大于20均线", "是否大于60均线", "是否大于120均线"]]
    df_tjl0[["是否大于20均线", "是否大于60均线", "是否大于120均线"]] = df_tjl0[["是否大于20均线", "是否大于60均线", "是否大于120均线"]].applymap(lambda x: 1 if x > 0 else 0)
    sckd = cal_scqd(df=df_tjl0)

    # 计算H市场强度
    df_h_tjl0 = df_h0[["date", "name", "industry", "是否大于20均线", "是否大于60均线", "是否大于120均线"]]
    df_h_tjl0[["是否大于20均线", "是否大于60均线", "是否大于120均线"]] = df_h_tjl0[["是否大于20均线", "是否大于60均线", "是否大于120均线"]].applymap(lambda x: 1 if x > 0 else 0)
    sckd_h = cal_scqd(df=df_h_tjl0)

    df_h = df_h.rename(columns={'name': "code_name", 'industry': "行业名"})
    H_s_df = H_hy_qiangdu_productor(df_h)

    thszxg_df = pd.read_csv(r'data/同花顺_自选股.csv')
    # 生成带自选股的df
    df = get_all_codes_with_zxg(df0, thszxg_df)
    cols = ['date', 'code', 'code_name', '行业名', '涨跌幅', "1指数涨跌幅", "滚动市盈率", "市净率",
            '大于60线', '大于20线', '大于120线', '1强度_指数', '60强度_指数']
    choosed_cols = []
    # 生成自选股
    zxg_df0 = get_zxg_df(df)

    # 生成带对应行业强度
    s_df = hy_qiangdu_productor(df=df)
    seleced_table = pd.DataFrame(s_df)
    # 自选股行业强度
    zxg_df = hy_qiangdu_productor(df=zxg_df0)





server = Flask(__name__)
app = dash.Dash(__name__, server=False)
scheduler = APScheduler()

server.config['SCHEDULER_API_ENABLED'] = True

scheduler.init_app(server)
app.init_app(server)
scheduler.start()


base_table_filters = []
hy_filter_cols = []
global df, df0, df_h0, df_h, zxg_df0, s_df, seleced_table, thszxg_df, zxg_df, H_s_df, cols, choosed_cols, sckd, sckd_h

init_all_data()


store_mem = dcc.Store(id = "store_data",data={"st_hy_cols":[]})
store_mem2 = dcc.Store(id = "store_data2",data={"st_hy_cols":[]})
store_mem_zxg = dcc.Store(id = "store_data_zxg",data={"zxg_new":[]})

A_shichangqd = html.Div(id="A_scqd",children = [
                                    # get_a_heat_map(df= sckd["是否大于20均线"],id="A_20ma_gd"),
                                    #  get_a_heat_map(df= sckd["是否大于60均线"],id="A_60ma_gd"),
                                    #  get_a_heat_map(df= sckd["是否大于120均线"],id="A_120ma_gd")
                        # html.Div(id='scqd20',children = [
                        #                                     get_a_heat_map(df= sckd["是否大于20均线"],id="A_20ma_gd")],
                        #                                     # get_a_scqd_table(df=sckd_h["是否大于20均线"],id="h_scqd20_table")],
                        #                                 # style={'width': 1800,'height': 1500, "left":50,"position":"absolute","align":"center"  },
                        #          ),
                        # html.Div(id='scqd60',children = [
                        #                                     get_a_heat_map(df= sckd["是否大于60均线"],id="A_60ma_gd")],
                        #                                     # get_a_scqd_table(df=sckd_h["是否大于20均线"],id="h_scqd20_table")],
                        #                                 # style={'width': 1800,'height': 1500, "left":50,"position":"absolute","align":"center"  },
                        #          ),
                        # html.Div(id='scqd120',children = [
                        #                                     get_a_heat_map(df= sckd["是否大于120均线"],id="A_120ma_gd")],
                        #                                 # get_a_scqd_table(df=sckd_h["是否大于20均线"],id="h_scqd20_table")],
                        #                                 # style={'width': 1800,'height': 1500, "left":50,"position":"absolute","align":"center"  },
                        #          )
                        ])

Astocks = html.Div([

            dcc.Link(id="refresh", children = ["点击重载"],refresh =True,href="www.a667.com:666",style={'width': 1550,'height': 30,"left":1630,"top":100,"position":"absolute","font-size":18}),

            html.Div(id = "Checklist_base_col",children=[
                        # dcc.Dropdown(id = "base_col",options=[{'label': i, 'value': i} for i in  [i for i in df.columns if i not in cols]] ,
                        #     value=[],multi=True,placeholder=" ---观察项列，选择框--- ",
                        #     style={'width': 600, 'height': 25, "left": 150, "top": 5, "position": "absolute",})
                dcc.Checklist(id = "base_col",
                 options=[{'label': i+"__", 'value': i} for i in  [i for i in df.columns if i not in cols] ],value=[],
                style = {
                         'fontWeight': '450',
                          "font-size":17,
                          "text-align": "center",
                         })
            ],
                style={'width': 1020,'height': 60, "left":55,"top":105,"position":"absolute",
                                                "word-break":"break-all",
                                                "word-spacing":"10px",
                                              'backgroundColor': 'rgb(50, 155, 30,0.1)','fontWeight': 'bold',
                                              'border': '1px solid rgb(220, 200, 120,0.5)',"text-align":"center",
                                              "border-radius":"5px"},),
            html.Div(id = "condition_div_show",
                     style={'width': 1550,'height': 30,"left":130,"top":180,"position":"absolute","font-size":18}),
            #  base_table
            html.Div([dash_table.DataTable(
        id='base_table',
        columns=[{"name": i, "id": i} for i in df.columns if i in cols+choosed_cols],
        # dropdown={"行业名": {"clearable" : True, "options" : [ {'label': k, 'value': k} for k in df['行业名'].unique()]}},
        loading_state ={"is_loading ":True},
        data=df.to_dict('records'),
        style_data_conditional=[{'if': {'row_index': 'even'},'backgroundColor': 'rgb(255, 255, 210,1)'}]
                               + data_bars(df, '1强度_指数') + data_bars(df, '60强度_指数')+
                               data_bars(df, '大于20线')+data_bars(df, '大于60线')+data_bars(df, '大于120线'),
                style_data={'if': {'row_index': 'odd'},'backgroundColor': 'rgb(208, 248, 248)'},
        style_header={'backgroundColor': 'rgb(250, 220, 220)','fontWeight': 'bold','border': '2px solid black',
                      'textWrap': 'normal','height': 'auto',
                      },
        style_cell={'textAlign': 'center',
                    "font-size":15,'fontWeight': '430',
                        'width': '70px','minWidth': '40px',  'maxWidth': '70px',
                        "height":35,
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'border': '2px solid black',
                        "color":'black'},
        # tooltip_header={i : {'value':i,'type': 'markdown' } for i in df.columns if i in cols+choosed_cols},
        # tooltip_data=[{column: {'value': str(column)+"\t = \t"+str(value), 'type': 'text',"duration":1000*5 }
        #                for column, value in row.items()} for row in df.to_dict('records')],
        fixed_rows={'headers': True},
        filter_action="custom",sort_action="custom",
        sort_mode="multi",
        column_selectable="multi",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        selected_cells=[],
        page_action="native",
        page_current= 0,
        page_size= 50,
        export_format='xlsx',
        export_headers='display',
        # fixed_columns={'headers': True,"data":4},
        style_table={'width': '1500px','height': 590,"left":20,"top":10,
                     },
            )],id = "base_table_div",
                style={'width': 1550,'height': 600, "left":50,"top":175,"position":"absolute",
                       'backgroundColor': 'rgb(151, 181, 151,0.1)', 'fontWeight': '410',
                       "font-size": 17, 'border': '1px solid rgb(220, 200, 120,0.5)',
                       "text-align": "center", "border-radius": "4px"},),
            # 雪球
            html.Div([dcc.Link(id = "Astock_link", children=[f" 没有选择股票:  雪球行情页面"],href="https://xueqiu.com/hq",
                               refresh=False,target="_blank")],id = "xq_link",
                      style={'width': 350,'height': 30, "left":75,"top":795,"position":"absolute","font-size":20,
                             'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
                             # 'border': '1px solid rgb(220, 200, 120,0.5)',
                             "text-align": "center", "border-radius": "5px"
                             }),
            # 同花顺财务
            html.Div([dcc.Link(id = "Astock_cw_link", children=[f" 财务查看:  同花顺页面"],href="http://q.10jqka.com.cn/",
                               refresh=False,target="_blank")],id = "ths_cw_link",
                      style={'width': 350,'height': 30, "left":450,"top":795,"position":"absolute","font-size":20,
                             'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
                             # 'border': '1px solid rgb(220, 200, 120,0.5)',
                             "text-align": "center", "border-radius": "5px"
                             }),
            # 问财行业对比
            html.Div([dcc.Link(id = "Astock_wencai_link", children=[f" 行业及产品的上下游:  问财页面"],href="http://www.iwencai.com",
                               refresh=False,target="_blank")],id = "ths_wc_link",
                      style={'width': 350,'height': 30, "left":830,"top":795,"position":"absolute","font-size":20,
                             'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
                             # 'border': '1px solid rgb(220, 200, 120,0.5)',
                             "text-align": "center", "border-radius": "5px"
                             }),
            # 行业下拉选择框
            html.Div(id = "Dropdown_hy",children=[
                dcc.Dropdown(id = "hy_dd",options=[{'label': i, 'value': i} for i in s_df["行业名"].tolist()],
                            value=[],multi=True,placeholder=" 行业选择框 ：      默认全选  ",
                             style={"font-size":18,'fontWeight': 500,'border': '1px solid rgb(220, 100, 120,0.5)', "border-radius": "5px"}
                             )],
                    style={'width': 500,'height': 40, "left":1230,"top":792,"position":"absolute",}),
            #des_div
            html.Div(id="base_des_div",style={'width': 200,'height': 360, "left":1630,"top":185,"position":"absolute",
                                              'backgroundColor': 'rgb(151, 151, 231,0.3)','fontWeight': '420',"font-size":12,
                                              'border': '1px solid rgb(220, 200, 120,0.5)',"text-align":"center",
                                              "box-shadow": "5px -3px 3px #898888",
                                              "border-radius":"3px"}),
        # des_div
        html.Div(id="info_tip_div", children=["复合筛选操作指南：",html.Br(),"数字操作 ：>0+<10 ",html.Br(),"字符串操作 ：银行*保险"],style={'width': 200, 'height': 200, "left": 1630, "top": 555, "position": "absolute",
                                           'backgroundColor': 'rgb(251, 251, 131,0.15)', 'fontWeight': '420', "font-size": 15,
                                           'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "center",
                                           "box-shadow": "5px -3px 3px #898888",
                                           "border-radius": "3px"}),

            #hy_table
            html.Div(id='hy_table',children = [dash_table.DataTable(
        id='hy_table0',
        columns=[{"name": i, "id": i, "deletable": False, "selectable": False} for i in s_df.columns if i != 'id'],
        data=s_df.to_dict('records'),
        style_data_conditional=[{'if': {'row_index': 'even'},'backgroundColor': 'rgb(248, 248, 228,1)'}]+
        data_bars(s_df, '60强度')+data_bars(s_df, '20强度')+data_bars(s_df, '120强度')+
        data_bars(s_df, '每日alpha%')+data_bars(s_df, '60alpha%')+
        data_bars(s_df, '每日alpha_mean')+data_bars(s_df, '60alpha_mean'),
        style_data={'if': {'row_index': 'odd'},'backgroundColor': 'rgb(228, 248, 248,1)'},

        style_header={'backgroundColor': 'rgb(230, 200, 200)','fontWeight': 'bold','border': '2px solid red',"height":40},
        style_cell_conditional=[
            {
                'if': {'column_id': '行业名'},
                'width': '70px'
            },
        ],
        style_cell={'textAlign': 'center',
                    "font-size":15,'fontWeight': '460',
                        'width': '60px', 'maxWidth': '60px','minWidth': '60px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'border': '2px solid grey',
                        "color":'black'},
        tooltip={i: {'value': i, 'use_with': 'both'} for i in s_df.columns},
        filter_action="custom",sort_action="custom",
        sort_mode="multi",
        column_selectable="multi",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 100,
        # fixed_columns={'headers': True, "data": 1},
        style_table={'width': 800,"top": 15,"left": 10,'overflowX': 'auto'},#
            )],
             style={'width': 830, 'height': 1000, "left": 50, "top": 850, "position": "absolute",
                    'backgroundColor': 'rgb(100, 155, 130,0.1)', 'fontWeight': 'bold',
                    'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "center",
                    "border-radius": "5px"
                    },),
            #selected_table
            html.Div(id='selected_table',children = [dash_table.DataTable(
            id='selected_table0',
            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in seleced_table.columns if i != 'id'],
            # fixed_columns={'headers': True,"data":1},
            data=seleced_table.to_dict('records'),
            style_data_conditional=data_bars(seleced_table, '60强度')+data_bars(seleced_table, '20强度')+data_bars(seleced_table, '120强度')+
                data_bars(seleced_table, '每日alpha%')+data_bars(seleced_table, '60alpha%')+data_bars(seleced_table, '每日alpha_mean')+data_bars(seleced_table, '60alpha_mean'),
            style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
            style_cell={'textAlign': 'center',"font-size":15,'fontWeight': '450',
                        'minWidth': '60px', 'width': '70px', 'maxWidth': '120px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'border': '2px solid grey',
                        "color": 'black'},
            tooltip={i: {'value': i, 'use_with': 'both'} for i in seleced_table.columns},
            sort_mode="multi",
            filter_action="custom", sort_action="custom",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=100,
            style_table={'width': 900, "left":25,"top":15},
            )],
                           style={'width': 950  ,'height': 1000, "left":900,"top":850,"position":"absolute",
                                  'backgroundColor': 'rgb(100, 155, 130,0.1)', 'fontWeight': 'bold',
                                  'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "center",
                                  "border-radius": "5px"
                                  },),

            html.Div(id='A_scgdqd_graph',children=[get_a_heat_map(df=sckd["是否大于20均线"],id="A_20ma_gd0")],
                           style={'width': 1500,'height': 200, "left":150,"top":1900,"position":"absolute",
                                  "border-radius":"5px"},),
             html.Div(id="scqd_tip_div", children=["市场强度20MA指南：",html.Br(),html.Br(),
                                                   "单个行业,从左到右,反应近期的资金热度。",html.Br(),
                                                   "整体，从左往右，反应市场资金在行业之间的流动，又称行业滚动。",html.Br(),html.Br(),
                                                   "制作原理：",html.Br(),html.Br(),
                                                   "统计每个行业个股大于MA20的个数，计算%。",html.Br(),
                                                   "MA20线反应市场短期资金成本，当价格在其上，"
                                                   "代表短期资金看好。容易形成短期趋势。",html.Br(),
                                                   "短期趋势形成会扩散成中期趋势。",html.Br(),
                                                   "趋势：资金成本不断抬高的过程。"],
             style={'width': 230, 'height': 600, "left": 1600, "top": 2000, "position": "absolute",
                    'backgroundColor': 'rgb(151, 51, 231,0.1)', 'fontWeight': '420', "font-size": 17,
                    'border': '2px solid rgb(220, 200, 120,0.5)', "text-align": "left",
                    "box-shadow": "5px -3px 3px #898888",
                    "border-radius": "3px"}),
])

H_shichangqd = html.Div(id="H_scqd",children = [
                            # html.Div(id='h_scqd20',children = [
                            #         get_a_heat_map(df= sckd_h["是否大于20均线"],id="H_20ma_gd")],),
                            # html.Div(id='h_scqd60',children = [
                            #                                     get_a_heat_map(df= sckd_h["是否大于60均线"],id="H_60ma_gd")],),
                            # html.Div(id='h_scqd120',children = [
                            #                                     get_a_heat_map(df= sckd_h["是否大于120均线"],id="H_120ma_gd")],),
                            ])

Hstocks =html.Div([
            html.Div(id='H_main_div',children = [
        dash_table.DataTable(
                            id='df_h_all',
                            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_h.columns],
                            data=df_h.to_dict('records'),
                            style_data_conditional=data_bars(df_h, '是否大于20均线') +data_bars(df_h, '是否大于60均线') +
                                                    data_bars(df_h, '是否大于120均线') ,
                            style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
                            style_cell={'textAlign': 'right',
                                        'minWidth': '40px', 'width': '70px', 'maxWidth': '150px',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'border': '2px solid grey',
                                        "color": 'black'},
                            fixed_rows={"headers":True},
                            tooltip={i: {'value': i, 'use_with': 'both'} for i in df_h.columns},
                            sort_mode="multi",
                            filter_action="custom", sort_action="custom",
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            selected_cells=[],
                            page_action="native",
                            page_current=0,
                            page_size=50,
                            style_table={"width": 1450, "height": 640, "left": 20, "margin": 10,
                         'overflowY': 'auto'})],
                            style={'width': 1500,'height': 650, "left":80,"top":180,"position":"absolute"},),
            html.Div([dcc.Link(id = "Hstock_link", children=[f" 当前没有选择股票:  雪球行情页面"],href="https://xueqiu.com/hq",
                               refresh=False,target="_blank")],id = "xq_link2",
                      style={'width': 350,'height': 30, "left":105,"top":765,"position":"absolute","font-size":20,
                             'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
                             # 'border': '1px solid rgb(220, 200, 120,0.5)',
                             "text-align": "center", "border-radius": "5px"
                             }),
            html.Div([dcc.Link(id = "Hstock_cw_link", children=[f" 财务查看:  同花顺页面"],href="http://q.10jqka.com.cn/hk/indexYs/",
                               refresh=False,target="_blank")],id = "ths_cw_link2",
                      style={'width': 350,'height': 30, "left":505,"top":765,"position":"absolute","font-size":20,
                             'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
                             # 'border': '1px solid rgb(220, 200, 120,0.5)',
                             "text-align": "center", "border-radius": "5px"
                             }),
            html.Div(id = "hk_Dropdown_hy",children=[
                            dcc.Dropdown(id = "hk_hy_dd",options=[{'label': i, 'value': i} for i in H_s_df["行业名"].tolist()],
                            value=[],multi=True,placeholder=" 行业选择框 ：      默认全选  ",
                             style={"font-size":18,'fontWeight': 500,'border': '1px solid rgb(220, 100, 120,0.5)', "border-radius": "5px"}
                             )],
                    style={'width': 350,'height': 30, "left":900,"top":765,"position":"absolute",}),

            html.Div(id="H_main_des_div",style={'width': 200,'height': 360, "left":1600,"top":190,"position":"absolute",
                                              'backgroundColor': 'rgb(151, 151, 231,0.3)','fontWeight': '420',"font-size":12,
                                              'border': '1px solid rgb(220, 200, 120,0.5)',"text-align":"center",
                                              "box-shadow": "5px -3px 3px #898888",
                                              "border-radius":"3px"}),

            html.Div(id='H_hy_div',children = [
                    dash_table.DataTable(
                        id='H_hy_table0',
                        columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in H_s_df.columns],
                        data=H_s_df.to_dict('records'),
                        style_data_conditional=data_bars(H_s_df, '60强度')+data_bars(H_s_df, '20强度')+data_bars(H_s_df, '120强度') ,
                        style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
                        style_cell={'textAlign': 'right',
                                    'minWidth': '50px', 'width': '60px', 'maxWidth': '80px',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'border': '2px solid grey',
                                    "color": 'black'},
                        tooltip={i: {'value': i,'use_with': 'both'} for i in H_s_df.columns},
                        sort_mode="multi",
                        filter_action="custom", sort_action="custom",
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current=0,
                        page_size=100,
                        style_table={"width": 850, "left": 20, "margin": 10,
                             'overflowY': 'auto'})],
                       style={'width': 900,'height': 1000, "left":80,"top":845,"position":"absolute"},),

            html.Div(id='H_seleced_div',children = [
                    dash_table.DataTable(
                    id='hk_seleced',
                    columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in H_s_df.columns],
                    data=H_s_df.to_dict('records'),
                    style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
                    style_header={'backgroundColor': 'rgb(240, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
                    style_cell={'textAlign': 'right',
                                'minWidth': '50px', 'width': '60px', 'maxWidth': '80px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'border': '2px solid grey',
                                "color": 'black'},
                    sort_mode="multi",
                    filter_action="custom",
                    sort_action="custom",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=50,
                    style_table={"width": 850, "height": 1000, "left": 20,"top":0, "margin": 10})],
                           style={'width': 900,'height': 1000, "left":950,"top":845,"position":"absolute","align":"center"  },),
            html.Div(id='H_scgdqd_graph',children=[get_a_heat_map(df= sckd_h["是否大于20均线"],id="H_20ma_gd0")],
                           style={'width': 1500,'height': 200, "left":150,"top":1900,"position":"absolute",
                                  "border-radius":"5px"},),
            html.Div(id="H_scqd_tip_div", children=["市场强度20MA指南：",html.Br(),html.Br(),
                                                   "单个行业,从左到右,反应近期的资金热度。",html.Br(),
                                                   "整体，从左往右，反应市场资金在行业之间的流动，又称行业滚动。",html.Br(),html.Br(),
                                                   "制作原理：",html.Br(),html.Br(),
                                                   "统计每个行业个股大于MA20的个数，计算%。",html.Br(),
                                                   "MA20线反应市场短期资金成本，当价格在其上，"
                                                   "代表短期资金看好。容易形成短期趋势。",html.Br(),
                                                   "短期趋势形成会扩散成中期趋势。",html.Br(),
                                                   "趋势：资金成本不断抬高的过程。"],
             style={'width': 230, 'height': 600, "left": 1600, "top": 2000, "position": "absolute",
                    'backgroundColor': 'rgb(151, 51, 231,0.1)', 'fontWeight': '420', "font-size": 17,
                    'border': '2px solid rgb(220, 200, 120,0.5)', "text-align": "left",
                    "box-shadow": "5px -3px 3px #898888",
                    "border-radius": "3px"}),

            ])

ths_stocks = html.Div([
            store_mem_zxg,
            html.Div(dcc.Upload(id='upload_zxg',children=html.Div([html.A("拖拽文件----点击上传csv, xls, 同花顺.ini 文件")]),style={
                            'width': '1000px',
                            'height': '60px',
                            'lineHeight': '60px',
                            'border': '3px dashed rgb(245, 150, 150,0.8)',
                            'borderRadius': '5px',
                            'backgroundColor': 'rgb(240, 220, 220,0.2)',
                            "font-size":20,
                            'textAlign': 'center',
                            'margin': '10px'
                        },multiple=False),
                           style={'width': 1000,'height': 70, "left":400,"top":100,"position":"absolute"}),
            html.Div(id='zxg_all_div',children = [
        dash_table.DataTable(
                            id='zxg_all0',
                            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in zxg_df0.columns],
                            data=zxg_df0.to_dict('records'),
                            style_data_conditional=(data_bars(zxg_df0, '涨跌幅') ),
                            style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
                            style_cell={'textAlign': 'right',
                                        'minWidth': '40px', 'width': '70px', 'maxWidth': '150px',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'border': '2px solid grey',
                                        "color": 'black'},
                            tooltip={i: {'value': i, 'use_with': 'both'} for i in zxg_df0.columns},
                            sort_mode="multi",
                            filter_action="custom", sort_action="custom",
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current=0,
                            page_size=20,
                            style_table={"width": 1400, "height": 500, "left": 20, "margin": 10,
                         'overflowY': 'auto'})],
                            style={'width': 1450,'height': 800, "left":100,"top":200,"position":"absolute"},),
            html.Div(id='zxg_table',children = [
        dash_table.DataTable(
                    id='zxg_table0',
                    columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in zxg_df.columns],
                    data=zxg_df.to_dict('records'),
                    style_data_conditional=(
                    data_bars(zxg_df, '60强度')+data_bars(zxg_df, '20强度')+data_bars(zxg_df, '120强度') ),
                    style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
                    style_cell={'textAlign': 'right',
                                'minWidth': '40px', 'width': '70px', 'maxWidth': '150px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'border': '2px solid grey',
                                "color": 'black'},
                    tooltip={i: {'value': i,'use_with': 'both'} for i in zxg_df.columns},
                    sort_mode="multi",
                    filter_action="custom", sort_action="custom",
                    row_deletable=False,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=100,
                    style_table={"width": 1400, "height": 600, "left": 20, "margin": 10,
                         'overflowY': 'auto'})],
                           style={'width': 1500,'height': 600, "left":100,"top":800,"position":"absolute"},),
            html.H4(id="zxg_update_info",children=["当前自选股样本"],style={"left":1650,"top":150,"position":"absolute","align":"center"}),
            html.Div(id='zxg_look',children = [
                    dash_table.DataTable(
                    id='zxg_look0',
                    columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in thszxg_df.columns],
                    data=thszxg_df.to_dict('records'),
                    style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
                    style_header={'backgroundColor': 'rgb(240, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
                    style_cell={'textAlign': 'right',
                                'minWidth': '50px', 'width': '70px', 'maxWidth': '100px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'border': '2px solid grey',
                                "color": 'black'},
                    page_action="native",
                    page_current=0,
                    page_size=50,
                    style_table={"width": 300, "height": 800, "left": 20,"top":20, "margin": 10,'overflowY': 'auto'})],
                           style={'width': 330,'height': 800, "left":1550,"top":200,"position":"absolute","align":"center"  },),
            ])

app.layout = html.Div([
dcc.Store(id = "store_dingshi_triger",data={"ds":[9,30]}),

                        store_mem,
                        store_mem2,
                        dcc.Interval(id="dingshi_tool",interval=10*1000,),
                        dcc.Tabs(id="tabs_main",value = "A股观察",children =[
                            dcc.Tab(label='A股观察', value='A股观察',children=[Astocks],),
                            dcc.Tab(label='H股观察', value='H股观察', children=[Hstocks],),
                            dcc.Tab(label='A股逐日-滚动强度', value='A股滚动强度', children=[A_shichangqd],),
                            dcc.Tab(label='H股逐日-滚动强度', value='H股滚动强度', children=[H_shichangqd],),
                            dcc.Tab(label='同花顺自选股观察', value='同花顺自选股观察', children=[ths_stocks])],),
                        html.Hr(id="hr",style={"color":"#ff6699"}),
])




#1   filter_base_table
@app.callback([Output('base_table', "data"),Output('base_table', "columns"),Output('condition_div_show', "children")],
                [Input('base_table', "filter_query"),
                 Input('base_table', "sort_by"),
                 Input('base_col', "value"),
                 Input("store_data","data"),

                 Input("store_data_zxg","data"),
                 ])
def filter_base_table(filter,sort_by,base_col,s_data,zxgdata):
    global df,base_table_filters,hy_filter_cols,cols,choosed_cols
    ctx = dash.callback_context
    # print(ctx.triggered[0]['prop_id'])
    dff = pd.DataFrame(df)
    dff= dff[[i for i in df.columns if i in cols+base_col]]
    choosed_cols  = base_col
    new_filter0=[]
    if filter !=None:
        if filter not in ["",''] :
            new_filter0 = filter.split("&&")
    # new_filter = deal_hycols(filter)
    new_filter = new_filter0+s_data["st_hy_cols"]
    # print("base_recieve:",new_filter,123,sort_by,(len(new_filter)==0) , (sort_by==None))
    if (len(new_filter)==0) and (sort_by==None):
        dff.to_dict('records')

        return dff.to_dict('records'),[{"name": i, "id": i} for i in dff.columns ],["当前筛选条件：无"]
    if (len(new_filter)>0):

        new_filtering_expressions = split_all_filter(new_filter)
        # print("new_filtering_expressions_recieve:", filtering_expressions)

        for filter_part in new_filtering_expressions:

            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                if "*" not in filter_value:
                    dff = dff.loc[(dff[col_name].str.contains(str(filter_value).strip(" ")))|
                                   (dff[col_name]==(str(filter_value).strip(" ")))]
                else:
                    con_str = [fv0.strip(" ") for fv0 in filter_value.split("*")]
                    dff = dff.loc[dff[col_name].isin(con_str)]

            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(str(filter_value))]

    if (sort_by != None):
        if len(sort_by) > 0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction']=='desc' for col in sort_by]
            dff0 = dff.sort_values(by = sort_list ,ascending=sort_func,inplace=False)
            show_df0 = dff0
        else:show_df0 = dff
    else:show_df0 = dff
    dff = show_df0
    new_filter_show = [html.Label(i+"  ",style={'backgroundColor': 'rgb(251, 151, 231,0.3)','fontWeight': '410',
                                                "font-size":17,'border': '1px solid rgb(220, 200, 120,0.5)',
                                                "text-align":"center","border-radius":"2px"}) for i in new_filter0 ]
    new_filter_show0 =[]
    for c in new_filter_show:
        new_filter_show0.append(c)
        new_filter_show0.append(" ， ")

    condition_show =["当前筛选条件："]+new_filter_show0
    return dff.to_dict('records'),[{"name": i, "id": i} for i in dff.columns ],condition_show

#2   filter_hangye_table
@app.callback(Output('hy_table0', "data"),
                [Input('hy_table0', "filter_query"),
                Input('hy_table0', "sort_by"),
                 Input("store_data_zxg","data")])
def filter_hangye_table(filter,sort_by,zxgdata):
    global df,s_df

    dff = pd.DataFrame(s_df)
    # print(filter,sort_by)
    if (filter == None) & (sort_by == None):
        return s_df.to_dict('records')
    if (filter != None):
        filtering_expressions = filter.split('&&')
        new_filtering_expressions = split_all_filter(filtering_expressions)

        for filter_part in new_filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]
        # show_df0 = dff
    if (sort_by != None):
        if len(sort_by) > 0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction'] == 'desc' for col in sort_by]
            dff0 = dff.sort_values(by=sort_list, ascending=sort_func, inplace=False)
            # print(sort_list,'\n',sort_func,dff0)
            show_df0 = dff0
        else:
            show_df0 = dff
    else:
        show_df0 = dff
    return show_df0.to_dict('records')

#4   filter_zxg_all
@app.callback(Output('zxg_all0', "data"),
                [Input('zxg_all0', "filter_query"),
                Input('zxg_all0', "sort_by"),
Input("store_data_zxg","data")
                 ])
def filter_zxg_all(filter,sort_by,zxgdata):
    global zxg_df0
    dff = pd.DataFrame(zxg_df0)
    # print(filter, sort_by)

    if (filter==None) & (sort_by==None):
        return dff.to_dict('records')
    if (filter != None):
        filtering_expressions = filter.split('&&')
        # new_filtering_expressions = []
        new_filtering_expressions = split_all_filter(filtering_expressions)
        for filter_part in new_filtering_expressions:


            # print(filter_part)
            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if (sort_by!=None) :
        if len(sort_by) >0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction']=='desc' for col in sort_by]
            dff0 = dff.sort_values(by = sort_list ,ascending=sort_func,inplace=False)
            # print(sort_list,'\n',sort_func,dff0)
            show_df0 = dff0

        else:show_df0 = dff
    else:show_df0 = dff

    return show_df0.to_dict('records')

#5   filter_zxg_table
@app.callback(Output('zxg_table0', "data"),
                [Input('zxg_table0', "filter_query"),
                Input('zxg_table0', "sort_by"),Input("store_data_zxg","data")])
def filter_zxg_table(filter,sort_by,zxgdata):
    global zxg_df
    dff = pd.DataFrame(zxg_df)
    # print(filter, sort_by)

    if (filter==None) & (sort_by==None):
        return dff.to_dict('records')
    if (filter != None):
        filtering_expressions = filter.split('&&')
        # new_filtering_expressions = []
        new_filtering_expressions = split_all_filter(filtering_expressions)
        for filter_part in new_filtering_expressions:


            # print(filter_part)
            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if (sort_by!=None) :
        if len(sort_by) >0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction']=='desc' for col in sort_by]
            dff0 = dff.sort_values(by = sort_list ,ascending=sort_func,inplace=False)
            # print(sort_list,'\n',sort_func,dff0)
            show_df0 = dff0

        else:show_df0 = dff
    else:show_df0 = dff

    return show_df0.to_dict('records')


#6   filter_zxg_all
@app.callback(Output('df_h_all', "data"),
    [Input('df_h_all', "filter_query"),
    Input('df_h_all', "sort_by")   ],
    Input("store_data2","data"))
def filter_hk_all(filter,sort_by,hk_hy_store):
    global df_h
    dff = pd.DataFrame(df_h)
    new_filter0 = []
    if filter != None:
        if filter not in ["", '']:
            new_filter0 = filter.split("&&")
    # new_filter = deal_hycols(filter)
    new_filter = new_filter0 + hk_hy_store["st_hy_cols"]

    # print(new_filter,"---",hk_hy_store)
    # print(dff.tail())
    if (len(new_filter)==0) & (sort_by==None):
        return dff.to_dict('records')
    if (len(new_filter)!=0):
        filtering_expressions = new_filter
        new_filtering_expressions = split_all_filter(filtering_expressions)
        for filter_part in new_filtering_expressions:
            # print(filter_part)
            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                if "*" not in filter_value:
                    dff = dff.loc[(dff[col_name].str.contains(str(filter_value).strip(" "))) |
                                  (dff[col_name] == (str(filter_value).strip(" ")))]
                else:
                    con_str = [fv0.strip(" ") for fv0 in filter_value.split("*")]
                    dff = dff.loc[dff[col_name].isin(con_str)]

            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if (sort_by!=None) :
        if len(sort_by) >0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction']=='desc' for col in sort_by]
            dff0 = dff.sort_values(by = sort_list ,ascending=sort_func,inplace=False)
            # print(sort_list,'\n',sort_func,dff0)
            show_df0 = dff0

        else:show_df0 = dff
    else:show_df0 = dff

    return show_df0.to_dict('records')


#6   filter_zxg_all
@app.callback(Output('H_hy_table0', "data"),
    [Input('H_hy_table0', "filter_query"),
    Input('H_hy_table0', "sort_by")   ])
def filter_hk_hy(filter,sort_by):
    global H_s_df
    dff = pd.DataFrame(H_s_df)

    if (filter==None) & (sort_by==None):
        return dff.to_dict('records')
    if (filter != None):
        filtering_expressions = filter.split('&&')
        # new_filtering_expressions = []
        new_filtering_expressions = split_all_filter(filtering_expressions)
        for filter_part in new_filtering_expressions:


            # print(filter_part)
            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if (sort_by!=None) :
        if len(sort_by) >0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction']=='desc' for col in sort_by]
            dff0 = dff.sort_values(by = sort_list ,ascending=sort_func,inplace=False)
            # print(sort_list,'\n',sort_func,dff0)
            show_df0 = dff0

        else:show_df0 = dff
    else:show_df0 = dff

    return show_df0.to_dict('records')

#   update_seleced_table
def filter_seleced_table(df,filter,sort_by):
    dff = pd.DataFrame(df)
    if (filter == None) & (sort_by == None):
        return dff
    if (filter != None):
        filtering_expressions = filter.split('&&')
        new_filtering_expressions = split_all_filter(filtering_expressions)

        for filter_part in new_filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)
            # print(col_name, operator, filter_value)
            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]
        # show_df0 = dff
    if (sort_by != None):
        if len(sort_by) > 0:
            sort_list = [col['column_id'] for col in sort_by]
            sort_func = [col['direction'] == 'asc' for col in sort_by]
            dff0 = dff.sort_values(by=sort_list, ascending=sort_func, inplace=False)
            # print(sort_list,'\n',sort_func,dff0)
            show_df0 = dff0
        else:
            show_df0 = dff
    else:
        show_df0 = dff
    return show_df0

@app.callback(Output('base_des_div', "children"),
    [Input('base_table', "data")])
def update_des_table(df0):
    df = pd.DataFrame(df0)
    # print(df.keys())
    qd_key = [i for i in df.keys() if "20" in i ]
    declist = [
        ["行业数量:", len(df["行业名"].unique())],
        ["股票数量:", len(df["code"].unique())],
        [">20MA:", df[df["大于20线"] > 0].shape[0]],
        [">60MA:", df[df["大于60线"] > 0].shape[0]],
        [">120MA:", df[df["大于120线"] > 0].shape[0]],
        ["20MA强度:%", round(100*df[df["大于20线"] > 0].shape[0] / len(df["code"].unique()) if len(df["code"].unique()) != 0 else 0,3)],
        ["60MA强度:%", round(100*df[df["大于60线"] > 0].shape[0] / len(df["code"].unique()) if len(df["code"].unique()) != 0 else 0,3)],
        ["120MA强度:%", round(100 * df[df["大于120线"] > 0].shape[0] / len(df["code"].unique()) if len(df["code"].unique()) != 0 else 0,3)],
        ["Alpha_20", round(df[df[qd_key[0]] > 0][qd_key[0]].mean(),3)],
    ]
    des_df = pd.DataFrame(declist,columns=["统计描述：","数值大小"])
    des_df=des_df.round(2)
    des_table0 = dash_table.DataTable(
        id='des_table',
        columns=[{"name": i, "id": i, "deletable": False, "selectable": False} for i in des_df.columns if i != 'id'],
        data=des_df.to_dict('records'),
        style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(258, 258, 228,0.5)'},
        style_header={'backgroundColor': 'rgb(230, 220, 200,0.5)', 'fontWeight': 'bold', 'border': '2px solid red', "height": 50},
        style_cell={'textAlign': 'center',
                    "font-size":13,
                    'width': '70px',
                    "height":32,
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'border': '2px solid grey',
                    "color": 'black'},
        #'width': 200,'height': 320, "left":1630,"top":130,"position":"absolute",
        style_table={'width': 150,"left":25,"top":10},
        )

    return des_table0

@app.callback(Output('H_main_des_div', "children"),
    [Input('df_h_all', "data")])
def update_hk_des_table(df0):
    df = pd.DataFrame(df0)
    # print(df.keys())
    qd_key = [i for i in df.keys() if "20" in i ]
    declist = [
        ["行业数量:", len(df["行业名"].unique())],
        ["股票数量:", len(df["code"].unique())],
        [">20MA:", df[df["是否大于20均线"] > 0].shape[0]],
        [">60MA:", df[df["是否大于60均线"] > 0].shape[0]],
        [">120MA:", df[df["是否大于120均线"] > 0].shape[0]],
        ["20MA强度:%", round(100*df[df["是否大于20均线"] > 0].shape[0] / len(df["code"].unique()) if len(df["code"].unique()) != 0 else 0,3)],
        ["60MA强度:%", round(100*df[df["是否大于60均线"] > 0].shape[0] / len(df["code"].unique()) if len(df["code"].unique()) != 0 else 0,3)],
        ["120MA强度:%", round(100 * df[df["是否大于120均线"] > 0].shape[0] / len(df["code"].unique()) if len(df["code"].unique()) != 0 else 0,3)],
            ]

    des_df = pd.DataFrame(declist,columns=["统计描述：","数值大小"])
    des_df=des_df.round(2)
    des_table0 = dash_table.DataTable(
        id='des_hk_table',
        columns=[{"name": i, "id": i, "deletable": False, "selectable": False} for i in des_df.columns if i != 'id'],
        data=des_df.to_dict('records'),
        style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(258, 258, 228,0.5)'},
        style_header={'backgroundColor': 'rgb(230, 220, 200,0.5)', 'fontWeight': 'bold', 'border': '2px solid red', "height": 50},
        style_cell={'textAlign': 'center',
                    "font-size":13,
                    'width': '70px',
                    "height":32,
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'border': '2px solid grey',
                    "color": 'black'},
        #'width': 200,'height': 320, "left":1630,"top":130,"position":"absolute",
        style_table={'width': 150,"left":25,"top":10},
        )

    return des_table0


@app.callback(
    Output('selected_table0', "data"),
    [Input('base_table', "data"),
     Input('selected_table0', "filter_query"),
    Input('selected_table0', "sort_by")] )
def update_seleced_table(df0,filter,sort_by):
    global seleced_table ,df,s_df
    #
    df0 = pd.DataFrame(df0)
    # print(df.shape ,df0.shape)
    #如果和原始数据一致，那就只进行排序
    if df.shape == df0.shape:
        seleced_df = s_df
    else:
        seleced_df = hy_qiangdu_productor(df=df0)
    seleced_table = filter_seleced_table(seleced_df,filter, sort_by)
    children0 = seleced_table.to_dict('records')

    return children0


@app.callback(
    Output('hk_seleced', "data"),
    [Input('df_h_all', "data"),
     Input('hk_seleced', "filter_query"),
    Input('hk_seleced', "sort_by")] )
def update_hk_seleced_table(df0,filter,sort_by):
    global seleced_table ,df,H_s_df

    df0 = pd.DataFrame(df0)
    if df_h.shape == df0.shape:
        seleced_df = H_s_df
    else:
        H_hy_qiangdu_productor(df0)
        seleced_df = H_hy_qiangdu_productor(df=df0)
    seleced_table = filter_seleced_table(seleced_df,filter, sort_by)
    children0 = seleced_table.to_dict('records')

    return children0


#   update_styles
@app.callback([Output('base_table', 'style_data_conditional'),
               Output('base_table', 'tooltip_data'),
               Output('Astock_link', 'href'),
               Output('Astock_link', 'children'),
               Output('Astock_cw_link', 'href'),
               Output('Astock_cw_link', 'children'),
               Output('Astock_wencai_link', 'href'),
               Output('Astock_wencai_link', 'children'),

               ],
              [Input('base_table', 'data'),
               Input('base_table', 'selected_cells')])
def update_styles(data,selected_cells):
    global s_df
    df = pd.DataFrame(data)

    if len(selected_cells) > 0:
        selected_code = str(df.iloc[int(selected_cells[0]["row"])]["code"]).upper()
        selected_code =selected_code[:2]+selected_code[3:]
        new_href = f"https://xueqiu.com/S/{selected_code}"
        link_children = [f"查看股票行情-雪球网：{df.iloc[int(selected_cells[0]['row'])]['code_name']}"]
        new_href2 = f"http://stockpage.10jqka.com.cn/{selected_code[2:]}/finance/#finance"
        link_children2 = [f"查看财务报表-同花顺：{df.iloc[int(selected_cells[0]['row'])]['code_name']}"]

        new_href3 =f"http://www.iwencai.com/unifiedwap/result?w={selected_code[2:]}&querytype=&issugs"
        link_children3 = [f"行业产品上下游-问财：{df.iloc[int(selected_cells[0]['row'])]['code_name']}"]


        style_1 =[{'if': { 'column_id': selected_cells[0]["column_id"] },'background_color': '#FFF099'},
                  {'if': {'row_index': selected_cells[0]["row"]},'background_color': '#FFF099'}]
    else:
        style_1, new_href,link_children = [], f"https://xueqiu.com/hq", [f" 没有选择股票:  雪球行情页面"]
        new_href2, link_children2 =  f"http://q.10jqka.com.cn/", [f" 查看财务报表:  同花顺页面"]
        new_href3, link_children3 = f"http://www.iwencai.com", [f" 行业及产品的上下游:  问财页面"]


    style_1.extend( [{'if': {'row_index': 'even'}, 'backgroundColor': 'rgb(255, 255, 210,0.6)'}])
    style_1.extend(data_bars(df, '1强度_指数'))
    style_1.extend(data_bars(df, '60强度_指数'))
    style_1.extend(data_bars(df, '大于20线'))
    style_1.extend(data_bars(df, '大于60线'))
    style_1.extend(data_bars(df, '大于120线'))
    tooltip_data = [{column: {'value': str(column) + "\t = \t" + str(value), 'type': 'text', "duration": 1000 * 5}
                     for column, value in row.items()} for row in df.to_dict('records')]
    return style_1,tooltip_data,new_href,link_children,new_href2,link_children2,new_href3, link_children3


#   update_styles
@app.callback([Output('df_h_all', 'style_data_conditional'),
               Output('df_h_all', 'tooltip_data'),
               Output('Hstock_link', 'href'),
               Output('Hstock_link', 'children'),
               Output('Hstock_cw_link', 'href'),
               Output('Hstock_cw_link', 'children'),
               ],
              [Input('df_h_all', 'data'),
               Input('df_h_all', 'selected_cells')])
def Hupdate_styles(data,selected_cells):
    df = pd.DataFrame(data)

    if len(selected_cells) > 0:
        selected_code = str(df.iloc[int(selected_cells[0]["row"])]["code"]).upper()
        new_href = f"https://xueqiu.com/S/{selected_code}"
        link_children = [f"查看股票行情：{df.iloc[int(selected_cells[0]['row'])]['code_name']}"]

        new_href2 = f"http://stockpage.10jqka.com.cn/HK{selected_code[1:]}/finance/#index"
        link_children2 = [f"查看：{df.iloc[int(selected_cells[0]['row'])]['code_name']}，财务报表"]


        style_1 =[{'if': { 'column_id': selected_cells[0]["column_id"] },'background_color': '#FFF099'},
                  {'if': {'row_index': selected_cells[0]["row"]},'background_color': '#FFF099'}]
    else:
        style_1, new_href,link_children = [], f"https://xueqiu.com/hq", [f" 当前没有选择股票:  雪球行情页面"]
        new_href2, link_children2 =  f"http://q.10jqka.com.cn/hk/indexYs/", [f" 查看财务报表:  同花顺页面"]

    style_1.extend( data_bars(df, '是否大于20均线') +data_bars(df, '是否大于60均线') +
                                                    data_bars(df, '是否大于120均线'))

    # style_1.extend(data_bars(df, '是否大于20均线'))
    # style_1.extend(data_bars(df, '是否大于60均线'))
    # style_1.extend(data_bars(df, '是否大于120均线'))
    tooltip_data = [{column: {'value': str(column) + "\t = \t" + str(value), 'type': 'text', "duration": 1000 * 5}
                     for column, value in row.items()} for row in df.to_dict('records')]
    return style_1,tooltip_data,new_href,link_children,new_href2,link_children2



@app.callback(Output('store_data', 'data'),
              [Input('hy_dd', 'value')])
def filter_hy_to_store_data(value_dd):
    global filter_hy_cols
    filter_str = ""
    selected_hys = list(value_dd)
    for ix, r in enumerate(selected_hys):
        filter_str += r if ix == 0 else "*"+r
    filter_hy_part = ["{行业名} contains "+filter_str] if len(filter_str) > 0 else []

    filter_hy_cols={"st_hy_cols":filter_hy_part}

    return filter_hy_cols

@app.callback(Output('store_data2', 'data'),
              [Input('hk_hy_dd', 'value')])
def filter_hk_hy_to_store_data(value_dd):
    global filter_hy_cols
    filter_str = ""
    selected_hys = list(value_dd)
    for ix, r in enumerate(selected_hys):
        filter_str += r if ix == 0 else "*"+r
    filter_hy_part = ["{行业名} contains "+filter_str] if len(filter_str) > 0 else []

    filter_hy_cols={"st_hy_cols":filter_hy_part}

    return filter_hy_cols



def parse_contents(contents, filename, date):
    def parse_ini_file(f):

        # with open(r"F:\vnpy_my_gitee\company\A股票_company\StockBlock.ini", mode='r') as f:

        text = f.read()
        print("导入成功")
        # print(text)
        a = text[text.find("[BLOCK_NAME_MAP_TABLE]"):text.find("[BLOCK_STOCK_CONTEXT]")]
        # print(a)
        zxgcol = [i for i in a.split("\n") if len(i) > 0]
        zxgcontent = text[text.find("[BLOCK_STOCK_CONTEXT]"):text.find("[@7]")].split("\n")
        key = {i.split("=")[0]: i.split("=")[-1] for i in zxgcol}
        # print({i.split("=")[0]:i.split("=")[-1] for i in zxgcol})
        # print("==================")
        df_zong = pd.DataFrame()
        df_zong_list = []
        for i in zxgcontent:
            if i[:2] in key.keys():
                # print("==================")
                # print({key[i[:2]]:[v.split(":")[-1] for v in i[3:].split(",")]})
                row0 = [[v.split(":")[-1], str(key[i[:2]]).strip('\r')] for v in i[3:].split(",") if "\r" not in v.split(":")[-1]]
                # if ("+\r" in row0[-1]):
                #     row0[-1] = row0[-1].strip('\r')
                #     if ("+" in row0[-1]):
                #         row0[-1] = row0[-1].strip('+')
                # print(row0[:])
                # print(row0[0])
                # print(row0[-1])

                df_zong_list.extend(row0)
                # df_zong[key[i[:2]]] = pd.Series([v.split(":")[-1] for v in i[3:].split(",")])
                # print([v.split(":")[-1] for v in i[3:].split(",")])
        zxg = pd.DataFrame(df_zong_list, columns=["code_num", "自选股分类"])
        zxg = pd.DataFrame(zxg.groupby("code_num").apply(lambda x: ",".join(x["自选股分类"].tolist())).reset_index())
        zxg.columns = ["code_num", "自选股分类"]
        zxg = zxg.drop_duplicates("code_num", keep="first")
        zxg.to_csv(r"同花顺_自选股.csv", header=True, mode="w")
        return zxg

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif "ini" in filename:
            df = parse_ini_file(f=io.StringIO(decoded.decode('GBK')))
    except Exception as e:
        print(e)
        #html.Div(['格式错误，请上传csv，xls,同花顺.ini文件'])
        return pd.DataFrame()

    return df


@app.callback(output=[Output('zxg_look0', 'data'),
              Output("zxg_update_info","children")],
              inputs=[Input('upload_zxg', 'contents')],
              state=(State('upload_zxg', 'filename'),State('upload_zxg', 'last_modified'))
              )
def update_output(list_of_contents, list_of_names, list_of_dates):
    global thszxg_df, zxg_df0, zxg_df, s_df, seleced_table,df0,df
    if list_of_contents is not None:
        df_ini = parse_contents(list_of_contents, list_of_names, list_of_dates)
        if df_ini.empty: return thszxg_df.to_dict("records"),"当前自选股样本"
        thszxg_df = df_ini
        # 生成带自选股的df
        df = get_all_codes_with_zxg(df0, thszxg_df)
        # 生成自选股
        zxg_df0 = get_zxg_df(df)
        # 生成带对应行业强度
        s_df = hy_qiangdu_productor(df=df)
        seleced_table = pd.DataFrame(s_df)
        # 自选股行业强度
        zxg_df = hy_qiangdu_productor(df=zxg_df0)
        return df_ini.to_dict("records"),"添加成功，F5，刷新查看"
    else:
        return thszxg_df.to_dict("records"),"当前自选股样本"

@app.callback(output=[Output('A_scgdqd_graph', 'children'),
                      Output('H_scgdqd_graph', 'children'),
                      Output('A_scqd', 'children'),
                      Output('H_scqd', 'children'),],
              inputs=Input('tabs_main', 'value'),
              )
def update_tabs(tab):

    if tab == "A股观察":
        return get_a_heat_map(df=sckd["是否大于20均线"],id="A_20ma_gd0"),[],[],[]
    elif tab == "H股观察":
        return [],get_a_heat_map(df= sckd_h["是否大于20均线"],id="H_20ma_gd0"),[],[]
    elif tab == "A股滚动强度":
        return [],[],[get_a_heat_map(df= sckd["是否大于20均线"],id="A_20ma_gd"),
                                     get_a_heat_map(df= sckd["是否大于60均线"],id="A_60ma_gd"),
                                     get_a_heat_map(df= sckd["是否大于120均线"],id="A_120ma_gd")],[]
    elif tab == "H股滚动强度":
        return [],[],[],[get_a_heat_map(df= sckd_h["是否大于20均线"],id="H_20ma_gd"),
                                     get_a_heat_map(df= sckd_h["是否大于60均线"],id="H_60ma_gd"),
                                     get_a_heat_map(df= sckd_h["是否大于120均线"],id="H_120ma_gd")]
    else:
        return [],[],[],[]

@scheduler.task(trigger ='cron', id='test', day='*', hour='7', minute='15') # day='*', hour='10-11', minute='*/1', minute='10-23/5',
def dingshi_update_all_data():
        try:
            print(datetime.datetime.now())
            print("更新A股")
            update_new_ASTOCKs() #更新A股
            time.sleep(2)
            print("更新H股")
            update_h_stocks()
            time.sleep(2)
            print("更新全局变量")
            init_all_data()
            time.sleep(2)
        except Exception as e:
            print(e)



# def dingshi_update_all_data2(data):
#     hour, minute =data
#     now_time = datetime.datetime.now()
#     target_runtime = now_time.replace(hour=hour, minute=minute, second=0)
#     while True:
#         if (datetime.datetime.now().hour==hour) and (datetime.datetime.now().minute==minute):
#             try:
#                 print("更新A股")
#                 update_new_ASTOCKs() #更新A股
#                 time.sleep(2)
#                 print("更新H股")
#                 update_h_stocks()
#                 time.sleep(2)
#
#                 init_all_data()
#             except Exception as e:
#                 print(e)
#             finally:
#                 now_time = datetime.datetime.now()
#                 target_runtime = now_time.replace(day=now_time.day + 1,
#                                        hour=hour, minute=minute, second=0)
#                 time.sleep(60)
#         else:
#             now_time = datetime.datetime.now()
#             sleep_time = (target_runtime - now_time).seconds
#
#             print(f"sleep_time :=={datetime.timedelta(seconds=(sleep_time - 10))} \n")
#             print(f"target_runtime:=={target_runtime}")
#
#             if float(sleep_time) > 10:
#                 time.sleep(sleep_time - 10)
#                 time.sleep(5)
#                 print("休息5s！")
#             elif float(sleep_time) < 0:
#                 now_time = datetime.datetime.now()
#                 target_runtime = now_time.replace(day=now_time.day + 1,
#                                  hour=hour, minute=minute, second=0)
#             else:
#                 time.sleep(2)




#
# def singleton(cls):
#     instance = cls()
#     print("1",cls)
#     instance.__call__ = lambda: instance
#     return instance
# @singleton
# class run_dingshiqi:
#     def __init__(self):
#         print("开始定时器！")
#         # t1 = threading.Thread(target=self.dingshi_update_all_data, args=())
#         # t1.start()
#
#         # app.run_server(host='0.0.0.0', port=8888, debug=True)
#
#         # t1.stop()
#         # t1.join()
#
#     def dingshi_update_all_data(self):
#         while True:
#             if (datetime.datetime.now().hour == 7) and (datetime.datetime.now().minute == 5):
#                 update_new_ASTOCKs()  # 更新A股
#                 time.sleep(2)
#                 update_h_stocks()
#                 time.sleep(2)
#
#                 init_all_data()
#
#                 # target_runtime = datetime.datetime.now()
#                 # sleep_time = target_runtime.replace(day=target_runtime.day + 1,
#                 #                                     hour=7, minute=5, second=0) - target_runtime
#                 # # print("next_runtime:",target_runtime.replace(day=(target_runtime+datetime.timedelta(days=1)),hour=7,minute=5,second=0))
#                 # # print(sleep_time-10)
#                 time.sleep(10)
#             else:
#                 target_runtime = datetime.datetime.now()
#                 print(target_runtime)
#                 sleep_time = target_runtime.replace(day=target_runtime.day + 1,
#                                                     hour=7, minute=5, second=0) - target_runtime
#
#                 # print("1,next_runtime:", target_runtime.replace(day=target_runtime.day + 1, hour=7, minute=5, second=0))
#                 print("sleep_time:", sleep_time.seconds - 10)
#                 time.sleep(10)

if __name__ == '__main__':

    # t1 = threading.Thread(target=dingshi_update_all_data, args=(17,39))
    # t1.start()
    # # print(threading.current_thread())
    # print(str(threading.enumerate()))
    # a_dsq = run_dingshiqi()
    # print(123)
    print(scheduler.get_jobs())

    server.run(host='0.0.0.0',port=8888)
