import base64
import datetime
import io
import numpy as np
import time
import dash
from dash.dependencies import Input, Output,State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go

import pandas as pd
# from dash.exceptions import PreventUpdate
from flask_apscheduler import APScheduler
from flask import Flask

from functools import reduce
import plotly.figure_factory as ff
from tool_funcs import *

np.seterr(divide='ignore', invalid='ignore')


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

server = Flask(__name__)
app = dash.Dash(__name__, server=False)
scheduler = APScheduler()
server.config['SCHEDULER_API_ENABLED'] = True

scheduler.init_app(server)
app.init_app(server)
scheduler.start()

global hc_config


df_indexs_data = load_local_indexs_data()
yinzis = ["yinzi01","yinzi02","yinzi03","yinzi031","yinzi04","yinzi05","yinzi06","yinzi07"]
hc_config = {'A': 'sh.000300', 'B': 'sz.399006', 'yinzi_name': 'yinzi03', 'yinzi_canshu': 20, 'pos_zq': 20, 'sxf': 0.001}

A_indexs_bt_sys = html.Div([
            html.H4("A股指数回测系统",id = "A_bt_sys_h4",
                     style={'width': 500,'height': 30,"left":130,"top":80,"position":"absolute","font-size":18})  ,
            html.Div(id="huice_info_tip", children=["回测介绍:"],
                         style={'width': 500, 'height': 330, "left": 1400, "top": 85, "position": "absolute",
                                'backgroundColor': 'rgb(251, 251, 131,0.15)', 'fontWeight': '420', "font-size": 16,
                                'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "left",
                                "box-shadow": "5px -3px 3px #898888",
                                "border-radius": "3px"}),

            html.Div(children = [
                #设置品种 A，B
                html.Div(children=[
                                    html.Div(children=["选择a品种："],
                                            style={ 'height': 25, "left": 100, "top": 8, "position": "absolute",}),

                                    dcc.Dropdown(id = "A_pinzhong",options=[{'label': i, 'value': i}
                                                                            for i in  [i for i in df_indexs_data["code_name"].tolist()]] ,
                                        value="沪深300指数",multi=False,placeholder=" ---选择a品种--- ",
                                        style={'width': 250, 'height': 25, "left": 150, "top": 5, "position": "absolute",}),

                                    html.Div(children=["选择b品种："],
                                            style={'height': 25, "left": 600, "top": 8, "position": "absolute", }),

                                    dcc.Dropdown(id = "B_pinzhong",options=[{'label': i, 'value': i }
                                                                            for i in  [i for i in df_indexs_data["code_name"].tolist()]] ,
                                        value="创业板指数(价格)",multi=False,placeholder=" ---选择b品种--- ",
                                        style={'width': 250, 'height': 25, "left": 420, "top": 5, "position": "absolute",})
                                    ],style={'width': 1500,'height': 30,"left":130,"top":35,"position":"absolute","font-size":18})  ,
                #设置回测因子和参数
                html.Div(children=[

                    html.Div(children=["选择回测因子："],
                                        style={'width': 250, 'height': 25, "left": 100, "top": 7, "position": "absolute",}),
                            dcc.Dropdown(id = "yinzi_cal",options=[{'label': i, 'value': i} for i in  yinzis ],
                                value=yinzis[0],multi=False,placeholder=" ---选择因子--- ",
                                style={'width': 250, 'height': 25, "left": 150, "top": 5, "position": "absolute",}),
                            html.Div(children=["设置因子参数："],
                                        style={ 'height': 25, "left": 600, "top": 7, "position": "absolute",}),
                            dcc.Dropdown(id = "yinzi_cs_set",options=[{'label': i, 'value': i} for i in  [5,10,15,20,25,30,40,50,60]] ,
                                value=20,multi= False,placeholder=" ---选择参数--- ",
                                style={'width': 250, 'height': 25, "left": 420, "top": 5, "position": "absolute",}),
                           ],style={'width': 1500,'height': 30,"left":130,"top":100,"position":"absolute","font-size":18})  ,
                #设置回测参数
                html.Div(children=[

                    html.Div(children=["持仓周期："],
                                        style={'width': 250, 'height': 25, "left": 100, "top": 7, "position": "absolute",}),
                            dcc.Dropdown(id = "jiaoyi_pos_zq",options=[{'label': i, 'value': i} for i in  [5,10,15,20,25,30,40,50,60] ],
                                value=20,multi=False,placeholder=" ---选择--- ",
                                style={'width': 250, 'height': 25, "left": 150, "top": 5, "position": "absolute",}),
                            html.Div(children=["交易手续费："],
                                        style={ 'height': 25, "left": 600, "top": 7, "position": "absolute",}),
                            dcc.Dropdown(id = "jiaoyi_sxf_set",options=[{'label': i/10000, 'value': i/10000} for i in  [5,10,15,20,25,30]] ,
                                value=10/10000,multi= False,placeholder=" ---选择--- ",
                                style={'width': 250, 'height': 25, "left": 420, "top": 5, "position": "absolute",}),
                           ],style={'width': 1500,'height': 30,"left":130,"top":170,"position":"absolute","font-size":18})  ,
                # 一键回测按钮
                html.Div(children=[html.Button(id = "btn_backtest",children=["一键回测"],
                                        style={'width': 350, 'height': 40, "left": 100, "top": 5, "position": "absolute",}),
                                   ],
                        style={'width': 1500,'height': 30,"left":130,"top":250,"position":"absolute","font-size":18})  ,
            ],
            style={'width': 1250, 'height': 330, "left": 80, "top": 85, "position": "absolute",
                                'backgroundColor': 'rgb(201, 201, 231,0.3)', 'fontWeight': '420',
                                "font-size": 16,
                                'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "left",
                                "box-shadow": "5px -3px 3px #898888",
                                "border-radius": "4px"}),
            # 回测jieguo
            html.Div(id="hc_res_div",children=[
                               ],
                    style={'width': 1500,'height': 30,"left":130,"top":450,"position":"absolute","font-size":18})  ,


        #
        #     html.Div(id = "Checklist_base_col",children=[
        #                 # dcc.Dropdown(id = "base_col",options=[{'label': i, 'value': i} for i in  [i for i in df.columns if i not in cols]] ,
        #                 #     value=[],multi=True,placeholder=" ---观察项列，选择框--- ",
        #                 #     style={'width': 600, 'height': 25, "left": 150, "top": 5, "position": "absolute",})
        #         dcc.Checklist(id = "base_col",
        #          options=[{'label': i+"__", 'value': i} for i in  [i for i in df.columns if i not in cols] ],value=[],
        #         style = {
        #                  'fontWeight': '450',
        #                   "font-size":17,
        #                   "text-align": "center",
        #                  })
        #     ],
        #         style={'width': 1020,'height': 60, "left":55,"top":105,"position":"absolute",
        #                                         "word-break":"break-all",
        #                                         "word-spacing":"10px",
        #                                       'backgroundColor': 'rgb(50, 155, 30,0.1)','fontWeight': 'bold',
        #                                       'border': '1px solid rgb(220, 200, 120,0.5)',"text-align":"center",
        #                                       "border-radius":"5px"},),
        #     html.Div(id = "condition_div_show",
        #              style={'width': 1550,'height': 30,"left":130,"top":180,"position":"absolute","font-size":18}),
        #     #  base_table
        #     html.Div([dash_table.DataTable(
        # id='base_table',
        # columns=[{"name": i, "id": i} for i in df.columns if i in cols+choosed_cols],
        # # dropdown={"行业名": {"clearable" : True, "options" : [ {'label': k, 'value': k} for k in df['行业名'].unique()]}},
        # loading_state ={"is_loading ":True},
        # data=df.to_dict('records'),
        # style_data_conditional=[{'if': {'row_index': 'even'},'backgroundColor': 'rgb(255, 255, 210,1)'}]
        #                        + data_bars(df, '1强度_指数') + data_bars(df, '60强度_指数')+
        #                        data_bars(df, '大于20线')+data_bars(df, '大于60线')+data_bars(df, '大于120线'),
        #         style_data={'if': {'row_index': 'odd'},'backgroundColor': 'rgb(208, 248, 248)'},
        # style_header={'backgroundColor': 'rgb(250, 220, 220)','fontWeight': 'bold','border': '2px solid black',
        #               'textWrap': 'normal','height': 'auto',
        #               },
        # style_cell={'textAlign': 'center',
        #             "font-size":15,'fontWeight': '430',
        #                 'width': '70px','minWidth': '40px',  'maxWidth': '70px',
        #                 "height":35,
        #                 'overflow': 'hidden',
        #                 'textOverflow': 'ellipsis',
        #                 'border': '2px solid black',
        #                 "color":'black'},
        # # tooltip_header={i : {'value':i,'type': 'markdown' } for i in df.columns if i in cols+choosed_cols},
        # # tooltip_data=[{column: {'value': str(column)+"\t = \t"+str(value), 'type': 'text',"duration":1000*5 }
        # #                for column, value in row.items()} for row in df.to_dict('records')],
        # fixed_rows={'headers': True},
        # filter_action="custom",sort_action="custom",
        # sort_mode="multi",
        # column_selectable="multi",
        # row_selectable="multi",
        # row_deletable=False,
        # selected_columns=[],
        # selected_rows=[],
        # selected_cells=[],
        # page_action="native",
        # page_current= 0,
        # page_size= 50,
        # export_format='xlsx',
        # export_headers='display',
        # # fixed_columns={'headers': True,"data":4},
        # style_table={'width': '1500px','height': 590,"left":20,"top":10,
        #              },
        #     )],id = "base_table_div",
        #         style={'width': 1550,'height': 600, "left":50,"top":175,"position":"absolute",
        #                'backgroundColor': 'rgb(151, 181, 151,0.1)', 'fontWeight': '410',
        #                "font-size": 17, 'border': '1px solid rgb(220, 200, 120,0.5)',
        #                "text-align": "center", "border-radius": "4px"},),
        #     # 雪球
        #     html.Div([dcc.Link(id = "Astock_link", children=[f" 没有选择股票:  雪球行情页面"],href="https://xueqiu.com/hq",
        #                        refresh=False,target="_blank")],id = "xq_link",
        #               style={'width': 350,'height': 30, "left":75,"top":795,"position":"absolute","font-size":20,
        #                      'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
        #                      # 'border': '1px solid rgb(220, 200, 120,0.5)',
        #                      "text-align": "center", "border-radius": "5px"
        #                      }),
        #     # 同花顺财务
        #     html.Div([dcc.Link(id = "Astock_cw_link", children=[f" 财务查看:  同花顺页面"],href="http://q.10jqka.com.cn/",
        #                        refresh=False,target="_blank")],id = "ths_cw_link",
        #               style={'width': 350,'height': 30, "left":450,"top":795,"position":"absolute","font-size":20,
        #                      'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
        #                      # 'border': '1px solid rgb(220, 200, 120,0.5)',
        #                      "text-align": "center", "border-radius": "5px"
        #                      }),
        #     # 问财行业对比
        #     html.Div([dcc.Link(id = "Astock_wencai_link", children=[f" 行业及产品的上下游:  问财页面"],href="http://www.iwencai.com",
        #                        refresh=False,target="_blank")],id = "ths_wc_link",
        #               style={'width': 350,'height': 30, "left":830,"top":795,"position":"absolute","font-size":20,
        #                      'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
        #                      # 'border': '1px solid rgb(220, 200, 120,0.5)',
        #                      "text-align": "center", "border-radius": "5px"
        #                      }),
        #     # 行业下拉选择框
        #     html.Div(id = "Dropdown_hy",children=[
        #         dcc.Dropdown(id = "hy_dd",options=[{'label': i, 'value': i} for i in s_df["行业名"].tolist()],
        #                     value=[],multi=True,placeholder=" 行业选择框 ：      默认全选  ",
        #                      style={"font-size":18,'fontWeight': 500,'border': '1px solid rgb(220, 100, 120,0.5)', "border-radius": "5px"}
        #                      )],
        #             style={'width': 500,'height': 40, "left":1230,"top":792,"position":"absolute",}),
        #     #des_div
        #     html.Div(id="base_des_div",style={'width': 200,'height': 360, "left":1630,"top":185,"position":"absolute",
        #                                       'backgroundColor': 'rgb(151, 151, 231,0.3)','fontWeight': '420',"font-size":12,
        #                                       'border': '1px solid rgb(220, 200, 120,0.5)',"text-align":"center",
        #                                       "box-shadow": "5px -3px 3px #898888",
        #                                       "border-radius":"3px"}),
        # # des_div
        # html.Div(id="info_tip_div", children=["复合筛选操作指南：",html.Br(),"数字操作 ：>0+<10 ",html.Br(),"字符串操作 ：银行*保险"],style={'width': 200, 'height': 200, "left": 1630, "top": 555, "position": "absolute",
        #                                    'backgroundColor': 'rgb(251, 251, 131,0.15)', 'fontWeight': '420', "font-size": 15,
        #                                    'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "center",
        #                                    "box-shadow": "5px -3px 3px #898888",
        #                                    "border-radius": "3px"}),
        #
        #     #hy_table
        #     html.Div(id='hy_table',children = [dash_table.DataTable(
        # id='hy_table0',
        # columns=[{"name": i, "id": i, "deletable": False, "selectable": False} for i in s_df.columns if i != 'id'],
        # data=s_df.to_dict('records'),
        # style_data_conditional=[{'if': {'row_index': 'even'},'backgroundColor': 'rgb(248, 248, 228,1)'}]+
        # data_bars(s_df, '60强度')+data_bars(s_df, '20强度')+data_bars(s_df, '120强度')+
        # data_bars(s_df, '每日alpha%')+data_bars(s_df, '60alpha%')+
        # data_bars(s_df, '每日alpha_mean')+data_bars(s_df, '60alpha_mean'),
        # style_data={'if': {'row_index': 'odd'},'backgroundColor': 'rgb(228, 248, 248,1)'},
        #
        # style_header={'backgroundColor': 'rgb(230, 200, 200)','fontWeight': 'bold','border': '2px solid red',"height":40},
        # style_cell_conditional=[
        #     {
        #         'if': {'column_id': '行业名'},
        #         'width': '70px'
        #     },
        # ],
        # style_cell={'textAlign': 'center',
        #             "font-size":15,'fontWeight': '460',
        #                 'width': '60px', 'maxWidth': '60px','minWidth': '60px',
        #                 'overflow': 'hidden',
        #                 'textOverflow': 'ellipsis',
        #                 'border': '2px solid grey',
        #                 "color":'black'},
        # tooltip={i: {'value': i, 'use_with': 'both'} for i in s_df.columns},
        # filter_action="custom",sort_action="custom",
        # sort_mode="multi",
        # column_selectable="multi",
        # row_selectable="multi",
        # row_deletable=False,
        # selected_columns=[],
        # selected_rows=[],
        # page_action="native",
        # page_current= 0,
        # page_size= 100,
        # # fixed_columns={'headers': True, "data": 1},
        # style_table={'width': 800,"top": 15,"left": 10,'overflowX': 'auto'},#
        #     )],
        #      style={'width': 830, 'height': 1000, "left": 50, "top": 850, "position": "absolute",
        #             'backgroundColor': 'rgb(100, 155, 130,0.1)', 'fontWeight': 'bold',
        #             'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "center",
        #             "border-radius": "5px"
        #             },),
        #     #selected_table
        #     html.Div(id='selected_table',children = [dash_table.DataTable(
        #     id='selected_table0',
        #     columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in seleced_table.columns if i != 'id'],
        #     # fixed_columns={'headers': True,"data":1},
        #     data=seleced_table.to_dict('records'),
        #     style_data_conditional=data_bars(seleced_table, '60强度')+data_bars(seleced_table, '20强度')+data_bars(seleced_table, '120强度')+
        #         data_bars(seleced_table, '每日alpha%')+data_bars(seleced_table, '60alpha%')+data_bars(seleced_table, '每日alpha_mean')+data_bars(seleced_table, '60alpha_mean'),
        #     style_data={'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'},
        #     style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'border': '2px solid black', "height": 40},
        #     style_cell={'textAlign': 'center',"font-size":15,'fontWeight': '450',
        #                 'minWidth': '60px', 'width': '70px', 'maxWidth': '120px',
        #                 'overflow': 'hidden',
        #                 'textOverflow': 'ellipsis',
        #                 'border': '2px solid grey',
        #                 "color": 'black'},
        #     tooltip={i: {'value': i, 'use_with': 'both'} for i in seleced_table.columns},
        #     sort_mode="multi",
        #     filter_action="custom", sort_action="custom",
        #     row_deletable=False,
        #     selected_columns=[],
        #     selected_rows=[],
        #     page_action="native",
        #     page_current=0,
        #     page_size=100,
        #     style_table={'width': 900, "left":25,"top":15},
        #     )],
        #                    style={'width': 950  ,'height': 1000, "left":900,"top":850,"position":"absolute",
        #                           'backgroundColor': 'rgb(100, 155, 130,0.1)', 'fontWeight': 'bold',
        #                           'border': '1px solid rgb(220, 200, 120,0.5)', "text-align": "center",
        #                           "border-radius": "5px"
        #                           },),
        #
        #     html.Div(id='A_scgdqd_graph',children=[get_a_heat_map(df=sckd["是否大于20均线"],id="A_20ma_gd0")],
        #                    style={'width': 1500,'height': 200, "left":150,"top":1900,"position":"absolute",
        #                           "border-radius":"5px"},),
        #      html.Div(id="scqd_tip_div", children=["市场强度20MA指南：",html.Br(),html.Br(),
        #                                            "单个行业,从左到右,反应近期的资金热度。",html.Br(),
        #                                            "整体，从左往右，反应市场资金在行业之间的流动，又称行业滚动。",html.Br(),html.Br(),
        #                                            "制作原理：",html.Br(),html.Br(),
        #                                            "统计每个行业个股大于MA20的个数，计算%。",html.Br(),
        #                                            "MA20线反应市场短期资金成本，当价格在其上，"
        #                                            "代表短期资金看好。容易形成短期趋势。",html.Br(),
        #                                            "短期趋势形成会扩散成中期趋势。",html.Br(),
        #                                            "趋势：资金成本不断抬高的过程。"],
        #      style={'width': 230, 'height': 600, "left": 1600, "top": 2000, "position": "absolute",
        #             'backgroundColor': 'rgb(151, 51, 231,0.1)', 'fontWeight': '420', "font-size": 17,
        #             'border': '2px solid rgb(220, 200, 120,0.5)', "text-align": "left",
        #             "box-shadow": "5px -3px 3px #898888",
        #             "border-radius": "3px"}),
])

app.layout = html.Div([
                        # dcc.Store(id = "store_dingshi_triger",data={"ds":[9,30]}),
                        #
                        # store_mem,
                        # store_mem2,
                        #
                        # dcc.Interval(id="dingshi_tool",interval=10*1000,),
                        dcc.Tabs(id="tabs_main",value = "A股index_回测sys",
                         children =[
                            dcc.Tab(label='A股index_回测sys', value='A股index_回测sys', children=[A_indexs_bt_sys], ),
                            # dcc.Tab(label='A股观察', value='A股观察',children=[Astocks],),
                            # dcc.Tab(label='H股观察', value='H股观察', children=[Hstocks],),
                            # dcc.Tab(label='A股逐日-滚动强度', value='A股滚动强度', children=[A_shichangqd],),
                            # dcc.Tab(label='H股逐日-滚动强度', value='H股滚动强度', children=[H_shichangqd],),
                            # dcc.Tab(label='同花顺自选股观察', value='同花顺自选股观察', children=[ths_stocks])
                            ],),
                        html.Hr(id="hr",style={"color":"#ff6699"}),
])



@app.callback(output=[
    Output('huice_info_tip', 'children'),
                      ],
              inputs=[Input('A_pinzhong', 'value'),Input('B_pinzhong', 'value'),
                    Input('yinzi_cal', 'value'),Input('yinzi_cs_set', 'value'),
                    Input('jiaoyi_pos_zq', 'value'),Input('jiaoyi_sxf_set', 'value'),
                      ],
              )
def run_huice_sys(A,B,yzname,yz_cs0,zq,sxf):
    global df_indexs_data , hc_config
    info = [html.Br(),"回测介绍：",html.Br(),html.Br(),"A品种:=",A,html.Br(),"A品种代码:=",str(df_indexs_data.loc[df_indexs_data["code_name"]==A,"code"].iloc[0]),html.Br(),
            "B品种:=",B,html.Br(),"B品种代码:=",str(df_indexs_data.loc[df_indexs_data["code_name"]==B,"code"].iloc[0]),html.Br(),
            "每次回测都是，实时抓取数据，每次回测时间不超过15s。",html.Br(),html.Br(),
            "回测原理简介，对两个自定义的指数进行选择持仓。假如是两个行业指数，那么会在两个行业之间进行轮动持仓。",html.Br(),html.Br(),
            "持仓的准则，根据选择的因子不同，而进行标准化选择持仓。"
            "统计收益情况，具体可对照：周期持仓表查看。"
            # "回测因子:=",str(yzname),html.Br(),"因子参数:=",str(yz_cs0),html.Br(),
            # "持仓周期:=", str(zq), html.Br(), "交易手续费:=", str(sxf),
            ]
    hc_config = {"A":str(df_indexs_data.loc[df_indexs_data["code_name"]==A,"code"].iloc[0]),
                 "B":str(df_indexs_data.loc[df_indexs_data["code_name"]==B,"code"].iloc[0]),
                 "yinzi_name":yzname,
                 "yinzi_canshu":yz_cs0,
                "pos_zq":zq,
                 "sxf":sxf}
    return [info]

@app.callback(
    Output('hc_res_div', 'children'),
    Input('btn_backtest', 'n_clicks')
    )
def update_output(n_clicks):
    global hc_config , df_indexs_data
    print(hc_config)
    hc_config_info = [html.Br(), "回测配置信息", html.Br(), html.Br(), "A品种:=", hc_config["A"],
             html.Br(),"B品种:=", hc_config["B"], html.Br(),  html.Br(),
            "回测因子:=", hc_config["yinzi_name"], html.Br(), "因子参数:=", hc_config["yinzi_canshu"], html.Br(),
            "持仓周期:=",  hc_config["pos_zq"], html.Br(), "交易手续费:=", hc_config["sxf"],
            ]

    yinzi_canshu_m = hc_config["yinzi_canshu"]
    n_day = int(hc_config["pos_zq"])
    sxf = float(hc_config["sxf"])
    yinzi_name = hc_config["yinzi_name"]
    A = hc_config["A"]
    B = hc_config["B"]
    info = df_indexs_data[df_indexs_data["code"].isin([A, B])].copy()
    stime = str(max(info["ipoDate"]))
    etime = datetime.datetime.now().strftime("%Y-%m-%d")


    df_A = get_data(index_code=A, stime=stime, etime=etime, yinzi_name=yinzi_name, yinzi_cs0=yinzi_canshu_m)
    df_B = get_data(index_code=B, stime=stime, etime=etime, yinzi_name=yinzi_name, yinzi_cs0=yinzi_canshu_m)

    df_pos = huice(df_A, df_B, n = yinzi_canshu_m,pos_zq=n_day,sxf=sxf)

    df_pos2 = df_pos[['candle_begin_time', 'signal', 'pos', "open_price", 'per_lr']].copy()
    df_pos2["pos2"] = df_pos2["pos"].shift(1)
    df_pos2.loc[df_pos2["signal"] == 0, "pos"] = df_pos2.loc[df_pos2["signal"] == 0, "pos2"]
    df_pos2["pos2"] = np.nan
    df_pos2.loc[(df_pos2["pos"] != df_pos2["pos"].shift()), "pos2"] = \
        df_pos2.loc[(df_pos2["pos"] != df_pos2["pos"].shift()), "candle_begin_time"]
    df_pos2["pos2"].fillna(method="ffill", inplace=True)
    decrib_per_trade, decrib_per_trade2 = zhubi_fx2(df=df_pos2)

    # print(decrib_per_trade, decrib_per_trade2)
    # df_pos["candle_begin_time"] = df_pos["candle_begin_time"].str[8:16].str.replace(" ", "日")
    h_info = ""
    fig = go.Figure()
    titleifo = f"资金曲线"
    df_pos["candle_begin_time"] = pd.to_datetime(df_pos["candle_begin_time"]). \
        apply(lambda x: x.to_pydatetime().strftime("%Y_n_%m_y_%d_r"))

    x_arr = df_pos["candle_begin_time"].tolist()
    for k in [ "hc_zj","A_jz", "B_jz"]:
        y_arr = df_pos[k].tolist()  # +[np.nan for i in range(500)]

        fig = fig.add_trace(go.Scatter(x=x_arr, y=y_arr, name=k, mode='lines', showlegend=True))

    fig.update_layout(
        xaxis=dict(visible=True, showline=True, showgrid=True, gridwidth=1, gridcolor='rgb(52, 20, 52)',
                   showticklabels=True, title='日期',
                   zeroline=True,
                   linecolor='rgb(52, 52, 52)',
                   tickfont=dict(family='Arial', size=12, color='rgb(52, 52, 52)', )),

        yaxis=dict(showline=True, showgrid=True, gridwidth=1, gridcolor='rgb(52, 20, 52)',
                   zeroline=True, showticklabels=True, title=f"资金曲线",
                   type='linear',
                   linecolor='rgb(52, 52, 52)',
                   tickfont=dict(family='Arial', size=12, color='rgb(52, 52, 52)', ), ),

        autosize=True,
        showlegend=True,
        plot_bgcolor="black",
        width=1200, height=650,
        title=dict(text=titleifo, font=dict(family='Arial', size=16, color='rgb(202, 52, 52)')))

    fig0 = dcc.Graph(figure=fig)
    # decrib_per_trade =decrib_per_trade.reset_index(drop=True)
    # decrib_per_trade2 =decrib_per_trade2.reset_index(drop=False)

    table0 = dash_table.DataTable(
        id='drs_df',
        columns=[{"name": i, "id": i} for i in decrib_per_trade2.columns],
        data=decrib_per_trade2.to_dict('records'),
        style_header={'backgroundColor': 'rgb(240, 200, 200)', 'font-size': 14,
                      'fontWeight': '450', 'border': '2px solid black', "height": 50},

        style_cell={'textAlign': 'right', 'font-size': 17,
                    'minWidth': '60px', 'width': '70px', 'maxWidth': '70px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'border': '2px solid grey',
                    "color": 'black'},

        fixed_columns={"headers": True, "data": 1},
        style_table={"left":15,"right":15,"top":10})

    table1 = dash_table.DataTable(
        id='drs_df2',
        columns=[{"name": i, "id": i} for i in decrib_per_trade.columns],
        data=decrib_per_trade.to_dict('records'),
        style_header={'backgroundColor': 'rgb(240, 200, 200)', 'font-size': 14,
                      'fontWeight': '450', 'border': '2px solid black', "height": 50},
        style_cell={'textAlign': 'right', 'font-size': 17,
                    'minWidth': '60px', 'width': '70px', 'maxWidth': '80px',
                    # 'overflow': 'hidden',
                    # 'textOverflow': 'ellipsis',
                    'border': '2px solid grey',
                    "color": 'black'},
        # fixed_columns={"headers": True, "data": 1},
        style_table={'width': 1600,'height': 450,"left":15,"right":15,"top":10,'overflowY': 'scroll'})


    return [fig0,html.Div(hc_config_info, style={'width': 140,'height': 220, "left":-100,"top":30,"position":"absolute",
                                    "font-size":15,'backgroundColor': 'rgb(51, 101, 131,0.2)', 'fontWeight': '410',
                                    "text-align": "left", "border-radius": "5px"
                             }),
            html.Div(children = ["回测结果：",table0], style={'width': 550,'height': 650, "left":1220,"top":20,"position":"absolute",
                                    "font-size":20,'backgroundColor': 'rgb(51, 151, 231,0.2)', 'fontWeight': '410',
                                    "text-align": "center", "border-radius": "5px"
                             }),

            html.Div(children = ["周期持仓表",table1], style={'width': 1700,'height': 550, "left":60,"top":700,"position":"absolute",
                                    "font-size":20,'backgroundColor': 'rgb(151, 151, 231,0.2)', 'fontWeight': '410',
                                    "text-align": "center", "border-radius": "5px"
                             })]


if __name__ == '__main__':

    print(scheduler.get_jobs())

    server.run(host='127.0.0.1',port=8888,debug=True)
