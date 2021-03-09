from 时间序列检验 import facter_deal,norm_test,adf_time_stats
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.graph_objects as go

import plotly.figure_factory as ff
from numba_策略开发.功能工具.因子库 import rsi_cci_atr01 ,std_zdf_tp
from numba_策略开发.功能工具.功能函数 import transfer_to_period_data

import pandas as pd


data_df = pd.read_csv(r'F:\vnpy_my_gitee\new_company\hsi_data_1min\HSI2019-2020.csv')
data_df = transfer_to_period_data(data_df,time_cols= "datetime",rule_type="1H")
data_df['factor'] =  rsi_cci_atr01(data_df,18)


# print(data_df['factor'].shape)
data_df['wl_delter'] = data_df['close']
data_df["wlyl"] = data_df['wl_delter'].diff(1).shift(-5)
adf_seq = data_df['factor']
df_res = facter_deal(data_df,f_s=["factor"],wlsy=["wlyl","datetime"])
df_res =df_res[['datetime',  'factor',  'wlyl', '一阶差分', '二阶差分', '10_ma', 'tahn', 'sigmoid', '一阶差分10_ma', '一阶差分tahn', '一阶差分sigmoid', '二阶差分10_ma', '二阶差分tahn', '二阶差分sigmoid']]
df_tj = df_res.describe()
df_tj = df_tj.reset_index(drop=False)

res_zong = pd.DataFrame()
for key in df_res.keys():
    if key == "datetime":continue
    p = norm_test(a=df_res[key])
    dfoutput = adf_time_stats(s=df_res[key])
    dfoutput['name:'] = key
    dfoutput['norm(<0.05):'] = p
    res_zong = res_zong.append(dfoutput, ignore_index=True)

    # desc_msg = "\n1.p值越小越好，要求小于给定的显著水平，p值小于0.05，等于0最好。"
    # desc_msg += "\n2.t%值，ADF值要小于t%值，1%， 5%， 10% 的三个level，都是一个临界值，\n    如果小于这个临界值，说明拒绝原假设。"
    # desc_msg+="\n3.其中，1% ： 严格拒绝原假设； \n    5%： 拒绝原假设； \n    10% 以此类推，程度越来越低。\n    如果，ADF小于1% level， 说明严格拒绝原假设。\n"
    # print(desc_msg)
res_zong = res_zong[['name:', 'adf_测试值', 'p_值(<0.05)', "norm(<0.05):", 'mean', 'std', '偏度', '峰度', '原假设小于_(1%)',
                     '原假设小于_(5%)', '原假设小于_(10%)', '延迟', '方差', '测试次数']]
# print(df_tj)
# print(res_zong.tail())
# exit()
# print(res_zong[['name:', 'adf_测试值', 'p_值(<0.05)', "norm(<0.05):", 'mean', 'std', '偏度', '峰度', '原假设小于_(1%)',
#                      '原假设小于_(5%)', '原假设小于_(10%)']])
res_zong = res_zong.round(4)

# df = pd.read_csv(r'F:\vnpy_my_gitee\new_company\numba_策略开发\nunba_BD\逐笔结果_资金曲线_std_BD.csv')
# df = df[['Unnamed: 0', '开始时间','结束时间', '持续分钟', '最大净值', '最小净值', '盈亏大小', '结束-最大',
#        '结束-最小',  'rsi']]
show_factors_test_cols = ['name:', 'adf_测试值', 'p_值(<0.05)', "norm(<0.05):", 'mean', 'std', '偏度', '峰度', '原假设小于_(1%)','原假设小于_(5%)', '原假设小于_(10%)',] # '延迟', '方差', '测试次数'
res_info = {}
df = res_zong
# print(df[show_cols])
PAGE_SIZE = 1500
graph_cols = df['name:'].tolist()



def get_A_table(id,df0):
    return  dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i, "deletable": False, 'selectable': True} for i in df0.keys()],
        fixed_rows={'headers': True},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(208, 248, 248)'}],
        style_data={'backgroundColor': 'rgb(238, 238, 238)',
                    'width': '40px', 'maxWidth': '50px', 'minWidth': '20px' },

        style_header={'backgroundColor': 'rgb(230, 230, 230)',
                      'fontWeight': 'bold',
                      'border': '2px solid black',
                      "height": 40},

        style_cell_conditional=[{'if': {'column_id': i for i in df0.keys()},
                                 }, ],
        style_cell={'textAlign': 'center',
                    'whiteSpace': 'normal',
                    'minWidth': '30px', 'width': '40px', 'maxWidth': '40px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'border': '2px solid grey',
                    'fontWeight': 'bold',
                    "color": 'black'},
        data=df0.to_dict('records'),
        page_current=0,
        page_size=PAGE_SIZE,
        row_deletable=False,
        editable=False,
        # selected_rows=[],
        # row_selectable='multi',
        selected_columns=[],
        selected_cells=[],
        column_selectable="multi",
        page_action='custom',
        filter_action='custom',
        filter_query = '',
        sort_action='custom',
        sort_mode='multi',
        sort_by=[],
        style_table={ },
        tooltip={i: {'value': i, 'use_with': 'both'} for i in df0.keys()},
    )

app = dash.Dash(__name__)
app.layout =    html.Div(className="row", children=[ dcc.Store(id='filter_df_store'),dcc.Link(id='clicking',children=["dfdf"] , href = 'https://www.baidu.com'),
                html.H4('factor分析', style={"left": 5, "top": 5, "position": "absolute"},),
                html.Div(id='factors_table-container',style={"width": 800, "left": 1100, "top": 100, "position": "absolute"},),
                html.Div(id='wlsy_col_figs', className="five columns",style={"width": 800, "left": 1100, "top": 500, "position": "absolute"},),
                html.Div(children=[get_A_table(id='factors_table',df0 = df[show_factors_test_cols])],style={'height': 100, "width": 1000, "left": 100, "top": 15, "position": "absolute"},),
                html.Div(children=[get_A_table(id="filter_main_table",df0=df_res.iloc[-100:])], style={'height': 100, "width": 1000, "left": 100, "top": 250, "position": "absolute"}),
                html.Div(children=[get_A_table(id="tj_table",df0=df_tj)], style={'height': 800, "width": 1000, "left": 100, "top": 500, "position": "absolute"}),
                html.Div(id='syqx_figs',style={"width": 1060, "left": 100, "top": 1100, "position": "absolute"}, ),

                                                 ])


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

@app.callback(Output('factors_table', "data"),Input('factors_table', "sort_by"),Input('factors_table', "filter_query"),)
def update_factors_table(sort_by, filter):
    global df
    filtering_expressions = filter.split(' && ')
    dff = df
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    return dff.to_dict('records')

@app.callback([Output('filter_main_table', "data"),Output('filter_df_store', "data")],
    [Input('filter_main_table', "sort_by"),Input('filter_main_table', "filter_query"),])
def update_filter_main_table(sort_by, filter):
    global df_res
    dff = df_res
    filtering_expressions = filter.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )
    return dff.iloc[-5:].to_dict('records'),dff.to_dict('records')

@app.callback(Output('tj_table', "data"),Input('filter_df_store', "data") )
def update_tj_table(data):
    data =pd.DataFrame(data)
    group_labels  =[ i  for i in data.keys() if i not in["index","datetime"]]
    data = data[group_labels]
    df_filter = pd.DataFrame(data)

    descr = df_filter.describe().round(3)
    descr.loc['总和'] = df_filter.sum()
    descr.loc['总和'] = df_filter.sum()
    descr.loc['夏普率'] = descr.loc['mean']/descr.loc['std']


    descr.loc['胜率'] = 100*df_filter[df_filter["wlyl"]>0].count()/df_filter.shape[0]
    descr.loc['赢'] = df_filter[df_filter["wlyl"]>0].sum()
    descr.loc['亏'] = df_filter[df_filter["wlyl"]<=0].sum()
    descr = descr[group_labels]
    descr= descr.loc[["总和","胜率","赢","亏","count","夏普率","mean","std","max","min","50%"]]

    descr.reset_index(drop=False,inplace=True)


    return descr.to_dict("records")

@app.callback( [ Output('filter_main_table',"style_data_conditional"),Output('tj_table', 'style_data_conditional'),Output('clicking', 'href'),],

                [Input('filter_main_table', 'selected_columns'),Input('filter_main_table', 'selected_cells'),Input('tj_table', 'selected_columns')])
def update_styles(selected_columns1,selected_cells,selected_columns2):
    print(123,selected_cells)
    selected_columns = set(selected_columns1) | set(selected_columns2)
    selected_cols = [c for c in selected_columns]

    condition0 = [{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(208, 248, 248)'}]

    condition0+=[{'if': {'column_id': i},'background_color': 'rgb(248, 148, 148)'} for i in selected_cols]

    return condition0,condition0,'https://www.baidu.com'

@app.callback(
    Output('factors_table-container', "children"),
    Input('filter_main_table', 'selected_columns'),
    Input('filter_df_store', "data"))
def update_graph(selected_columns,data):
    data0 = pd.DataFrame(data)

    if len(selected_columns)==0:
        hist_data = [data0.iloc[:][data0.keys()[1]].tolist() ]
        group_labels = [data0.keys()[1]]
    else:
        hist_data = [data0[i].tolist() for i in selected_columns]
        group_labels = [i for i in selected_columns]
    fig0 = ff.create_distplot(hist_data, group_labels, show_hist=True, show_curve=True, bin_size=0.1)
    figs =dcc.Graph(figure=fig0)

    return html.Div(children =figs)



@app.callback(
    Output('wlsy_col_figs', "children"),
    Input('filter_main_table', 'selected_columns'),
    Input('filter_df_store', "data")

)
def update_graph2(selected_columns,data):
    data0 = pd.DataFrame(data)
    if len(selected_columns) == 0:
        figs = [dcc.Graph(figure=px.scatter(data0[["factor", "wlyl"]],
                                            x="factor", y="wlyl",
                                             size_max=2)) ]
    else:

        figs =[dcc.Graph(figure=px.scatter(data0[[k,"wlyl"]],
                                       x=k, y="wlyl",
                      size_max=2)) for k in selected_columns]

    return html.Div(children =figs)


@app.callback(
    Output('syqx_figs', "children"),
    Input('filter_df_store', "data")
)
def update_wlyl_graph(data):
    df =pd.DataFrame(data)
    df.sort_values(by='datetime',ascending=True,inplace=True)
    df['syqx'] = df['wlyl'].cumsum()
    df['syqx_with_sxf'] = (df['wlyl'] - 3).cumsum()
    df['sxf'] = df['syqx']-df['syqx_with_sxf']
    # df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].str[:]
    # print(132,df["datetime"].to_numpy())
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["datetime"].to_numpy(),y=df["syqx"].to_numpy(), name='收益曲线',mode='lines'))
    fig.add_trace(go.Scatter(x=df["datetime"].to_numpy(), y=df["syqx_with_sxf"],name='收益曲线-手续费', mode='lines'))
    fig.add_trace(go.Scatter(x=df["datetime"].to_numpy(), y=df["sxf"],name='手续费', mode='lines'))

    fig.update_layout(
        xaxis=dict(visible=True ,showline=True,showgrid=False,showticklabels=True,title='收益曲线分析', zeroline=True,
                  linecolor='rgb(204, 204, 204)',tickfont=dict(family='Arial',size=12,color='rgb(182, 182, 182)',)),
        yaxis=dict(showline=True,showgrid=True,zeroline=True,showticklabels=True,title="盈亏变化", type='linear',
                   linecolor='rgb(204, 204, 204)',tickfont=dict(family='Arial',size=12,color='rgb(182, 182, 182)',),),
        autosize=False,
        showlegend=True,
        plot_bgcolor='white',
        width=1000,height = 500
    )




    return dcc.Graph(figure=fig)



if __name__ == '__main__':

    app.run_server(debug=True)