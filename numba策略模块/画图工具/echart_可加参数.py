import os,time
import datetime as dt
import pandas as pd
import numpy as np
import talib
import random
from datetime import timedelta
from pyecharts import options as opts
from pyecharts.globals import ThemeType

from pyecharts.charts import Kline, Line, Bar, Grid,Scatter,Page,Tab
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


def transfer_to_period_data(df, rule_type):
    """
    将数据转换为其他周期的数据
    :param df:
    :param rule_type:
    :return:
    """

    # =====转换为其他分钟数据
    period_df = df.resample(rule=rule_type, on='candle_begin_time', label='left', closed='left').agg(
        {'open': 'first',
         'high': 'max',
         'low': 'min',
         'close': 'last',
         'volume': 'sum',
         })
    period_df.dropna(subset=['open'], inplace=True)  # 去除一天都没有交易的周期
    period_df = period_df[period_df['volume'] > 0]  # 去除成交量为0的交易周期
    period_df.reset_index(inplace=True)
    df = period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]

    return df



def draw_charts(kline_data ,canshu ={} ,canshu2={},vol_bar=False,markline_show1 =False ,markline_show2 =False, path = '0501'):
    '''
    df['candle_begin_time','open','high','low','close','volume']
    [['candle_begin_time','open','high','low','close','volume']]

    kdata = df[['open', 'high', 'low', 'close']].values.tolist()
    df['candle_begin_time'].values.tolist()
    :return:
    '''

    df = kline_data.copy()

    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    time_list = df['candle_begin_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).values.tolist()
    vol = df['volume'].values.tolist()
    kline = Kline()
    kline.add_xaxis(xaxis_data=time_list)
    signal_pos_list = []
    if 'signal' in df.columns:
        print('signal,存在！')
        df['pos'] = df['signal'].shift(1)
        for i  in df[df['pos'] > 0].index:
            bar0 = df.iloc[i]
            sig_pos0 = opts.MarkPointItem(name="做多",
                               coord=[bar0['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),bar0.low -2],
                               value='买',
                               symbol ='circle',
                               symbol_size =[20,40],
                               itemstyle_opts = {'color': 'red'})

            signal_pos_list.append(sig_pos0)
        for i in df[df['pos'] < 0].index:
            bar0 = df.iloc[i]
            sig_pos0 = opts.MarkPointItem(name="做空",
                                          coord=[bar0['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),bar0.high +3],
                                          value='卖',
                                          symbol='circle',
                                          symbol_size=[20, 40],
                                          itemstyle_opts={'color': 'blue'})
            signal_pos_list.append(sig_pos0)
        for i  in df[df['pos'] == 0].index:
            bar0 = df.iloc[i]
            sig_pos0 = opts.MarkPointItem(name="平仓",
                               coord=[bar0['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),bar0.low - 2],
                               value='平',
                               symbol ='triangle',
                               symbol_size =[20,40],
                               itemstyle_opts = {'color': 'green'})

            signal_pos_list.append(sig_pos0)
    else :
        df['pos'] =None
    markline =[]
    if markline_show1 and ('signal' in df.columns) :
        area_index_list =[i for i  in df[(df['pos'] ==0)|(df['pos'] >0)|(df['pos'] <0)].index]
        for ix,i in enumerate(area_index_list):
            if ix+1 > len(area_index_list)-1:
                break
            i_now = df.iloc[area_index_list[ix]]
            i_next = df.iloc[area_index_list[ix+1]]
            if (i_now['pos'] >0) or (i_now['pos'] <0) :
                log_info = f"价差：={i_next['open']-i_now['open']}--({i_next['open']}-{i_now['open']})"
            else :
                log_info =f"平仓：{i_next['open']}---开仓：{i_now['open']}"
            sig_area = [{"xAxis": i_now['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            "yAxis": i_now['open'] ,
                            "value": None},
                        {"xAxis": i_next['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),
                         "yAxis": i_now['open'],
                         "value":log_info}]
            sig_area_v = [{"xAxis": i_next['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            "yAxis": i_now['open'] ,
                            "value": None},
                        {"xAxis": i_next['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),
                         "yAxis": i_next['open'],
                         "value":None}]
            markline.append(sig_area)
            markline.append(sig_area_v)

    elif markline_show2 and ('signal' in df.columns):
        area_index_list =[i for i in  df[(df['pos'] ==0)|(df['pos'] >0)|(df['pos'] <0)].index]
        for ix,i in enumerate(area_index_list):
            i_now = df.iloc[area_index_list[ix]]
            i_1_now = df.iloc[area_index_list[ix-1]] if ix != 0 else 0
            if i_now['pos'] == 0:
                log_info = f"交易价: {round(i_now['open'], 1)} +- 2__盈亏:{round(i_now['open'], 1) - round(i_1_now['open'], 1)}"
            else:
                log_info = f"交易价: {round(i_now['open'],1)} +- 2"

            sig_area = [{"xAxis": i_now['candle_begin_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            "yAxis": i_now['open'] ,
                            "value": None},
                        {"xAxis": (i_now['candle_begin_time']+timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                         "yAxis": i_now['open'],
                         "value":log_info}]

            markline.append(sig_area)


    colors = {'red': 'rgb((220,20,60)','red2': 'rgb((250,20,40)',
              'yangzi': 'rgb(255,0,255)', 'zi': 'rgb(128,0,128)',
              'sehnzi': 'rgb(148,0,211)', 'blue': 'rgb(0,0,255)',
              'qing': 'rgb(0,255,255)', 'black': 'rgb(0,0,0)',
              'shengreen': 'rgb(157,255,212)', 'green': 'rgb(0,255,0)',
              'yellow': 'rgb(255,255,0)', 'glod': 'rgb(218,165,32)',
              'orange': 'rgb(255,165,0)', 'brown': 'rgb(165,42,42)'}
    kline.add_xaxis(xaxis_data=time_list)
    kline.add_yaxis(series_name="oclh",
                    xaxis_index=1,
                    yaxis_index=1,
                    y_axis =df.loc[:, ['open', 'close', 'low', 'high']].values.tolist(),
                    markline_opts=opts.MarkLineOpts(data=markline,
                                                    linestyle_opts=opts.LineStyleOpts(type_='dotted',width=3,color='red',opacity=0.5),
                                                    label_opts=opts.LabelOpts(position="right", color="blue", font_size=13),),
                    #官网给出的样本
                    markpoint_opts=opts.MarkPointOpts(data=signal_pos_list),
                    itemstyle_opts=opts.ItemStyleOpts(color="#ec0090", color0="#00aa3c"), )

    kline.set_global_opts(
                    legend_opts=opts.LegendOpts(is_show=True,pos_top=30, pos_left="left",orient='vertical'),
                    datazoom_opts=[ opts.DataZoomOpts(
                                        is_show=False,
                                        type_="inside",
                                        xaxis_index=[0, 1],
                                        range_start=90,
                                        range_end=100,
                                        orient='vertical'),
                                    opts.DataZoomOpts(
                                        is_show=True,
                                        xaxis_index=[0, 1],
                                        type_="slider",
                                        pos_top="20%",
                                        range_start=90,
                                        range_end=100,orient='vertical'),],
                    yaxis_opts =opts.AxisOpts(
                                is_scale=True,
                                splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)),),
                    title_opts=opts.TitleOpts(
                        title = 'K_line',
                        pos_top='middle',
                        title_textstyle_opts=opts.TextStyleOpts(
                            color='black',font_weight='bold' ,font_size=20)),
                    tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                trigger_on='"mousemove"',#click|mousemove
                                axis_pointer_type="cross",
                                is_show_content=True,
                                is_always_show_content=True,
                                background_color="rgba(20, 105, 245, 0.1)",
                                border_width=1,
                                border_color= "#ccc",
                                position = ['70%','30%'],
                                textstyle_opts=opts.TextStyleOpts(font_size=10,color="#000"),),
                    visualmap_opts=opts.VisualMapOpts(
                                is_show=True,
                                dimension=2,
                                series_index=5,
                                is_piecewise=True,
                                pieces=[{"value": 1, "color": "#00da3c"},{"value": -1, "color": "#ec0000"},],),
                    axispointer_opts=opts.AxisPointerOpts(
                                is_show=True,
                                link=[{"xAxisIndex": "all"}],
                                label=opts.LabelOpts(background_color="#777"),),
                    brush_opts=opts.BrushOpts(
                                x_axis_index="all",
                                brush_link="all",
                                out_of_brush={"colorAlpha": 0.1},
                                brush_type="lineX",),
    )

    if len(canshu.keys())>0:
        cos = list(colors.values())
        line =  Line()
        for k,v in canshu.items():
            line.add_xaxis(xaxis_data=time_list)
            line.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
            co = cos.pop()
            line.add_yaxis(
            series_name=k,
            y_axis = [y for y in v.values.tolist() if y != np.nan],
            xaxis_index=1,
            yaxis_index=1,
            is_smooth=False,
            is_connect_nones=False,# 是否连接空数据
            is_symbol_show=False,#是否显示值的位置，默认显示。
            color = co,
            is_hover_animation = False, # 是否开启 hover 在拐点标志上的提示动画效果。
            linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.9,color=co),
            label_opts=opts.LabelOpts(is_show =True,position='middle',distance=2,rotate=5,color = 'rgb(165,42,42)'),
            itemstyle_opts=opts.ItemStyleOpts(color=co),)
        kline = kline.overlap(line)

    grid_chart = Grid(init_opts=opts.InitOpts(width = "1500px",height= "700px",theme=ThemeType.DARK))
    grid_chart.add(kline,grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%",pos_top='4%' ,height="70%"))

    if( vol_bar == True) or (len(canshu2.keys())==0):
        bar = Bar()
        bar.add_xaxis(xaxis_data=time_list)
        bar.add_yaxis(
            series_name="volume",
            y_axis=vol,
            xaxis_index=1,
            yaxis_index=2,
            label_opts=opts.LabelOpts(is_show=False), )

        bar.set_global_opts(xaxis_opts=opts.AxisOpts(
            type_="category",
            is_scale=True,
            grid_index=1,
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=False),
            split_number=20,
            min_="dataMin",
            max_="dataMax", ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_scale=True,
                split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False), ),
            legend_opts=opts.LegendOpts(is_show=True), )
        grid_chart.add(bar,grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="80%", height="15%"),)

    if len(canshu2.keys())>0 :
        line2 = Line()
        line2.add_xaxis(xaxis_data=time_list)
        for i, k in enumerate(canshu2.keys()):
            co = random.choice(list(colors.values()))
            line2.add_yaxis(
                series_name=k,
                y_axis=[y for y in canshu2[k].values.tolist()],
                xaxis_index=1,
                yaxis_index=i + 1,
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=True,
                color=co,
                z_level=0,
                linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5, color=co),
                label_opts=opts.LabelOpts(is_show=False), )
            if k == list(canshu2.keys())[-1]: continue
            line2.extend_axis(yaxis=opts.AxisOpts(name=k, type_="value", position="right", min_=None, max_=None,
                                                  axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color=co)),
                                                  axislabel_opts=opts.LabelOpts(formatter="{value}"), ))
        line2.set_global_opts(xaxis_opts=opts.AxisOpts(
            type_="category", is_scale=True, grid_index=1, split_number=20, min_="dataMin", max_="dataMax",
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=False), ),
            yaxis_opts=opts.AxisOpts(name=k, grid_index=i + 1, position='right',
                                     splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)),
                                     min_=None, max_=None, is_scale=True, offset=50, ), )
        grid_chart.add(line2,grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="75%", height="23%"),is_control_axis_index=True)


    tab = Tab()

    zijin_data = (kline_data['per_lr'].cumsum())
    zijin_data.fillna(method='ffill', inplace=True)
    zijin_line = Line(init_opts=opts.InitOpts(width = "1500px",height= "700px",theme=ThemeType.DARK)).add_xaxis(time_list)
    zijin_line.add_yaxis(series_name="zijin_line:", y_axis=zijin_data.values, color="#FF0000")
    zijin_line.set_global_opts(

            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
            title_opts=opts.TitleOpts(title="资金曲线变化"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            yaxis_opts = opts.AxisOpts(
                is_scale=True,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False), ),
                 legend_opts = opts.LegendOpts(is_show=True)
    )
    tab.add(grid_chart,'kline' )
    tab.add(zijin_line,'资金曲线变化' )
    tab.render(f"{path}.html")
    html0 = tab.render(f"{path}.html")
    if os.path.exists(html0):
        print("ok!保存在：")
        print(html0)
    else:
        print('保存失败！')


def draw_line_charts(input_data,input_data2,canshu =[],path=''):
    canshu_list = []
    max0 = int(max(input_data[canshu].max())*1.001)
    min0 =  int(min(input_data[canshu].min())*0.995)
    print(max0,min0)
    for i in canshu:
        canshu_dict0 = {}
        canshu_dict0[i] = input_data[i]
        canshu_list.append(canshu_dict0)
    print(input_data.keys())
    print(input_data.tail(100))
    input_data['candle_begin_time'] = pd.to_datetime(input_data['candle_begin_time'])
    time_list = input_data['candle_begin_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).values.tolist()

    tab = Tab()
    line_base = Line(init_opts=opts.InitOpts(width="1500px", height="700px", theme=ThemeType.DARK)).add_xaxis(time_list)
    line_base.add_yaxis(series_name="close:",
                        y_axis=input_data['close'].values,
                        is_symbol_show=False,

                        color="#FF0000")
    line_base.set_global_opts(
        datazoom_opts=[opts.DataZoomOpts(range_start=95,
            range_end=100,), opts.DataZoomOpts(
            type_="inside",
            range_start=95,
            range_end=100,
            )],
        title_opts=opts.TitleOpts(title="资金曲线变化"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            axislabel_opts=opts.LabelOpts(is_show=False),
            axisline_opts=opts.AxisLineOpts(is_show=False),
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False), ),
            legend_opts=opts.LegendOpts(is_show=True)    )
    for ix,v in enumerate(canshu_list):
        v0:str
        value0:dict
        for k,v0 in v.items():
            name =k
            value0 = v0
        data0 = pd.DataFrame(value0)
        data0.fillna(method='bfill',inplace=True)
        # print(data0)
        line_base.add_yaxis(
            series_name=name,
            y_axis=data0.values.tolist(),
            yaxis_index=2,
            linestyle_opts=opts.LineStyleOpts(),
            label_opts=opts.LabelOpts(is_show=False),
        )
        line_base.extend_axis(
            yaxis=opts.AxisOpts(
            name="资金情况",
            name_location="start",
            type_="value",
            max_= max0,
            min_= min0,
            is_inverse=False,
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),))
    tab.add(line_base,'资金变化')

    table0 = Table().add(list(input_data2.columns), input_data2.values.tolist()).set_global_opts(
                    title_opts=ComponentTitleOpts(title="策略统计", subtitle=str(dt.datetime.now())))
    table0.add_js_funcs('''
    document.write("<p style='color:#FF0000;'>
        注释：none </p>"
                        ''')
    tab.add(table0,'策略统计表')

    html0 =tab.render(path)
    if os.path.exists(html0):
        print("ok!保存在：")
        print(html0)
    else:
        print('保存失败！')





if __name__ == '__main__':

    # s_time = '2019-11-1'
    # e_time = '2019-12-19'
    # datapath = r'C:\Users\ASUS\Desktop\task\quanter\coin_dates\huobi_data\btcnew.csv'
    # df = pd.read_csv(filepath_or_buffer=datapath, date_parser='candle_begin_time')
    # # df['candle_begin_time_BG'] = pd.to_datetime(df['candle_begin_time_BG'])
    # df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    # df = df[df['candle_begin_time'] >= pd.to_datetime(s_time)]
    # df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
    # # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # df = transfer_to_period_data(df, rule_type='30T')

    df_time_list = [['2019-11-30', '2020-6-30'], ['2020-05-04 09:15:00', '2020-05-10 09:15:00']]
    s_time, e_time = df_time_list[1]

    datapath =r'F:\task\恒生股指期货\hsi_data_1min\HSI202001-202010.csv'
    df = pd.read_csv(filepath_or_buffer=datapath, index_col=1)
    # print(df.tail())
    # exit()
    df['candle_begin_time'] = df['datetime']
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    df = df[df['candle_begin_time'] >= pd.to_datetime(s_time)]
    df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]

    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)


    df['ma2'] = df['close'].rolling(30).mean()
    df['ma'] = df['close'].rolling(60).mean()

    df['ma_mom'] = df['ma'] -  df['ma'].shift(60)
    df['ma_bias'] = df['close'] -  df['ma'].shift(60)

    df.loc[((df['ma_mom']>0) & (df['ma_mom'].shift(1)<0)),'signal'] = 1
    df.loc[((df['ma_mom']<0) & (df['ma_mom'].shift(1)>0)),'signal'] = 0
    draw_line_charts(df,canshu=['ma_bias','ma_mom'])
    # draw_charts(df,canshu={'ma':df['ma']} ,canshu2={'ma_mom':df['ma_mom'],'ma_bias':df['ma_bias']} ,markline_show=True, path = '20201123')
