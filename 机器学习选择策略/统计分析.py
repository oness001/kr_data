import datetime
import time
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as stats

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 斯皮尔曼秩相关系数和p检验
stats.spearmanr(a=[],b=[])

def plot_fenbutu(a, b):


    fig = plt.figure(figsize = (10,6))
    ax1 = fig.add_subplot(2,1,1)  # 创建子图1
    ax1.scatter(a,b)
    plt.grid()
    # 绘制数据分布图

    # 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
    k_value, b_value, r_value, p_value, std_err = stats.linregress(a, b)
    ax1.plot(a, a*k_value+b_value, color='green', label='other')

    ax2 = fig.add_subplot(2,1,2)  # 创建子图2
    a.hist(bins=30, alpha=0.5, ax=ax2)
    a.plot(kind='kde', secondary_y=True, ax=ax2,c='g')
    b.hist(bins=30,alpha = 0.5,ax = ax2)
    b.plot(kind = 'kde', secondary_y=True,ax = ax2,c='r')
    plt.grid()
    plt.show()
    # 绘制直方图
    # 呈现较明显的正太性


def plot_fenbutu02(a,b,c):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)  # 创建子图1
    ax1.scatter(a, b,c='r')
    ax1.scatter(a, c,c='g')


    plt.grid()
    plt.show()
    # 绘制数据分布图

    # # 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
    # k_value, b_value, r_value, p_value, std_err = stats.linregress(a, b)
    # ax1.plot(a, a * k_value + b_value, color='green', label='other')

def plot_3d(data):
    fig = plt.figure()
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = fig.add_subplot(111, projection='3d')


    flag =['*','+','^','o','*','+','^','o','*','+','^','o','*','+','^','o']
    color_list = ['r','b','y','g','pink','black','#3d0101','#005001','#353541','#300541',]
    for i, z0 in enumerate(set(data['s_Time'].tolist())):
        print(i,z0)

        ax.scatter(xs=0, ys=data[data['s_Time']==z0]['未来=月终收益'], zs=data[data['s_Time']==z0]['预测值'], c=color_list[i], s=30, alpha=1, label='', marker=flag[i])

    ax.set_xticklabels(list(set(data['s_Time'].tolist())), fontsize=10)
    ax.set_yticklabels([" ", " ", "predict_price", " ", " "], fontsize=10)
    ax.set_zlabel('True_price', fontsize=16)

    plt.tight_layout(rect=(0, 0, 1, 1))
    # plt.savefig('student_score.pdf')
    plt.show()

import random


def echart_plot_3d(data):
    from pyecharts.charts import Bar3D
    from pyecharts import options as opts


    print(data.tail())

    df = data[['s_Time', '未来=月终收益' ,'预测值' ]].values.tolist()



    (Bar3D(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        "",
        df,
        xaxis3d_opts=opts.Axis3DOpts( type_="category"),
        yaxis3d_opts=opts.Axis3DOpts( type_="value"),
        zaxis3d_opts=opts.Axis3DOpts( type_="value"),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=100000),
        title_opts=opts.TitleOpts(title="predict"),
    )
    .render("predict.html")
)

def dong_scatter(data,info='',path0=''):
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode
    from pyecharts.charts import Scatter, Timeline
    # endlist = ['canshu', 'celue_name', '预测_s_Time', '未来end_3', '最终预测值1', 'hg最终预测值1']
    data['aim'] = data['canshu'] + data[ 'celue_name']
    data.rename(columns={'hg最终预测值1': '预测值', '未来end_3': '未来=月终收益','预测_s_Time':'s_Time'}, inplace=True)

    print(data.columns)

    print(data.tail())
    title = '动态图'

    df = data
    # data['预测值'] = data['预测值'].apply(lambda x: int(x))
    min_pre = min(data['预测值'].values.tolist())
    max_pre = max(data['预测值'].values.tolist())
    # df['未来=月终收益']=df['未来=月终收益'].apply(lambda x: int(x))
    # df['预测值']=df['预测值'].apply(lambda x: int(x))

    df['s_Time'] = pd.to_datetime(df['s_Time'])#.apply(lambda x:x.strftime(format="%Y-%m-%d"))
    df.sort_values(by=['s_Time'], ascending=True, inplace=True)
    tl = Timeline()
    timelist = list(set(df['s_Time'].tolist()))
    print(list(set(df['s_Time'].tolist())))
    sorted(timelist)
    # df_date = [time.strftime('%Y-%m-%d',time.localtime(i/1000000000) ) for i in timelist]
    # print(df_date)

    for k,i in enumerate(sorted(timelist)):
        # print(k,i)
        xdata = df.loc[df['s_Time'] == i, '预测值'].values.tolist()
        ydata = df.loc[df['s_Time']==i,['未来=月终收益','预测值']].values.tolist()
        # print(ydata)
        Scatter0 = (
            Scatter()
            .add_xaxis(xdata)
            .add_yaxis('未来=月终收益',ydata,label_opts = opts.LabelOpts(is_show=False))
            .set_series_opts()

            .set_global_opts(
                xaxis_opts=opts.AxisOpts(name = '预测值：',type_="value",axistick_opts=opts.AxisTickOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(name='真实值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),

                title_opts =opts.TitleOpts(title=f"{title}==:{i}月份的数据"),

                tooltip_opts = opts.TooltipOpts(formatter=JsCode("function (params) { return '真实：'+params.value[1] +' == 预测：'+ params.value[2];}")),
                visualmap_opts=opts.VisualMapOpts(min_=min_pre,max_= max_pre),
                ))
        tl.add(Scatter0, "{}月".format(i))
    tl.render(path0+f"{info}.html")
    print(path0+f"{info}.html")

def dong_scatter2(qian_n =10,data={},info='',print_data=[0,0,0],path0=''):
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode
    from pyecharts.charts import Scatter, Timeline,Line,Page
    from pyecharts.components import Table


    # endlist = ['canshu', 'celue_name', '预测_s_Time', '最终预测值1', 'hg最终预测值1',
    #            '未来end_3', '未来max_back_3', '未来sharp_rate_3', '未来trade_nums_3', '未来win_rate_3']
    data['aim'] = data['canshu'] + data[ 'celue_name']
    dict0 = {'hg最终预测值1': '回归=预测值', '最终预测值1':'分类=预测值', '预测_s_Time': 's_Time',
             '未来end_3': '未来=月终收益','未来max_back_3':'未来=最大回撤', '未来sharp_rate_3':'未来=夏普率', '未来trade_nums_3':'未来=交易次数','未来win_rate_3': '未来=胜率'}
    data.rename(columns=dict0, inplace=True)

    data = data.applymap(lambda x:round(x,3) if isinstance(x,float) else x)

    # print(data.columns)

    # print(data.tail())
    title = f'动态图:{qian_n}:'

    df = data
    # data['回归=预测值'] = data['回归=预测值'].apply(lambda x: int(x))

    # df['未来=月终收益']=df['未来=月终收益'].apply(lambda x: int(x))
    # df['回归=预测值']=df['回归=预测值'].apply(lambda x: int(x))

    df['s_Time'] = pd.to_datetime(df['s_Time'])#.apply(lambda x:x.strftime(format="%Y-%m-%d"))
    df.sort_values(by=['s_Time'], ascending=True, inplace=True)
    tl = Timeline()
    tl_fl = Timeline()

    timelist = list(set(df['s_Time'].tolist()))
    # sorted(timelist)
    # df_date = [time.strftime('%Y-%m-%d',time.localtime(i/1000000000) ) for i in timelist]
    # print(df_date)
    tjdf = pd.DataFrame()
    timelist =sorted(timelist)
    # 时间散点图
    df_zong = pd.DataFrame()
    for k,i in enumerate(timelist):
        df1 = df[df['s_Time'] == i]
        df0 = df1[df1['分类=预测值']==df1['分类=预测值'].max()]

        tjindex = [x for x in list(df.keys()) if str(x).startswith('未来')]
        # print(df[df['s_Time']==i][tjindex].mean())
        print(i)
        df0.sort_values(by =['回归=预测值'], ascending=True, inplace=True)
        print(df0.tail(30))
        print(df0.shape)

        df0_ = df0.iloc[-1*qian_n:]
        tjdf[i] = df0_[tjindex].mean()
        df_zong = df_zong.append(df0_)
        xdata = [t for t in df0['回归=预测值'].values.tolist()]
        yd1 = df0[['未来=月终收益', '回归=预测值', 'aim']]
        if yd1.shape[0] >qian_n:
            ydata1 = yd1.iloc[:].values.tolist()
            ydata2 = yd1.iloc[-1*qian_n:].values.tolist()
            min_pre = yd1.iloc[-1*qian_n:]['回归=预测值'].mean()
            max_pre = yd1.iloc[-1*qian_n:]['回归=预测值'].max()
        else:
            ydata1 = yd1.values.tolist()
            # ydata2 = yd1.values.tolist()
            min_pre = df0['回归=预测值'].mean()
            max_pre = df0['回归=预测值'].max()
        xdata2 = [t for t in df1['分类=预测值'].values.tolist()]
        yd2 = df1[['未来=月终收益', '分类=预测值', 'aim']].values.tolist()


        # print(ydata)
        Scatter0 = (
            Scatter()
            .add_xaxis(xdata)
            .add_yaxis('未来=月终收益',ydata1,label_opts = opts.LabelOpts(is_show=False,),)
            # .add_yaxis('未来=月终收益', ydata2, label_opts=opts.LabelOpts(is_show=False, ), symbol='triangle')

            .set_global_opts(
                xaxis_opts=opts.AxisOpts(name = '预测值：',type_= "value",axistick_opts=opts.AxisTickOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(name = '真实值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                title_opts =opts.TitleOpts(title=f"{title}==:{i}",pos_left=30),
                tooltip_opts = opts.TooltipOpts(formatter=JsCode("function (params) { return '==真实：'+params.value[1] +' <br/>== 预测：'+ params.value[2]+' <br/>== 策略：'+ params.value[3];}"),
                                                ),
                visualmap_opts=opts.VisualMapOpts(min_=min_pre,max_= max_pre),
                ))
        min_pre =  df0['未来=月终收益'].mean()
        max_pre = df0['未来=月终收益'].max()
        Scatter1 = (
            Scatter()
            .add_xaxis(xdata2)
            .add_yaxis('未来=月终收益',yd2,label_opts = opts.LabelOpts(is_show=False,),)
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(name = '预测值：',type_= "value",axistick_opts=opts.AxisTickOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(name = '真实值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),

                title_opts =opts.TitleOpts(title=f"{title}==:{i}",pos_left=30),

                tooltip_opts = opts.TooltipOpts(formatter=JsCode("function (params) { return '==真实：'+params.value[1] +' <br/>== 预测：'+ params.value[2]+' <br/>== 策略：'+ params.value[3];}"),
                                                ),
                visualmap_opts=opts.VisualMapOpts(min_=min_pre,max_= max_pre),
                ))
        tl.add(Scatter0, f"{i}")
        tl_fl.add(Scatter1, f"{i}")

    tjdf.fillna(0,inplace=True)
    tjdf = tjdf.applymap(lambda x:round(x,3) if isinstance(x,float) else x)

    # print(tjdf)
    # exit()

    tl2 = Timeline()
    for wl in tjdf.index:
        wl0 = tjdf.loc[wl]
        # print(wl0)

        line =  Line().add_xaxis(list(wl0.index)).add_yaxis(wl,wl0.values.tolist())\
            .set_global_opts(title_opts=opts.TitleOpts(title=f'前{qian_n}名策略\n统计量：{wl}—变化',pos_left=30))
        tl2.add(line,wl)
        
    # 统计表格
    talble = Table()
    # [t.strftime(format="%Y-%m-%d")for t   in tjdf.keys()]
    tjdf.columns = tjdf.columns.map(lambda x:x.strftime(format="%Y-%m-%d"))
    tjdf['统计量名称'] = tjdf.index
    tjdf['平均总计'] =tjdf.mean(axis = 1)
    # print(tjdf['总计'])
    # exit()
    # print(tjdf.values)
    tjdf = tjdf.applymap(lambda x:round(x,3) if isinstance(x,float) else x)
    tjdf_columns = list(tjdf.keys())[-2:]+list(tjdf.keys())[:-2]
    tjdf = tjdf[tjdf_columns]
    talble.add(list(tjdf.keys()),tjdf.values.tolist()).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"前{qian_n}名策略的真实未来，的历史统计表格"))

    # 策略推荐表格
    talble2 = Table()
    tjcelue_data = df[df['s_Time'] == timelist[-1]][['s_Time','canshu', 'celue_name',  '分类=预测值', '回归=预测值']]
    tjcelue_data= tjcelue_data[tjcelue_data['回归=预测值'].notnull()]
    tjcelue_data.sort_values(by='回归=预测值',ascending=True,inplace=True)
    tjcelue_data = tjcelue_data.iloc[-1*qian_n:]
    talble2.add(tjcelue_data.columns.tolist(),tjcelue_data.values.tolist()
                ).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"推荐前{qian_n}名策略"))

    talble3 = Table()
    headers_list = ['介绍：','本次筛选个数','过滤因子']
    match_df=[print_data]
    talble3.add(headers_list,match_df).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"整体介绍"))



    page = Page()
    page.add(
        tl,tl_fl,tl2,talble,talble2,talble3
    )


    page.render(path0+f"{info}.html")
    print(path0+f"{info}.html")
    return df_zong


def dong_scatter3(qian_n=10, data={},data2=[], info='', print_data=[0, 0, 0], path0=''):
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode
    from pyecharts.charts import Scatter, Timeline, Line, Page,Tab
    from pyecharts.components import Table

    # endlist = ['canshu', 'celue_name', '预测_s_Time', '最终预测值1', 'hg最终预测值1',
    #            '未来end_3', '未来max_back_3', '未来sharp_rate_3', '未来trade_nums_3', '未来win_rate_3']
    data['aim'] = data['canshu'] + data['celue_name']
    dict0 = {'hg最终预测值1': '回归=预测值', '最终预测值1': '分类=预测值', '预测_s_Time': 's_Time',
             '未来end_3': '未来=月终收益', '未来max_back_3': '未来=最大回撤', '未来sharp_rate_3': '未来=夏普率', '未来trade_nums_3': '未来=交易次数', '未来win_rate_3': '未来=胜率'}
    data.rename(columns=dict0, inplace=True)

    data = data.applymap(lambda x: round(x, 3) if isinstance(x, float) else x)

    # print(data.columns)

    # print(data.tail())
    title = f'动态图:{qian_n}:'

    df = data
    # data['回归=预测值'] = data['回归=预测值'].apply(lambda x: int(x))

    # df['未来=月终收益']=df['未来=月终收益'].apply(lambda x: int(x))
    # df['回归=预测值']=df['回归=预测值'].apply(lambda x: int(x))

    df['s_Time'] = pd.to_datetime(df['s_Time'])  # .apply(lambda x:x.strftime(format="%Y-%m-%d"))
    df.sort_values(by=['s_Time'], ascending=True, inplace=True)
    tl = Timeline()
    tl_fl = Timeline()

    timelist = list(set(df['s_Time'].tolist()))
    # sorted(timelist)
    # df_date = [time.strftime('%Y-%m-%d',time.localtime(i/1000000000) ) for i in timelist]
    # print(df_date)
    tjdf = pd.DataFrame()
    timelist = sorted(timelist)
    # 时间散点图
    df_zong = pd.DataFrame()
    for k, i in enumerate(timelist):
        df1 = df[df['s_Time'] == i]
        df0 = df1[df1['分类=预测值'] == df1['分类=预测值'].max()]

        tjindex = [x for x in list(df.keys()) if str(x).startswith('未来')]
        # print(df[df['s_Time']==i][tjindex].mean())
        print(i)
        df0.sort_values(by=['回归=预测值'], ascending=True, inplace=True)
        print(df0.tail(30))
        print(df0.shape)

        df0_ = df0.iloc[-1 * qian_n:]
        tjdf[i] = df0_[tjindex].mean()
        df_zong = df_zong.append(df0_)
        xdata = [t for t in df0['回归=预测值'].values.tolist()]
        yd1 = df0[['未来=月终收益', '回归=预测值', 'aim']]
        if yd1.shape[0] > qian_n:
            ydata1 = yd1.iloc[:].values.tolist()
            ydata2 = yd1.iloc[-1 * qian_n:].values.tolist()
            min_pre = yd1.iloc[-1 * qian_n:]['回归=预测值'].mean()
            max_pre = yd1.iloc[-1 * qian_n:]['回归=预测值'].max()
        else:
            ydata1 = yd1.values.tolist()
            # ydata2 = yd1.values.tolist()
            min_pre = df0['回归=预测值'].mean()
            max_pre = df0['回归=预测值'].max()
        xdata2 = [t for t in df1['分类=预测值'].values.tolist()]
        yd2 = df1[['未来=月终收益', '分类=预测值', 'aim']].values.tolist()

        # print(ydata)
        Scatter0 = (
            Scatter()
                .add_xaxis(xdata)
                .add_yaxis('未来=月终收益', ydata1, label_opts=opts.LabelOpts(is_show=False, ), )
                # .add_yaxis('未来=月终收益', ydata2, label_opts=opts.LabelOpts(is_show=False, ), symbol='triangle')

                .set_global_opts(
                xaxis_opts=opts.AxisOpts(name='预测值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(name='真实值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                title_opts=opts.TitleOpts(title=f"{title}==:{i}", pos_left=30),
                tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                    "function (params) { return '==真实：'+params.value[1] +' <br/>== 预测：'+ params.value[2]+' <br/>== 策略：'+ params.value[3];}"),
                                              ),
                visualmap_opts=opts.VisualMapOpts(min_=min_pre, max_=max_pre),
            ))
        min_pre = df0['未来=月终收益'].mean()
        max_pre = df0['未来=月终收益'].max()
        Scatter1 = (
            Scatter()
                .add_xaxis(xdata2)
                .add_yaxis('未来=月终收益', yd2, label_opts=opts.LabelOpts(is_show=False, ), )
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(name='预测值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(name='真实值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),

                title_opts=opts.TitleOpts(title=f"{title}==:{i}", pos_left=30),

                tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                    "function (params) { return '==真实：'+params.value[1] +' <br/>== 预测：'+ params.value[2]+' <br/>== 策略：'+ params.value[3];}"),
                                              ),
                visualmap_opts=opts.VisualMapOpts(min_=min_pre, max_=max_pre),
            ))
        tl.add(Scatter0, f"{i}")
        tl_fl.add(Scatter1, f"{i}")

    tjdf.fillna(0, inplace=True)
    tjdf = tjdf.applymap(lambda x: round(x, 3) if isinstance(x, float) else x)

    # print(tjdf)
    # exit()

    tl2 = Timeline()
    for wl in tjdf.index:
        wl0 = tjdf.loc[wl]
        # print(wl0)

        line = Line().add_xaxis(list(wl0.index)).add_yaxis(wl, wl0.values.tolist()) \
            .set_global_opts(title_opts=opts.TitleOpts(title=f'前{qian_n}名策略\n统计量：{wl}—变化', pos_left=30))
        tl2.add(line, wl)

    # 统计表格
    talble = Table()
    # [t.strftime(format="%Y-%m-%d")for t   in tjdf.keys()]
    tjdf.columns = tjdf.columns.map(lambda x: x.strftime(format="%Y-%m-%d"))
    tjdf['统计量名称'] = tjdf.index
    tjdf['平均总计'] = tjdf.mean(axis=1)
    # print(tjdf['总计'])
    # exit()
    # print(tjdf.values)
    tjdf = tjdf.applymap(lambda x: round(x, 3) if isinstance(x, float) else x)
    tjdf_columns = list(tjdf.keys())[-2:] + list(tjdf.keys())[:-2]
    tjdf = tjdf[tjdf_columns]
    talble.add(list(tjdf.keys()), tjdf.values.tolist()).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"前{qian_n}名策略的真实未来，的历史统计表格"))

    # 策略推荐表格
    talble2 = Table()
    tjcelue_data = df[df['s_Time'] == timelist[-1]][['s_Time', 'canshu', 'celue_name', '分类=预测值', '回归=预测值']]
    tjcelue_data = tjcelue_data[tjcelue_data['分类=预测值'].notnull()]
    tjcelue_data.sort_values(by='分类=预测值', ascending=True, inplace=True)
    tjcelue_data = tjcelue_data.iloc[-1 * qian_n:]
    talble2.add(tjcelue_data.columns.tolist(), tjcelue_data.values.tolist()
                ).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"推荐前{qian_n}名策略"))

    talble3 = Table()
    headers_list = ['介绍：', '本次筛选个数', '过滤因子']
    match_df = [print_data]
    talble3.add(headers_list, match_df).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"整体介绍"))

    talble4 = Table()

    new_columns = list(data2.keys())
    newdata = data2.values.tolist()
    talble4.add(new_columns, newdata).set_global_opts(title_opts=opts.ComponentTitleOpts(title=f"逐月—策略详情：文件位置："+path0.split('\\')[0]))
    tab0 = Tab()


    tab0.add(tl,'回归')
    tab0.add(tl_fl,'分类')
    tab0.add(tl2,'逐月统计图')
    tab0.add(talble,'逐月统计表格')
    tab0.add(talble2,'最新月推荐策略')
    tab0.add(talble3,'备注描述')
    tab0.add(talble4,'每月策略详情表')

    tab0.render(path0 + f"{info}.html")
    print(path0 + f"{info}.html")
    return df_zong


def corr_plot(data:list):
    from pyecharts import options as opts
    from pyecharts.charts import HeatMap
    from pyecharts.faker import Faker

    corr = (
        HeatMap()
        .add_xaxis('series0')
        .add_yaxis(
            "series1",
            data,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="HeatMap-Label 显示"),
            visualmap_opts=opts.VisualMapOpts(),
        )
        .render("相关性图.html")
    )