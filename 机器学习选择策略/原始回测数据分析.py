import pandas as pd
import pickle
pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)
pd.set_option('display.float_format', lambda x: '%.5f' % x)








# 查看每个指标的总体收益分布
def anylize_to_html_01(data={}, info=''):
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode
    from pyecharts.charts import Scatter,Bar, Timeline, Line, Page,Tab
    from pyecharts.components import Table
    tab0 = Tab()
    data.loc[(data['max_back_1'] !=0), '赢撤比'] = -1 * data['end_1'] / data['max_back_1']
    data.loc[(data['max_back_1'] ==0), '赢撤比'] = 0


    data =data[data['end_1'] !=0]
    clm = list(set(data['celue_name'].tolist()))
    print(clm)

    title = f'所有参数的，每个指标的统计状态:'
    tjdf = pd.DataFrame()
    i=0
    tj_index = {'end_1':'最终收益','sharp_rate_1':'夏普率',
                'max_back_1':'最大回撤','win_rate_2':'胜率',
                'trade_nums_1':'交易次数','profit_days_1':'盈利天数',
                '赢撤比':'赢撤比'}

    s_tj_index = []
    for clm0 in clm:
        print(clm0)
        df = data[data['celue_name'] == clm0].copy()

        for k,v in df.groupby('para1'):
            print(k)
            index_0 = i + 1
            i=i+1
            tjdf.loc[index_0,'para1'] = k
            tjdf.loc[index_0,'clm'] = clm0

            for tj_ix in tj_index.keys():
                tjdf.loc[index_0,tj_index[tj_ix]] = v[tj_ix].mean()
            tjdf.loc[index_0,'赢撤比2'] = -1*v['end_1'].mean()/v['max_back_1'].mean() if v['max_back_1'].mean() !=0 else 0

    print(tjdf)
    print(tjdf.keys())
    tj_index2 = list(tjdf.keys())
    tj_index2.remove('para1')
    tj_index2.remove('clm')
    clm2 = tj_index2
    for x,clm0 in enumerate(clm):
        df0 = tjdf[tjdf['clm'] == clm0].copy()
        tl = Timeline()
        for k, i in enumerate(tj_index2):
            xdata,ydata1=df0['para1'],df0[['para1',i]]
            print(xdata)
            print(ydata1)
            # exit()
            bar0 = (
                Bar()
                    .add_xaxis(xdata.tolist())
                    .add_yaxis('均值', ydata1.values.tolist(), label_opts=opts.LabelOpts(is_show=False, ), )
                    .set_global_opts(
                    xaxis_opts=opts.AxisOpts(offset=0,name='参数名：', type_="value", is_show=True,axistick_opts=opts.AxisTickOpts(is_show=True)),
                    yaxis_opts=opts.AxisOpts(offset=0,name='均值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                    title_opts=opts.TitleOpts(title=f"{title}==:{i}"),
                    legend_opts = opts.LegendOpts(pos_right=True),
                    tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                        "function (params) { return '==para1：'+params.value[0] +' <br/>== 指标：'+ params.value[1]}"),
                    ),
                ))


            tl.add(bar0, f"{i}")
        tab0.add(tl, clm0)

    for x, clm0 in enumerate(clm2):
        tl = Timeline()
        tjdf0 =tjdf[tjdf['最终收益']!=0].copy()
        tjdf0.sort_values(clm0,ascending=True,inplace=True)
        df0 = tjdf0.iloc[-1*int(tjdf0.shape[0]*0.12):].copy()
        index_list = list(df0.keys())
        index_list.remove('clm')
        df0 = df0[index_list]
        for k, i in enumerate(index_list):
            xdata, ydata1 = df0['para1'], df0[[i,'para1']]
            print( ydata1)
            sca0 = (
                Scatter()
                    .add_xaxis(xdata.tolist())
                    .add_yaxis('分布值', ydata1.values.tolist(), label_opts=opts.LabelOpts(is_show=False, ), )
                    .set_global_opts(
                    xaxis_opts=opts.AxisOpts(offset=0,name='参数名：', type_="value", is_show=True, axistick_opts=opts.AxisTickOpts(is_show=True)),
                    yaxis_opts=opts.AxisOpts(offset=0
                                             , name='分布值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                    title_opts=opts.TitleOpts(title=f"{title}==:{i}"),
                    legend_opts=opts.LegendOpts(pos_right=True),
                    tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                        "function (params) { return '==para1：'+params.value[0] +' <br/>== 指标：'+ params.value[1]}"),
                    ),
                ))

            tl.add(sca0, f"{i}")
        tab0.add(tl, clm0)


    tab0.render(f"{info}.html")
    print(f"{info}.html")
    return

# 细分
def anylize_to_html_02(data={}, fl_style='para1',info='',add_info=''):
    # 根据生成的原始数据
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode
    from pyecharts.charts import Scatter,Bar, Timeline, Line, Page,Tab
    from pyecharts.components import Table
    tab0 = Tab()
    data.loc[(data['max_back_1'] !=0), '赢撤比'] = -1 * data['end_1'] / data['max_back_1']
    data.loc[(data['max_back_1'] ==0), '赢撤比'] = 0

    data =data[data['end_1'] !=0]

    clm = list(set(data['celue_name'].tolist()))
    print(clm)

    title = f'所有参数的，每个指标的统计状态:'
    tjdf = pd.DataFrame()
    tj_dict = {}
    # 生成分类数据tjdf
    i=0
    tj_index = {'end_1':'最终收益','sharp_rate_1':'夏普率',
                'max_back_1':'最大回撤','win_rate_2':'胜率',
                'trade_nums_1':'交易次数','profit_days_1':'盈利天数',
                '赢撤比':'赢撤比'}
    for clm0 in clm:
        print(clm0)
        df = data[data['celue_name'] == clm0].copy()

        for k,v in df.groupby(fl_style):
            print(k)
            index_0 = i + 1
            i=i+1
            tjdf.loc[index_0,fl_style] = k
            tjdf.loc[index_0,'clm'] = clm0

            for tj_ix in tj_index.keys():
                tjdf.loc[index_0,tj_index[tj_ix]] = v[tj_ix].mean()
            tjdf.loc[index_0,'赢撤比2'] = -1*v['end_1'].mean()/v['max_back_1'].mean() if v['max_back_1'].mean() !=0 else 0

    print(tjdf)
    print(tjdf.keys())
    # exit()
    filter_col ='夏普率'

    df0_dict = {} #策略名+参数的字典。
    tjdf2 = pd.DataFrame()
    for x,clm0 in enumerate(clm):
        # 每个策略类，按照filter_col排序，根据每个策略回测数量，选取比较高的排名的策略
        df0 = tjdf[tjdf['clm'] == clm0].copy()
        df0.sort_values(filter_col,ascending=True,inplace=True)
        df0 = df0.iloc[-1*int(df0.shape[0]*0.35):]
        df0_dict[str(df0.iloc[-1]['clm'])] = list(sorted(set(df0[fl_style].tolist())))
        tjdf2 =tjdf2.append(df0,ignore_index=True)
    tjdf2.sort_values('最终收益',ascending=True,inplace=True)
    tjdf2 =tjdf2.iloc[int(tjdf2.shape[0]*0.1):]
    print(tjdf2)
    tj_index2 = list(tjdf2.keys())
    tj_index2.remove(fl_style)
    tj_index2.remove('clm')
    clm2 = tj_index2
    for x, clm0 in enumerate(clm2):
        tl = Timeline()
        tjdf0 =tjdf2[tjdf2['最终收益']!=0].copy()
        tjdf0.sort_values(clm0,ascending=True,inplace=True)
        # df0 = tjdf0.iloc[-1*int(tjdf0.shape[0]*0.12):].copy()
        df0 = tjdf0
        index_list = list(df0.keys())
        index_list.remove('clm')
        df0 = df0[index_list]
        for k, i in enumerate(index_list):
            xdata, ydata1 = df0[fl_style], df0[[i,fl_style]]
            print(add_info)
            # print( ydata1)
            sca0 = (
                Scatter()
                    .add_js_funcs(f"document.write('%s')"%add_info)
                    .add_xaxis(xdata.tolist())
                    .add_yaxis('分布值', ydata1.values.tolist(), label_opts=opts.LabelOpts(is_show=False, ), )
                    .set_global_opts(
                    xaxis_opts=opts.AxisOpts(offset=0,name='参数名：', type_="value", is_show=True, axistick_opts=opts.AxisTickOpts(is_show=True)),
                    yaxis_opts=opts.AxisOpts(offset=0
                                             , name='分布值：', type_="value", axistick_opts=opts.AxisTickOpts(is_show=True)),
                    title_opts=opts.TitleOpts(title=f"{title}==:{i}"),
                    legend_opts=opts.LegendOpts(pos_right=True),
                    tooltip_opts=opts.TooltipOpts(formatter=JsCode(
                        "function (params) { return '==canshu：'+params.value[0] +' <br/>== 指标：'+ params.value[1]}"),
                        ),)
            )

            tl.add(sca0, f"{i}")
        tab0.add(tl, clm0)


    tl.render(f"{info}.html")
    print(f"{info}.html")
    return df0_dict


def to_train_data01(data,df_dict,fl_style='para1'):
    #将传入的原始的data数据，按照df——dict的分类方式（筛选方式进行分类），在整合到一起保存。
    # 之后需要调用to_train_data01进行分类保存成学习数据
    to_train_data0 = pd.DataFrame()
    for k,v in df_dict.items():
        data0 = data[data['celue_name']==k]
        for i in v:
            d0 = data0[data0[fl_style]==i]
            to_train_data0=to_train_data0.append(d0)

    print(to_train_data0.iloc[-100:])
    to_train_data0.to_pickle(f'{fl_style}_traindata.pkl')
    return to_train_data0


def to_teain_data02(df1,docu_path = r'F:\new_0811\huice_log\train_res\para1_traindata'):
    #将df1数据分成多类，以满足，一次学习的所需要数据！

    for i in ['01', '02', '03', '04']:
        new_dict = {}
        for clm in sorted(set(df1['celue_name'].tolist())):
            if str(clm).split('_')[-2] == i:
                df0 = df1[df1['celue_name'] == clm].copy()
                df0.sort_values('res_time', inplace=True)
                new_dict[clm] = {}
                for y in sorted(set(df0['res_time'].tolist())):
                    new_dict[clm][y] = df0[df0['res_time'] == y].copy()
                print(new_dict[clm].keys())
                # exit()
        print(new_dict.keys())

        with open(docu_path + r'\para_mebd%s.pkl' % i, 'wb') as f:
            pickle.dump(new_dict, f)
            print(docu_path + r'\para_mebd%s.pkl' % i)

if __name__ == '__main__':



    if 1 == True:
        df =pd.DataFrame()
        for bh in [
            '01'
            ,'02'
            ,'03'
            ,'04'
        ]:
            df0 = pd.DataFrame()
            train_path = r'F:\new_0811\huice_log\train_res\MEBD_%s_每月训练数据_周期1_回测周期3_10-13.pickle' % str(bh)
            print(train_path)
            with open(train_path, 'rb') as f:
                file_0 = pickle.load(f)
            for k, v in file_0.items():
                print(k)
                if str(k).endswith('3'): continue
                for t in v.keys():
                    print(t)
                    df = df.append(v[t], ignore_index=True)
                    df0 = df0.append(v[t], ignore_index=True)

        df.sort_values('res_time', inplace=True)
        df.reset_index(inplace=True, drop=True)
        print(df.sample(100))
        print(df.shape)
        df.to_pickle('all_data_df.pkl')

    if 0 == 1:
        # df1 = pd.read_pickle(r'F:\new_0811\all_data_df策略01类.pkl')
        add_info = '策略分类分析处理选择步骤：<br/>' \
        '1.筛选每个策略的按照某指标排序的参数前n名。<br/>' \
        '2.所有策略统计到一起。形成一个df表格。<br/>' \
        '3.df进行从原始回测文件里面提取相应的策略。<br/>'
        df1 = pd.read_pickle(r'F:\new_0811\all_data_df.pkl')
        print(df1.tail())
        print(df1.shape)
        print(sorted(set(df1['celue_name'].tolist())))
        df_dict = anylize_to_html_02(data=df1,fl_style='para2',info='para2_fl_df',add_info =add_info)
        df = to_train_data01(data=df1,df_dict=df_dict,fl_style='para2')
        to_teain_data02(df1=df, docu_path=r'F:\new_0811\huice_log\train_res\para2_traindata')
