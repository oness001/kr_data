
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
# 设置 DataFrame 的列名与列对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# 设置 DataFrame 的每列都可以显示
pd.set_option('display.max_columns',500)
pd.set_option('display.width',5000)
# pd.set_option('mode.chained_assignment',None)


def facter_deal(data_df,f_s=["factor"],wlsy=["wlyl","datetime"]):
    data_df.fillna(0,inplace=True)
    data_df["一阶差分"] = data_df[f_s[0]].diff(1)
    data_df["二阶差分"] = data_df["一阶差分"].diff(1)
    data_df["10_ma"] = data_df[f_s[0]].rolling(10).mean()
    data_df["tahn"] = np.tanh(data_df[f_s[0]])
    data_df["sigmoid"] = 1 / (1 + np.exp(data_df[f_s[0]]))

    data_df["一阶差分10_ma"] = pd.Series(data_df["一阶差分"]).rolling(10).mean()
    data_df["一阶差分tahn"] = np.tanh(data_df["一阶差分"])
    data_df["一阶差分sigmoid"] = 1 / (1 + np.exp(data_df["一阶差分"]))

    data_df["二阶差分10_ma"] = pd.Series(data_df["二阶差分"]).rolling(10).mean()
    data_df["二阶差分tahn"] = np.tanh(data_df["二阶差分"])
    data_df["二阶差分sigmoid"] = 1 / (1 + np.exp(data_df["二阶差分"]))

    return data_df

def adf_time_stats(s):
    # print(type(s))
    s.fillna(0,inplace=True)
    dftest = adfuller(s)
    dfoutput = pd.Series()
    for key,value in zip(['adf_测试值', 'p_值(<0.05)', '延迟', '测试次数'],dftest[:4]):
        dfoutput[key] = round(value,6)
    for key, value in dftest[4].items():
        dfoutput['原假设小于_(%s)' % key] = str(round(value,6)) +" 真假："+ str(dfoutput['adf_测试值']<value)

    dfoutput['mean' ] = s.mean()
    dfoutput['方差' ] = s.var()
    dfoutput['std' ] = s.std()
    dfoutput['偏度' ] = s.skew()
    dfoutput['峰度' ] = s.kurt()
    return dfoutput
def norm_test(a):
    from scipy import stats
    # import numpy as np
    a.dropna(inplace=True)
    # kstest（K-S检验）
    K,p = stats.kstest(a, 'norm')
    # print(f"kstest（K-S检验）:",K, round(p,5) , p>0.05)

    # # normaltest
    # N,p = stats.normaltest(a)
    # print(f"normaltest:",N,p,N>p)
    #
    # # Anderson-Darling test
    # A,C,p = stats.anderson(a,dist='norm')
    # print(f"Anderson test:",A,C,A<C,p)
    return p

if __name__ =="__main__":
    import dash
    from dash.dependencies import Input, Output
    import dash_table
    import dash_core_components as dcc
    import dash_html_components as html
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import plotly.figure_factory as ff
    import numpy as np


    adf_seq = np.round(np.random.normal(0, 1.2, size=(1035,)), 2)

    a = 10
    b= 10
    dash =1
    if a == 1:

        df_res = facter_deal(adf_seq)
        res_zong = pd.DataFrame()
        for key in df_res.keys():
            p = norm_test(a=df_res[key])
            dfoutput = adf_time_stats(s=df_res[key])
            dfoutput['name:'] = key
            dfoutput['norm(<0.05):'] = p
            dfoutput['data'] = df_res[key]

            res_zong=res_zong.append(dfoutput,ignore_index=True)

            # desc_msg = "\n1.p值越小越好，要求小于给定的显著水平，p值小于0.05，等于0最好。"
            # desc_msg += "\n2.t%值，ADF值要小于t%值，1%， 5%， 10% 的三个level，都是一个临界值，\n    如果小于这个临界值，说明拒绝原假设。"
            # desc_msg+="\n3.其中，1% ： 严格拒绝原假设； \n    5%： 拒绝原假设； \n    10% 以此类推，程度越来越低。\n    如果，ADF小于1% level， 说明严格拒绝原假设。\n"
            # print(desc_msg)
        res_zong = res_zong[['name:','adf_测试值','p_值',"norm(<0.05):", 'mean', 'std', '偏度','峰度', '原假设小于 (1%)—的值','原假设小于 (5%)—的值', '原假设小于 (10%)—的值',   '延迟', '方差', '测试次数',"data"]]

        res_zong=res_zong.round(4)


        # print(res_zong['name'].to_list())

        hist_data = [res_zong.iloc[int(i)]["data"] for i in range(res_zong.shape[0])]
        print(res_zong)
        print(hist_data)
        group_labels = res_zong['name:'].to_list()  # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels,show_hist=True,show_curve=True, bin_size=0.1)
        fig.show()


        exit()
        exit()
        stats.probplot(df_res[key], dist="norm", plot=plt)
        plt.show()
    if  b ==  1 :
        from sklearn.datasets import load_iris
        import numpy as np

        # 导入IRIS数据集
        iris = load_iris()
        iris = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_legth', 'petal_width'])
        if True:
            #一个总体均值的z检验
            np.mean(iris['petal_legth'])
            '''
            原假设：鸢尾花花瓣平均长度是4.2 
            备择假设：鸢尾花花瓣平均长度不是4.2
            '''

            import statsmodels.stats.weightstats

            z, pval = statsmodels.stats.weightstats.ztest(iris['petal_legth'], value=4.2)
            print(z, pval)

            '''
            P=0.002 <5%, 拒绝原假设，接受备则假设。
            '''
        if True:
            import scipy.stats
            t, pval = scipy.stats.ttest_1samp(iris['petal_legth'], popmean=4.0)
            print(t, pval)
            '''
            P=0.0959 > 5%, 接受原假设，即花瓣长度为4.0。 
            '''
    if dash ==1:
        import plotly.graph_objects as go  # or plotly.express as px
        fig = go.Figure()
        import dash

        df_res = facter_deal(adf_seq)
        res_zong = pd.DataFrame()
        for key in df_res.keys():
            p = norm_test(a=df_res[key])
            dfoutput = adf_time_stats(s=df_res[key])
            dfoutput['name:'] = key
            dfoutput['norm(<0.05):'] = p
            dfoutput['data'] = df_res[key]

            res_zong = res_zong.append(dfoutput, ignore_index=True)
            # desc_msg = "\n1.p值越小越好，要求小于给定的显著水平，p值小于0.05，等于0最好。"
            # desc_msg += "\n2.t%值，ADF值要小于t%值，1%， 5%， 10% 的三个level，都是一个临界值，\n    如果小于这个临界值，说明拒绝原假设。"
            # desc_msg+="\n3.其中，1% ： 严格拒绝原假设； \n    5%： 拒绝原假设； \n    10% 以此类推，程度越来越低。\n    如果，ADF小于1% level， 说明严格拒绝原假设。\n"
            # print(desc_msg)
        res_zong = res_zong[['name:', 'adf_测试值', 'p_值', "norm(<0.05):", 'mean', 'std', '偏度', '峰度', '原假设小于 (1%)—的值', '原假设小于 (5%)—的值', '原假设小于 (10%)—的值', '延迟', '方差',
             '测试次数', "data"]]
        res_zong = res_zong.round(4)
        hist_data = [res_zong.iloc[int(i)]["data"] for i in range(len(res_zong))]
        group_labels = [res_zong.iloc[int(i)]["name:"] for i in range(len(res_zong))]
        print(hist_data)
        fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_curve=True, bin_size=0.1)
        # print(res_zong[res_zong.keys()])

        # fig.show()
        app = dash.Dash(__name__)
        print(res_zong.keys()[:-2])
        # exit()

        app.layout = html.Div([html.Div([dcc.Graph(figure=fig)]),
                              dash_table.DataTable(id='datatable-paging',
                                                    columns=[{"name": i, "id": i} for i in res_zong.keys()[:-1]],
                                                    data = res_zong[res_zong.keys()[:-1]].to_dict('records'),
                                                   )])

        app.run_server(debug=True, use_reloader=False)
