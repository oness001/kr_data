import os
import signal
import traceback
import pandas as pd
import numpy as np
import sys, time
import pickle
import datetime as dt

from dateutil.relativedelta import relativedelta

from vnpy.event import EventEngine
from vnpy.app.ib_cta_backtester.engine import BacktesterEngine



pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


from typing import List
from vnpy.trader.object import TradeData


def flag_encoding(train_data,to_col='celue_name',new_col = 'clm'):

    clm = set(sorted(train_data[to_col].to_list()))
    print(f'需要生成flag：{to_col}')
    clm_dict= {}
    for i, c in enumerate(clm):
        clm_dict[c]={f'{new_col}{i}':1}
    for ix, c in enumerate(clm_dict.keys()):
        for i, c0 in enumerate(clm):
            if f'{new_col}{i}' in clm_dict[c].keys():continue
            else: clm_dict[c][f'{new_col}{i}'] = 0

    for i in clm:
        con = train_data[to_col] == i
        # 同名策略名统一命名。
        for m,v in clm_dict[i].items():
            train_data.loc[con,m]=v

    # print(clm_dict)
    return train_data


# 特征因子计算
def  cal_yinzi0(celuename,train_da0, s_t, e_t, res_t, train_res):
    '''
    :param d: 计算周期因子
    :param s_t: 开始周期时间
    :param now_time: 当前时间
    :param train_res: 【是否要传入训练结果，样本训练结果】
    :return:
    ['end', 'max_back', 'trade_nums', 'sharp_rate', 'all_tradedays',
       'profit_days', 'win_rate', 'win_mean_num', 'loss_mean_num', 'max_num',
       'min_num', 'mean_num', 'std_num',

    '''
    df_zong0 = pd.DataFrame()  # 保存参数的统计因子
    # 计算每个参数的统计因子
    train_da0 = train_da0.copy()
    # 遍历所有参数，计算本期的统计量
    for i, d in train_da0.groupby('canshu'):
        try:
            d.fillna(0, inplace=True)
            # 训练开始
            df_zong0.loc[i, 'train_s'] = s_t
            # 训练结束
            df_zong0.loc[i, 'train_e'] = e_t
            #  预测时间
            df_zong0.loc[i, 'res_t'] = res_t

            df_zong0.loc[i, 'canshu'] = i
            df_zong0.loc[i, 'celuename'] = celuename
            if float(d['end'].sum()) == 0 or d['end'].sum() == np.NAN:
                continue

            df_zong0.loc[i, '本周期总收益'] = float(d['end'].sum())
            df_zong0.loc[i, '最近周期收益'] = float(d.iloc[-1]['end'].sum())
            df_zong0.loc[i, '最大回撤'] = d['max_back'].min()
            df_zong0.loc[i, '最大值'] = (d['end'].cumsum()).max()
            df_zong0.loc[i, '收益std'] = (d['end'].std())
            df_zong0.loc[i, '偏度'] = (d['end'].skew())
            df_zong0.loc[i, '峰度'] = (d['end'].kurt())

            df_zong0.loc[i, '平均月收益'] = d['end'].mean()
            df_zong0.loc[i, '平均月最大收益'] = d['max_sum'].mean()
            df_zong0.loc[i, '平均月最大回撤'] = d['max_back'].mean()
            df_zong0.loc[i, '平均月夏普率'] = d['sharp_rate'].mean()
            df_zong0.loc[i, '平均月交易次数'] = d['trade_nums'].mean()
            df_zong0.loc[i, '月均交易天数'] = d['total_days'].mean()
            df_zong0.loc[i, '月均盈利天数'] = d['profit_days'].mean()
            df_zong0.loc[i, '月均开单收益std'] = d['std_num'].mean()
            df_zong0.loc[i, '月均开单最大收益'] = d['max_num'].mean()
            df_zong0.loc[i, '月均亏单平均亏损'] = d['loss_mean_num'].mean()

            df_zong0.loc[i, '月均胜单平均盈利'] = d['win_mean_num'].mean()
            df_zong0.loc[i, '月均胜单平均盈利偏度'] = d['win_mean_num'].skew()
            df_zong0.loc[i, '月均胜单平均盈利std'] = d['win_mean_num'].std()

            df_zong0.loc[i, '月均交易胜率'] = d['win_rate'].mean()
            df_zong0.loc[i, '月均交易胜率偏度'] = d['win_rate'].skew()
            df_zong0.loc[i, '月均交易胜率std'] = d['win_rate'].std()

            df_zong0.loc[i, '月均开单平均收益'] = d['mean_num'].mean()
            df_zong0.loc[i, '月均开单平均收益偏度'] = d['mean_num'].skew()
            df_zong0.loc[i, '月均开单平均收益std'] = d['mean_num'].std()

            df_zong0.loc[i, '回撤std'] = (d['max_back'].std() * -1)

            df_zong0.loc[i, '盈撤比'] = (d['max_sum'].mean() / (-1 * d['max_back'].mean())) if (d['max_back'].mean()) != 0 else 0
            df_zong0.loc[i, '盈利因子01'] = d['max_sum'].sum() * d['end'].mean()  / (d['end'].std()) if (d['end'].std() != 0) else 0

            if train_res.empty:
                df_zong0.loc[i, '预测周期真实收益'] = float(0)  # c=1, y=1,
            else:
                # 训练结果
                d1 = train_res
                df_zong0.loc[i, '预测周期真实收益'] = float(d1.loc[d1['canshu'] == i, 'end'].sum())

        except Exception as e:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            print('单参数统计出错', d.tail())
        finally:
            df_zong0.fillna(0, inplace=True)
            # print(df_zong0.tail())

    return df_zong0


# 计算所有数据的特征因子
def cal_all_yinzi(celuename,zong_t, hc_zq, gd_zq, data):
    '''
    生成目标月份的统计数据。
    :param data: 原始数据
    :param index_name:统计列表名字
    :return: 以时间为序列的统计特征字典或者df
      因子1，因子2。。。
    dt1
    dt2
    。
    。
    。
    '''
    yinzi_dict = {}
    try:
        i = 0
        # 月份循环
        while True:
            pass
            # 回测的时间
            hc_end = zong_t[-1]
            now_t0 = zong_t[0] + relativedelta(months=int(i + hc_zq))
            train_s0 = zong_t[0] + relativedelta(months=int(i))
            train_e0 = now_t0 - relativedelta(months=int(gd_zq))
            # 当前月为，训练结果月，大于回测结束时间，训练结果为空，一旦训练数据的结束月大于回测结束时间，跳出。
            if now_t0 > hc_end:
                train_res0 = pd.DataFrame()
                if train_e0 > zong_t[-1]:
                    break
            else:train_res0 = data[data['s_time'] == now_t0].copy()

            # 本次训练数据
            train_da0 = data[data['s_time'] >= train_s0]
            train_da0 = train_da0[train_da0['s_time'] <= train_e0]

            train_da0.sort_values(by=['s_time'], ascending=True, inplace=True)

            print('训练数据月份：',now_t0.strftime(format="%Y-%m-%d"))

            dt_yinzi = cal_yinzi0(celuename,train_da0, train_s0, train_e0, res_t=now_t0, train_res=train_res0)

            yinzi_dict[now_t0.strftime(format="%Y-%m-%d")] = dt_yinzi

            if dt_yinzi.empty == False:
                print(f'{now_t0}完成，为空值。')
                # print(dt_yinzi.sample(5))
            else:
                print(f'*****'*3)
            print(f',最后:')
            print(dt_yinzi.tail())
            i = i + gd_zq

    except Exception as e:
        print(traceback.format_exc())
        print(e)

    print(f'所有的统计数据生成！\n特征列名字：{yinzi_dict[zong_t[-1].strftime(format="%Y-%m-%d")].columns}')
    # exit()

    return yinzi_dict

# 训练前的数据处理
def data_precess(train_data,yinzi = [], pre_style = 'max_min'):
    import sklearn.preprocessing as spp

    train_data.fillna(0, inplace=True)
    train_data = train_data[train_data['平均月收益'] != 0].copy()
    train_data0 = pd.DataFrame()

    # 数据处理
    if pre_style == 'max_min':
        train_data0 = spp.MinMaxScaler().fit_transform(train_data[yinzi])

    elif pre_style == 'max_abs':
        train_data0 = spp.MaxAbsScaler().fit_transform(train_data[yinzi])

    elif pre_style == 'standar':
        train_data0 = spp.StandardScaler().fit_transform(train_data[yinzi])

    elif pre_style == 'normal':
        train_data0 = spp.Normalizer().fit_transform(train_data[yinzi])


    train_data0 = pd.DataFrame(train_data0, columns=yinzi, index=train_data.index)
    train_data0.loc[:, '预测周期真实收益'] = pd.Series(train_data['预测周期真实收益'])

    return train_data0


def data_fliter_fb(data):

    # 过滤算法

    data = data[data['平均月收益'] > data['平均月收益'].mean()]
    # data = data[data['效率因子01'] > 0]


    return pd.DataFrame(data)


# 生成训练集
def generate_train_data(da,zong_t = [], hc_zq=6,gd_zq=1):
    '''

    以实际结果为训练结果，训练结果的月份为标记月份，

    最总生成，以结果月为序列的训练集。
    最后一个月为预测数据集，月份比传入最后回测日期大一个月，最后一个月的结果为零，就是训练集的结果为零。

    将下周期的结果来验证本周期的选择

    :return:
    '''

    zong_t = zong_t
    # 滚动周期
    gd_zq = gd_zq
    # 每次回测周期
    hc_zq = hc_zq
    # 总周期数据进行处理

    #  注意，必须先进行字段处理干净，才能选择日期，否则报错。
    da['s_time'] = pd.to_datetime(da['s_time'])
    da = da[pd.to_datetime(da['s_time'])>=zong_t[0]]
    da = da[ pd.to_datetime(da['s_time'])<=zong_t[-1]]

    #   简单处理， 目标特征设置为float格式
    to_nums = ['para1', 'para2', 'para3', 'para4','win_rate', 'win_mean_num', 'loss_mean_num', 'max_num', 'min_num',
                'mean_num', 'std_num', 'max_sum', 'end', 'max_back', 'trade_nums',
                'sharp_rate', 'total_days', 'profit_days', 'total_slippage']
    da[to_nums] = da[to_nums].applymap(float)
    da['canshu'] = da['canshu'].apply(lambda x:str(x))
    celuename = da.iloc[-1]['celue_name']
    da['canshu_'] = da['canshu']
    da.set_index('canshu_', drop=True, inplace=True)
    #  上小下大
    da.sort_values(by=['s_time'], ascending=True, inplace=True)
    da.fillna(0,inplace=True)
    #   计算da数据的因子(统计指标）==》生成所有月份的训练数据集
    print(da.head(2))
    print(da.tail(2))
    # exit()
    df_yinzi = cal_all_yinzi(celuename,zong_t,hc_zq,gd_zq,da)

    # print(df_yinzi.sample(10))
    # exit()
    #   查看一下。
    for i,v in df_yinzi.items():
        print(i)
        print(v.tail(3))
        print('=='*10)
    #  保存成pickle
    # path = pkl_path0 + str(celuename) + 'train_data.pickle'
    # with open(path,mode='wb') as f:
    #     pickle.dump(df_yinzi, f)
    # print(f'训练数据集，保存完毕！\n{path}')
    print(f'训练数据集，生成完毕！')

    return df_yinzi

# 学习，预测
def train_and_predict(train_data,res_path,algo,res_flag= '.pickle' ,zong_t = [], hc_zq=6, gd_zq=1):

    # 导入训练集
    df_yinzi = train_data.copy()
    print(df_yinzi.keys())
    # exit()

    i = 0
    df_zong = {}  # 收集每一阶段的优势参数
    while True:

        # 当前的回测的时间=训练结果月
        now_t0 = zong_t[0] + relativedelta(months=int(i + hc_zq))
        now_ = now_t0.strftime(format="%Y-%m-%d")
        # 训练结束时间
        train_t0 = now_t0 - relativedelta(months=int(gd_zq))
        # 预测结束时间，预测结果月
        pre_t0 = now_t0 + relativedelta(months=int(gd_zq))
        pre_ = pre_t0.strftime(format="%Y-%m-%d")
        print('----------')

        print(train_t0,'=====>',now_,pre_)

        # 训练数据结束月=回测结束，意味着，没有训练结果了
        if now_t0 > zong_t[-1]:
            break

        df_zong[now_t0] = {}
        # 训练数据
        train_da0 =  pd.DataFrame(df_yinzi[now_].copy())
        # exit()
        # 预测数据
        pre_da0 =  pd.DataFrame(df_yinzi[pre_].copy())
        train_da0['canshu_index'] =train_da0['canshu']
        pre_da0['canshu_index'] =pre_da0['canshu']

        train_da0.set_index('canshu_index', drop=True, inplace=True)
        pre_da0.set_index('canshu_index', drop=True, inplace=True)
        # print(train_da0.tail())
        # print(pre_da0.tail())
        # exit()

        # 机器学习预测
        try:
            print(now_t0, '开始训练。。。')
            yinzi0 = [  '本周期总收益',
       '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
       '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
       '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
       '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
       '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']

            res0 = ['预测周期真实收益', '预期偏差', 'end_预测值']

            # 训练
            model, df_pre0 = eval(algo[0])(train_data=train_da0, model_list={}, yinzi=yinzi0,Train=True)
            if isinstance(df_pre0, str):
                print(f'{df_pre0}，下一循环。')
                df_zong[now_t0]['预测数据'] = pd.DataFrame()
                df_zong[now_t0]['预测model'] = model
                i= i+gd_zq
                continue
            else:
                df_zong[now_t0]['预测model'] = model


            res_index = 'end_预测值'
            # 预测# 最后一个月，只给出预测数据，
            if now_t0 > zong_t[-1]:
                model_list, pre_data = eval(algo[0])(train_data=pre_da0, model_list=model, yinzi=yinzi0, Train=False,last=True)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，最后一个月，无预测数据。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                else:
                    pre_data.dropna(axis=0, subset=[res_index], inplace=True)
                    pre_data.fillna(0, inplace=True)
                    pre_data.sort_values(by=[res_index], ascending=True, inplace=True)
                    df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
                    break
            #  预测
            else:
                # 预测
                model_list, pre_data = eval(algo[0])(train_data=pre_da0, model_list=model,yinzi=yinzi0, Train=False)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，下一循环。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                    i = i + gd_zq
                    continue

            # print(pre_data)

            pre_data.dropna(axis=0, subset=[res_index], inplace=True)
            pre_data.fillna(0, inplace=True)
            pre_data.sort_values(by=[res_index], ascending=True, inplace=True)
            # print(pre_data.columns)

            res = ['celuename', 'end_预测值','预测周期真实收益', 'canshu', '1_预测值', '2_预测值', 'train_s',
       'train_e', 'res_t',  '本周期总收益', '最近周期收益', '最大回撤', '最大值',
       '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益', '平均月最大回撤', '平均月夏普率', '平均月交易次数',
       '月均交易天数', '月均盈利天数', '月均开单收益std', '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利',
       '月均胜单平均盈利偏度', '月均胜单平均盈利std', '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std',
       '月均开单平均收益', '月均开单平均收益偏度', '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
            if pre_data.empty:pass
            else:print(pre_data[res].tail())

            df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
            i =i+gd_zq
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            i =i+1
            continue

    print(df_zong.keys())

    res_path += res_flag
    with open(res_path,mode='wb') as f:
        pickle.dump(df_zong,f)
        print(f'逐月训练结果，已保存！to:\n{res_path}')

    return res_path

def pre_mode(train_path,res_path0,zong_t,res_flag = f'_pre_res12.pickle',algo=['cal_zuhe3'],mode=2,time_lidu=1):
        with open(train_path, 'rb') as fe:
            train_res = pickle.load(fe)
            if mode == 1:
                for k, v in train_res.items():
                    print(k)
                    train_and_predict(train_data=v,res_path = res_path0,algo =algo,res_flag= f'\s_ftx\\非贪心{k}'+res_flag, zong_t=zong_t, hc_zq=6, gd_zq=1)
                    time.sleep(2)

            if mode == 2:
                train_data={}
                # print(list(train_data.keys())[0])
                for i in train_res[list(train_res.keys())[0]].keys():
                    df_ =pd.DataFrame()
                    for k, v in train_res.items():
                        if k.split('_')[-1] == str(time_lidu):
                            df_ = df_.append(v[i],ignore_index=True)
                    train_data[i] = df_
                    # print(i)
                    # print(df_.shape)
                    # print(set(train_data[i]['celuename'].to_list()))
                    # exit()

                train_and_predict(train_data=train_data,res_path = res_path0,algo =algo,res_flag= f'\\ftx\\非贪心集中'+res_flag, zong_t=zong_t, hc_zq=6, gd_zq=1)
                time.sleep(5)

            if mode == 3:
                for k, v in train_res.items():
                    cum_df_ = pd.DataFrame()
                    train_data = {}
                    for t in v.keys():
                        cum_df_ = cum_df_.append(v[t],ignore_index=True)
                        train_data[t] = cum_df_
                        # print(k,t,cum_df_.shape)
                    # exit()
                    train_and_predict(train_data=train_data,res_path = res_path0,algo =algo,res_flag= f'\s_tx\\贪心{k}'+res_flag, zong_t=zong_t, hc_zq=6, gd_zq=1)
                    time.sleep(5)

            if mode == 4:
                #   这一步骤：重新排序traindata，以时间为key，alldata为value。
                train_data={}
                for i in train_res[list(train_res.keys())[0]].keys():
                    df_ = pd.DataFrame()
                    for k, v in train_res.items():
                        if k.split('_')[-1] == str(time_lidu):
                            df_ = df_.append(v[i],ignore_index=True)
                    train_data[i] = df_

                #   这一步骤，累加训练数据，key= 时间，value=累计向前添加。
                cum_df_ = pd.DataFrame()
                train_data0 = {}
                for t, v in train_data.items():
                    cum_df_ = cum_df_.append(v, ignore_index=True)
                    train_data0[t] = cum_df_
                    # print(t)
                    # print(v.shape)


                train_and_predict(train_data=train_data0,res_path = res_path0,algo =algo,res_flag=f'\\tx\\贪心集中'+res_flag, zong_t=zong_t, hc_zq=6, gd_zq=1)
                time.sleep(5)


#结果分析
def analyze_res2(res_path,info = ''):
    cols =[]
    df_zong = pd.DataFrame()

    df_pre = pd.DataFrame()
    with open(res_path,mode='rb') as f:
        pre_res = pickle.load(f)
        print(pre_res.keys())


    cl_name = ''
    #   筛选当前策略的每周期的参数
    for i,v in pre_res.items():
        df0 = pd.DataFrame()

        print(f'滚动观察：{i}')
        # print(v.keys()) # >>dict_keys(['预测数据', '预测model'])
        # continue
        if len(v.keys()) == 0:
            print(f'{i} =》没有数据，不用操作！')
            df0.at[0,'s_Time'] = i

            df_pre = df_pre.append(df0, ignore_index=True, )
            continue
        else:
            print(v.keys())
            if '预测数据' in v.keys():
                pass
            else:
                v['预测数据'] = pd.DataFrame()

            if v['预测数据'].empty:
                continue


        print(f'{i}:有数据。')

        v = v['预测数据']
        # print(v)
        # exit()
        cl_name = v.iloc[-1]['celuename']
        df0['预测周期真实收益'] = v['预测周期真实收益']
        df0['s_Time'] = i
        #参考col
        cols =['end_预测值', 'celuename']
        for index_ in cols :
            if index_ == 'end_预测值':
                wei = len(str(int(v.iloc[-1][index_]))) -5
                if wei > 0:
                    base = pow(10,wei)
                    df0[index_] =  v[index_]/base
                    continue
            df0[index_] = v[index_]

        df0=df0[['s_Time','预测周期真实收益']+cols]
        df0.sort_values(by='end_预测值',inplace=True)
        # df0 = df0[df0['end_预测值']>0].copy() #   筛选条件
        # df0 = df0[df0['end_预测值']>df0['end_预测值'].mean()].copy() #   筛选条件

        df_pre = df_pre.append(df0,ignore_index=True,)

    df_pre['策略'] = cl_name
    df_pre.fillna(0,inplace=True)
    df_zong =df_zong.append(df_pre,ignore_index=True)

    df_zong.sort_values(by = 's_Time',inplace=True)
    # df_zong = df_zong[df_zong['end_预测值'] > 0].copy()
    print(df_zong[[ '策略','s_Time','预测周期真实收益']+cols].head())

    df_corr = pd.DataFrame(index=cols)
    # for i in cols:
        # # corr_list.append(np.corrcoef(df_zong[i],df_zong[j])[0])
        # df_corr.loc[i,'相关性'] = min(np.corrcoef(df_zong[i],df_zong['预测周期真实收益'])[0])

    print('相关矩阵：')
    print(df_zong.tail(300))
    # exit()
    dong_scatter(data = df_zong[['s_Time','end_预测值','预测周期真实收益','策略']],info = info,path0 = str(res_path).strip('.'+res_path.split('.')[-1]))

    # echart_plot_3d(data = df_zong[['s_Time','end_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['end_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['end_预测值'])

    # print(df_zong)


#结果分析
def analyze_res3(res_path,info = ''):
    '''
    1.读取数据
    2.滚动增加每月记录

    '''
    tj_n = 0
    with open(res_path,mode='rb') as f:
        pre_res = pickle.load(f)
        print(pre_res.keys())

    # with open(train_path,mode='rb') as f:
    #     train_data = pickle.load(f)
    #     print(train_data.keys())

    cl_name = ''
    #   筛选当前策略的每周期的参数
    tongji = pd.DataFrame()
    dfz_0 = pd.DataFrame()
    for i,v in pre_res.items():
        print(f'滚动观察：{i}')
        print('keys：', v.keys())
        #每期的结果汇总
        for k in v.keys():
            if 'data_' not in k :
                continue
            v[k] = flag_encoding(v[k],to_col='model_name', new_col='modelname')
            # print(v[k].keys())
            # exit()
            v[k]['策略'] = v[k]['celue_name']+v[k]['canshu']+v[k]['model_name']
            v[k]['s_Time'] = i
            dfz_0 = dfz_0.append(v[k],ignore_index=True)
            # print(v[k].sample(10))
            # 统计分类数据
            tongji_0 = False
            if tongji_0 ==True:
                for fl, fldf in v[k].groupby('预测值'):
                    if tj_n >0:fldf =fldf.sample(tj_n)

                    if '未来end_3' not in fldf.keys():
                        break

                    index_tj = str(i)+'_'+str(fl)
                    # print(fldf.sample(10))
                    tongji.loc[index_tj,'预测值分类'] = fl
                    tongji.loc[index_tj, '平均收益'] = fldf['未来end_3'].mean()
                    tongji.loc[index_tj, '平均收益std'] = fldf['未来end_3'].std()

                    tongji.loc[index_tj, '平均max收益'] = fldf['未来max_sum_3'].mean()
                    tongji.loc[index_tj, '平均sharp'] = fldf['未来sharp_rate_3'].mean()
                    tongji.loc[index_tj, '平均max_back'] = fldf['未来max_back_3'].mean()
                    tongji.loc[index_tj, '平均win_rate'] = fldf['未来win_rate_3'].mean()
                    tongji.loc[index_tj, '平均trade_nums'] = fldf['未来trade_nums_3'].mean()

    # 统计总体分类数据
    pass
    # for fl, fldf in tongji.groupby('预测值分类'):
    #     index_tj = '总体_' + str(fl)
    #     tongji.loc[index_tj, '预测值分类'] = fl
    #     tongji.loc[index_tj, '平均收益'] = fldf['平均收益'].mean()
    #     tongji.loc[index_tj, '平均收益std'] = fldf['平均收益'].std()
    #
    #     tongji.loc[index_tj, '平均max收益'] = fldf['平均max收益'].mean()
    #     tongji.loc[index_tj, '平均sharp'] = fldf['平均sharp'].mean()
    #     tongji.loc[index_tj, '平均max_back'] = fldf['平均max_back'].mean()
    #     tongji.loc[index_tj, '平均win_rate'] = fldf['平均win_rate'].mean()
    #     tongji.loc[index_tj, '平均trade_nums'] = fldf['平均trade_nums'].mean()
    #
    # to_csvpath = res_path.strip('.pickle')+'.csv'
    # p = 1
    # while True:
    #     if os.path.exists(to_csvpath):
    #         p+=1
    #         to_csvpath = res_path.strip('.pickle')+f'_{p}.csv'
    #     else:break
    # tj_title =pd.DataFrame(data = {'平均max_back':f'每周期分类后的{tj_n}个策略的基本情况。'},index=['描述:'])
    # # tongji.loc['描述：','预测值分类']  = '每周期分类后的n个策略的基本情况。'
    # tongji=  tj_title.append(tongji)
    # tongji.to_csv(to_csvpath,mode='w')
    #
    # print('csv文件：保存：》》',to_csvpath)
    #
    print(dfz_0.shape)

    dfz_0 = dfz_0[(dfz_0['预测值'] >=5)]
    dfz_0 = dfz_0[(dfz_0['预测值'] <=7)]
    #
    # print(dfz_0.tail())
    print(dfz_0.shape)
    # print(pre_res.keys())

    # exit()
    index_0 = ['best_score','end_3', 'max_back_3','max_sum_3',  'sharp_rate_3',
       'sj_acc_score', 'sj_ba_acc_score', 'trade_nums_3', 'win_rate_3',
       '市场_kurt_1', '市场_kurt_2', '市场_kurt_3', '市场_mean_m1', '市场_mean_m2',
       '市场_mean_m3', '市场_mm_m1', '市场_mm_m2', '市场_mm_m3', '市场_skew_1',
       '市场_skew_2', '市场_skew_3', '市场_std_mean1', '市场_std_mean2',
       '市场_std_mean3', '市场_std_mm1', '市场_std_mm2', '市场_std_mm3',
       '市场_vol_kurt1', '市场_vol_kurt2', '市场_vol_kurt3', '市场_vol_mm1',
       '市场_vol_mm2', '市场_vol_mm3', '市场_vol_mm_m1', '市场_vol_mm_m2',
       '市场_vol_mm_m3', '市场_vol_skew1', '市场_vol_skew2', '市场_vol_skew3',
       '市场_vol_std_mm1', '市场_vol_std_mm2', '市场_vol_std_mm3',
          '测试集_预测值', '预测值','clm0', 'clm1', 'clm2', 'clm3',
       'clm4', 'clm5','para1', 'para2', 'para3', 'para4']
    train_data = dfz_0[dfz_0['s_Time'] !=pd.to_datetime('2020-8-1')]
    pre_data = dfz_0[dfz_0['s_Time'] ==pd.to_datetime('2020-8-1')]
    train_data['canshu_index'] = train_data['canshu']
    pre_data['canshu_index'] = pre_data['canshu']
    # 添加策略名因子
    train_data.set_index('canshu_index', drop=True, inplace=True)
    pre_data.set_index('canshu_index', drop=True, inplace=True)
    # print(train_data['未来end_3'].sample(100))
    print(train_data.shape)
    print(train_data.keys())
    # exit()

    # exit()
    # '未来end_3', '未来max_back_3', '未来max_sum_3', '未来sharp_rate_3', '未来trade_nums_3', '未来win_rate_3',
    res_to_df = {}
    res_to_df, suanfa_list = second_fl01(train_data=train_data, res_to_df=res_to_df, yinzi=index_0,resinx='未来end_3', Train=True, n_jobs=3,pac_k=40, q = 6)
    res_to_df, suanfa_list = second_fl01(train_data=pre_data, res_to_df=res_to_df, yinzi=index_0,resinx='未来end_3', Train=False, n_jobs=3,pac_k=40,
                                         q = 6)
    baocun ={}
    baocun[sorted(dfz_0.keys())[-1]] = res_to_df
    path_erji = res_path.strip('.pickle') + f'二级预测结果2.pickle'
    with open(path_erji,'wb') as f0:
        pickle.dump(baocun,f0)
        print('保存成功！')
        for k,v in res_to_df.items():
            if str(k).startswith('data'):
                print(v.keys())
                print(v[v['最终预测值']>=5].sample(50))
        exit()


    dfz_0.sort_values(by = 's_Time',inplace=True)
    # print(dfz_0[['s_Time','策略','预测周期真实收益', '预测周期真实分类',  '预测值']])
    t = dfz_0.iloc[-1]['s_Time']
    y = dfz_0[dfz_0['s_Time'] == t]

    # exit()
    dong_scatter(data = dfz_0,info = info,path0 = str(res_path).strip('.'+res_path.split('.')[-1]))

    # echart_plot_3d(data = df_zong[['s_Time','end_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['end_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['end_预测值'])

    # print(df_zong)

def analyze_res4(res_path,info = ''):
    '''
    1.读取数据
    2.滚动增加每月记录

    '''
    tj_n = 2
    with open(res_path,mode='rb') as f:
        pre_res = pickle.load(f)
        print(pre_res.keys())

    # with open(train_path,mode='rb') as f:
    #     train_data = pickle.load(f)
    #     print(train_data.keys())

    cl_name = ''
    #   筛选当前策略的每周期的参数
    tongji = pd.DataFrame()
    dfz_0 = pd.DataFrame()
    for i,v in pre_res.items():
        print(f'滚动观察：{i}')
        print('keys：', v.keys())
        #每期的结果汇总
        for k in v.keys():
            if 'data_' not in k :
                continue
            # v[k] = flag_encoding(v[k],to_col='model_name', new_col='modelname')
            # print(v[k].keys())
            # exit()
            v[k]['策略'] = v[k]['celue_name']+v[k]['canshu']+v[k]['model_name']
            v[k]['s_Time'] = i
            dfz_0 = dfz_0.append(v[k],ignore_index=True)
            # print(v[k].sample(10))
            # exit()
            # 统计分类数据
            tongji_0 = True
            if tongji_0 ==True:
                for fl, fldf in v[k].groupby('预测集_预测值'):
                    # print(fl)
                    # print(fldf)
                    # exit()
                    # if tj_n >0:fldf =fldf.sample(tj_n)

                    if '未来max_sum_3' not in fldf.keys():
                        break

                    index_tj = str(i)+'_'+str(fl)
                    # print(fldf.sample(10))
                    tongji.loc[index_tj,'预测值分类'] = fl

                    # tongji.loc[index_tj,'预测值自身准确度'] = fl
                    tongji.loc[index_tj, '平均收益'] = fldf['未来end_3'].mean()
                    tongji.loc[index_tj, '平均收益std'] = fldf['未来end_3'].std()
                    tongji.loc[index_tj, '平均max收益'] = fldf['未来max_sum_3'].mean()
                    tongji.loc[index_tj, '平均sharp'] = fldf['未来sharp_rate_3'].mean()
                    tongji.loc[index_tj, '平均max_back'] = fldf['未来max_back_3'].mean()
                    tongji.loc[index_tj, '平均win_rate'] = fldf['未来win_rate_3'].mean()
                    tongji.loc[index_tj, '平均trade_nums'] = fldf['未来trade_nums_3'].mean()

    # 统计总体分类数据
    pass
    for fl, fldf in tongji.groupby('预测值分类'):
        print(fldf)

        index_tj = '总体_' + str(fl)
        tongji.loc[index_tj, '预测值分类'] = fl
        tongji.loc[index_tj, '平均收益'] = fldf['平均收益'].mean()
        tongji.loc[index_tj, '平均收益std'] = fldf['平均收益'].std()

        tongji.loc[index_tj, '平均max收益'] = fldf['平均max收益'].mean()
        tongji.loc[index_tj, '平均sharp'] = fldf['平均sharp'].mean()
        tongji.loc[index_tj, '平均max_back'] = fldf['平均max_back'].mean()
        tongji.loc[index_tj, '平均win_rate'] = fldf['平均win_rate'].mean()
        tongji.loc[index_tj, '平均trade_nums'] = fldf['平均trade_nums'].mean()

    to_csvpath = res_path.strip('.pickle')+'.csv'
    p = 1
    while True:
        if os.path.exists(to_csvpath):
            p+=1
            to_csvpath = res_path.strip('.pickle')+f'_{p}.csv'
        else:break
    tj_title =pd.DataFrame(data = {'平均max_back':f'每周期分类后的{tj_n}个策略的基本情况。'},index=['描述:'])
    # tongji.loc['描述：','预测值分类']  = '每周期分类后的n个策略的基本情况。'
    tongji=  tj_title.append(tongji)
    tongji.to_csv(to_csvpath,mode='w')

    #
    print('csv文件：保存：》》',to_csvpath)
    exit()
    #
    print(dfz_0.shape)

    dfz_0 = dfz_0[(dfz_0['预测值'] >=5)]
    dfz_0 = dfz_0[(dfz_0['预测值'] <=7)]
    #
    # print(dfz_0.tail())
    print(dfz_0.shape)
    # print(pre_res.keys())

    # exit()
    index_0 = ['best_score','end_3', 'max_back_3','max_sum_3',  'sharp_rate_3',
       'sj_acc_score', 'sj_ba_acc_score', 'trade_nums_3', 'win_rate_3',
       '市场_kurt_1', '市场_kurt_2', '市场_kurt_3', '市场_mean_m1', '市场_mean_m2',
       '市场_mean_m3', '市场_mm_m1', '市场_mm_m2', '市场_mm_m3', '市场_skew_1',
       '市场_skew_2', '市场_skew_3', '市场_std_mean1', '市场_std_mean2',
       '市场_std_mean3', '市场_std_mm1', '市场_std_mm2', '市场_std_mm3',
       '市场_vol_kurt1', '市场_vol_kurt2', '市场_vol_kurt3', '市场_vol_mm1',
       '市场_vol_mm2', '市场_vol_mm3', '市场_vol_mm_m1', '市场_vol_mm_m2',
       '市场_vol_mm_m3', '市场_vol_skew1', '市场_vol_skew2', '市场_vol_skew3',
       '市场_vol_std_mm1', '市场_vol_std_mm2', '市场_vol_std_mm3',
          '测试集_预测值', '预测值','clm0', 'clm1', 'clm2', 'clm3',
       'clm4', 'clm5','para1', 'para2', 'para3', 'para4']
    train_data = dfz_0[dfz_0['s_Time'] !=pd.to_datetime('2020-8-1')]
    pre_data = dfz_0[dfz_0['s_Time'] ==pd.to_datetime('2020-8-1')]
    train_data['canshu_index'] = train_data['canshu']
    pre_data['canshu_index'] = pre_data['canshu']
    # 添加策略名因子
    train_data.set_index('canshu_index', drop=True, inplace=True)
    pre_data.set_index('canshu_index', drop=True, inplace=True)
    # print(train_data['未来end_3'].sample(100))
    print(train_data.shape)
    print(train_data.keys())
    # exit()

    # exit()
    # '未来end_3', '未来max_back_3', '未来max_sum_3', '未来sharp_rate_3', '未来trade_nums_3', '未来win_rate_3',
    res_to_df = {}
    res_to_df, suanfa_list = second_fl01(train_data=train_data, res_to_df=res_to_df, yinzi=index_0,resinx='未来end_3', Train=True, n_jobs=3,pac_k=40, q = 6)
    res_to_df, suanfa_list = second_fl01(train_data=pre_data, res_to_df=res_to_df, yinzi=index_0,resinx='未来end_3', Train=False, n_jobs=3,pac_k=40,
                                         q = 6)
    baocun ={}
    baocun[sorted(dfz_0.keys())[-1]] = res_to_df
    path_erji = res_path.strip('.pickle') + f'二级预测结果2.pickle'
    with open(path_erji,'wb') as f0:
        pickle.dump(baocun,f0)
        print('保存成功！')
        for k,v in res_to_df.items():
            if str(k).startswith('data'):
                print(v.keys())
                print(v[v['最终预测值']>=5].sample(50))
        exit()


    dfz_0.sort_values(by = 's_Time',inplace=True)
    # print(dfz_0[['s_Time','策略','预测周期真实收益', '预测周期真实分类',  '预测值']])
    t = dfz_0.iloc[-1]['s_Time']
    y = dfz_0[dfz_0['s_Time'] == t]

    # exit()
    dong_scatter(data = dfz_0,info = info,path0 = str(res_path).strip('.'+res_path.split('.')[-1]))

    # echart_plot_3d(data = df_zong[['s_Time','end_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['end_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['end_预测值'])

    # print(df_zong)



#结果分析
def analyze_ress(res_paths,info = '9-2_tx'):
    cols =[]
    df_zong = pd.DataFrame()
    for r in res_paths:

        df_pre = pd.DataFrame()
        with open(r,mode='rb') as f:
            pre_res = pickle.load(f)
            print(pre_res.keys())
        cl_name = ''
        #   筛选当前策略的每周期的参数
        for i,v in pre_res.items():
            df0 = pd.DataFrame()

            print(f'滚动观察：{i}')
            # print(v.keys()) # >>dict_keys(['预测数据', '预测model'])
            # continue
            if len(v.keys()) == 0:
                print(f'{i} =》没有数据，不用操作！')
                df0.at[0,'s_Time'] = i

                df_pre = df_pre.append(df0, ignore_index=True, )
                continue
            else:
                print(v.keys())
                if '预测数据' in v.keys():
                    pass
                else:
                    v['预测数据'] = pd.DataFrame()

                if v['预测数据'].empty:
                    continue


            print(f'{i}:有数据。')

            v = v['预测数据']
            # print(v)
            # exit()
            cl_name = v.iloc[-1]['celuename']
            df0['预测周期真实收益'] = v['预测周期真实收益']
            df0['s_Time'] = i
            #参考col
            cols =['end_预测值',
                   '本周期总收益','最近周期收益','最大回撤', '平均月最大回撤','平均月收益','平均月夏普率','平均月交易次数']
            for index_ in cols :
                if index_ == 'end_预测值':
                    wei = len(str(int(v.iloc[-1][index_]))) -5
                    if wei > 0:
                        base = pow(10,wei)
                        df0[index_] =  v[index_]/base
                        continue
                df0[index_] = v[index_]

            df0=df0[['s_Time','预测周期真实收益']+cols]
            df0.sort_values(by='end_预测值',inplace=True)
            # df0 = df0[df0['end_预测值']>0].copy() #   筛选条件
            # df0 = df0[df0['end_预测值']>df0['end_预测值'].mean()].copy() #   筛选条件

            df_pre = df_pre.append(df0,ignore_index=True,)

        df_pre['策略'] = cl_name
        df_pre.fillna(0,inplace=True)
        df_zong =df_zong.append(df_pre,ignore_index=True)

    df_zong.sort_values(by = 's_Time',inplace=True)
    df_zong = df_zong[df_zong['end_预测值'] > 0].copy()
    print(df_zong[[ '策略','s_Time','预测周期真实收益']+cols])

    df_corr = pd.DataFrame(index=cols)
    for i in cols:
        # # corr_list.append(np.corrcoef(df_zong[i],df_zong[j])[0])
        df_corr.loc[i,'相关性'] = min(np.corrcoef(df_zong[i],df_zong['预测周期真实收益'])[0])

    print('相关矩阵：')
    print(df_corr)
    # exit()
    dong_scatter(data = df_zong[['s_Time','end_预测值','预测周期真实收益','策略']],info = info,path0=res_paths[0].rstrip(res_paths[0].split('\\')[-1]))
    # echart_plot_3d(data = df_zong[['s_Time','end_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['end_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['end_预测值'])

    print(df_zong)


def fenge_canshu():
    local_data_path = r'F:\new_0811\huice_log\train_res\MEBD策略模型_03_训练数据_time0_1_6_new_09-07.pickle'
    if os.path.exists(local_data_path):
        print('该地址文件存在：')
        with open(local_data_path, 'rb') as f:
            res_pic = pickle.load(f)
            newdf = {}
            print(type(res_pic))
            print(f'{local_data_path}\n本地的内容：', res_pic.keys())
            for k, v in res_pic.items():
                print(k)
                newdf[k] = {}
                for y, da in v.items():
                    print(k)
                    da['canshu'] = da['canshu'].apply(lambda x: str(x).strip(']'))
                    da['canshu'] = da['canshu'].apply(lambda x: str(x).strip('['))
                    da['para1'] = da['canshu'].apply(lambda x: str(x).split(',')[0])
                    da['para2'] = da['canshu'].apply(lambda x: str(x).split(',')[1])
                    da['para3'] = da['canshu'].apply(lambda x: str(x).split(',')[2])
                    da['para4'] = da['canshu'].apply(lambda x: str(x).split(',')[3])
                    newdf[k][y] = da
        with open(local_data_path, 'wb') as f0:
            pickle.dump(newdf, f0)

def clm_tezheng(train_data):

    clm = set(sorted(train_data['celue_name'].to_list()))
    clm_dict= {}
    for i, c in enumerate(clm):
        clm_dict[c]={'clm%s'%i:1}
    for ix, c in enumerate(clm_dict.keys()):
        for i, clm in enumerate(clm):
            if 'clm%s' % i in clm_dict[c].keys():continue
            else: clm_dict[c]['clm%s' % i] = 0

    for i in clm:
        con = train_data['celue_name'] == i
        # 同名策略名统一命名。
        for m,v in clm_dict[i].items():
            train_data.loc[con,m]=v



def show_local_data(local_data_path,show =False):

    if os.path.exists(local_data_path):
        print('该地址文件存在：')
        with open(local_data_path,'rb') as f:
            res_pic = pickle.load(f)
            print(type(res_pic))
            print(f'{local_data_path}\n第一层key的本地的内容：',res_pic.keys())
            if show == True:
                for k,v in res_pic.items():
                    print(k)
                    # if k < datetime.datetime(2020, 8, 1, 0, 0):continue
                    print('\n第二层key',v.keys())
                    # print(set(res_pic['s_Time'].tolist()))
                    # exit()
                    # print('\n第三层key',v['data'].keys())

                    print('\n第三层key',v['data'].shape)
                    # v['data_randomforest_classify_grid']['new']= v['data_randomforest_classify_grid'][['canshu','celue_name']].apply(lambda x: str(x['canshu'])+str(x['celue_name']))
                    # v['data']['new'] =v['data']['canshu']+v['data']['celue_name']
                    print('\n第三层key',v['data'].tail())
                    # exit()



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    from 统计分析 import *
    from 机器学习函数 import *
    from 算法组合 import *
    # 单策略运行
    zong_t = [dt.datetime(2019, 6, 1), dt.datetime(2020, 6, 1)]

    if 1 == True:

        path_ = os.getcwd() + r'\huice_log' + '\MEBD03\dema_tp_03_2019-06-01_3T_8_18.csv'
        data_path = r'F:\new_0811\DB_in_out_strategies\MEBD策略模型_01.pickle'
        train_path = r'F:\new_0811\huice_log\train_res\MEBD03_time1_traindatas_特征.pkl'
        res_path0 = r'F:\new_0811\huice_log\MEBD03_pre_res'
        #开始预测
        # pre_mode(train_path, res_path0, zong_t,res_flag = f'_pre_res_cal_zuhe2.pickle', algo=['cal_zuhe2'], mode=4, time_lidu=1)
        show_path =  r'F:\new_0811\huice_log\MEBD03_pre_res\tx\贪心集中zuhe_911_022_MEBD_03_训练结果_10_1f09-22.pickle'
        print('chakan')
        show_local_data(show_path,show=True)
        exit()
        # fenge_canshu()

        #   集中统计结果，的数据
        if 1 == True:
            # res_paths = [r'F:\new_0811\huice_log\res_data\s_ftx\非贪心ma_tp_03_1_pre_res_cal_zuhe1.pickle',
            #              r'F:\new_0811\huice_log\res_data\s_ftx\非贪心ema_tp_03_1_pre_res_cal_zuhe1.pickle',
            #              r'F:\new_0811\huice_log\res_data\s_ftx\非贪心dema_tp_03_1_pre_res_cal_zuhe1.pickle',
            #              r'F:\new_0811\huice_log\res_data\s_ftx\非贪心wma_tp_03_1_pre_res_cal_zuhe1.pickle',
            #              r'F:\new_0811\huice_log\res_data\s_ftx\非贪心kama_tp_03_1_pre_res_cal_zuhe1.pickle',
            #              r'F:\new_0811\huice_log\res_data\s_ftx\非贪心T3_tp_03_1_pre_res_cal_zuhe1.pickle',
            #
            #              ]
            # res_paths=[r'F:\new_0811\huice_log\res_data\s_tx\ma_tp_03_1_pre_res_cal_zuhe1.pickle',
            #            r'F:\new_0811\huice_log\res_data\s_tx\ema_tp_03_1_pre_res_cal_zuhe1.pickle',
            #            r'F:\new_0811\huice_log\res_data\s_tx\dema_tp_03_1_pre_res_cal_zuhe1.pickle',
            #            r'F:\new_0811\huice_log\res_data\s_tx\wma_tp_03_1_pre_res_cal_zuhe1.pickle',
            #            r'F:\new_0811\huice_log\res_data\s_tx\kama_tp_03_1_pre_res_cal_zuhe1.pickle',
            #            r'F:\new_0811\huice_log\res_data\s_tx\T3_tp_03_1_pre_res_cal_zuhe1.pickle',
            #            ]
            # res_paths =[r'F:\new_0811\huice_log\res_data\ftx\非贪心集中_pre_res_cal_zuhe1.pickle']

            res_path = r'F:\new_0811\huice_log\MEBD03_pre_res\tx\贪心集中zuhe_911_02mebd03_训练结果20_09-15.pickle'
            analyze_res4(res_path=res_path, info='')


