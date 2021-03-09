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

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from typing import List
from vnpy.trader.object import TradeData


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
            df_zong0.loc[i, 'para1'] = float(d['para1'].mean())
            df_zong0.loc[i, 'para2'] = float(d['para2'].mean())
            df_zong0.loc[i, 'para3'] = float(d['para3'].mean())
            df_zong0.loc[i, 'para4'] = float(d['para4'].mean())

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
    :return: 以时间为序列的统计特征字典{时间：数据，时间：数据。。。}
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

            print(f'{celuename}训练数据月份：',now_t0.strftime(format="%Y-%m-%d"))
            # 每次计算一次
            dt_yinzi = cal_yinzi0(celuename,train_da0, train_s0, train_e0, res_t=now_t0, train_res=train_res0)

            yinzi_dict[now_t0.strftime(format="%Y-%m-%d")] = dt_yinzi

            if dt_yinzi.empty == False:
                print(f'{now_t0}完成。')
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

# 学习，预测
def train_and_predict(train_data,res_path,algo,res_flag= '.pickle' ,zong_t = [], hc_zq=6, gd_zq=1,n_jobs=2):

    # 导入训练集
    df_yinzi = train_data.copy()
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


        # 训练数据结果月>回测结束时间，意味着，没有预测数据。跳出。
        if now_t0 > zong_t[-1]:
            print(f'训练数据结束，{train_t0}')
            break
        else:
            print( now_,'=====>', pre_)

        df_zong[now_t0] = {}
        # 训练数据
        train_da0 =  pd.DataFrame(df_yinzi[now_].copy())

        # 预测数据
        pre_da0 =  pd.DataFrame(df_yinzi[pre_].copy())
        pre_da0 = pre_da0[pre_da0['res_t'] == pre_].copy()
        # print(pre_da0.head(10))
        #
        # print(pre_da0.tail(10))
        # time.sleep(2)
        # continue

        train_da0['canshu_index'] =train_da0['canshu']
        pre_da0['canshu_index'] =pre_da0['canshu']

        train_da0.set_index('canshu_index', drop=True, inplace=True)
        pre_da0.set_index('canshu_index', drop=True, inplace=True)
        yinzi0 = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        # yinzi0 = corr_jw(train_data=train_da0, yinzi=yinzi0, res_index='预测周期真实收益')
        # exit()

        # print(train_da0.tail())
        # print(pre_da0.tail())
        # exit()

        # 机器学习预测
        try:

            print(now_t0, '开始训练。。。')


            res0 = ['预测周期真实收益', '预期偏差', 'end_预测值']
            # 训练
            model, df_pre0 = eval(algo)(train_data=train_da0, model_list={}, yinzi=yinzi0,Train=True,n_jobs=n_jobs)
            if isinstance(df_pre0, str):
                print(f'{df_pre0}，下一循环。')
                df_zong[now_t0]['预测数据'] = pd.DataFrame()
                df_zong[now_t0]['预测model'] = model
                i= i+gd_zq
                continue
            else:
                df_zong[now_t0]['预测model'] = model


            res_index = 'end_预测值'
            # 预测， 最后一个月，只给出预测数据，
            if pre_t0 > zong_t[-1]: #预测数据的真实值大于目前数据，代表没有未来数据，仅需要预测。
                model_list, pre_data = eval(algo)(train_data=pre_da0, model_list=model, yinzi=yinzi0, Train=False,last=True)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，最后一个月，无预测数据。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                else:
                    print(f'最终预测')
                    pre_data.dropna(axis=0, subset=[res_index], inplace=True)
                    pre_data.fillna(0, inplace=True)
                    pre_data.sort_values(by=[res_index], ascending=True, inplace=True)
                    df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
                break
            else:
                # 预测
                model_list, pre_data = eval(algo)(train_data=pre_da0, model_list=model,yinzi=yinzi0, Train=False)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，下一循环。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                    i = i + gd_zq
                    continue

            # print(pre_data)

            pre_data.dropna(axis=0, subset=[res_index], inplace=True)
            pre_data.fillna(0, inplace=True)
            pre_data.sort_values(by=[res_index], ascending=True, inplace=True)

            res = ['celuename', 'end_预测值','预测周期真实收益', 'canshu', 'train_s',
                   'train_e', 'res_t',  '本周期总收益', '最近周期收益', '最大回撤', '最大值',
                   '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益', '平均月最大回撤', '平均月夏普率', '平均月交易次数',
                   '月均交易天数', '月均盈利天数', '月均开单收益std', '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利',
                   '月均胜单平均盈利偏度', '月均胜单平均盈利std', '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std',
                   '月均开单平均收益', '月均开单平均收益偏度', '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']

            if pre_data.empty:pass
            else:print(pre_data.tail())

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

# 学习，预测2
def train_and_predict2(train_data,res_path,algo,res_flag= '.pickle' ,zong_t = [], hc_zq=6, gd_zq=1,n_jobs=2):
    '''02
    动态处理每次训练的因子，进行动态选择。
基本流程：
1.在每次日期循环时,生成两个数据集，训练数据和测试数据（预测数据）。
2.对两个数据进行必要的加工。
3.对每次的训练数据进行，因子重构：这个重构意味：每次因子都不同。选择标准一致。

    '''
    # 导入训练集
    df_yinzi = train_data.copy()

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
        # 训练数据结果月>回测结束时间，意味着，没有预测数据。跳出。
        if now_t0 > zong_t[-1]:
            print(f'训练数据结束，{train_t0}')
            break
        else:
            print( now_,'=====>', pre_)

        df_zong[now_t0] = {}
        # 训练数据
        train_da0 =  pd.DataFrame(df_yinzi[now_].copy())

        # 预测数据
        pre_da0 =  pd.DataFrame(df_yinzi[pre_].copy())
        pre_da0 = pre_da0[pre_da0['res_t'] == pre_].copy()


        train_da0['canshu_index'] =train_da0['canshu']
        pre_da0['canshu_index'] =pre_da0['canshu']

        train_da0.set_index('canshu_index', drop=True, inplace=True)
        pre_da0.set_index('canshu_index', drop=True, inplace=True)
        yinzi = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        # 因子重构
        train_da0,new_yinzi = build_yinzi(train_data=train_da0, yinzi=yinzi,res_index = '预测周期真实收益')
        train_da0,yinzi0 = corr_jw(train_data=train_da0,n=10, yinzi=new_yinzi, res_index='预测周期真实收益')
        yinzi = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        pre_da0, new_yinzi = build_yinzi(train_data=pre_da0, yinzi=yinzi, res_index='预测周期真实收益')
        # print(pre_da0.tail())

        # pre_da0, yinzi0 = corr_jw(train_data=pre_da0, n=10,yinzi=new_yinzi, res_index='预测周期真实收益')

        # exit()
        print(f'筛选出：{yinzi0}')
        # exit()

        # print(train_da0.tail())
        # print(pre_da0.tail())
        # exit()

        # 机器学习预测
        try:

            print(now_t0, '开始训练。。。')


            res0 = ['预测周期真实收益', '预期偏差', 'end_预测值']
            # 训练
            model, df_pre0 = eval(algo)(train_data=train_da0, model_list={}, yinzi=yinzi0,Train=True,n_jobs=n_jobs)
            if isinstance(df_pre0, str):
                print(f'{df_pre0}，下一循环。')
                df_zong[now_t0]['预测数据'] = pd.DataFrame()
                df_zong[now_t0]['预测model'] = model
                i= i+gd_zq
                continue
            else:
                df_zong[now_t0]['预测model'] = model


            res_index = 'end_预测值'
            # 预测， 最后一个月，只给出预测数据，
            if pre_t0 > zong_t[-1]: #预测数据的真实值大于目前数据，代表没有未来数据，仅需要预测。
                print('最后一个月！')

                model_list, pre_data = eval(algo)(train_data=pre_da0, model_list=model, yinzi=yinzi0, Train=False,last=True)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，最后一个月，无预测数据。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                else:
                    print(f'最终预测')
                    pre_data.dropna(axis=0, subset=[res_index], inplace=True)
                    pre_data.fillna(0, inplace=True)
                    pre_data.sort_values(by=[res_index], ascending=True, inplace=True)
                    df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
                break
            else:
                # 预测
                model_list, pre_data = eval(algo)(train_data=pre_da0, model_list=model,yinzi=yinzi0, Train=False)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，下一循环。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                    i = i + gd_zq
                    continue

            # print(pre_data)

            pre_data.dropna(axis=0, subset=[res_index], inplace=True)
            pre_data.fillna(0, inplace=True)
            pre_data.sort_values(by=[res_index], ascending=True, inplace=True)

            res = ['celuename', 'end_预测值','预测周期真实收益', 'canshu', 'train_s',
                   'train_e', 'res_t',  '本周期总收益', '最近周期收益', '最大回撤', '最大值',
                   '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益', '平均月最大回撤', '平均月夏普率', '平均月交易次数',
                   '月均交易天数', '月均盈利天数', '月均开单收益std', '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利',
                   '月均胜单平均盈利偏度', '月均胜单平均盈利std', '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std',
                   '月均开单平均收益', '月均开单平均收益偏度', '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']

            if pre_data.empty:pass
            else:print(pre_data.tail())

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

# 学习，预测3
def train_and_predict3(train_data,res_path,algo,res_flag= '.pickle' ,zong_t = [], hc_zq=6, gd_zq=1,n_jobs=2):
    '''02
    动态处理每次训练的因子，进行动态选择。
基本流程：
1.在每次日期循环时,生成两个数据集，训练数据和测试数据（预测数据）。
2.对两个数据进行必要的加工。
3.对每次的训练数据进行，因子重构：这个重构意味：每次因子都不同。选择标准一致。

    '''
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
        train_t0_ = train_t0.strftime(format="%Y-%m-%d")

        # 预测结束时间，预测结果月
        pre_t0 = now_t0 + relativedelta(months=int(gd_zq))
        pre_ = pre_t0.strftime(format="%Y-%m-%d")
        print('----------')
        print(train_t0_,'====>',now_, '=====>', pre_)

        # 训练数据结果月>回测结束时间，意味着，没有预测数据。跳出。
        if now_t0 > zong_t[-1]:
            print(f'训练数据结束，{train_t0}')
            break
        # exit()
            # print( now_,'=====>', pre_)

        df_zong[now_t0] = {}
        # 训练数据
        train_da0 =  pd.DataFrame(df_yinzi[now_].copy())

        # 预测数据
        pre_da0 =  pd.DataFrame(df_yinzi[pre_].copy())
        pre_da0 = pre_da0[pre_da0['res_time'] == pre_].copy()


        train_da0['canshu_index'] =train_da0['canshu']
        pre_da0['canshu_index'] =pre_da0['canshu']

        # 添加因子
        train_da0 = clm_tezheng(train_da0)
        pre_da0 = clm_tezheng(pre_da0)

        # print(pre_da0.head(5))

        train_da0.set_index('canshu_index', drop=True, inplace=True)
        pre_da0.set_index('canshu_index', drop=True, inplace=True)
        flag_ = ['celuename', 'canshu', 'celue_name', 's_time', 'e_time', 'res_time','预测周期真实收益','canshu_index']
        yinzi = [i for i in list(train_da0.keys()) if i not in flag_]
        # print(yinzi)
        # exit()

        # # 因子重构
        # train_da0,new_yinzi = build_yinzi(train_data=train_da0, yinzi=yinzi,res_index = '预测周期真实收益')
        # train_da0,yinzi0 = corr_jw(train_data=train_da0,n=20, yinzi=yinzi, res_index='预测周期真实收益')

        # pre_da0, new_yinzi = build_yinzi(train_data=pre_da0, yinzi=yinzi, res_index='预测周期真实收益')
        # pre_da0, yinzi0 = corr_jw(train_data=pre_da0, n=20,yinzi=new_yinzi, res_index='预测周期真实收益')
        # exit()

        yinzi0 = yinzi
        print(f'本次执行的因子数：{len(yinzi0)}')
        print(f'本次执行的因子数：{(yinzi0)}')

        # exit()

        # print(train_da0.tail())
        # print(pre_da0.tail())
        # exit()

        # 机器学习预测
        try:

            print(now_t0, '开始训练。。。')


            # 训练
            model, df_pre0 = eval(algo)(train_data=train_da0, model_list={}, yinzi=yinzi0,Train=True,n_jobs=n_jobs)
            if isinstance(df_pre0, str):
                print(f'{df_pre0}，下一循环。')
                df_zong[now_t0]['预测数据'] = pd.DataFrame()
                df_zong[now_t0]['预测model'] = model
                i= i+gd_zq
                continue
            else:
                df_zong[now_t0]['预测model'] = model


            res_index = 'end_预测值'
            # 预测， 最后一个月，只给出预测数据，
            if pre_t0 > zong_t[-1]: #预测数据的真实值大于目前数据，代表没有未来数据，仅需要预测。
                print('最后一个月！')
                model_list, pre_data = eval(algo)(train_data=pre_da0, model_list=model, yinzi=yinzi0, Train=False,last=True)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，最后一个月，无预测数据。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                else:
                    print(f'最终预测')
                    pre_data.dropna(axis=0, subset=[res_index], inplace=True)
                    pre_data.fillna(0, inplace=True)
                    pre_data.sort_values(by=[res_index], ascending=True, inplace=True)
                    df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
                break
            else:
                # 预测
                model_list, pre_data = eval(algo)(train_data=pre_da0, model_list=model,yinzi=yinzi0, Train=False)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，下一循环。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                    i = i + gd_zq
                    continue

            # print(pre_data)

            # pre_data.dropna(axis=0, subset=[res_index], inplace=True)
            pre_data.fillna(0, inplace=True)
            pre_data.sort_values(by=[res_index], ascending=True, inplace=True)

            res = ['celuename', 'end_预测值','预测周期真实收益', 'canshu', 'train_s',
                   'train_e', 'res_t',  '本周期总收益', '最近周期收益', '最大回撤', '最大值',
                   '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益', '平均月最大回撤', '平均月夏普率', '平均月交易次数',
                   '月均交易天数', '月均盈利天数', '月均开单收益std', '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利',
                   '月均胜单平均盈利偏度', '月均胜单平均盈利std', '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std',
                   '月均开单平均收益', '月均开单平均收益偏度', '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']

            if pre_data.empty:pass
            else:print(pre_data.tail())

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

# 匹配911，二级全预测
def train_and_predict4(train_data,res_path,algo,res_flag= '.pickle' ,zong_t = [], hc_zq=6, gd_zq=1,n_jobs=2,q_n=10,test=False):
    '''02
    动态处理每次训练的因子，进行动态选择。
    基本流程：
    1.在每次日期循环时,生成两个数据集，训练数据和测试数据（预测数据）。
    2.对两个数据进行必要的加工。
    3.对每次的训练数据进行，因子重构：这个重构意味：每次因子都不同。选择标准一致。

    '''
    # 导入训练集
    df_yinzi = train_data.copy()
    print(df_yinzi.keys())
    print(zong_t)
    # exit()

    i = 0
    df_zong = {}  # 收集每一阶段的优势参数
    while True:
        # 当前的回测的时间=训练结果月
        now_t0 = zong_t[0] + relativedelta(months=int(i + hc_zq))
        now_ = now_t0.strftime(format="%Y-%m-%d")
        # 训练结束时间
        train_t0 = now_t0 - relativedelta(months=int(gd_zq))
        train_t0_ = train_t0.strftime(format="%Y-%m-%d")

        # 预测结束时间，预测结果月
        pre_t0 = now_t0 + relativedelta(months=int(gd_zq))
        pre_ = pre_t0.strftime(format="%Y-%m-%d")
        pre_pre = ( pre_t0 + relativedelta(months=int(1))).strftime(format="%Y-%m-%d")

        print('----------')
        print(train_t0_,'====>',now_, '=====>', pre_)


        # 训练数据结果月>回测结束时间，意味着，没有预测数据。跳出。
        if now_t0 > zong_t[-1]:
            print(f'训练数据结束，{train_t0}')
            break

        df_zong[now_t0] = {}
        # 训练数据:包含多个月
        train_da0 =  pd.DataFrame(df_yinzi[now_].copy())

        # 预测数据：只有一个月！
        pre_da0 =  pd.DataFrame(df_yinzi[pre_].copy())

        pre_da0 = pre_da0[pre_da0['res_time'] == pre_].copy()
        train_da0['canshu_index'] =train_da0['canshu']
        pre_da0['canshu_index'] =pre_da0['canshu']
        # 添加策略名因子
        train_da0 = clm_tezheng(train_da0)
        pre_da0 = clm_tezheng(pre_da0)
        train_da0.set_index('canshu_index', drop=True, inplace=True)
        pre_da0.set_index('canshu_index', drop=True, inplace=True)
        flag_ = ['celuename', 'canshu', 'celue_name', 's_time', 'e_time', 'res_time','预测周期真实收益','canshu_index']
        yinzi = [i for i in list(train_da0.keys()) if i not in flag_]
        # print(yinzi)
        # print(train_da0[train_da0['res_time']==now_].tail())
        # print(pre_da0.tail())
        # exit()

        # # 因子重构
        # train_da0,new_yinzi = build_yinzi(train_data=train_da0, yinzi=yinzi,res_index = '预测周期真实收益')
        # train_da0,yinzi0 = corr_jw(train_data=train_da0,n=20, yinzi=yinzi, res_index='预测周期真实收益')

        # pre_da0, new_yinzi = build_yinzi(train_data=pre_da0, yinzi=yinzi, res_index='预测周期真实收益')
        # pre_da0, yinzi0 = corr_jw(train_data=pre_da0, n=20,yinzi=new_yinzi, res_index='预测周期真实收益')
        # exit()
        yinzi0 = yinzi
        print(f'本次执行的因子数：{len(yinzi0)}')
        print(f'本次执行的因子数：{(yinzi0)}')
        # exit()


        # 机器学习预测
        try:
            print(now_t0, '开始训练。。。')
            # 训练
            if test:
                train_da0 = train_da0.sample(600)
                pre_da0 = pre_da0.sample(600)
                pac_k = 10
            else:
                pac_k = 40
            res_to_df, suanfa_list = eval(algo)(train_data=train_da0, res_to_df={}, yinzi=yinzi0,Train=True,n_jobs=n_jobs,pac_k = pac_k,q=q_n)

            # 预测， 最后一个月，只给出预测数据，
            print(pre_t0, '开始预测。。。')
            if pre_t0 > zong_t[-1]: #预测数据的真实值大于目前数据，代表没有未来数据，仅需要预测。
                print('最后一个月！',pre_t0)
                res_to_df, suanfa_list = eval(algo)(train_data=pre_da0, res_to_df=res_to_df, yinzi=yinzi0,
                                                    Train=False,last=True,n_jobs=n_jobs,pac_k =pac_k,q=q_n)
            else:
                # 预测
                print('预测月分：',pre_t0)
                res_to_df, suanfa_list = eval(algo)(train_data=pre_da0, res_to_df=res_to_df, yinzi=yinzi0,
                                                    Train=False,last=False,n_jobs=n_jobs,pac_k = pac_k,q=q_n)
            # print(res_to_df.keys())
            for k,v in res_to_df.items():
                if str(k).startswith('data')==False:continue

                add_index = ['canshu','celue_name']+ [x for x in df_yinzi[pre_].keys() if str(x).startswith('市场_')]
                add_index+= [x for x in pre_da0.keys() if str(x).startswith('para')]
                add_index+= [x for x in pre_da0.keys() if str(x).startswith('clm')]

                add_index+= ['end_3','max_sum_3','sharp_rate_3','max_back_3','win_rate_3','trade_nums_3']

                print(add_index)

                v = pd.merge(v,pre_da0[add_index],on=['canshu','celue_name'])
                print(v.tail(3))
                if pre_t0 > zong_t[-1]:
                    print('跳出！')
                    v['未来end_3'] = 0
                    res_to_df[k] = v
                    break

                pre_df = df_yinzi[pre_pre][['canshu','celue_name','end_3','max_sum_3','sharp_rate_3','max_back_3','win_rate_3','trade_nums_3']]
                li_ = ['end_3', 'max_sum_3', 'sharp_rate_3', 'max_back_3', 'win_rate_3', 'trade_nums_3']
                newli_ = ['未来'+str(x) for x in li_]

                pre_df.rename(columns=dict(zip(li_,newli_)),inplace =True)

                res_to_df[k] = pd.merge(v,pre_df,on=['canshu','celue_name'])

                print(res_to_df[k].tail(5))
                # exit()
            # res_to_df = pd.merge(res_to_df,pre_da0[['end_3','max_sum_3','sharp_rate_3','max_back_3','win_rate_3','trade_nums_3','win_mean_num_3','loss_mean_num_3']],left_on=['canshu','celue_name'])
            df_zong[now_t0]= res_to_df
            i =i+gd_zq
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            i =i+1
            continue
    print(df_zong.keys())
    time0 = time.strftime("%m-%d", time.localtime())

    res_path += res_flag+time0+'最新月份数据.pickle'
    with open(res_path,mode='wb') as f:
        pickle.dump(df_zong,f)
        print(f'逐月训练结果，已保存！to:\n{res_path}')

    return df_zong

# 预测模式选择
def pre_mode(train_path, res_path0, zong_t, res_flag=f'_pre_res12.pickle',zuhe_cal='train_and_predict2', algo='cal_zuhe3', mode=2, hg_zq = 2, time_lidu=1,n_jobs=2,q_n = 10,test =False):

    with open(train_path, 'rb') as fe:
        train_res = pickle.load(fe)
        print(f'训练数据包含策略：',list(train_res.keys()))

        # s_ftx
        if mode == 1:
            for k, v in train_res.items():
                print(k)
                eval(zuhe_cal)(train_data=v, res_path=res_path0, algo=algo, res_flag=f'\s_ftx\\{k}_贪心集中{algo}{res_flag}' + res_flag,
                                  zong_t=zong_t, hc_zq=6, gd_zq=1)
                time.sleep(2)
        # ftxjz
        if mode == 2:
            train_data = {}
            print(list(train_res.keys())[0])
            for i in train_res[list(train_res.keys())[0]].keys():
                df_ = pd.DataFrame()
                for k, v in train_res.items():
                    if k.split('_')[-1] == str(time_lidu):
                        df_ = df_.append(v[i], ignore_index=True)
                train_data[i] = df_
                print(set(train_data[i]['celuename'].to_list()))
                # exit()

            eval(zuhe_cal)(train_data=train_data, res_path=res_path0, algo=algo, res_flag=f'\\ftx\\非贪心集中{algo}{res_flag}' + res_flag,
                              zong_t=zong_t, hc_zq=6, gd_zq=1)
            time.sleep(5)
        # s_tx
        if mode == 3:
            for k, v in train_res.items():
                cum_df_ = pd.DataFrame()
                train_data = {}
                for t in v.keys():
                    print(k)
                    cum_df_ = cum_df_.append(v[t], ignore_index=True)
                    train_data[t] = cum_df_
                eval(zuhe_cal)(train_data=v, res_path=res_path0, algo=algo, res_flag=f'\s_tx\\{k}{algo}{res_flag}' + res_flag,
                                  zong_t=zong_t, hc_zq=6, gd_zq=1)
                time.sleep(5)
        # txjz
        if mode == 4:
            train_data = {} #按月份合并所有策略
            print(f'每个策略，包含月份：', )
            yuefen_list = sorted(train_res[list(train_res.keys())[0]].keys())
            if not isinstance(yuefen_list[0],str):
                yuefen_list = [datetime.datetime.strftime(i,"%Y-%m-%d" ) for i in yuefen_list]
            for i in yuefen_list: #遍历月份
                if datetime.datetime.strptime(i,"%Y-%m-%d" )<(zong_t[0]+relativedelta(months=int(hg_zq))):continue
                df_ = pd.DataFrame()
                # 每一个策略都添加历史的:hg_zq个
                for k, v in train_res.items():
                    # 遍历月份添加
                    if k.split('_')[-1] == str(time_lidu):
                        for y in range(hg_zq):
                            add_y = datetime.datetime.strptime(i, "%Y-%m-%d") - relativedelta(months=int(y))
                            add_y = add_y.strftime(format="%Y-%m-%d")
                            df_ = df_.append(v[add_y], ignore_index=True)
                train_data[i] = df_

            # 贪心向前！，贪心算法要hgzq=0才可以
            if hg_zq == 0:
                print('贪心向前，hg_zq = 0')
                train_data0 = {}
                cum_df_ = pd.DataFrame()
                for t,v in train_data.items():
                    # print(t)
                    cum_df_ = cum_df_.append(v, ignore_index=True)
                    train_data0[t] = cum_df_
            else:
                train_data0 = train_data
                print(f'贪心周期固定，hg_zq = {hg_zq}')

            print(train_data0.keys())
            print(train_data0['2020-09-01'].keys())
            print(train_data0['2020-09-01'].tail())
            # print(train_data0['2020-10-01'].tail())
            # exit()
            resdf = eval(zuhe_cal)(train_data=train_data0, res_path=res_path0, algo=algo, res_flag=f'\\贪心集中{algo}{res_flag}',
                              zong_t=zong_t, hc_zq=hg_zq, gd_zq=1,n_jobs=n_jobs,q_n=q_n,test=test)
            time.sleep(5)
            return resdf

def pre_mode_mulfile(train_paths:str, res_path0:str, zong_t:list,index_list=[], res_flag=f'.pickle', algo='cal_zuhe3', mode=2, time_lidu=1,n_jobs=2):
    #  加载多个文件的训练数据。
    '''
    for i in train_paths:
        class_name = (i.split('_')[-4].split('\\')[-1])
        with open(i, 'rb') as fe:
            train_data = pickle.load(fe)
            for k,v in train_data.items():
                print('key:',k)
                print(v.keys())
                for t in v.keys():
                    # print(v[t]['celuename'].tail())
                    if class_name in v[t].iloc[-1]['celuename']:
                        print('class_in')
                        continue
                    v[t]['celuename'] = class_name + '_' + v[t]['celuename'] +'_' + v[t]['canshu']

        with open(i,'wb') as f:
            pickle.dump(train_data,f)
            

        # exit()
    '''
    # for i in train_paths:
    #     with open(i, 'rb') as fe:
    #         train_data = pickle.load(fe)
    #         for k, v in train_data.items():
    #             print('key:', k)
    #             print(v['2020-07-01'].sample(50))
    #             exit()
    # exit()
    celue_class = []
    train_datas = {}
    #  遍历所有月份下的所有训练文件
    for y in ['2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01']:
        yue_data = pd.DataFrame()
        for path_0 in train_paths:
            with open(path_0, 'rb') as fe:
                train_res = pickle.load(fe)
                for k, v in train_res.items():
                    # print('key:', k)
                    celue_class.append('_'+ k[:-4])
                    if k.split('_')[-1]!=f'{time_lidu}':continue
                    yue_data =yue_data.append(v[y])
        train_datas[y] = yue_data
        # print(train_datas[y].sample(30))
        # print(train_datas[y].shape)
        # print(set(celue_class))
        # exit()
    # print(train_datas['2020-07-01'].tail(1000))
    #     # exit()
    index_list = list(set(celue_class)) if len(index_list)==0 else index_list
    # 同类贪心>>生成多个文档。
    if mode == 1 and len(index_list)>0:
        for clname in index_list:
            cum_df0 =pd.DataFrame()
            train_data0= {}
            for k,v in train_datas.items():
                v['celuename_in'] = v['celuename'].apply(lambda x:1 if clname in x else 0)
                cum_df0 =cum_df0.append(v[v['celuename_in'] ==1].copy())
                train_data0[k] = cum_df0
            print(train_data0.keys())
            print(f'现在运行策略类：{clname}')

            train_and_predict2(train_data=train_data0, res_path=res_path0, algo=algo, res_flag=f'\mul_files_tx\\{algo}{clname}{res_flag}',
                              zong_t=zong_t, hc_zq=6, gd_zq=1, n_jobs=n_jobs)
            time.sleep(2)

    # 多类别贪心生成一个文档。
    if mode == 2 and len(index_list)>0:
        cum_df0 =pd.DataFrame()
        train_data0= {}
        for k,v in train_datas.items():
            tianjia_ = pd.DataFrame()
            for clname in index_list:
                v['celuename_in'] = v['celuename'].apply(lambda x:1 if (clname in x) else 0)
                tianjia_ = tianjia_.append(v[v['celuename_in'] ==1].copy())
            cum_df0 = cum_df0.append(tianjia_)
            train_data0[k] = cum_df0
            # print(cum_df0.shape)

        print(train_data0.keys())
        # exit()
        train_and_predict(train_data=train_data0, res_path=res_path0, algo=algo,
                          res_flag=f'\mul_to_1_tx\\贪心集中{algo}{res_flag}',
                              zong_t=zong_t, hc_zq=6, gd_zq=1,n_jobs = n_jobs)
        time.sleep(2)


#结果分析
def analyze_res(res_paths,info = ''):
    cols =[]
    df_zong = pd.DataFrame()
    for r in res_paths:

        df_pre = pd.DataFrame()
        with open(r,mode='rb') as f:
            pre_res = pickle.load(f)

        cl_name = ''
        for i,v in pre_res.items():
            df0 = pd.DataFrame()
            print(f'滚动观察：{i}')
            if v['预测数据'].empty:
                print(f'{i} =》没有数据，不用操作！')
                df0.at[0,'s_Time'] = i

                df_pre = df_pre.append(df0, ignore_index=True, )
                continue

            print(f'{i}:有数据。')

            v= v['预测数据']
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
            # print(df0)
            # time.sleep(0.1)

            df0.sort_values(by='end_预测值',inplace=True)
            df0 = df0.iloc[-5:].copy()
            df_pre = df_pre.append(df0,ignore_index=True,)

        df_pre['策略'] = cl_name
        df_pre.fillna(0,inplace=True)
        df_zong =df_zong.append(df_pre,ignore_index=True)

    df_zong.sort_values(by = 's_Time',inplace=True)
    df_zong = df_zong[df_zong['end_预测值'] > 0].copy()
    print(df_zong[[ '策略','s_Time','预测周期真实收益']+cols])
    corr_list= []
    df_corr = pd.DataFrame(index=cols)
    for i in cols:
        for j in cols:

            # corr_list.append(np.corrcoef(df_zong[i],df_zong[j])[0])
            df_corr.loc[i,j] = min(np.corrcoef(df_zong[i],df_zong[j])[0])

    print('相关矩阵：')

    print(df_corr)
    exit()

    dong_scatter(data = df_zong[['s_Time','end_预测值','预测周期真实收益']],info = info)
    # echart_plot_3d(data = df_zong[['s_Time','end_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['end_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['end_预测值'])

    print(df_zong)

def clm_tezheng(train_data,to_col='celue_name',new_col = 'clm'):

    clm = set(sorted(train_data[to_col].to_list()))
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


if __name__ == '__main__':
    from 统计分析 import *
    from 机器学习函数 import *
    from 算法组合 import *


    sys.path.append(r"F:\task\恒生股指期货\KRData_master\KRData")

    '''
    更新或者重测：.更新的话，比如本地是八月最新，预测九月，当前已经到十月初，那么就是跟新八月和九月，预测十月。
    比如，hg周期=2，就是每次多添加最近2个周期的数据作为训练数据。
    要预测回测八月九月，八月-hg周期2=6月开始》最终得到对九月的预测。同理，九月-7月》预测十月数据。
    zong_t = [dt.datetime(2020,6, 1), dt.datetime(2020,9, 1)]，最早的训练数据，最新的训练数据.
    全部重测的话，
    1就是要确定，重测开始时间，满足开始时间的月份有训练数据。
    '''
    #【  最新月-回滚周期 ，最新的训练数据】
    zong_t = [dt.datetime(2020,7, 1), dt.datetime(2020,10, 1)]

    timeflag =( datetime.datetime.now()).strftime('%m_%d')

    # 进行第一次机器学习和训练
    if 0 ==True:
        # 学习结果保存地址

        res_path0 = r'F:\new_0811\huice_log\res_data\固定贪心训练结果'
        for bh in ['01','02','03','04']:
            #训练数据地址
            train_path = r'F:\new_0811\huice_log\train_data\MEBD_%s_每月训练数据_周期1_回测周期3_%s.pickle'%(str(bh),timeflag)
            flag0 = train_path.split('\\')[-1].split('.')[0]
            # 分类
            q_n = 20
            ld = 1 #粒度
            tolocaldf =pre_mode(train_path, res_path0, zong_t, res_flag = f'_{flag0}训练结果_{q_n}_{ld}f',
                     zuhe_cal ='train_and_predict4', algo='zuhe_911_022', mode = 4,hg_zq = 2
                     , time_lidu = ld,n_jobs=3,q_n = q_n ,test = False)
            print(tolocaldf.keys())
            print('训练结束，预测数据已保存')
    # 结果更新
    if 1 == True:
        for bh in ['01', '02', '03', '04']:
            newpath = r'F:\new_0811\huice_log\res_data\固定贪心训练结果\贪心集中zuhe_911_022_MEBD_%s_每月训练数据_周期1_回测周期3_11-11训练结果_20_1f11-12最新月份数据.pickle'%str(bh)
            local_trainres_path = r'F:\new_0811\huice_log\res_data\tx\贪心集中zuhe_911_022_MEBD_%s_训练结果_20_1f10-14最新月份数据.pickle' % str(bh)

            # 最新数据
            with open(newpath, 'rb') as f:
                newdf = pickle.load(f)
                print(newdf.keys())
            # 之前数据
            with open(local_trainres_path, 'rb') as f:
                local_dict = pickle.load(f)
                print(local_dict.keys())
            # 开始更新
            for k, v in newdf.items():
                print(k)
                print(v['data'].tail())
                local_dict[k] = newdf[k]
                print(f'{k}_{v.keys()}')
            # continue
            with open(newpath, 'wb') as f:
                pickle.dump(local_dict, f)
                print(f'新数据保存到{newpath}')
            for k, v in local_dict.items():
                print(f'{k}_{v.keys()}')

    if 0 == True:
        df =pd.DataFrame()
        for bh in [
            # '01'
            # ,'02'
            # ,'03'
            # ,'04'
                   ]:
            train_path = r'F:\new_0811\huice_log\train_res\MEBD_%s_每月训练数据_周期1_回测周期3_10-13.pickle'%str(bh)
            print(train_path)
            with open(train_path,'rb') as f:
                file_0 = pickle.load(f)
            for k,v in file_0.items():
                print(k)
                if str(k).endswith('3'):continue
                for t in v.keys():
                    print(t)
                    df = df.append(v[t],ignore_index=True)
        df.sort_values('res_time',inplace=True)
        df.reset_index(inplace=True,drop=True)
        print(df.sample(100))
        print(df.shape)
        df.to_pickle('all_data_df策略01类.pkl')

    if 0==True:
        df1 =pd.read_pickle(r'F:\new_0811\all_data_df策略01类.pkl')
        print(df1.keys())
        print(df1.tail())
