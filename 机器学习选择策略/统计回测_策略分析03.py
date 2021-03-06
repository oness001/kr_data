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

    #  注意，必须先进行字段处理干净，才能选择日期，否则报错。
    da['s_time'] = pd.to_datetime(da['s_time'])
    da = da[pd.to_datetime(da['s_time']) >= zong_t[0]]
    da = da[ pd.to_datetime(da['s_time']) <= zong_t[-1]]

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
    # 计算该策略的统计量特征
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
        train_da0,yinzi0 = corr_jw(train_data=train_da0,n=20, yinzi=new_yinzi, res_index='预测周期真实收益')
        yinzi = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        pre_da0, new_yinzi = build_yinzi(train_data=pre_da0, yinzi=yinzi, res_index='预测周期真实收益')

        pre_da0, yinzi0 = corr_jw(train_data=pre_da0, n=20,yinzi=new_yinzi, res_index='预测周期真实收益')

        # exit()
        yinzi0 = yinzi0
        print(f'本次执行的因子数：{len(yinzi0)}')
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


# 预测模式选择
def pre_mode(train_path, res_path0, zong_t, res_flag=f'_pre_res12.pickle',zuhe_cal='train_and_predict2', algo='cal_zuhe3', mode=2, time_lidu=1,n_jobs=2):
    with open(train_path, 'rb') as fe:
        train_res = pickle.load(fe)
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
            print(list(train_res.keys()))
            for i in train_res[list(train_res.keys())[0]].keys(): #遍历月份
                if datetime.datetime.strptime(i,"%Y-%m-%d" )<(zong_t[0]+relativedelta(months=int(6))):continue
                df_ = pd.DataFrame()
                for k, v in train_res.items():

                    if k.split('_')[-1] == str(time_lidu):
                        df_ = df_.append(v[i], ignore_index=True)
                # print(df_.tail())

                train_data[i] = df_
            # exit()
            # print(set(train_data[i]['celuename'].to_list()))
            # exit()
            train_data0 = {}
            cum_df_ = pd.DataFrame()
            for t,v in train_data.items():
                # print(t)
                cum_df_ = cum_df_.append(v, ignore_index=True)
                train_data0[t] = cum_df_

            eval(zuhe_cal)(train_data=train_data0, res_path=res_path0, algo=algo, res_flag=f'\\tx\\贪心集中{algo}{res_flag}',
                              zong_t=zong_t, hc_zq=6, gd_zq=1,n_jobs=n_jobs)
            time.sleep(5)

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


# 打包原回测文件生成训练数据。
def pack_to_traindatas(data_path, train_path, zong_t,time_lidu,hc_zq=3,watch_data=False):
    '''
   1.读取回测文件。
   2.读取训练数据地址，保存一个空文件。
   3.按策略名字生成训练数据数据  》数据格式：{时间：数据，时间：数据。。。}
   4.最终整理所有数据：》{策略名字：{时间：数据，时间：数据。。。}，
                        策略名字：{时间：数据，时间：数据。。。}，
                        策略名字：{时间：数据，时间：数据。。。}}

    '''
    if watch_data:
        with open(data_path, 'rb') as f0:
            train_data = pickle.load(f0)
            for k, v in train_data.items():
                if k.split('_')[-1] == f'{time_lidu}':
                    print(k)
                    print(v.sample(10))
        return print('查看回测数据完毕。')

    #  1.读取回测文件
    train_datas = {}
    with open(data_path, 'rb') as f0:
        resdf = pickle.load(f0)

    #  2.1读取训练数据地址，保存一个空文件。
    time0 = time.strftime("%m-%d", time.localtime())
    train_path = train_path.split('.')[0] + f'time0_{time_lidu}_{hc_zq}_new_{time0}' + '.pickle'
    if os.path.exists(train_path):  # 读取保存地址
        print(f'地址存在{train_path}。')

    #  2.2读取训练数据地址，保存一个空文件。
    else:

        print(f'文件地址可用，新建一个空文件:{train_path}')
        with open(train_path, 'wb') as f_new:
            pickle.dump(train_datas, f_new)
            print(f'空文件保存在：{train_path}')

    #  3.按策略名字生成训练数据数据
    for k in resdf.keys():  # 遍历里面的策略名
        da = resdf[k]
        #   生成数据
        if len(da.keys()) > 0:
            print(f'{k}开始生成训练数据')
            # 数据格式：{时间：数据，时间：数据。。。}
            trainda = generate_train_data(da, zong_t=zong_t, hc_zq=hc_zq)
            train_datas[k] = trainda
            print(f'训练数据已经添加，当前：{k}')
            with open(train_path, 'wb') as f1:
                print('已存在》key：', train_datas.keys())
                pickle.dump(train_datas, f1)
                print(f'训练数据存放完毕：\n{k}保存在：{train_path}')
    with open(train_path, 'wb') as f1:
        print('生成完毕：已存在key：')
        for i in train_datas.keys():
            print(i)
        print(f'\n保存在：{train_path}')

def updata_to_local_traindatas(data_path,train_path,clnames=[],dates=[],hc_zq=6):
    with open(data_path,'rb') as f:
        hc_res = pickle.load(f)

    with open(train_path,'rb') as f1:
        local_train_res = pickle.load(f1)

    for clm in clnames:
        # 拿到指定策略的本地回测数据
        hc_local_df = hc_res[clm]
        # 检查一下更新的月份是否存在
        for y in dates:
            if y in local_train_res[clm].keys():
                print(f'{clm}训练数据的月份{y}，存在！')
                continue
            # 不存在，需要更新！


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




if __name__ == '__main__':

    '进行第一次学习，两个步骤：    1.生成训练数据。    2.生成学习结果。'

    from 统计分析 import *
    from 机器学习函数 import *
    from 算法组合 import *
    from DB_in_out_strategies.查看_下载_更新_db回测数据_strategies_res import show_local_data
    # 单策略运行
    zong_t = [dt.datetime(2019,6, 1), dt.datetime(2020, 6, 1)]
    # show_local_data(r'F:\new_0811\huice_log\res_data\tx\贪心集中cal_zuhe_gl_02_9_7_2.pickle',show=True)
    # exit()
    # 生成训练数据
    if 1 == True:
        path_ = os.getcwd() + r'\huice_log' + '\MEBD03\dema_tp_03_2019-06-01_3T_8_18.csv'
        data_path = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to03_09-09.pickle'
        train_path = r'F:\new_0811\huice_log\train_res\MEBD_03_训练数据.pkl'
        res_path0 = r'F:\new_0811\huice_log\MEBD03_pre_res'
        time_lidu = 1
        zong_t = [dt.datetime(2019, 4, 1), dt.datetime(2020, 8, 1)]
        pack_to_traindatas(data_path, train_path, zong_t,time_lidu=1, watch_data=False)
    # 进行算法学习
    if 0 ==True:
        res_path0 = r'F:\new_0811\huice_log\MEBD03_pre_res'

        train_paths = [r'F:\new_0811\huice_log\train_res\MEBD策略模型_03_训练数据_time0_1_6_new_09-07.pickle',
                       # r'F:\new_0811\huice_log\train_data\MEBD02_time1_traindatas_特征.pkl',
                       # r'F:\new_0811\huice_log\train_data\MEBD03_time1_traindatas_特征.pkl',
        #
                       ]
        # pre_mode_mulfile(train_paths, res_path0, zong_t,index_list=['_ma_','_wma_','_T3_'], res_flag=f'_9_3.pickle', algo='cal_zuhe_gl_01', mode=1, time_lidu=1,n_jobs =2)
        for train_path in train_paths:
            pre_mode(train_path, res_path0, zong_t, res_flag = f'_9_8_2_test.pickle',zuhe_cal='train_and_predict3', algo='cal_zuhe_gl_02', mode=4, time_lidu=1)



