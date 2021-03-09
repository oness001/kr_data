from numba import jit
import numpy as np
import pandas as pd

@jit(nopython=True,error_model="numpy")
def cal_tongji(df_input: np.array):
    df1 = df_input.copy()
    df1[:, 4][np.isnan(df1[:, 4])] = 0
    zjline = np.cumsum(df1[:, 4])
    maxback_line = zjline.copy()
    end_zj, max_zj, max_back, yc_rate, sharp_rate, stn_rate, yk_risk_rate, mean_zj, counts, sl, yk_rate =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 返回统计数据
    max_log = 0
    end_zj = zjline[-1]
    max_zj = max(zjline)
    counts= len(df1[:, 5][np.where(~np.isnan(df1[:, 5]))])
    if (counts == 0 ) or (end_zj == 0):

        res0 = np.array([end_zj, max_zj, max_back, yc_rate, sharp_rate, stn_rate, yk_risk_rate, mean_zj, counts, sl, yk_rate])
        return res0

    else:
        # std_zj = np.std(zjline)
        # print(zjline[-1],zjline.shape[0])
        for i in range(1, zjline.shape[0]):
            max_log = max(max(zjline[i], zjline[i - 1]), max_log)
            maxback_line[i] = max_log - zjline[i]
        max_back = max(maxback_line)
        y_ = (df1[:, 5][np.where(df1[:, 5] > 0)])
        k_ = (df1[:, 5][np.where(df1[:, 5] <= 0)])
        all = df1[:, 5][np.where(~np.isnan(df1[:, 5]))]

        # print(1,max_back,len(k_),len(all_))
        sl = y_.shape[0] / counts if counts > 1 else 1
        yc_rate = end_zj / max_back if max_back != 0 else 0
        sharp_rate = np.mean(all) / np.std(all) if counts >1 else 0
        yk_rate =  y_.mean()/(k_.mean()* -1) if len(k_) >1 else y_.sum()
        stn_rate = np.mean(all) / np.std(k_) if len(k_) >1 else 0
        yk_risk_rate = np.std(y_) / np.std(k_) if len(k_) >1 else 0
        mean_zj =  np.std(y_) - np.std(k_) if len(k_) >1 else 0
        res0 = np.array([end_zj, max_zj, max_back, yc_rate, sharp_rate,stn_rate,yk_risk_rate, mean_zj, counts, sl, yk_rate])
        return res0
@jit(nopython=True)
def cal_tongji_pro_1(df_input: np.array):
    df1 = df_input.copy()
    df1[:, 4][np.isnan(df1[:, 4])] = 0
    zjline = np.cumsum(df1[:, 4])
    maxback_line = zjline.copy()
    end_zj, max_zj, max_back, yc_rate, sharp_rate, stn_rate, yk_risk_rate, mean_zj, counts, sl, yk_rate =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jingzhi_array = zjline
    # 返回统计数据

    max_log = 0



    end_zj = zjline[-1]
    max_zj = max(zjline)
    counts= len(df1[:, 5][np.where(~np.isnan(df1[:, 5]))])
    if (counts == 0 ) or (end_zj == 0):

        res0 = np.array([end_zj, max_zj, max_back, yc_rate, sharp_rate, stn_rate, yk_risk_rate, mean_zj, counts, sl, yk_rate])
        return res0,jingzhi_array

    else:
        # std_zj = np.std(zjline)
        # print(zjline[-1],zjline.shape[0])
        for i in range(1, zjline.shape[0]):
            max_log = max(max(zjline[i], zjline[i - 1]), max_log)
            maxback_line[i] = max_log - zjline[i]
        max_back = max(maxback_line)
        y_ = (df1[:, 5][np.where(df1[:, 5] > 0)])
        k_ = (df1[:, 5][np.where(df1[:, 5] <= 0)])
        all = df1[:, 5][np.where(~np.isnan(df1[:, 5]))]

        # print(1,max_back,len(k_),len(all_))
        sl = y_.shape[0] / counts if counts > 1 else 1
        yc_rate = end_zj / max_back if max_back != 0 else 0
        sharp_rate = np.mean(all) / np.std(all) if counts >1 else 0
        yk_rate =  y_.mean()/(k_.mean()* -1) if len(k_) >1 else y_.sum()
        stn_rate = np.mean(all) / np.std(k_) if len(k_) >1 else 0
        yk_risk_rate = np.std(y_) / np.std(k_) if len(k_) >1 else 0
        mean_zj =  np.std(y_) - np.std(k_) if len(k_) >1 else 0
        jingzhi_array = zjline
        res0 = np.array([end_zj, max_zj, max_back, yc_rate, sharp_rate,stn_rate,yk_risk_rate, mean_zj, counts, sl, yk_rate])

        return res0,jingzhi_array



@jit(nopython=True)
def cal_per_pos(i, df1: np.array, open_pos_size, open_bar, clsoe_bar, last_close, sxf=1, slip=1):
    last_signal = df1[i - 1][1]
    last_bar_pos = df1[i - 1][2]
    last_open_prcie = df1[i - 1][3]

    # 开多===上一根信号=1 仓位 ==0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点
    if last_signal >= 1 and last_bar_pos == 0:
        df1[i][2] = open_pos_size
        df1[i][3] = open_bar + sxf + slip
        df1[i][4] = (clsoe_bar - open_bar) * open_pos_size - (sxf + slip)

    # # # 空转多===上一根信号=1 仓位 < 0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点， 结束平仓胜负【5】
    elif last_signal == 1 and last_bar_pos < 0:
        df1[i][2] = open_pos_size
        df1[i][3] = open_bar + sxf + slip
        df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
        df1[i][4] += (clsoe_bar - open_bar) * open_pos_size - (sxf + slip)
        df1[i][5] = open_bar - (sxf + slip) - last_open_prcie

    # # 多转空===上一根信号=-1 仓位 > 0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点， 结束平仓胜负【5】
    elif (last_signal == -1) and (last_bar_pos > 0):
        df1[i][2] = -1 * open_pos_size
        df1[i][3] = open_bar - (sxf + slip)
        # 先平多，在开空
        df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
        df1[i][4] += (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)
        df1[i][5] = open_bar - (sxf + slip) - last_open_prcie

    # 开空===上一根信号=-1， 【2】仓位==0,   【3】记录持仓价，  【4】当日盈亏-手续费和滑点
    elif last_signal < 0 and last_bar_pos == 0:  # ===开空
        df1[i][2] = -1 * open_pos_size
        df1[i][3] = open_bar - (sxf + slip)
        df1[i][4] = (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)

    # 平仓===上一根信号=0 【2】仓位 != 0,记录平仓价【3】，当日盈亏【4】-手续费和滑点
    elif last_signal == 0 and abs(last_bar_pos) > 0:
        df1[i][2] = 0  # 平仓
        df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
        if last_bar_pos > 0:
            df1[i][3] = open_bar - (sxf + slip)
            df1[i][5] = open_bar  - last_open_prcie- (sxf + slip)
        elif last_bar_pos < 0:
            df1[i][3] = open_bar + sxf + slip
            df1[i][5] = open_bar + sxf + slip - last_open_prcie
    # 仓位不变===记录开/平仓价【3】，当日盈亏【4】，变化点数乘以仓位
    else:
        df1[i][2] = last_bar_pos
        df1[i][3] = last_open_prcie
        df1[i][4] = (clsoe_bar - last_close) * df1[i][2]

    return df1, df1[i][3], df1[i][2]
@jit(nopython=True)
def to_deal_stop_price(type0,stop_up,stop_dn,high_bar,low_bar,now_pos):
    '''
    type0:   1:buy ,  -1:sell ,  0:close

    :param stop_up:
    :param stop_dn:
    :param high_bar:
    :param low_bar:
    :return:
    '''
    up_con1 = (stop_up <  high_bar)
    up_con2 = (stop_up > low_bar)
    dn_con1 = (stop_dn <  high_bar)
    dn_con2 = (stop_dn >  low_bar)
    # 上穿 0.1，1.1，-1.1
    if (up_con1 and up_con2 and (not dn_con2)) :
        if abs(now_pos) > 0 and type0 == 0: return 0.1,stop_up
        elif now_pos == 0 and type0 == 1:   return 1.1 ,stop_up
        elif now_pos == 0 and type0 == -1:  return -1.1 ,stop_up

    # 下穿 0.1，1.1，-1.1
    if (dn_con1 and dn_con2 and (not up_con1)):
        if abs(now_pos) > 0 and type0 == 0:return 0.1, stop_dn
        elif now_pos == 0 and type0 == 1:return 1.1, stop_dn
        elif now_pos == 0 and type0 == -1:return -1.1, stop_dn

    # 同时上下,采用不利原则。0.1 ，1.5
    elif up_con1 and dn_con2:
        # 采用不利原则，当日不利平仓
        if (now_pos) > 0 and type0 == 0:return 0.1,stop_dn
        elif (now_pos) < 0 and type0 == 0:return 0.1,stop_up
        # 亏损原则，一开一平
        elif now_pos == 0 and abs(type0) == 1:return 1.5,(stop_up - stop_dn)

    # 直接超越，次日买入 0，1，-1
    elif (up_con1 and not up_con2) or (not dn_con1 and dn_con2):
        if  abs(now_pos) >0 and type0 == 0 :return 0,np.nan
        elif now_pos == 0 and type0 == 1:return 1,np.nan
        elif now_pos == 0 and type0 == -1:return -1,np.nan
    else:
        return np.nan ,np.nan

@jit(nopython=True)
def cal_per_pos_with_stop_order(i, df1: np.array, open_pos_size, open_bar, clsoe_bar, last_close,stop_price, sxf=1, slip=1):
    last_signal = df1[i - 1][1]
    last_bar_pos = df1[i - 1][2]
    last_open_prcie = df1[i - 1][3]

    # 开多===上一根信号=1 仓位 ==0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点
    if last_signal == 1 and last_bar_pos == 0:
        df1[i][2] = open_pos_size
        df1[i][3] = open_bar + sxf + slip
        df1[i][4] = (clsoe_bar - open_bar) * open_pos_size - (sxf + slip)

    # # # 空转多===上一根信号=1 仓位 < 0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点， 结束平仓胜负【5】
    elif last_signal == 1 and last_bar_pos < 0:
        df1[i][2] = open_pos_size
        df1[i][3] = open_bar + sxf + slip
        df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
        df1[i][4] += (clsoe_bar - open_bar) * open_pos_size - (sxf + slip)
        df1[i][5] = open_bar - (sxf + slip) - last_open_prcie

    # # 多转空===上一根信号=-1 仓位 > 0 ,记录持仓价【3】，当日盈亏【4】-手续费和滑点， 结束平仓胜负【5】
    elif (last_signal == -1) and (last_bar_pos > 0):
        df1[i][2] = -1 * open_pos_size
        df1[i][3] = open_bar - (sxf + slip)
        # 先平多，在开空
        df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
        df1[i][4] += (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)
        df1[i][5] = open_bar - (sxf + slip) - last_open_prcie

    # 开空===上一根信号=-1， 【2】仓位==0,   【3】记录持仓价，  【4】当日盈亏-手续费和滑点
    elif last_signal == -1 and last_bar_pos == 0:  # ===开空
        df1[i][2] = -1 * open_pos_size
        df1[i][3] = open_bar - (sxf + slip)
        df1[i][4] = (clsoe_bar - open_bar) * -1 * open_pos_size - (sxf + slip)

    # 平仓===上一根信号=0 【2】仓位 != 0,记录平仓价【3】，当日盈亏【4】-手续费和滑点
    elif (last_signal == 0)or(last_signal == 0.2) and abs(last_bar_pos) > 0:
        df1[i][2] = 0  # 平仓
        df1[i][4] = (open_bar - last_close) * last_bar_pos - (sxf + slip)
        if last_bar_pos > 0:
            df1[i][3] = open_bar - (sxf + slip)
            df1[i][5] = open_bar - (sxf + slip) - last_open_prcie
        elif last_bar_pos < 0:
            df1[i][3] = open_bar + sxf + slip
            df1[i][5] = open_bar + sxf + slip - last_open_prcie


    # 开仓停止单。
    elif abs(last_signal) == 1.1  and last_bar_pos == 0:  # ===开空
        if last_signal > 0:

            df1[i][2] = 1 * open_pos_size
            df1[i][3] = stop_price + (sxf + slip)
            df1[i][4] = (clsoe_bar - stop_price) * 1 * open_pos_size - (sxf + slip)
        elif last_signal < 0:
            df1[i][2] = -1 * open_pos_size
            df1[i][3] = stop_price - (sxf + slip)
            df1[i][4] = (clsoe_bar - stop_price) * -1 * open_pos_size - (sxf + slip)
    # 双向开仓停止单。亏！！！
    elif abs(last_signal) == 1.5 and last_bar_pos == 0:  # ===开空
       #特殊！！！
        df1[i][2] = 1 * open_pos_size
        df1[i][3] = (clsoe_bar+open_bar)/2
        df1[i][4] = stop_price* open_pos_size - (sxf + slip)*2
    # 平仓停止单
    elif last_signal == 0.1 and last_bar_pos != 0:
        df1[i][2] = 0  # 平仓
        df1[i][4] = (stop_price - last_close) * last_bar_pos - (sxf + slip)
        if last_bar_pos > 0:
            df1[i][3] = stop_price - (sxf + slip)
            df1[i][5] = stop_price  - last_open_prcie- (sxf + slip)
        elif last_bar_pos < 0:
            df1[i][3] = stop_price + sxf + slip
            df1[i][5] = stop_price  - last_open_prcie + sxf + slip



    # 仓位不变===记录开/平仓价【3】，当日盈亏【4】，变化点数乘以仓位
    else:
        df1[i][2] = last_bar_pos
        df1[i][3] = last_open_prcie
        df1[i][4] = (clsoe_bar - last_close) * df1[i][2]

    return df1, df1[i][3], df1[i][2]

def huice_hsi(df,bzj = 1/5, leverage_rate=1, c_rate=20, hycs=50, slip =1, min_margin_rate=(0.8),is_print=True):

    """
    基于恒生指数期数的回测框架
    :param df:  带有signal和pos的数据
    bzj = 1/10:  一手合约所需现金比例
    :param leverage_rate:  最多提供?倍杠杆，
    :param c_rate:  手续费0.002(固定模式,百分比),,固定金额:如20元
    :param  hycs:  合约乘数
    :param min_margin_rate:  低保证金比例，必须占到投入保证金的?%以上.这里取50%,否则爆仓
    slip = 1 :交易滑点.
    :return: df 带资金曲线, 处理好的df:可以进行可视化
    """

    def max_huiche(df):
        df = df[['candle_begin_time', 'equity_curve']].copy()
        # 计算当日之前的资金曲线的最高点
        df['max2here'] = df['equity_curve'].expanding().max()
        # 计算到历史最高值到当日的跌幅，drowdwon
        df['dd2here'] = (df['max2here'] - df['equity_curve']) * 100 / df['max2here']
        # 计算最大回撤，以及最大回撤结束时间
        end_date, max_draw_down = tuple(df.sort_values(by=['dd2here']).iloc[-1][['candle_begin_time', 'dd2here']])
        # 计算最大回撤开始时间
        start_date = df[df['candle_begin_time'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['candle_begin_time']
        # 将无关的变量删除
        df.drop(['max2here', 'dd2here'], axis=1, inplace=True)

        return max_draw_down, start_date, end_date
    def max_huiche2(df):
        df = df[['candle_begin_time', 'open_lr']].copy()
        # 计算当日之前的资金曲线的最高点
        df['max2here'] = df['open_lr'].expanding().max()
        # 计算到历史最高值到当日的跌幅，drowdwon
        df['dd2here'] = (df['max2here'] - df['open_lr'])
        # 计算最大回撤，以及最大回撤结束时间
        end_date, max_draw_down = tuple(df.sort_values(by=['dd2here']).iloc[-1][['candle_begin_time', 'dd2here']])
        # print(df['dd2here'],df.sort_values(by=['dd2here']) )
        # 计算最大回撤开始时间
        start_date = df[df['candle_begin_time'] <= end_date].sort_values(by='open_lr', ascending=False).iloc[0]['candle_begin_time']
        # 将无关的变量删除
        df.drop(['max2here', 'dd2here'], axis=1, inplace=True)

        return max_draw_down, start_date, end_date
    # =====基本参数

    init_cash = 1000000
    min_margin =  bzj * min_margin_rate  # 最低保证金
    df['pos'] = df['signal'].shift()
    df['pos'].fillna(method='ffill',inplace = True)
    df['pos'].fillna(value=0,inplace = True)
    print('已经生成pos！')
    df['close_手价值'] = df['close']* hycs

    # =====根据pos计算资金曲线
    # ===计算涨跌幅
    df['change'] = df['close'].pct_change(1)  # 根据收盘价计算涨跌幅
    df['next_open_diancha'] = (df['open'].shift(-1) - df['close']) *df['pos']# 从今天开盘买入，到今天收盘的涨跌幅


    # ===选取开仓、平仓条件
    condition0 = df['pos'] != np.nan
    condition1 = df['pos'] != 0
    condition2 = df['pos'] != df['pos'].shift(1)#上一根
    open_pos_condition = condition0 & condition1 & condition2

    condition0 = df['pos'] != np.nan
    condition1 = df['pos'] != 0
    condition2 = df['pos'] != df['pos'].shift(-1)
    close_pos_condition =condition0& condition1 & condition2

    # ===对每次交易进行分组
    df.loc[open_pos_condition, 'start_time'] = df['candle_begin_time']
    df['start_time'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

    # ===计算仓位变动,  # 开仓时仓位价值大小
    # 开仓价计算
    df.loc[open_pos_condition, 'open_price'] = df.loc[open_pos_condition, 'open'] + slip * df.loc[open_pos_condition, 'pos']
    df.loc[open_pos_condition, 'open_position'] =  leverage_rate * (df.loc[open_pos_condition, 'open_price'])*hycs
    df['open_price'].fillna(method='ffill', inplace=True)
    df['open_position'].fillna(method='ffill', inplace=True)
    df['open_price'].fillna(0, inplace=True)

    df['diancha'] = (df['close'] - df['open_price']) * df['pos']

    df.loc[close_pos_condition, 'diancha'] = df.loc[close_pos_condition, 'diancha'] + (df.loc[close_pos_condition, 'next_open_diancha'])

    df['position'] = (df['diancha'])*hycs+df['open_position']

    df.loc[open_pos_condition, 'position'] = df.loc[open_pos_condition, 'open_position'] + df['diancha']*hycs
    # 开仓后每天的仓位的变动
    # group_num = len(df.groupby('start_time'))
    # #每次开仓,仓位价值变动
    # if group_num > 1:#x['close']:指对每个分组里面的['close']进行修改 position_2
    #
    #     # p = df.groupby('start_time').apply(lambda x: x.iloc[0]['open_position'] * x['close'] / x.iloc[0]['open_price'])
    #     p= df.groupby('start_time').apply(lambda x:  (x['diancha']+x['open_price'])*hycs )
    #
    #     p = p.reset_index(level=[0])  # 还原第一列index为列,并生成一个0123的新的index.
    #     print(p)
    #     df['position'] = p[0]
    #     # exit()
    # elif group_num == 1:
    #     p = df.groupby('start_time')[['open_position', 'diancha']].apply(lambda x: (x['diancha'])*hycs + x.iloc[0]['open_position'])
    #     df['position'] = p.T.iloc[:, 0]

    # 平仓时仓位:下跟进行仓位交易,本周期,记录平仓后(换仓后的仓位价值)



    # 计算持仓利润

    df.loc[close_pos_condition, 'diancha'] = (df['open'].shift(-1) - df['open_price']) * df['pos']
    df.loc[df['pos'] != 0 , 'sxf_fee'] = c_rate
    df.loc[close_pos_condition, 'sxf_fee'] =c_rate*2
    df['sxf_fee'].fillna(value=0, inplace=True)

    df['profit'] = (df['position'] - df['open_position'])  #持仓盈利或者损失
    df['profit'].fillna(method='ffill', inplace=True)
    df['profit'].fillna(value=0, inplace=True)

    # 计算持仓利润最小值
    # 多仓#
    df.loc[df['pos'] < 0, 'profit_min'] = hycs*(df['high']-df['open_price'])*(-1)  # 最小持仓盈利或者损失
    # 空仓#
    df.loc[df['pos'] > 0, 'profit_min'] = hycs*(df['low'] -df['open_price'])*(1)  # 最小持仓盈利或者损失

    try:
        # ===计算资金曲线:
        # 累计利润
        df['open_lr'] = 0
        chicang_profit = [0] #累计利润，计算时，profit已经去除滑点只需统计累计利润
        for i in range(1,df.shape[0],1):
            # 无仓位
            if df.iloc[i]['pos'] == 0 and df.iloc[i-1]['pos'] == 0:
                df.loc[i, 'open_lr'] = chicang_profit[0]
                pass
            # 0仓>开仓
            elif df.iloc[i]['pos'] != 0 and df.iloc[i-1]['pos'] == 0:
                chicang_profit[0] -= c_rate #累计持仓利润先剪掉手续费
                #利润累计。
                df.loc[i,'open_lr'] = chicang_profit[0] + df.iloc[i]['profit']
            #持仓态
            elif df.iloc[i]['pos'] != 0 and df.iloc[i]['pos'] == df.iloc[i-1]['pos']:
                df.loc[i,'open_lr'] = chicang_profit[0] + df.iloc[i]['profit']
            # 有仓位》》平仓
            elif df.iloc[i-1]['pos'] != 0 and df.iloc[i]['pos'] == 0:
                # 平，今日平仓，手续费扣除到上日，累计利润记录
                chicang_profit[0] += (df.iloc[i - 1]['profit'] - c_rate)
                df.loc[i - 1, 'open_lr'] = chicang_profit[0]
                df.loc[i, 'open_lr'] = chicang_profit[0] + df.iloc[i]['profit']

            # 换仓:空>多,多>空
            elif df.iloc[i]['pos'] != df.iloc[i-1]['pos'] and df.iloc[i]['pos'] != 0:
                # 平，记录》在赋值
                chicang_profit[0] += ( df.iloc[i-1]['profit'] - c_rate)
                df.loc[i-1,'open_lr'] = chicang_profit[0]
                # 开
                chicang_profit[0] -= c_rate
                df.loc[i,'open_lr'] = chicang_profit[0] + df.iloc[i]['profit']

            else:
                print('没有这种情况')
        df['open_lr'].fillna(value = 0, inplace=True)

        # 每次都恢复初始保证金init_cash
        df['cash'] = init_cash
        df['cash'] += ( df['profit'] - df['sxf_fee'])
        #不追加保证金，一次足额，保证金
        df['true_cash'] = init_cash + df['open_lr']

        # 单笔资金曲线
        df['cash_rate'] = df['cash'] /init_cash
        df['cash_rate'].fillna(value = 1, inplace=True)


        df['cash_min'] = df['true_cash'] + (df['profit_min'] - df['profit'] )  # 实际最小资金

        # ===判断是否会爆仓
        _index = df[df['cash_min'] <= min_margin * df['close_手价值']].index
        if len(_index) > 0:
            print(df.iloc[_index]['candle_begin_time'], '有爆仓!\n\n')
            df.loc[_index, '强平'] = 1
            df['强平'].fillna(method='ffill',inplace=True)

        # print(df[['candle_begin_time', 'signal', 'pos', 'open', 'close', 'open_price',
        #           'profit','sxf_fee','open_lr', 'true_cash','cash', 'open_position', 'position',
        #           'close_手价值','start_time']].iloc[500:1000])
        # exit()



        #复利资金曲线
        df['equity_change'] = df['cash'].pct_change()
        df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'cash'] /init_cash  - 1  # 开仓日的收益率
        df['equity_change'].fillna(value=0, inplace=True)
        df['equity_change'] =df['equity_change']+1
        df['equity_curve'] = (df['equity_change']).cumprod()
        df['pos'].fillna(value=0, inplace=True)



        # 统计数据

        df.loc[close_pos_condition, 'profit_count'] = df.loc[close_pos_condition, 'profit']
        count = df[df['profit_count'] != 0]['profit_count'].count()
        yingli = df[df['profit_count'] > 0]['profit_count'].count()

        yingli_ = df[df['profit_count'] > 0]['profit_count'].sum()
        kuisun_ = df[df['profit_count'] < 0]['profit_count'].sum()

        if count != 0 :shenglv = yingli/count
        else:shenglv = 0
        # 是否有亏损
        if kuisun_ != 0 :
            yingkuibi = abs(yingli_/kuisun_)
            yingli_max = df[df['profit_count'] > 0]['profit_count'].max()
            kuisun_min = df[df['profit_count'] < 0]['profit_count'].min()
            # ying_t_s = pd.to_datetime(df[df['profit_count'] == yingli_max]['start_time'].values[0])
            # ying_t_e = pd.to_datetime(df[df['profit_count'] == yingli_max]['candle_begin_time'].values[0])
            # kui_t_s = pd.to_datetime(df[df['profit_count'] == kuisun_min]['start_time'].values[0])
            # kui_t_e = pd.to_datetime(df[df['profit_count'] == kuisun_min]['candle_begin_time'].values[0])
        else:
            yingkuibi = 0
            yingli_max = df[df['profit_count'] > 0]['profit_count'].max()
            kuisun_min = df[df['profit_count'] < 0]['profit_count'].min()
            # ying_t_s = pd.to_datetime(df[df['profit_count'] == yingli_max]['start_time'].values[0])
            # ying_t_e = pd.to_datetime(df[df['profit_count'] == yingli_max]['candle_begin_time'].values[0])
            # kui_t_s = '无亏损记录'
            # kui_t_e = '无亏损记录'

    except Exception as e:
        print(e)

    zuidahuiche,s_mb_time ,e_mb_time= max_huiche(df)

    if is_print:
        print('====复利模式:')
        print('最高收益:', round(df['equity_curve'].max() * 100 , 2),'%')
        print('最低收益:', round(df['equity_curve'].min() * 100, 3), '%')
        print('最终收益:', round(df.iloc[-1]['equity_curve'] * 100, 3), '%')
        print('最大回撤:', round(zuidahuiche, 3),'%')
        print('最大回撤开始与结束时间:', s_mb_time,e_mb_time)

        print('\n====单利模式:')
        zuidahuiche2, s_mb_time, e_mb_time = max_huiche2(df)
        print('最高收益:', round(df['open_lr'].max() , 2), '元')
        print('最大亏损:', round(df['open_lr'].min(), 3), '元')
        print('最终收益:', round(df.iloc[-1]['open_lr'], 3), '元')
        print('最大回撤:', round(zuidahuiche2, 3), '元')
        print('最大回撤开始与结束时间:', s_mb_time,e_mb_time)

        print('\n====统计:')
        print('满仓手续费总大小:', round(count * c_rate*2, 2))
        # print('夏普比率:', round(df['profit'].mean()*100/df['profit'].std(), 2), '%')
        print('开单次数:',round(count,1))
        print('赢单次数:',round(yingli,1))
        print('亏单次数:',round(count - yingli,1) )
        print('开单胜率:',round(shenglv*100,2),'%')
        print('盈亏比:', round(yingkuibi,3))
        # print('利润平均值:', round(df['profit'].mean() , 2),'%')
        # print('利润标准差:', round( df['profit'].std(), 2),'%')
        # print('总赢单大小:',round(yingli_,3))
        # print('总亏单大小:',round(kuisun_,3))
        # print('最大一次赢单:', round(yingli_max,3))
        # print('最大赢单开始时间:', ying_t_s)
        # print('最大赢单结束时间:', ying_t_e)
        # print('最大一次亏单:', round(kuisun_min,3))
        # print('最大亏单开始时间:',kui_t_s)
        # print('最大亏单结束时间:',kui_t_e)

        # ===删除不必要的数据

        df.drop(['change', 'start_time',
                 'profit_min', 'cash_min','profit_count','equity_change'], axis=1, inplace=True)
        # exit()



    return df,[round(df['open_lr'].max() , 2),round(df.iloc[-1]['open_lr'] , 3),round(df['open_lr'].min(), 3),round(zuidahuiche2, 3),round(count,2),round(shenglv*100,3)]




@jit(nopython=True)
def cal_tongji_hsi(df_input: np.array):
    df1 = df_input.copy()
    # 返回统计数据
    df1[:, 4][np.isnan(df1[:, 4])] = 0
    zjline = np.cumsum(df1[:, 4])
    maxback_line = zjline.copy()
    # print(zjline.shape)
    # print(zjline[-1])

    max_log = 0

    for i in range(1, zjline.shape[0]):
        max_log = max(max(zjline[i], zjline[i - 1]), max_log)
        maxback_line[i] = max_log - zjline[i]

    end_zj = zjline[-1]
    max_zj = max(zjline)
    std_zj = np.std(zjline)
    mean_zj = np.std(zjline)

    max_back = max(maxback_line)
    y_ = (df1[:, 5][np.where(df1[:, 5] > 0)])
    k_ = (df1[:, 5][np.where(df1[:, 5] <= 0)])
    counts = (y_.shape[0] + k_.shape[0])
    sl = y_.shape[0] / (y_.shape[0] + k_.shape[0]) if counts != 0 else 1
    yk_rate =  y_.sum()/(k_.sum()* -1) if (k_.sum() !=0) else 1
    yc_rate = end_zj / max_back if max_back != 0 else 0
    sharp_rate = mean_zj / std_zj if std_zj != 0 else 0

    res0 = np.array([end_zj, max_zj, (max_back), yc_rate, sharp_rate, mean_zj, (counts), sl, yk_rate])
    return res0

@jit(nopython=True)
def cal_signal_hsi(df0, df1, df2, cs0, sxf=1, slip=1):
    '''
    # df0 == 原始time，ohlcc,:np.array ：['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'days', 'huors', 'minutes']
    # df1 == 信号统计数据列:np.array ：['candle_begin_time','signal', 'pos', 'opne_price', 'per_lr', 'sl']
    # df2 == 指标列:np.array
    '''

    # 配置临时变量
    open_pos_size = 1

    for i in range(10, df0.shape[0]):  # 类似于vnpy:on_bar
        # 交易所日盘，开放
        if True:
            open_bar = df0[i][1]
            # high_bar = df0[i][2]
            # low_bar = df0[i][3]
            last_close = df0[i - 1][4]
            clsoe_bar = df0[i][4]
            # === 仓位统计。
            df1, now_open_prcie, now_pos = cal_per_pos(i, df1, open_pos_size, open_bar, clsoe_bar, last_close, sxf=1, slip=1)
    res0 = cal_tongji_hsi(df_input=df1)
    res0 = np.concatenate((res0, cs0))
    return df0, df1, df2, res0



