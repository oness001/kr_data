from numba import jit
import numpy as np

@jit(nopython=True)
def cal_tongji(df_input: np.array):
    df1 = df_input.copy()
    # 返回统计数据
    df1[:, 4][np.isnan(df1[:, 4])] = 0
    zjline = np.cumsum(df1[:, 4])
    maxback_line = zjline.copy()
    max_log = 0
    for i in range(1, zjline.shape[0]):
        max_log = max(max(zjline[i], zjline[i - 1]), max_log)
        maxback_line[i] = max_log - zjline[i]

    end_zj = zjline[-1]
    max_zj = max(zjline)
    std_zj = np.std(zjline)
    mean_zj = np.mean(zjline)
    max_back = max(maxback_line)
    y_ = (df1[:, 5][np.where(df1[:, 5] > 0)])
    k_ = (df1[:, 5][np.where(df1[:, 5] <= 0)])
    counts = (y_.shape[0] + k_.shape[0])
    sl = y_.shape[0] / (y_.shape[0] + k_.shape[0]) if counts != 0 else 1
    yk_rate = 1 / (y_.sum() * -1 / k_.sum()) if k_.sum() != 0 else 1
    yc_rate = end_zj / max_back if max_back != 0 else 0
    sharp_rate = mean_zj / std_zj if std_zj != 0 else 0

    res0 = np.array([end_zj, max_zj, (max_back), yc_rate, sharp_rate, mean_zj, (counts), sl, yk_rate])
    return res0

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
            df1[i][5] = open_bar - (sxf + slip) - last_open_prcie
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
def put_stop_order(type_bs,stop_price,high_bar,low_bar):
    if abs(type_bs) != 1:
        return np.array([np.nan,np.nan])
    stop_signal = np.nan
    if high_bar <= stop_price:
        stop_signal = np.nan if type_bs ==1 else -1
    elif  (low_bar >= stop_price) :
        stop_signal = np.nan if type_bs == -1 else 1
    elif  (high_bar > stop_price) or(low_bar < stop_price) :
        stop_signal = 1.1 *type_bs
    return np.array([stop_signal,stop_price])

@jit(nopython=True)
def deal_per_bar_stop_order(stop_order1=np.array([np.nan,np.nan]),stop_order2=np.array([np.nan,np.nan])):
    # per_bar_stop_order=np.full([2,2], np.nan)
    # per_bar_stop_order[0]=stop_order1
    # per_bar_stop_order[1]=stop_order2
    '''
    只接受最多2个停止单子，一共有9个组合，，
    分别为：(np.nan,1,1.1) ,(np.nan,-1,-1.1)的组合任意组合
    原则：同时开仓方向一样，以停止单先成交之后不成交。
    同时开仓方向不一样以亏损为原则，不考虑赚钱的情况，即是成交先后！

    返回一个数组：[信号，价格]
    '''
    if np.isnan(stop_order1[0])  :
        if not np.isnan(stop_order2[0]):return stop_order2
        else: return np.array([np.nan,np.nan])
    elif np.isnan(stop_order2[0]) :
        if not np.isnan(stop_order1[0]):return stop_order1
        else: return np.array([np.nan,np.nan])
    elif abs(stop_order1[0])==abs(stop_order2[0])==1: 
        return np.array([np.nan,np.nan])
    elif abs(abs(stop_order1[0])-abs(stop_order2[0]))==0.1:
        if abs(stop_order1[0]) ==1.1:return np.array([0.1*stop_order1[0],stop_order1[1]])
        else:return np.array([0.1*stop_order2[0],stop_order2[1]])
    elif abs(stop_order1[0]) ==abs(stop_order2[0]) ==1.1:
        cha = abs(stop_order1[1]-stop_order2[1])
        return np.array([0.5,cha])

    else: return np.array([np.nan,np.nan])


@jit(nopython=True)
def cal_per_pos_by_stop_order(i, df1: np.array, open_pos_size, open_bar, clsoe_bar,last_open, last_close,stop_order: np.array, sxf=1, slip=1):
    last_signal = df1[i - 1][1]
    last_bar_pos = df1[i - 1][2]
    last_open_prcie = df1[i - 1][3]
    stop_price = stop_order[1]

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
    
    # 平仓停止单
    elif last_signal == 1.1 and last_bar_pos != 0:
        df1[i][2] = 0  # 平仓
        df1[i][4] = (stop_price - last_open) * last_bar_pos - (sxf + slip)
        if last_bar_pos > 0:
            df1[i][3] = stop_price - (sxf + slip)
            df1[i][5] = stop_price  - last_open_prcie- (sxf + slip)
        elif last_bar_pos < 0:
            df1[i][3] = stop_price + sxf + slip
            df1[i][5] = stop_price  - last_open_prcie + sxf + slip

    # 双向开仓停止单。亏！！！
    elif abs(last_signal) == 0.5 :  
        #特殊！！！
        df1[i][2] = 1 * open_pos_size
        df1[i][3] = (clsoe_bar+open_bar)/2
        df1[i][4] = (clsoe_bar-last_close)*open_pos_size-(abs(stop_price)+ (sxf + slip)*2)* max(1,abs(open_pos_size)) 

        df1[i][4] = stop_price* abs(open_pos_size) - (sxf + slip)*2
    elif abs(last_signal) == 0.11  :  
       #特殊！！！
        df1[i][2] = 1 * open_pos_size
        df1[i][3] = abs(stop_price - open_bar)
        df1[i][4] = (clsoe_bar-last_close)*open_pos_size-(abs(stop_price - open_bar)+ (sxf + slip)*2)* max(1,abs(open_pos_size)) 

    # 仓位不变===记录开/平仓价【3】，当日盈亏【4】，变化点数乘以仓位
    else:
        df1[i][2] = last_bar_pos
        df1[i][3] = last_open_prcie
        df1[i][4] = (clsoe_bar - last_close) * df1[i][2]

    return df1, df1[i][3], df1[i][2]
