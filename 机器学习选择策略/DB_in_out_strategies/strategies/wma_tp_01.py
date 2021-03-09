import datetime as dt
import numpy as np
import pandas as pd
import talib
# import pandas_talib as ptalib

from tzlocal import get_localzone

from vnpy.app.ib_cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)



class wma_tp_01(CtaTemplate):
    """"""
    #=== 注明所需要的变量
    # 交易时间设置
    DAY_START =  dt.time(9, 50)
    DAY_END = dt.time(16, 00)
    close_pos_time = dt.time(hour=15,minute=59)

    trading_time_hours = [9,10,11,12,13,14,15,16]
    trading_time_mins= [50,5]
    is_trading = False

    orderid_condition = []
    orderid_sl = []
    load_days = 3
    ma_len ,bd = 60 ,4
    dk_len = 20
    acc = 7
    stop_loss = 100
    zhouqi = 3
    parameters = ['ma_len','bd',"dk_len",'acc','zhouqi']

    up,do,me = 0 ,0,0
    acc0 = -10

    variables = []
    open_long_price = 0
    open_short_price = 0
    close_long_price = 0
    close_short_price = 0
    # 止盈止损价预设
    stop_loss_price = 0
    stop_yingli_price = 0

    records = ['up', 'do','stop_loss_price']


    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):

        super(wma_tp_01, self).__init__( cta_engine, strategy_name, vt_symbol, setting )

        #===k-bar合成管理功能函数
        self.bg = BarGenerator(self.on_bar, self.zhouqi, self.on_3min_bar)
        #===运行态，保存3min_bar功能函数
        self.am = ArrayManager(size=150)


    # === 新的bar更新之后，怎么操作：
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.bg.update_bar(bar)


    #=== 每个5min——bar怎么操作：
    def on_3min_bar(self, bar: BarData):
        """"""
        self.am.update_bar(bar)
        am = self.am
        if not am.inited: return
        # === 交易订单处理
        for orderid in self.orderid_condition:
            self.cancel_order(orderid)
        self.orderid_condition.clear()
        # 指标计算
        me = talib.WMA(am.close, self.ma_len)
        # me_2 = talib.MA(am.high, self.ma_len)
        # me_3 = talib.MA(am.low, self.ma_len)

        # 计算指标波动率
        atr = am.atr(self.ma_len)

        up = me + atr * self.bd
        do = me - atr * self.bd

        mo = (am.close[-1] - am.close[self.dk_len])
        canshu = [me, up, do, atr]

        stop_loss_condition, self.stop_loss_price = self.stop_loss_func(am=am, canshu=canshu, bar=bar)

        # 判断在？交易时间段
        if 0 == self.if_can_trading(bar):
            return
        else:
            pass

        # 准备变量
        per_pos = 1  # 每次开仓手数（平仓）
        # 开平价格设置:追入买进
        open_long_price = am.close[-1] + 10
        open_short_price = am.close[-1] - 10
        close_long_price = am.close[-1] - 10
        close_short_price = am.close[-1] + 10

        # === 整理仓位 === 是否要进行，盘尾日内平仓
        if self.pos != 0 and self.if_today_clsoe(bar):
            if self.pos > 0:
                orderid = self.sell(bar.close_price - 1, abs(self.pos))
                self.orderid_condition.extend(orderid)  # 尾部添加orderid
            elif self.pos < 0:
                orderid = self.cover(bar.close_price + 1, abs(self.pos))
                self.orderid_condition.extend(orderid)
            return
        self.cancel_all()  # 清仓

        # === 计算开平，止盈止损，条件
        # open_long_condition = False
        # open_short_condition = False
        # close_long_condition = False
        # close_short_condition = False
        # stop_loss_condition = False
        # stop_yingli_condition = False

        if self.pos == 0:
            open_long_condition = am.close[-1] > up[-1] and am.close[-2] <= up[-2] and mo >= 0
            open_short_condition = am.close[-1] < do[-1] and am.close[-2] >= do[-2] and mo <= 0

            # close_long_condition  =  am.close[-1]  < me[-1] and am.close[-2]  >= me[-2]     #平多
            # close_short_condition = am.close[-1]  > me[-1] and am.close[-2]  <= me[-2]      # 平多

            # === 判断开平仓
            if open_long_condition:
                self.buy(open_long_price, per_pos, stop=False)
            elif open_short_condition:
                self.sell(open_short_price, per_pos, stop=False)
            else:
                # print('无交易')
                pass

        elif self.pos > 0:
            close_long_condition = am.close[-1] < me[-1] and am.close[-2] >= me[-2]
            open_short_condition = am.close[-1] < do[-1] and am.close[-2] >= do[-2] and mo <= 0
            stop_yingli_condition = False
            # === 判断止盈止损
            if stop_loss_condition:
                self.sell(close_long_price, per_pos, stop=False)
            else:
                # === 判断开平仓
                if open_short_condition:
                    self.sell(open_short_price, per_pos * 2, stop=False)
                elif close_long_condition:
                    self.sell(close_long_price, per_pos, stop=False)
                else:
                    pass

        elif self.pos < 0:
            close_short_condition = am.close[-1] > me[-1] and am.close[-2] <= me[-2]
            open_long_condition = am.close[-1] > up[-1] and am.close[-2] <= up[-2] and mo >= 0

            # === 判断止盈止损
            if stop_loss_condition:
                self.buy(close_short_price, per_pos, stop=False)
            # === 判断开平仓
            else:

                if open_long_condition:
                    self.buy(open_long_price, per_pos * 2, stop=False)
                elif close_short_condition:
                    self.buy(close_short_price, per_pos, stop=False)
                else:
                    # print('无交易')
                    pass
            # print(bar.datetime, '当前仓位:', self.pos)
            # print(stop_loss_condition)



        else:
            pass

        self.up, self.do, self.me = up[-1], do[-1], me[-1]
        self.put_event()

    #===初始化策略
    def on_init(self):
        """
        Callback when strategy is inited.
        """
        # ===提前加载k_bar
        self.load_bar(days = self.load_days, callback=self.init_bar)

    #过滤bar，
    def init_bar(self, bar: BarData):
        if bar.datetime.replace(second=0) >= dt.datetime.now(get_localzone()).replace(second=0) - dt.timedelta(minutes=1):
            self.bg.bar = bar
        else:
            self.on_bar(bar)

    #=== 启动策略执行操作
    def on_start(self):
        """
        Callback when strategy is started.
        """
    #=== 策略停止执行操作
    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
    #=== 每个tick怎么操作：策略主逻辑。
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)




    #=== 如果有断网等情况，，恢复数据和参数等数据
    def on_recover(self, variables):

        for n, v in variables.items():
            setattr(self, n, v)

    #===下单之后，返回的结果：》订单状态
    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass
    #=== 每次有成交撮合，成功的信息，则返回成交订单状态
    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
    #=== 停止单的信息状态返回
    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

    #=== 计算当天的开盘时间与收盘时间，返回是否开始交易
    def if_can_trading(self,bar:BarData,):
        # 计算当天的开盘时间与收盘时间
        if (dt.time(bar.datetime.hour, bar.datetime.minute) >= self.DAY_START) and\
                (dt.time(bar.datetime.hour,bar.datetime.minute) <= self.DAY_END):
            is_trading = True
        else:is_trading = False

        return is_trading

    #=== 判断今日尾盘是否清仓
    def if_today_clsoe(self,bar):
        return bar.datetime.time() >= self.close_pos_time

    def stop_loss_func(self,am,canshu,bar,acc0 = -10):
        me = canshu[0]
        stop_loss_price_shift = self.stop_loss_price

        if self.pos > 0:
            if am.close[-1] > am.open[-1]:
                self.acc0 += self.acc
                self.stop_loss_price = me[-1] + self.acc0
            else:self.stop_loss_price = self.stop_loss_price
            self.stop_loss_price =  max(stop_loss_price_shift,self.stop_loss_price)
            if am.close[-1] < self.stop_loss_price:
                return True ,self.stop_loss_price
            else:
                return False ,self.stop_loss_price

        elif self.pos < 0:
            if am.close[-1] < am.open[-1]:
                self.acc0 += self.acc
                self.stop_loss_price = me[-1] - self.acc0

            else:self.stop_loss_price = self.stop_loss_price

            self.stop_loss_price =  min(stop_loss_price_shift,self.stop_loss_price)

            if am.close[-1] > self.stop_loss_price :
                return True ,self.stop_loss_price
            else:
                return False ,self.stop_loss_price
        elif self.pos == 0:
            self.stop_loss_price = me[-1]
            self.acc0 = acc0
            return False, self.stop_loss_price
        else:
            print('情况出错了')
            self.stop_loss_price = me[-1]
            self.acc0 = acc0
            return False ,self.stop_loss_price

    def open_triopen_trigger(self,):
        # 记录trend的时间周期
        self.time_go

        if self.trend == 1:
            self.time_go
            pass
        elif self.trend == -1:
            pass
        elif self.trend ==0:
            pass
        else:pass