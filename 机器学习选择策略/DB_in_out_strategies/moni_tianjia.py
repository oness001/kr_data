#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 0014 15:07
# @Author  : Hadrianl 
# @File    : genarate_new_strategies


from typing import List, Iterable
from mongoengine import StringField, Document, BinaryField, register_connection
from vnpy.app.ib_cta_strategy import CtaEngine
from vnpy.event import EventEngine
import os

class StrategyCode(Document):
    name = StringField(required=True)
    class_name = StringField(required=True)
    hash = StringField(max_length=64)
    data = BinaryField()

    meta = {
        'db_alias': 'VNPY_BACKTEST',
        "indexes": [
            {
                "fields": ("class_name",),
            },
            {
                "fields": ("name", "class_name",),
                "unique": True
            },
            {
                "fields": ("name", "class_name", "hash"),
                "unique": True,
            }
        ],
    }

def create_strategies(strategies_list: List, vt_symbol:str):
    s_path = 'strategies'
    if os.path.exists(s_path):
        answer = input('strategies文件夹已存在，是否删除[Y/N][y/n]')
        if answer.lower() in ['y', 'yes']:
            recursive_rm(s_path)


    os.mkdir(s_path)
    with open(os.path.join(s_path, '__init__.py'), 'w') as f:
        for sn, ps in strategies_list:
            f.writelines(f'# {sn} --> {ps}')

    init()
    for s, _ in strategies_list:
        # 数据库获取数据
        strategy_code = StrategyCode.objects(class_name=s).first()

        binary_file = strategy_code.data

        # 文件是否存在
        f_path = rf'{s_path}/{s}.py'
        if not os.path.exists(f_path):
            with open(file=f_path, mode='wb') as f:
                f.write(binary_file)


    cta_engine = CtaEngine(None, EventEngine())
    cta_engine.load_strategy_class()

    for sn, ps in strategies_list:
        cta_engine.add_strategy(sn, f"{sn}_TEST", vt_symbol, ps)
        print(f'添加：{sn}=={ps}')

def init():
    from vnpy.trader.setting import get_settings
    db_settting = get_settings('database.')
    register_connection(
        'VNPY_BACKTEST',
        db='VNPY_BACKTEST',
        host=db_settting['host'],
        port=db_settting['port'],
        username=db_settting['user'],
        password=db_settting['password'],
        authentication_source=db_settting['authentication_source'],
    )

def recursive_rm(path):
    for fp in os.listdir(path):
        subpath = os.path.join(path, fp)
        if os.path.isdir(subpath):
            recursive_rm(subpath)
        else:
            os.remove(subpath)
            print(f'删除{subpath}')
    else:
        os.rmdir(path)
        print(f'文件夹{path}已删除')


if __name__ == '__main__':
    import pickle ,time

    # ["setting": {'ma_len': canshu0[0], 'bd': canshu0[1], "dk_len": canshu0[2],
    #             'acc': canshu0[3], 'zhouqi': zq}]


    #本次策略来源：分类：1(2)，5(2)，6(1),策略地址：F:\new_0811\huice_log\res_data\tx\贪心集中zuhe_911_01_训练结果_09-15二级预测结果2.pickle
    moni_list = [
        ('ma_tp_03',{'ma_len': 90, 'bd': 1, "dk_len": 70,'acc': 0, 'zhouqi': 1}),
        ('kama_tp_03', {'ma_len': 100, 'bd': 1, "dk_len": 40, 'acc': 0, 'zhouqi': 1}),
        ('kama_tp_03', {'ma_len': 40, 'bd': 3, "dk_len": 70, 'acc': 2, 'zhouqi': 1}),
        ('kama_tp_03', {'ma_len': 60, 'bd': 3, "dk_len": 30, 'acc': 0, 'zhouqi': 1}),
        ('ema_tp_03', {'ma_len': 100, 'bd':2, "dk_len": 30, 'acc': 4, 'zhouqi': 1}),

    ]
    with open(os.getcwd()+r'\moni_celue.pickle' ,'wb') as fp:
        pickle.dump(moni_list,fp)
        time.sleep(3)




    vt_symbol = input("请输入合约名称(HSI-20200929-HKD-FUT.HKFE)：")

    sl_file_name = os.getcwd()+r'\moni_celue.pickle'
    with open(sl_file_name, 'rb') as f:
        sl = pickle.load(f)
        print('pickle,存在！')

    if not isinstance(sl, Iterable):
        raise TypeError("pickle对象不可迭代，请检查pickle对象格式")

    create_strategies(sl, vt_symbol)