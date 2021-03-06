from mongoengine import DateTimeField, StringField, Document, BinaryField, FloatField, BooleanField, DictField, register_connection
import pandas as pd
import pickle
import os
import time
import datetime
pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)

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

class StrategiesSet(Document):
    hash = StringField(max_length=64)
    class_name = StringField()
    vt_symbol = StringField()
    interval = StringField()
    start = DateTimeField()
    end = DateTimeField()
    rate = FloatField()
    slippage = FloatField()
    size = FloatField()
    pricetick = FloatField()
    capital = FloatField()
    inverse = BooleanField()
    setting = DictField()
    result = DictField()

    meta = {
        'db_alias': 'VNPY_BACKTEST',
        "indexes": [
            {
                "name": "ss_unique",
                "fields": ("class_name", "vt_symbol", "interval", "start", "end", "rate", "slippage", "size", "pricetick", "capital", "inverse", "setting"),
                "unique": True,
            },
            {
                "fields": ("class_name",),
            },
        ],
    }

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

# DB_data??????all_to_local
def DBdata_to_dfs(res_path, moxing=['04']):
    '''
    1.?????????????????????????????????????????????
    2.??????moxing????????????db??????
    3.?????????????????????????????????
    '''

    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    result_pickle = {}

    # with open(res_path, 'rb') as f0:
    #     local_res = pickle.load(f0)
    #     print('??????keys???', local_res.keys())
    try:
        for a0 in all0:
            res = StrategiesSet.objects(class_name=a0, result__ne={})
            all = StrategiesSet.objects(class_name=a0)
            print(f'?????????{a0}??? ????????????{res.count()}??? ?????????{all.count()}, ????????????{float(res.count()) * 100 / float(all.count())}%')
            # continue
            # ???????????????????????????
            if res.count() == all.count():
                if a0.split('_')[-1] not in moxing: continue
                print(a0)
                # ????????????
                # if a0 in [l.strip('_' + l.split('_')[-1]) for l in local_res.keys()]:
                #     print(f'?????????{a0},??????????????????')
                #     continue

                all = StrategiesSet.objects(class_name=a0)
                result_dict = {}
                i = 0
                cols = []

                for index_, s0 in enumerate(all):
                    result_dict0 = {}
                    result_dict0['celue_name'] = s0.class_name + str('_') + str(s0.setting['zhouqi'])
                    result_dict0['s_time'] = s0.start
                    result_dict0['para1'] = s0.setting['ma_len']
                    result_dict0['para2'] = s0.setting['bd']
                    result_dict0['para3'] = s0.setting['dk_len']
                    result_dict0['para4'] = s0.setting['acc']
                    result_dict0['canshu'] = [s0.setting['ma_len'], s0.setting['bd'], s0.setting['dk_len'],
                                              s0.setting['acc']]
                    # ????????????
                    if s0.result['win_rate'] != None:
                        result_dict0['win_rate'] = s0.result['win_rate']
                        result_dict0['win_mean_num'] = s0.result['win_mean_num']
                        result_dict0['loss_mean_num'] = s0.result['loss_mean_num']
                        result_dict0['max_num'] = s0.result['max_num']
                        result_dict0['min_num'] = s0.result['min_num']
                        result_dict0['mean_num'] = s0.result['mean_num']
                        result_dict0['std_num'] = s0.result['std_num']
                        result_dict0['max_sum'] = s0.result['max_sum']
                        result_dict0['end'] = s0.result['total_net_pnl']
                        result_dict0['max_back'] = s0.result['max_drawdown']
                        result_dict0['trade_nums'] = s0.result['total_trade_count']
                        result_dict0['sharp_rate'] = s0.result['sharpe_ratio']
                        result_dict0['total_days'] = s0.result['total_days']
                        result_dict0['profit_days'] = s0.result['profit_days']
                        result_dict0['total_slippage'] = s0.result['total_slippage']
                        result_dict0['win_rate'] = s0.result['win_rate']
                        result_dict0['win_mean_num'] = s0.result['win_mean_num']
                        result_dict0['win_mean_num'] = s0.result['loss_mean_num']
                        result_dict0['max_num'] = s0.result['max_num']
                        result_dict0['min_num'] = s0.result['min_num']
                        result_dict0['mean_num'] = s0.result['mean_num']
                        result_dict0['std_num'] = s0.result['std_num']
                        cols = result_dict0.keys()

                    result_dict[index_] = result_dict0

                df_res0 = pd.DataFrame(result_dict).T
                # print(df_res0)
                df_res0 = df_res0[list(cols)]
                for name in [c for c in list(set(df_res0['celue_name'].tolist()))]:
                    result_pickle[str(name)] = df_res0.loc[df_res0['celue_name'] == name, :].copy()
                    print(f'{name}???????????????')
                # result_pickle[str(a0)] = df_res0
            else:
                print(f'{a0}??????????????????')
        if len(result_pickle.keys()) == 0:
            print('????????????????????? ')
            exit()
        print('?????????key???', result_pickle.keys())
        res_in = input('???????????????y/n')
        if res_in == 'y':
            time0 = time.strftime("%m-%d", time.localtime())
            respath = res_path.split('.')[0] + moxing[0]+'_'+time0 + '.pickle'
            with open(respath,'wb') as f:
                pickle.dump(result_pickle, f)
                print('dict ????????????keys???', result_pickle.keys())
                print(f'????????????????????????{respath}')
        else:
            print('???????????????')
    except Exception as e:
        print(e)

    return result_pickle

def show_DB_data():
    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    done = []
    undone = []
    for a0 in all0:
        res = StrategiesSet.objects(class_name=a0, result__ne={})
        all = StrategiesSet.objects(class_name=a0)
        print(f'?????????{a0}??? ????????????{res.count()}??? ?????????{all.count()}, ????????????{float(res.count())*100/float(all.count())}%' )
        if float(res.count())/float(all.count())==1:
            done.append(a0)
        else:
            undone.append(a0)

    print(f'\n????????????{done}\n????????????{undone}')
    return [all0]

def show_local_data(local_data_path,show =False):

    if os.path.exists(local_data_path):
        print('????????????????????????')
        with open(local_data_path,'rb') as f:
            res_pic = pickle.load(f)
            print(type(res_pic))
            print(f'{local_data_path}\n?????????key?????????????????????',res_pic.keys())
            if show == True:
                for k,v in res_pic.items():
                    print(k)
                    print('\n?????????key?????????',v.keys())
                    # print('\n?????????key??????',v[].keys())
                    print(sorted(set((v['s_time'].tolist()))))
                    # for k ,da in v['data_randomforest_classify_grid'].groupby('???????????????'):
                    #     print(k)
                    #     print(da.shape)
                    #     print(da[['end_3','max_back_3','??????_????????????','?????????_????????????']].mean())
                    #     print(da.sample(2))
                    # print((sorted(set(v['s_time'].to_list()))))
                    # exit()
                    # print(v['2020-09-01']['????????????????????????'].tail(10))
                    # print(v['2020-08-01']['????????????????????????'].loc[v['2020-08-01']['????????????????????????'] !=0].tail(10))
                    exit()
                    v.sort_values(by='s_time',inplace=True)
                    print(v.head(1))
                    print(v.tail(1))

                    print(k,v.shape)
                    print('\n===========')
        return res_pic
    else:
        print('???????????????????????????????????????')
        return None

def save_to_local(to_pickle_data,to_pickle_flag):

    path0 = os.getcwd()+'\\'
    for i in list(set([k.split('_')[-2] for k in to_pickle_data.keys()])):

        cl_to_pickle_data = {}

        for k in to_pickle_data.keys():
            if k.split('_')[-2] == i:
                cl_to_pickle_data[k] = to_pickle_data[k]


        to_path = path0 + str(to_pickle_flag) + f'_{i}.pickle'
        if os.path.exists(to_path):
            to_path = path0 + str(to_pickle_flag) + f'_{i}_new.pickle'
            print(f'????????????{to_path}????????????!')
            new = input('????????????????????????y/n???')
            if new == 'y':
                with open(to_path, 'wb') as f:
                    pickle.dump(cl_to_pickle_data, f)
                    print(f'{cl_to_pickle_data.keys()}\n?????????????????????{to_path}')
            # print(cl_to_pickle_data.keys())
            continue
        else:
            to_path = path0 + str(to_pickle_flag) + f'_{i}.pickle'
            # print(to_path,'\n', cl_to_pickle_data.keys())
            # continue

            with open(to_path, 'wb') as f:
                pickle.dump(cl_to_pickle_data, f)
                print(f'{cl_to_pickle_data.keys()}\n?????????????????????{to_path}')


    return


# 1===?????????????????????????????????????????????????????????(????????????????????????
def update_newDBdata_to_local(local_res_path, moxing=['04'], new_time=datetime.datetime(2020, 7, 1)):
    time0 = time.strftime("%m-%d", time.localtime())
    res_path = local_res_path + '_' + time0 + '.pickle'

    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    result_pickle = {}
    print(all0)
    # ?????????????????????
    for a0 in all0:
        res = StrategiesSet.objects(class_name=a0, result__ne={})
        all = StrategiesSet.objects(class_name=a0)
        # print(f'?????????{a0}??? ????????????{res.count()}??? ?????????{all.count()}, ????????????{float(res.count()) * 100 / float(all.count())}%')
        # ???????????????????????????
        if res.count() == all.count():

            if a0.split('_')[-1] not in moxing: continue
            print(a0)
            # ??????newtime????????????==?????????????????????
            all = StrategiesSet.objects(class_name=a0, end__gt=new_time)
            if not (len(all)):continue

            result_dict = {}
            cols = []
            for index_, s0 in enumerate(all):
                result_dict0 = {}
                result_dict0['celue_name'] = s0.class_name + str('_') + str(s0.setting['zhouqi'])
                result_dict0['s_time'] = s0.start
                result_dict0['para1'] = s0.setting['ma_len']
                result_dict0['para2'] = s0.setting['bd']
                result_dict0['para3'] = s0.setting['dk_len']
                result_dict0['para4'] = s0.setting['acc']
                result_dict0['canshu'] = [s0.setting['ma_len'], s0.setting['bd'], s0.setting['dk_len'],
                                          s0.setting['acc']]
                # ????????????
                if s0.result['win_rate'] != None:
                    result_dict0['win_rate'] = s0.result['win_rate']
                    result_dict0['win_mean_num'] = s0.result['win_mean_num']
                    result_dict0['loss_mean_num'] = s0.result['loss_mean_num']
                    result_dict0['max_num'] = s0.result['max_num']
                    result_dict0['min_num'] = s0.result['min_num']
                    result_dict0['mean_num'] = s0.result['mean_num']
                    result_dict0['std_num'] = s0.result['std_num']
                    result_dict0['max_sum'] = s0.result['max_sum']
                    result_dict0['end'] = s0.result['total_net_pnl']
                    result_dict0['max_back'] = s0.result['max_drawdown']
                    result_dict0['trade_nums'] = s0.result['total_trade_count']
                    result_dict0['sharp_rate'] = s0.result['sharpe_ratio']
                    result_dict0['total_days'] = s0.result['total_days']
                    result_dict0['profit_days'] = s0.result['profit_days']
                    result_dict0['total_slippage'] = s0.result['total_slippage']
                    result_dict0['win_rate'] = s0.result['win_rate']
                    result_dict0['win_mean_num'] = s0.result['win_mean_num']
                    result_dict0['win_mean_num'] = s0.result['loss_mean_num']
                    result_dict0['max_num'] = s0.result['max_num']
                    result_dict0['min_num'] = s0.result['min_num']
                    result_dict0['mean_num'] = s0.result['mean_num']
                    result_dict0['std_num'] = s0.result['std_num']
                    cols = result_dict0.keys()

                result_dict[index_] = result_dict0

            df_res0 = pd.DataFrame(result_dict).T
            df_res0 = df_res0[list(cols)]
            print(df_res0.tail())
            # exit()
            for name in [c for c in list(set(df_res0['celue_name'].tolist()))]:
                todata = df_res0.loc[df_res0['celue_name'] == name].copy()
                todata.sort_values(by='s_time', ascending=True, inplace=True)
                result_pickle[str(name)] = todata
                print(f'{name}???????????????')
            # result_pickle[str(a0)] = df_res0
        else:
            print(f'{a0}??????????????????')

    if len(result_pickle.keys()) == 0:
        print('????????????????????? ')
        exit()

    print('?????????key???', result_pickle.keys())
    res_in = input(f'???????????????y/n:\n?????????{res_path}')
    if res_in == 'y':
        with open(res_path, 'wb') as f:
            pickle.dump(result_pickle, f)
            print('dict ????????????keys???', result_pickle.keys())
            print(f'????????????????????????{res_path}')

    else:
        print('???????????????')



    return result_pickle
# 2===????????????????????????????????????(????????????????????????
def update_newdata_to_dfs(new_data, newpath0,local_res_path):
    '''
    new_data?????????{????????????????????????}
    :param new_data:
    :param local_res_path:
    :return:
    '''

    local_res_path0 = newpath0

    # ?????????????????????????????????==??????????????????
    with open(local_res_path, 'rb') as f0:
        local_res = pickle.load(f0)
        print('??????keys???', local_res.keys())
    # ??????????????????????????????????????????
    for k, v in local_res.items():
        # ??????????????????????????? ??????????????????????????????
        if k in new_data.keys():
            if new_data[k].iloc[0]['s_time'] > v.iloc[-1]['s_time']:
                print(f'{k},?????????????????????')
                v = v.append(new_data[k])
                local_res[k] = v
            else:
                print(f'{k}?????????????????????????????????????????????')
                continue

    time0 = time.strftime("%m-%d", time.localtime())
    path0 = local_res_path0 +"_"+ time0 +'_allnew.pickle'
    with open(path0, 'wb') as f:
        pickle.dump(local_res, f)
        print('dict ????????????keys???', local_res.keys())
        print(f'????????????????????????{path0}')

    return local_res_path0 + time0

# ???????????????????????????
def updata_to_local(local_path,updatas_names=[],new_time=''):
    '''????????????????????????????????????:
    1.?????????????????????????????????????????????
    2.??????????????????????????????
    updatas = {clm,:data=df,
                clm,:data=df,
                    }
    '''
    time0 = time.strftime("%m-%d", time.localtime())
    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    result_pickle = {}
    for a0 in updatas_names:
        res = StrategiesSet.objects(class_name=a0, result__ne={})
        all = StrategiesSet.objects(class_name=a0)
        print(f'?????????{a0}??? ????????????{res.count()}??? ?????????{all.count()}, ????????????{float(res.count()) * 100 / float(all.count())}%')
        # ???????????????????????????
        if res.count() == all.count():
            print(a0,'???????????????')
            # ??????newtime????????????==?????????????????????
            all = StrategiesSet.objects(class_name=a0, end__gt=new_time)
            result_dict = {}
            cols = []
            for index_, s0 in enumerate(all):
                result_dict0 = {}
                result_dict0['celue_name'] = s0.class_name + str('_') + str(s0.setting['zhouqi'])
                result_dict0['s_time'] = s0.start
                result_dict0['para1'] = s0.setting['ma_len']
                result_dict0['para2'] = s0.setting['bd']
                result_dict0['para3'] = s0.setting['dk_len']
                result_dict0['para4'] = s0.setting['acc']
                result_dict0['canshu'] = [s0.setting['ma_len'], s0.setting['bd'], s0.setting['dk_len'],
                                          s0.setting['acc']]
                # ????????????
                if s0.result['win_rate'] != None:
                    result_dict0['win_rate'] = s0.result['win_rate']
                    result_dict0['win_mean_num'] = s0.result['win_mean_num']
                    result_dict0['loss_mean_num'] = s0.result['loss_mean_num']
                    result_dict0['max_num'] = s0.result['max_num']
                    result_dict0['min_num'] = s0.result['min_num']
                    result_dict0['mean_num'] = s0.result['mean_num']
                    result_dict0['std_num'] = s0.result['std_num']
                    result_dict0['max_sum'] = s0.result['max_sum']
                    result_dict0['end'] = s0.result['total_net_pnl']
                    result_dict0['max_back'] = s0.result['max_drawdown']
                    result_dict0['trade_nums'] = s0.result['total_trade_count']
                    result_dict0['sharp_rate'] = s0.result['sharpe_ratio']
                    result_dict0['total_days'] = s0.result['total_days']
                    result_dict0['profit_days'] = s0.result['profit_days']
                    result_dict0['total_slippage'] = s0.result['total_slippage']
                    result_dict0['win_rate'] = s0.result['win_rate']
                    result_dict0['win_mean_num'] = s0.result['win_mean_num']
                    result_dict0['win_mean_num'] = s0.result['loss_mean_num']
                    result_dict0['max_num'] = s0.result['max_num']
                    result_dict0['min_num'] = s0.result['min_num']
                    result_dict0['mean_num'] = s0.result['mean_num']
                    result_dict0['std_num'] = s0.result['std_num']
                    cols = result_dict0.keys()

                result_dict[index_] = result_dict0
            df_res0 = pd.DataFrame(result_dict).T
            # ???????????????df??????
            df_res0 = df_res0[list(cols)]
            # print(df_res0.tail())
            # exit()
            # ????????????????????????????????????????????????
            for name in [c for c in list(set(df_res0['celue_name'].tolist()))]:
                todata = df_res0.loc[df_res0['celue_name'] == name].copy()
                todata.sort_values(by='s_time', ascending=True, inplace=True)
                result_pickle[str(name)] = todata
                print(f'{name}???????????????')
                # result_pickle[str(a0)] = df_res0
        else:
            print(f'{a0}????????????????????????')


    if len(result_pickle.keys()) == 0:
        print('???????????????????????????????????? ')
        exit()

    print('????????????????????????????????????\n', result_pickle.keys())

    with open(local_path,'rb') as lf:
        local_df = pickle.load(lf)

    for clm in result_pickle.keys():
        if clm in local_df.keys():
            new_data  = pd.DataFrame(local_df[clm]).append(result_pickle[clm],ignore_index=True)
            new_data.drop_duplicates(subset=['s_time','canshu'],keep='first',inplace=True)
            local_df[clm] = new_data
        else:
            local_df[clm] = result_pickle[clm]
    with open(local_path,'wb') as tof:
        pickle.dump(local_df,tof)






if __name__ == '__main__':
    '''
    ????????????????????????????????????????????????????????????
    ????????????????????????????????????????????????
    '''




    #??????DBdata
    if 0==1:
        show_DB_data()

    # ??????:1.????????????????????????????????????????????????
    if 0 ==1:
        for  i  in ["02","03","01","04"]:
            newpath0 = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to%s'%i
            res_pickle_path0 = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to%s10-13_allnew.pickle'%i

            newdata = update_newDBdata_to_local(local_res_path=newpath0, moxing=[i], new_time = datetime.datetime(2020, 9, 30))
            update_newdata_to_dfs(new_data=newdata, newpath0=newpath0,local_res_path=res_pickle_path0)

    #??????????????????????????????
    if 0 == 1 :pass

        # # ??????DB????????????df????????????res??????????????????????????????
        # res_path = os.getcwd() + r'\res_df_in_dict_to.pickle'
        # to_pickle_data = DBdata_to_dfs(res_path,moxing=['04'])
    #?????????????????????
    if 1== 1:
        for i in range(1,5):
            i = str(i)
            res_pickle_path = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to0%s_11-11_allnew.pickle'%i
            with open(res_pickle_path,'rb') as f:
                to_pickle_data = pickle.load(f)
                for k in [i for i in list(to_pickle_data.keys()) if 'T3' in i ]:

                        print('??????T3???')
                        to_pickle_data.pop(k)

                print(to_pickle_data.keys())
            with open(res_pickle_path,'wb') as f:
                pickle.dump(to_pickle_data,f)

            # print(sorted(set(to_pickle_data['dema_tp_02_1']['s_time'].tolist())))
        # save_to_local(to_pickle_data, to_pickle_flag='MEBD????????????')
        # show_local_data(res_pickle_path0,show=True)









