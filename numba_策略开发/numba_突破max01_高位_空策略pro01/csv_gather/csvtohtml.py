import pandas as pd
import os

def csv_to_html(path_0,path_1 = None):

    df0 = pd.read_csv(path_0, index_col=0)
    df0.sort_values('最后收益', ascending=False, inplace=True)
    df0 = df0.iloc[:100].copy()
    if not path_1:
        print(df0.tail())
        print('--- 仅仅查看文件，没有html地址！ ---')
    else:
        df0.to_html(path_1)
        print('保存至：',path_1)

    return df0

if __name__ == "__main__":

    name = 'max_突破06_atr_12_02=20_26.csv'
    path_0 = os.getcwd() + r'\%s' % name
    path_1 = r'F:\task\恒生股指期货\numba_突破max01_空策略pro01\策略介绍html\csv_html\%s'%name.strip('csv')+'html'

    df0 = csv_to_html(path_0)
    df0['s_time'] = '2018-1-1'
    df0.loc[df0['最后收益']<df0['最后收益'].mean(),'s_time'] = '2019-1-1'
    df0.set_index(keys =['s_time','最后收益'],inplace =True)
    df0.sort_values(by='最后收益',ascending=False,inplace =True)
    print(df0)

