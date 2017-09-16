import os
import datetime
import pandas as pd
from pandas import DataFrame
from util.COptHolding import COptHolding


def extract_pandl(holding_file):
    """
    从给定的期权持仓文件提取P&L值
    :param holding_file:
    :return:
    """
    str_date = os.path.basename(holding_file).split(sep='.')[0][-8:]
    holding_date = datetime.datetime.strptime(str_date, '%Y%m%d')
    holding = COptHolding()
    holding.load_holdings(holding_file)
    return holding_date, holding.pandl, holding.nav


def extract_pandls(pandl_dir):
    """
    从指定期权策略holding文件夹中提取P&L时间序列数据，并保存在holding文件夹下
    :param pandl_dir: 持仓文件所在的文件夹
    :return:
    """
    pandl_filename = os.path.join(pandl_dir, 'pandls.csv')
    with open(pandl_filename, 'wt') as f:
        f.write('date,pandl,nav\n')
        for dirname in os.listdir(pandl_dir):
            file_path = os.path.join(pandl_dir, dirname)
            if os.path.isfile(file_path) and os.path.basename(file_path)[:7] == 'holding':
                holding_date, fpandl, fnav = extract_pandl(file_path)
                f.write('%s,%0.2f,%0.2f\n' % (holding_date.strftime('%Y-%m-%d'), fpandl, fnav))

if __name__ == '__main__':
    # extract_pandls('../opt_holdings/vol_trend_strategy')
    deviations = ['0.005', '0.010', '0.015', '0.020', '0.025', '0.030', '0.035', '0.040']
    dict_pandls = {}        # pandl字典
    dict_drawdown = {}      # 最大回撤字典
    for deviation in deviations:
        dict_pandls[deviation] = {}
        dict_drawdown[deviation] = {}
    for days in range(40, 61):
        for deviation in deviations:
            dir_of_pandl = os.path.join('../opt_holdings', 'vol_trend_%d_%s' % (days, deviation))
            print('extract pandl on param: days = %d, deviation = %s' % (days, deviation))
            extract_pandls(dir_of_pandl)
            filename_of_pandl = os.path.join(dir_of_pandl, 'pandls.csv')
            df_pandl = pd.read_csv(filename_of_pandl, index_col=0, parse_dates=[0])
            # df_pandl.columns = ['pandl', 'nav']
            # 提取区间pandl
            dict_pandls[deviation][days] = df_pandl.ix['2017-08-24', 'pandl']/1000000.0
            # 计算区间的最大回撤
            max_nav = 0.0
            min_nav = 0.0
            max_drawdown = 0.0
            for trading_day, pandl_data in df_pandl.iterrows():
                fnav = pandl_data['nav']
                if fnav > max_nav:
                    max_nav = fnav
                    min_nav = fnav
                elif fnav < min_nav:
                    min_nav = fnav
                drawdown = min_nav / max_nav - 1.0
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
            dict_drawdown[deviation][days] = max_drawdown
    df_pandls = DataFrame(dict_pandls)
    df_drawdown = DataFrame(dict_drawdown)
    df_pandls.index.name = 'days'
    df_pandls.to_csv('./pandl_dist.csv')
    df_drawdown.index.name = 'days'
    df_drawdown.to_csv('./maxdrawdown_dist.csv')
