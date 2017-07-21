import os
import datetime
from util.COptHolding import COptHolding

def extract_pandl(holding_file):
    """
    从给定的期权持仓文件提取P&L值
    :param holding_file:
    :return:
    """
    str_date = os.path.basename(holding_file).split(sep='.')[0][-8:]
    holding_date = datetime.datetime.strptime(str_date, '%Y%m%d')
    Holding = COptHolding()
    Holding.load_holdings(holding_file)
    return holding_date, Holding.pandl


def extract_pandls(pandl_dir, pandl_filename):
    """
    从指定期权策略holding文件夹中提取P&L时间序列数据
    :param pandl_dir:
    :param pandl_filename:
    :return:
    """
    with open(pandl_filename, 'wt') as f:
        for dirname in os.listdir(pandl_dir):
            file_path = os.path.join(pandl_dir, dirname)
            if os.path.isfile(file_path) and file_path[-4:] == '.txt':
                holding_date, fpandl = extract_pandl(file_path)
                f.write('%s,%0.2f\n' % (holding_date.strftime('%Y-%m-%d'), fpandl))

if __name__ == '__main__':
    extract_pandls('../opt_holdings', './pandls.csv')