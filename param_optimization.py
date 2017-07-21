from multiprocessing import Pool
import os
import OptVolTrading
import datetime


def do_optimizing(begdate, enddate, optname, configname):
    print('Run task %s(%s)...' % (configname, os.getpid()))
    vol_strategy = OptVolTrading.CVolTradingStrategy(optname, configname)
    vol_strategy.on_vol_trading_interval(begdate, enddate)

if __name__ == '__main__':
    opt_name = 'VolTrade'
    tmbeg_date = datetime.date(2017, 6, 26)
    tmend_date = datetime.date(2017, 6, 27)
    # for k in range(1, 5):
    #     config_name = 'test' + str(k)
    #     pid = os.fork()
    #     if pid == 0:
    #         do_optimizing(tmbeg_date, tmend_date, opt_name, config_name)
    #     else:
    #         print("'%s'进程已启动，pid=%s" % (config_name, pid))
    p = Pool()
    for k in range(1, 5):
        config_name = 'test' + str(k)
        p.apply_async(do_optimizing, args=(tmbeg_date, tmend_date, opt_name, config_name,))
    p.close()
    p.join()
    print('All optimizing has done.')
