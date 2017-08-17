from util.util import COptTradeData
from util.COptHolding import COption
from util.COptHolding import COptHolding
from configparser import ConfigParser
import pandas as pd
import datetime
import os
from enum import Enum, auto


class MktStatus(Enum):
    """市场状态类，市场状态分为：牛市、熊市、震荡市"""
    Bullish = auto()
    Bearish = auto()
    Volatile = auto()


def mean_average(index_code, end_date, days, threshold):
    """
    均线系统，返回指定指数的市场状态(MktStatus）
    :param index_code: 指数的代码
    :param end_date: 截止日期,类型=datetime.date
    :param days: 均线系统采用的交易日数量
    :param threshold: 偏离度阈值
    :return: 指定指数的市场状态，MktStatus
    """
    file_path = './data/' + index_code + '.csv'
    k_data = pd.read_csv(file_path, parse_dates=[0])
    # idx = k_data[k_data.date == end_date].index.values[0]
    fma = k_data[k_data.date <= end_date].tail(days).mean().close
    flast = k_data[k_data.date == end_date].iloc[0].close
    if flast > fma * (1.0+threshold):
        status = MktStatus.Bullish
    elif flast < fma * (1.0+threshold):
        status = MktStatus.Bearish
    else:
        status = MktStatus.Volatile
    return status


class CVolTrendTradingStrategy(object):
    """结合趋势判断的波动率交易（比率价差））"""
    def __init__(self, portname, configname):
        """
        策略初始化
        :param portname: 组合名称
        :param configname:  采用的配置项名称
        """
        self.portname = portname
        self.configname = configname
        self.marginratio = 0.0          # 保证金占比
        self.ma_days = 0                # 均线系统所采用的天数
        self.ma_deviation = 0.0         # 偏离均线的阈值
        self.spread_ratio = None        # 比率价差中买入平值期权和卖出虚值期权的数量之比，如‘1:2’

        self.opts_data = {}                 # 期权基础数据（样本期权），字典类型，map<optcode,COption>
        self.underlying_quote_1min = None   # 期权标的1分钟行情数据，DataFrame数据
        self.commission_per_unit = 0.0      # 每张期权交易佣金
        self.opt_holdings_path = None       # 持仓数据文件夹路径

        # self.opt_holdings = COptHolding(self.opt_holdings_path + self.configname + 'log.txt')   # 策略的期权持仓类
        # 策略的期权持仓类
        self.opt_holdings = COptHolding(os.path.join(self.opt_holdings_path, self.configname, 'log.txt'))
        # 策略的认购比率和认沽比率价差中所含期权代码tuple(code1,code2)，其中第一个为平值期权代码，第二个为虚值期权代码
        self.call_ratio = None  # 认购比率空仓时默认为None
        self.put_ratio = None   # 认沽比率空仓时默认为None

        # 导入相关参数
        self.load_param()

    def load_param(self):
        """导入策略的参数值"""
        cfg = ConfigParser()
        cfg.read('config.ini')
        self.marginratio = cfg.getfloat(self.configname, 'marginratio')
        self.ma_days = cfg.getint(self.configname, 'ma_days')
        self.ma_deviation = cfg.getfloat(self.configname, 'ma_deviation')
        self.spread_ratio = cfg.get(self.configname, 'spread_ratio')
        self.opt_holdings_path = cfg.get('path', 'opt_holdings_path')

    def load_opt_basic_data(self, trading_day):
        """
        导入期权基本信息数据
        导入当月期权合约信息，如果trading_day为当月期权合约的最后交易日，那么导入次月合约
        :param trading_day: 日期（类型=datetime.date）
        :return: 如果导入成功=True，如果导入失败=False
        """
        self.opts_data = {}
        header_name = {'opt_code', 'trade_code', 'opt_name', 'underlying_code', 'secu_type', 'opt_type',
                       'exercise_type', 'strike', 'multiplier', 'end_month', 'listed_date', 'expire_date',
                       'exercise_date', 'delivery_date'}
        opts_basics = pd.read_csv('./data/OptBasics.csv', usecols=range(14), parse_dates=[10, 11, 12, 13],
                                  dtype={'期权代码': str})
        opts_basics.columns = header_name
        opts_basics.set_index(keys='opt_code', inplace=True)
        opts_basics = opts_basics[(opts_basics.expire_date > trading_day) & (opts_basics.listed_date <= trading_day) &
                                  (opts_basics.multiplier == 10000)]
        if len(opts_basics) == 0:
            return False
        # 选择当月合约，即当前交易合约中到期日最小的合约
        opts_basics = opts_basics[opts_basics.expire_date == min(opts_basics.expire_date)]
        for optcode, optdata in opts_basics.iterrows():
            if optcode not in self.opts_data:
                if optdata['opt_type'] == '认购':
                    opt_type = 'Call'
                else:
                    opt_type = 'Put'
                if optdata['exercise_type'] == '欧式':
                    exercise_type = 'European'
                else:
                    exercise_type = 'American'
                enddate = optdata['expire_date'].to_pydatetime().date()
                self.opts_data[optcode] = COption(optcode, optdata['opt_name'], opt_type, exercise_type,
                                                  float(optdata['strike']), int(optdata['multiplier']), enddate)
        # 如果当前日期为当月合约最后交易日的前一天，那么把持仓状态改为'transform’
        if opts_basics.iloc[0, 10] - datetime.timedelta(days=1) == trading_day:
            self.opt_holdings.status = 'transform'
        return True

    def load_opt_holdings(self, trading_day):
        """
        导入策略的期权持仓数据
        :param trading_day: 持仓日期（类型=datetime.date）
        :return:
        """
        strdate = trading_day.strftime('%Y%m%d')
        holding_filename = self.opt_holdings_path + self.configname + '/holding_' + self.portname + '_' + strdate + '.txt'
        self.opt_holdings.load_holdings(holding_filename)

    def load_opt_1min_quote(self, trading_day):
        """
        导入指定日期期权（含持仓期权）的1分钟行情数据
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        strdate = trading_day.strftime('%Y-%m-%d')
        for optcode, opt in self.opts_data.items():
            strfilepath = '../opt_quote/' + strdate + '/' + optcode + '.csv'
            opt.quote_1min = pd.read_csv(strfilepath, usecols=range(7), index_col=0, parse_dates=[0])
        for optcode, holding in self.opt_holdings.holdings.items():
            strfilepath = '../opt_quote/' + strdate + '/' + optcode + '.csv'
            holding.COption.quote_1min = pd.read_csv(strfilepath, usecols=range(7), index_col=0, parse_dates=[0])

    def load_underlying_1min_quote(self, trading_day):
        """
        导入期权标的1分钟行情数据
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        strdate = trading_day.strftime('%Y-%m-%d')
        strfilepath = '../opt_quote/' + strdate + '/510050ETF.csv'
        self.underlying_quote_1min = pd.read_csv(strfilepath, usecols=range(7), index_col=0, parse_dates=[0])

    def calc_opt_margin(self, trading_day):
        """
        计算样本期权和持仓期权的开仓保证金，每个交易日开盘前计算一次
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        # 1.读取标的日K线时间序列，获取标的的前收盘价
        underlying_quote = pd.read_csv('./data/underlying_daily_quote.csv', index_col=0, parse_dates=[0])
        underlying_pre_close = float(underlying_quote.ix[trading_day, 'pre_close'])
        # 2.读取样本期权的日行情
        strdate = trading_day.strftime('%Y-%m-%d')
        strfilepath = '../opt_quote/' + strdate + '/50OptionDailyQuote.csv'
        opts_quote = pd.read_csv(strfilepath, usecols=range(1, 14), parse_dates=[0], encoding='gb18030',
                                 dtype={'option_code': str})
        opts_quote.set_index(keys='option_code', inplace=True)
        # 3.计算样本期权的开仓保证金
        for optcode, opt in self.opts_data.items():
            if optcode in opts_quote.index:
                opt_pre_settle = float(opts_quote.ix[optcode, 'pre_settle'])
                opt.calc_margin(opt_pre_settle, underlying_pre_close)
            else:
                opt.margin = 3000.0
        # 4.计算持仓期权的开仓保证金
        self.opt_holdings.calc_margin(trading_day)

    def get_atm_strike(self, underlying_price):
        """
        取得平价的行权价
        :param underlying_price:  标的最新价格
        :return:
        """
        atm_strike = -1
        for _, copt in self.opts_data.items():
            if atm_strike == -1:
                atm_strike = copt.strike
            elif abs(copt.strike - underlying_price) < abs(atm_strike - underlying_price):
                atm_strike = copt.strike
        return atm_strike

    def get_ratiospread_opts(self, underlying_price, opt_type):
        """
        取得比率价差的平值、虚值期权（虚值一档）
        :param underlying_price: 标的最新价格
        :param opt_type: 期权类型，'Call' or 'Put'
        :return: tuple(COption, COption)，分别为平值、虚值期权类，期权类型由opt_type指定
        """
        atm_strike = self.get_atm_strike(underlying_price)
        if atm_strike == -1:
            return None, None
        else:
            atm_opt = otm_opt = None    # atm_opt=平值期权，otm_opt=虚值期权
            for _, copt in self.opts_data.items():
                if copt.opt_type == opt_type:
                    if copt.strike == atm_strike:
                        atm_opt = copt
                    if opt_type == 'Call':
                        if (otm_opt is None) & (copt.strike > atm_strike):
                            otm_opt = copt
                        elif (otm_opt is not None) & (atm_strike < copt.strike < otm_opt.strike):
                            otm_opt = copt
                    elif opt_type == 'Put':
                        if (otm_opt is None) & (copt.strike < atm_strike):
                            otm_opt = copt
                        elif (otm_opt is not None) & (otm_opt.strike < copt.strike < atm_strike):
                            otm_opt = copt
            return atm_opt, otm_opt

    def get_spread_ratio(self):
        """
        取得价差比率
        :return: tuple(平值期权数量,虚值期权数量），如(1,2)
        """
        atm_ratio = otm_ratio = None
        atm_ratio, otm_ratio = tuple(int(n) for n in self.spread_ratio.split(':'))
        return atm_ratio, otm_ratio

    def do_position(self, trading_beg_datetime, trading_min_num, opt_type, capital):
        """
        进行建仓操作
        :param trading_beg_datetime: 建仓开始时的交易时间
        :param trading_min_num: 建仓所需的分钟数，整型
        :param opt_type: 期权类型，Call=认购比率，Put=认沽比率
        :param capital: 建仓所采用的资本金（多头使用的金额+空头占用的保证金）
        :return: 建仓完成时的交易时间，=最后一笔建仓时间的后一分钟
        """
        # 1.取得平值期权、虚值一档的期权，并设置比率价差的代码
        funderlying_price = self.underlying_quote_1min.ix[trading_beg_datetime, 'close']
        atm_opt, otm_opt = self.get_ratiospread_opts(funderlying_price, opt_type)
        if opt_type == 'Call':
            self.call_ratio = (atm_opt.code, otm_opt.code)
        else:
            self.put_ratio = (atm_opt.code, otm_opt.code)

        # 2.取得平值期权、虚值期权的单位数量
        atm_ratio, otm_ratio = self.get_spread_ratio()
        # 3.计算平值期权、虚值期权的建仓数量
        funit_cost = atm_opt.quote_1min.ix[trading_beg_datetime, 'close'] * atm_opt.multiplier * atm_ratio + otm_opt.margin * otm_ratio
        num_of_units = int(round(capital / trading_min_num / funit_cost, 0))
        atm_vol, otm_vol = (atm_ratio * num_of_units, otm_ratio * num_of_units)
        # 4.在接下来的trading_min_num分钟里建仓，每分钟平值、虚值的建仓量分别为atm_vol和otm_vol
        trade_datas = []
        for min_num in range(trading_min_num):
            trading_datetime = trading_beg_datetime + datetime.timedelta(minutes=min_num)
            atm_price = atm_opt.quote_1min.ix[trading_datetime, 'close']
            trade_datas.append(COptTradeData(atm_opt.code, 'buy', 'open', atm_price, atm_vol,
                                             atm_price * atm_vol * atm_opt.multiplier, atm_vol * self.commission_per_unit,
                                             trading_datetime, atm_opt))
            otm_price = otm_opt.quote_1min.ix[trading_datetime, 'close']
            trade_datas.append(COptTradeData(otm_opt.code, 'sell', 'open', otm_price, otm_vol,
                                             otm_price * otm_vol * otm_opt.multiplier, 0.0, trading_datetime, otm_opt))
        # 5.更新持仓数据
        self.opt_holdings.update_holdings(trade_datas)
        # 6.返回建仓完成时的交易时间，=最后一笔建仓时间的后一分钟
        return trading_beg_datetime + datetime.timedelta(minutes=trading_min_num)

    def do_liquidation(self, trading_beg_datetime, trading_min_num, opt_type):
        """
        进行平仓操作
        :param trading_beg_datetime: 平仓开始时的交易时间
        :param trading_min_num: 平仓所需的分钟数，整型
        :param opt_type: 期权类型，Call=认购比率，Put=认沽比率
        :return: 平仓完成时的交易时间，=最后一笔平仓时间的后一分钟
        """
        # 1.取得平值期权、虚值期权的代码
        if opt_type == 'Call':
            atm_code, otm_code = self.call_ratio
        else:
            atm_code, otm_code = self.put_ratio
        # 2.根据平仓的分钟数，计算每分钟平仓的数量
        atm_vol = int(self.opt_holdings.holdings[atm_code].holdingvol / trading_min_num + 0.5)
        otm_vol = int(self.opt_holdings.holdings[otm_code].holdingvol / trading_min_num + 0.5)
        # 3.在接下来的trading_min_num分钟里平仓，每分钟平仓平值、虚值的数量分别为atm_vol和opt_vol
        for min_num in range(trading_min_num):
            trade_datas = []
            trading_datetime = trading_beg_datetime + datetime.timedelta(minutes=min_num)
            atm_price = self.opts_data[atm_code].quote_1min.ix[trading_datetime, 'close']
            otm_price = self.opts_data[otm_code].quote_1min.ix[trading_datetime, 'close']
            # 如果时间超过14:59:00，将剩余的全部平仓
            if trading_datetime.time() >= datetime.time(14, 59, 0):
                atm_vol = self.opt_holdings.holdings[atm_code].holdingvol
                if atm_vol > 0:
                    trade_datas.append(COptTradeData(atm_code, 'sell', 'close', atm_price, atm_vol, atm_price * atm_vol,
                                                     atm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[atm_code]))
                otm_vol = self.opt_holdings.holdings[otm_code].holdingvol
                if otm_vol > 0:
                    trade_datas.append(COptTradeData(otm_code, 'buy', 'close', otm_price, otm_vol, otm_price * otm_vol,
                                                     otm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[otm_code]))
            else:
                if atm_vol > self.opt_holdings.holdings[atm_code].holdingvol:
                    atm_vol = self.opt_holdings.holdings[atm_code].holdingvol
                if otm_vol > self.opt_holdings.holdings[otm_code].holdingvol:
                    otm_vol = self.opt_holdings.holdings[otm_code].holdingvol
                if atm_vol > 0:
                    trade_datas.append(COptTradeData(atm_code, 'sell', 'close', atm_price, atm_vol, atm_price * atm_vol,
                                                     atm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[atm_code]))
                if otm_vol > 0:
                    trade_datas.append(COptTradeData(otm_code, 'buy', 'close', otm_price, otm_vol, otm_price * otm_vol,
                                                     otm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[otm_code]))
            # 更新持仓数据
            self.opt_holdings.update_holdings(trade_datas)
        # 返回平仓完成时的交易时间，=最后一笔平仓时间的后一分钟
        return trading_beg_datetime + datetime.timedelta(minutes=trading_min_num)
