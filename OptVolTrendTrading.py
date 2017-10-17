from util.util import COptTradeData
from util.COption import COption
from util.COptHolding import COptHolding
from configparser import ConfigParser, RawConfigParser
import pandas as pd
import datetime
import os
from enum import Enum, auto
import time


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
    if flast > fma * (1.0 + threshold):
        status = MktStatus.Bullish
    elif flast < fma * (1.0 - threshold):
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

        self.opts_data = {}                 # 所有未到期期权基础数据（样本期权），字典类型，map<optcode,COption>
        self.trading_opts_data = {}         # 参与交易期权的基础数据，字典类型，map<optcode,COption>
        self.underlying_quote_1min = None   # 期权标的1分钟行情数据，DataFrame数据
        self.commission_per_unit = 0.0      # 每张期权交易佣金
        self.opt_holdings_path = None       # 持仓数据文件夹路径

        # 策略的认购比率和认沽比率价差中所含期权代码tuple(code1,code2)，其中第一个为平值期权代码，第二个为虚值期权代码
        self.call_ratio = None  # 认购比率空仓时默认为None
        self.put_ratio = None   # 认沽比率空仓时默认为None

        # 交易日历
        self.calendar = None
        # 距离期权到期日多少天开次月期权
        self.open_nextmonth_opt_days = None
        # 移仓天数
        self.transform_days = None

        # 是否导入了交易日交易数据
        self.is_loaded_tradingdata = False

        # 导入相关参数
        self.load_param()

        # 策略的期权持仓类
        self.opt_holdings = COptHolding(os.path.join(self.opt_holdings_path, self.configname, 'log.txt'))

    def load_param(self):
        """导入策略的参数值"""
        cfg = ConfigParser()
        cfg.read('config.ini')
        self.marginratio = cfg.getfloat(self.configname, 'marginratio')
        self.ma_days = cfg.getint(self.configname, 'ma_days')
        self.ma_deviation = cfg.getfloat(self.configname, 'ma_deviation')
        self.spread_ratio = cfg.get(self.configname, 'spread_ratio')
        self.opt_holdings_path = cfg.get('path', 'opt_holdings_path')
        self.open_nextmonth_opt_days = cfg.getint(self.configname, 'open_nextmonth_opt_days')
        self.transform_days = cfg.getint(self.configname, 'transform_days')
        self.calendar = pd.read_csv('./data/tradingdays.csv', parse_dates=[0, 1])
        self.commission_per_unit = cfg.getfloat('trade', 'commission')

    def load_opt_basic_data(self, trading_day):
        """
        导入期权基本信息数据
        导入当月期权合约信息，如果trading_day为当月期权合约的最后交易日，那么导入次月合约
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        self.opts_data = {}
        self.trading_opts_data = {}
        header_name = ['opt_code', 'trade_code', 'opt_name', 'underlying_code', 'secu_type', 'opt_type',
                       'exercise_type', 'strike', 'multiplier', 'end_month', 'listed_date', 'expire_date',
                       'exercise_date', 'delivery_date']
        opts_basics = pd.read_csv('./data/OptBasics.csv', usecols=range(14), parse_dates=[10, 11, 12, 13],
                                  dtype={'期权代码': str})
        opts_basics.columns = header_name
        opts_basics.set_index(keys='opt_code', inplace=True)
        opts_basics = opts_basics[(opts_basics.expire_date >= trading_day) & (opts_basics.listed_date <= trading_day) &
                                  (opts_basics.multiplier == 10000)]

        expire_date = self.calendar[self.calendar.tradingday >= trading_day].iloc[self.open_nextmonth_opt_days - 1, 0].date()
        trading_opts_basics = opts_basics[(opts_basics.expire_date >= expire_date) &
                                          (opts_basics.listed_date <= trading_day)]
        # 选择当月合约，即当前交易合约中到期日最小的合约
        if len(trading_opts_basics) > 0:
            trading_opts_basics = trading_opts_basics[trading_opts_basics.expire_date == min(trading_opts_basics.expire_date)]

        # 导入未到期期权的基本信息数据
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
                # 导入参与交易期权的基本信息数据
                if optcode in trading_opts_basics.index:
                    self.trading_opts_data[optcode] = self.opts_data[optcode]

        # 如果当前日期为当月合约最后交易日的前一天，那么把持仓状态改为'transform’
        # if opts_basics.iloc[0, 10] - datetime.timedelta(days=1) == trading_day:
        #     self.opt_holdings.status = 'transform'
        # return True

    def load_opt_holdings(self, trading_day):
        """
        导入策略的期权持仓数据
        :param trading_day: 持仓日期（类型=datetime.date）
        :return:
        """
        strdate = trading_day.strftime('%Y%m%d')
        holding_filename = self.opt_holdings_path + self.configname + '/holding_' + self.portname + '_' + strdate + '.txt'
        self.opt_holdings.load_holdings(holding_filename)

    def load_settings(self, trading_day):
        """
        导入策略指定日期的设置
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        strdate = trading_day.strftime('%Y%m%d')
        settings_filename = '%s%s/settings_%s_%s.ini' % (self.opt_holdings_path, self.configname, self.portname, strdate)
        cfg = ConfigParser()
        cfg.read(settings_filename)
        str_call_ratio = cfg.get(self.configname, 'call_ratio')
        str_put_ratio = cfg.get(self.configname, 'put_ratio')
        if str_call_ratio != 'none':
            self.call_ratio = tuple(str_call_ratio.split(','))
        else:
            self.call_ratio = None
        if str_put_ratio != 'none':
            self.put_ratio = tuple(str_put_ratio.split(','))
        else:
            self.put_ratio = None

    def save_settings(self, trading_day):
        """
        保存策略的设置
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        strdate = trading_day.strftime('%Y%m%d')
        settings_filename = '%s%s/settings_%s_%s.ini' % (self.opt_holdings_path, self.configname, self.portname, strdate)
        with open(settings_filename, 'wt') as f:
            f.write('[' + self.configname + ']\n')
            if self.call_ratio is not None:
                f.write('call_ratio=%s,%s\n' % (self.call_ratio[0], self.call_ratio[1]))
            else:
                f.write('call_ratio=none\n')
            if self.put_ratio is not None:
                f.write('put_ratio=%s,%s\n' % (self.put_ratio[0], self.put_ratio[1]))
            else:
                f.write('put_ratio=none\n')

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

    def load_trading_datas(self, trading_day):
        """
        导入交易日交易相关数据，包含期权基本信息数据、期权分钟数据、标的分钟数据
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        if ~self.is_loaded_tradingdata:
            self.load_opt_basic_data(trading_day)
            self.load_opt_1min_quote(trading_day)
            self.load_underlying_1min_quote(trading_day)
            self.calc_opt_margin(trading_day)
            self.is_loaded_tradingdata = True

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
        for _, copt in self.trading_opts_data.items():
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
            for _, copt in self.trading_opts_data.items():
                if copt.opt_type == opt_type:
                    if copt.strike == atm_strike:
                        atm_opt = copt
                    else:
                        if opt_type == 'Call':
                            if otm_opt is None:
                                if copt.strike > atm_strike:
                                    otm_opt = copt
                            else:
                                if atm_strike < copt.strike < otm_opt.strike:
                                    otm_opt = copt
                        elif opt_type == 'Put':
                            if otm_opt is None:
                                if copt.strike < atm_strike:
                                    otm_opt = copt
                            else:
                                if otm_opt.strike < copt.strike < atm_strike:
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
        :return: 建仓完成时的交易时间，=最后一笔建仓时间的后一分钟，但如果len(self.trading_opts_data)==0，那么返回None
        """
        # 0.如果参与交易期权的基础数据为空，那么返回None
        if len(self.trading_opts_data) == 0:
            return None
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

    def do_liquidation(self, trading_beg_datetime, trading_min_num, opt_type, liquidation_size):
        """
        进行平仓操作
        :param trading_beg_datetime: 平仓开始时的交易时间
        :param trading_min_num: 平仓所需的分钟数，整型
        :param opt_type: 期权类型，Call=认购比率，Put=认沽比率
        :param liquidation_size: 平仓规模，HALF=半仓，ALL=全仓
        :return: 平仓完成时的交易时间，=最后一笔平仓时间的后一分钟
        """
        # 1.取得平值期权、虚值期权的代码，以及平值、虚值的期权类实例
        if opt_type == 'Call':
            atm_code, otm_code = self.call_ratio
        else:
            atm_code, otm_code = self.put_ratio
        atm_opt = COption(atm_code)
        otm_opt = COption(otm_code)
        # 2.根据平仓规模，计算平值、虚值期权平仓数量以及认购或认沽比率价差的代码
        if liquidation_size == 'HALF':
            atm_liquid_vol = int(self.opt_holdings.holdings[atm_code].holdingvol * 0.5 + 0.5)
            otm_liquid_vol = int(self.opt_holdings.holdings[otm_code].holdingvol * 0.5 + 0.5)
        else:
            atm_liquid_vol = self.opt_holdings.holdings[atm_code].holdingvol
            otm_liquid_vol = self.opt_holdings.holdings[otm_code].holdingvol
            if opt_type == 'Call':
                self.call_ratio = None
            else:
                self.put_ratio = None

        # 3.根据平仓的分钟数，计算每分钟平仓的数量
        atm_vol = int(atm_liquid_vol / trading_min_num + 0.5)
        otm_vol = int(otm_liquid_vol / trading_min_num + 0.5)
        # 4.在接下来的trading_min_num分钟里平仓，每分钟平仓平值、虚值的数量分别为atm_vol和opt_vol
        trade_datas = []
        is_liquid_done = False
        for min_num in range(trading_min_num):
            trading_datetime = trading_beg_datetime + datetime.timedelta(minutes=min_num)
            atm_price = self.opts_data[atm_code].quote_1min.ix[trading_datetime, 'close']
            otm_price = self.opts_data[otm_code].quote_1min.ix[trading_datetime, 'close']
            # 如果时间超过14:59:00，将剩余的全部平仓
            if trading_datetime.time() >= datetime.time(14, 59, 0):
                atm_vol = atm_liquid_vol
                if atm_vol > 0:
                    trade_datas.append(COptTradeData(atm_code, 'sell', 'close', atm_price, atm_vol,
                                                     atm_price * atm_vol * atm_opt.multiplier,
                                                     atm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[atm_code]))
                otm_vol = otm_liquid_vol
                if otm_vol > 0:
                    trade_datas.append(COptTradeData(otm_code, 'buy', 'close', otm_price, otm_vol,
                                                     otm_price * otm_vol * otm_opt.multiplier,
                                                     otm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[otm_code]))
                is_liquid_done = True
            else:
                if (atm_vol > atm_liquid_vol) & (otm_vol > otm_liquid_vol):
                    is_liquid_done = True
                if atm_vol > atm_liquid_vol:
                    atm_vol = atm_liquid_vol
                if otm_vol > otm_liquid_vol:
                    otm_vol = otm_liquid_vol
                if atm_vol > 0:
                    trade_datas.append(COptTradeData(atm_code, 'sell', 'close', atm_price, atm_vol,
                                                     atm_price * atm_vol * atm_opt.multiplier,
                                                     atm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[atm_code]))
                if otm_vol > 0:
                    trade_datas.append(COptTradeData(otm_code, 'buy', 'close', otm_price, otm_vol,
                                                     otm_price * otm_vol * otm_opt.multiplier,
                                                     otm_vol * self.commission_per_unit, trading_datetime,
                                                     self.opts_data[otm_code]))
            atm_liquid_vol -= atm_vol
            otm_liquid_vol -= otm_vol
            if is_liquid_done:
                break
        # 5.更新持仓数据
        self.opt_holdings.update_holdings(trade_datas)
        # 6.返回平仓完成时的交易时间，=最后一笔平仓时间的后一分钟
        return trading_beg_datetime + datetime.timedelta(minutes=trading_min_num)

    def transfer_status(self, pre_status, status, trading_datetime):
        """
        处理转变状态相关的交易
        :param pre_status: 交易前状态
        :param status: 交易后状态
        :param trading_datetime: 交易时间，类型=datetime.datetime
        :return:
        """
        self.load_trading_datas(trading_datetime.date())
        funderlying_price = self.underlying_quote_1min.ix[trading_datetime, 'close']
        fnav = self.opt_holdings.net_asset_value(trading_datetime)
        if pre_status == 'NONE':
            if status == 'PUT_RATIO':
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'PUT_RATIO'
            elif status == 'CALL_RATIO':
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'CALL_RATIO'
            elif status == 'CALL_PUT_RATIO':
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_HALF_RATIO'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'
        elif pre_status == 'CALL_HALF_RATIO':
            if status == 'PUT_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'PUT_RATIO'
            elif status == 'CALL_RATIO':
                atm_opt, otm_opt = self.get_ratiospread_opts(funderlying_price, 'Call')
                if (atm_opt is not None) and (otm_opt is not None):
                    if self.call_ratio == (atm_opt.code, otm_opt.code):
                        if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                            self.opt_holdings.status = 'CALL_RATIO'
                    else:
                        self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                        self.opt_holdings.status = 'NONE'
                        if self.do_position(trading_datetime, 20, 'Call', fnav * 0.8) is not None:
                            self.opt_holdings.status = 'CALL_RATIO'
            elif status == 'CALL_PUT_RATIO':
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'
            elif status == 'CALL_HALF_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_HALF_RATIO'
        elif pre_status == 'CALL_RATIO':
            if status == 'PUT_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'PUT_RATIO'
            elif status == 'CALL_PUT_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Call', 'HALF')
                self.opt_holdings.status = 'CALL_HALF_RATIO'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'
            elif status == 'CALL_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'CALL_RATIO'
        elif pre_status == 'PUT_HALF_RATIO':
            if status == 'PUT_RATIO':
                atm_opt, otm_opt = self.get_ratiospread_opts(funderlying_price, 'Put')
                if (atm_opt is not None) and (otm_opt is not None):
                    if self.put_ratio == (atm_opt.code, otm_opt.code):
                        if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                            self.opt_holdings.status = 'PUT_RATIO'
                    else:
                        self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                        self.opt_holdings.status = 'NONE'
                        if self.do_position(trading_datetime, 20, 'Put', fnav * 0.8) is not None:
                            self.opt_holdings.status = 'PUT_RATIO'
            elif status == 'CALL_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'CALL_RATIO'
            elif status == 'CALL_PUT_RATIO':
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'
            elif status == 'PUT_HALF_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'PUT_HALF_RATIO'
        elif pre_status == 'PUT_RATIO':
            if status == 'CALL_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'CALL_RATIO'
            elif status == 'CALL_PUT_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Put', 'HALF')
                self.opt_holdings.status = 'PUT_HALF_RATIO'
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'
            elif status == 'PUT_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'PUT_RATIO'
        elif pre_status == 'CALL_PUT_RATIO':
            if status == 'PUT_RATIO':
                atm_opt, otm_opt = self.get_ratiospread_opts(funderlying_price, 'Put')
                self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'PUT_HALF_RATIO'
                if (atm_opt is not None) and (otm_opt is not None):
                    if self.put_ratio == (atm_opt.code, otm_opt.code):
                        if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                            self.opt_holdings.status = 'PUT_RATIO'
                    else:
                        self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                        self.opt_holdings.status = 'NONE'
                        if self.do_position(trading_datetime, 20, 'Put', fnav * 0.8) is not None:
                            self.opt_holdings.status = 'PUT_RATIO'
            elif status == 'CALL_RATIO':
                atm_opt, otm_opt = self.get_ratiospread_opts(funderlying_price, 'Call')
                self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'CALL_HALF_RATIO'
                if (atm_opt is not None) and (otm_opt is not None):
                    if self.call_ratio == (atm_opt.code, otm_opt.code):
                        if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                            self.opt_holdings.status = 'CALL_RATIO'
                    else:
                        self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                        self.opt_holdings.status = 'NONE'
                        if self.do_position(trading_datetime, 20, 'Call', fnav * 0.8) is not None:
                            self.opt_holdings.status = 'CALL_RATIO'
            elif status == 'CALL_PUT_RATIO':
                self.do_liquidation(trading_datetime, 20, 'Call', 'ALL')
                self.do_liquidation(trading_datetime, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_datetime, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_HALF_RATIO'
                if self.do_position(trading_datetime, 20, 'Put', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'

    def do_term_transfer(self, trading_day):
        """
        处理移仓交易
        :param trading_day: 交易日期，类型=datetime.date
        :return:
        """
        trading_time = datetime.datetime(trading_day.year, trading_day.month, trading_day.day, 14, 30, 0)
        if self.opt_holdings.status == 'CALL_HALF_RATIO':
            call_expire_date = self.opt_holdings.holdings[self.call_ratio[0]].COption.end_date
            if self.calendar[self.calendar.tradingday <= call_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                self.load_trading_datas(trading_day)
                fnav = self.opt_holdings.net_asset_value(trading_time)
                self.do_liquidation(trading_time, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_time, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_HALF_RATIO'
        elif self.opt_holdings.status == 'CALL_RATIO':
            call_expire_date = self.opt_holdings.holdings[self.call_ratio[0]].COption.end_date
            if self.calendar[self.calendar.tradingday <= call_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                self.load_trading_datas(trading_day)
                fnav = self.opt_holdings.net_asset_value(trading_time)
                self.do_liquidation(trading_time, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_time, 20, 'Call', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'CALL_RATIO'
        elif self.opt_holdings.status == 'PUT_HALF_RATIO':
            put_expire_date = self.opt_holdings.holdings[self.put_ratio[0]].COption.end_date
            if self.calendar[self.calendar.tradingday <= put_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                self.load_trading_datas(trading_day)
                fnav = self.opt_holdings.net_asset_value(trading_time)
                self.do_liquidation(trading_time, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_time, 20, 'Put', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'PUT_HALF_RATIO'
        elif self.opt_holdings.status == 'PUT_RATIO':
            put_expire_date = self.opt_holdings.holdings[self.put_ratio[0]].COption.end_date
            if self.calendar[self.calendar.tradingday <= put_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                self.load_trading_datas(trading_day)
                fnav = self.opt_holdings.net_asset_value(trading_time)
                self.do_liquidation(trading_time, 20, 'Put', 'ALL')
                self.opt_holdings.status = 'NONE'
                if self.do_position(trading_time, 20, 'Put', fnav * 0.8) is not None:
                    self.opt_holdings.status = 'PUT_RATIO'
        elif self.opt_holdings.status == 'CALL_PUT_RATIO':
            call_expire_date = self.opt_holdings.holdings[self.call_ratio[0]].COption.end_date
            if self.calendar[self.calendar.tradingday <= call_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                self.load_trading_datas(trading_day)
                fnav = self.opt_holdings.net_asset_value(trading_time)
                self.do_liquidation(trading_time, 20, 'Call', 'ALL')
                self.opt_holdings.status = 'PUT_HALF_RATIO'
                if self.do_position(trading_time, 20, 'Call', fnav * 0.4) is not None:
                    self.opt_holdings.status = 'CALL_PUT_RATIO'
                    put_expire_date = self.opt_holdings.holdings[self.put_ratio[0]].COption.end_date
                    if self.calendar[self.calendar.tradingday <= put_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                        self.do_liquidation(trading_time, 20, 'Put', 'ALL')
                        self.opt_holdings.status = 'CALL_HALF_RATIO'
                        if self.do_position(trading_time, 20, 'Put', fnav * 0.4) is not None:
                            self.opt_holdings.status = 'CALL_PUT_RATIO'
                else:
                    self.do_term_transfer(trading_day)
            else:
                put_expire_date = self.opt_holdings.holdings[self.put_ratio[0]].COption.end_date
                if self.calendar[self.calendar.tradingday <= put_expire_date].iloc[-self.transform_days, 0].date() == trading_day:
                    self.load_trading_datas(trading_day)
                    fnav = self.opt_holdings.net_asset_value(trading_time)
                    self.do_liquidation(trading_time, 20, 'Put', 'ALL')
                    self.opt_holdings.status = 'CALL_HALF_RATIO'
                    if self.do_position(trading_time, 20, 'Put', fnav * 0.4) is not None:
                        self.opt_holdings.status = 'CALL_PUT_RATIO'


    def on_vol_trading(self, trading_day, pre_trading_day):
        """
        指定某一交易日期，进行基于均线系统的波动率交易
        :param trading_day: 交易日期，类型=datetime.date
        :param pre_trading_day: 前一交易日期，类型=datetime.date
        :return:
        """
        self.is_loaded_tradingdata = False
        # 调用均线系统，计算当天的市场状态
        mkt_status = mean_average('sh000016', pre_trading_day, self.ma_days, self.ma_deviation)
        with open(self.opt_holdings_path + self.configname + '/log.txt', 'at') as f:
            f.write(trading_day.strftime('%Y-%m-%d') + str(mkt_status) + '\n')
        print("%s of %s" % (trading_day.strftime('%Y-%m-%d'), self.configname))
        # 导入期权持仓数据
        self.load_opt_holdings(pre_trading_day)
        # 导入策略设置
        self.load_settings(pre_trading_day)
        # 根据期权持仓状态和市场状态进行不同操作,NONE=空仓，CALL_RATIO=认购比率，PUT_RATIO=认沽比率，CALL_PUT_RATIO=认购、认沽比率
        trading_time = datetime.datetime(trading_day.year, trading_day.month, trading_day.day, 9, 30, 0)
        # 当前持仓状态为“空仓”
        if self.opt_holdings.status == 'NONE':
            # 市场状态为牛市，建仓认沽比率价差
            if mkt_status == MktStatus.Bullish:
                self.transfer_status('NONE', 'PUT_RATIO', trading_time)
            # 市场状态为熊市，建仓认购比率价差
            elif mkt_status == MktStatus.Bearish:
                self.transfer_status('NONE', 'CALL_RATIO', trading_time)
            # 市场状态为震荡市，建仓认购、认沽比率价差
            elif mkt_status == MktStatus.Volatile:
                self.transfer_status('NONE', 'CALL_PUT_RATIO', trading_time)
        # 当前持仓状态为“半仓认购比率”
        elif self.opt_holdings.status == 'CALL_HALF_RATIO':
            # 持仓状态为“半仓认购比率”、当前市场状态为“牛市”，平仓全部认购比率价差、开仓认沽比率价差（全仓）
            if mkt_status == MktStatus.Bullish:
                self.transfer_status('CALL_HALF_RATIO', 'PUT_RATIO', trading_time)
            # 持仓状态为“半仓认购比率”、当前市场状态为“熊市”，继续开仓认购比率价差至满仓
            elif mkt_status == MktStatus.Bearish:
                self.transfer_status('CALL_HALF_RATIO', 'CALL_RATIO', trading_time)
            # 持仓状态为“半仓认购比率”、当前市场状态为“震荡市”，开仓认沽比率价差（半仓）
            elif mkt_status == MktStatus.Volatile:
                self.transfer_status('CALL_HALF_RATIO', 'CALL_PUT_RATIO', trading_time)
        # 当前持仓状态为“认购比率”
        elif self.opt_holdings.status == 'CALL_RATIO':
            # 持仓状态为“认购比率”、当前市场状态为“牛市”，平仓全部认购比率价差、开仓认沽比率价差
            if mkt_status == MktStatus.Bullish:
                self.transfer_status('CALL_RATIO', 'PUT_RATIO', trading_time)
            # 持仓状态为“认购比率”、当前市场状态为“熊市”，不做操作
            elif mkt_status == MktStatus.Bearish:
                pass
            # 持仓状态为“认购比率”、当前市场状态为“震荡市”，平认购比率价差一半仓位、开仓认沽比率差价
            elif mkt_status == MktStatus.Volatile:
                self.transfer_status('CALL_RATIO', 'CALL_PUT_RATIO', trading_time)
        # 当前持仓状态为“半仓认沽比率”
        elif self.opt_holdings.status == 'PUT_HALF_RATIO':
            # 持仓状态为“半仓认沽比率”、当前市场状态为“牛市”，继续开仓认沽比率价差至满仓
            if mkt_status == MktStatus.Bullish:
                self.transfer_status('PUT_HALF_RATIO', 'PUT_RATIO', trading_time)
            # 持仓状态为“半仓认沽比率”、当前市场状态为“熊市”，平仓全部认沽比率价差、开仓认购比率价差（全仓）
            if mkt_status == MktStatus.Bearish:
                self.transfer_status('PUT_HALF_RATIO', 'CALL_RATIO', trading_time)
            # 持仓状态为“半仓认沽比率”、当前市场状态为“震荡市”，开仓认购比率价差（半仓）
            if mkt_status == MktStatus.Volatile:
                self.transfer_status('PUT_HALF_RATIO', 'CALL_PUT_RATIO', trading_time)
        # 当前持仓状态为“认沽比率”
        elif self.opt_holdings.status == 'PUT_RATIO':
            # 持仓状态为“认沽比率”，当前市场状态为“牛市”，不做操作
            if mkt_status == MktStatus.Bullish:
                pass
            # 持仓状态为“认沽比率”，当前市场状态为“熊市”，平仓全部认沽比率价差、开仓认购比率价差
            elif mkt_status == MktStatus.Bearish:
                self.transfer_status('PUT_RATIO', 'CALL_RATIO', trading_time)
            # 持仓状态为“认沽比率”，当前市场状态为“震荡市“，认沽比率价差平仓一半、开仓认购比率价差
            elif mkt_status == MktStatus.Volatile:
                self.transfer_status('PUT_RATIO', 'CALL_PUT_RATIO', trading_time)
        # 当前持仓状态为“认购认沽比率”
        elif self.opt_holdings.status == 'CALL_PUT_RATIO':
            # 持仓状态为“认购认沽比率”，当前市场状态为“牛市”，平仓认购比率价差的仓位、开仓认沽比率价差
            if mkt_status == MktStatus.Bullish:
                self.transfer_status('CALL_PUT_RATIO', 'PUT_RATIO', trading_time)
            # 持仓状态为“认购认沽比率”，当前市场状态为“熊市”，平仓认沽比率价差的仓位、开仓认购比率
            elif mkt_status == MktStatus.Bearish:
                self.transfer_status('CALL_PUT_RATIO', 'CALL_RATIO', trading_time)
            # 持仓状态为“认购认沽比率”，当前市场状态为“震荡市”，不做操作
            elif mkt_status == MktStatus.Volatile:
                pass
        # 如果当天为移仓日期，那么收盘前移仓
        self.do_term_transfer(trading_day)
        # if self.call_ratio is not None:
        #     call_expire_date = self.opt_holdings.holdings[self.call_ratio[0]].COption.end_date
        # else:
        #     call_expire_date = None
        # if self.put_ratio is not None:
        #     put_expire_date = self.opt_holdings.holdings[self.put_ratio[0]].COption.end_date
        # else:
        #     put_expire_date = None
        # if (call_expire_date is None) and (put_expire_date is None):
        #     expire_date = None
        # elif call_expire_date is None:
        #     expire_date = put_expire_date
        # elif put_expire_date is None:
        #     expire_date = call_expire_date
        # else:
        #     expire_date = min(call_expire_date, put_expire_date)
        # if (expire_date is not None) and (self.calendar[self.calendar.tradingday <= expire_date].iloc[-self.transform_days, 0].date() == trading_day):
        #     trading_time = datetime.datetime(trading_day.year, trading_day.month, trading_day.day, 14, 30, 0)
        #     self.transfer_status(self.opt_holdings.status, self.opt_holdings.status, trading_time)
        # 每个交易日结束，计算持仓保证金、持仓净值、P&L，并保存持仓数据、P&L及策略设置
        self.opt_holdings.calc_margin(trading_day)
        self.opt_holdings.p_and_l(trading_day)
        self.opt_holdings.net_asset_value(trading_day)
        holding_filename = self.opt_holdings_path + self.configname + '/holding_' + self.portname + '_' + trading_day.strftime('%Y%m%d') + '.txt'
        self.opt_holdings.save_holdings(holding_filename)
        self.save_settings(trading_day)

    def on_vol_trading_interval(self, beg_date, end_date):
        """
        指定日期区间，进行波动率交易
        :param beg_date: 开始日期，类型=datetime.date
        :param end_date: 结束日期，类型=datetime.date
        :return:
        """
        df_tradingdays = self.calendar[(self.calendar.tradingday >= beg_date) & (self.calendar.tradingday <= end_date)]
        for _, tradingdays in df_tradingdays.iterrows():
            trading_day = tradingdays['tradingday'].date()
            pre_trading_day = tradingdays['pre_tradingday'].date()
            self.on_vol_trading(trading_day, pre_trading_day)

# 结合均线系统的比率价差交易入口
if __name__ == '__main__':
    vol_strategy = CVolTrendTradingStrategy('VolTrade', 'vol_trend_strategy')
    print('ma_days = %d, ma_deviation = %0.3f' % (vol_strategy.ma_days, vol_strategy.ma_deviation))
    tmbeg_date = datetime.date(2017, 8, 25)
    tmend_date = datetime.date(2017, 10, 17)
    vol_strategy.on_vol_trading_interval(tmbeg_date, tmend_date)

    # tmbeg_date = datetime.date(2015, 2, 9)
    # tmend_date = datetime.date(2017, 8, 24)
    # config = ConfigParser()
    # config.read('config.ini')
    # deviations = ['0.005','0.010','0.015', '0.020', '0.025', '0.030', '0.035', '0.040']
    # for days in range(60, 61):
    #     for deviation in deviations:
    #         # 添加配置项
    #         section = 'vol_trend_%d_%s' % (days, deviation)
    #         if config.has_section(section) == False:
    #             config.add_section(section)
    #             config.set(section, 'ma_days', str(days))
    #             config.set(section, 'ma_deviation', deviation)
    #             config.set(section, 'spread_ratio', '1:3')
    #             config.set(section, 'transform_days', '5')
    #             config.set(section, 'open_nextmonth_opt_days', '10')
    #             config.set(section, 'marginratio', '0.9')
    #             with open('config.ini', 'w') as f:
    #                 config.write(f)
    #         # 执行该配置对应的测试
    #         vol_strategy = CVolTrendTradingStrategy('VolTrade', section)
    #         print('\nma_days = %d, ma_deviation = %0.3f' % (vol_strategy.ma_days, vol_strategy.ma_deviation))
    #         vol_strategy.on_vol_trading_interval(tmbeg_date, tmend_date)
    #         # 暂停5分钟
    #         time.sleep(400)
