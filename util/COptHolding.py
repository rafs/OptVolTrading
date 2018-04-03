import datetime
from util.util import CGreeks, CSingleOptHolding
from util.COption import COption
import pandas as pd


class COptHolding(object):
    """期权持仓类"""

    def __init__(self, str_logfile_path=None):
        """
        初始化持仓数据及持仓汇总希腊字母
        class attributes:
        holdings: {'code': CSingleOptHolding}
        mode: 模式，逐步加仓、恒定gamma
        status: 状态，onposition=建仓，positioned=建仓完成，onliquidation=平仓
        cashinflow: 资金流入(卖出开仓、卖出平仓)
        cashoutflow: 资金流出(买入开仓、买入平仓)
        commission: 佣金
        pandl: 盈亏
        capital: 规模
        nav: 组合净值
        gammaexposure: gamma暴露
        greeks: CGreeks类
        logger: 日志类
        logfilepath: 日志文件路径
        """
        self.holdings = {}
        self.mode = ''
        self.status = ''
        self.cashinflow = 0.0
        self.cashoutflow = 0.0
        self.commission = 0.0
        self.pandl = 0.0
        self.capital = 0.0
        self.nav = 0.0
        self.gammaexposure = 0.0
        self.holdingmv = 0.0
        self.greeks = CGreeks()
        # self.logger = None
        self.logfilepath = str_logfile_path

    def load_holdings(self, holding_filename):
        """加载持仓数据,同时加载期权的分钟行情数据"""
        # 先清空持仓数据
        self.holdings = {}
        with open(holding_filename, 'rt', encoding='UTF-8') as f:
            bIsPandLSection = False
            bIsHoldingSection = False
            while True:
                strline = f.readline().strip('\n')
                if not strline:
                    break
                if strline == '[begin of P&L]':
                    bIsPandLSection = True
                    strline = f.readline().strip('\n')
                if strline == '[end of P&L]':
                    bIsPandLSection = False
                    strline = f.readline().strip('\n')
                if strline == '[begin of holdings]':
                    bIsHoldingSection = True
                    strline = f.readline().strip('\n')
                if strline == '[end of holdings]':
                    bIsHoldingSection = False

                if bIsPandLSection:
                    ind_name, ind_value = strline.split(sep='=')
                    if ind_name == 'mode':
                        self.mode = ind_value
                    elif ind_name == 'status':
                        if self.status != 'onliquidation':
                            self.status = ind_value
                    elif ind_name == 'cashinflow':
                        self.cashinflow = float(ind_value)
                    elif ind_name == 'cashoutflow':
                        self.cashoutflow = float(ind_value)
                    elif ind_name == 'commission':
                        self.commission = float(ind_value)
                    elif ind_name == 'pandl':
                        self.pandl = float(ind_value)
                    elif ind_name == 'capital':
                        self.capital = float(ind_value)
                    elif ind_name == 'nav':
                        self.nav = float(ind_value)
                    elif ind_name == 'gammaexposure':
                        self.gammaexposure = float(ind_value)
                if bIsHoldingSection:
                    holding_data = strline.split(sep='|')
                    dict_holding = {}
                    for data in holding_data:
                        ind_name, ind_value = data.split(sep='=')
                        dict_holding[ind_name] = ind_value
                    if dict_holding['code'] not in self.holdings:
                        holding_side = int(dict_holding['holdingside'])
                        holding_vol = int(dict_holding['holdingvol'])
                        holding_opt = COption(dict_holding['code'])
                        self.holdings[dict_holding['code']] = CSingleOptHolding(holding_side, holding_vol, holding_opt)
                    else:
                        # strmsg = "持仓文件：" + holding_filename + "中期权" + dict_holding['code'] + "的持仓数据重复\n"
                        strmsg = "holding data of option:%s in holding file: %s was duplicated.\n" % (dict_holding['code'], holding_filename)
                        # self.logger.error(strmsg)
                        with open(self.logfilepath, 'at', encoding='UTF-8') as f:
                            f.write(strmsg)

    def save_holdings(self, holding_filename):
        """保存持仓数据"""
        # 如果当前的状态为onliquidation，那么把状态改为onposition
        if self.status == 'onliquidation':
            self.status = 'onposition'
        with open(holding_filename, 'wt', encoding='UTF-8') as f:
            f.write('[begin of P&L]\n')
            f.write('mode=%s\n' % self.mode)
            f.write('status=%s\n' % self.status)
            f.write('cashinflow=%0.2f\n' % self.cashinflow)
            f.write('cashoutflow=%0.2f\n' % self.cashoutflow)
            f.write('commission=%0.2f\n' % self.commission)
            f.write('pandl=%0.2f\n' % self.pandl)
            f.write('capital=%0.2f\n' % self.capital)
            f.write('nav=%0.2f\n' % self.nav)
            f.write('gammaexposure=%0.2f\n' % self.gammaexposure)
            f.write('gamma_mv=%0.2f\n' % self.greeks.gamma_mv)
            f.write('delta_mv=%0.2f\n' % self.greeks.delta_mv)
            f.write('total_margin=%0.2f\n' % self.total_margin())
            f.write('holding_mv=%0.2f\n' % self.holdingmv)
            f.write('[end of P&L]\n\n')

            f.write('[begin of holdings]\n')
            for opt_code, opt_holding in self.holdings.items():
                f.write('code=%s|holdingside=%d|holdingvol=%d\n' % (opt_code, opt_holding.holdingside, opt_holding.holdingvol))
            f.write('[end of holdings]\n')

    def calc_greeks(self, underlying_price, risk_free, dividend_rate, vol, calc_datetime=datetime.datetime.now()):
        """计算期权持仓的汇总希腊字母值"""
        self.greeks.delta = 0.0
        self.greeks.gamma = 0.0
        self.greeks.vega = 0.0
        self.greeks.delta_mv = 0.0
        self.greeks.gamma_mv = 0.0
        # 遍历持仓，先计算每个持仓期权的希腊字母，然后汇总希腊字母值
        for code, holding in self.holdings.items():
            holding.COption.calc_greeks(underlying_price, risk_free, dividend_rate, vol, calc_datetime)
            self.greeks.delta += (holding.COption.greeks.delta * holding.COption.multiplier *
                                  holding.holdingside * holding.holdingvol)
            self.greeks.gamma += (holding.COption.greeks.gamma * holding.COption.multiplier *
                                  holding.holdingside * holding.holdingvol)
            self.greeks.vega += (holding.COption.greeks.vega * holding.COption.multiplier *
                                 holding.holdingside * holding.holdingvol)
            self.greeks.delta_mv += (holding.COption.greeks.delta_mv * holding.holdingside * holding.holdingvol)
            self.greeks.gamma_mv += (holding.COption.greeks.gamma_mv * holding.holdingside * holding.holdingvol)

    def calc_margin(self, trading_day):
        """
        计算持仓期权的开仓保证金
        :param trading_day: 日期（类型=datetime.date）
        :return:
        """
        # 1.读取标的日K线时间序列
        underlying_quote = pd.read_csv('./data/underlying_daily_quote.csv', index_col=0, parse_dates=[0], encoding='UTF-8')
        underlying_pre_close = float(underlying_quote.ix[trading_day, 'pre_close'])
        # 2.读取样本期权的日行情
        strdate = trading_day.strftime('%Y-%m-%d')
        strfilepath = '../opt_quote/' + strdate + '/50OptionDailyQuote.csv'
        opts_quote = pd.read_csv(strfilepath, usecols=range(1, 14), parse_dates=[0], encoding='gb18030', dtype={'option_code':str})
        opts_quote.set_index(keys='option_code', inplace=True)
        # 3.计算持仓期权的开仓保证金
        for optcode, holding in self.holdings.items():
            if optcode in opts_quote.index:
                opt_pre_settle = float(opts_quote.ix[optcode, 'pre_settle'])
                holding.COption.calc_margin(opt_pre_settle, underlying_pre_close)
            else:
                holding.COption.margin = 3000.0

    def total_margin(self):
        """返回持仓期权的合计保证金"""
        fmargin = 0.0
        for optcode, optholding in self.holdings.items():
            if optholding.holdingside == -1:
                fmargin += optholding.COption.margin * optholding.holdingvol
        return fmargin

    def margin_ratio(self):
        """返回保证金占资金规模的比例"""
        return self.total_margin() / self.capital

    def get_least_timevalue_opts(self, underlying_price, trading_datetime, exclusions=None):
        """
        取得持仓中时间价值最小的认购、认沽期权
        :param underlying_price: 标的最新价格
        :param trading_datetime: 交易时间(类型=datetime.datetime)
        :param exclusions: 需要排除的代码列表
        :return: tuple(认购COption,认沽COption)
        """
        if exclusions is None:
            exclusions = []
        opt_call = None
        opt_put = None
        for optcode, holding in self.holdings.items():
            if holding.holdingvol > 0 and optcode not in exclusions:
                if holding.COption.opt_type == "Call":
                    if opt_call is None:
                        opt_call = holding.COption
                    else:
                        if holding.COption.time_value(underlying_price, trading_datetime) < \
                                opt_call.time_value(underlying_price, trading_datetime):
                            opt_call = holding.COption
                else:
                    if opt_put is None:
                        opt_put = holding.COption
                    else:
                        if holding.COption.time_value(underlying_price, trading_datetime) < \
                                opt_put.time_value(underlying_price, trading_datetime):
                            opt_put = holding.COption
        return opt_call, opt_put

    def get_minmax_gamma_opts(self, exclusions=None):
        """
        取得持仓中gamma值最大的认购认沽期权，以及gamma值最小的认购认沽期权
        :param exclusions: 需要排除的代码列表
        :return: dict{'min'/'max':tuple(COption of call, COption of put)}
        """
        if exclusions is None:
            exclusions = []
        min_gamma_call = None
        min_gamma_put = None
        max_gamma_call = None
        max_gamma_put = None
        for optcode, holding in self.holdings.items():
            if holding.holdingvol > 0 and optcode not in exclusions:
                if holding.COption.opt_type == 'Call':
                    if min_gamma_call is None:
                        min_gamma_call = holding.COption
                    if max_gamma_call is None:
                        max_gamma_call = holding.COption
                    if holding.COption.greeks.gamma < min_gamma_call.greeks.gamma:
                        min_gamma_call = holding.COption
                    if holding.COption.greeks.gamma > max_gamma_call.greeks.gamma:
                        max_gamma_call = holding.COption
                else:
                    if min_gamma_put is None:
                        min_gamma_put = holding.COption
                    if max_gamma_put is None:
                        max_gamma_put = holding.COption
                    if holding.COption.greeks.gamma < min_gamma_put.greeks.gamma:
                        min_gamma_put = holding.COption
                    if holding.COption.greeks.gamma > max_gamma_put.greeks.gamma:
                        max_gamma_put = holding.COption
        return {'min': (min_gamma_call, min_gamma_put), 'max': (max_gamma_call, max_gamma_put)}

    def verify_update_tradedata(self, tradedata):
        """
        校验单条交易记录的有效性，若有效，那么同时更新交易数据至持仓数据
        :param tradedata: 单条期权交易记录
        :return: 校验通过并返回True，校验错误返回False
        """
        if tradedata.code in self.holdings:
            if tradedata.tradeside == 'buy' and tradedata.openclose == 'open':
                if self.holdings[tradedata.code].holdingside == 1:
                    self.holdings[tradedata.code].holdingvol += tradedata.tradevol
                    self.cashoutflow += tradedata.tradevalue
                    self.commission += tradedata.commission
                    # return True
                else:
                    self.holdings[tradedata.code].holdingvol -= tradedata.tradevol
                    self.cashoutflow += tradedata.tradevalue
                    self.commission += tradedata.commission
            elif tradedata.tradeside == 'buy' and tradedata.openclose == 'close':
                if self.holdings[tradedata.code].holdingside == -1:
                    self.holdings[tradedata.code].holdingvol -= tradedata.tradevol
                    self.cashoutflow += tradedata.tradevalue
                    self.commission += tradedata.commission
                    # return True
                else:
                    self.holdings[tradedata.code].holdingvol += tradedata.tradevol
                    self.cashoutflow += tradedata.tradevalue
                    self.commission += tradedata.commission
            elif tradedata.tradeside == 'sell' and tradedata.openclose == 'open':
                if self.holdings[tradedata.code].holdingside == -1:
                    self.holdings[tradedata.code].holdingvol += tradedata.tradevol
                    self.cashinflow += tradedata.tradevalue
                    self.commission += tradedata.commission
                    # return True
                else:
                    self.holdings[tradedata.code].holdingvol -= tradedata.tradevol
                    self.cashinflow += tradedata.tradevalue
                    self.commission += tradedata.commission
            elif tradedata.tradeside == 'sell' and tradedata.openclose == 'close':
                if self.holdings[tradedata.code].holdingside == 1:
                    self.holdings[tradedata.code].holdingvol -= tradedata.tradevol
                    self.cashinflow += tradedata.tradevalue
                    self.commission += tradedata.commission
                    # return True
                else:
                    self.holdings[tradedata.code].holdingvol += tradedata.tradevol
                    self.cashinflow += tradedata.tradevalue
                    self.commission += tradedata.commission
        else:
            if tradedata.openclose == 'close':
                tradedata.openclose = 'open'
            if tradedata.tradeside == 'buy' and tradedata.openclose == 'open':
                # holding = CSingleOptHolding(side=1, vol=tradedata.tradevol, opt=COption(tradedata.code))
                holding = CSingleOptHolding(side=1, vol=tradedata.tradevol, opt=tradedata.opt)
                self.cashoutflow += tradedata.tradevalue
                self.commission += tradedata.commission
                self.holdings[tradedata.code] = holding
                # return True
            elif tradedata.tradeside == 'sell' and tradedata.openclose == 'open':
                # holding = CSingleOptHolding(side=-1, vol=tradedata.tradevol, opt=COption(tradedata.code))
                holding = CSingleOptHolding(side=-1, vol=tradedata.tradevol, opt=tradedata.opt)
                self.cashinflow += tradedata.tradevalue
                self.commission += tradedata.commission
                self.holdings[tradedata.code] = holding
                # return True
            else:
                return False
        # 如果该交易记录对应期权持仓量=0，那么删除该条期权持仓
        if self.holdings[tradedata.code].holdingvol == 0:
            del_holding = self.holdings.pop(tradedata.code)
            if del_holding is not None:
                # self.logger.info("期权%s的持仓量等于0，该持仓已删除。" % tradedata.code)
                # strmsg = "期权%s的持仓量等于0，该持仓已删除。\n" % tradedata.code
                strmsg = "holding vol of option: %s is equal to 0, the holding data was deleted.\n" % tradedata.code
            else:
                # self.logger.error("删除期权%s持仓数据失败！" % tradedata.code)
                # strmsg = "删除期权%s持仓数据失败！\n" % tradedata.code
                strmsg = "failed to delete holding data of option: %s\n" % tradedata.code
            with open(self.logfilepath, 'at', encoding='UTF-8') as f:
                f.write(strmsg)
        # 如果该交易记录对应的期权持仓量<0, 修改持仓记录
        elif self.holdings[tradedata.code].holdingvol < 0:
            self.holdings[tradedata.code].holdingside *= -1
            self.holdings[tradedata.code].holdingvol *= -1
        return True

    def update_holdings(self, tradedatas):
        """
        根据给定的期权交易数据列表，更新持仓数据
        :param tradedatas: 期权交易数据(COptTradeData)列表
        :return: 无
        """
        # 遍历期权交易数据列表，更新每条期权交易数据
        with open(self.logfilepath, 'at', encoding='UTF-8') as f:
            for tradedata in tradedatas:
                log_msg = "trade info: time=%s,code=%s,tradeside=%s,openclose=%s,price=%f,vol=%d,value=%f,commission=%f\n" % \
                          (tradedata.time.strftime('%Y-%m-%d %H:%M:%S'), tradedata.code, tradedata.tradeside, tradedata.openclose,
                           tradedata.tradeprice, tradedata.tradevol, tradedata.tradevalue, tradedata.commission)
                # self.logger.info(log_msg)
                f.write(log_msg)
                if not self.verify_update_tradedata(tradedata):
                    # print('%s,期权%s的交易数据出错\n' % (tradedata.time.strftime('%H:%M:%S'), tradedata.code))
                    # self.logger.error('上一条交易数据错误')
                    f.write('the last trading data was wrong.\n')
        # 更新持仓期权的保证金
        if tradedatas:
            self.calc_margin(tradedatas[0].time.date())

    def holding_mv(self, trading_datetime, holdingside = None):
        """
        计算持仓期权的总市值
        :param trading_datetime: 计算时间，类型=datetime.datetime或者datetime.date
        :param holdingside: None（计算全部持仓市值）；1（计算多头持仓市值）；-1（计算空头持仓市值）
        :return:
        """
        opt_mv = 0.0
        # 如果trading_datetime是datetime.date类型，那么导入持仓期权当天的日行情
        # if isinstance(trading_datetime, datetime.date):
        if type(trading_datetime).__name__ == 'date':
            str_dailyquote_path = '../opt_quote/%s/50OptionDailyQuote.csv' % trading_datetime.strftime('%Y-%m-%d')
            opt_daily_hq = pd.read_csv(str_dailyquote_path, usecols=range(1,14), parse_dates=[0], encoding='gb18030', dtype={'option_code':str})
            opt_daily_hq.set_index(keys='option_code', inplace=True)
            if holdingside is None:
                for optcode, optholding in self.holdings.items():
                    opt_price = opt_daily_hq.ix[optcode, 'close']
                    opt_mv += optholding.holdingside * optholding.holdingvol * opt_price * optholding.COption.multiplier
            else:
                for optcode, optholding in self.holdings.items():
                    if optholding.holdingside == holdingside:
                        opt_price = opt_daily_hq.ix[optcode, 'close']
                        opt_mv += holdingside * optholding.holdingvol * opt_price * optholding.COption.multiplier
        # elif isinstance(trading_datetime, datetime.datetime):
        elif type(trading_datetime).__name__ in ['datetime','Timestamp']:
            if holdingside is None:
                for optcode, optholding in self.holdings.items():
                    opt_price = optholding.COption.quote_1min.ix[trading_datetime, 'close']
                    opt_mv += optholding.holdingside * optholding.holdingvol * opt_price * optholding.COption.multiplier
            else:
                for optcode, optholding in self.holdings.items():
                    if optholding.holdingside == holdingside:
                        opt_price = optholding.COption.quote_1min.ix[trading_datetime, 'close']
                        opt_mv += holdingside * optholding.holdingvol * opt_price * optholding.COption.multiplier
        self.holdingmv = opt_mv
        return opt_mv

    def p_and_l(self, trading_datetime):
        """
        计算给定时间点时的盈亏及组合净值
        :param trading_datetime: 计算时间，类型=datetime.datetime或者datetime.date
        :return:
        """
        self.pandl = self.holding_mv(trading_datetime) + self.cashinflow - self.cashoutflow - self.commission
        # self.nav = self.capital + self.pandl
        return self.pandl

    def net_asset_value(self, trading_datetime):
        """
        计算持仓净值
        :param trading_datetime: 计算时间，类型=datetime.datetime或者datetime.date
        :return:
        """
        self.nav = self.capital + self.p_and_l(trading_datetime)
        return self.nav

    def capital_available(self, trading_datetime):
        """
        计算可用资金
        :param tading_datetime: 计算时间，类型=datetime.datetime或者datetime.date
        :return: 可用资金，= nav - margin - 多头mv
        """
        return self.net_asset_value(trading_datetime) - self.total_margin() - self.holding_mv(trading_datetime, 1)
