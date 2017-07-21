import datetime
from util.util import COptTradeData
from util.COption import COption
from util.COptHolding import COptHolding
from configparser import ConfigParser
import pandas as pd


class CVolTradingStrategy(object):
    """期权波动率交易策略类"""
    def __init__(self, portname, configname):
        """
        策略初始化
        :param portname: 组合名称
        :param configname: 采用的配置项名称
        """
        self.portname = portname
        self.configname = configname
        self.marginratio = 0.0                  # 保证金占比上限
        self.marginratio_low = 0.0              # 保证金占比下限
        self.endday_deltathreshold = 0.0        # delta日终阈值比例
        self.endday_deltaadjratio = 0.0         # delta日终调整比例
        self.intraday_deltathreshold = 0.0      # delta盘中阈值比例
        self.intraday_deltaadjratio = 0.0       # delta盘中调整比例

        self.opts_data = {}                     # 期权基础数据（样本期权），字典类型,map<optcode,COption>
        self.underlying_quote_1min = None       # 期权标的1分钟行情数据,DataFrame数据
        # 无风险利率历史数据，1年期中国国债到期收益率，DataFrame数据（index='date',columns=['riskfree']）
        self.df_riskfree = None
        # 期权标的历史波动率，DateFrame数据（index='date',columns=['HV5','HV10','HV20','HV60']）
        self.df_HistVol = None
        self.commission_per_unit = 0.0          # 每张期权交易佣金
        self.underlying_price = 0.0             # 标的最新价格
        self.min_pandl_path = None              # 分钟P&L文件夹路径
        self.opt_holdings_path = None           # 持仓数据文件夹路径

        # 平仓相关参数
        self.liquidation_vols = None            # 每次平仓量dict<optcode, liquidationvol}

        # 导入相关数据及参数
        self.load_param()
        self.load_riskfree_hist()
        self.load_hist_vol()

        self.opt_holdings = COptHolding(self.opt_holdings_path + self.configname + '/log.txt')  # 期权的持仓

        # 设置日志
        # self.logger = None
        # self.set_logging()

    def load_param(self):
        """导入endday_adj_strategy的参数值"""
        cfg = ConfigParser()
        cfg.read('config.ini')
        self.marginratio = cfg.getfloat(self.configname, 'marginratio')
        self.marginratio_low = cfg.getfloat(self.configname, 'marginratio_low')
        self.endday_deltathreshold = cfg.getfloat(self.configname, 'endday_deltathreshold')
        self.endday_deltaadjratio = cfg.getfloat(self.configname, 'endday_deltaadjratio')
        self.intraday_deltathreshold = cfg.getfloat(self.configname, 'intraday_deltathreshold')
        self.intraday_deltaadjratio = cfg.getfloat(self.configname, 'intraday_deltaadjratio')
        self.commission_per_unit = cfg.getfloat('trade', 'commission')
        self.min_pandl_path = cfg.get('path', 'min_pandl_path')
        self.opt_holdings_path = cfg.get('path', 'opt_holdings_path')

    # def set_logging(self):
    #     """设置日志参数"""
    #     # 创建logger
    #     self.logger = logging.getLogger('OptVolTrading')
    #     # 设置level为DEBUG
    #     self.logger.setLevel(logging.DEBUG)
    #     # 创建一个handler，用于写入日志文件
    #     fh = logging.FileHandler('logger.log')
    #     # 创建一个handler，用于输出到控制台
    #     ch = logging.StreamHandler()
    #     # 定义handler的输出格式formatter
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     fh.setFormatter(formatter)
    #     ch.setFormatter(formatter)
    #     # 给logger添加handler
    #     self.logger.addHandler(fh)
    #     self.logger.addHandler(ch)
    #
    #     # 设置持仓的日志类
    #     self.opt_holdings.logger = logging.getLogger('OptVolTrading.holdings')
    #     self.opt_holdings.logger.setLevel(logging.DEBUG)
    #     self.opt_holdings.logger.addHandler(fh)
    #     self.opt_holdings.logger.addHandler(ch)

    def load_riskfree_hist(self):
        """导入无风险利率历史数据"""
        self.df_riskfree = pd.read_csv('./data/riskfree.csv', index_col=0, parse_dates=[0])
        self.df_riskfree['riskfree'] = self.df_riskfree['riskfree']/100.0

    def load_hist_vol(self):
        """导入标的历史波动率数据"""
        self.df_HistVol = pd.read_csv('./data/Historical_Vol.csv', index_col=0, parse_dates=[0])

    def load_opt_basic_data(self, trading_day):
        """
        导入期权基本信息数据
        导入当月期权合约信息，如果trading_day为当月期权合约的最后交易日，那么导入次月合约
        :param trading_day:日期（类型=datetime.date） 
        :return: 如果导入成功=True，如果导入失败=False
        """
        self.opts_data = {}
        header_names = ['opt_code', 'trade_code', 'opt_name', 'underlying_code', 'secu_type', 'opt_type', 'exercise_type',
                        'strike', 'multiplier', 'end_month', 'listed_date', 'expire_date', 'exercise_date', 'delivery_date']
        opts_basics = pd.read_csv('./data/OptBasics.csv', usecols=range(14), parse_dates=[10, 11, 12, 13],  dtype={'期权代码':str})
        opts_basics.columns = header_names
        # opts_basics['opt_code'] = opts_basics['opt_code'].astype('str') # 将'期权代码'字段转化为字符串类型
        opts_basics.set_index(keys='opt_code', inplace=True)
        # strdate = trading_day.strftime('%Y-%m-%d')
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
                    opt_type = "Put"
                if optdata['exercise_type'] == '欧式':
                    exercise_type = 'European'
                else:
                    exercise_type = 'American'
                # enddate = datetime.datetime.strptime(optdata['expire_date'], '%Y-%m-%d').date()
                enddate = optdata['expire_date'].to_pydatetime().date()
                self.opts_data[optcode] = COption(optcode, optdata['opt_name'], opt_type, exercise_type,
                                                  float(optdata['strike']), int(optdata['multiplier']), enddate)
        # # 如果当前日期为当月期权上市交易日期，那么把持仓状态改为'onposition'
        # if opts_basics.loc[0, 'listed_date'] == trading_day:
        #     self.opt_holdings.status = 'onposition'
        # 如果当前日期为当月合约最后交易日的前一天，那么把持仓状态改为‘onliquidation’
        if opts_basics.iloc[0, 10] - datetime.timedelta(days=1) == trading_day:
            self.opt_holdings.status = 'onliquidation'
            self.liquidation_vols = None
        return True

    def load_opt_holdings(self, trading_day):
        """
        导入策略的期权持仓数据
        :param trading_day:持仓日期（类型=datetime.date） 
        :return: 
        """
        strdate = trading_day.strftime("%Y%m%d")
        holding_filename = self.opt_holdings_path + self.configname + '/holding_' + self.portname + '_' + strdate + '.txt'
        self.opt_holdings.load_holdings(holding_filename)
        # 同时将持仓加入self.opts_data
        for optcode, holding in self.opt_holdings.holdings.items():
            if optcode not in self.opts_data:
                self.opts_data[optcode] = COption(optcode)

    def calc_opt_greeks(self, trading_datetime):
        """
        计算期权基础数据中的希腊字母值，标的价格有变化时调用本方法
        :param trading_datetime: 
        :return: 
        """
        underlying_price = float(self.underlying_quote_1min.ix[trading_datetime, 'close'])
        riskfree = float(self.df_riskfree.ix[trading_datetime.date(), 'riskfree'])
        vol = float(self.df_HistVol.ix[trading_datetime.date(), 'HV60'])
        for optcode, opt in self.opts_data.items():
            opt.calc_greeks(underlying_price, riskfree, 0.0, vol, trading_datetime)

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
            strfilepath = '../opt_quote/' + strdate + '/' +  optcode + '.csv'
            holding.COption.quote_1min = pd.read_csv(strfilepath, usecols=range(7), index_col=0, parse_dates=[0])

    def load_underlying_1min_quote(self, trading_day):
        """
        导入期权标的证券的1分钟行情数据
        :param trading_day: 日期（类型=datetime.date）
        :return: 
        """
        strdate = trading_day.strftime('%Y-%m-%d')
        strfilepath = '../opt_quote/' + strdate + '/510050ETF.csv'
        self.underlying_quote_1min = pd.read_csv(strfilepath, usecols=range(7), index_col=0, parse_dates=[0])

    def calc_opt_margin(self, trading_day):
        """
        计算样本期权和持仓期权的开仓保证金,每个交易日开盘前计算一次
        :param trading_day: 日期（类型=datetime.date）
        :return: 
        """
        # 1.读取标的日K线时间序列
        underlying_quote = pd.read_csv('./data/underlying_daily_quote.csv', index_col=0, parse_dates=[0])
        underlying_pre_close = float(underlying_quote.ix[trading_day, 'pre_close'])
        # 2.读取样本期权的日行情
        strdate = trading_day.strftime('%Y-%m-%d')
        strfilepath = '../opt_quote/' + strdate + '/50OptionDailyQuote.csv'
        opts_quote = pd.read_csv(strfilepath, usecols=range(1, 14), parse_dates=[0], encoding='gb18030', dtype={'option_code':str})
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

    def get_atm_opts(self, underlying_price):
        """
        取得平价认购、认沽期权
        :param underlying_price:标的最新价格 
        :return: tuple(COption, COption)，分别为认购、认沽期权类
        """
        if underlying_price <= 3.0:
            fspan = 0.05
        else:
            fspan = 0.1
        # 遍历样本期权，选取平价认购、认沽期权
        atm_call = None
        atm_put = None
        for optcode, copt in self.opts_data.items():
            # 如果已经找到平值认购、认沽期权，则跳出
            if (atm_call is not None) and (atm_put is not None):
                break
            if abs(copt.strike - underlying_price) <= fspan/2.0:
                if copt.opt_type == 'Call':
                    atm_call = copt
                else:
                    atm_put = copt
        if (atm_call is None) or (atm_put is None):
            atm_call, atm_put = self.get_near_atm_opts(underlying_price)
        return atm_call, atm_put

    def get_near_atm_opts(self, underlying_price):
        """
        取得最接近平值的认购、认沽期权，本方法在标的大幅波动、当前挂牌期权已无平价期权时调用
        :param underlying_price:  标的最新价格
        :return: tuple(COption, COption)，分别为认购、认沽期权类
        """
        # 遍历样本期权，选取abs(strike - underlying_price)最小的认购、认沽期权
        opt_call = None
        opt_put = None
        for optcode, copt in self.opts_data.items():
            if opt_call is None:
                opt_call = copt
            if opt_put is None:
                opt_put = copt
            if copt.opt_type == 'Call' and abs(copt.strike - underlying_price) < abs(opt_call.strike - underlying_price):
                opt_call = copt
            if copt.opt_type == 'Put' and abs(copt.strike - underlying_price) < abs(opt_put.strike - underlying_price):
                opt_put = copt
        return opt_call, opt_put

    def do_position(self, trading_datetime, margin, is_mkt_neutral=True):
        """
        进行建仓操作
        :param trading_datetime:交易时间（类型=datetime.datetime） 
        :param margin:用于建仓的保证金
        :param is_mkt_neutral:是否市场中性建仓，默认为True；如果为False，那么认购、认沽数量相等
        :return: 
        """
        # 1.取得平值认购、认沽期权
        atm_call, atm_put = self.get_atm_opts(self.underlying_price)
        # 2.计算所需卖出认购、认沽期权的数量
        if is_mkt_neutral:
            call_vol = int(round(-atm_put.greeks.delta * margin / (atm_call.greeks.delta * atm_put.margin -
                                                                   atm_put.greeks.delta * atm_call.margin), 0))
            put_vol = int(round(atm_call.greeks.delta * margin / (atm_call.greeks.delta * atm_put.margin -
                                                                  atm_put.greeks.delta * atm_call.margin), 0))
        else:
            call_vol = put_vol = int(round(margin / (atm_call.margin + atm_put.margin), 0))
        # if int(self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'volume'] * 0.2) < call_vol:
        #     put_vol = int(round(int(self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'volume'] * 0.2) /
        #                         call_vol * put_vol, 0))
        #     call_vol = int(self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'volume'] * 0.2)
        # if int(self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'volume'] * 0.2) < put_vol:
        #     call_vol = int(round(int(self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'volume'] * 0.2) /
        #                          put_vol * call_vol, 0))
        #     put_vol = int(self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'volume'] * 0.2)

        if self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'volume'] < call_vol:
            put_vol = max(int(round(self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'volume'] / call_vol * put_vol, 0)), 1)
            call_vol = max(self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'volume'], 1)
        if self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'volume'] < put_vol:
            call_vol = max(int(round(self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'volume'] / put_vol * call_vol, 0)), 1)
            put_vol = max(self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'volume'], 1)

        # 3.生成交易清单
        trade_datas = []
        call_trade_price = self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'close']
        trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', call_trade_price, call_vol,
                                         call_trade_price * call_vol * atm_call.multiplier, 0.0,
                                         trading_datetime, atm_call))
        put_trade_price = self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'close']
        trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', put_trade_price, put_vol,
                                         put_trade_price * put_vol * atm_put.multiplier, 0.0,
                                         trading_datetime, atm_put))
        # 4.更新持仓数据
        self.opt_holdings.update_holdings(trade_datas)

    def do_liquidation(self, trading_datetime):
        """
        进行平仓交易
        :param trading_datetime: 交易时间，类型=datetime.datetime
        :return:

        平仓方法：
        (1)将期权持仓分成20份，每分钟平仓一份，20分钟平仓完成
        """
        # 如果持仓已清空，直接退出
        if len(self.opt_holdings.holdings) == 0:
            return
        # 如果是第一次进入平仓操作，先计算每次平仓量
        if self.liquidation_vols is None:
            self.liquidation_vols = {}
            for optcode, holding in self.opt_holdings.holdings.items():
                if holding.holdingvol > 0 and optcode not in self.liquidation_vols:
                    self.liquidation_vols[optcode] = int(round(holding.holdingvol / 20.0, 0))
        # 根据设定的每一次平仓数据，进行平仓操作，如果是当天最后一分钟，那么平仓剩余的持仓量
        trade_datas = []
        for optcode, liquidvol in self.liquidation_vols.items():
            if (optcode in self.opt_holdings.holdings) and (self.opt_holdings.holdings[optcode].holdingvol > 0):
                trade_price = self.opts_data[optcode].quote_1min.ix[trading_datetime, 'close']
                if trading_datetime.time() >= datetime.time(14, 59, 0):
                    trade_vol = self.opt_holdings.holdings[optcode].holdingvol
                else:
                    if liquidvol <= self.opt_holdings.holdings[optcode].holdingvol:
                        trade_vol = liquidvol
                    else:
                        trade_vol = self.opt_holdings.holdings[optcode].holdingvol
                trade_datas.append(COptTradeData(optcode, 'buy', 'close', trade_price, trade_vol,
                                                 trade_price * trade_vol * self.opts_data[optcode].multiplier,
                                                 trade_vol * self.commission_per_unit,
                                                 trading_datetime, self.opts_data[optcode]))
        # 更新持仓数据
        self.opt_holdings.update_holdings(trade_datas)

    def is_fully_invested(self, calc_type):
        """
        组合是否已经达到了满仓
        :param calc_type: 计算类型，'margin'=逐步加仓模式下的计算保证金是否达到上限；
                         'gamma'=在gamma恒定模式下计算gamma暴露是否达到了上限或者保证金是否达到上限
        :return:
        """
        if calc_type == 'margin':
            if self.opt_holdings.margin_ratio() < self.marginratio - 0.01:
                return False
            else:
                return True
        elif calc_type == 'gamma':
            if ((abs(self.opt_holdings.greeks.gamma_mv) > abs(self.opt_holdings.gammaexposure)) or
                    (self.opt_holdings.margin_ratio() > self.marginratio - 0.01)):
                return True
            else:
                return False

    def calc_opt_adjvol(self, delta_adj_amount, short_opt, cover_opt):
        """
        计算“开新仓、平旧仓”中的开新仓数量和平旧仓数量
        :param delta_adj_amount: 需要调整的delta值(delta市值)
        :param short_opt: 卖开新仓的期权
        :param cover_opt: 平旧仓的期权
        :return: 开新仓期权的数量和平旧仓期权的数量
        """
        short_vol = int(round(cover_opt.margin * delta_adj_amount / (short_opt.margin * cover_opt.greeks.delta_mv -
                                                                     cover_opt.margin * short_opt.greeks.delta_mv), 0))
        cover_vol = int(round(short_opt.margin * delta_adj_amount / (short_opt.margin * cover_opt.greeks.delta_mv -
                                                                     cover_opt.margin * short_opt.greeks.delta_mv), 0))
        return short_vol, cover_vol

    def adj_delta(self, delta_adj_amount, trading_datetime):
        """
        调整持仓的delta值，调整的delta额度由delta_adj_amount确定
        :param delta_adj_amount:需要调整的delta额度(delta市值)
        :param trading_datetime:交易时间（类型=datetime.datetime）
        :return:

        算法：
        (1)如果还未达到保证金上限，卖开加仓，加仓平值期权
           a.如果delta调整值小于0，卖开call
           b.如果delta调整值大于0，卖开put
        (2)如果已达到保证金上限，卖开新仓、买平旧仓，开仓平值期权、平仓时间价值最小期权
           a.如果delta调整值小于0，卖开call、买平put
           b.如果delta调整值大于0，卖开put、买平call
        """
        # 1.取得平值认购、认沽期权
        atm_call, atm_put = self.get_atm_opts(self.underlying_price)
        trade_datas = []
        # 2.生成交易清单
        if not self.is_fully_invested('gamma'):     # 如果还未达到保证金上限，卖开加仓
            if delta_adj_amount < 0:    # delta调整值小于0，卖开call
                call_vol = int(round(abs(delta_adj_amount / atm_call.greeks.delta_mv), 0))
                call_trade_price = self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'close']
                trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', call_trade_price, call_vol,
                                                 call_trade_price * call_vol * atm_call.multiplier, 0.0,
                                                 trading_datetime, atm_call))
            else:                       # delta调整值大于0，卖开put
                put_vol = int(round(abs(delta_adj_amount / atm_put.greeks.delta_mv), 0))
                put_trade_price = self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'close']
                trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', put_trade_price, put_vol,
                                                 put_trade_price * put_vol * atm_put.multiplier, 0.0,
                                                 trading_datetime, atm_put))
        else:   # 如果已经达到保证金上限，卖开新仓、买平旧仓
            exclusions = []     # 需要排除的代码列表
            # 取得持仓中时间价值最低的认购、认沽期权
            opt_call_close, opt_put_close = self.opt_holdings.get_least_timevalue_opts(self.underlying_price,
                                                                                       trading_datetime, exclusions)
            if delta_adj_amount < 0:    # delta调整至小于0，卖开call、买平put
                # 计算开平仓数量
                call_vol, put_vol = self.calc_opt_adjvol(delta_adj_amount, atm_call, opt_put_close)
                delta_remain = delta_adj_amount
                # 如果计算出来的认沽期权平仓量大于其持仓量，那么先根据保证金相等原则更新交易量，然后再选取时间价值次小的认沽期权进行交易
                while put_vol > self.opt_holdings.holdings[opt_put_close.code].holdingvol:
                    put_vol = self.opt_holdings.holdings[opt_put_close.code].holdingvol
                    call_vol = int(round(opt_put_close.margin * put_vol / atm_call.margin, 0))
                    call_trade_price = self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'close']
                    put_trade_price = self.opts_data[opt_put_close.code].quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', call_trade_price, call_vol,
                                                     call_trade_price * call_vol * atm_call.multiplier, 0.0,
                                                     trading_datetime, atm_call))
                    trade_datas.append(COptTradeData(opt_put_close.code, 'buy', 'close', put_trade_price, put_vol,
                                                     put_trade_price * put_vol * opt_put_close.multiplier,
                                                     put_vol * self.commission_per_unit, trading_datetime,
                                                     opt_put_close))
                    delta_remain -= (-call_vol * atm_call.greeks.delta_mv + put_vol * opt_put_close.greeks.delta_mv)
                    if abs(delta_remain) < abs(delta_adj_amount * 0.05):
                        call_vol = put_vol = 0
                        break
                    else:
                        exclusions.append(opt_put_close.code)
                        _, opt_put_close = self.opt_holdings.get_least_timevalue_opts(self.underlying_price,
                                                                                      trading_datetime, exclusions)
                        call_vol, put_vol = self.calc_opt_adjvol(delta_remain, atm_call, opt_put_close)
                if call_vol > 0:
                    call_trade_price = self.opts_data[atm_call.code].quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', call_trade_price, call_vol,
                                                     call_trade_price * call_vol * atm_call.multiplier, 0.0,
                                                     trading_datetime, atm_call))
                if put_vol > 0:
                    put_trade_price = self.opts_data[opt_put_close.code].quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(opt_put_close.code, 'buy', 'close', put_trade_price, put_vol,
                                                     put_trade_price * put_vol * opt_put_close.multiplier,
                                                     put_vol * self.commission_per_unit, trading_datetime,
                                                     opt_put_close))
            else:                       # delta调整值大于0，卖开put、买平call
                # 计算开平仓数量
                put_vol, call_vol = self.calc_opt_adjvol(delta_adj_amount, atm_put, opt_call_close)
                delta_remain = delta_adj_amount
                # 如果计算出来的认购期权平仓量大于其持仓量，那么先根据保证金相等原则更新交易量，然后再选取时间价值次小的认购期权进行交易
                while call_vol > self.opt_holdings.holdings[opt_call_close.code].holdingvol:
                    call_vol = self.opt_holdings.holdings[opt_call_close.code].holdingvol
                    put_vol = int(round(opt_call_close.margin * call_vol / atm_put.margin, 0))
                    call_trade_price = self.opts_data[opt_call_close.code].quote_1min.ix[trading_datetime, 'close']
                    put_trade_price = self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(opt_call_close.code, 'buy', 'close', call_trade_price, call_vol,
                                                     call_trade_price * call_vol * opt_call_close.multiplier,
                                                     call_vol * self.commission_per_unit, trading_datetime,
                                                     opt_call_close))
                    trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', put_trade_price, put_vol,
                                                     put_trade_price * put_vol * atm_put.multiplier, 0.0,
                                                     trading_datetime, atm_put))
                    delta_remain -= (-put_vol * atm_put.greeks.delta_mv + call_vol * opt_call_close.greeks.delta_mv)
                    if abs(delta_remain) < delta_adj_amount * 0.05:
                        call_vol = put_vol = 0
                        break
                    else:
                        exclusions.append(opt_call_close.code)
                        opt_call_close, _ = self.opt_holdings.get_least_timevalue_opts(self.underlying_price,
                                                                                       trading_datetime, exclusions)
                        put_vol, call_vol = self.calc_opt_adjvol(delta_remain, atm_put, opt_call_close)
                if call_vol > 0:
                    call_trade_price = self.opts_data[opt_call_close.code].quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(opt_call_close.code, 'buy', 'close', call_trade_price, call_vol,
                                                     call_trade_price * call_vol * opt_call_close.multiplier,
                                                     call_vol * self.commission_per_unit, trading_datetime,
                                                     opt_call_close))
                if put_vol > 0:
                    put_trade_price = self.opts_data[atm_put.code].quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', put_trade_price, put_vol,
                                                     put_trade_price * put_vol * atm_put.multiplier,
                                                     0.0, trading_datetime, atm_put))
        # 3.更新持仓
        self.opt_holdings.update_holdings(trade_datas)

    def adj_gamma(self, gamma_adj_amount, trading_datetime):
        """
        调整持仓的gamma值，调整gamma的额度由gamma_adj_amount确定
        :param gamma_adj_amount:需要调整的gamma额度(1%gamma市值)
        :param trading_datetime:交易时间（类型=datetime.datetime）
        :return:

        算法:
        (一)上调gamma值，即gamma_adj_amount < 0
           1.如果保证金已达到上限：加仓平值认购、认沽期权，平仓gamma最小(gamma相等，取时间价值较小)的认购或认沽期权，并保持delta中性，
             解以下三元一次方程，得加仓的认购、认沽期权数量，和平仓的期权数量
             (1) -delta_{atm,c} * Vol_{atm,c} - delta_{atm,p} * Vol_{atm,p} = -delta_{close} * Vol_{close}, delta中性
             (2) -Γ_{atm,c} * Vol_{atm,c} - Γ_{atm,p} * Vol_{atm,p} + Γ_{close} * Vol_{close} = gamma_adj_amount
             (3) Margin_{atm,c} * Vol_{atm,c} + margin_{atm,p} * Vol_{atm,p} = margin_{close} * Vol_{close}, 保证金不增加
           2.如果保证金未达到上限：加仓平值认购、认沽期权，保持delta中性。解以下二元一次方程，得加仓的认购、认沽期权数量:
            (1) -delta_{atm,c} * Vol_{atm,c} = delta_{atm,p} * Vol_{atm,p}, delta中性
            (2) -Γ_{atm,c} * Vol_{atm,c} - Γ_{atm,p} * Vol_{atm,p} = gamma_adj_amount
        (二)下调gamma值，即gamma_adj_amount > 0
            平仓持仓中gamma值最大的认购、认沽期权，保持delta中性。解以下二元一次方程，得平仓的认购、认沽期权数量：
            (1) delta_{atm,c} * Vol_{atm,c} = -delta_{atm,p} * Vol_{atm,p}, delta中性
            (2) Γ_{atm,c} * Vol_{atm,c} + Γ_{atm,p} * Vol_{atm,p} = gamma_adj_amount
        """
        # 1.取得持仓中gamma值最大的认购、认沽期权及gamma值最小的认购、认沽期权
        exclusions = []
        # dict_minmax_gamma_opts = self.opt_holdings.get_minmax_gamma_opts(exclusions)
        trade_datas = []
        # 2.生成交易清单
        # 2.1 如果需要调整的gamma值小于0，即增大gamma暴露
        if gamma_adj_amount < 0:
            # 取得平值认购、认沽期权
            atm_call, atm_put = self.get_atm_opts(self.underlying_price)
            # 2.1.1 如果保证金已达到上限，那么加仓平值认购、认沽期权，同时平仓gamma最小(gamma相等，取时间价值较小)的认购或认沽期权
            #       同时保持delta中性
            if self.is_fully_invested('margin'):
                exclusions.append(atm_call.code)
                exclusions.append(atm_put.code)
                dict_minmax_gamma_opts = self.opt_holdings.get_minmax_gamma_opts(exclusions)
                gamma_remain = gamma_adj_amount
                if (dict_minmax_gamma_opts['min'][0] is None) and (dict_minmax_gamma_opts['min'][1] is None):
                    return
                elif dict_minmax_gamma_opts['min'][0] is None:
                    opt_close = dict_minmax_gamma_opts['min'][1]
                elif dict_minmax_gamma_opts['min'][1] is None:
                    opt_close = dict_minmax_gamma_opts['min'][0]
                else:
                    if dict_minmax_gamma_opts['min'][0].greeks.gamma < dict_minmax_gamma_opts['min'][1].greeks.gamma:
                        opt_close = dict_minmax_gamma_opts['min'][0]
                    elif dict_minmax_gamma_opts['min'][0].greeks.gamma > dict_minmax_gamma_opts['min'][1].greeks.gamma:
                        opt_close = dict_minmax_gamma_opts['min'][1]
                    else:
                        if dict_minmax_gamma_opts['min'][0].time_value(self.underlying_price, trading_datetime) < \
                                dict_minmax_gamma_opts['min'][1].time_value(self.underlying_price, trading_datetime):
                            opt_close = dict_minmax_gamma_opts['min'][0]
                        else:
                            opt_close = dict_minmax_gamma_opts['min'][1]

                # while opt_close == atm_call or opt_close == atm_put:
                #     exclusions.append(opt_close.code)
                #     dict_minmax_gamma_opts = self.opt_holdings.get_minmax_gamma_opts(exclusions)
                #     if dict_minmax_gamma_opts['min'][0] is None:
                #         opt_close = dict_minmax_gamma_opts['min'][1]
                #     elif dict_minmax_gamma_opts['min'][1] is None:
                #         opt_close = dict_minmax_gamma_opts['min'][0]
                #     else:
                #         if dict_minmax_gamma_opts['min'][0].greeks.gamma < dict_minmax_gamma_opts['min'][1].greeks.gamma:
                #             opt_close = dict_minmax_gamma_opts['min'][0]
                #         elif dict_minmax_gamma_opts['min'][0].greeks.gamma > dict_minmax_gamma_opts['min'][1].greeks.gamma:
                #             opt_close = dict_minmax_gamma_opts['min'][1]
                #         else:
                #             if dict_minmax_gamma_opts['min'][0].time_value(self.underlying_price, trading_datetime) < \
                #                     dict_minmax_gamma_opts['min'][1].time_value(self.underlying_price, trading_datetime):
                #                 opt_close = dict_minmax_gamma_opts['min'][0]
                #             else:
                #                 opt_close = dict_minmax_gamma_opts['min'][1]

                a = atm_put.greeks.delta_mv * opt_close.margin - opt_close.greeks.delta_mv * atm_put.margin
                b = opt_close.greeks.delta_mv * atm_call.margin - atm_call.greeks.delta_mv * opt_close.margin
                c = opt_close.greeks.gamma_mv * atm_put.greeks.delta_mv - \
                    atm_put.greeks.gamma_mv * opt_close.greeks.delta_mv
                d = opt_close.greeks.gamma_mv * atm_call.greeks.delta_mv - \
                    atm_call.greeks.gamma_mv * opt_close.greeks.delta_mv
                # vol_atmcall = int(round(gamma_remain * opt_close.greeks.delta_mv * a / (b * c + a * d), 0))
                # vol_atmput = int(round((gamma_remain * opt_close.greeks.delta_mv - d * vol_atmcall) / c, 0))
                # vol_close = int(round((atm_call.greeks.delta_mv * vol_atmcall + atm_put.greeks.delta_mv * vol_atmput) /
                #                       opt_close.greeks.delta_mv, 0))
                vol_atmcall = gamma_remain * opt_close.greeks.delta_mv * a / (b * c + a * d)
                vol_atmput = (gamma_remain * opt_close.greeks.delta_mv - d * vol_atmcall) / c
                vol_close = (atm_call.greeks.delta_mv * vol_atmcall + atm_put.greeks.delta_mv * vol_atmput) / opt_close.greeks.delta_mv
                vol_atmcall = int(round(vol_atmcall, 0))
                vol_atmput = int(round(vol_atmput, 0))
                vol_close = int(round(vol_close, 0))
                # 如果计算出来的平仓期权的平仓数量大于其持仓数量，那么根据该期权持仓量更新交易量，然后再选择gamma次小的持仓期权进行平仓
                while vol_close > self.opt_holdings.holdings[opt_close.code].holdingvol:
                    vol_close = self.opt_holdings.holdings[opt_close.code].holdingvol
                    vol_atmcall = (opt_close.margin * vol_close * atm_put.greeks.delta_mv -
                                   atm_put.margin * opt_close.greeks.delta_mv * vol_close) / \
                                  (atm_call.margin * atm_put.greeks.delta_mv - atm_put.margin * atm_call.greeks.delta_mv)
                    vol_atmput = (opt_close.greeks.delta_mv * vol_close - atm_call.greeks.delta_mv * vol_atmcall) / atm_put.greeks.delta_mv
                    # vol_close = int(round(vol_close, 0))
                    vol_atmcall = int(round(vol_atmcall, 0))
                    vol_atmput = int(round(vol_atmput))
                    closeopt_trade_price = opt_close.quote_1min.ix[trading_datetime, 'close']
                    atmcall_trade_price = atm_call.quote_1min.ix[trading_datetime, 'close']
                    atmput_trade_price = atm_put.quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(opt_close.code, 'buy', 'close', closeopt_trade_price, vol_close,
                                                     closeopt_trade_price * vol_close * opt_close.multiplier,
                                                     vol_close * self.commission_per_unit, trading_datetime, opt_close))
                    trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', atmcall_trade_price, vol_atmcall,
                                                     atmcall_trade_price * vol_atmcall * atm_call.multiplier, 0.0,
                                                     trading_datetime, atm_call))
                    trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', atmput_trade_price, vol_atmput,
                                                     atmput_trade_price * vol_atmput * atm_put.multiplier, 0.0,
                                                     trading_datetime, atm_put))
                    gamma_remain -= (opt_close.greeks.gamma_mv * vol_close - atm_call.greeks.gamma_mv * vol_atmcall -
                                     atm_put.greeks.gamma_mv * vol_atmput)
                    if abs(gamma_remain) < abs(gamma_adj_amount * 0.05):
                        vol_close = vol_atmcall = vol_atmput = 0
                        break
                    else:
                        exclusions.append(opt_close.code)
                        dict_minmax_gamma_opts = self.opt_holdings.get_minmax_gamma_opts(exclusions)
                        if (dict_minmax_gamma_opts['min'][0] is None) and (dict_minmax_gamma_opts['min'][1] is None):
                            return
                        elif dict_minmax_gamma_opts['min'][0] is None:
                            opt_close = dict_minmax_gamma_opts['min'][1]
                        elif dict_minmax_gamma_opts['min'][1] is None:
                            opt_close = dict_minmax_gamma_opts['min'][0]
                        else:
                            if dict_minmax_gamma_opts['min'][0].greeks.gamma < dict_minmax_gamma_opts['min'][1].greeks.gamma:
                                opt_close = dict_minmax_gamma_opts['min'][0]
                            elif dict_minmax_gamma_opts['min'][0].greeks.gamma > dict_minmax_gamma_opts['min'][1].greeks.gamma:
                                opt_close = dict_minmax_gamma_opts['min'][1]
                            else:
                                if dict_minmax_gamma_opts['min'][0].time_value(self.underlying_price, trading_datetime) < \
                                        dict_minmax_gamma_opts['min'][1].time_value(self.underlying_price,
                                                                                    trading_datetime):
                                    opt_close = dict_minmax_gamma_opts['min'][0]
                                else:
                                    opt_close = dict_minmax_gamma_opts['min'][1]
                        a = atm_put.greeks.delta_mv * opt_close.margin - opt_close.greeks.delta_mv * atm_put.margin
                        b = opt_close.greeks.delta_mv * atm_call.margin - atm_call.greeks.delta_mv * opt_close.margin
                        c = opt_close.greeks.gamma_mv * atm_put.greeks.delta_mv - \
                            atm_put.greeks.gamma_mv * opt_close.greeks.delta_mv
                        d = opt_close.greeks.gamma_mv * atm_call.greeks.delta_mv - \
                            atm_call.greeks.gamma_mv * opt_close.greeks.delta_mv
                        vol_atmcall = gamma_remain * opt_close.greeks.delta_mv * a / (b * c + a * d)
                        vol_atmput = (gamma_remain * opt_close.greeks.delta_mv - d * vol_atmcall) / c
                        vol_close = (atm_call.greeks.delta_mv * vol_atmcall + atm_put.greeks.delta_mv * vol_atmput) / opt_close.greeks.delta_mv
                        vol_atmcall = int(round(vol_atmcall, 0))
                        vol_atmput = int(round(vol_atmput, 0))
                        vol_close = int(round(vol_close, 0))
                if vol_atmcall > 0:
                    atmcall_trade_price = atm_call.quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', atmcall_trade_price, vol_atmcall,
                                                     atmcall_trade_price * vol_atmcall * atm_call.multiplier, 0.0,
                                                     trading_datetime, atm_call))
                if vol_atmput > 0:
                    atmput_trade_price = atm_put.quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', atmput_trade_price, vol_atmput,
                                                     atmput_trade_price * vol_atmput * atm_put.multiplier, 0.0,
                                                     trading_datetime, atm_put))
                if vol_close > 0:
                    closeopt_trade_price = opt_close.quote_1min.ix[trading_datetime, 'close']
                    trade_datas.append(COptTradeData(opt_close.code, 'buy', 'close', closeopt_trade_price, vol_close,
                                                     closeopt_trade_price * vol_close * opt_close.multiplier,
                                                     vol_close * self.commission_per_unit, trading_datetime, opt_close))
            # 2.1.2 如果保证金未达到上限，那么加仓平值认购、认沽期权，保持delta中性
            else:
                denominator = (atm_put.greeks.gamma_mv * atm_call.greeks.delta_mv -
                               atm_call.greeks.gamma_mv * atm_put.greeks.delta_mv)
                vol_atmcall = int(round(gamma_adj_amount * atm_put.greeks.delta_mv / denominator, 0))
                vol_atmput = int(round(-gamma_adj_amount * atm_call.greeks.delta_mv / denominator, 0))
                atmcall_trade_price = atm_call.quote_1min.ix[trading_datetime, 'close']
                atmput_trade_price = atm_put.quote_1min.ix[trading_datetime, 'close']
                trade_datas.append(COptTradeData(atm_call.code, 'sell', 'open', atmcall_trade_price, vol_atmcall,
                                                 atmcall_trade_price * vol_atmcall * atm_call.multiplier, 0.0,
                                                 trading_datetime, atm_call))
                trade_datas.append(COptTradeData(atm_put.code, 'sell', 'open', atmput_trade_price, vol_atmput,
                                                 atmput_trade_price * vol_atmput * atm_put.multiplier, 0.0,
                                                 trading_datetime, atm_put))
        # 2.2 如果需要调整的gamma值大于0，即缩小gamma暴露。平仓持仓中gamma值最大的认购、认沽期权，保持delta中性
        #     但如果当前持仓保证金占比小于保证金占比下限，则不缩小gamma暴露
        else:
            if self.opt_holdings.margin_ratio() < self.marginratio_low:
                return
            dict_minmax_gamma_opts = self.opt_holdings.get_minmax_gamma_opts(exclusions)
            gamma_remain = gamma_adj_amount
            close_call = dict_minmax_gamma_opts['max'][0]
            close_put = dict_minmax_gamma_opts['max'][1]
            close_call_vol = (gamma_adj_amount * close_put.greeks.delta_mv /
                              (close_call.greeks.gamma_mv * close_put.greeks.delta_mv -
                               close_put.greeks.gamma_mv * close_call.greeks.delta_mv))
            close_put_vol = (-gamma_adj_amount * close_call.greeks.delta_mv /
                             (close_call.greeks.gamma_mv * close_put.greeks.delta_mv -
                              close_put.greeks.gamma_mv * close_call.greeks.delta_mv))
            close_call_vol = int(round(close_call_vol))
            close_put_vol = int(round(close_put_vol))
            # 如果计算出来的认购或认沽期权平仓量大于其持仓量，那么更新平仓量，并寻找持仓中gamma次大的认购、认沽期权进行平仓
            while True:
                if close_call_vol > self.opt_holdings.holdings[close_call.code].holdingvol:
                    close_call_vol = self.opt_holdings.holdings[close_call.code].holdingvol
                    exclusions.append(close_call.code)
                    if close_put_vol <= self.opt_holdings.holdings[close_put.code].holdingvol:
                        close_put_vol = int(round(-close_call.greeks.delta_mv * close_call_vol /
                                                  close_put.greeks.delta_mv, 0))
                    else:
                        close_put_vol = self.opt_holdings.holdings[close_put.code].holdingvol
                        exclusions.append(close_put.code)
                else:
                    if close_put_vol <= self.opt_holdings.holdings[close_put.code].holdingvol:
                        break
                    else:
                        close_put_vol = self.opt_holdings.holdings[close_put.code].holdingvol
                        close_call_vol = int(round(-close_put.greeks.delta_mv * close_put_vol /
                                                   close_call.greeks.delta_mv, 0))
                        exclusions.append(close_put.code)
                call_trade_price = close_call.quote_1min.ix[trading_datetime, 'close']
                put_trade_price = close_put.quote_1min.ix[trading_datetime, 'close']
                trade_datas.append(COptTradeData(close_call.code, 'buy', 'close', call_trade_price, close_call_vol,
                                                 call_trade_price * close_call_vol * close_call.multiplier,
                                                 close_call_vol * self.commission_per_unit, trading_datetime,
                                                 close_call))
                trade_datas.append(COptTradeData(close_put.code, 'buy', 'close', put_trade_price, close_put_vol,
                                                 put_trade_price * close_put_vol * close_put.multiplier,
                                                 close_put_vol * self.commission_per_unit, trading_datetime, close_put))
                gamma_remain -= (close_call.greeks.gamma_mv * close_call_vol + close_put.greeks.gamma_mv * close_put_vol)
                if abs(gamma_remain) < gamma_adj_amount * 0.05:
                    close_call_vol = close_put_vol = 0
                    break
                dict_minmax_gamma_opts = self.opt_holdings.get_minmax_gamma_opts(exclusions)
                close_call = dict_minmax_gamma_opts['max'][0]
                close_put = dict_minmax_gamma_opts['max'][1]
                close_call_vol = (gamma_remain * close_put.greeks.delta_mv /
                                  (close_call.greeks.gamma_mv * close_put.greeks.delta_mv -
                                   close_put.greeks.gamma_mv * close_call.greeks.delta_mv))
                close_put_vol = (gamma_remain * close_call.greeks.delta_mv /
                                 (close_call.greeks.gamma_mv * close_put.greeks.delta_mv -
                                  close_put.greeks.gamma_mv * close_call.greeks.delta_mv))
                close_call_vol = int(round(close_call_vol, 0))
                close_put_vol = int(round(close_put_vol, 0))
            if close_call_vol > 0:
                call_trade_price = close_call.quote_1min.ix[trading_datetime, 'close']
                trade_datas.append(COptTradeData(close_call.code, 'buy', 'close', call_trade_price, close_call_vol,
                                                 call_trade_price * close_call_vol * close_call.multiplier,
                                                 close_call_vol * self.commission_per_unit, trading_datetime,
                                                 close_call))
            if close_put_vol > 0:
                put_trade_price = close_put.quote_1min.ix[trading_datetime, 'close']
                trade_datas.append(COptTradeData(close_put.code, 'buy', 'close', put_trade_price, close_put_vol,
                                                 put_trade_price * close_put_vol * close_put.multiplier,
                                                 close_put_vol * self.commission_per_unit, trading_datetime, close_put))
        # 3.更新持仓
        self.opt_holdings.update_holdings(trade_datas)

    def dynamic_adj(self, trading_datetime):
        """
        动态调整
        :param trading_datetime: 交易时间，类型=datetime.datetime
        :return:
        """
        is_delta_adjusted = False
        # 1.delta调整
        # 如果盘中delta市值超限，进行调整
        # if trading_datetime.time() >= datetime.time(9, 30, 0) and trading_datetime.time() < datetime.time(14, 55, 0):
        if datetime.time(9, 30, 0) <= trading_datetime.time() < datetime.time(14, 55, 0):
            # 如果盘中$Delta绝对值大于gamma暴露值，那么进行delta调整
            if abs(self.opt_holdings.greeks.delta_mv) > abs(self.opt_holdings.gammaexposure * self.intraday_deltathreshold):
                if self.opt_holdings.greeks.delta_mv > 0:
                    delta_adj_amount = abs(self.opt_holdings.gammaexposure * self.intraday_deltaadjratio) - self.opt_holdings.greeks.delta_mv
                else:
                    delta_adj_amount = abs(self.opt_holdings.greeks.delta_mv) - abs(self.opt_holdings.gammaexposure * self.intraday_deltaadjratio)
                self.adj_delta(delta_adj_amount, trading_datetime)
                self.opt_holdings.calc_greeks(self.underlying_price,
                                              self.df_riskfree.ix[trading_datetime.date(), 'riskfree'], 0.0,
                                              self.df_HistVol.ix[trading_datetime.date(), 'HV60'], trading_datetime)
                is_delta_adjusted = True
        # 如果收盘前delta市值超限，进行调整
        # elif trading_datetime.time() >= datetime.time(14, 55, 0) and trading_datetime.time() < datetime.time(14, 56, 0):
        elif datetime.time(14, 55, 0) <= trading_datetime.time() < datetime.time(14, 56, 0):
            if abs(self.opt_holdings.greeks.delta_mv) > abs(self.opt_holdings.gammaexposure * self.endday_deltathreshold):
                if self.opt_holdings.greeks.delta_mv > 0:
                    delta_adj_amount = abs(self.opt_holdings.gammaexposure * self.endday_deltaadjratio) - self.opt_holdings.greeks.delta_mv
                else:
                    delta_adj_amount = abs(self.opt_holdings.greeks.delta_mv) - abs(self.opt_holdings.gammaexposure * self.endday_deltaadjratio)
                self.adj_delta(delta_adj_amount, trading_datetime)
                self.opt_holdings.calc_greeks(self.underlying_price,
                                              self.df_riskfree.ix[trading_datetime.date(), 'riskfree'], 0.0,
                                              self.df_HistVol.ix[trading_datetime.date(), 'HV60'], trading_datetime)
                is_delta_adjusted = True
        # 2.如果已经进行delta调整，那么检查gamma市值是否超限，若超限则调整
        if is_delta_adjusted:
            gamma_adj_amount = 0.0
            if self.opt_holdings.greeks.gamma_mv < self.opt_holdings.gammaexposure * 1.1:
                gamma_adj_amount = self.opt_holdings.gammaexposure * 1.1 - self.opt_holdings.greeks.gamma_mv
            elif self.opt_holdings.greeks.gamma_mv > self.opt_holdings.gammaexposure * 0.9:
                gamma_adj_amount = self.opt_holdings.gammaexposure * 0.9 - self.opt_holdings.greeks.gamma_mv
            if abs(gamma_adj_amount) > abs(self.opt_holdings.gammaexposure) * 0.01:
                self.adj_gamma(gamma_adj_amount, trading_datetime)
                self.opt_holdings.calc_greeks(self.underlying_price,
                                              self.df_riskfree.ix[trading_datetime.date(), 'riskfree'], 0.0,
                                              self.df_HistVol.ix[trading_datetime.date(), 'HV60'], trading_datetime)

    def on_vol_trading(self, trading_day, pre_trading_day):
        """
        指定某一交易日期，进行波动率交易
        :param trading_day: 交易日期，类型=datetime.date
        :param pre_trading_day: 前一交易日期，类型=datetime.date
        :return:
        """
        with open(self.opt_holdings_path + self.configname + '/log.txt', 'at') as f:
            f.write(trading_day.strftime('%Y-%m-%d') + '\n')
        print("%s of %s" % (trading_day.strftime('%Y-%m-%d'), self.configname))
        # 导入期权基本信息数据
        is_load_success = False
        is_load_success = self.load_opt_basic_data(trading_day)
        # 导入期权持仓数据
        self.load_opt_holdings(pre_trading_day)
        # 导入期权（含持仓期权）分钟行情数据
        self.load_opt_1min_quote(trading_day)
        # 导入标的分钟行情数据
        self.load_underlying_1min_quote(trading_day)
        # 计算期权（含持仓期权）开仓保证金
        self.calc_opt_margin(trading_day)
        # 读取当天的无风险利率、标的历史波动率
        frisk_free = float(self.df_riskfree.ix[trading_day, 'riskfree'])
        fhist_vol = float(self.df_HistVol.ix[trading_day, 'HV60'])
        if is_load_success:
            # 遍历标的分钟行情数据，进行波动率交易的动态调整
            for trading_datetime, quotedata in self.underlying_quote_1min.iterrows():
                # print(trading_datetime)
                if trading_datetime.time() < datetime.time(9, 30, 0):
                    continue
                if trading_datetime.time() > datetime.time(15, 0, 0):
                    break
                # 设置标的最新价格
                self.underlying_price = quotedata['close']
                # 计算样本期权的希腊字母值
                self.calc_opt_greeks(trading_datetime)
                # 计算持仓期权的希腊字母值
                self.opt_holdings.calc_greeks(self.underlying_price, frisk_free, 0.0, fhist_vol, trading_datetime)
                # 根据策略状态进行不同交易
                if self.opt_holdings.status == 'onposition':
                    if not self.is_fully_invested('margin'):
                        fmargin = self.opt_holdings.capital * self.marginratio - self.opt_holdings.total_margin()
                        self.do_position(trading_datetime, fmargin, True)
                    else:
                        self.opt_holdings.status = 'positioned'
                        self.opt_holdings.gammaexposure = self.opt_holdings.greeks.gamma_mv
                        self.dynamic_adj(trading_datetime)
                elif self.opt_holdings.status == 'positioned':
                    self.dynamic_adj(trading_datetime)
                elif self.opt_holdings.status == 'onliquidation' and trading_datetime.time() < datetime.time(14, 30, 0):
                    self.dynamic_adj(trading_datetime)
                elif self.opt_holdings.status == 'onliquidation' and trading_datetime.time() >= datetime.time(14, 30, 0):
                    self.do_liquidation(trading_datetime)
                else:
                    # self.logger.critical('持仓状态设置错误。')
                    with open(self.opt_holdings_path + self.configname + '/log.txt', 'at') as f:
                        f.write('holding status setting error.\n')
                # 每分钟结束，计算策略P&L、greeks，并保存
                self.opt_holdings.calc_greeks(self.underlying_price, frisk_free, 0.0, fhist_vol, trading_datetime)
                with open(self.min_pandl_path + self.configname + '/' + trading_datetime.strftime('%Y-%m-%d') + '.csv', 'at') as f:
                    f.write(trading_datetime.strftime('%H:%M:%S,'))
                    f.write('pandl=%f,' % self.opt_holdings.p_and_l(trading_datetime))
                    f.write('delta_mv=%f,' % self.opt_holdings.greeks.delta_mv)
                    f.write('gamma_mv=%f\n' % self.opt_holdings.greeks.gamma_mv)
        # 每个交易日结束保存持仓数据、P&L
        holding_filename = self.opt_holdings_path + self.configname + '/holding_' + self.portname + '_' + trading_day.strftime('%Y%m%d') + '.txt'
        self.opt_holdings.save_holdings(holding_filename)

    def on_vol_trading_interval(self, beg_date, end_date):
        """
        指定时间区间，进行波动率交易
        :param beg_date: 开始日期，类型=datetime.date
        :param end_date: 结束日期，类型=datetime.date
        :return:
        """
        # 读取交易日期数据
        df_tradingdays = pd.read_csv('./data/tradingdays.csv', parse_dates=[0, 1])
        df_tradingdays = df_tradingdays[(df_tradingdays.tradingday >= beg_date) & (df_tradingdays.tradingday <= end_date)]
        for _, tradingdays in df_tradingdays.iterrows():
            trading_day = tradingdays['tradingday']
            pre_trading_day = tradingdays['pre_tradingday']
            self.on_vol_trading(trading_day, pre_trading_day)

# 波动率交易入口
if __name__ == '__main__':
    vol_strategy = CVolTradingStrategy('VolTrade', 'endday_adj_strategy')
    tmbeg_date = datetime.date(2015, 2, 9)
    tmend_date = datetime.date(2017, 6, 27)
    vol_strategy.on_vol_trading_interval(tmbeg_date, tmend_date)
