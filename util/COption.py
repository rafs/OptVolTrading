import math
import datetime
from scipy.stats import norm
import csv
from util.util import CGreeks


class COption(object):
    """期权类"""

    def __init__(self, code, opt_name=None, opt_type=None, exercise_type=None, strike=None,
                 multiplier=None, end_date=None):
        """
        初始化期权类
        Parameters:
        --------
        code:期权代码
        opt_type:期权类型，Call, Put
        exercise_type:行权方式，European, American
        strike:行权价
        multiplier:合约单位
        end_date:到期日，datetime.date类
        """
        self.code = code
        self.name = opt_name
        self.opt_type = opt_type
        self.exercise_type = exercise_type
        self.strike = strike
        self.multiplier = multiplier
        self.end_date = end_date
        self.greeks = CGreeks()                 # 期权的希腊字母值
        self.margin = None                      # 期权的开仓保证金
        self.quote_1min = None                  # 期权的1分钟行情数据,DataFrame格式
        if opt_name is None:
            self.load_param()

    def calc_greeks(self, underlying_price, risk_free, dividend_rate, vol, calc_datetime=datetime.datetime.now()):
        """
        计算期权的希腊字母值
        Parameters:
        underlyingprice:标的最新价格
        risk_free:无风险利率，取最新的一年期shibor利率
        dividend_rate:股息率，=0
        vol:波动率，标的的60日历史波动率
        calc_datetime:计算greek的时间(datetime.datetime类)，默认为今天
        """
        tau = ((self.end_date - calc_datetime.date()).days + COption.time_remain_of_day(calc_datetime)) / 365.0
        d1 = (math.log(underlying_price / self.strike) +
              (risk_free - dividend_rate + vol * vol / 2.0) * tau) / vol / math.sqrt(tau)
        if self.opt_type.lower() == 'call':
            self.greeks.delta = math.exp(-dividend_rate * tau) * norm.cdf(d1)
        elif self.opt_type.lower() == 'put':
            self.greeks.delta = math.exp(-dividend_rate * tau) * (norm.cdf(d1) - 1.0)
        else:
            self.greeks.delta = 0.0

        self.greeks.gamma = (math.exp(-dividend_rate * tau - d1 * d1 / 2.0) / vol / underlying_price /
                             math.sqrt(2.0 * math.pi * tau))
        self.greeks.vega = (math.exp(-dividend_rate * tau - d1 * d1 / 2.0) * underlying_price *
                            math.sqrt(tau / 2.0 / math.pi) / 100.0)

        self.greeks.delta_mv = self.greeks.delta * self.multiplier * underlying_price
        self.greeks.gamma_mv = 0.01 * self.greeks.gamma * self.multiplier * underlying_price ** 2

    def load_param(self):
        """从期权基本合约资料.csv文件导入本期权的相应参数"""
        # with open('期权合约基本资料.csv', 'rb', newline='') as f:
        with open('./data/OptBasics.csv', newline='') as f:
            f.readline()    # 忽略表头
            opts_params = csv.reader(f, delimiter=',')
            for opt_params in opts_params:
                if opt_params[0] == self.code:
                    self.name = opt_params[2]
                    if opt_params[5] == '认购':
                        self.opt_type = 'Call'
                    else:
                        self.opt_type = 'Put'
                    if opt_params[6] == '欧式':
                        self.exercise_type = 'European'
                    else:
                        self.exercise_type = 'American'
                    self.strike = float(opt_params[7])
                    self.multiplier = int(opt_params[8])
                    self.end_date = datetime.datetime.strptime(opt_params[11], '%Y-%m-%d').date()
                    break

    def calc_margin(self, opt_pre_settle, underlying_pre_close):
        """
        计算期权的开仓保证金
        :param opt_pre_settle:期权前结算价
        :param underlying_pre_close: 标的前收盘价
        :return:
        """
        if self.opt_type == 'Call':
            self.margin = (opt_pre_settle + max(0.12*underlying_pre_close - max(self.strike-underlying_pre_close, 0),
                                                0.07*underlying_pre_close)) * self.multiplier
        else:
            self.margin = min(opt_pre_settle + max(0.12*underlying_pre_close - max(underlying_pre_close-self.strike, 0),
                                                   0.07*self.strike), self.strike) * self.multiplier

    def time_value(self, underlying_price, trading_datetime):
        """
        返回期权的时间价值
        :param underlying_price:标的的价格
        :param trading_datetime:交易时间(类型=datetime.datetime)
        :return:
        """
        opt_price = self.quote_1min.ix[trading_datetime, 'close']
        if self.opt_type == 'Call':
            return opt_price - max(underlying_price - self.strike, 0.0)
        else:
            return opt_price - max(self.strike - underlying_price, 0.0)

    def __eq__(self, opt):
        """
        重载等于运算符
        :param opt:COption类
        :return: 两者代码相等返回True，否则返回False
        """
        return self.code == opt.code

    def maturity(self, calc_date_time, unit='days'):
        """
        计算期权剩余期限
        Parameters:
        --------
        :param calc_date_time: datetime.datetime
            计算日期
        :param unit: str
            unit='days', 返回剩余期限以'天'为单位；unit='years', 返回剩余期限以'年'为单位
        :return: float
            该期权剩余期限，以距离最后交易日的自然日天数为基础计算
        """
        tau = (self.end_date - calc_date_time.date()).days + self.time_remain_of_day(calc_date_time)
        if unit == 'years':
            tau /= 365.
        return tau

    @classmethod
    def time_remain_of_day(cls, calc_date_time):
        """
        计算当天剩余时间（单位=天），主要用于计算期权希腊字母时剩余时间
        Parameters:
        ---------
        calc_date_time:时间，datetime.datetime类型
        """
        y = calc_date_time.year
        m = calc_date_time.month
        d = calc_date_time.day
        if calc_date_time < datetime.datetime(y, m, d, 9, 30, 0, 0):
            time_remain = 1.0
        elif calc_date_time < datetime.datetime(y, m, d, 11, 30, 0, 0):
            time_remain = 1.0 - (calc_date_time - datetime.datetime(y, m, d, 9, 30, 0, 0)).seconds / 14400.0
        elif calc_date_time < datetime.datetime(y, m, d, 13, 0, 0, 0):
            time_remain = 0.5
        elif calc_date_time < datetime.datetime(y, m, d, 15, 0, 0, 0):
            time_remain = (datetime.datetime(y, m, d, 15, 0, 0, 0) - calc_date_time).seconds / 14400.0
        else:
            time_remain = 0.0
        return time_remain
