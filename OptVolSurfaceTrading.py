#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 波动率曲面交易策略
# @Filename: OptVolSurfaceTrading
# @Date:   : 2018-02-01 16:43
# @Author  : YuJun
# @Email   : yujun_mail@163.com

from util.COption import COption
import vol_surface_model as vsm
import pandas as pd
from pandas import DataFrame, Series
import datetime
import logging
from configparser import ConfigParser
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class CVolSurfaceTradingStrategy(object):
    """波动率曲面交易策略"""
    def __init__(self, portname, configname):
        self.parname = portname
        self.configname = configname
        self.opts_data = {}                 # 当月、次月期权基础数据，字典类型, map<optcode, COption>
        self.monitor_data = DataFrame()     # 监控数据，含最新行情及波动率数据
        self.underlying_price = 0           # underlying的最新价格
        self.risk_free = 0.038              # 无风险利率
        self.q = 0                          # underlying的股息率
        self.sv_model = None                # 随机波动率模型
        self.vol_par = []                   # 波动率模型参数

        self.underlying_quote_1min = None   # 标的分钟行情
        self.trading_opt_expdate = None     # 当前参与套利交易期权的到期日期
        self.arb_holding_pairs = None       # 策略的套利持仓对数据, pd.DataFrame

        self.trading_date = None            # 当前交易日期, datetime.date
        self.trading_time = None            # 当期交易时间, datetime.datetime

    def load_param(self):
        """导入策略的参数"""
        cfg = ConfigParser()
        cfg.read('config.ini')
        self.sv_model = cfg.get(self.configname, 'sv_model')

    def load_vol_param(self, trading_day):
        """导入波动率模型参数"""
        par = pd.read_csv(Path('./data/%s_par.csv' % self.sv_model), parse_dates=[0], header=0)
        par = par[par.date <= trading_day].iloc[0]
        self.vol_par = [par['alpha'], par['beta'], par['rho'], par['nu']]

    def load_opt_basic_data(self, trading_day):
        """
        导入期权基本信息数据
        :param trading_day: datetime.date
        :return:
        """
        self.opts_data = {}
        # 读取期权基本数据
        header_name = ['opt_code', 'trade_code', 'opt_name', 'underlying_code', 'secu_type', 'opt_type',
                       'exercise_type', 'strike', 'multiplier', 'end_month', 'listed_date', 'expire_date',
                       'exercise_date', 'delivery_date']
        opts_basics = pd.read_csv('./data/OptBasics.csv', usecols=range(14), parse_dates=[10, 11, 12, 13],
                                  names=header_name, dtype={'opt_code': str}, header=0)
        # 选取当月、次月合约
        opts_basics = opts_basics[(opts_basics.expire_date >= trading_day) & (opts_basics.listed_date <= trading_day)]
        expire_dats_used = sorted(list(set(opts_basics.expire_date)))[:2]
        opts_basics = opts_basics[opts_basics.expire_date.apply(lambda x: True if x in expire_dats_used else False)]
        # 构建self.opts_data
        for idx, opt_data in opts_basics.iterrows():
            if opt_data['opt_type'] == '认购':
                opt_type = 'Call'
            else:
                opt_type = 'Put'
            if opt_data['exercise_type'] == '欧式':
                exercise_type = 'European'
            else:
                exercise_type = 'American'
            end_date = opt_data['expire_date'].to_pydatetime().date()
            self.opts_data[opt_data['opt_code']] = COption(opt_data['opt_code'], opt_data['opt_name'], opt_type,
                                                           exercise_type, opt_data['strike'], opt_data['multiplier'],
                                                           end_date)

    def init_monitor_data(self, trading_day):
        """
        初始化moniter_data
        :param trading_day: datetime.date
            交易日期
        :return:
        """
        self.monitor_data = DataFrame()
        for opt_code, copt in self.opts_data.items():
            # 剔除剩余期限（自然日）小于7天的期权
            # if copt.maturity(datetime.datetime(trading_day.year, trading_day.month, trading_day.day,0, 0, 0), unit='days') < 7:
            #     continue
            single_opt = Series({'expire_date': copt.end_date, 'opt_type': copt.opt_type,
                                 'strike': round(copt.strike,3), 'code': copt.code, 'name': copt.name,
                                 'ask_volume': 0, 'ask_price': 0, 'ask_imp_vol': 0, 'mid_imp_vol': 0,
                                 'model_imp_vol': 0, 'model_price': 0, 'bid_volume': 0, 'bid_price': 0,
                                 'bid_imp_vol': 0, 'long_spread': 0, 'short_spread': 0, 'delta': 0,
                                 'maturity': copt.maturity(datetime.datetime.combine(trading_day, datetime.time()))})
            # single_opt = Series([copt.end_date, copt.strike, copt.code, copt.name, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                     index=['expire_date', 'strke', 'code', 'name', 'ask_volume', 'ask_price', 'ask_imp_vol',
            #                            'mid_imp_vol', 'model_imp_vol', 'model_price', 'bid_volume', 'bid_price', 'bid_imp_vol'])
            self.monitor_data = self.monitor_data.append(single_opt, ignore_index=True)
        self.monitor_data.sort_values(by=['opt_type','expire_date'], inplace=True)
        self.monitor_data = self.monitor_data.set_index(['opt_type', 'expire_date'])
        # 如果当月合约的剩余期限（自然日）小于等于7天，那么交易次月合约
        if self.monitor_data.index.levels[1][0].date - trading_day + 1 <= 7:
            self.trading_opt_expdate = self.monitor_data.index.levels[1][1]
        else:
            self.trading_opt_expdate = self.monitor_data.index.levels[1][0]

    def update_monitor_data(self, df_opt_quote, underlying_quote, quote_type='M'):
        """
        更新monitor_data
        Parameters:
        --------
        :param df_opt_quote: pd.DataFrame
            期权分钟行情数据<code,date_time,open,high,low,close,volume,amount>, index=code
            期权实时行情数据
        :param underlying_quote: pd.Series
            underlying分钟行情数据<code,date_time,open,high,low,close,volume,amount>, index=code
            underlying实时行情数据
        :param quote_type: str
            行情数据类型, 'M'=分钟行情数据, 'R'=实时行情数据
        :return:
        """
        # 更新行情数据及波动率数据
        if quote_type == 'M':
            self.underlying_price = underlying_quote['close']
            for idx, monitor_data in self.monitor_data.iterrows():
                if monitor_data['code'] in df_opt_quote.index:
                    self.monitor_data.loc[idx, 'ask_price'] = df_opt_quote[monitor_data['code'], 'close']
                    self.monitor_data.loc[idx, 'bid_price'] = df_opt_quote[monitor_data['code'], 'close']
                    trading_time = df_opt_quote[monitor_data['code'], 'date_time']
                    mkt_price = (self.monitor_data.loc[idx, 'ask_price'] + self.monitor_data.loc[idx, 'bid_price']) / 2.
                    tau = self.opts_data[monitor_data['code']].maturity(trading_time, 'years')
                    imp_vol = vsm.opt_imp_vol(self.underlying_price, monitor_data['strike'], self.risk_free, self.q,
                                              tau, monitor_data['opt_type'], mkt_price)
                    self.monitor_data.loc[idx, 'ask_imp_vol'] = imp_vol
                    self.monitor_data.loc[idx, 'bid_imp_vol'] = imp_vol
                    self.monitor_data.loc[idx, 'mid_imp_vol'] = imp_vol
                    if self.sv_model == 'sabr':
                        self.monitor_data.loc[idx, 'model_imp_vol'] = vsm.SABR(self.vol_par[0], self.vol_par[1], self.vol_par[2], self.vol_par[3],
                                                                               self.underlying_price, monitor_data['strike'], tau)
                        self.monitor_data.loc[idx, 'model_price'] = vsm.bs_model(self.underlying_price, monitor_data['strike'], self.risk_free,
                                                                                 self.q, self.monitor_data.loc[idx, 'model_imp_vol'], tau,
                                                                                 monitor_data['opt_type'])
                        self.monitor_data.loc[idx, 'long_spread'] = self.monitor_data.loc[idx, 'ask_price'] - self.monitor_data.loc[idx, 'model_price']
                        self.monitor_data.loc[idx, 'short_spread'] = self.monitor_data.loc[idx, 'bid_price'] - self.monitor_data.loc[idx, 'model_price']
                    # self.monitor_data.loc[idx, 'time_value'] = self.opts_data[monitor_data['code']].time_value(self.underlying_price, trading_time)
                    self.monitor_data.loc[idx, 'delta'] = self.opts_data[monitor_data['code']].greeks.delta
                else:
                    logging.info('%s期权的行情没有更新!' % monitor_data['code'])
        elif quote_type == 'R':
            pass

    def sorted_arbitrade_spread(self):
        """
        分别按认购、认沽对self.monitor_data的数据进行排序（降序）
        :return: tuple, 返回4个排序后的monitor_data
            1. 认购期权shortspread降序排列
            2. 认购期权longspread升序排列
            3. 认沽期权shortspread降序排列
            4. 认沽期权longspread升序排列
        """
        df_monitor_data = self.monitor_data.set_index([])
        # 剔除delta小于0.9和小于0.1的深度s实值/虚值合约
        df_monitor_data = self.monitor_data[(self.monitor_data.delta < 0.9) & (self.monitor_data.delta > 0.1)]
        # 认购期权shortspread降序排列
        call_shortspread_desc = df_monitor_data.loc['Call', self.trading_opt_expdate].sort_values(by='short_spread', ascending=False)
        # 认购期权longspread升序排列
        call_longspread_asc = df_monitor_data.loc['Call', self.trading_opt_expdate].sort_values(by='long_spread', ascending=True)
        # 认沽期权shortspread降序排列
        put_shortspread_desc = df_monitor_data.loc['Put', self.trading_opt_expdate].sort_values(by='short_spread', ascending=False)
        # 认沽期权longspread升序排列
        put_longspread_asc = df_monitor_data.loc['Put', self.trading_opt_expdate].sort_values(by='long_spread', ascending=True)

        return (call_shortspread_desc, call_longspread_asc, put_shortspread_desc, put_longspread_asc)


    def load_opt_1min_quote(self, trading_day):
        """
        导入指定日期期权的1分钟行情数据
        Parameters:
        --------
        :param trading_day: datetime.date
            日期
        :return:
        """
        cfg = ConfigParser()
        cfg.read('config.ini')
        quote_path = cfg.get('path', 'opt_quote_path')
        strdate = trading_day.strftime('%Y-%m-%d')
        for optcode, opt in self.opts_data.items():
            file_path = Path(quote_path, strdate, '%s.csv' % optcode)
            opt.quote_1min = pd.read_csv(file_path, usecols=range(7), index_col=0, parse_dates=[0])

    def load_underlying_1min_quote(self, trading_day):
        """
        导入期权标的1分钟行情数据
        :param trading_day: datetime.date
            日期
        :return:
        """
        cfg = ConfigParser()
        cfg.read('config.ini')
        quote_path = cfg.get('path', 'opt_quote_path')
        strdate = trading_day.strftime('%Y-%m-%d')
        file_path = Path(quote_path, strdate, '510050ETF.csv')
        self.underlying_quote_1min = pd.read_csv(file_path, usecols=range(7), index_col=0, parse_dates=[0])

    def handle_arb_holding_pars(self, handle_type, trade_data=None):
        """
        处理套利持仓对数据
        Parameters:
        --------
        :param handle_type: str
            处理方式, 'add'=添加套利对; 'scan'=扫描套利对数据，检查是否需要平仓; 'save'=保存套利对数据
        :param trade_data: list, 默认为None
            当handle_type='add'时，trade_data为交易数据, trade_data列表的元素为COptTradeData类
            trade_data[0]为套利对中买入期权交易, trade_data[1]为套利对中卖出期权交易
        :return:
        """
        if handle_type == 'add':
            assert len(trade_data) == 2
            if self.arb_holding_pairs is None:
                self.arb_holding_pairs = DataFrame()
            single_pair = Series()
            single_pair['date_time'] = trade_data[0].time           # 交易时间
            single_pair['long_code'] = trade_data[0].code           # 多头期权代码
            single_pair['long_volume'] = trade_data[1].tradevol     # 多头期权数量
            single_pair['long_cost'] = trade_data[0].tradeprice     # 多头期权成本
            single_pair['long_last'] = self.opts_data[trade_data[0].code].quote_1min.loc[trade_data[0].time, 'close']   # 多头期权的现价
            single_pair['long_model_price'] = self.monitor_data[self.monitor_data.code==trade_data[0].code].iloc[0]['model_price']  # 多头期权的理论价格
            single_pair['short_code'] = trade_data[1].code          # 空头期权代码
            single_pair['short_volume'] = trade_data[1].tradevol    # 空头期权数量
            single_pair['short_cost'] = trade_data[1].tradeprice    # 空头期权成本
            single_pair['short_last'] = self.opts_data[trade_data[1].code].quote_1min.loc[trade_data[1].time, 'close']  # 空头期权的现价
            single_pair['short_model_price'] = self.monitor_data[self.monitor_data.code==trade_data[1].code].iloc[0]['model_price'] # 空头期权的理论价格
            # 盈利空间
            single_pair['profit_spread'] = (single_pair['long_model_price'] - single_pair['long_cost']) * single_pair['long_volume'] + (single_pair['short_cost'] - single_pair['short_model_price']) * single_pair['short_volume']
            # 已实现盈亏
            single_pair['realized_profit'] = (single_pair['long_last'] - single_pair['long_cost']) * single_pair['long_volume'] + (single_pair['short_cost'] - single_pair['short_last']) * single_pair['short_volume']
            single_pair['profit_ratio'] = single_pair['realized_profit'] / single_pair['profit_spread'] # 实现盈亏占比
            self.arb_holding_pairs.append(single_pair, ignore_index=True)
        elif handle_type == 'scan':
            # 遍历self.arb_holding_pairs, 更新最新价格、理论价格、盈利空间、已实现盈亏及实现盈利占比
            for _, arb_pair in self.arb_holding_pairs.iterrows():
                arb_pair['long_last'] = self.opts_data[arb_pair['long_code']].quote_1min.loc[self.trading_time, 'close']
                arb_pair['long_model_price'] = self.monitor_data[self.monitor_data.code==arb_pair['long_code']].iloc[0]['model_price']
                arb_pair['short_last'] = self.opts_data[arb_pair['short_code']].quote_1min.loc[self.trading_time, 'close']
                arb_pair['short_model_price'] = self.monitor_data[self.monitor_data.code==arb_pair['short_code']].iloc[0]['model_price']



if __name__ == '__main__':
    # pass
    s = CVolSurfaceTradingStrategy('VolSurfaceTrade', 'vol_surface_strategy')
    s.load_opt_basic_data(datetime.date(2017,1,2))
    s.init_monitor_data(datetime.date(2017,1,2))
    # s.update_monitor_data(None, None, 'M')
    print(s.monitor_data)
    print(s.monitor_data.index)
