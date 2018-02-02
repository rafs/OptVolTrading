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


class CVolSurfaceTradingStrategy(object):
    """波动率曲面交易策略"""
    def __init__(self):
        self.opts_data = {}                 # 当月、次月期权基础数据，字典类型, map<optcode, COption>
        self.monitor_data = DataFrame()     # 监控数据，含最新行情及波动率数据
        self.underlying_price = 0           # underlying的最新价格
        self.risk_free = 0.038              # 无风险利率
        self.q = 0                          # underlying的股息率

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
            if copt.maturity(datetime.datetime(trading_day.year, trading_day.month, trading_day.day,0, 0, 0), unit='days') < 7:
                continue
            single_opt = Series({'expire_date': copt.end_date, 'opt_type': copt.opt_type,
                                 'strike': round(copt.strike,3), 'code': copt.code, 'name': copt.name,
                                 'ask_volume': 0, 'ask_price': 0, 'ask_imp_vol': 0, 'mid_imp_vol': 0,
                                 'model_imp_vol': 0, 'model_price': 0, 'bid_volume': 0, 'bid_price': 0,
                                 'bid_imp_vol': 0})
            # single_opt = Series([copt.end_date, copt.strike, copt.code, copt.name, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                     index=['expire_date', 'strke', 'code', 'name', 'ask_volume', 'ask_price', 'ask_imp_vol',
            #                            'mid_imp_vol', 'model_imp_vol', 'model_price', 'bid_volume', 'bid_price', 'bid_imp_vol'])
            self.monitor_data = self.monitor_data.append(single_opt, ignore_index=True)
        self.monitor_data.sort_values(by='opt_type', inplace=True)
        self.monitor_data = self.monitor_data.set_index(['opt_type', 'expire_date'])

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
        # 更新行情数据
        if quote_type == 'M':
            self.underlying_price = underlying_quote['close']
            for idx, monitor_data in self.monitor_data.iterrows():
                if monitor_data['code'] in df_opt_quote.index:
                    self.monitor_data.loc[idx, 'ask_price'] = df_opt_quote[monitor_data['code'], 'close']
                    self.monitor_data.loc[idx, 'bid_price'] = df_opt_quote[monitor_data['code'], 'close']
                    imp_vol = vsm.opt_imp_vol(self.underlying_price, monitor_data['strike'], self.risk_free, self.q,
                                              self.opts_data[monitor_data['code']].maturity())
        elif quote_type == 'R':
            pass



if __name__ == '__main__':
    # pass
    s = CVolSurfaceTradingStrategy()
    s.load_opt_basic_data(datetime.date(2017,1,2))
    s.init_monitor_data(datetime.date(2017,1,2))
    # s.update_monitor_data(None, None, 'M')
    print(s.monitor_data)
