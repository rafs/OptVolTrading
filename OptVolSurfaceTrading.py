#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 波动率曲面交易策略
# @Filename: OptVolSurfaceTrading
# @Date:   : 2018-02-01 16:43
# @Author  : YuJun
# @Email   : yujun_mail@163.com

from util.COption import COption
import vol_surface_model as vsm
from util.util import COptTradeData
from util.COptHolding import COptHolding
from util.util import Utils
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import datetime
import logging
from configparser import ConfigParser
from pathlib import Path
import csv
import time
import calendar

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class CVolSurfaceTradingStrategy(object):
    """波动率曲面交易策略"""
    def __init__(self, portname, configname):
        self.portname = portname
        self.configname = configname
        self.opts_data = {}                 # 当月、次月期权基础数据，字典类型, map<optcode, COption>
        self.monitor_data = DataFrame()     # 监控数据，含最新行情及波动率数据
        self.underlying_price = 0           # underlying的最新价格
        self.risk_free = 0                  # 无风险利率, float
        self.hist_vol = 0                   # 标的历史波动率, float
        self.q = 0                          # underlying的股息率, float
        self.sv_model = None                # 随机波动率模型名称, str
        self.vol_par = None                 # 波动率模型参数, dict<'Call': [alpha,beta,rho,nu], 'Put': [alpha,beta,rho,nu]>

        self.underlying_quote_1min = None   # 标的分钟行情数据, pd.DataFrame
        self.trading_opt_expdate = None     # 当前参与套利交易期权的到期日期, datetime.datetime
        self.arb_holding_pairs = None       # 策略的套利持仓对数据, pd.DataFrame

        self.trading_date = None            # 当前交易日期, datetime.date
        self.trading_time = None            # 当期交易时间, datetime.datetime
        self.pre_trading_date = None        # 前一交易日日期, datetime.date
        self.commission_per_unit = None     # 每张期权交易佣金, float
        self.pair_holding_days = None       # 策略套利持仓对最大持有交易日天数, int
        self.margin_ratio = 0.9             # 保证金最高比例
        self.opt_holdings_path = ''         # 持仓数据文件夹路径
        self.calendar = None                # 交易日历

        self.expected_ret_threshold = None  # 套利交易预期收益率阀值(dict{'call','put'})

        self.trade_num_in_1min = 0          # 一分钟内已交易的次数
        self.max_trade_num_1min = 5         # 一分钟内最大的交易次数
        self.trading_slip_point = 3         # 交易滑点

        # 导入相关参数
        self._load_param()

        # 策略的期权持仓类
        self.opt_holdings = COptHolding(Path(self.opt_holdings_path, self.configname, 'log.txt'))

    def _calibrate_sv_model(self):
        """
        对随机波动率模型参数进行校准
        :return:
        """
        # if self.underlying_quote_1min is None:
        #     self._load_underlying_1min_quote(trading_day)
        # 遍历样本期权，分别提取认购、认沽期权的行权价list、maturity list、隐含波动率list
        call_strikes = []
        call_maturities = []
        call_imp_vols = []
        put_strikes = []
        put_maturities = []
        put_imp_vols = []
        for idx, monitor_data in self.monitor_data.iterrows():
            if self.opts_data[monitor_data['code']].maturity(self.trading_time, 'days') <= 7:   # 剔除距离到期日小于7个自然日的期权
                continue
            if idx[0] == 'Call':
                call_strikes.append(monitor_data['strike'])
                call_maturities.append(monitor_data['maturity'])
                call_imp_vols.append(monitor_data['mid_imp_vol'])
            elif idx[0] == 'Put':
                put_strikes.append(monitor_data['strike'])
                put_maturities.append(monitor_data['maturity'])
                put_imp_vols.append(monitor_data['mid_imp_vol'])
        # 校准波动率模型参数
        call_sv_par = list(vsm.sabr_calibration(0.5, self.underlying_price, call_strikes, call_maturities, call_imp_vols))
        call_sv_par.insert(0,self.trading_date.strftime('%Y-%m-%d'))
        call_sv_par.insert(1, 'Call')
        put_sv_par = list(vsm.sabr_calibration(0.5, self.underlying_price, put_strikes, put_maturities, put_imp_vols))
        put_sv_par.insert(0, self.trading_date.strftime('%Y-%m-%d'))
        put_sv_par.insert(1, 'Put')
        # 保存波动率模型参数
        with open('./data/%s_par.csv' % self.sv_model, 'a', newline='', encoding='UTF-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows([call_sv_par, put_sv_par])
        # 计算期权理论波动率，并保存
        call_model_impvols = []
        alpha = call_sv_par[2]
        beta = call_sv_par[3]
        rho = call_sv_par[4]
        nu = call_sv_par[5]
        for j in range(len(call_strikes)):
            call_model_impvols.append(vsm.SABR(alpha, beta, rho, nu, self.underlying_price, call_strikes[j], call_maturities[j]))
        put_model_impvols = []
        alpha = put_sv_par[2]
        beta = put_sv_par[3]
        rho = put_sv_par[4]
        nu = put_sv_par[5]
        for j in range(len(put_strikes)):
            put_model_impvols.append(vsm.SABR(alpha, beta, rho, nu, self.underlying_price, put_strikes[j], put_maturities[j]))
        df_call_imp = DataFrame({'maturity': call_maturities, 'strike': call_strikes, 'mkt_imp_vol': call_imp_vols, 'model_imp_vol': call_model_impvols, 'opt_type': ['Call']*len(call_maturities)})
        df_put_imp = DataFrame({'maturity': put_maturities, 'strike': put_strikes, 'mkt_imp_vol': put_imp_vols, 'model_imp_vol': put_model_impvols, 'opt_type': ['Put']*len(put_maturities)})
        file_path = Path(self.opt_holdings_path, self.configname, 'impvol_%s_%s.csv' % (self.portname, self.trading_date.strftime('%Y%m%d')))
        decimals = Series([6, 3, 6, 6], index=['maturity', 'strike', 'mkt_imp_vol', 'model_imp_vol'])
        pd.concat([df_call_imp, df_put_imp], axis=0, ignore_index=True).round(decimals).to_csv(file_path, index=False, columns=['maturity', 'strike', 'opt_type', 'mkt_imp_vol', 'model_imp_vol'])

    def calibrate_sv_model(self, start_date, end_date=None):
        """
        对随机波动率模型参数进行校准(这是对外提供的接口)
        Parameters:
        --------
        :param start_date: datetime.date
            开始日期
        :param end_date: datetime.date, 默认None
            结束日期
        :return: tulple
            返回认购、认沽期权校准参数tuple(call_sv_par, put_sv_par)
        """
        # self._load_param()
        if end_date is None:
            end_date = start_date
        df_tradingdays = self.calendar[(self.calendar.tradingday >= start_date) & (self.calendar.tradingday <= end_date)]
        for _, tradingdays in df_tradingdays.iterrows():
            trading_day = tradingdays['tradingday'].date()
            logging.info('Calibration sv moded parameters at %s' % trading_day.strftime('%Y-%m-%d'))
            self.trading_date = trading_day
            self.trading_time = datetime.datetime.combine(trading_day, datetime.time(15, 0, 0))
            # 导入期权基本信息数据
            self._load_opt_basic_data(trading_day)
            # 导入期权和标的的分钟行情数据
            self._load_opt_1min_quote(trading_day)
            self._load_underlying_1min_quote(trading_day)
            # 导入无风险利率及标的历史波动率
            self._load_risk_free(trading_day)
            self._load_hist_vol(trading_day)
            # 初始化monitor_data
            self._init_monitor_data(trading_day)
            # 用收盘数据更新monitor
            df_opt_quote = DataFrame()
            for idx, monitor_data in self.monitor_data.iterrows():
                if len(self.opts_data[monitor_data['code']].quote_1min) == 0:
                    continue
                # trading_date_time = datetime.datetime.combine(trading_day, datetime.time())
                # if self.opts_data[monitor_data['code']].maturity(trading_date_time, 'days') <= 5:
                #     continue
                opt_quote = self.opts_data[monitor_data['code']].quote_1min.iloc[-1].copy()
                opt_quote['code'] = monitor_data['code']
                df_opt_quote = df_opt_quote.append(opt_quote, ignore_index=False)
            df_opt_quote.index.name = 'date_time'
            df_opt_quote.reset_index(inplace=True)
            df_opt_quote.set_index(keys='code', inplace=True)
            underlying_quote = self.underlying_quote_1min.iloc[-1]
            # underlying_quote['code'] = '510050'
            self._update_monitor_data(df_opt_quote, underlying_quote, quote_type='M', update_model=False)
            # 校准随机波动率模型参数
            self._calibrate_sv_model()

    def _load_param(self):
        """导入策略的参数"""
        cfg = ConfigParser()
        cfg.read('config.ini')
        self.sv_model = cfg.get(self.configname, 'sv_model')
        self.opt_holdings_path = cfg.get('path', 'opt_holdings_path')
        self.commission_per_unit = cfg.getfloat('trade', 'commission')
        self.pair_holding_days = cfg.getint('vol_surface_strategy', 'pair_holding_days')
        self.calendar = pd.read_csv('./data/tradingdays.csv', parse_dates=[0,1], encoding='UTF-8')

    def _load_vol_param(self, trading_day):
        """导入波动率模型参数"""
        par = pd.read_csv(Path('./data/%s_par.csv' % self.sv_model), parse_dates=[0], header=0, encoding='UTF-8')
        par = par[par.date == trading_day]
        call_par = par[par.type == 'Call'].iloc[0]
        put_par = par[par.type == 'Put'].iloc[0]
        self.vol_par = {'Call': [call_par['alpha'], call_par['beta'], call_par['rho'], call_par['nu']],
                        'Put': [put_par['alpha'], put_par['beta'], put_par['rho'], put_par['nu']]}

    def _load_expected_return_threshold(self, trading_day, days):
        """
        取得给定交易日期的几天波动率曲面交易机会预期收益率的中位数
        :param trading_day: datetime.date
            交易日期
        :param days: int
            天数
        """
        # # 计算取得trading_day日期前一个月的开始、结束时间
        # year = trading_day.year
        # month = trading_day.month
        # month -= 1
        # if month <= 0:
        #     month = 12
        #     year -= 1
        # wday, monthrange = calendar.monthrange(year, month)
        # start_time = '%s 09:30:00' % datetime.date(year, month, 1).strftime('%Y-%m-%d')
        # end_time = '%s 15:00:00' % datetime.date(year, month, monthrange).strftime('%Y-%m-%d')

        # 计算取得trading_day前days天数的开始、结束时间
        tradingdays_range = self.calendar[self.calendar.tradingday < trading_day].tail(days)
        start_time = '%s 09:30:00' % tradingdays_range.iloc[0].tradingday.strftime('%Y-%m-%d')
        end_time = '%s 15:00:00' % tradingdays_range.iloc[-1].tradingday.strftime('%Y-%m-%d')
        # 读取波动率曲面交易机会数据, 计算给定交易日期前days天的预期收益率的中位数
        trade_chance_path = './data/trade_chance_%s.csv' % self.portname
        df_trade_chances = pd.read_csv(trade_chance_path, header=0, encoding='UTF-8')
        sample_trade_chances = df_trade_chances[(df_trade_chances.datetime >= start_time) & (df_trade_chances.datetime <= end_time)]
        if len(sample_trade_chances) == 0:
            logging.info('Trade chances of previous %d days is empty.' % days)
            return
        call_sample = sample_trade_chances[sample_trade_chances.opt_type == 'Call']
        put_sample = sample_trade_chances[sample_trade_chances.opt_type == 'Put']
        call_expected_ret_threshold = np.percentile(Utils.clean_extreme_value(np.array(call_sample['expected_return']), 'MAD'), 75)
        # call_expected_ret_threshold = call_sample['expected_return'].quantile(0.75)
        if call_expected_ret_threshold is np.nan:
            call_expected_ret_threshold = 0.04
        if call_expected_ret_threshold < 0.04:
            call_expected_ret_threshold = 0.04
        put_expected_ret_threshold = np.percentile(Utils.clean_extreme_value(np.array(put_sample['expected_return']), 'MAD'), 75)
        # put_expected_ret_threshold = put_sample['expected_return'].quantile(0.75)
        if put_expected_ret_threshold is np.nan:
            put_expected_ret_threshold = 0.04
        if put_expected_ret_threshold < 0.04:
            put_expected_ret_threshold = 0.04
        self.expected_ret_threshold = {'Call': call_expected_ret_threshold, 'Put': put_expected_ret_threshold}
        logging.info('Call ret threshold = %.4f, Put ret threshold = %.4f.' % (call_expected_ret_threshold, put_expected_ret_threshold))


    def _load_trading_datas(self, trading_day):
        """
        导入交易日交易相关数据，包含期权基本信息数据、期权分钟数据、标的分钟数据及随机波动率模型模型参数
        :param trading_day: datetime.date
            交易日期
        :return:
        """
        self._load_opt_basic_data(trading_day)
        self._load_opt_holdings(self.pre_trading_date)
        self._load_underlying_1min_quote(trading_day)
        self._load_opt_1min_quote(trading_day)
        self._load_risk_free(trading_day)
        self._load_hist_vol(trading_day)
        # pre_trading_day = self.calendar[self.calendar.tradingday == trading_day].iloc[0]['pre_tradingday']
        self._load_vol_param(self.pre_trading_date)
        self._handle_arb_holding_pairs('load')
        self._calc_opt_margin()
        self._load_expected_return_threshold(trading_day, 10)

    def _load_opt_basic_data(self, trading_day):
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
                                  names=header_name, dtype={'opt_code': str}, header=0, encoding='UTF-8')
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

    def _init_monitor_data(self, trading_day):
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
            tau = copt.maturity(datetime.datetime.combine(trading_day, datetime.time()), 'days')
            if tau <= 1:
                continue
            single_opt = Series({'expire_date': copt.end_date, 'opt_type': copt.opt_type,
                                 'strike': round(copt.strike,3), 'code': copt.code, 'name': copt.name,
                                 'ask_volume': 0, 'ask_price': 0, 'ask_imp_vol': 0, 'mid_imp_vol': 0,
                                 'model_imp_vol': 0, 'model_price': 0, 'bid_volume': 0, 'bid_price': 0,
                                 'bid_imp_vol': 0, 'long_spread': 0, 'short_spread': 0, 'delta': 0,
                                 'maturity': tau/365.})
            self.monitor_data = self.monitor_data.append(single_opt, ignore_index=True)
        self.monitor_data.sort_values(by=['opt_type','expire_date'], inplace=True)
        self.monitor_data = self.monitor_data.set_index(['opt_type', 'expire_date'])
        # 如果当月合约的剩余期限（自然日）小于等于7天，那么交易次月合约
        if (self.monitor_data.index.levels[1][0].date() - trading_day).days + 1 <= 7:
            self.trading_opt_expdate = self.monitor_data.index.levels[1][1]
        else:
            self.trading_opt_expdate = self.monitor_data.index.levels[1][0]

    def _update_monitor_data(self, df_opt_quote, underlying_quote, quote_type='M', update_model=True):
        """
        更新monitor_data
        Parameters:
        --------
        :param df_opt_quote: pd.DataFrame
            期权分钟行情数据<code,date_time,open,high,low,close,volume,amount>, index=code
            期权实时行情数据
        :param underlying_quote: pd.Series
            underlying分钟行情数据<code,date_time,open,high,low,close,volume,amount>
            underlying实时行情数据
        :param quote_type: str
            行情数据类型, 'M'=分钟行情数据, 'R'=实时行情数据
        :param update_model: bool, 默认True
            是否更新与随机波动率模型相关的数据
        :return:
        """
        # 更新行情数据及波动率数据
        # if quote_type == 'M':
        #     self.underlying_price = underlying_quote['close']
        #     k = 0
        #     for idx, monitor_data in self.monitor_data.iterrows():
        #         if monitor_data['code'] in df_opt_quote.index:
        #             self.monitor_data.iloc[k]['ask_price'] = df_opt_quote.loc[monitor_data['code'], 'close']
        #             self.monitor_data.iloc[k]['bid_price'] = df_opt_quote.loc[monitor_data['code'], 'close']
        #             trading_time = df_opt_quote.loc[monitor_data['code'], 'date_time']
        #             # mkt_price = (self.monitor_data.loc[idx, 'ask_price'] + self.monitor_data.loc[idx, 'bid_price']) / 2.
        #             mkt_price = (monitor_data['ask_price'] + monitor_data['bid_price']) / 2.
        #             tau = self.opts_data[monitor_data['code']].maturity(trading_time, 'years')
        #             opt_type = idx[0]
        #             imp_vol = vsm.opt_imp_vol(self.underlying_price, monitor_data['strike'], self.risk_free, self.q,
        #                                       tau, opt_type, mkt_price)
        #             self.monitor_data.iloc[k]['ask_imp_vol'] = imp_vol
        #             self.monitor_data.iloc[k]['bid_imp_vol'] = imp_vol
        #             self.monitor_data.iloc[k]['mid_imp_vol'] = imp_vol
        #             if update_model:
        #                 if self.sv_model == 'sabr':
        #                     self.monitor_data.iloc[k]['model_imp_vol'] = vsm.SABR(self.vol_par[0], self.vol_par[1], self.vol_par[2], self.vol_par[3],
        #                                                                           self.underlying_price, monitor_data['strike'], tau)
        #                     self.monitor_data.iloc[k]['model_price'] = vsm.bs_model(self.underlying_price, monitor_data['strike'], self.risk_free,
        #                                                                             self.q, monitor_data['model_imp_vol'], tau, monitor_data['opt_type'])
        #                     self.monitor_data.iloc[k]['long_spread'] = monitor_data['ask_price'] - monitor_data['model_price']
        #                     self.monitor_data.iloc[k]['short_spread'] = monitor_data['bid_price'] - monitor_data['model_price']
        #             # self.monitor_data.loc[idx, 'time_value'] = self.opts_data[monitor_data['code']].time_value(self.underlying_price, trading_time)
        #             self.monitor_data.iloc[k]['maturity'] = tau
        #             self.monitor_data.iloc[k]['delta'] = self.opts_data[monitor_data['code']].greeks.delta
        #         else:
        #             logging.info('%s期权的行情没有更新!' % monitor_data['code'])
        #         k += 1
        # elif quote_type == 'R':
        #     pass

        # 更新行情数据及波动率数据
        self.monitor_data.reset_index(inplace=True)
        if quote_type == 'M':
            self.underlying_price = underlying_quote['close']
            for idx, monitor_data in self.monitor_data.iterrows():
                if monitor_data['code'] in df_opt_quote.index:
                    self.monitor_data.loc[idx, 'ask_price'] = df_opt_quote.loc[monitor_data['code'], 'close']
                    self.monitor_data.loc[idx, 'bid_price'] = df_opt_quote.loc[monitor_data['code'], 'close']
                    trading_time = df_opt_quote.loc[monitor_data['code'], 'date_time']
                    # mkt_price = (self.monitor_data.loc[idx, 'ask_price'] + self.monitor_data.loc[idx, 'bid_price']) / 2.
                    mkt_price = (self.monitor_data.loc[idx, 'ask_price'] + self.monitor_data.loc[idx, 'bid_price']) / 2.
                    tau = self.opts_data[monitor_data['code']].maturity(trading_time, 'years')
                    opt_type = monitor_data['opt_type']
                    imp_vol = vsm.opt_imp_vol(self.underlying_price, monitor_data['strike'], self.risk_free, self.q, tau, opt_type, mkt_price)
                    self.monitor_data.loc[idx, 'ask_imp_vol'] = imp_vol
                    self.monitor_data.loc[idx, 'bid_imp_vol'] = imp_vol
                    self.monitor_data.loc[idx, 'mid_imp_vol'] = imp_vol
                    self.monitor_data.loc[idx, 'maturity'] = tau
                    if update_model:
                        if self.sv_model == 'sabr':
                            # self.monitor_data.loc[idx, 'model_imp_vol'] = vsm.SABR(self.vol_par[0], self.vol_par[1], self.vol_par[2], self.vol_par[3],
                            #                                                       self.underlying_price, monitor_data['strike'], tau)
                            self.monitor_data.loc[idx, 'model_imp_vol'] = self._calc_model_price(self.monitor_data.loc[idx])
                            self.monitor_data.loc[idx, 'model_price'] = vsm.bs_model(self.underlying_price, monitor_data['strike'], self.risk_free,
                                                                                    self.q, self.monitor_data.loc[idx, 'model_imp_vol'], tau, monitor_data['opt_type'])
                            self.monitor_data.loc[idx, 'long_spread'] = self.monitor_data.loc[idx, 'ask_price'] - self.monitor_data.loc[idx, 'model_price']
                            self.monitor_data.loc[idx, 'short_spread'] = self.monitor_data.loc[idx, 'bid_price'] - self.monitor_data.loc[idx, 'model_price']
                    # self.monitor_data.loc[idx, 'time_value'] = self.opts_data[monitor_data['code']].time_value(self.underlying_price, trading_time)
                    self.monitor_data.loc[idx, 'delta'] = self.opts_data[monitor_data['code']].greeks.delta
                else:
                    logging.info('%s期权的行情没有更新!' % monitor_data['code'])
        elif quote_type == 'R':
            pass

        self.monitor_data = self.monitor_data.set_index(['opt_type', 'expire_date'])

    def _sorted_arbitrage_spread(self):
        """
        分别按认购、认沽对self.monitor_data的数据进行排序（降序）
        :return: tuple, 返回4个排序后的monitor_data
            1. 认购期权shortspread降序排列
            2. 认购期权longspread升序排列
            3. 认沽期权shortspread降序排列
            4. 认沽期权longspread升序排列
        """
        # 剔除delta小于0.9和小于0.1的深度s实值/虚值合约
        df_monitor_data = self.monitor_data[(abs(self.monitor_data.delta) < 0.8) & (abs(self.monitor_data.delta) > 0.2)]
        # 认购期权shortspread降序排列
        call_shortspread_desc = df_monitor_data.loc['Call', self.trading_opt_expdate].sort_values(by='short_spread', ascending=False)
        # 认购期权longspread升序排列
        call_longspread_asc = df_monitor_data.loc['Call', self.trading_opt_expdate].sort_values(by='long_spread', ascending=True)
        # 认沽期权shortspread降序排列
        put_shortspread_desc = df_monitor_data.loc['Put', self.trading_opt_expdate].sort_values(by='short_spread', ascending=False)
        # 认沽期权longspread升序排列
        put_longspread_asc = df_monitor_data.loc['Put', self.trading_opt_expdate].sort_values(by='long_spread', ascending=True)

        return (call_shortspread_desc, call_longspread_asc, put_shortspread_desc, put_longspread_asc)


    def _load_opt_1min_quote(self, trading_day):
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
        for optcode, holding in self.opt_holdings.holdings.items():
            file_path = Path(quote_path, strdate, '%s.csv' % optcode)
            holding.COption.quote_1min = pd.read_csv(file_path, usecols=range(7), index_col=0, parse_dates=[0], encoding='UTF-8')

    def _load_underlying_1min_quote(self, trading_day):
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
        self.underlying_quote_1min = pd.read_csv(file_path, usecols=range(7), index_col=0, parse_dates=[0], encoding='UTF-8')

    def _load_risk_free(self, trading_day):
        """
        导入无风险利率
        :param trading_day: datetime.date
        :return:
        """
        df_riskfree = pd.read_csv('./data/riskfree.csv', parse_dates=[0], encoding='UTF-8')
        self.risk_free = float(df_riskfree[df_riskfree.date <= trading_day].iloc[-1]['riskfree']) / 100.

    def _load_hist_vol(self, trading_day):
        """
        导入标的历史波动率
        :param trading_day: datetime.date
        :return:
        """
        df_hist_vol = pd.read_csv('./data/Historical_Vol.csv', parse_dates=[0], encoding='UTF-8')
        self.hist_vol = float(df_hist_vol[df_hist_vol.date <= trading_day].iloc[-1]['HV60'])

    def _load_opt_holdings(self, trading_day):
        """
        导入策略的期权持仓数据
        Parameters:
        --------
        :param trading_day: datetime.date
            持仓日期
        :return:
        """
        strdate = trading_day.strftime('%Y%m%d')
        holding_filename = Path(self.opt_holdings_path, self.configname, 'holding_%s_%s.csv' % (self.portname, strdate))
        self.opt_holdings.load_holdings(holding_filename)

    def _handle_arb_holding_pairs(self, handle_type, trade_data=None):
        """
        处理套利持仓对数据
        Parameters:
        --------
        :param handle_type: str
            处理方式, 'add'=添加套利对; 'scan'=扫描套利对数据，检查是否需要平仓; 'save'=保存套利对数据, ‘load’=导入前一交易日套利对数据
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
            single_pair['date_time'] = trade_data[0].time.strftime('%Y-%m-%d %H:%M:%S')     # 交易时间, str
            single_pair['long_code'] = trade_data[0].code           # 多头期权代码
            single_pair['long_volume'] = trade_data[0].tradevol     # 多头期权数量
            single_pair['long_cost'] = trade_data[0].tradeprice     # 多头期权成本
            single_pair['long_last'] = self.opts_data[trade_data[0].code].quote_1min.loc[trade_data[0].time, 'close']   # 多头期权的现价
            single_pair['long_model_price'] = self.monitor_data[self.monitor_data.code==trade_data[0].code].iloc[0]['model_price']  # 多头期权的理论价格
            single_pair['short_code'] = trade_data[1].code          # 空头期权代码
            single_pair['short_volume'] = trade_data[1].tradevol    # 空头期权数量
            single_pair['short_cost'] = trade_data[1].tradeprice    # 空头期权成本
            single_pair['short_last'] = self.opts_data[trade_data[1].code].quote_1min.loc[trade_data[1].time, 'close']  # 空头期权的现价
            single_pair['short_model_price'] = self.monitor_data[self.monitor_data.code==trade_data[1].code].iloc[0]['model_price'] # 空头期权的理论价格
            # 盈利空间
            # long_profit_spread = (single_pair['long_model_price'] - single_pair['long_cost']) * single_pair['long_volume'] * self.opts_data[single_pair['long_code']].multiplier
            # short_profit_spread = (single_pair['short_cost'] - single_pair['short_model_price']) * single_pair['short_volume'] * self.opts_data[single_pair['short_code']].multiplier
            # single_pair['profit_spread'] = long_profit_spread + short_profit_spread
            single_pair['profit_spread'] = self._profit_spread(single_pair)
            # 预期年化收益率
            money_ocupied = self.opts_data[single_pair['short_code']].margin*single_pair['short_volume'] + single_pair['long_volume']*single_pair['long_cost']*self.opts_data[single_pair['long_code']].multiplier
            single_pair['expect_return'] = single_pair['profit_spread'] / money_ocupied
            # 已实现盈亏
            # single_pair['realized_profit'] = (single_pair['long_last'] - single_pair['long_cost']) * single_pair['long_volume'] + (single_pair['short_cost'] - single_pair['short_last']) * single_pair['short_volume']
            single_pair['realized_profit'] = self._realized_profit(single_pair)
            single_pair['profit_ratio'] = single_pair['realized_profit'] / single_pair['profit_spread'] # 实现盈亏占比
            single_pair['holding_days'] = 1                                                             # 持仓天数(按交易日计算)
            self.arb_holding_pairs = self.arb_holding_pairs.append(single_pair, ignore_index=True)
            # 添加持仓
            logging.info('建仓, 盈利空间 = %.2f., 预期收益 = %.2f%%' % (single_pair['profit_spread'], single_pair['expect_return']*100))
            self.opt_holdings.update_holdings(trade_data)
            self.trade_num_in_1min += 1
        elif handle_type == 'scan':
            # trade_datas = []
            # 遍历self.arb_holding_pairs, 更新最新价格、理论价格、盈利空间、已实现盈亏、预期年化收益率及实现盈利占比
            to_be_deleted = []
            for idx, arb_pair in self.arb_holding_pairs.iterrows():
                # 如果一分钟内交易次数达到上限, 跳出循环
                if self.trade_num_in_1min >= self.max_trade_num_1min:
                    break
                self.arb_holding_pairs.loc[idx, 'long_last'] = self.opts_data[arb_pair['long_code']].quote_1min.loc[self.trading_time, 'close']
                self.arb_holding_pairs.loc[idx, 'long_model_price'] = self.monitor_data[self.monitor_data.code==arb_pair['long_code']].iloc[0]['model_price']
                self.arb_holding_pairs.loc[idx, 'short_last'] = self.opts_data[arb_pair['short_code']].quote_1min.loc[self.trading_time, 'close']
                self.arb_holding_pairs.loc[idx, 'short_model_price'] = self.monitor_data[self.monitor_data.code==arb_pair['short_code']].iloc[0]['model_price']
                self.arb_holding_pairs.loc[idx, 'profit_spread'] = self._profit_spread(self.arb_holding_pairs.loc[idx])
                money_ocupied = self.opts_data[arb_pair['short_code']].margin*arb_pair['short_volume'] + arb_pair['long_volume']*arb_pair['long_cost']*self.opts_data[arb_pair['long_code']].multiplier
                self.arb_holding_pairs.loc[idx, 'expect_return'] = self.arb_holding_pairs.loc[idx, 'profit_spread'] / money_ocupied
                self.arb_holding_pairs.loc[idx, 'realized_profit'] = self._realized_profit(self.arb_holding_pairs.loc[idx])
                self.arb_holding_pairs.loc[idx, 'profit_ratio'] = self.arb_holding_pairs.loc[idx, 'realized_profit'] / self.arb_holding_pairs.loc[idx, 'profit_spread']
                # 如果盈利空间>0, 且已实现盈利占比>50%, 平仓
                if self.arb_holding_pairs.loc[idx, 'profit_spread'] > 0 and self.arb_holding_pairs.loc[idx, 'profit_ratio'] > 0.5:
                    logging.info('已实现盈利=%.2f, %.2f%%>50%%, 平仓.' % (self.arb_holding_pairs.loc[idx, 'realized_profit'], self.arb_holding_pairs.loc[idx, 'profit_ratio'] * 100))
                    self._liquidate_arb_pair(self.arb_holding_pairs.loc[idx])
                    to_be_deleted.append(idx)
                    continue
                # 如果盈利空间<=0, 平仓
                # if self.arb_holding_pairs.loc[idx, 'profit_spread'] <= 0:
                #     logging.info('盈利空间<0,实现盈利 = %.2f, 平仓.' % self.arb_holding_pairs.loc[idx, 'realized_profit'])
                #     self._liquidate_arb_pair(self.arb_holding_pairs.loc[idx])
                #     to_be_deleted.append(idx)
                #     continue
                # 如果预期年化收益率小于无风险利率, 平仓
                # if arb_pair['expect_return'] < 0.:
                #     self._liquidate_arb_pair(arb_pair)
                #     to_be_deleted.append(idx)
                # 如果持有天数大于self.pair_holding_days, 平仓
                if arb_pair['holding_days'] > self.pair_holding_days:
                    logging.info('持有天数>2, 实现盈利 = %.2f, 平仓.' % self.arb_holding_pairs.loc[idx, 'realized_profit'])
                    self._liquidate_arb_pair(self.arb_holding_pairs.loc[idx])
                    to_be_deleted.append(idx)
                    continue
            if len(to_be_deleted) > 0:
                self.arb_holding_pairs.drop(to_be_deleted, axis=0, inplace=True)
        elif handle_type == 'save':
            arb_pair_header = ['date_time', 'long_code', 'long_volume', 'long_cost', 'long_last', 'long_model_price',
                               'short_code', 'short_volume', 'short_cost', 'short_last', 'short_model_price',
                               'profit_spread', 'expect_return', 'realized_profit', 'profit_ratio', 'holding_days']
            holding_filename = Path(self.opt_holdings_path, self.configname, 'arb_pairs_%s_%s.csv' % (self.portname, self.trading_date.strftime('%Y%m%d')))
            if len(self.arb_holding_pairs) > 0:
                self.arb_holding_pairs.to_csv(holding_filename, columns=arb_pair_header, index=False, float_format='%.4f')
        elif handle_type == 'load':
            self.arb_holding_pairs = DataFrame()
            arb_pair_header = ['date_time', 'long_code', 'long_volume', 'long_cost', 'long_last', 'long_model_price',
                               'short_code', 'short_volume', 'short_cost', 'short_last', 'short_model_price',
                               'profit_spread', 'expect_return', 'realized_profit', 'profit_ratio', 'holding_days']
            holding_filename = Path(self.opt_holdings_path, self.configname, 'arb_pairs_%s_%s.csv' % (self.portname, self.pre_trading_date.strftime('%Y%m%d')))
            if holding_filename.exists():
                self.arb_holding_pairs = pd.read_csv(holding_filename, header=0, names=arb_pair_header, parse_dates=[0], dtype={'long_code': str, 'short_code': str}, encoding='UTF-8')
                # 将套利持仓对的‘持仓天数’增加1
                for idx, arb_pair in self.arb_holding_pairs.iterrows():
                    # arb_pair['holding_days'] += 1
                    self.arb_holding_pairs.loc[idx, 'holding_days'] += 1

    def _liquidate_arb_pair(self, arb_pair):
        """
        对给定的套利持仓对进行平仓操作
        Parameters:
        --------
        :param arb_pair: pd.Series
        :return:
        """
        trade_datas = []
        trade_code = arb_pair['long_code']
        trade_price = arb_pair['long_last'] - 0.0001 * self.trading_slip_point
        trade_volume = arb_pair['long_volume']
        trade_value = trade_price * trade_volume * self.opts_data[trade_code].multiplier
        trade_commission = trade_volume * self.commission_per_unit
        trade_datas.append(COptTradeData(trade_code, 'sell', 'close', trade_price, trade_volume,
                                         trade_value, trade_commission, self.trading_time, self.opts_data[trade_code]))
        trade_code = arb_pair['short_code']
        trade_price = arb_pair['short_last'] + 0.0001 * self.trading_slip_point
        trade_volume = arb_pair['short_volume']
        trade_value = trade_price * trade_volume * self.opts_data[trade_code].multiplier
        trade_commission = trade_volume * self.commission_per_unit
        trade_datas.append(COptTradeData(trade_code, 'buy', 'close', trade_price, trade_volume, trade_value,
                                         trade_commission, self.trading_time, self.opts_data[trade_code]))
        self.opt_holdings.update_holdings(trade_datas)
        self.trade_num_in_1min += 1

    def _profit_spread(self, arb_pair):
        """
        计算给定套利持仓对的盈利空间
        Parameters:
        --------
        :param arb_pair: pd.Series
            套利持仓对的数据
        :return: float
        -------
            返回套利持仓对的盈利空间
        """
        long_profit_spread = (arb_pair['long_model_price'] - arb_pair['long_cost']) * arb_pair['long_volume'] * self.opts_data[arb_pair['long_code']].multiplier
        short_profit_spread = (arb_pair['short_cost'] - arb_pair['short_model_price']) * arb_pair['short_volume'] * self.opts_data[arb_pair['short_code']].multiplier
        commission = (2 * arb_pair['long_volume'] + arb_pair['short_volume']) * self.commission_per_unit
        return long_profit_spread + short_profit_spread - commission

    def _realized_profit(self, arb_pair):
        """
        计算给定套利持仓对的已实现盈利
        Parameters:
        --------
        :param arb_pair: pd.Series
            套利持仓对的数据
        :return: float
        --------
            返回套利持仓对的已实现盈利
        """
        long_realized_profit = (arb_pair['long_last'] - arb_pair['long_cost']) * arb_pair['long_volume'] * self.opts_data[arb_pair['long_code']].multiplier
        short_realized_profit = (arb_pair['short_cost'] - arb_pair['short_last']) * arb_pair['short_volume'] * self.opts_data[arb_pair['short_code']].multiplier
        commission = (2 * arb_pair['long_volume'] + arb_pair['short_volume']) * self.commission_per_unit
        return long_realized_profit + short_realized_profit - commission

    def _calc_opt_margin(self):
        """
        计算样本期权和持仓期权的开仓保证金，每个交易日开盘前计算一次
        :return:
        """
        # 1.读取标的日K线时间序列，获取标的的前收盘价
        underlying_quote = pd.read_csv('./data/underlying_daily_quote.csv', index_col=0, parse_dates=[0], encoding='UTF-8')
        underlying_pre_close = float(underlying_quote.loc[self.trading_date, 'pre_close'])
        # 2.读取样本期权的日行情
        cfg = ConfigParser()
        cfg.read('config.ini')
        quote_path = cfg.get('path', 'opt_quote_path')
        strdate = self.trading_date.strftime('%Y-%m-%d')
        strfilepath = Path(quote_path, strdate, '50OptionDailyQuote.csv')
        opts_quote = pd.read_csv(strfilepath, usecols=range(1, 14), parse_dates=[0], encoding='gb18030',
                                 dtype={'option_code': str})
        opts_quote.set_index(keys='option_code', inplace=True)
        # 3.计算样本期权的开仓保证金
        for optcode, opt in self.opts_data.items():
            if optcode in opts_quote.index:
                opt_pre_settle = float(opts_quote.loc[optcode, 'pre_settle'])
                opt.calc_margin(opt_pre_settle, underlying_pre_close)
            else:
                opt.margin = 3000.0
        # 4.计算持仓期权的开仓保证金
        self.opt_holdings.calc_margin(self.trading_date)

    def _calc_opt_greeks(self):
        """
        计算期权基础数据中的希腊字母值
        :return:
        """
        for optcode, copt in self.opts_data.items():
            copt.calc_greeks(self.underlying_price, self.risk_free, self.q, self.hist_vol, self.trading_time)

    def _calc_model_price(self, monitor_data):
        """
        计算给定的单条的monitor_data期权的理论隐含波动率
        :return:
        """
        if self.sv_model == 'sabr':
            if monitor_data['opt_type'] == 'Call':
                alpha = self.vol_par['Call'][0]
                beta = self.vol_par['Call'][1]
                rho = self.vol_par['Call'][2]
                nu = self.vol_par['Call'][3]
            else:
                alpha = self.vol_par['Put'][0]
                beta = self.vol_par['Put'][1]
                rho = self.vol_par['Put'][2]
                nu = self.vol_par['Put'][3]
            model_imp_vol = vsm.SABR(alpha, beta, rho, nu, self.underlying_price, monitor_data['strike'], monitor_data['maturity'])
        else:
            model_imp_vol = 0.
        return model_imp_vol

    def _is_fully_invested(self, trading_datetime):
        """
        是否达到了满仓
        :return: bool
        """
        # if self.opt_holdings.margin_ratio() < self.margin_ratio - 0.01:
        #     return False
        # else:
        #     return True

        money_ocupied = self.opt_holdings.total_margin() + self.opt_holdings.holding_mv(trading_datetime, 1)
        nav = self.opt_holdings.net_asset_value(trading_datetime)
        ratio = money_ocupied / nav
        if ratio < self.margin_ratio - 0.01:
            return False
        else:
            return True

    def _handle_trade_chance(self, long_monitor_data, short_monitor_data, date_time, handle_type):
        """
        处理交易机会
        Parameters:
        --------
        :param long_monitor_data: pd.Series
            多头期权的monitor_data
        :param short_monitor_data: pd.Series
            空头期权的monitor_data
        :param date_time: datetime.datetime
            交易时间
        :param handle_type: str
            处理方式, 'trade'=交易, 'save'=仅保存交易机会数据, 不进行交易
        :return:
        """
        long_opt_code = long_monitor_data['code']
        long_opt_delta = self.opts_data[long_opt_code].greeks.delta
        long_opt_price = long_monitor_data['ask_price']
        short_opt_code = short_monitor_data['code']
        short_opt_delta = self.opts_data[short_opt_code].greeks.delta
        short_opt_price = short_monitor_data['bid_price']
        if abs(short_opt_delta) > abs(long_opt_delta):
            short_opt_volume = 10
            long_opt_volume = int(short_opt_volume * abs(short_opt_delta) / abs(long_opt_delta) + 0.5)
        else:
            long_opt_volume = 10
            short_opt_volume = int(long_opt_volume * abs(long_opt_delta) / abs(short_opt_delta) + 0.5)
        money_ocupied = self.opts_data[short_opt_code].margin * short_opt_volume + long_opt_volume * long_opt_price * self.opts_data[long_opt_code].multiplier
        commission = (2 * long_opt_volume + short_opt_volume) * self.commission_per_unit
        profit_spread = abs(short_monitor_data['short_spread']) * short_opt_volume * self.opts_data[short_opt_code].multiplier \
                        + abs(long_monitor_data['long_spread']) * long_opt_volume * self.opts_data[long_opt_code].multiplier - commission
        expected_return = profit_spread / money_ocupied
        opt_type = self.opts_data[long_opt_code].opt_type
        if handle_type == 'trade':
            if (self.trade_num_in_1min < self.max_trade_num_1min) and (not self._is_fully_invested(date_time)) and (expected_return > self.expected_ret_threshold[opt_type]):
            # if (not self._is_fully_invested()) and expected_return > 0.01:
                trade_datas = []
                trade_price = long_opt_price + 0.0001 * self.trading_slip_point
                trade_value = trade_price * long_opt_volume * self.opts_data[long_opt_code].multiplier
                trade_datas.append(COptTradeData(long_opt_code, 'buy', 'open', trade_price, long_opt_volume,
                                                 trade_value, long_opt_volume*self.commission_per_unit,
                                                 date_time, self.opts_data[long_opt_code]))
                trade_price = short_opt_price - 0.0001 * self.trading_slip_point
                trade_value = trade_price * short_opt_volume * self.opts_data[short_opt_code].multiplier
                trade_datas.append(COptTradeData(short_opt_code, 'sell', 'open', trade_price, short_opt_volume,
                                                 trade_value, 0., date_time, self.opts_data[short_opt_code]))
                self._handle_arb_holding_pairs('add', trade_datas)
        elif handle_type == 'save':
            trade_chance = [date_time.strftime('%Y-%m-%d %H:%M:%S'), opt_type, short_opt_code, short_opt_volume, round(short_opt_price, 4),
                            round(short_opt_delta, 4), long_opt_code, long_opt_volume, round(long_opt_price, 4), round(long_opt_delta, 4),
                            round(profit_spread, 4), round(money_ocupied, 2), round(expected_return, 4)]
            # filepath = Path(self.opt_holdings_path, self.configname, 'trade_chance_%s.csv' % self.portname)
            with open('./data/trade_chance_%s.csv' % self.portname, 'a', newline='', encoding='UTF-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(trade_chance)

    def on_vol_trading(self, start_date, end_date=None):
        """
        指定某一交易日期，进行波动率曲面交易
        :param start_date: datetime.date
            开始日期
        :param end_date: datetime.date
            结束日期
        :return:
        """
        if end_date is None:
            end_date = start_date
        df_tradingdays = self.calendar[(self.calendar.tradingday >= start_date) & (self.calendar.tradingday <= end_date)]
        for _, tradingday_info in df_tradingdays.iterrows():
            trading_day = tradingday_info['tradingday'].date()
            self.trading_date = trading_day
            self.pre_trading_date = self.calendar[self.calendar.tradingday == trading_day].iloc[0]['pre_tradingday']
            self.trading_time = datetime.datetime.combine(trading_day, datetime.time(9, 30, 0))
            # 导入交易相关数据
            self._load_trading_datas(trading_day)
            # 初始化monitor_data
            self._init_monitor_data(trading_day)
            # 遍历标的分钟行情的时间戳, 进行波动率曲面交易
            for date_time, underlying_price in self.underlying_quote_1min.iterrows():
                logging.info('%s searching trade chance.' % date_time.strftime('%Y-%m-%d %H:%M:%S'))
                self.trading_time = date_time
                self.trade_num_in_1min = 0
                if date_time.time() < datetime.time(9, 30, 0):
                    continue
                if date_time.time() > datetime.time(15, 0, 0):
                    break
                self.underlying_price = underlying_price['close']
                self._calc_opt_greeks()
                self.opt_holdings.calc_greeks(self.underlying_price, self.risk_free, self.q, self.hist_vol, date_time)
                # 更新monitor_data
                df_opt_quote = DataFrame()
                for idx, monitor_data in self.monitor_data.iterrows():
                    if len(self.opts_data[monitor_data['code']].quote_1min) == 0:
                        continue
                    opt_quote = self.opts_data[monitor_data['code']].quote_1min.loc[date_time].copy()
                    opt_quote['code'] = monitor_data['code']
                    df_opt_quote = df_opt_quote.append(opt_quote, ignore_index=False)
                df_opt_quote.index.name = 'date_time'
                df_opt_quote.reset_index(inplace=True)
                df_opt_quote.set_index(keys='code', inplace=True)
                # underlying_quote = self.underlying_quote_1min.loc[date_time]
                self._update_monitor_data(df_opt_quote, underlying_price, quote_type='M', update_model=True)
                # scan策略的arb_holding_pairs, 将符合平仓条件的套利持仓对平仓
                self._handle_arb_holding_pairs('scan')
                # 搜索套利机会, 若有, 建仓
                call_shortspread_desc, call_longspread_asc, put_shortspread_desc, put_longspread_asc = self._sorted_arbitrage_spread()
                # 计算认购期权套利机会
                if call_shortspread_desc.iloc[0]['short_spread'] > 0 and call_longspread_asc.iloc[0]['long_spread'] < 0:
                    self._handle_trade_chance(call_longspread_asc.iloc[0], call_shortspread_desc.iloc[0], date_time, 'save')
                    self._handle_trade_chance(call_longspread_asc.iloc[0], call_shortspread_desc.iloc[0], date_time, 'trade')
                if put_shortspread_desc.iloc[0]['short_spread'] > 0 and put_longspread_asc.iloc[0]['long_spread'] < 0:
                    self._handle_trade_chance(put_longspread_asc.iloc[0], put_shortspread_desc.iloc[0], date_time, 'save')
                    self._handle_trade_chance(put_longspread_asc.iloc[0], put_shortspread_desc.iloc[0], date_time, 'trade')
            # 每个交易日结束, 校准随机波动率参数, 保存持仓数据、P&L
            self._calibrate_sv_model()

            self.opt_holdings.calc_margin(trading_day)
            self.opt_holdings.p_and_l(trading_day)
            self.opt_holdings.net_asset_value(trading_day)
            holding_filename = Path(self.opt_holdings_path, self.configname, 'holding_%s_%s.csv' % (self.portname, self.trading_date.strftime('%Y%m%d')))
            self.opt_holdings.save_holdings(holding_filename)
            self._handle_arb_holding_pairs(handle_type='save')
            time.sleep(10)

    def trade_chance_analyzing(self):
        """波动率曲面交易机会分析"""
        # 波动率曲面策略交易机会文件路径
        trade_chance_filepath = '/Users/davidyujun/Dropbox/OptVolTrading/data/trade_chance_VolSurfaceTrade.csv'
        # 导入交易机会数据
        vs_trade_chances = pd.read_csv(trade_chance_filepath, header=0,
                                       dtype={'short_opt_code': str, 'long_opt_code': str}, encoding='UTF-8')
        trading_days = pd.read_csv('/Users/davidyujun/Dropbox/OptVolTrading/data/tradingdays.csv', header=0, encoding='UTF-8')
        pre_trading_day = trading_days.iloc[0]['tradingday']
        df_tradechance_convergence = DataFrame()
        t = 0
        for _, trade_chance in vs_trade_chances.iterrows():
            t += 1
            if t % 10000 == 0:
                print(trade_chance['datetime'][:10], ', ', str(t))
            trading_day = trade_chance['datetime'][:10]
            short_opt_code = trade_chance['short_opt_code']
            short_opt_vol = trade_chance['short_opt_volume']
            short_opt_cost = trade_chance['short_opt_price']
            long_opt_code = trade_chance['long_opt_code']
            long_opt_vol = trade_chance['long_opt_volume']
            long_opt_cost = trade_chance['long_opt_price']
            profit_spread = trade_chance['profit_spread']
            if trading_day != pre_trading_day:
                holding_days = trading_days[trading_days.tradingday >= trading_day].head(6)
                pre_trading_day = trading_day
            ser_tradechance_convergence = Series()
            k = 0
            for holding_day in list(holding_days['tradingday']):
                opt_quote_path = '/Users/davidyujun/Dropbox/opt_quote/%s/50OptionDailyQuote.csv' % holding_day
                if not Path(opt_quote_path).exists():
                    ser_tradechance_convergence = Series()
                    break
                df_opt_quote = pd.read_csv(opt_quote_path, header=0, dtype={'option_code': str}, encoding='GB2312')
                df_opt_quote.set_index('option_code', inplace=True)
                if short_opt_code not in df_opt_quote.index:
                    print(short_opt_code, ' not at day: ', holding_day)
                    ser_tradechance_convergence = Series()
                    break
                if long_opt_code not in df_opt_quote.index:
                    print(long_opt_code, ' not at day: ', holding_day)
                    ser_tradechance_convergence = Series()
                    break
                short_opt_close = df_opt_quote.loc[short_opt_code, 'close']
                long_opt_close = df_opt_quote.loc[long_opt_code, 'close']
                realized_profit_amount = (short_opt_vol * (short_opt_cost - short_opt_close) + long_opt_vol * (
                            long_opt_close - long_opt_cost)) * 10000
                realized_profit_ratio = round(realized_profit_amount / profit_spread, 4)
                day_label = 'day' + str(k)
                ser_tradechance_convergence[day_label] = realized_profit_ratio
                k += 1
            if len(ser_tradechance_convergence) > 0:
                ser_tradechance_convergence['datetime'] = trade_chance['datetime']
                ser_tradechance_convergence['short_opt_code'] = short_opt_code
                ser_tradechance_convergence['long_opt_code'] = long_opt_code
                df_tradechance_convergence = df_tradechance_convergence.append(ser_tradechance_convergence,
                                                                               ignore_index=True)
        df_tradechance_convergence.to_csv('~/Dropbox/OptVolTrading/data/trade_chance_convergence.csv',
                                          index=False, columns=['datetime','short_opt_code','long_opt_code','day0','day1','day2','day3','day4','day5'])


if __name__ == '__main__':
    # pass
    s = CVolSurfaceTradingStrategy('VolSurfaceTrade', 'vol_surface_strategy')
    # s._load_opt_basic_data(datetime.date(2017,1,2))
    # s._init_monitor_data(datetime.date(2017,1,2))
    # s.update_monitor_data(None, None, 'M')
    # print(s.monitor_data)
    # print(s.monitor_data.index)
    # s.calibrate_sv_model(datetime.date(2017, 12, 28), datetime.date(2018, 1, 2))
    s.on_vol_trading(datetime.date(2015, 11, 13), datetime.date(2015, 12, 31))
    # s.trade_chance_analyzing()
