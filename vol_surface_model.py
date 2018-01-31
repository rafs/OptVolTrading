#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 期权波动率曲面模型
# @Filename: VolSurface_Trading_SABR
# @Date:   : 2018-01-30 19:14
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import math
from scipy.stats import norm
from scipy import optimize
from collections import Iterable
import numpy as np


def calc_vega(sigma, S, K, r, q, tau, type, mkt_price):
    """
    计算期权的vega
    Parameters:
    --------
    :param S: float or array-like of float
        underlying的价格
    :param K: float or array-like of float
        行权价
    :param r: float
        无风险利率
    :param q: float
        underlying的股息率
    :param sigma: float
        underlying的波动率
    :param tau: float or array-like of float
        maturity
    --------
    :return: float or array-like of float
        期权的vega值
    """
    if isinstance(S, Iterable):
        vega = []
        assert len(S) == len(K), "length of 'S' should equal to length of 'K'"
        assert len(S) == len(tau), "length of 'S' should equal to length of 'tau'"
        for i in range(len(S)):
            d1 = (math.log(S[i]/K[i]) + (r - q + sigma * sigma / 2.0) * tau[i]) / sigma / math.sqrt(tau[i])
            # vega.append(math.exp(-q * tau[i] - d1 * d1 / 2.0) * S[i] * math.sqrt(tau[i] / 2.0 / math.pi) / 100.0)
            vega.append(S[i] * math.exp(-q * tau[i]) * math.sqrt(tau[i]) * norm.pdf(d1) / 100.0)
    elif isinstance(S, float):
        d1 = (math.log(S / K) + (r - q + sigma * sigma / 2.0) * tau) / sigma / math.sqrt(tau)
        # vega = math.exp(-q * tau - d1 * d1 / 2.0) * S * math.sqrt(tau / 2.0 / math.pi) / 100.0
        vega = S * math.exp(-q * tau) * math.sqrt(tau) * norm.pdf(d1) / 100.0
    else:
        vega = None
    return vega

def bs_model(s, k, r, q, sigma, tau, type):
    """
    利用BS model计算欧式期权理论价格
    Parameters:
    --------
    :param s: float
        underlying的价格
    :param k: float
        行权价
    :param r: float
        无风险利率
    :param q: float
        underlying的股息率
    :param sigma: float
        underlying的波动率
    :param tau: float
        maturity
    :param type: str
        期权类型, 'C':认购期权; ‘P’:认沽期权
    --------
    :return: float
        期权的理论价格
    """
    d1 = (math.log(s/k) + (r - q + sigma * sigma / 2.0) * tau) / sigma / math.sqrt(tau)
    d2 = d1 - sigma * math.sqrt(tau)
    if type == 'C':
        opt_value = s * math.exp(-q * tau) * norm.cdf(d1) - k * math.exp(-r * tau) * norm.cdf(d2)
    else:
        opt_value = -s * math.exp(-q * tau) * norm.cdf(-d1) + k * math.exp(-r * tau) * norm.cdf(-d2)
    return opt_value

def imp_vol_calc_objfunc(sigma, s, k, r, q, tau, type, mkt_price):
    """
    计算期权隐含波动率的优化函数
    Parameters:
    --------
    :param sigma: float
        underlying的波动率
    :param s: float
        underlying的价格
    :param k: float
        行权价
    :param r: float
        无风险利率
    :param q: float
        underlying的股息率
    :param tau: float
        maturity
    :param type: str
        期权类型，'C':认购期权; 'P': 认沽期权
    :param mkt_price: float
        期权的市场价格
    --------
    :return:
    """
    return bs_model(s, k, r, q, sigma, tau, type) - mkt_price

def opt_imp_vol(s, k, r, q, tau, type, mkt_price):
    """
    计算期权的隐含波动率
    Parameters:
    --------
    :param s:
    :param k:
    :param r:
    :param q:
    :param tau:
    :param type:
    :param mkt_price:
    --------
    :return:
    """
    # imp_vol = optimize.newton(imp_vol_calc_objfunc, x0=sigma_init, fprime=calc_vega, args=(s, k, r, q, tau, type, mkt_price), tol=0.00001)
    try:
        imp_vol = optimize.bisect(imp_vol_calc_objfunc, 0.01, 2, args=(s, k, r, q, tau, type, mkt_price))
    except RuntimeError:
        imp_vol = 0.0
    # if imp_vol < 0:
    #     imp_vol = 0.0
    return imp_vol

def SABR(alpha, beta, rho, nu, s, k, tau):
    """
    stochastic-alpha-beta-rho model计算期权的隐含波动率
    Parameters:
    --------
    :param alpha: float
        时刻t的瞬时波动率
    :param beta: float
        弹性系数
    :param rho: float
        标的价格随机变量与波动率随机变量之间的相关系数
    :param nu: float
        波动率的波动率
    :param s: float
        标的价格
    :param k: float
        行权价格
    :param tau: float
        期权剩余期限（单位：年）
    --------
    :return: float
        SABR model估计的期权隐含波动率
    """
    if s == k:  # ATM formula
        V = (s * k) ** ((1. - beta)/2.)
        logSK = math.log(s/k)
        A = 1. + ( ((1.-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2.-3.*(rho**2))/24.) ) * tau
        B = 1. + (1./24.)*(((1.-beta)*logSK)**2) + (1./1920.)*(((1-beta)*logSK)**4)
        VOL = (alpah/V) * A
    else:   # not-ATM formula
        V = (s * k) ** ((1. - beta)/2.)
        logSK = math.log(s/k)
        z = (nu/alpha)*V*logSK
        x = math.log( (math.sqrt(1.-2.*rho*z+z**2) + z - rho) / (1.-rho) )
        A = 1. + ( ((1.-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2.-3.*(rho**2))/24.) ) * tau
        B = 1. + (1./24.)*(((1-beta)*logSK)**2) + (1./1920.)*(((1.-beta)*logSK)**4)
        VOL = (nu * logSK * A) / (x * B)
    return VOL

def sabr_objfunc(par, beta, s, K, tau, MKT):
    """
    用于sabr模型校准的对象函数
    :param par: array-like of float
        sarb模型的参数, par[0]:alpha, par[1]:rho, par[2]:nu
    :param beta: float
        sabr模型的beta参数
    :param s: float
        underlying价格
    :param K: array-like of float
        期权行权价
    :param tau: array-like of float
        期权maturity
    :param MKT: array-like of float
        期权的隐含波动率
    --------
    :return: float
        sum square of difference between sabr-vol and imp-vol
    """
    sum_sq_diff = 0.
    for j in range(len(K)):
        sabr_vol = SABR(par[0], beta, par[1], par[2], s, K[j], tau[j])
        diff = sabr_vol - MKT[j]
        sum_sq_diff += diff**2
    obj = math.sqrt(sum_sq_diff)
    return obj

def sabr_calibration(beta, s, K, tau, MKT):
    """
    对sabr模型进行校准
    Parameters:
    --------
    :param beta: float
        sabr模型的beta参数值
    :param s: float
        underlying的价格
    :param K: array-like of float
        期权的行权价
    :param tau: array-like of float
        期权maturity
    :param MKT: array-like of float
        期权的隐含波动率
    --------
    :return: tuple of float
        经过市场隐含波动率校准后的sabr模型参数, (alpha, rho, nu)
    """
    starting_par = np.array([0.001, 0, 0.001])    # 参数[alpha, rho, nu]的初始值
    bnds = ((0.001, None), (-0.999, 0.999), (0.001, None))
    res = optimize.minimize(sabr_objfunc, starting_par, args=(beta, s, K, tau, MKT), bounds=bnds, method='SLSQP')
    return (res.x[0], res.x[1], res.x[2])


if __name__ == '__main__':
    # pass
    s = 3.113
    k = 3.1
    r = 0.035101
    q = 0.0
    tau = 29/365
    type = 'C'
    mkt_price = 0.0667
    imp_vol = opt_imp_vol(s, k, r, q, tau, type, mkt_price)
    print('imp_vol = ', imp_vol)
