import numpy as np


class CGreeks(object):
    """期权希腊字母值结构体"""

    __slots__ = ['delta', 'delta_mv', 'gamma', 'gamma_mv', 'vega']

    def __init__(self, delta=0.0, delta_mv=0.0, gamma=0.0, gamma_mv=0.0, vega=0.0):
        self.delta = delta
        self.delta_mv = delta_mv    # delta市值
        self.gamma = gamma
        self.gamma_mv = gamma_mv    # 1%gamma市值
        self.vega = vega


class CSingleOptHolding(object):
    """期权持仓数据结构体"""

    __slots__ = ['holdingside', 'holdingvol', 'COption']

    def __init__(self, side=0, vol=0, opt=None):
        self.holdingside = side     # 持仓方向，1=long, -1=short
        self.holdingvol = vol       # 持仓量
        self.COption = opt          # 持仓期权，COption类


class COptTradeData(object):
    """期权交易数据结构体"""

    __slots__ = ['code', 'tradeside', 'openclose', 'tradeprice', 'tradevol', 'tradevalue', 'commission', 'time', 'opt']

    def __init__(self, code=None, tradeside=None, openclose=None, price=0.0, vol=0, value=0.0, commission=0.0,
                 time=None, opt=None):
        self.code = code                # 代码
        self.tradeside = tradeside      # 交易方向，'buy' or 'sell'
        self.openclose = openclose      # 开平仓，'open' or 'close'
        self.tradeprice = price         # 成交价格
        self.tradevol = vol             # 成交数量
        self.tradevalue = value         # 成交金额
        self.commission = commission    # 佣金
        self.time = time                # 成交时间(datetime.datetime)
        self.opt = opt                  # 交易的期权(COption类)


class Utils(object):

    @classmethod
    def clean_extreme_value(cls, arr_data, method='MAD'):
        """
        对数据进行去极值处理
        :param arr_data: np.array
            需要进行去极值的原始数据, 数组的每一列各自进行去极值极值操作
        :param method: 去极值算法
        :return: np.array
            去极值处理后的数据
        """
        raw_data = arr_data.copy()
        m = np.median(raw_data, axis=0)
        mad = np.median(np.fabs(raw_data - m), axis=0)
        fupper = m + mad * 5.2
        flower = m - mad * 5.2
        for k in range(raw_data.shape[1]):
            if method == 'MAD':
                raw_data[:, k][raw_data[:, k] > fupper[k]] = fupper[k]
                raw_data[:, k][raw_data[:, k] < flower[k]] = flower[k]
        return raw_data
