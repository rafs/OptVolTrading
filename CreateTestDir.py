import os
import shutil

beg_days = 41
end_days = 60
deviations = ['0.005', '0.010', '0.015', '0.020', '0.025', '0.030', '0.035', '0.040']
holding_path = '/Users/davidyujun/Dropbox/opt_holdings'
source_file = '/Users/davidyujun/Dropbox/opt_holdings/vol_trend_strategy/holding_VolTrade_20150206.txt'
for days in range(beg_days, end_days + 1):
    for deviation in deviations:
        path = os.path.join(holding_path, 'vol_trend_%d_%s' % (days, deviation))
        os.makedirs(path)
        target_file = os.path.join(path, 'holding_VolTrade_20150206.txt')
        # os.system('copy %s %s' % (source_file, target_file))
        shutil.copy(source_file, target_file)
        with open(os.path.join(path, 'settings_VolTrade_20150206.ini'), 'wt') as f:
            f.write('[vol_trend_%d_%s]\n' % (days, deviation))
            f.write('call_ratio=none\n')
            f.write('put_ratio=none\n')
