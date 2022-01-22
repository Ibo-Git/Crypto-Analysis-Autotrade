# import math
# import os
# from functools import reduce

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.cuda.amp import autocast
# from torch.utils import data
# from tqdm import tqdm
# import requests

# get_url_ohlc = 'https://api.kraken.com/0/public/OHLC?interval=1&pair=ZRXUSD'
# asset_codes = ['ZRX', '1INCH', 'AAVE', 'GHST', 'ALGO', 'ANKR', 'ANT', 'REP', 'REPV2', 'AXS', 'BADGER', 'BAL', 'BNT', 'BAND', 'BAT', 'XBT', 'BCH', 'ADA', 'CTSI', 'LINK', 'CHZ', 'COMP', 'ATOM', 'CQT', 'CRV', 'DASH', 'MANA', 'XDG', 'DYDX', 'EWT', 'ENJ', 'MLN', 'EOS', 'ETH', 'ETC', 'FIL', 'FLOW', 'GNO', 'ICX', 'INJ', 'KAR', 'KAVA', 'KEEP', 'KSM', 'KNC', 'LSK', 'LTC', 'LPT', 'LRC', 'MKR', 'MINA', 'MIR', 'XMR', 'MOVR', 'NANO', 'OCEAN', 'OMG', 'OXT', 'OGN', 'OXY', 'PAXG', 'PERP', 'DOT', 'MATIC', 'QTUM', 'REN', 'RARI', 'RAY', 'XRP', 'SRM', 'SDN', 'SC', 'SOL', 'XLM', 'STORJ', 'SUSHI', 'SNX', 'TBTC', 'XTZ', 'GRT', 'SAND', 'TRX', 'UNI', 'WAVES', 'WBTC', 'YFI', 'ZEC']


# response = requests.get(get_url_ohlc)
# response = response.json()['result']['ZRXUSD']

# peak = {
#     'start_value': 0
# }

# last_row_set = False
# last_row = None

# for index, row in enumerate(response):
#     row = [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])]
#     if not last_row_set: 
#         last_row_set = True
#         last_row = row
#         continue
        
#     market_value = row[4]
#     last_market_value = last_row[4]

#     # Start recording new peak
#     if market_value > last_market_value and peak['start_value'] == 0:
#         peak = {
#             'start_index': index,
#             'start_value': market_value,
#             'start_timestamp': row[0],
#             'peak_max': -999999
#         }
#     elif market_value <= last_market_value and peak['start_value'] != 0:
#         # Peak finished        
#         if last_market_value > peak['peak_max']:
#             peak['peak_max'] = last_market_value
#             peak['peak_max_timestamp'] = last_row[0]

#             peak['end_index'] = index - 1
#             peak['end_value'] = peak['peak_max']
#             peak['percentage_increase'] = (peak['end_value'] / peak['start_value']) - 1

#             # peak['percentage_increase_after_fees'] = peak['percentage_increase'] - fee_percentage_inout
#             peak['end_timestamp'] = peak['peak_max_timestamp']
#             # peak['timedelta64'] = peak['end_timestamp'] - peak['start_timestamp']
#             # peaks.append(peak)

            
#             plt.plot(range(peak['start_index'], peak['end_index']), [o[4] for o in response[peak['start_index']:peak['end_index']]])

#             # Reset peak
#             peak = {
#                 'start_value': 0
#             }

#     last_row = row

# plt.show()
# breakpoint = None