import yfinance as yf
import datetime
import numpy as np 


xbtusd_data = yf.download(tickers='ATOM1-USD', period = '60d', interval = '2m')

fees_percentage = 0.0026
fee_percentage_inout = 0.00521355524363
min_percentage_increase = 0.05
percentage_risk_margin = fees_percentage
peaks = []

peak = {
    'start_value': 0
}

last_row_set = False
last_row = None

for index, row in xbtusd_data.iterrows():
    # Peak starts when value goes from down to up
    # Peak ends when value goes from up to down

    if not last_row_set: 
        last_row_set = True
        last_row = row
        continue

    market_value = row['Adj Close']
    last_market_value = last_row['Adj Close']

    # Start recording new peak
    if market_value > last_market_value and peak['start_value'] == 0:
        peak = {
            'start_value': market_value,
            'start_dt64': row.name.asm8,
            'peak_max': -999999
        }
    elif market_value <= last_market_value and peak['start_value'] != 0:
        # Peak finished        
        if last_market_value > peak['peak_max']:
            peak['peak_max'] = last_market_value
            peak['peak_max_dt64'] = last_row.name.asm8

        # Only finish peak if risk margin is exceeded
        if ((peak['peak_max'] / market_value) - 1) > percentage_risk_margin:
            peak['end_value'] = peak['peak_max']
            peak['percentage_increase'] = (peak['end_value'] / peak['start_value']) - 1

            # Check if peak is worhty to be added to list
            if peak['percentage_increase'] > min_percentage_increase:
                peak['percentage_increase_after_fees'] = peak['percentage_increase'] - fee_percentage_inout
                peak['end_dt64'] = peak['peak_max_dt64']
                peak['timedelta64'] = peak['end_dt64'] - peak['start_dt64']
                peaks.append(peak)

            # Reset peak
            peak = {
                'start_value': 0
            }    

        

    last_row = row

print(len(peaks))
print(np.timedelta64((sum(p['timedelta64'] for p in peaks) / len(peaks)), 'm'))
print(sum(p['percentage_increase'] for p in peaks))
print(sum(p['percentage_increase_after_fees'] for p in peaks))
end = None

    