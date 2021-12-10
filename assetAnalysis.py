import yfinance as yf


xbtusd_data = yf.download(tickers='BTC-USD', period = '60d', interval = '2m')
