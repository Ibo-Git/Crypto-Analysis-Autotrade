import requests
from datetime import datetime



# get current bitcoin price
TICKER_API_URL = 'https://api.kraken.com/0/public/Ticker'
FEE_API_URL = 'https://api.kraken.com/0/public/AssetPairs?pair=XXBTZUSD'

# da fees immer fix, braucht man das nich mehr, kannst einmal auslesen und speichern :D
def get_conversion_fee():
    response = requests.get(FEE_API_URL)
    response_json = response.json()
    return response_json['result']['XXBTZUSD']['fees']

class Sandbox:
    def __init__(self, currencyList):
        self.currencies = currencyList

    def currency_conversion(self, currency_1, currency_2, price = 'a'):
        pair = {'pair': currency_1 + currency_2}
        response = requests.get(TICKER_API_URL, pair)
        response_json = response.json()

        if not response_json['error']:
            return float(response_json['result'][currency_1 + currency_2][price][0])
        else:
            raise Exception(response_json['error'][0])

    def convert(self, currency_1, currency_2, amount_currency_1):
        try:
            # funktioniert nur in eine Richtung: FIX!!! -> Verkaufen von BTC muss umgerechnet werden!  
            value = Sandbox.currency_conversion(currency_1, currency_2)
        except:
            pass

        try:
            if amount_currency_1 > self.currencies[currency_1].balance:
                raise ValueError('You can not sell more than you have bought.')
            elif amount_currency_1  < 0:
                raise ValueError('You must sell more than 0$.')
            
            # add to history
            now = datetime.now()
            transaction_date = now.strftime("%Y/%m/%d - %H:%M:%S")

        except:
            pass


class Currency:
    def __init__(self, code, balance = 0):
        self.code = code
        self.balance = balance



def main():
    test123 = Sandbox()
    test123.convert('XETH', 'XXBT', 100)


if __name__ == '__main__':
    main()