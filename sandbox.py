import requests





# get current bitcoin price
TICKER_API_URL = 'https://api.coinmarketcap.com/v1/ticker/'

def get_latest_crypto_price(crypto):
  response = requests.get(TICKER_API_URL + crypto)
  response_json = response.json()
  return float(response_json[0]['price_usd'])






class Sandbox:
    def __init__(self, balance = 0):
        self._balance = balance

    @property
    def _balance(self):
        return self._balance

    @_balance.setter
    def balance(self, value):
        self._balance = value

    def buy_crypto(self, name, amount):
        #usd = get_latest_crypto_price(name) # e.g. bitcoin
        #fees = get_fee()
        # set balance
        return

    def sell_crypto(self, name, amount):
        # set balance

        if amount > self.balance:
            raise ValueError('You can not sell more than you have bought.')
        elif amount < 0:
            raise ValueError('You must sell more than 0$.')


def main():
    test123 = Sandbox()

if __name__ == '__main__':
    main()