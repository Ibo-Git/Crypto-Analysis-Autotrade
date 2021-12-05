import requests
from datetime import datetime



# get current bitcoin price
TICKER_API_URL = 'https://api.kraken.com/0/public/Ticker'
FEE_API_URL = 'https://api.kraken.com/0/public/AssetPairs?pair=XXBTZUSD'

API_URLS = {
    'AssetPairTickerInfo': 'https://api.kraken.com/0/public/Ticker'
}

# da fees immer fix, braucht man das nich mehr, kannst einmal auslesen und speichern :D
def get_conversion_fee():
    response = requests.get(FEE_API_URL)
    response_json = response.json()
    return response_json['result']['XXBTZUSD']['fees']

class Sandbox:
    def __init__(self):
        self.assets = {}
        self.transaction_history = []

    def asset_pairs_ticker_info_raw(asset_code_1, asset_code_2):
        # Todo (2021-12-05), cs: Make sure the pairs are concatenated in the correct order (XXBTUSD is fine USDXXBT not)
        pair_code = asset_code_1 + asset_code_2
        data = { 'pair': pair_code }
        response = requests.get(API_URLS['AssetPairTickerInfo'], data)
        response_json = response.json()

        if len(response_json['error']) == 0:
            return response_json['result'][pair_code]
        else:
            raise Exception(response_json['error'][0])


    def asset_pairs_ticker_info(asset_code_1, asset_code_2, price = 'a'):
        raw_info = Sandbox.asset_pairs_ticker_info_raw(asset_code_1, asset_code_2)

        return (float(raw_info[price][0]))

    

    def trade(self, from_asset_code, to_asset_code, from_asset_amount):
        asset_conversion_rate = Sandbox.asset_pairs_ticker_info(from_asset_code, to_asset_code)

        # Handle fees
        fee_percentage = 0.0026
        from_asset_fee_amount = from_asset_amount * fee_percentage
        from_asset_amount_after_fee = from_asset_amount - from_asset_fee_amount        
        to_asset_amount = asset_conversion_rate * from_asset_amount_after_fee

        # Update asset amounts
        if (from_asset_code not in self.assets): self.assets[from_asset_code] = Asset(from_asset_code)
        if (to_asset_code not in self.assets): self.assets[to_asset_code] = Asset(to_asset_amount)

        self.assets[from_asset_code].amount -= from_asset_amount
        self.assets[to_asset_code].amount += to_asset_amount

        # Append to history
        self.transaction_history.append(TransactionHistoryEvent(
            transaction_date=datetime.now(),
            from_asset_code=from_asset_code,
            to_asset_code=to_asset_code,
            from_asset_amount=from_asset_amount,
            to_asset_amount=to_asset_amount,
            asset_conversion_rate=asset_conversion_rate,
            from_asset_fee_amount=from_asset_fee_amount,
            fee_percentage=fee_percentage,            
            from_asset_amount_after_fee=from_asset_amount_after_fee
        ))



class Asset:
    def __init__(self, code):
        self.code  = code
        self.amount = 0

class TransactionHistoryEvent:
    def __init__(self, transaction_date, from_asset_code, to_asset_code, from_asset_amount, to_asset_amount, asset_conversion_rate, from_asset_fee_amount, fee_percentage, from_asset_amount_after_fee):
        self.transaction_date = transaction_date
        self.from_asset_code = from_asset_code
        self.to_asset_code = to_asset_code
        self.from_asset_amount = from_asset_amount
        self.to_asset_amount = to_asset_amount
        self.asset_conversion_rate = asset_conversion_rate
        self.from_asset_fee_amount = from_asset_fee_amount
        self.fee_percentage = fee_percentage
        self.from_asset_amount_after_fee = from_asset_amount_after_fee
        



def main():
    sandbox = Sandbox()
    sandbox.trade('XXBT', 'ZUSD', 100)


if __name__ == '__main__':
    main()