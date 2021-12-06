import requests
from datetime import datetime


API_URLS = {
    'AssetPairTickerInfo': 'https://api.kraken.com/0/public/Ticker',
    'AssetPairs': 'https://api.kraken.com/0/public/AssetPairs'
}

class Sandbox:
    def __init__(self):
        self.assets = {}
        self.transaction_history = []
        self.get_asset_pairs()

    def get_asset_pairs(self):
        response = requests.get(API_URLS['AssetPairs'])
        response_json = response.json()      

        pair_codes = {}

        for key in response_json['result']:
            pair = response_json['result'][key]['wsname'].split('/')
            
            if pair[0] not in pair_codes:
                pair_codes[pair[0]] = {pair[1]}
            else:
                pair_codes[pair[0]].add(pair[1])

        self.pair_codes = pair_codes


    def asset_pairs_ticker_info_raw(asset_code_1, asset_code_2):
        pair_code = asset_code_1 + asset_code_2
        data = { 'pair': pair_code }
        response = requests.get(API_URLS['AssetPairTickerInfo'], data)
        response_json = response.json()

        if len(response_json['error']) == 0:
            for key in response_json['result']:
                return response_json['result'][key]
        else:
            raise Exception(response_json['error'][0])


    def asset_pairs_ticker_info(self, asset_code_1, asset_code_2, asset_amount):
        
        if asset_code_2 in self.pair_codes[asset_code_1]:
            raw_info = Sandbox.asset_pairs_ticker_info_raw(asset_code_1, asset_code_2)

            if asset_amount >= 0:
                return (float(raw_info['a'][0]))
            else:
                return (float(raw_info['b'][0]))

        elif asset_code_1 in self.pair_codes[asset_code_2]:
            raw_info = Sandbox.asset_pairs_ticker_info_raw(asset_code_2, asset_code_1)
            if asset_amount >= 0:
                return 1/(float(raw_info['b'][0]))
            else:
                return 1/(float(raw_info['a'][0]))
        else:
            raise Exception('Pair code not available.')


    def trade(self, from_asset_amount, from_asset_code, to_asset_code):
        asset_conversion_rate = self.asset_pairs_ticker_info(from_asset_code, to_asset_code, from_asset_amount)

        # Handle fees
        fee_percentage = 0.0026
        
        if from_asset_amount >= 0:
            to_asset_amount = asset_conversion_rate * from_asset_amount
            to_asset_fee_amount = to_asset_amount * fee_percentage
            to_asset_amount_after_fee = to_asset_amount + to_asset_fee_amount
        else:
            to_asset_amount = asset_conversion_rate * from_asset_amount
            to_asset_fee_amount = - to_asset_amount * fee_percentage
            to_asset_amount_after_fee = to_asset_amount + to_asset_fee_amount



        # Update asset amounts
        if (from_asset_code not in self.assets): self.assets[from_asset_code] = Asset(from_asset_code)
        if (to_asset_code not in self.assets): self.assets[to_asset_code] = Asset(to_asset_amount)

        self.assets[from_asset_code].amount += from_asset_amount
        self.assets[to_asset_code].amount -= to_asset_fee_amount

        # Append to history
        self.transaction_history.append(TransactionHistoryEvent(
            transaction_date=datetime.now(),
            from_asset_code=from_asset_code,
            to_asset_code=to_asset_code,
            from_asset_amount=from_asset_amount,
            to_asset_amount=to_asset_amount,
            asset_conversion_rate=asset_conversion_rate,
            from_asset_fee_amount=to_asset_fee_amount,
            fee_percentage=fee_percentage,            
            from_asset_amount_after_fee=to_asset_amount_after_fee
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
    sandbox.trade(10, 'USD', 'XBT')
    sandbox.trade(-10, 'USD', 'XBT')


if __name__ == '__main__':
    main()