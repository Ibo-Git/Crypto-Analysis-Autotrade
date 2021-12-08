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


    def asset_pairs_ticker_info(self, asset_code_1, asset_code_2):
        raw_info = Sandbox.asset_pairs_ticker_info_raw(asset_code_1, asset_code_2)
        return {'a': float(raw_info['a'][0]), 'b': float(raw_info['b'][0]) }

        # if asset_code_2 in self.pair_codes[asset_code_1]:
        #     raw_info = Sandbox.asset_pairs_ticker_info_raw(asset_code_1, asset_code_2)

        #     if asset_volume >= 0:
        #         return float(raw_info['a'][0])
        #     else:
        #         return float(raw_info['b'][0])
        # elif asset_code_1 in self.pair_codes[asset_code_2]:
        #     raw_info = Sandbox.asset_pairs_ticker_info_raw(asset_code_2, asset_code_1)

        #     if asset_volume >= 0:
        #         return 1 / (float(raw_info['b'][0]))
        #     else:
        #         return 1 / (float(raw_info['a'][0]))
        # else:
        #     raise Exception('Pair code not available.')


    def trade(self, from_asset_volume, from_asset_code, to_asset_code):
        # Set general asset_codes
        base_asset_code = None
        secondary_asset_code = None
        assets_reversed = None

        if to_asset_code in self.pair_codes[from_asset_code]:
            base_asset_code = from_asset_code
            secondary_asset_code = to_asset_code
            assets_reversed = False
        elif from_asset_code in self.pair_codes[to_asset_code]:
            base_asset_code = to_asset_code
            secondary_asset_code = from_asset_code            
            assets_reversed = True


        transaction_type = 'buy' if from_asset_volume * (-1 if assets_reversed else 1) >= 0 else 'sell'
        base_asset_conversion_rate = self.asset_pairs_ticker_info(base_asset_code, secondary_asset_code)['a' if transaction_type == 'buy' else 'b']
        base_asset_volume = from_asset_volume
        secondary_asset_volume = from_asset_volume * -1 * base_asset_conversion_rate

        if assets_reversed:
            base_asset_volume = (from_asset_volume * -1) / base_asset_conversion_rate
            secondary_asset_volume = from_asset_volume

        # Calculate amounts & fees
        base_fee_percentage = 0.0026
        base_asset_volume_after_fee = base_asset_volume
        secondary_asset_volume_after_fee = secondary_asset_volume
        
        if not assets_reversed:
            # buy/sell the exact base amount -> fees on secondary
            if transaction_type == 'buy':
                # buying exact base amount with secondary currency -> add fee
                secondary_asset_volume_after_fee += secondary_asset_volume * base_fee_percentage
            else:
                # selling exact base amount with secondary currency -> subtract fee
                secondary_asset_volume_after_fee -= secondary_asset_volume * base_fee_percentage
        else:
            # buy/sell the exact secondary amount -> fees on base
            if transaction_type == 'buy':
                # buying base amount with secondary currency -> add fee
                base_asset_volume_after_fee -= base_asset_volume * base_fee_percentage
            else:
                # selling base amount with secondary currency -> subtract fee
                base_asset_volume_after_fee += base_asset_volume * base_fee_percentage

        print(f'trade({from_asset_volume}, {from_asset_code}, {to_asset_code})')
        print(f'transaction_type: {transaction_type}')
        print(f'conversion_rate: {base_asset_conversion_rate}')
        print(f'base_asset_volume: {base_asset_volume}')
        print(f'base_asset_volume_after_fee: {base_asset_volume_after_fee}')        
        print(f'secondary_asset_volume: {secondary_asset_volume}')
        print(f'secondary_asset_volume_after_fee: {secondary_asset_volume_after_fee}')

        # # Update asset volumes
        if (base_asset_code not in self.assets): self.assets[base_asset_code] = Asset(base_asset_code)
        if (secondary_asset_code not in self.assets): self.assets[secondary_asset_code] = Asset(secondary_asset_code)

        self.assets[base_asset_code].volume += base_asset_volume_after_fee
        self.assets[secondary_asset_code].volume += secondary_asset_volume_after_fee

        # Append to history
        self.transaction_history.append(TransactionHistoryEvent(
            transaction_date=datetime.now(),
            transaction_type=transaction_type,
            base_asset_conversion_rate=base_asset_conversion_rate,
            base_asset_volume=base_asset_volume,
            base_asset_volume_after_fee=base_asset_volume_after_fee,
            secondary_asset_volume=secondary_asset_volume,
            secondary_asset_volume_after_fee=secondary_asset_volume_after_fee,
        ))


class Asset:
    def __init__(self, code):
        self.code  = code
        self.volume = 0


class TransactionHistoryEvent:
    def __init__(self, transaction_date, transaction_type, base_asset_conversion_rate, base_asset_volume, base_asset_volume_after_fee, secondary_asset_volume, secondary_asset_volume_after_fee):
            self.transaction_date=transaction_date,
            self.transaction_type=transaction_type,
            self.base_asset_conversion_rate=base_asset_conversion_rate,
            self.base_asset_volume=base_asset_volume,
            self.base_asset_volume_after_fee=base_asset_volume_after_fee,
            self.secondary_asset_volume=secondary_asset_volume,
            self.secondary_asset_volume_after_fee=secondary_asset_volume_after_fee,

def main():
    sandbox = Sandbox()
    sandbox.trade(10, 'XBT', 'EUR')    
    sandbox.trade(-10, 'XBT', 'EUR')
    sandbox.trade(433000, 'EUR', 'XBT')
    sandbox.trade(-433000, 'EUR', 'XBT')


if __name__ == '__main__':
    main()