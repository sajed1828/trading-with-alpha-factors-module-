import pandas as pd
import numpy as np
import ta
from data_resorces import data_source

class alpha_factor:
    def __init__(self):
        pass

    @staticmethod    
    def ta_factor_indcators(df):
        df = pd.DataFrame(df, columns=['close', 'high', 'low', 'open', 'volume'])

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
        df['stoch_rsi'] = stoch_rsi.stochrsi()

        # ROC
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()

        return    pd.DataFrame(df, columns=["date", "open","high","low","close","volume",
                                   "returns","vwap", "adv20", "rsi","stoch_rsi","macd","macd_signal",
                                   "macd_diff","sma_50","sma_20","bb_bbm","bb_bbh","bb_bbl"])


    def Volume_factors(self, df):
        df = self.__get_data(df)
        # Add your volume factor logic
        return df.dropna()

    def Volatility_factors(self, df):
        df = self.__get_data(df)
        # Add volatility logic
        return df.dropna()

    def Price_factors(self, df):
        df = self.__get_data(df)
        # Add price-related factors
        return df.dropna()

    def Fundamental_factors(self, df):
        df = self.__get_data(df)
        # Add fundamentals logic
        return df.dropna()

    def __get_data(self, df):
        return data_source.get_link(df)

