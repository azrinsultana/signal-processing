import numpy as np
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from modules.chart import generate_signal_plot
import pandas as pd
from modules.chart import generate_grid_signal_plot
# def prepare_signal(raw_data, short_ma, long_ma, rsi_period, rsi_low, rsi_high,atr_period,atr_mult ):
#     dataset = filter_by_slope(
#         remove_low_volatility_signals(
#             prior_signal_making_zero(
#                 signal_propagate(
#                     shift_signals(raw_data)
#                 )
#             )
#         )
#     )
#     return dataset
def prepare_signal(raw_data, short_ma, long_ma, rsi_period, rsi_low, rsi_high,atr_period,atr_mult ):
     dataset,params_dict  = best_combination(


                     shift_signals(raw_data), short_ma, long_ma, rsi_period, rsi_low, rsi_high,atr_period,atr_mult


         )
     print("from the perpare singla", params_dict)
     return dataset,params_dict
#########
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df, period):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def generate_signal(df, short_ma, long_ma, rsi_period,
                    rsi_low, rsi_high, atr_period, atr_mult):

    data = df.copy()

    data['ma_short'] = data['close'].rolling(short_ma).mean()
    data['ma_long'] = data['close'].rolling(long_ma).mean()
    data['rsi'] = compute_rsi(data['close'], rsi_period)
    data['atr'] = compute_atr(data, atr_period)

    data['signal'] = 0  # HOLD

    buy_cond = (
        (data['ma_short'] > data['ma_long'] + atr_mult * data['atr']) &
        (data['rsi'] < rsi_low)
    )

    sell_cond = (
        (data['ma_short'] < data['ma_long'] - atr_mult * data['atr']) &
        (data['rsi'] > rsi_high)
    )

    data.loc[buy_cond, 'signal'] = 1
    data.loc[sell_cond, 'signal'] = -1

    return data['signal']
def best_combination(df, short_ma, long_ma, rsi_period, rsi_low, rsi_high,atr_period,atr_mult):
    print("selecting best signal by grid search")
    short_ma_list = [5, 10, 15, 20, 30]
    long_ma_list = [40, 60, 80, 100, 150]

    rsi_periods = [7, 14]
    rsi_lows = range(20, 41, 5)  # wider oversold
    rsi_highs = range(60, 91, 5)  # wider overbought

    atr_periods = [7, 14]
    atr_mults = [0.25, 0.5, 0.75, 1.0]

    results = []

    for s in short_ma_list:
        for l in long_ma_list:
            if s >= l:
                continue
            for rsi_p in rsi_periods:
                for rsi_low in rsi_lows:
                    for rsi_high in rsi_highs:
                        if rsi_low >= rsi_high:
                            continue
                        for atr_p in atr_periods:
                            for atr_m in atr_mults:
                                signals = generate_signal(
                                    df, s, l, rsi_p,
                                    rsi_low, rsi_high,
                                    atr_p, atr_m
                                )

                                counts = signals.value_counts(normalize=True)
                                buy = counts.get(1, 0)
                                sell = counts.get(-1, 0)
                                hold = counts.get(0, 0)

                                score = (
                                        abs(buy - 0.25) +
                                        abs(sell - 0.25) +
                                        abs(hold - 0.50)
                                )

                                results.append({
                                    'short_ma': s,
                                    'long_ma': l,
                                    'rsi_period': rsi_p,
                                    'rsi_low': rsi_low,
                                    'rsi_high': rsi_high,
                                    'atr_period': atr_p,
                                    'atr_mult': atr_m,
                                    'buy_pct': buy,
                                    'sell_pct': sell,
                                    'hold_pct': hold,
                                    'score': score
                                })

    results_df = pd.DataFrame(results)
    best_combinations = results_df.sort_values('score').head(10)
    best = best_combinations.iloc[0]

    df['signal'] = generate_signal(
        df,
        short_ma=int(best['short_ma']),
        long_ma=int(best['long_ma']),
        rsi_period=int(best['rsi_period']),
        rsi_low=int(best['rsi_low']),
        rsi_high=int(best['rsi_high']),
        atr_period=int(best['atr_period']),
        atr_mult=best['atr_mult']
    )
    # df.rename(columns={'Signal': 'signal'}, inplace=True)
    print("generated grid signal", df['signal'].value_counts())
    # generate_grid_signal_plot(
    #     df,
    #     rsi_low=int(best["rsi_low"]),
    #     rsi_high=int(best["rsi_high"]),
    #     atr_mult=best["atr_mult"],
    #     window=1000
    # )
    print(best)
    print("generated signal using the grid search, from best signal",df['signal'].value_counts())
    params_dict = {
        'short_ma': int(best['short_ma']),
        'long_ma': int(best['long_ma']),
        'rsi_period':int(best['rsi_period']),
        'rsi_low': int(best['rsi_low']),
        'rsi_high': int(best['rsi_high']),
        'atr_period': int(best['atr_period']),
        'atr_mult': best['atr_mult']
    }
    return df, params_dict
#########
def visualize_dataset(df, processed, limit=3000):
    df.reset_index(inplace=True, drop=True)
    generate_signal_plot(df, val_limit=limit)
    # generate_signal_plot(generate_signal_only_extrema(df), val_limit=limit)
    generate_signal_plot(shift_signals(df), val_limit=limit)
    # generate_signal_plot(signal_propagate(shift_signals(df)), val_limit=limit)

    # processed = remove_low_volatility_signals(
    #     prior_signal_making_zero(
    #         signal_propagate(
    #             shift_signals(df)
    #         )
    #     )
    # )
    generate_signal_plot(processed, val_limit=limit)
    # generate_signal_plot(filter_by_slope(processed), val_limit=limit)
    # generate_signal_plot(filter_by_slope(processed, look_ahead=30), val_limit=limit)

#
# def filter_by_slope(df, look_ahead=24, slope_threshold=0):
#     df = df.copy()
#     print(f"Before: {df['Signal'].value_counts()}")
#
#     s = df["Signal"].to_numpy()
#     close = df["close"].to_numpy()
#
#     n = look_ahead
#     x = np.arange(n)
#     x_mean = x.mean()
#     x_var = ((x - x_mean) ** 2).sum()
#
#     signal_idx = np.where(s != 0)[0]
#
#     for i in signal_idx:
#         if i + n >= len(close):
#             continue
#
#         y = close[i:i+n]
#         y_mean = y.mean()
#
#         # Fast LR slope
#         slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
#
#         if s[i] == 1 and slope <= slope_threshold:
#             s[i] = 0
#         elif s[i] == -1 and slope >= -slope_threshold:
#             s[i] = 0
#
#     df["Signal"] = s
#     print(
#         "Filtering BUY/SELL by slope: ",
#         np.unique(s, return_counts=True)
#     )
#     return df
#
#
# def generate_signal_only_extrema(df, cluster_length=30):
#     df = df.copy()
#
#     close = df["close"].to_numpy()
#     signal = np.zeros(len(df), dtype=np.int8)
#
#     max_idx = argrelextrema(close, np.greater, order=cluster_length)[0]
#     min_idx = argrelextrema(close, np.less, order=cluster_length)[0]
#
#     signal[max_idx] = -1   # Sell
#     signal[min_idx] = 1    # Buy
#
#     df["Signal"] = signal
#     return df
#
#
# def generate_consecutive_signal_label(df, col='Signal'):
#     df = df.copy()
#     signal = df[col].to_numpy(dtype=int)
#
#     # --- Step 1: Propagate last non-zero signal forward ---
#     propagated_signal = np.zeros_like(signal)
#     last_value = 0
#     for i, current_value in enumerate(signal):
#         if current_value != 0:
#             last_value = current_value
#         propagated_signal[i] = last_value
#
#     return propagated_signal
#
#
# # Propagate signals consecutively
# def signal_propagate(df_signals):
#     df = df_signals.copy()
#
#     s = df["Signal"]
#     df["Signal"] = s.replace(0, np.nan).ffill().fillna(0).astype(int)
#
#     print(f"Filtered Signals After Propagation: {df['Signal'].value_counts()}")
#     return df
#
#
# def prior_signal_making_zero(df_signal, reset_length=5):
#     df = df_signal.copy()
#
#     s = df["Signal"].to_numpy()
#     s_new = s.copy()
#
#     # Detect sign changes (1 → -1 or -1 → 1)
#     flip_idx = np.where(s[1:] * s[:-1] == -1)[0] + 1
#
#     for i in flip_idx:
#         start = max(0, i - reset_length)
#         s_new[start:i+1] = 0
#
#     df["Signal"] = s_new
#     print(f"After nullify prior {reset_length} signal: {np.unique(s_new, return_counts=True)}")
#     return df
#

def shift_signals(df, delay=3):
    """
    Shift non-zero signals forward by `delay` bars (fast, vectorized).
    """
    df = df.copy()

    s = df["signal"].to_numpy()
    shifted = s.copy()

    # Reset all signals
    s[:] = 0

    if delay < len(s):
        s[delay:] = shifted[:-delay]

    df["signal"] = s
    print(f"Shift signals forward by delay {delay} bars.")
    return df


def remove_low_volatility_signals(df, threshold_percentile=20, atr_period=14):
    """
    Set signals to 0 when ATR is below a certain percentile threshold (low volatility).
    """
    df = df.copy()

    # Check required columns
    for col in ['high', 'low', 'close', 'signal']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['ATR'] = atr.average_true_range()

    # Calculate volatility threshold
    atr_threshold = df['ATR'].quantile(threshold_percentile / 100)

    # Mask: where ATR is too low, nullify the signal
    low_volatility = df['ATR'] < atr_threshold
    df.loc[low_volatility, 'signal'] = 0

    df.drop(columns=['ATR'], inplace=True)

    print(f"Low-volatility threshold (ATR percentile {threshold_percentile}%) = {atr_threshold:.6f}")
    print(f"After nullify prior signal: {df['signal'].value_counts()}")
    return df


def generate_atr_sma_signals(df, atr_period=14, atr_multiplier=1.5, sma_period=50, low_vol_percentile=20):
    """
    ATR-based breakout signals with SMA trend filter and low-volatility filter.

    df: DataFrame with 'high', 'low', 'close' columns
    atr_period: ATR lookback period
    atr_multiplier: multiplier for ATR breakout
    sma_period: SMA period for trend filter
    low_vol_percentile: percentile to remove low-volatility signals
    """
    df = df.copy()

    # ---------------------
    # ATR Calculation
    # ---------------------
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['ATR'] = atr.average_true_range()

    # ---------------------
    # SMA Trend Filter
    # ---------------------
    df['SMA'] = df['close'].rolling(sma_period).mean()

    # ---------------------
    # ATR Breakout Bands
    # ---------------------
    df['Upper_Band'] = df['close'].shift(1) + atr_multiplier * df['ATR']
    df['Lower_Band'] = df['close'].shift(1) - atr_multiplier * df['ATR']

    # ---------------------
    # Initial Signals
    # ---------------------
    df['signal'] = 0
    df.loc[df['close'] > df['Upper_Band'], 'signal'] = 1   # Buy
    df.loc[df['close'] < df['Lower_Band'], 'signal'] = -1  # Sell

    # ---------------------
    # Trend Filter
    # ---------------------
    df.loc[(df['signal'] == 1) & (df['close'] < df['SMA']), 'signal'] = 0
    df.loc[(df['signal'] == -1) & (df['close'] > df['SMA']), 'signal'] = 0

    # ---------------------
    # Low-volatility Filter
    # ---------------------
    atr_threshold = df['ATR'].quantile(low_vol_percentile / 100)
    df.loc[df['ATR'] < atr_threshold, 'signal'] = 0

    print(f"Low-volatility threshold (ATR percentile {low_vol_percentile}%) = {atr_threshold:.6f}")
    print(f"Signal counts after trend & volatility filter:\n{df['signal'].value_counts()}")

    # ---------------------
    # Final Output
    # ---------------------
    return df[['close', 'high', 'low', 'ATR', 'SMA', 'Upper_Band', 'Lower_Band', 'signal']]


def ema_crossover_signal(df, fast=9, slow=21):
    df = df.copy()
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['signal'] = 0
    df.loc[df['EMA_fast'] > df['EMA_slow'], 'signal'] = 1
    df.loc[df['EMA_fast'] < df['EMA_slow'], 'signal'] = -1
    
    # Optional: remove conflicting signals with trend SMA(50)
    df['SMA50'] = df['close'].rolling(50).mean()
    df.loc[(df['Signal']==1) & (df['close']<df['SMA50']), 'signal']=0
    df.loc[(df['Signal']==-1) & (df['close']>df['SMA50']), 'signal']=0
    
    return df[['close','EMA_fast','EMA_slow','SMA50','signal']]


def bollinger_signal(df, period=20, std_mult=2):
    df = df.copy()
    df['SMA'] = df['close'].rolling(period).mean()
    df['STD'] = df['close'].rolling(period).std()
    df['Upper'] = df['SMA'] + std_mult*df['STD']
    df['Lower'] = df['SMA'] - std_mult*df['STD']
    
    df['signal'] = 0
    df.loc[df['close'] < df['Lower'], 'signal'] = 1   # Buy
    df.loc[df['close'] > df['Upper'], 'signal'] = -1  # Sell
    
    return df[['close','SMA','Upper','Lower','signal']]


def rsi_signal(df, period=14, lower=30, upper=70):
    df = df.copy()
    df['RSI'] = RSIIndicator(df['close'], period).rsi()
    
    df['signal'] = 0
    df.loc[df['RSI'] < lower, 'signal'] = 1
    df.loc[df['RSI'] > upper, 'signal'] = -1
    
    # Trend filter optional
    df['SMA50'] = df['close'].rolling(50).mean()
    df.loc[(df['signal']==1) & (df['close']<df['SMA50']), 'signal']=0
    df.loc[(df['signal']==-1) & (df['close']>df['SMA50']), 'signal']=0
    
    return df[['close','RSI','SMA50','signal']]


def large_engulfing_signal(df, atr_period=14, sma_period=50, min_body_multiplier=0.5):
    """
    Detect large bullish/bearish engulfing candles and generate signals
    df: DataFrame with 'open', 'high', 'low', 'close'
    atr_period: ATR lookback for filtering small candles
    sma_period: SMA period for trend filter
    min_body_multiplier: min candle body size relative to ATR
    """
    df = df.copy()

    # ---------------------
    # ATR filter for candle size
    # ---------------------
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['ATR'] = atr.average_true_range()
    
    # Compute candle body
    df['Body'] = abs(df['close'] - df['open'])

    # SMA trend filter
    df['SMA'] = df['close'].rolling(sma_period).mean()

    # Initialize signal
    df['signal'] = 0

    # Loop over candles to detect engulfing
    for i in range(1, len(df)):
        prev_open = df.loc[df.index[i-1], 'open']
        prev_close = df.loc[df.index[i-1], 'close']
        curr_open = df.loc[df.index[i], 'open']
        curr_close = df.loc[df.index[i], 'close']
        curr_body = df.loc[df.index[i], 'Body']
        curr_atr = df.loc[df.index[i], 'ATR']

        # Minimum body filter
        if curr_body < curr_atr * min_body_multiplier:
            continue

        # Bullish Engulfing
        if (curr_close > curr_open) and (curr_close > prev_open) and (curr_open < prev_close):
            # Trend filter
            if curr_close > df.loc[df.index[i], 'SMA']:
                df.loc[df.index[i], 'signal'] = 1

        # Bearish Engulfing
        elif (curr_close < curr_open) and (curr_close < prev_open) and (curr_open > prev_close):
            if curr_close < df.loc[df.index[i], 'SMA']:
                df.loc[df.index[i], 'signal'] = -1

    print(f"Engulfing signals generated: {df['signal'].value_counts()}")
    return df[['open','high','low','close','ATR','SMA','Body','signal']]


