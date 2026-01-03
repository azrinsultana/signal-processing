import numpy as np
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from modules.chart import generate_signal_plot, plot_trading_signals
import pandas as pd


def prepare_signal(raw_data, short_ma, long_ma, rsi_period, rsi_low, rsi_high,atr_period,atr_mult ):
     dataset,params_dict  = best_combination(


                     shift_signals(raw_data), short_ma, long_ma, rsi_period, rsi_low, rsi_high,atr_period,atr_mult


         )
     print("from the prepare signals", params_dict)
     dataset=correct_signal_with_extreme(dataset,cluster_length=25)
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
    print("generated grid signal", df['signal'].value_counts())

    print(best)
    print("generated signal using the grid search, from best signal", df['signal'].value_counts())
    params_dict = {
        'short_ma': int(best['short_ma']),
        'long_ma': int(best['long_ma']),
        'rsi_period': int(best['rsi_period']),
        'rsi_low': int(best['rsi_low']),
        'rsi_high': int(best['rsi_high']),
        'atr_period': int(best['atr_period']),
        'atr_mult': best['atr_mult']
    }

    print("generated grid signal", df['signal'].value_counts())
    plot_trading_signals(df, "grid search signals", signal_col="signal", limit=10000)

    return df, params_dict
#########




def generate_extreme_signal(df, cluster_length=25):

    close = df["close"].to_numpy()
    signal_extreme = np.zeros(len(df), dtype=np.int8)

    max_idx = argrelextrema(close, np.greater, order=cluster_length)[0]
    min_idx = argrelextrema(close, np.less, order=cluster_length)[0]

    signal_extreme[max_idx] = -1  # Sell
    signal_extreme[min_idx] = 1  # Buy

    return signal_extreme


def correct_signal_with_extreme(df, cluster_length=25):

    df = df.copy()

    # Original signal from your MA+RSI+ATR strategy
    signal_original = df["signal"].to_numpy()

    # Generate extreme signal
    signal_extreme = generate_extreme_signal(df, cluster_length=cluster_length)

    # Replace original signal where extreme signal is BUY/SELL
    corrected_signal = np.where(signal_extreme != 0, signal_extreme, signal_original)

    df["signal"] = corrected_signal

    # Optional: check distribution
    counts = df["signal"].value_counts(normalize=True) * 100
    print("Corrected signal distribution (%):\n", counts.round(2))

    # Plot first 10k signals
    plot_trading_signals(df, title="Corrected Signals with Extreme Overlay", signal_col="signal", limit=10000)

    return df





def visualize_dataset(df, processed, limit=3000):
    df.reset_index(inplace=True, drop=True)
    generate_signal_plot(df, val_limit=limit)
    # generate_signal_plot(generate_signal_only_extrema(df), val_limit=limit)
    generate_signal_plot(shift_signals(df), val_limit=limit)

    generate_signal_plot(processed, val_limit=limit)



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




