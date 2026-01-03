import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_trading_signals(df, title, signal_col="signal", limit=10000):
    df = df.copy().iloc[:limit].reset_index(drop=True)

    # Resolve signal column safely (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    signal_col = col_map.get(signal_col.lower())

    if signal_col is None:
        raise ValueError("Signal column not found")

    # Force numeric
    df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    print("Signal counts:")
    print(df[signal_col].value_counts())

    x = np.arange(len(df))
    close = df["close"].values
    signal = df[signal_col].values

    plt.figure(figsize=(10, 7))
    plt.plot(x, close, label="Close Price", linewidth=1.5)

    # BUY
    buy_idx = signal == 1
    plt.scatter(
        x[buy_idx],
        close[buy_idx],
        marker="^",
        color="green",
        label="BUY",
        s=25,
        zorder=3
    )

    # SELL
    sell_idx = signal == -1
    plt.scatter(
        x[sell_idx],
        close[sell_idx],
        marker="v",
        color="red",
        label="SELL",
        s=25,
        zorder=3
    )

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def generate_signal_plot(data_plot, title, val_limit=5000000):
    df = data_plot.iloc[:val_limit, :].copy()

    # Auto-detect close column
    close_col = None
    for col in df.columns:
        if col.lower() == "close":
            close_col = col
            break

    if close_col is None:
        raise KeyError("No 'Close' or 'close' column found in dataframe.")

    # Plot Close Prices and signals
    plt.figure(figsize=(10, 7))
    plt.plot(df.index, df[close_col], c='black', alpha=0.7,
             label='Close Price', linewidth=0.5)

    # Combine Buy and Sell Signals
    if "signal" not in df.columns:
        raise KeyError("No 'Signal' column found in dataframe.")

    signals = df[df['signal'] != 0]

    # Plot Sell Signals
    sell_signals = signals[signals['signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals[close_col],
                c='red', label='Sell Signal', marker='o', s=14)

    # Plot Buy Signals
    buy_signals = signals[signals['signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals[close_col],
                c='blue', label='Buy Signal', marker='o', s=14)

    # Chart Customization
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title='Signal Type')
    plt.tight_layout()
    plt.show()


def generate_grid_signal_plot(
    df,
    rsi_low,
    rsi_high,
    atr_mult=1.0,
    window=5000000,
    title="Trading Signal Visualization based on grid search of the values"
):


    df = df.iloc[-window:].copy()

    # -----------------------------
    # Buy / Sell masks
    # -----------------------------
    buy = df[df["signal"] == 1]
    sell = df[df["signal"] == -1]

    # -----------------------------
    # ATR range
    # -----------------------------
    # upper_atr = df["close"] + atr_mult * df["atr"]
    # lower_atr = df["close"] - atr_mult * df["atr"]

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ===== Price + Signal =====
    axes[0].plot(df.index, df["close"], label="Close Price")
    axes[0].scatter(buy.index, buy["close"], marker="^", s=80, label="Buy")
    axes[0].scatter(sell.index, sell["close"], marker="v", s=80, label="Sell")
    axes[0].set_title("Price with Buy/Sell Signals")
    axes[0].legend()
    axes[0].grid(True)

    # ===== RSI =====
    axes[1].plot(df.index, df["rsi"], label="RSI")
    axes[1].axhline(rsi_low, linestyle="--", label="RSI Low")
    axes[1].axhline(rsi_high, linestyle="--", label="RSI High")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("RSI Indicator")
    axes[1].legend()
    axes[1].grid(True)

    # ===== ATR =====
    axes[2].plot(df.index, df["atr"], label="ATR")
    # axes[2].fill_between(
    #     df.index, lower_atr, upper_atr, alpha=0.3, label="ATR Range"
    # )
    axes[2].set_title("ATR Volatility Range")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
