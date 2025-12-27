import matplotlib.pyplot as plt


def generate_signal_plot(data_plot, val_limit=5000000):
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
    if "Signal" not in df.columns:
        raise KeyError("No 'Signal' column found in dataframe.")

    signals = df[df['Signal'] != 0]

    # Plot Sell Signals
    sell_signals = signals[signals['Signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals[close_col],
                c='red', label='Sell Signal', marker='o', s=14)

    # Plot Buy Signals
    buy_signals = signals[signals['Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals[close_col],
                c='blue', label='Buy Signal', marker='o', s=14)

    # Chart Customization
    plt.title('Buy/Sell Signals on Close Price on test data', fontsize=16)
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
