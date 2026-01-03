An end-to-end Machine Learning pipeline for financial trading signal generation using classical ML and deep learning models, test with another dataset to verify whether our model can learn complex pattern and produce good output in another secnario.

**Step 1: signal generation using multiple indicators, MA, RSI, ATR** 

short_moving_average_periods = [5, 10, 15, 20, 30]

long_moving_average_periods = [40, 60, 80, 100, 150]

rsi_lookback_periods = [7, 14]

rsi_oversold_thresholds = range(20, 41, 5)

rsi_overbought_thresholds = range(60, 91, 5)

atr_lookback_periods = [7, 14]
atr_multipliers = [0.25, 0.5, 0.75, 1.0]

**create a gird search fucntion based on those values which searched to find the most optimal one with 14000 searches.**
Best optmial values are
hort_ma       30.000000
long_ma       150.000000
rsi_period      7.000000
rsi_low        40.000000
rsi_high       60.000000
atr_period     14.000000
atr_mult        0.250000

The geenrated signals ratio are buy:sell: hold: = 16:17:67

