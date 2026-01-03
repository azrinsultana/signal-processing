An end-to-end Machine Learning pipeline for financial trading signal generation using classical ML and deep learning models, test with another dataset to verify whether our model can learn complex pattern and produce good output in another secnario.

## signal generation using multiple indicators, MA, RSI, ATR

short_moving_average_periods = [5, 10, 15, 20, 30]

long_moving_average_periods = [40, 60, 80, 100, 150]

rsi_lookback_periods = [7, 14]

rsi_oversold_thresholds = range(20, 41, 5)

rsi_overbought_thresholds = range(60, 91, 5)

atr_lookback_periods = [7, 14]
atr_multipliers = [0.25, 0.5, 0.75, 1.0]

**create a gird search fucntion based on those values which searched to find the most optimal one with 14000 searches.**
Best optmial values are,

short_ma = 30, long_ma = 150, rsi_period = 7, rsi_low = 40, rsi_high = 60, atr_period = 14, atr_mult = 0.25

The geenrated signals ratio are buy:sell: hold: = 16:17:67

The geenrated signal again refined by identifying local highs and lows in the price series using generate_extreme_signal. This produces cleaner, more meaningful BUY and SELL points 

## Output
LightGBM model performance
===== 20% Test Classification Report ======
              precision    recall  f1-score   support

        Hold       0.95      0.86      0.90      8057
        Sell       0.80      0.88      0.83      2294
         Buy       0.76      0.95      0.84      2007

    accuracy                           0.88     12358
   macro avg       0.83      0.90      0.86     12358
weighted avg       0.89      0.88      0.88     12358





