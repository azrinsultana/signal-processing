import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from modules.utility import save_classification_report, create_run_id
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
def xgbmodel(X_processed, y_mapped, sample_weight=None, report_dir="reports"):
    run_id = create_run_id()
    # ------------------------------------
    # 2. 80-20 chronological split
    # ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_mapped, test_size=0.2, shuffle=False
    )

    # ------------------------------------
    # 3. Model
    # ------------------------------------
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss"
    )

    # ------------------------------------
    # 4. Fit on training data
    # ------------------------------------

    if sample_weight is not None:
        xgb_model.fit(X_train, y_train, sample_weight=sample_weight[:len(y_train)])
    else:
        xgb_model.fit(X_train, y_train)

    # ------------------------------------
    # 5. Evaluate on test (20%)
    # ------------------------------------
    y_pred = xgb_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Hold', 'Sell', 'Buy'])
    print("====== 20% Test Classification Report ======")
    print(report)

    report_path = save_classification_report(
        report_text=report,
        run_id=run_id,
        report_dir=report_dir,
        metadata={
            "Train size": len(X_train),
            "Test size": len(X_test),
            "Model": "XGBoost"
        }
    )

    print(f"[✓] Report saved → {report_path}")

    return xgb_model, run_id





def lgbmodel(X_processed, y_mapped, sample_weight=None, report_dir="reports"):
    run_id = create_run_id()


    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_mapped, test_size=0.2, shuffle=False
    )

    # ------------------------------------
    # 3. LightGBM Model
    # ------------------------------------
    lgb_clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    if sample_weight is not None:
        lgb_clf.fit(X_train, y_train, sample_weight=sample_weight[:len(y_train)])
    else:
        lgb_clf.fit(X_train, y_train)

    y_pred = lgb_clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Hold', 'Sell', 'Buy'])
    print("====== 20% Test Classification Report ======")
    print(report)

    report_path = save_classification_report(
        report_text=report,
        run_id=run_id,
        report_dir=report_dir,
        metadata={
            "Train size": len(X_train),
            "Test size": len(X_test),
            "Model": "LightGBM"
        }
    )
    print(f"[✓] Report saved → {report_path}")

    return lgb_clf, run_id

################ CNN ############

def cnn1d_model(X_processed, y_mapped, sample_weight=None, report_dir="reports"):
    run_id = create_run_id()

    # ------------------------------------
    # 1. Ensure NumPy arrays
    # ------------------------------------
    X = np.asarray(X_processed, dtype=np.float32)
    y = np.asarray(y_mapped, dtype=np.int32)

    # If labels are (-1,0,1) → map to (0,1,2)
    unique_labels = np.unique(y)
    if set(unique_labels) == {-1, 0, 1}:
        label_map = {-1: 1, 0: 0, 1: 2}
        y = np.vectorize(label_map.get)(y)

    # One-hot encoding
    y_cat = to_categorical(y, num_classes=3)

    # ------------------------------------
    # 2. 80-20 chronological split
    # ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, shuffle=False
    )

    # Sample weights
    if sample_weight is not None:
        sw_train = sample_weight[:len(y_train)]
    else:
        sw_train = None


    # (samples, timesteps/features, channels)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # ------------------------------------
    # 4. Build 1D CNN
    # ------------------------------------
    model = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=X_train.shape[1:]),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ------------------------------------
    # 5. Train
    # ------------------------------------
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        sample_weight=sw_train,
        verbose=1
    )

    # ------------------------------------
    # 6. Evaluate
    # ------------------------------------
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=["Hold", "Sell", "Buy"]
    )

    print("====== 20% Test Classification Report ======")
    print(report)

    report_path = save_classification_report(
        report_text=report,
        run_id=run_id,
        report_dir=report_dir,
        metadata={
            "Train size": len(X_train),
            "Test size": len(X_test),
            "Model": "1D CNN"
        }
    )

    print(f"[✓] Report saved → {report_path}")

    return model, run_id


############# hybrid cnn-lstm ##########

def cnn_lstm_model(X_processed, y_mapped, sample_weight=None, report_dir="reports"):
    run_id = create_run_id()

    # ------------------------------------
    # 1. Ensure NumPy arrays
    # ------------------------------------
    X = np.asarray(X_processed, dtype=np.float32)
    y = np.asarray(y_mapped, dtype=np.int32)

    # Map (-1, 0, 1) → (0, 1, 2)
    if set(np.unique(y)) == {-1, 0, 1}:
        y = np.vectorize({-1: 1, 0: 0, 1: 2}.get)(y)

    y_cat = to_categorical(y, num_classes=3)

    # ------------------------------------
    # 2. 80-20 chronological split
    # ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, shuffle=False
    )

    if sample_weight is not None:
        sw_train = sample_weight[:len(y_train)]
    else:
        sw_train = None


    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # ------------------------------------
    # 4. CNN + LSTM Architecture
    # ------------------------------------
    model = Sequential([

        # CNN feature extractor
        Conv1D(32, kernel_size=3, activation="relu",
               input_shape=X_train.shape[1:]),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Temporal modeling
        LSTM(64, return_sequences=False),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ------------------------------------
    # 5. Train
    # ------------------------------------
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        sample_weight=sw_train,
        verbose=1
    )

    # ------------------------------------
    # 6. Evaluate
    # ------------------------------------
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=["Hold", "Sell", "Buy"]
    )

    print("====== 20% Test Classification Report ======")
    print(report)

    report_path = save_classification_report(
        report_text=report,
        run_id=run_id,
        report_dir=report_dir,
        metadata={
            "Train size": len(X_train),
            "Test size": len(X_test),
            "Model": "CNN + LSTM"
        }
    )

    print(f"[✓] Report saved → {report_path}")

    return model, run_id


###########  attension LSTM ##########

def cnn_lstm_model(X_processed, y_mapped, sample_weight=None, report_dir="reports"):
    run_id = create_run_id()

    # -------------------------------
    # Hyperparameters (internal)
    # -------------------------------
    cnn_filters = [32, 64]
    cnn_kernel_sizes = [3, 3]
    lstm_units = 64
    dense_units = 64
    dropout_rate = 0.3
    learning_rate = 0.001
    batch_size = 256
    epochs = 50

    # -------------------------------
    # 1. Ensure NumPy arrays
    # -------------------------------
    X = np.asarray(X_processed, dtype=np.float32)
    y = np.asarray(y_mapped, dtype=np.int32)

    # Map (-1, 0, 1) → (0, 1, 2)
    if set(np.unique(y)) == {-1, 0, 1}:
        y = np.vectorize({-1: 1, 0: 0, 1: 2}.get)(y)

    y_cat = to_categorical(y, num_classes=3)

    # -------------------------------
    # 2. 80-20 chronological split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, shuffle=False
    )

    sw_train = sample_weight[:len(y_train)] if sample_weight is not None else None

    # -------------------------------
    # 3. Reshape for CNN + LSTM
    # -------------------------------
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # -------------------------------
    # 4. Build CNN + LSTM Model
    # -------------------------------
    model = Sequential()
    input_shape = X_train.shape[1:]

    # CNN layers
    for i, (filters, kernel) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
        if i == 0:
            model.add(Conv1D(filters, kernel_size=kernel, activation="relu", input_shape=input_shape))
        else:
            model.add(Conv1D(filters, kernel_size=kernel, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

    # LSTM layer
    model.add(LSTM(lstm_units, return_sequences=False))

    # Dense + Dropout
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------------
    # 5. Train
    # -------------------------------
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        sample_weight=sw_train,
        verbose=1
    )

    # -------------------------------
    # 6. Evaluate
    # -------------------------------
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=["Hold", "Sell", "Buy"]
    )
    print("====== 20% Test Classification Report ======")
    print(report)

    report_path = save_classification_report(
        report_text=report,
        run_id=run_id,
        report_dir=report_dir,
        metadata={
            "Train size": len(X_train),
            "Test size": len(X_test),
            "Model": "CNN + LSTM",
            "CNN filters": cnn_filters,
            "CNN kernels": cnn_kernel_sizes,
            "LSTM units": lstm_units,
            "Dense units": dense_units,
            "Dropout": dropout_rate,
            "Learning rate": learning_rate,
            "Batch size": batch_size,
            "Epochs": epochs
        }
    )
    print(f"[✓] Report saved → {report_path}")

    return model, run_id




def predict_with_new_dataset(X_new, pipe, model, test_df_features):
    # ------------------------------------
    # 1. Preprocess
    # ------------------------------------
    X_new_processed = pipe.transform(X_new)

    X_new_processed = pd.DataFrame(
        X_new_processed,
        columns=test_df_features.columns,
        index=test_df_features.index
    )

    # ------------------------------------
    # 2. Handle CNN vs non-CNN
    # ------------------------------------
    X_input = X_new_processed.values.astype(np.float32)

    # CNN → needs (N, F, 1)
    if hasattr(model, "input_shape") and len(model.input_shape) == 3:
        X_input = X_input[..., np.newaxis]

    # ------------------------------------
    # 3. Predict
    # ------------------------------------
    y_pred = model.predict(X_input)

    # CNN outputs probabilities → convert to class index
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    unique, counts = np.unique(y_pred, return_counts=True)
    print(dict(zip(unique, counts)), len(y_pred), len(test_df_features))

    # ------------------------------------
    # 4. Map classes → trading signals
    # ------------------------------------
    mapping = {0: 0, 1: -1, 2: 1}  # Hold, Sell, Buy
    y_pred_labels = pd.Series(y_pred).map(mapping)

    y_pred_labels.index = test_df_features.index

    test_df_features = test_df_features.copy()
    test_df_features["signal"] = y_pred_labels

    return test_df_features
