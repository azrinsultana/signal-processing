#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import pandas as pd

from modules.chart import generate_signal_plot
from modules.dataset_loader import load_dataset, load_test_dataset
from modules.feature_generate import extract_fast_features
from modules.feature_selection import select_best_features
from modules.models import (
    lgbmodel,
    predict_with_new_dataset, cnn1d_model, cnn_lstm_model
)
from modules.preprocessing import (
    rename_col,
    handling_nan_after_feature_generate,
    prepare_dataset_for_model, rename_test_col
)
from modules.signal_label_processing import (
    prepare_signal,
    visualize_dataset
)
from modules.simulator import run_backtesting_simulator
from modules.utility import (
    load_model,
    save_model,
    find_project_root,
    str2bool
)


# ---------------------------
# VISUALIZATION
# ---------------------------


class SignalMLPipeline:


    def __init__(self, data_dir_, file_name_, test_file_path_, n_features=20, visualize=False):
        self.visualize = visualize
        self.y = None
        self.X = None
        self.data_dir = data_dir_
        self.file_name = file_name_
        self.test_file_path = test_file_path_
        self.n_features = n_features

        # Will be filled during pipeline
        self.raw_data = None
        self.dataset = None
        self.df_features = None
        self.selected_features = None
        self.pipe = None
        # self.model = None
        self.models = {}
        self.reports = {}

        self.short_ma = None
        self.long_ma = None
        self.rsi_period = None
        self.rsi_low = None
        self.rsi_high = None
        self.atr_period = None
        self.atr_mult = None

        self.step_functions = {
            1: ("load", self.load_and_prepare_raw_data),
            2: ("label", self.generate_labels),
            3: ("simulation", self._simulation),
            4: ("visualize", self.visualize_current_dataset),  # ← moved here
            5: ("features", self.extract_features),
            6: ("select", self._feature_selection_wrapper),
            7: ("train", self._train_wrapper),
            8: ("save", self.save),
            9: ("test", self.test_new_dataset)
        }

    # ---------------------------
    # VISUALIZATION WRAPPER
    # ---------------------------
    def visualize_current_dataset(self):
        if not self.visualize:
            print(">>> Visualization disabled (use --visualize true)")
            return

        print(">>> Visualizing dataset with all preprocessing steps...")
        if not hasattr(self, "dataset") or self.dataset is None:
            print("[!] dataset missing → generating labels")
            self.generate_labels()

        visualize_dataset(self.raw_data, self.dataset)

    # wrappers for functions with return values
    def _feature_selection_wrapper(self):
        self.X, self.y = self.feature_selection()
    def _simulation(self):
        simulation_results = run_backtesting_simulator(df=self.dataset)
        print(simulation_results)

    def _train_wrapper(self):
        if not hasattr(self, "X") or not hasattr(self, "y"):
            raise RuntimeError(
                "Features are not generated yet. Run step 4 (select) before step 5 (train)."
            )
        self.train_model(self.X, self.y)

    # ---------------------------------------------------
    # Flexible pipeline with start/end control
    # ---------------------------------------------------
    def run_pipeline(self, start_step_=1, end_step_=9):
        print(f"\n>>> Running pipeline from step {start_step_} to {end_step_}\n")

        for step in range(start_step_, end_step_ + 1):
            if not hasattr(self, "raw_data"):
                print("Auto-running Step 1: LOAD (required for LABEL)")
                self.load_and_prepare_raw_data()

            if not hasattr(self, "dataset"):
                print("Auto-running Step 2: LABEL (required for FEATURES)")
                self.generate_labels()

            # dependency for STEP 4 (feature selection)
            if step == 4:
                if not hasattr(self, "df_features"):
                    print("Auto-running Step 3: FEATURES (required for SELECT)")
                    self.extract_features()

            # dependency for STEP 5 (training)
            if step == 5:
                if not hasattr(self, "X"):
                    print("Auto-running Step 4: SELECT (required for TRAIN)")
                    self._feature_selection_wrapper()

            step_name, func = self.step_functions[step]
            print(f"=== Step {step}: {step_name.upper()} ===")
            func()

    # ---------------------------
    # LOAD + CLEAN DATA
    # ---------------------------
    def load_and_prepare_raw_data(self):
        print("Loading dataset...")
        dt = load_dataset(self.data_dir, self.file_name)
        dt = rename_col(dt)
        self.raw_data = dt
        print(self.raw_data.head(), self.raw_data['signal'].value_counts())

    # ---------------------------
    # SIGNAL GENERATION
    # ---------------------------
    def generate_labels(self):

        signal_params = {
            'raw_data':  self.raw_data,
            'short_ma': self.short_ma,
            'long_ma': self.long_ma,
            'rsi_period': self.rsi_period,
            'rsi_low': self.rsi_low,
            'rsi_high': self.rsi_high,
            'atr_period': self.atr_period,
            'atr_mult': self.atr_mult
        }

        self.dataset , params_dict= prepare_signal(**signal_params)
        print(self.dataset["signal"].value_counts())
        if params_dict:
            self.short_ma = params_dict["short_ma"]
            self.long_ma= params_dict["long_ma"]
            self.rsi_period = params_dict["rsi_period"]
            self.rsi_low = params_dict["rsi_low"]
            self.rsi_high = params_dict["rsi_high"]
            self.atr_period = params_dict["atr_period"]
            self.atr_mult = params_dict["atr_mult"]

        save_path = os.path.join(self.data_dir, 'cleaned_generated_signal.csv')

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.dataset.to_csv(save_path, index=False)
        print(f"Dataset with generated Signal saved at {save_path}")

    # ---------------------------
    # FEATURE EXTRACTION + CLEANING
    # ---------------------------
    def extract_features(self):
        print("Extracting features...")
        df_feat = extract_fast_features(self.dataset, self.short_ma, self.long_ma, self.rsi_period, self.atr_period)
        df_feat = handling_nan_after_feature_generate(df_feat)
        self.df_features = df_feat

    # ---------------------------
    # FEATURE SELECTION
    # ---------------------------
    def feature_selection(self):
        print("Performing feature selection...")

        df = self.df_features.dropna(subset=["signal"]).copy()
        X = df.drop(columns=["signal"])
        y = df["signal"]
        X = X.fillna(X.mean())

        selected_features, votes, masks, pipe = select_best_features(X, y, self.n_features)
        selected_features = selected_features[selected_features != "time"]

        self.selected_features = list(selected_features)
        self.pipe = pipe

        # Save internally
        self.X = X
        self.y = y

        pd.Series(self.selected_features).to_csv("selected_features.csv", index=False)
        print("Selected Features:", self.selected_features)

        return X, y

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    def train_model(self, X, y):
        print("Preparing dataset for model training...")

        # ------------------------------------
        # 1. Select features
        # ------------------------------------
        x_selected = X[self.selected_features].copy()
        print("Selected feature shape:", x_selected.shape)

        # ------------------------------------
        # 2. Preprocess (shared for all models)
        # ------------------------------------
        x_processed, y_mapped, pipe, sample_weight_ = prepare_dataset_for_model(
            x_selected,
            y,
            sample_weight=True
        )

        self.pipe = pipe

        self.models = {}
        self.reports = {}

        print("\n=== Training LightGBM ===")
        lgb_model, lgb_run = lgbmodel(
            x_processed,
            y_mapped,
            sample_weight_,
            report_dir="reports"
        )
        self.models["LightGBM"] = lgb_model

        # ------------------------------------
        # 5. Train 1D CNN
        # ------------------------------------
        print("\n=== Training 1D CNN ===")
        cnn_model, cnn_run = cnn1d_model(
            x_processed,
            y_mapped,
            sample_weight_,
            report_dir="reports"
        )
        self.models["CNN"] = cnn_model

        # ------------------------------------
        # 6. Train CNN + LSTM
        # ------------------------------------
        print("\n=== Training CNN + LSTM ===")
        cnn_lstm_model_, cnn_lstm_run = cnn_lstm_model(
            x_processed,
            y_mapped,
            sample_weight_,
            report_dir="reports"
        )
        self.models["CNN_LSTM"] = cnn_lstm_model_

        print("\n[✓] Training completed for all models:")
        for name in self.models:
            print("   -", name)

    def save(self):

        os.makedirs("saved_models", exist_ok=True)

        save_model(self.pipe, self.selected_features, None)  # no save_dir parameter needed

        for name, model in self.models.items():
            if name == "LightGBM":
                model_path = os.path.join("saved_models", f"{name}_model.pkl")
                import joblib
                joblib.dump(model, model_path)
            else:
                model_path = os.path.join("saved_models", f"{name}_model.keras")
                model.save(model_path)
            print(f"[✓] Saved {name} model → {model_path}")

        print("[✓] All models and pipeline saved successfully.")

    def load(self):

        self.pipe, self.selected_features, _ = load_model(load_dir="saved_models")
        self.models = {}
        for name in ["LightGBM", "CNN", "CNN_LSTM"]:
            if name == "LightGBM":
                model_path = os.path.join("saved_models", f"{name}_model.pkl")
                import joblib
                self.models[name] = joblib.load(model_path)
            else:
                model_path = os.path.join("saved_models", f"{name}_model.keras")
                from keras.models import load_model as keras_load_model
                self.models[name] = keras_load_model(model_path)
            print(f"[✓] Loaded {name} model → {model_path}")

        print("[✓] All models and pipeline loaded successfully.")

    def test_new_dataset(self):
        print(f"Loading external test dataset...{self.test_file_path}")

        # ----------------------------
        # Load & prepare test dataset
        # ----------------------------
        test_df = load_test_dataset(self.test_file_path)
        test_df['signal'] = 0
        test_df = rename_test_col(test_df)

        print("Extracting features from test dataset...")
        test_df_features = extract_fast_features(
            test_df.iloc[-10000:, :],
            self.short_ma, self.long_ma,
            self.rsi_period, self.atr_period
        )
        test_df_features = handling_nan_after_feature_generate(test_df_features)

        x_test = test_df_features[self.selected_features].copy()

        # ----------------------------
        # Predict for all models
        # ----------------------------
        all_results = {}

        for model_name, model in self.models.items():
            print(f"\n>>> Predicting signals using {model_name}...")
            result_df = predict_with_new_dataset(
                x_test, self.pipe, model,
                test_df_features[self.selected_features]
            )

            # Add 'close' for plotting
            result_df['close'] = test_df_features['close']
            result_df.reset_index(drop=True, inplace=True)


            all_results[model_name] = result_df

            # Plot signals for this model
            print(f">>> Generating signal plot for {model_name}...")
            generate_signal_plot(result_df, title=f"{model_name} Signals", val_limit=10000)

        print("\n[✓] Prediction completed for all models.")
        return all_results


    # ---------------------------
    # RUN FULL PIPELINE
    # ---------------------------
    def run_full_pipeline(self):
        self.load_and_prepare_raw_data()
        self.generate_labels()
        self.extract_features()
        X, y = self.feature_selection()
        self.train_model(X, y)
        self.save()
        self.test_new_dataset()

        print("\nPipeline completed successfully.")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal ML Pipeline")

    parser.add_argument("--start", type=str, default="1", help="Start step (number or name)")
    parser.add_argument("--end", type=str, default="9", help="End step (number or name)")
    parser.add_argument("--visualize", type=str2bool, default=False, help="Enable visualization (true/false)")

    args = parser.parse_args()

    step_map = {
        "load": 1,
        "label": 2,
        "visualize": 3,
        "simulation": 4,
        "features": 5,
        "select": 6,
        "train": 7,
        "save": 8,
        "test": 9
    }


    def convert_step(x):
        if x.isdigit():
            return int(x)
        return step_map[x.lower()]


    start_step = convert_step(args.start)
    end_step = convert_step(args.end)

    # Configure paths
    PROJECT_ROOT = find_project_root()
    DATASETS_DIR = PROJECT_ROOT / "datasets"

    if os.path.exists(DATASETS_DIR):
        data_dir = DATASETS_DIR
    else:
        data_dir = r"D:\azrin\education\versity\3rd semester\foundation of computer programming\projects\datasets"

    training_file = "Cleaned_Signal_EURUSD_for_training_635_635_60000.csv"
    test_file = os.path.join(data_dir, 'GBPUSD_H1_20140525_20251021.csv')

    # Create pipeline instance
    pipeline = SignalMLPipeline(
        data_dir_=data_dir,
        file_name_=training_file,
        test_file_path_=test_file,
        n_features=20,
        visualize=args.visualize,

    )

    # Run with step control
    pipeline.run_pipeline(start_step, end_step)
