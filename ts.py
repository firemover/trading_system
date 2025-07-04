import sys
sys.path.append('/opt/intel/openvino/python/python3.9')

import os
import shutil
import numpy as np
import pandas as pd
import json
import logging
import time
import subprocess
import onnx
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pybit.unified_trading import HTTP
from openvino.runtime import Core, serialize, PartialShape
import colorama
from colorama import Fore, Style
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import requests
import math

colorama.init(autoreset=True)

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

class SquareBar(TqdmCallback):
    COLORS = [
        Fore.GREEN,  # completed
        Fore.YELLOW, # running
        Fore.RED     # error
    ]
    def __init__(self, epochs, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.bar = None

    def on_train_begin(self, logs=None):
        self.bar = tqdm(
            total=self.epochs,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} Epochs',
            ncols=40,
            leave=True
        )

    def on_epoch_end(self, epoch, logs=None):
        color = self.COLORS[0]
        squares = color + '■' * (epoch + 1) + Style.RESET_ALL
        squares += ' ' * (self.epochs - (epoch + 1))
        self.bar.set_description_str(squares)
        self.bar.update(1)
        if epoch + 1 == self.epochs:
            self.bar.close()

    def on_train_end(self, logs=None):
        if self.bar:
            self.bar.close()

class Backtester:
    def __init__(self, trading_system):
        self.ts = trading_system
        self.results = []
        self.trade_history = []
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.current_position = None
        self.commission_rate = 0.0004
        
    def run_backtest(self, start_date=None, end_date=None):
        try:
            logging.info("Starting backtest...")
            
            df = self.ts.get_historical_data()
            
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            if len(df) < self.ts.config['trading']['predict_window'] * 2:
                raise ValueError("Not enough data for backtesting")
            
            X, y = self.ts.preprocess_data(df)
            
            for i in tqdm(range(len(X)), desc="Backtesting"):
                window_data = df.iloc[i:i+self.ts.config['trading']['predict_window']]
                current_price = window_data['close'].iloc[-1]
                next_price = df['close'].iloc[i+self.ts.config['trading']['predict_window']]
                
                features = window_data[self.ts.config['model']['input_features']]
                scaled = self.ts.scaler.transform(features)
                input_data = np.expand_dims(scaled, axis=0).astype(np.float32)
                
                prediction = self.ts.predict(input_data)
                if prediction is None:
                    continue
                
                direction = "Buy" if prediction >= self.ts.config['model']['threshold'] else "Sell"
                confidence = prediction if direction == "Buy" else 1 - prediction
                
                if confidence < self.ts.config['model'].get('min_confidence', 0.6):
                    continue
                
                self.execute_trade_simulation(
                    direction=direction,
                    current_price=current_price,
                    next_price=next_price,
                    timestamp=window_data.index[-1]
                )
            
            report = self.generate_report(df)
            logging.info("Backtest completed successfully")
            return report
            
        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}", exc_info=True)
            raise
    
    def execute_trade_simulation(self, direction, current_price, next_price, timestamp):
        if self.current_position is not None:
            self.close_position(next_price, timestamp)
        
        amount = self.current_balance * (self.ts.config['trading']['max_trade_percentage'] / 100) / current_price
        amount = round(amount, 8)
        
        if direction == "Buy":
            stop_loss = current_price * (1 - self.ts.config['trading']['stop_loss'] / 100)
            take_profit = current_price * (1 + self.ts.config['trading']['take_profit'] / 100)
            entry_type = "long"
        else:
            stop_loss = current_price * (1 + self.ts.config['trading']['stop_loss'] / 100)
            take_profit = current_price * (1 - self.ts.config['trading']['take_profit'] / 100)
            entry_type = "short"
        
        trade = {
            'entry_time': timestamp,
            'entry_price': current_price,
            'direction': direction,
            'entry_type': entry_type,
            'amount': amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'exit_time': None,
            'exit_price': None,
            'pnl': None,
            'pnl_pct': None,
            'exit_reason': None
        }
        
        self.current_position = trade
        self.trade_history.append(trade.copy())
        
        logging.info(
            f"Backtest {direction} at {current_price:.2f}, "
            f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
            f"Amount: {amount:.4f}, Balance: {self.current_balance:.2f}"
        )
    
    def close_position(self, price, timestamp, reason="market"):
        if self.current_position is None:
            return
        
        position = self.current_position
        entry_price = position['entry_price']
        amount = position['amount']
        
        if position['direction'] == "Buy":
            pnl = (price - entry_price) * amount
        else:
            pnl = (entry_price - price) * amount
        
        pnl -= (entry_price * amount * self.commission_rate)
        pnl -= (price * amount * self.commission_rate)
        
        pnl_pct = (pnl / (entry_price * amount)) * 100
        self.current_balance += pnl
        
        position['exit_time'] = timestamp
        position['exit_price'] = price
        position['pnl'] = pnl
        position['pnl_pct'] = pnl_pct
        position['exit_reason'] = reason
        
        logging.info(
            f"Backtest close {position['direction']} at {price:.2f}, "
            f"PnL: {pnl:.2f} ({pnl_pct:.2f}%), "
            f"New balance: {self.current_balance:.2f}, "
            f"Reason: {reason}"
        )
        
        self.current_position = None
    
    def generate_report(self, df):
        if not self.trade_history:
            return {"error": "No trades were executed during backtest"}
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] <= 0]) if losing_trades > 0 else 0
        profit_factor = -avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio(df)
        
        report = {
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": self.trade_history
        }
        
        logging.info("\n" + "="*50)
        logging.info("BACKTEST REPORT")
        logging.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logging.info(f"Initial Balance: {self.initial_balance:.2f}")
        logging.info(f"Final Balance: {self.current_balance:.2f} ({total_pnl_pct:.2f}%)")
        logging.info(f"Total Trades: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.2%}")
        logging.info(f"Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}")
        logging.info(f"Profit Factor: {profit_factor:.2f}")
        logging.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logging.info("="*50 + "\n")
        
        return report
    
    def calculate_max_drawdown(self):
        if not self.trade_history:
            return 0
        
        balance = self.initial_balance
        peak = balance
        max_drawdown = 0
        
        for trade in self.trade_history:
            balance += trade['pnl']
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def calculate_sharpe_ratio(self, df):
        if not self.trade_history:
            return 0
        
        daily_pnl = {}
        for trade in self.trade_history:
            date = trade['exit_time'].date()
            if date in daily_pnl:
                daily_pnl[date] += trade['pnl']
            else:
                daily_pnl[date] = trade['pnl']
        
        if not daily_pnl:
            return 0
        
        pnl_series = pd.Series(daily_pnl.values())
        avg_daily_return = pnl_series.mean()
        std_daily_return = pnl_series.std()
        
        if std_daily_return == 0:
            return 0
        
        sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(365)
        return sharpe_ratio

class TradingSystem:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.session = self.connect_to_bybit()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.core = Core()
        self.model = None
        self.force_retrain = False
        self.cached_data = None
        logging.info("Initializing TradingSystem class and model...")
        self.initialize_model()

    def load_config(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Config load error: {str(e)}")
            raise

    def setup_logging(self):
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            level=self.config['logging']['log_level'].upper(),
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        color_formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(color_formatter)
        logging.getLogger('').addHandler(console)
        logging.info("Trading system initialized")

    def connect_to_bybit(self):
        try:
            session = HTTP(
                api_key=self.config['bybit_api']['api_key'],
                api_secret=self.config['bybit_api']['api_secret'],
                testnet=self.config['bybit_api']['testnet'],
                recv_window=self.config['bybit_api']['recv_window']
            )
            logging.info("Bybit API connection established")
            return session
        except Exception as e:
            logging.error(f"Bybit connection failed: {str(e)}")
            raise

    def get_historical_data(self):
        try:
            if self.cached_data is not None:
                return self.cached_data

            total_needed = self.config['trading']['train_window']
            all_records = []
            last_start = None
            symbol = self.config['trading']['symbol']
            interval = str(self.config['trading']['interval'])
            max_limit = 1000

            while len(all_records) < total_needed:
                limit = min(max_limit, total_needed - len(all_records))
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                }
                if last_start:
                    params["start"] = last_start
                data = self.session.get_kline(**params)
                if not data or 'result' not in data or not data['result'] or 'list' not in data['result']:
                    raise ValueError("Invalid API response format")
                batch = data['result']['list']
                if not batch:
                    break
                all_records = batch + all_records
                last_start = int(batch[-1][0]) - 1

            df = pd.DataFrame(
                all_records,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            )
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            upper_band, lower_band = self.calculate_bollinger_bands(df['close'])
            df['upper_band'] = upper_band
            df['lower_band'] = lower_band
            logging.info(f"Loaded {len(df)} historical records")
            self.cached_data = df.dropna()
            return self.cached_data
        except Exception as e:
            logging.error(f"Historical data error: {str(e)}")
            raise

    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def preprocess_data(self, df):
        try:
            features = df[self.config['model']['input_features']]
            scaled = self.scaler.fit_transform(features)
            
            X = np.array([
                scaled[i:i+self.config['trading']['predict_window']] 
                for i in range(len(scaled)-self.config['trading']['predict_window']])
            ], dtype=np.float32)
            
            y = np.array([
                1 if df['close'].iloc[i+self.config['trading']['predict_window']] > df['close'].iloc[i] else 0
                for i in range(len(df)-self.config['trading']['predict_window']])
            ], dtype=np.float32)
            
            return X, y
        except Exception as e:
            logging.error(f"Data preprocessing failed: {str(e)}")
            raise

    def initialize_model(self):
        try:
            model_path = self.config['openvino']['model_path']
            logging.info(f"Checking for model at path: {model_path}")

            if not os.path.exists(model_path) or self.force_retrain:
                logging.warning("Model not found or retrain forced, starting training...")
                self.train_and_save_model()
            else:
                device = self.select_device()
                logging.info(f"Loading OpenVINO model from {model_path} on device {device}")
                model = self.core.read_model(model_path)
                self.model = self.core.compile_model(model, device)
                logging.info("Model loaded successfully in OpenVINO runtime.")
                logging.info(f"Model inputs: {[input.any_name for input in self.model.inputs]}")
                logging.info(f"Model outputs: {[output.any_name for output in self.model.outputs]}")
                logging.info(f"Model loaded on {device} device")

                df = self.get_historical_data()
                features = df[self.config['model']['input_features']]
                self.scaler.fit(features)
                logging.info("MinMaxScaler fitted on historical data for inference.")
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}", exc_info=True)
            raise

    def select_device(self):
        available_devices = self.core.available_devices
        logging.info(f"Available OpenVINO devices: {available_devices}")
        device = self.config['openvino']['device']
        logging.info(f"Configured device: {device}")
        if device == "AUTO":
            selected = "MYRIAD" if "MYRIAD" in available_devices else "CPU"
            logging.info(f"Device AUTO selected: {selected}")
            return selected
        if device not in available_devices:
            logging.error(f"Device {device} not available. Available: {available_devices}")
            raise RuntimeError(f"Device {device} not available")
        logging.info(f"Selected device: {device}")
        return device

    def train_and_save_model(self):
        try:
            import tensorflow as tf
            from tf2onnx import convert

            logging.info("Starting data preparation for model training...")
            df = self.get_historical_data()
            X_train, y_train = self.preprocess_data(df)
            logging.info(f"Data prepared. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            logging.info("Building LSTM model architecture...")
            inputs = tf.keras.Input(
                shape=(self.config['trading']['predict_window'],
                       len(self.config['model']['input_features'])),
                name='model_input'
            )
            x = tf.keras.layers.LSTM(
                self.config['model']['hidden_units'],
                return_sequences=True,
                name='lstm_layer_1'
            )(inputs)
            x = tf.keras.layers.LSTM(
                self.config['model']['hidden_units']//2,
                name='lstm_layer_2'
            )(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            outputs = tf.keras.layers.Dense(
                1, 
                activation='sigmoid',
                name='model_output'
            )(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            logging.info("LSTM model compiled successfully.")

            logging.info("Starting model training...")
            history = model.fit(
                X_train, y_train,
                epochs=self.config['model']['train_epochs'],
                batch_size=self.config['model']['batch_size'],
                validation_split=0.2,
                callbacks=[SquareBar(self.config['model']['train_epochs'], verbose=0)]
            )
            logging.info("Model training completed.")

            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
            logging.info(f"Training accuracy: {train_acc:.2%}, Validation accuracy: {val_acc:.2%}")
            if train_acc < 0.7 or val_acc < 0.6:
                logging.warning("Model accuracy is low, consider adjusting parameters")

            saved_model_path = 'model/temp_saved_model'
            os.makedirs('model', exist_ok=True)
            logging.info(f"Saving TensorFlow model to {saved_model_path}...")
            tf.saved_model.save(model, saved_model_path)
            logging.info("TensorFlow SavedModel saved successfully.")

            onnx_output_path = 'model/temp.onnx'
            logging.info("Starting ONNX conversion...")
            convert_command = [
                "python", "-m", "tf2onnx.convert",
                "--saved-model", saved_model_path,
                "--output", onnx_output_path,
                "--opset", "13"
            ]
            subprocess.run(convert_command, check=True)
            logging.info("ONNX conversion completed successfully.")

            try:
                onnx_model = onnx.load(onnx_output_path)
                inputs = [input.name for input in onnx_model.graph.input]
                outputs = [output.name for output in onnx_model.graph.output]
                logging.info(f"ONNX model inputs: {inputs}")
                logging.info(f"ONNX model outputs: {outputs}")
                onnx.checker.check_model(onnx_model)
                logging.info("ONNX model is valid")
                input_name = inputs[0]
                logging.info(f"Using input name for OpenVINO conversion: {input_name}")
            except Exception as e:
                logging.error(f"ONNX model verification failed: {str(e)}")
                raise

            logging.info("Starting OpenVINO IR conversion...")
            ov_model = self.core.read_model(onnx_output_path)
            try:
                new_shape = PartialShape([1, self.config['trading']['predict_window'], 
                                       len(self.config['model']['input_features'])])
                ov_model.reshape({input_name: new_shape})
                logging.info(f"Model reshaped successfully for input: {input_name}")
            except Exception as e:
                logging.error(f"Model reshape failed: {str(e)}")
                raise

            serialize(ov_model, self.config['openvino']['model_path'])
            logging.info(f"OpenVINO IR model saved to {self.config['openvino']['model_path']}")

            device = self.select_device()
            logging.info(f"Compiling OpenVINO model for device: {device}")
            self.model = self.core.compile_model(ov_model, device)
            logging.info(f"Model successfully loaded and compiled on {device} device")

        except Exception as e:
            logging.error(f"Model training or export failed: {str(e)}", exc_info=True)
            raise
        finally:
            if os.path.exists(onnx_output_path):
                os.remove(onnx_output_path)
                logging.info(f"Removed temporary ONNX file: {onnx_output_path}")
            if os.path.exists(saved_model_path):
                shutil.rmtree(saved_model_path)
                logging.info(f"Removed temporary SavedModel directory: {saved_model_path}")

    def predict(self, input_data):
        try:
            if np.isnan(input_data).any():
                logging.error("Input data contains NaN values")
                return None

            if input_data.shape[0] != 1:
                input_data = np.expand_dims(input_data, axis=0)

            input_name = next(iter(self.model.inputs)).any_name
            results = self.model.infer_new_request({input_name: input_data})

            logging.debug(f"Results keys: {list(results.keys())}")
            logging.debug(f"Model outputs: {[output.any_name for output in self.model.outputs]}")

            output_key = list(results.keys())[0]
            prediction = results[output_key][0][0]

            if np.isnan(prediction):
                logging.error("Prediction returned NaN")
                return None

            logging.debug(f"Raw prediction value: {prediction:.4f}")
            return float(prediction)

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}", exc_info=True)
            return None

    def run_trading_cycle(self):
        try:
            if self.has_open_position():
                logging.info("Skipping cycle: position already open.")
                return False

            df = self.get_historical_data()
            if df.empty:
                logging.error("No historical data received")
                return False

            latest_data = df[self.config['model']['input_features']][-self.config['trading']['predict_window']:]
            scaled = self.scaler.transform(latest_data)

            if np.isnan(scaled).any():
                logging.error("Input data contains NaN values after scaling")
                return False

            input_data = np.expand_dims(scaled, axis=0).astype(np.float32)

            prediction = self.predict(input_data)
            if prediction is None:
                logging.warning("Prediction returned None")
                return False

            confidence = prediction if prediction >= 0.5 else 1 - prediction
            if confidence < self.config['model'].get('min_confidence', 0.6):
                logging.info(f"Prediction confidence too low: {confidence:.2%}, skipping trade")
                self.force_retrain = True
                return False

            direction = "Buy" if prediction >= self.config['model']['threshold'] else "Sell"
            logging.info(f"Prediction: {direction} (Confidence: {confidence:.2%}, Value: {prediction:.4f})")

            return self.execute_trade(direction)

        except Exception as e:
            logging.error(f"Trading cycle error: {str(e)}", exc_info=True)
            return False

    def get_symbol_step(self, symbol):
        try:
            info = self.session.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            if (
                info
                and "result" in info
                and "list" in info["result"]
                and len(info["result"]["list"]) > 0
            ):
                instrument = info["result"]["list"][0]
                min_qty = float(instrument.get("lotSizeFilter", {}).get("minOrderQty", 0.0))
                qty_step = float(instrument.get("lotSizeFilter", {}).get("qtyStep", 0.0))
                return min_qty, qty_step
            else:
                logging.warning(f"Could not fetch instrument info for {symbol}, using default step 0.001")
                return 0.001, 0.001
        except Exception as e:
            logging.error(f"Failed to get symbol step: {str(e)}")
            return 0.001, 0.001

    def execute_trade(self, direction):
        try:
            balance_resp = self.session.get_wallet_balance(
                coin="USDT",
                accountType="UNIFIED"
            )
            logging.debug(f"UNIFIED wallet balance response: {balance_resp}")

            account_list = balance_resp.get('result', {}).get('list', [])
            if not account_list or 'coin' not in account_list[0]:
                logging.error("No account or coin info in API response. Full response: %s", balance_resp)
                return False

            coin_list = account_list[0]['coin']
            usdt_balance = next((item for item in coin_list if item.get('coin') == 'USDT'), None)
            logging.debug(f"UNIFIED USDT balance entry: {usdt_balance}")

            balance_str = usdt_balance.get('availableToWithdraw') or usdt_balance.get('walletBalance') or usdt_balance.get('equity')
            if not usdt_balance or not balance_str or balance_str == "":
                logging.error("USDT balance not found or empty in API response. Full response: %s", balance_resp)
                return False
            balance = float(balance_str)

            tickers = self.session.get_tickers(
                category="linear",
                symbol=self.config['trading']['symbol']
            )
            price = float(tickers['result']['list'][0]['lastPrice'])

            min_qty, qty_step = self.get_symbol_step(self.config['trading']['symbol'])

            amount = balance * (self.config['trading']['max_trade_percentage'] / 100) / price
            amount = max(min_qty, amount)
            amount = (int(amount / qty_step)) * qty_step
            amount = round(amount, 8)

            if amount < min_qty:
                logging.error(f"Trade amount {amount} is below minimum lot size {min_qty}")
                return False

            if direction == "Buy":
                stop_loss = price * (1 - self.config['trading']['stop_loss'] / 100)
                take_profit = price * (1 + self.config['trading']['take_profit'] / 100)
            else:
                stop_loss = price * (1 + self.config['trading']['stop_loss'] / 100)
                take_profit = price * (1 - self.config['trading']['take_profit'] / 100)

            order = self.session.place_order(
                category="linear",
                symbol=self.config['trading']['symbol'],
                side=direction,
                orderType="Market",
                qty=str(amount),
                stopLoss=str(stop_loss),
                takeProfit=str(take_profit)
            )

            logging.info(f"Executed {direction} order for {amount} {self.config['trading']['symbol']}")
            return True

        except Exception as e:
            logging.error(f"Trade execution failed: {str(e)}", exc_info=True)
            return False

    def has_open_position(self):
        try:
            positions = self.session.get_positions(
                category="linear",
                symbol=self.config['trading']['symbol']
            )
            for pos in positions.get('result', {}).get('list', []):
                if float(pos.get('size', 0)) != 0:
                    logging.info(f"Open position detected: {pos}")
                    return True
            return False
        except Exception as e:
            logging.error(f"Failed to check open positions: {str(e)}")
            return False

    def backtest(self, start_date=None, end_date=None):
        backtester = Backtester(self)
        return backtester.run_backtest(start_date, end_date)

    def run(self):
        logging.info("Starting trading bot")
        retrain_interval = self.config['model'].get('retrain_interval_hours', 1) * 3600
        position_retrain_interval = self.config['model'].get('position_retrain_interval_hours', retrain_interval / 3600) * 3600

        last_retrain = time.time()
        last_position_retrain = None
        position_open_time = None

        try:
            while True:
                start_time = time.time()

                if self.has_open_position():
                    if position_open_time is None:
                        position_open_time = time.time()
                        logging.info("Open position detected, skipping trading cycles.")
                    else:
                        elapsed = time.time() - position_open_time
                        if (last_position_retrain is None) or (time.time() - last_position_retrain >= position_retrain_interval):
                            if elapsed >= position_retrain_interval:
                                logging.info("Open position held too long, retraining model by interval (position)...")
                                self.train_and_save_model()
                                self.initialize_model()
                                last_position_retrain = time.time()
                                position_open_time = time.time()
                    logging.info("Skipping cycle: position already open.")
                    time.sleep(60)
                    continue
                else:
                    position_open_time = None

                if (time.time() - last_retrain >= retrain_interval) and (
                    last_position_retrain is None or time.time() - last_position_retrain >= position_retrain_interval
                ):
                    logging.info("Retraining model by schedule...")
                    self.train_and_save_model()
                    self.initialize_model()
                    last_retrain = time.time()

                if getattr(self, "force_retrain", False):
                    logging.info("Forcing retraining due to low prediction confidence...")
                    self.train_and_save_model()
                    self.initialize_model()
                    last_retrain = time.time()
                    self.force_retrain = False

                if not self.run_trading_cycle():
                    logging.warning("Cycle failed, retrying in 1 minute")
                    time.sleep(60)
                    continue

                cycle_time = time.time() - start_time
                sleep_time = max(0, self.config['trading']['interval'] * 60 - cycle_time)
                logging.info(f"Cycle completed in {cycle_time:.2f}s, next in {sleep_time:.1f}s")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logging.info("Trading system stopped by user")
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        trading_bot = TradingSystem()
        
        # Запуск бэктеста
        backtest_report = trading_bot.backtest(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Сохранение отчета в файл
        with open("backtest_report.json", "w") as f:
            json.dump(backtest_report, f, indent=2, default=str)
        
        # Запуск реальной торговли (опционально)
        # trading_bot.run()
        
    except Exception as e:
        logging.error(f"System initialization failed: {str(e)}", exc_info=True)