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

class TradingSystem:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.session = self.connect_to_bybit()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.core = Core()
        self.model = None
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
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
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
            data = self.session.get_kline(
                category="linear",
                symbol=self.config['trading']['symbol'],
                interval=str(self.config['trading']['interval']),
                limit=self.config['trading']['train_window']
            )
            
            if not data or 'result' not in data or not data['result'] or 'list' not in data['result']:
                raise ValueError("Invalid API response format")
                
            df = pd.DataFrame(
                data['result']['list'],
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            )
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            
            logging.info(f"Loaded {len(df)} historical records")
            return df.dropna()
            
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

    def preprocess_data(self, df):
        try:
            features = df[self.config['model']['input_features']]
            scaled = self.scaler.fit_transform(features)
            
            X = np.array([
                scaled[i:i+self.config['trading']['predict_window']] 
                for i in range(len(scaled)-self.config['trading']['predict_window'])
            ], dtype=np.float32)
            
            y = np.array([
                1 if df['close'].iloc[i+self.config['trading']['predict_window']] > df['close'].iloc[i] else 0
                for i in range(len(df)-self.config['trading']['predict_window'])
            ], dtype=np.float32)
            
            return X, y
        except Exception as e:
            logging.error(f"Data preprocessing failed: {str(e)}")
            raise

    def initialize_model(self):
        try:
            model_path = self.config['openvino']['model_path']
            logging.info(f"Checking for model at path: {model_path}")

            if not os.path.exists(model_path):
                logging.warning("Model not found, starting training and export process...")
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

            # 1. Prepare data
            logging.info("Starting data preparation for model training...")
            df = self.get_historical_data()
            X_train, y_train = self.preprocess_data(df)
            logging.info(f"Data prepared. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            # 2. Build model
            logging.info("Building LSTM model architecture...")
            inputs = tf.keras.Input(
                shape=(self.config['trading']['predict_window'],
                       len(self.config['model']['input_features'])),
                name='model_input'
            )
            x = tf.keras.layers.LSTM(
                self.config['model']['hidden_units'],
                name='lstm_layer'
            )(inputs)
            outputs = tf.keras.layers.Dense(
                1, 
                activation='sigmoid',
                name='model_output'
            )(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            logging.info("LSTM model compiled successfully.")

            # 3. Train model
            logging.info("Starting model training...")
            history = model.fit(X_train, y_train,
                     epochs=self.config['model']['train_epochs'],
                     batch_size=self.config['model']['batch_size'],
                     validation_split=0.2)
            logging.info("Model training completed.")

            # Проверка точности
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
            logging.info(f"Training accuracy: {train_acc:.2%}, Validation accuracy: {val_acc:.2%}")
            if train_acc < 0.7 or val_acc < 0.6:
                logging.warning("Model accuracy is low, consider adjusting parameters")

            # 4. Save as SavedModel
            saved_model_path = 'model/temp_saved_model'
            os.makedirs('model', exist_ok=True)
            logging.info(f"Saving TensorFlow model to {saved_model_path}...")
            tf.saved_model.save(model, saved_model_path)
            logging.info("TensorFlow SavedModel saved successfully.")

            # 5. Convert to ONNX
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

            # Verify ONNX model
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

            # 6. Convert to OpenVINO IR
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

            # 7. Load compiled model
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

            # Log available result keys and model outputs
            logging.debug(f"Results keys: {list(results.keys())}")
            logging.debug(f"Model outputs: {[output.any_name for output in self.model.outputs]}")

            # Use the first key from results
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
            # Получаем данные
            df = self.get_historical_data()
            if df.empty:
                logging.error("No historical data received")
                return False
                
            # Подготавливаем входные данные
            latest_data = df[self.config['model']['input_features']][-self.config['trading']['predict_window']:]
            scaled = self.scaler.transform(latest_data)
            
            if np.isnan(scaled).any():
                logging.error("Input data contains NaN values after scaling")
                return False
                
            input_data = np.expand_dims(scaled, axis=0).astype(np.float32)
            
            # Получаем предсказание
            prediction = self.predict(input_data)
            if prediction is None:
                logging.warning("Prediction returned None")
                return False
                
            # Обрабатываем результат
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            if confidence < self.config['model'].get('min_confidence', 0.6):
                logging.info(f"Prediction confidence too low: {confidence:.2%}, skipping trade")
                return False
                
            direction = "BUY" if prediction >= self.config['model']['threshold'] else "SELL"
            logging.info(f"Prediction: {direction} (Confidence: {confidence:.2%}, Value: {prediction:.4f})")
            
            return self.execute_trade(direction)
            
        except Exception as e:
            logging.error(f"Trading cycle error: {str(e)}", exc_info=True)
            return False

    def execute_trade(self, direction):
        try:
            # Получаем баланс
            balance_resp = self.session.get_wallet_balance(
                coin="USDT",
                accountType="UNIFIED"  # or "CONTRACT" depending on your Bybit account type
            )
            balance_list = balance_resp['result']['list']
            usdt_balance = next((item for item in balance_list if item['coin'] == 'USDT'), None)
            if not usdt_balance or 'availableToWithdraw' not in usdt_balance:
                logging.error("USDT balance not found in API response")
                return False
            balance = float(usdt_balance['availableToWithdraw'])
            
            # Получаем текущую цену
            tickers = self.session.get_tickers(
                category="linear",
                symbol=self.config['trading']['symbol']
            )
            price = float(tickers['result']['list'][0]['lastPrice'])
            
            # Рассчитываем количество
            amount = balance * (self.config['trading']['max_trade_percentage'] / 100) / price
            amount = round(amount, 4)
            
            if amount <= 0:
                logging.error("Invalid trade amount calculated")
                return False
                
            # Выполняем ордер
            order = self.session.place_active_order(
                category="linear",
                symbol=self.config['trading']['symbol'],
                side=direction,
                order_type="Market",
                qty=str(amount),
                stopLoss=str(price * (1 - self.config['trading']['stop_loss'] / 100)),
                takeProfit=str(price * (1 + self.config['trading']['take_profit'] / 100))
            )
            
            logging.info(f"Executed {direction} order for {amount} {self.config['trading']['symbol']}")
            return True
            
        except Exception as e:
            logging.error(f"Trade execution failed: {str(e)}", exc_info=True)
            return False

    def run(self):
        logging.info("Starting trading bot")
        try:
            while True:
                start_time = time.time()
                
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
        trading_bot.run()
    except Exception as e:
        logging.error(f"System initialization failed: {str(e)}", exc_info=True)