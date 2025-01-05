import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Data Preprocessing
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    if 'Date' not in data.columns:
        raise ValueError("The 'Date' column is missing in the dataset.")

    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.timestamp)
    data.replace(0, np.nan, inplace=True)
    data.bfill(inplace=True)

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10',
                'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'MACD', 'MACD_signal',
                'MACD_diff', 'RSI', 'BB_high', 'BB_low', 'BB_mid', 'Stochastic_K',
                'Stochastic_D', 'ADX', 'OBV', 'AD_Line', 'CCI', 'ATR', 'Williams_R',
                'Fib_Level_0', 'Fib_Level_23.6', 'Fib_Level_38.2', 'Fib_Level_61.8',
                'Fib_Level_100']

    scaler = MinMaxScaler()
    train_data.loc[:, features] = scaler.fit_transform(train_data[features])
    val_data.loc[:, features] = scaler.transform(val_data[features])
    test_data.loc[:, features] = scaler.transform(test_data[features])

    return train_data, val_data, test_data, scaler

# Create Sequences
def create_sequences(data, sequence_length):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10',
                     'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'MACD', 'MACD_signal',
                     'MACD_diff', 'RSI', 'BB_high', 'BB_low', 'BB_mid', 'Stochastic_K',
                     'Stochastic_D', 'ADX', 'OBV', 'AD_Line', 'CCI', 'ATR', 'Williams_R',
                     'Fib_Level_0', 'Fib_Level_23.6', 'Fib_Level_38.2', 'Fib_Level_61.8',
                     'Fib_Level_100']].values
    target = data['Close'].values

    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

# Model Creation
def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Dropout(0.4),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.4),
        Dense(100, activation='relu'),
        Dense(1)
    ])
    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Train the Model
def train_model(train_data, val_data, sequence_length, batch_size, epochs):
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)

    model = create_model((sequence_length, X_train.shape[2]))

    checkpoint = ModelCheckpoint("cnn_lstm_model.keras", save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    # Save the final model
    model.save("final_cnn_lstm_model.keras")

    return model, history

# Load the Model
def load_saved_model(model_path):
    return load_model(model_path)

# Evaluate the Model
def evaluate_model(model, test_data, sequence_length, scaler):
    X_test, y_test = create_sequences(test_data, sequence_length)
    y_pred = model.predict(X_test)

    y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], 30)), y_pred), axis=1))[:, -1]
    y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 30)), y_test.reshape(-1, 1)), axis=1))[:, -1]

    rmse = np.sqrt(np.mean((y_test_rescaled - y_pred_rescaled) ** 2))
    mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
    accuracy = 100 - (rmse * 100)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    return y_test_rescaled, y_pred_rescaled

# Visualization
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.legend()
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

# Main Execution
file_path = "/content/data.csv"
sequence_length = 100
batch_size = 64
epochs = 50

# Preprocess Data
train_data, val_data, test_data, scaler = preprocess_data(file_path)

# Train the Model
model, history = train_model(train_data, val_data, sequence_length, batch_size, epochs)

# Load the Model
loaded_model = load_saved_model("final_cnn_lstm_model.keras")

# Evaluate and Plot Results
y_test_rescaled, y_pred_rescaled = evaluate_model(loaded_model, test_data, sequence_length, scaler)
plot_results(y_test_rescaled, y_pred_rescaled)
