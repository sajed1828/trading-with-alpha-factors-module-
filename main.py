from alpha_zero_101 import Alpha_Zero
from alpha_core import alpha_factor
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import requests
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import ta  # Make sure ta is installed: pip install ta

        
def complute_all_alpha_zero(df):
    for i in range(1,18):
        func= getattr(Alpha_Zero, f'alpha_{i}')
        df[f'alpha_factors_{i}'] = func(df)
    return df.dropna()

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "rsi", "stoch_rsi",
    "macd_line", "macd_signal", "macd_diff", "sma_50", "sma_20",
    "bb_bbm", "bb_bbh", "bb_bbl", "returns", "vwap"
] + [f"alpha_factors_{i}" for i in range(1, 18) if i != 28 and i != 19]  

class genlenDataset(Dataset):
    def __init__(self, df, qen_len=10):
        self.qen_len = qen_len
        self.scaler = MinMaxScaler()
        print("Original df shape:", df.shape)
        print("Missing values per column:\n", df.isna().sum())
        print("Number of inf values:\n", np.isinf(df.select_dtypes(include=[np.number])).sum())
        
        data = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
        scaled = self.scaler.fit_transform(data)
        self.target = scaled[:, FEATURE_COLUMNS.index("close")]
        self.data = scaled

    def __len__(self):
        return len(self.data) - self.qen_len
    
    def __getitem__(self, idx):
        X = self.data[idx: idx+self.qen_len]
        y = self.target[idx+self.qen_len]
        return torch.tensor(X,dtype= torch.float), torch.tensor(y,dtype= torch.float)     

class LSTM_modul(nn.Module):
    def __init__(self, input_size=14, hiddan_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hiddan_size, 6)
        self.droup = nn.Dropout(0.2)
        self.fc = nn.Linear(hiddan_size , 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.droup(out[:, -1 ,:])
        return self.fc(out)   

def train_model(dataloader, model, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device) , y_batch.to(device)
            optimizer.total_loss = 0
            optimizer.zero_grad()
            output = model(x_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(dataloader)

def predict_next(df, model, scaler, seq_len=10):
    df = df[FEATURE_COLUMNS].copy()
    df = df.replace([np.inf, - np.inf], np.nan).dropna()
    
    scaled = scaler.transform(df.values[-seq_len:])

    input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    
    with torch.no_grad():
        predictions =  model(input_tensor).cpu().numpy()
    
    
    dummy = np.zeros((1, len(FEATURE_COLUMNS)))
    close_index = FEATURE_COLUMNS.index("close")
        
    dummy[0, 3] = predictions[0][0]
    inv =scaler.inverse_transform(dummy)
    return inv[0, close_index] 


def get_latest_candles(symbol="BTCUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        data = requests.get(url).json()
    except Exception as e:
        print("❌ Error fetching data:", e)
        return None

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df = df.astype({
        'open': float, 'high': float, 'low': float,
        'close': float, 'volume': float
    })

    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['stoch_rsi'] = ta.momentum.stochrsi(close=df['close'], window=14)
    macd = ta.trend.MACD(close=df['close'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()

    bb_indicator  = ta.volatility.BollingerBands(close=df['close'])
    df['bb_bbm'] = bb_indicator.bollinger_mavg()
    df['bb_bbh'] = bb_indicator.bollinger_hband()
    df['bb_bbl'] = bb_indicator.bollinger_lband()

    df["returns"] = df["close"].pct_change()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
 
    return df.dropna()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modul = LSTM_modul().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(modul.parameters(), lr=0.01)

latest_output = ''

if __name__ == "__main__":
    path = r"C:\Users\User\Documents\clever-trade-bot-ai-main (8)\clever-trade-bot-ai-main\P_project_with_python\Data_sources\Alpha_factors\data.csv"
    #df = get_latest_candles(symbol="BTCUSDT", interval="1m", limit=2000)
    
    df = pd.read_csv(path)
    
    #df = alpha_factor.ta_factor_indcators(df)
    #df = complute_all_alpha_zero(df)
    print(df)
    #df.to_csv('Data.csv')
    df.dropna(inplace=True)  
    print(df)

    input_size = len(FEATURE_COLUMNS)
    modul = LSTM_modul(input_size=input_size).to(device)
    dataset = genlenDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    initial_loss = train_model(dataloader, modul, criterion, optimizer, epochs=3)
    print(f"🧪 Initial training loss: {initial_loss:.4f}")

    while True:
        new_df = get_latest_candles(symbol="BTCUSDT", interval="1m", limit=200)
        if new_df is not None:
            new_df = complute_all_alpha_zero(new_df)
            #new_df["returns"] = new_df["close"].pct_change()
            #typical_price = (new_df["high"] + new_df["low"] + new_df["close"]) / 3
            #new_df["vwap"] = (typical_price * new_df["volume"]).cumsum() / new_df["volume"].cumsum()
            new_df.dropna(inplace=True)

            dataset = genlenDataset(new_df)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            loss = train_model(dataloader, modul, criterion, optimizer, epochs=1)
            prediction = predict_next(new_df, modul, dataset.scaler, seq_len=10)

            trend = "🔼 Up" if prediction > new_df["close"].iloc[-1] else "🔽 Down"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔮 Close ≈ {prediction:.2f} USDT | Loss: {loss:.4f} | Trend: {trend}")

            torch.save(modul.state_dict(), "model.pth") 
        time.sleep(60)

