import os 
import sys 
import yfinance as yf 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

trained_models_dir = 'trained_models'
datasets_dir = 'datasets' 

def list_models() -> list[str]:
    if not os.path.exists(trained_models_dir): 
        os.mkdir(trained_models_dir)
        return []
    
    return os.listdir(trained_models_dir)


def download_and_split_dataset(ticker_symbol: str): 
    if ticker_symbol is None: 
        print('please pass a ticker symbol') 

    dataset = yf.download(
        ticker_symbol, 
        period='1y',
        interval='1d',
    )


    data = dataset[['Close']].dropna()

    if not os.path.exists(f'./{datasets_dir}'):
        os.mkdir(f'./{datasets_dir}')    
        
    data.to_csv(f'./{datasets_dir}/{ticker_symbol}_dataset.csv')

    data['Prev_Close'] = data['Close'].shift(1)
    data = data.dropna()

    x = data[['Prev_Close']]
    y = data['Close']

    return train_test_split(x, y, test_size=0.2, shuffle=False)


ticker_symbol = input("enter a ticker symbol :")

models = list_models()

if ticker_symbol in models:
    model: LinearRegression = joblib.load(f'./{trained_models_dir}/{ticker_symbol}')
else: 
    x_train, x_test, y_train, y_test = download_and_split_dataset(ticker_symbol)
    linear_regression = LinearRegression()

    print(f'training {ticker_symbol} model')
    linear_regression.fit(x_train, y_train)

    y_predictions = linear_regression.predict(x_test)

    r2 = r2_score(y_test, y_predictions)
    mae = mean_absolute_error(y_test, y_predictions)
    print(f'training complete for {ticker_symbol}')

    print("\n--- Model Performance ---")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    joblib.dump(linear_regression, f'./{trained_models_dir}/{ticker_symbol}')