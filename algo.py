import requests
import pandas as pd
import numpy as np
import time
from telegram import Bot
import os

# Получение токена и chat_id из переменных окружения
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('CHAT_ID')

# Функция для получения последних N свечей
def kline(symb, tf, N):
    url = 'https://api.bybit.com'
    path = '/v5/market/kline'
    URL = url + path
    end_ds = int(time.time() * 1000)
    batch_size = 200
    dtf = int(tf) * 60 * 1000  # интервал в миллисекундах

    start_ds = end_ds - N * dtf
    end_ds = int(time.time() * 1000)
    
    batch_count = N // batch_size + int(N % batch_size != 0)

    dfs = pd.DataFrame()
    for i in range(batch_count):
        batch_start = start_ds + i * batch_size * dtf
        batch_end = min(end_ds, batch_start + batch_size * dtf)

        params = {
            'category': 'linear',
            'symbol': symb,
            'interval': tf,
            'start': batch_start,
            'end': batch_end
        }

        r = requests.get(URL, params=params)
        data = r.json()

        if 'result' in data and 'list' in data['result']:
            df = pd.DataFrame(data['result']['list'])

            m = pd.DataFrame()
            m['Date'] = pd.to_datetime(df.iloc[:,0].astype(float), unit='ms')
            m['Open'] = df.iloc[:, 1].astype(float)
            m['High'] = df.iloc[:, 2].astype(float)
            m['Low'] = df.iloc[:, 3].astype(float)
            m['Close'] = df.iloc[:, 4].astype(float)
            m['Volume'] = df.iloc[:, 5].astype(float)

            m = m.sort_values(by='Date')

            dfs = pd.concat([dfs, m], ignore_index=True)

    return dfs

# Функция для получения последних 100 свечей
def get_latest_klines(symb, tf, N=100):
    url = 'https://api.bybit.com'
    path = '/v5/market/kline'
    URL = url + path
    end_ds = int(time.time() * 1000)
    dtf = int(tf) * 60 * 1000  # интервал в миллисекундах

    start_ds = end_ds - N * dtf

    params = {
        'category': 'linear',
        'symbol': symb,
        'interval': tf,
        'start': start_ds,
        'end': end_ds
    }

    r = requests.get(URL, params=params)
    data = r.json()

    if 'result' in data and 'list' in data['result']:
        df = pd.DataFrame(data['result']['list'])

        m = pd.DataFrame()
        m['Date'] = pd.to_datetime(df.iloc[:,0].astype(float), unit='ms')
        m['Open'] = df.iloc[:, 1].astype(float)
        m['High'] = df.iloc[:, 2].astype(float)
        m['Low'] = df.iloc[:, 3].astype(float)
        m['Close'] = df.iloc[:, 4].astype(float)
        m['Volume'] = df.iloc[:, 5].astype(float)

        m = m.sort_values(by='Date')

        return m

    return None

# Функция для отправки сообщений в Telegram канал
def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    bot.send_message(chat_id=chat_id, text=message)

# Параметры стратегии
params = {
    'offset': 0.013,  # Offset может принимать отрицательные значения
    'leftBars': 50,
    'rightBars': 5,
    'filterPeriod': 5,
    'priceDropThreshold': 15,
    'averagingDrop': 0.07  # Порог для усреднения (7%)
}

symbol = 'WIFUSDT'
timeframe = '5'
num_bars = 500

data = kline(symbol, timeframe, num_bars)

# Преобразование столбца 'Date' в тип datetime и установка его как индекс
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)  # Убедитесь, что данные отсортированы по времени

# Вычисление уровней поддержки и сопротивления на основе предыдущих данных
def calculate_pivots(df, leftBars, rightBars):
    df['pivot_high'] = df['High'].rolling(window=leftBars + rightBars + 1, center=True).apply(lambda x: x.max() if x.size == leftBars + rightBars + 1 else np.nan)
    df['pivot_low'] = df['Low'].rolling(window=leftBars + rightBars + 1, center=True).apply(lambda x: x.min() if x.size == leftBars + rightBars + 1 else np.nan)
    return df

data = calculate_pivots(data, params['leftBars'], params['rightBars'])

# Сдвиг уровней на 30 баров
data['adjustedResistance'] = data['pivot_high'].shift(40) + params['offset']
data['adjustedSupport'] = data['pivot_low'].shift(40) - params['offset']

# Определение сигналов покупки и продажи
data['buySignal'] = (data['Close'].shift(1) < data['adjustedSupport'].shift(1)) & (data['Close'] > data['adjustedSupport'])
data['sellSignal'] = (data['Close'].shift(1) > data['adjustedResistance'].shift(1)) & (data['Close'] < data['adjustedResistance'])

# Параметры плеча и начальной сделки
leverage = 50
initial_investment = 1  # инвестиция в 1 доллар
price_drop_threshold = params['priceDropThreshold'] / 100  # Порог для стоп-лосса
averaging_drop = params['averagingDrop']  # Порог для усреднения

# Выполнение торговых сигналов в реальном времени
trades = []
last_buy_price = np.nan
buy_signal_count = 0
can_sell_signal = False
trade_open = False  # Флаг для отслеживания открытых сделок
stop_loss_count = 0  # Счётчик стоп-лоссов
contracts = 0
avg_price = 0
averaging_count = 0  # Счётчик усреднений
total_averages = 0  # Общее количество усреднений

while True:
    latest_klines = get_latest_klines(symbol, timeframe)
    
    if latest_klines is not None:
        data = latest_klines  # Перезаписываем данные последними 100 свечами
        data = calculate_pivots(data, params['leftBars'], params['rightBars'])
        data['adjustedResistance'] = data['pivot_high'].shift(40) + params['offset']
        data['adjustedSupport'] = data['pivot_low'].shift(40) - params['offset']
        data['buySignal'] = (data['Close'].shift(1) < data['adjustedSupport'].shift(1)) & (data['Close'] > data['adjustedSupport'])
        data['sellSignal'] = (data['Close'].shift(1) > data['adjustedResistance'].shift(1)) & (data['Close'] < data['adjustedResistance'])

        if not trade_open:
            if data['buySignal'].iloc[-1]:
                buy_signal_count += 1
                last_buy_price = data['Close'].iloc[-1]
                can_sell_signal = True
                trade_open = True
                contracts = (initial_investment * leverage) / last_buy_price  # Вычисление количества контрактов
                avg_price = last_buy_price
                trades.append({'Type': 'BUY', 'Price': last_buy_price, 'Time': data.index[-1], 'Contracts': contracts, 'Investment': initial_investment * leverage})

                message = f"Сигнал на покупку на уровне {last_buy_price:.2f}"
                send_telegram_message(bot_token, chat_id, message)

        if trade_open:
            if can_sell_signal and data['sellSignal'].iloc[-1]:
                sell_price = data['Close'].iloc[-1]
                pnl = (sell_price - last_buy_price) * contracts
                message = f"Сигнал на продажу на уровне {sell_price:.2f}\nPNL: {pnl:.2f}"
                send_telegram_message(bot_token, chat_id, message)
                trades.append({'Type': 'SELL', 'Price': sell_price, 'Time': data.index[-1], 'PNL': pnl, 'Contracts': contracts})
                trade_open = False
                contracts = 0  # Обнуление количества контрактов после закрытия сделки
                avg_price = 0
                total_averages += averaging_count  # Увеличение общего количества усреднений на значение текущего счётчика
                averaging_count = 0  # Сброс счётчика усреднений

            if data['Close'].iloc[-1] < last_buy_price * (1 - price_drop_threshold):
                sell_price = data['Close'].iloc[-1]
                pnl = (sell_price - last_buy_price) * contracts
                message = f"Цена упала на 15% от уровня покупки {last_buy_price:.2f}. Срабатывание стоп-лосса на уровне {sell_price:.2f}\nPNL: {pnl:.2f}"
                send_telegram_message(bot_token, chat_id, message)
                trades.append({'Type': 'STOP LOSS', 'Price': sell_price, 'Time': data.index[-1], 'PNL': pnl, 'Contracts': contracts})
                trade_open = False
                contracts = 0  # Обнуление количества контрактов после стоп-лосса
                avg_price = 0
                total_averages += averaging_count  # Увеличение общего количества усреднений на значение текущего счётчика
                averaging_count = 0  # Сброс счётчика усреднений
                stop_loss_count += 1

            if data['Close'].iloc[-1] < last_buy_price * (1 - averaging_drop):
                averaging_price = data['Close'].iloc[-1]
                buy_signal_count += 1
                contracts += (initial_investment * leverage) / averaging_price  # Увеличение количества контрактов
                avg_price = (avg_price * averaging_count + averaging_price) / (averaging_count + 1)  # Пересчёт средней цены
                averaging_count += 1
                message = f"Сигнал на усреднение на уровне {averaging_price:.2f}"
                send_telegram_message(bot_token, chat_id, message)
                trades.append({'Type': 'AVERAGE', 'Price': averaging_price, 'Time': data.index[-1], 'Contracts': contracts})

    # Задержка между запросами (например, 1 минута)
    time.sleep(300)
