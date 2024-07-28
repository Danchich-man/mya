import requests
import pandas as pd
import numpy as np
import time
import asyncio

from telegram import Bot

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
async def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

# Параметры стратегии
params = {
     'offset': 0.013,  # Offset может принимать отрицательные значения
    'leftBars': 50,
    'rightBars': 5,
    'filterPeriod': 5,
    'priceDropThreshold': 15,
    'averagingDrop': 0.07
}

symbol = 'WIFUSDT'
timeframe = '1'
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

# Telegram параметры
bot_token = '7402328372:AAEeOZBlHv1BGKKng-KnupD3h_bvyOhLzug'
chat_id = '-1002117465854'

# Функция для исполнения торговых сигналов
async def execute_trading():
    # Сообщение в начале торгов
    await send_telegram_message(bot_token, chat_id, "Торги начинаются!")

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
            data['adjustedResistance'] = data['pivot_high'].shift(30) + params['offset']
            data['adjustedSupport'] = data['pivot_low'].shift(30) - params['offset']
            data['buySignal'] = (data['Close'].shift(1) < data['adjustedSupport'].shift(1)) & (data['Close'] > data['adjustedSupport'])
            data['sellSignal'] = (data['Close'].shift(1) > data['adjustedResistance'].shift(1)) & (data['Close'] < data['adjustedResistance'])
            
            # Проверка сигнала на покупку
            if data['buySignal'].iloc[-1] and not trade_open and buy_signal_count < 3:
                last_buy_price = data['Close'].iloc[-1]
                avg_price = last_buy_price
                contracts = initial_investment / last_buy_price  # Рассчитываем количество контрактов на 1 доллар
                buy_signal_count += 1
                can_sell_signal = True
                trades.append({'type': 'buy', 'price': last_buy_price, 'contracts': contracts})
                trade_open = True  # Устанавливаем флаг открытия сделки
                stop_loss_count = 0  # Сброс счётчика стоп-лоссов
                await send_telegram_message(bot_token, chat_id, f"Buy Signal at {last_buy_price}")
            
            # Проверка сигнала на продажу
            elif data['sellSignal'].iloc[-1] and trade_open:
                sell_price = data['Close'].iloc[-1]
                profit = (sell_price - avg_price) * contracts * leverage
                trades.append({'type': 'sell', 'price': sell_price, 'contracts': contracts, 'profit': profit})
                can_sell_signal = False
                trade_open = False  # Сбрасываем флаг открытия сделки
                await send_telegram_message(bot_token, chat_id, f"Sell Signal at {sell_price}. Profit: {profit}")

            # Проверка стоп-лосса
            elif trade_open and (data['Close'].iloc[-1] < last_buy_price * (1 - price_drop_threshold)):
                stop_loss_price = data['Close'].iloc[-1]
                loss = (stop_loss_price - last_buy_price) * contracts * leverage
                trades.append({'type': 'stop_loss', 'price': stop_loss_price, 'contracts': contracts, 'loss': loss})
                stop_loss_count += 1
                trade_open = False  # Сбрасываем флаг открытия сделки
                await send_telegram_message(bot_token, chat_id, f"Stop-Loss Triggered at {stop_loss_price}. Loss: {loss}")

            # Усреднение позиций
            elif trade_open and averaging_count < 3 and (data['Close'].iloc[-1] < last_buy_price * (1 - averaging_drop)):
                average_price = data['Close'].iloc[-1]
                avg_price = (avg_price * contracts + average_price * contracts) / (contracts * 2)
                total_averages += 1
                averaging_count += 1
                trades.append({'type': 'average', 'price': average_price, 'contracts': contracts})
                await send_telegram_message(bot_token, chat_id, f"Averaging Position at {average_price}")

            # Проверка на завершение торгов после 3 стоп-лоссов или 3 сигналов на покупку
            if stop_loss_count >= 3 or buy_signal_count >= 3:
                break

        # Задержка перед следующей итерацией
        await asyncio.sleep(10)  # Ожидание 10 секунд перед следующей проверкой

    # Сообщение в конце торгов
    await send_telegram_message(bot_token, chat_id, "Торги завершены!")

# Запуск функции исполнения торговых сигналов
asyncio.run(execute_trading())
