from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict

app = FastAPI(title="MaveTrade Pattern Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYMBOL_MAP = {
    'XAUUSD': 'GC=F',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'JPY=X',
    'BTCUSD': 'BTC-USD',
}

TIMEFRAME_MAP = {
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
}

class ScanRequest(BaseModel):
    symbol: str
    timeframe: str
    min_pips: float
    direction: str
    start_date: str
    end_date: str

@app.get("/")
async def root():
    return {
        "message": "MaveTrade Pattern Scanner API",
        "status": "online",
        "version": "1.0.0"
    }

@app.post("/scan-patterns")
async def scan_patterns(request: ScanRequest):
    try:
        print(f"📥 Request: {request.symbol} {request.timeframe}")
        
        ticker = SYMBOL_MAP.get(request.symbol, request.symbol)
        interval = TIMEFRAME_MAP.get(request.timeframe, '1h')
        
        print(f"📊 Downloading {ticker}...")
        
        data = yf.download(
            ticker, 
            start=request.start_date, 
            end=request.end_date, 
            interval=interval,
            progress=False
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        print(f"✅ Downloaded {len(data)} candles")
        
        data = calculate_indicators(data)
        movements = identify_movements(data, request.min_pips, request.direction)
        patterns = analyze_patterns(data, movements)
        statistics = calculate_statistics(movements)
        
        return {
            "success": True,
            "total_candles": len(data),
            "total_movements": len(movements),
            "patterns": patterns[:10],
            "date_range": f"{request.start_date} to {request.end_date}",
            "statistics": statistics
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    return data

def identify_movements(data: pd.DataFrame, min_pips: float, direction: str) -> List[Dict]:
    movements = []
    pip_multiplier = 10000
    
    data = data.reset_index()
    data_list = data.to_dict('records')
    
    for i in range(len(data_list) - 10):
        entry_price = data_list[i]['Close']
        max_high = entry_price
        min_low = entry_price
        max_drawdown_pips = 0
        
        for j in range(i + 1, min(i + 11, len(data_list))):
            if data_list[j]['High'] > max_high:
                max_high = data_list[j]['High']
            if data_list[j]['Low'] < min_low:
                min_low = data_list[j]['Low']
            
            if direction in ['buy', 'both']:
                current_drawdown = (entry_price - data_list[j]['Low']) * pip_multiplier
                if current_drawdown > max_drawdown_pips:
                    max_drawdown_pips = current_drawdown
            
            if direction in ['sell', 'both']:
                current_drawdown = (data_list[j]['High'] - entry_price) * pip_multiplier
                if current_drawdown > max_drawdown_pips:
                    max_drawdown_pips = current_drawdown
            
            movement_up = (max_high - entry_price) * pip_multiplier
            movement_down = (entry_price - min_low) * pip_multiplier
            
            entry_time = data_list[i].get('Date') or data_list[i].get('Datetime') or str(i)
            
            if (direction in ['buy', 'both']) and movement_up >= min_pips:
                movements.append({
                    'index': i,
                    'direction': 'buy',
                    'pips': movement_up,
                    'entry_price': entry_price,
                    'target_price': max_high,
                    'duration_bars': j - i,
                    'entry_time': str(entry_time),
                    'max_drawdown': max_drawdown_pips
                })
                break
            
            if (direction in ['sell', 'both']) and movement_down >= min_pips:
                movements.append({
                    'index': i,
                    'direction': 'sell',
                    'pips': movement_down,
                    'entry_price': entry_price,
                    'target_price': min_low,
                    'duration_bars': j - i,
                    'entry_time': str(entry_time),
                    'max_drawdown': max_drawdown_pips
                })
                break
    
    return movements

def analyze_patterns(data: pd.DataFrame, movements: List[Dict]) -> List[Dict]:
    patterns = []
    data_list = data.reset_index().to_dict('records')
    
    rsi_oversold = [
        m for m in movements 
        if m['direction'] == 'buy' and m['index'] > 0
        and not pd.isna(data_list[m['index'] - 1].get('RSI'))
        and data_list[m['index'] - 1]['RSI'] < 35
    ]
    
    if len(rsi_oversold) >= 5:
        buy_movements = [m for m in movements if m['direction'] == 'buy']
        patterns.append({
            'id': 1,
            'conditions': ['RSI < 35', 'Direction: Buy'],
            'winrate': (len(rsi_oversold) / len(buy_movements) * 100) if buy_movements else 0,
            'total_events': len(rsi_oversold),
            'winning_events': len(rsi_oversold),
            'avg_profit': sum(m['pips'] for m in rsi_oversold) / len(rsi_oversold),
            'max_drawdown': 0,
            'avg_duration_hours': sum(m['duration_bars'] for m in rsi_oversold) / len(rsi_oversold),
            'best_hours': []
        })
    
    ema_cross = []
    for m in movements:
        if m['index'] < 2:
            continue
        prev = data_list[m['index'] - 1]
        prev2 = data_list[m['index'] - 2]
        if pd.isna(prev.get('EMA_20')) or pd.isna(prev.get('EMA_50')):
            continue
        if pd.isna(prev2.get('EMA_20')) or pd.isna(prev2.get('EMA_50')):
            continue
        if m['direction'] == 'buy':
            if prev2['EMA_20'] < prev2['EMA_50'] and prev['EMA_20'] > prev['EMA_50']:
                ema_cross.append(m)
        else:
            if prev2['EMA_20'] > prev2['EMA_50'] and prev['EMA_20'] < prev['EMA_50']:
                ema_cross.append(m)
    
    if len(ema_cross) >= 5:
        patterns.append({
            'id': 2,
            'conditions': ['EMA(20) crosses EMA(50)', 'Direction matches cross'],
            'winrate': (len(ema_cross) / len(movements) * 100) if movements else 0,
            'total_events': len(ema_cross),
            'winning_events': len(ema_cross),
            'avg_profit': sum(m['pips'] for m in ema_cross) / len(ema_cross),
            'max_drawdown': 0,
            'avg_duration_hours': sum(m['duration_bars'] for m in ema_cross) / len(ema_cross),
            'best_hours': []
        })
    
    return sorted(patterns, key=lambda x: x['winrate'], reverse=True)

def calculate_statistics(movements: List[Dict]) -> Dict:
    if not movements:
        return {
            'sl_aggressive': 0,
            'sl_balanced': 0,
            'sl_conservative': 0,
            'stats_by_sl': []
        }
    
    drawdowns = [m['max_drawdown'] for m in movements]
    
    sl_aggressive = float(np.percentile(drawdowns, 50))
    sl_balanced = float(np.percentile(drawdowns, 75))
    sl_conservative = float(np.percentile(drawdowns, 90))
    
    stats_by_sl = []
    
    for sl_level, sl_value in [
        ('Aggressive', sl_aggressive),
        ('Balanced', sl_balanced),
        ('Conservative', sl_conservative)
    ]:
        winning_trades = []
        losing_trades = []
        
        for m in movements:
            if m['max_drawdown'] > sl_value:
                losing_trades.append(sl_value)
            else:
                winning_trades.append(m['pips'])
        
        total_trades = len(winning_trades) + len(losing_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = sum(losing_trades) if losing_trades else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        avg_win = (total_profit / win_count) if win_count > 0 else 0
        avg_loss = sl_value
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
        
        stats_by_sl.append({
            'level': sl_level,
            'sl_pips': round(sl_value, 2),
            'win_rate': round(win_rate, 2),
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2)
        })
    
    return {
        'sl_aggressive': round(sl_aggressive, 2),
        'sl_balanced': round(sl_balanced, 2),
        'sl_conservative': round(sl_conservative, 2),
        'stats_by_sl': stats_by_sl
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
