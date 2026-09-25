import yaml

from src.data_loader import load_market_data
from src.features import FEATURE_COLUMNS, add_technical_features, build_training_frame
from src.liquidity import add_liquidity_features
from src.macro import add_macro_features
from src.models import predict_probabilities, train_signal_model
from src.multitimeframe import add_multitimeframe_features
from src.regime import add_multitimeframe_regime
from src.session import add_session_features
from src.session_entry import add_session_entry_filter

cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
market_cfg = cfg['market']
model_cfg = cfg['model']

df = load_market_data(
    symbol=market_cfg['symbol'],
    start_date=market_cfg['start_date'],
    end_date=market_cfg['end_date'],
    use_live_data=cfg['trading'].get('use_live_data', False),
)

enriched = add_technical_features(df)
enriched = add_multitimeframe_features(enriched)
enriched = add_session_features(enriched)
enriched = add_session_entry_filter(enriched, cfg.get('execution', {}).get('entry_windows') or None)
enriched = add_liquidity_features(enriched)
enriched = add_multitimeframe_regime(enriched)
enriched = add_macro_features(enriched)

enriched[FEATURE_COLUMNS] = enriched[FEATURE_COLUMNS].replace([float('inf'), float('-inf')], float('nan')).fillna(0.0)
X, y = build_training_frame(enriched, horizon=model_cfg['horizon'], threshold=model_cfg['threshold'])
model = train_signal_model(X, y)
probs = predict_probabilities(model, X)
enriched['signal_probability'] = probs

def compose_signal(row, probability):
    regime_bias = float(row.get('regime_bias', 0.0))
    macro_bias = float(row.get('macro_bias', 0.0))
    session_bias = float(row.get('session_bias', 0.0))
    liquidity_bias = float(row.get('liquidity_sweep', 0.0))
    trend_bias = 1.0 if float(row['sma_fast']) >= float(row['sma_slow']) else -1.0
    is_entry_session = bool(row.get('is_entry_session', True))

    long_ok = (
        is_entry_session
        and probability >= cfg['model']['probability_threshold']
        and (regime_bias >= 0 or trend_bias >= 0)
        and macro_bias >= 0
        and session_bias >= 0
    )
    short_ok = (
        is_entry_session
        and probability <= (1.0 - cfg['model']['probability_threshold'])
        and (regime_bias <= 0 or trend_bias <= 0)
        and macro_bias <= 0
        and session_bias <= 0
    )
    if long_ok and liquidity_bias in (0.0, 1.0):
        return 1
    if short_ok and liquidity_bias in (0.0, -1.0):
        return -1
    return 0

enriched['signal'] = enriched.apply(lambda r: compose_signal(r, float(r['signal_probability'])), axis=1)
row = enriched.iloc[-1]

print('latest_signal=', int(row['signal']))
print('latest_probability=', float(row['signal_probability']))
print('model_threshold=', cfg['model']['probability_threshold'])
print('entry_session=', bool(row.get('is_entry_session', False)))
print('trend_bias=', 1.0 if float(row['sma_fast']) >= float(row['sma_slow']) else -1.0)
print('regime_bias=', float(row.get('regime_bias', 0.0)))
print('macro_bias=', float(row.get('macro_bias', 0.0)))
print('session_bias=', float(row.get('session_bias', 0.0)))
print('liquidity_sweep=', float(row.get('liquidity_sweep', 0.0)))
print('close=', float(row['Close']))
print('ret_1=', float(row['ret_1']))
print('ret_5=', float(row['ret_5']))
print('sma_fast=', float(row['sma_fast']))
print('sma_slow=', float(row['sma_slow']))
print('rsi=', float(row['rsi']))
print('macd=', float(row['macd']))
print('atr=', float(row['atr']))
print('volatility=', float(row['volatility']))
print('macro_score=', float(row['macro_score']))
print('regime_score=', float(row['regime_score']))
print('session=', row.get('session', ''))
