STRATEGIES = {
        'aggressive': {
            'PRICE_THRESHOLD_PERCENT': 5,
            'VOLUME_SPIKE_MULTIPLIER': 1.2,
            'SCORE_WEIGHTS': {
                'price': 1.0,
                'volume': 2.0,
                'macd': 10.0,
                'rsi': 0.1
            }
        },
        'conservative': {
            'PRICE_THRESHOLD_PERCENT': 10,
            'VOLUME_SPIKE_MULTIPLIER': 2.0,
            'SCORE_WEIGHTS': {
                'price': 1.5,
                'volume': 2.5,
                'macd': 5.0,
                'rsi': 0.2,
            }
        }
}
