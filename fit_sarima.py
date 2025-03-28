from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(orders, df):
    p, d, q, P, D, Q, m = orders
    
    sarima = SARIMAX(df, order=(p, d, q), seasonal_order=(P, D, Q, m), freq='h')
    res = sarima.fit()
    aic = res.aic
    
    print(f'({p}, {d}, {q}, {P}, {D}, {Q}, {m}): {aic}')
    
    return (p, d, q, P, D, Q, m), aic