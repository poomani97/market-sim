import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title='Market Simulator', layout='wide')

INDICES = {
    'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK', 'Sensex': '^BSESN',
    'Nifty Next 50': '^NSMIDCP', 'Nifty IT': '^CNXIT', 'Nifty Auto': '^CNXAUTO',
    'Nifty Pharma': '^CNXPHARMA', 'Nifty Metal': '^CNXMETAL',
    'Nifty FMCG': '^CNXFMCG', 'Nifty Energy': '^CNXENERGY', 'India VIX': '^INDIAVIX',
}
GLOBAL = {
    'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'Dow Jones': '^DJI',
    'Nikkei 225': '^N225', 'Hang Seng': '^HSI',
}
COMMODITIES = {
    'Gold (USD)': 'GC=F', 'Silver (USD)': 'SI=F', 'Crude Oil WTI': 'CL=F',
    'Brent Crude': 'BZ=F', 'Natural Gas': 'NG=F', 'Copper': 'HG=F',
}
ETFS = {
    'Gold BeES ETF': 'GOLDBEES.NS', 'Silver BeES ETF': 'SILVERBEES.NS',
    'Nifty BeES ETF': 'NIFTYBEES.NS', 'Bank BeES ETF': 'BANKBEES.NS',
}
FOREX = {
    'USD/INR': 'USDINR=X', 'EUR/INR': 'EURINR=X', 'GBP/INR': 'GBPINR=X',
    'JPY/INR': 'JPYINR=X', 'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X', 'Dollar Index': 'DX-Y.NYB',
}
GROUPS = {
    'Indices': INDICES, 'Global Indices': GLOBAL, 'Commodities': COMMODITIES,
    'Gold/Silver ETFs': ETFS, 'Forex': FOREX,
}
FALLBACK = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'SBIN',
            'AXISBANK', 'ITC', 'LT', 'KOTAKBANK', 'M&M']
SKIP = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50', 'NIFTYFPI'}


@st.cache_data(ttl=6 * 3600)
def get_fno():
    url = 'https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv'
    h = {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,*/*'}
    try:
        s = requests.Session()
        s.get('https://www.nseindia.com', headers=h, timeout=10)
        r = s.get(url, headers=h, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        names = df['SYMBOL'].astype(str).str.strip().tolist()
        names = [n for n in names if n and n not in SKIP and n != 'Symbol']
    except Exception:
        names = FALLBACK
    return {n: n + '.NS' for n in names}


def universe():
    u = {}
    for cat, d in GROUPS.items():
        for name, sym in d.items():
            u[name] = (cat, sym)
    for name, sym in get_fno().items():
        u[name] = ('F&O Stocks', sym)
    return u


@st.cache_data(ttl=3600)
def get_prices(tickers, period='2y'):
    df = yf.download(list(tickers), period=period, auto_adjust=True,
                     progress=False, threads=True)
    close = df['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def simulate(close, p, keep=False):
    close = close.dropna().tail(p['lookback'] + 1)
    r = np.log(close / close.shift(1)).dropna()
    if len(r) < 60:
        return None
    if p['ewma']:
        sigma = float(np.sqrt((r ** 2).ewm(alpha=0.06).mean().iloc[-1]))
    else:
        sigma = float(r.std())
    mu = float(r.mean()) if p['drift'] else 0.0
    z = np.random.normal(mu, sigma, (p['sims'], p['days']))
    paths = np.exp(np.cumsum(z, axis=1))
    final = (paths[:, -1] - 1) * 100
    move = sigma * np.sqrt(p['days']) * 100
    band = p['frac'] * move
    up = float((final > band).mean() * 100)
    down = float((final < -band).mean() * 100)
    side = 100 - up - down
    if abs(up - down) < 3:
        lean = 'NEUTRAL'
    else:
        lean = 'UP' if up > down else 'DOWN'
    out = {'Last': float(close.iloc[-1]), 'Up %': up, 'Sideways %': side,
           'Down %': down, 'Move 1sd %': move, 'Lean': lean}
    if keep:
        out['paths'] = paths
        out['final'] = final
    return out


st.sidebar.title('Settings')
P = {
    'sims': st.sidebar.selectbox('Simulations', [10000, 5000, 2000, 1000], 0),
    'days': st.sidebar.slider('Days ahead', 1, 10, 1),
    'frac': st.sidebar.slider('Sideways band (fraction of expected move)', 0.1, 1.0, 0.25, 0.05),
    'lookback': st.sidebar.slider('History used (days)', 60, 500, 250),
    'drift': st.sidebar.checkbox('Include past average drift', True),
    'ewma': st.sidebar.checkbox('Weight recent days more (EWMA volatility)', True),
}
st.sidebar.caption('Drift on means the lean follows the past trend. Drift off makes UP and DOWN equal.')

U = universe()
st.title('Market Simulator')
t1, t2, t3 = st.tabs(['Analyze one', 'Scanner', 'Backtest'])

# ---------- Tab 1: analyze one symbol ----------
with t1:
    c1, c2 = st.columns([1, 2])
    cat = c1.selectbox('Category', ['All'] + list(GROUPS) + ['F&O Stocks'])
    q = c1.text_input('Search (nifty, gold, usd, reliance ...)').lower()
    names = [n for n, (c, s) in U.items()
             if (cat == 'All' or c == cat) and (q in n.lower() or q in s.lower())]
    if not names:
        st.warning('No match found')
    else:
        name = c2.selectbox(str(len(names)) + ' matches', names)
        if st.button('Run simulation', type='primary'):
            sym = U[name][1]
            close = get_prices((sym,)).iloc[:, 0]
            res = simulate(close, P, True)
            if res is None:
                st.error('Not enough price data for this symbol')
            else:
                last = res['Last']
                st.subheader(name + ' (' + sym + ')  last price ' + format(last, ',.2f'))
                a, b, c, d = st.columns(4)
                a.metric('Up chance', format(res['Up %'], '.1f') + '%')
                b.metric('Sideways chance', format(res['Sideways %'], '.1f') + '%')
                c.metric('Down chance', format(res['Down %'], '.1f') + '%')
                d.metric('Expected move +/-1sd', format(res['Move 1sd %'], '.2f') + '%')
                st.write('Lean: ' + res['Lean'])
                p5, p50, p95 = np.percentile(res['final'], [5, 50, 95])
                lo = last * (1 + p5 / 100)
                md = last * (1 + p50 / 100)
                hi = last * (1 + p95 / 100)
                st.write('After ' + str(P['days']) + ' day(s): 90% of simulations end between '
                         + format(lo, ',.2f') + ' and ' + format(hi, ',.2f')
                         + ', median ' + format(md, ',.2f') + '.')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
                pp = np.hstack([np.ones((150, 1)), res['paths'][:150]])
                ax1.plot(last * pp.T, alpha=0.25)
                ax1.axhline(last, color='black')
                ax1.set_title('150 simulated paths')
                ax2.hist(res['final'], bins=60)
                ax2.axvline(0, color='black')
                ax2.set_title('Final change % across all simulations')
                st.pyplot(fig)

# ---------- Tab 2: scanner ----------
with t2:
    sc = st.selectbox('Scan which group?', list(GROUPS) + ['F&O Stocks', 'Everything'], index=5)
    if st.button('Run scan', type='primary'):
        items = [(n, s) for n, (c, s) in U.items() if sc == 'Everything' or c == sc]
        tickers = tuple(dict.fromkeys(s for _, s in items))
        with st.spinner('Downloading prices for ' + str(len(tickers)) + ' symbols ...'):
            px = get_prices(tickers)
        rows = []
        bar = st.progress(0)
        for i, (n, s) in enumerate(items):
            if s in px.columns:
                out = simulate(px[s], P)
                if out:
                    rows.append({'Symbol': n, **out})
            bar.progress((i + 1) / len(items))
        st.session_state['scan'] = pd.DataFrame(rows)
    df = st.session_state.get('scan')
    if df is not None and len(df) > 0:
        df = df.round(1)
        st.write('Scanned ' + str(len(df)) + ' symbols')
        cols = ['Symbol', 'Up %', 'Sideways %', 'Down %', 'Lean']
        x, y, z = st.columns(3)
        x.write('Most likely UP')
        x.dataframe(df.sort_values('Up %', ascending=False).head(15)[cols], hide_index=True)
        y.write('Most likely DOWN')
        y.dataframe(df.sort_values('Down %', ascending=False).head(15)[cols], hide_index=True)
        z.write('Most likely SIDEWAYS')
        z.dataframe(df.sort_values('Sideways %', ascending=False).head(15)[cols], hide_index=True)
        f = st.text_input('Filter the full table by name').lower()
        full = df[df['Symbol'].str.lower().str.contains(f)] if f else df
        st.dataframe(full.sort_values('Symbol'), hide_index=True, use_container_width=True)

# ---------- Tab 3: backtest ----------
with t3:
    st.write('Replays up to 6 years, one day ahead: was the model lean right the next day?')
    keys = list(U)
    bn = st.selectbox('Symbol to backtest', keys, index=keys.index('Nifty 50'))
    if st.button('Run backtest'):
        c = get_prices((U[bn][1],), '6y').iloc[:, 0].dropna()
        r = np.log(c / c.shift(1))
        d = pd.DataFrame({'mu': r.rolling(P['lookback']).mean(),
                          'sd': r.rolling(P['lookback']).std(),
                          'nxt': (c.shift(-1) / c - 1) * 100}).dropna()
        band = P['frac'] * d['sd']
        d['p_up'] = 1 - norm.cdf((band - d['mu']) / d['sd'])
        d['p_dn'] = norm.cdf((-band - d['mu']) / d['sd'])
        d['lean'] = np.where(d['p_up'] > d['p_dn'], 'UP', 'DOWN')
        d['strength'] = (d['p_up'] - d['p_dn']).abs()
        base = float((d['nxt'] > 0).mean() * 100)
        st.write('Days tested: ' + str(len(d)) + '. Base rate of up days: ' + format(base, '.1f') + '%')
        rows = []
        for label, m in [('Lean UP', d['lean'] == 'UP'), ('Lean DOWN', d['lean'] == 'DOWN'),
                         ('Strongest 20% UP', (d['lean'] == 'UP') & (d['strength'] >= d['strength'].quantile(0.8))),
                         ('Strongest 20% DOWN', (d['lean'] == 'DOWN') & (d['strength'] >= d['strength'].quantile(0.8)))]:
            x = d[m]
            if len(x) < 20:
                continue
            hit = (x['nxt'] > 0).mean() * 100 if 'UP' in label else (x['nxt'] < 0).mean() * 100
            rows.append({'Signal': label, 'Days': len(x), 'Hit rate %': round(hit, 1),
                         'Avg next-day move %': round(x['nxt'].mean(), 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption('A hit rate close to the base rate means no edge. Above about 55% on 100+ days is worth a closer look.')

st.caption('Statistical estimates from past volatility. Not a prediction and not investment advice.')
