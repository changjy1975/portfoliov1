import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import pytz
import numpy as np

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 狀態初始化
# ==========================================
if "sort_col" not in st.session_state:
    st.session_state.sort_col = "獲利(原幣)"
if "sort_asc" not in st.session_state:
    st.session_state.sort_asc = False
if "last_updated" not in st.session_state:
    st.session_state.last_updated = "尚未更新"

# ==========================================
# 核心功能函數
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def remove_stock(symbol):
    df = load_data()
    df = df[df["股票代號"] != symbol]
    save_data(df)

def get_exchange_rate():
    try:
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.fast_info.last_price
        if rate is None or pd.isna(rate):
             rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 32.5

def get_current_prices(symbols):
    if not symbols: return {}
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            price = ticker.fast_info.last_price
            if price is None or pd.isna(price):
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty: price = hist["Close"].iloc[-1]
            if price is None or pd.isna(price):
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPreviousClose')
            prices[symbol] = price
        except:
            prices[symbol] = None
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

def get_historical_pl_trend(portfolio_df, period="1y"):
    """計算過去一段時間的每日總損益趨勢"""
    symbols = portfolio_df["股票代號"].unique().tolist()
    if not symbols: return None
    try:
        # 下載歷史股價與匯率
        data = yf.download(symbols, period=period, interval="1d")['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=symbols[0])
        usd_twd_hist = yf.download("USDTWD=X", period=period, interval="1d")['Close']
        
        combined_df = data.ffill().dropna()
        usd_twd_hist = usd_twd_hist.reindex(combined_df.index, method='ffill')
        
        daily_total_value_twd = pd.Series(0.0, index=combined_df.index)
        total_invested_twd = 0.0

        for _, row in portfolio_df.iterrows():
            sym, qty, cost_unit = row["股票代號"], row["股數"], row["平均持有單價"]
            if sym in combined_df.columns:
                if identify_currency(sym) == "USD":
                    daily_val = combined_df[sym] * qty * usd_twd_hist
                    invested = cost_unit * qty * usd_twd_hist.iloc[-1]
                else:
                    daily_val = combined_df[sym] * qty
                    invested = cost_unit * qty
                daily_total_value_twd += daily_val
                total_invested_twd += invested

        return pd.DataFrame({"累計損益": daily_total_value_twd - total_invested_twd}, index=combined_df.index)
    except: return None

# ==========================================
# 分析邏輯 (RSI, 技術分析, 投資組合分析)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1wk")
        if df.empty: return None, "無法獲取資料"
        df_recent = df.tail(26) 
        current_price = df['Close'].iloc[-1]
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_curr = calculate_rsi(df['Close'], 14).iloc[-1]
        
        advice, color = ("過熱，分批獲利", "red") if rsi_curr > 70 else ("超賣，分批佈局", "green") if rsi_curr < 30 else ("趨勢向上，持股續抱", "orange") if current_price > ma_20 else ("趨勢偏弱，觀望", "gray")
        return {"current_price": current_price, "high_6m": df_recent['High'].max(), "low_6m": df_recent['Low'].min(), "rsi": rsi_curr, "trend": "多頭 🐂" if current_price > ma_20 else "整理 🐻", "entry_target": max(df_recent['Low'].min() * 1.02, ma_20), "exit_target": df_recent['High'].max() * 0.98, "advice": advice, "advice_color": color, "history_df": df_recent}, None
    except Exception as e: return None, str(e)

def perform_portfolio_analysis(portfolio_df):
    symbols = portfolio_df["股票代號"].unique().tolist()
    if not symbols: return None, "無資料"
    try:
        hist_data = yf.download(" ".join(symbols), period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(hist_data, pd.Series): hist_data = hist_data.to_frame(name=symbols[0])
        returns = hist_data.pct_change().dropna()
        perf_list = []
        for sym in hist_data.columns:
            s = hist_data[sym].dropna()
            re = s.pct_change().dropna()
            cagr = ((s.iloc[-1]/s.iloc[0])**(1/(len(s)/252)) - 1) * 100
            perf_list.append({"股票代號": sym, "CAGR (%)": cagr, "年化波動率 (%)": re.std()*np.sqrt(252)*100, "Sharpe Ratio": (re.mean()*252)/(re.std()*np.sqrt(252))})
        
        suggestions = [f"⚠️ 集中度風險：{row['股票代號']}" for _, row in portfolio_df.iterrows() if row["現值(TWD)"]/portfolio_df["現值(TWD)"].sum() > 0.3]
        return {"corr_matrix": returns.corr(), "suggestions": suggestions or ["✅ 配置健康"], "perf_df": pd.DataFrame(perf_list)}, None
    except Exception as e: return None, str(e)

# ==========================================
# 介面組件
# ==========================================
COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def display_headers(key):
    cols = st.columns(COLS_RATIO)
    labels = [("代號","股票代號"), ("股數","股數"), ("均價","平均持有單價"), ("現價","最新股價"), ("總成本","總投入成本(原幣)"), ("現值","現值(原幣)"), ("獲利","獲利(原幣)"), ("報酬%","獲利率(%)")]
    for col, (l, n) in zip(cols[:-1], labels):
        if col.button(f"{l} {'▲' if st.session_state.sort_asc and st.session_state.sort_col==n else '▼'}", key=f"h_{n}_{key}"):
            st.session_state.sort_asc = not st.session_state.sort_asc if st.session_state.sort_col==n else False
            st.session_state.sort_col = n
            st.rerun()
    cols[-1].write("**管理**")

def display_stock_rows(df, cur):
    for _, row in df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc).iterrows():
        c = st.columns(COLS_RATIO)
        color = "red" if row["獲利(原幣)"] > 0 else "green"
        fmt = "{:,.0f}" if cur == "TWD" else "{:,.2f}"
        c[0].write(f"**{row['股票代號']}**"); c[1].write(f"{row['股數']:.2f}"); c[2].write(f"{row['平均持有單價']:.2f}")
        c[3].write(f"{row['最新股價']:.2f}"); c[4].write(fmt.format(row['總投入成本(原幣)'])); c[5].write(fmt.format(row['現值(原幣)']))
        c[6].markdown(f":{color}[{fmt.format(row['獲利(原幣)'])}]"); c[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if c[8].button("🗑️", key=f"del_{row['股票代號']}"): remove_stock(row['股票代號']); st.rerun()

# ==========================================
# 主程式執行
# ==========================================
col_refresh, col_time = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 刷新全部數據"):
        st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
with col_time:
    st.markdown(f"<div style='color: gray;'>最後更新: {st.session_state.last_updated}</div>", unsafe_allow_html=True)

df_record = load_data()
tab1, tab2, tab3 = st.tabs(["📊 庫存與資產配置", "🧠 AI 技術分析", "⚖️ 組合再平衡"])

if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    portfolio = df_record.groupby(["股票代號", "幣別"]).agg({"股數": "sum", "持有成本單價": "mean"}).reset_index()
    portfolio.rename(columns={"持有成本單價": "平均持有單價"}, inplace=True)
    
    current_prices = get_current_prices(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
    portfolio = portfolio.dropna(subset=["最新股價"])
    
    portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
    
    rate_factor = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
    portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * rate_factor
    portfolio["總投入成本(TWD)"] = portfolio["總投入成本(原幣)"] * rate_factor

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 總資產 (TWD)", f"${portfolio['現值(TWD)'].sum():,.0f}")
        c2.metric("💳 總成本 (TWD)", f"${portfolio['總投入成本(TWD)'].sum():,.0f}")
        total_p = portfolio['現值(TWD)'].sum() - portfolio['總投入成本(TWD)'].sum()
        c3.metric("📈 總獲利", f"${total_p:,.0f}", f"{(total_p/portfolio['總投入成本(TWD)'].sum()*100):.2f}%")
        
        st.markdown("---")
        st.subheader("📈 累計損益變動趨勢 (TWD)")
        period_choice = st.select_slider("區間", options=["1mo", "3mo", "6mo", "1y"], value="1y")
        trend_df = get_historical_pl_trend(portfolio, period_choice)
        if trend_df is not None:
            fig = px.line(trend_df, y="累計損益", title="投資組合損益走勢")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.subheader("📦 詳細庫存")
        for b in ["TWD", "USD"]:
            st.caption(f"{'🇹🇼 台股' if b=='TWD' else '🇺🇸 美股'}")
            sub = portfolio[portfolio["幣別"] == b]
            if not sub.empty:
                display_headers(b.lower()); display_stock_rows(sub, b)
            else: st.write("無持倉")

    with tab2:
        sel = st.selectbox("分析對象", portfolio["股票代號"].tolist())
        res, err = analyze_stock_technical(sel)
        if res:
            col_a, col_b = st.columns(2)
            col_a.metric("RSI 指標", f"{res['rsi']:.1f}")
            col_b.success(f"建議：{res['advice']}")
            st.line_chart(res['history_df']['Close'])

    with tab3:
        if st.button("啟動深度分析"):
            res, err = perform_portfolio_analysis(portfolio)
            if res:
                st.plotly_chart(px.imshow(res['corr_matrix'], text_auto=".2f"), use_container_width=True)
                st.dataframe(res['perf_df'])

with st.sidebar:
    st.header("📝 新增投資")
    with st.form("add"):
        s = st.text_input("代號", "2330.TW").upper()
        q = st.number_input("股數", value=1000.0)
        c = st.number_input("單價", value=500.0)
        if st.form_submit_button("新增"):
            df = pd.concat([load_data(), pd.DataFrame([{"股票代號":s,"股數":q,"持有成本單價":c}])], ignore_index=True)
            save_data(df); st.rerun()
    if st.button("🚨 清空"): 
        if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()
