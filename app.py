import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 頁面與使用者設定
# ==========================================
st.set_page_config(page_title="台美股投資戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# --- 使用者切換設定 (動態路徑) ---
with st.sidebar:
    st.header("👤 帳戶切換")
    user_list = ["主要帳戶", "投資帳戶B", "家人代操"] 
    current_user = st.selectbox("請選擇使用者：", user_list)
    
    # 動態決定檔案路徑
    DATA_FILE = f"portfolio_{current_user}.csv"
    st.info(f"📁 當前數據庫: `{DATA_FILE}`")

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
# 核心功能函數 (支援動態 DATA_FILE)
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

# ==========================================
# 統計分析與技術指標邏輯
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
        if df.empty: return None, "無法獲取歷史資料"
        df_recent = df.tail(26) 
        current_price = df['Close'].iloc[-1]
        high_6m, low_6m = df_recent['High'].max(), df_recent['Low'].min()
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_curr = calculate_rsi(df['Close'], 14).iloc[-1]
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        
        if rsi_curr > 70: advice, color = "過熱，建議分批獲利", "red"
        elif rsi_curr < 30: advice, color = "超賣，可考慮分批佈局", "green"
        else: advice, color = "趨勢持穩，觀望或波段操作", "gray"

        return {
            "current_price": current_price, "high_6m": high_6m, "low_6m": low_6m,
            "rsi": rsi_curr, "trend": trend, "advice": advice, 
            "advice_color": color, "history_df": df_recent, "ma_20": ma_20
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 介面顯示組件 (解決排序與管理按鈕問題)
# ==========================================
COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def update_sort(column_name):
    if st.session_state.sort_col == column_name:
        st.session_state.sort_asc = not st.session_state.sort_asc
    else:
        st.session_state.sort_col = column_name
        st.session_state.sort_asc = False

def display_headers(key_suffix):
    cols = st.columns(COLS_RATIO)
    headers = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本(原幣)"), ("現值", "現值(原幣)"), ("獲利", "獲利(原幣)"), ("報酬率%", "獲利率(%)")]
    for col, (label, col_name) in zip(cols[:-1], headers):
        arrow = "▲" if st.session_state.sort_asc and st.session_state.sort_col == col_name else "▼" if st.session_state.sort_col == col_name else ""
        if col.button(f"{label} {arrow}", key=f"h_{col_name}_{key_suffix}_{current_user}"):
            update_sort(col_name); st.rerun()
    cols[-1].write("管理")

def display_stock_rows(df, currency_type):
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        color = "red" if row["獲利(原幣)"] > 0 else "green"
        
        c1.write(f"**{row['股票代號']}**")
        c2.write(f"{row['股數']:.2f}")
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{row['最新股價']:.2f}")
        c5.write(fmt.format(row['總投入成本(原幣)']))
        c6.write(fmt.format(row['現值(原幣)']))
        c7.markdown(f":{color}[{fmt.format(row['獲利(原幣)'])}]")
        c8.markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        # 垃圾桶按鈕 Key 必須包含使用者名稱，防止切換時衝突
        if c9.button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            remove_stock(row['股票代號']); st.rerun()

# ==========================================
# 主程式區
# ==========================================
df_record = load_data()

# 頂部控制
col_refresh, col_time = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 刷新"):
        st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%H:%M:%S")
        st.rerun()
with col_time:
    st.caption(f"最後更新: {st.session_state.last_updated} | 使用者: {current_user}")

tab1, tab2, tab3 = st.tabs(["📊 庫存", "🧠 技術分析", "⚖️ 組合分析"])

with tab1:
    with st.sidebar:
        st.divider()
        st.subheader(f"新增持股 ({current_user})")
        with st.form("add_form"):
            s_in = st.text_input("股票代號", "2330.TW").upper()
            q_in = st.number_input("股數", min_value=0.0, value=100.0)
            c_in = st.number_input("持有成本", min_value=0.0, value=600.0)
            if st.form_submit_button("確認新增"):
                df = load_data()
                save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])], ignore_index=True))
                st.rerun()
        if st.button("🚨 清空本帳戶資料"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()

    if df_record.empty:
        st.info(f"帳戶 [{current_user}] 尚無資料。")
    else:
        # --- 核心計算 (解決 KeyError) ---
        usd_rate = get_exchange_rate()
        df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
        
        # 聚合計算：確保重新命名為 '平均持有單價'
        portfolio = df_record.groupby(["股票代號", "幣別"]).agg({
            "股數": "sum",
            "持有成本單價": "mean"
        }).reset_index().rename(columns={"持有成本單價": "平均持有單價"})
        
        # 補齊顯示所需欄位
        portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["平均持有單價"]
        prices = get_current_prices(portfolio["股票代號"].tolist())
        portfolio["最新股價"] = portfolio["股票代號"].map(prices)
        portfolio = portfolio.dropna(subset=["最新股價"]) # 過濾價格獲取失敗的
        
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        
        # 換算台幣
        portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
        portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
        
        # 總結看板
        t_val = portfolio["現值(TWD)"].sum()
        t_profit = (portfolio["獲利(原幣)"] * portfolio["匯率因子"]).sum()
        st.metric(f"💰 {current_user} 總資產", f"${t_val:,.0f} TWD", f"總獲利: ${t_profit:,.0f}")

        # 列表顯示
        for label, cur in [("🇹🇼 台股列表", "TWD"), ("🇺🇸 美股列表", "USD")]:
            sub = portfolio[portfolio["幣別"] == cur]
            if not sub.empty:
                st.subheader(label)
                display_headers(cur.lower())
                display_stock_rows(sub, cur)

# --- Tab 2 & 3 保持類似邏輯 ---
with tab2:
    if not df_record.empty:
        target = st.selectbox("分析目標", portfolio["股票代號"].tolist(), key="tech_select")
        res, err = analyze_stock_technical(target)
        if not err:
            st.metric("目前價格", f"{res['current_price']:.2f}", res['trend'])
            st.success(f"建議：{res['advice']}")
            st.line_chart(res['history_df']['Close'])

with tab3:
    st.write("這部分會根據目前持股進行權重分析...")
    if not df_record.empty:
        fig = px.pie(portfolio, values="現值(TWD)", names="股票代號", title=f"{current_user} 資產分佈")
        st.plotly_chart(fig, use_container_width=True)
