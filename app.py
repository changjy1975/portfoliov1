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
# 頂部控制區
# ==========================================
col_refresh, col_time = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 刷新全部數據"):
        st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
with col_time:
    st.markdown(f"<div style='padding-top: 10px; color: gray;'>最後更新時間: {st.session_state.last_updated} (台股來源: Yahoo Fast Info)</div>", unsafe_allow_html=True)

st.divider()

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
            price = None
            try:
                price = ticker.fast_info.last_price
            except:
                price = None

            if price is None or pd.isna(price):
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
            
            if price is None or pd.isna(price):
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or info.get('previousClose')
            
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
        # 1. 抓取歷史股價
        data = yf.download(symbols, period=period, interval="1d")['Close']
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=symbols[0])
        
        # 2. 抓取歷史匯率
        usd_twd_hist = yf.download("USDTWD=X", period=period, interval="1d")['Close']
        
        # 確保資料對齊
        combined_df = data.ffill().dropna()
        usd_twd_hist = usd_twd_hist.reindex(combined_df.index, method='ffill')
        
        daily_total_value_twd = pd.Series(0, index=combined_df.index)
        total_invested_twd = 0

        for _, row in portfolio_df.iterrows():
            sym = row["股票代號"]
            qty = row["股數"]
            cost_unit = row["平均持有單價"]
            currency = identify_currency(sym)
            
            if sym in combined_df.columns:
                if currency == "USD":
                    # 美股：每日市值換算為 TWD
                    daily_val = combined_df[sym] * qty * usd_twd_hist
                    # 成本以目前的匯率估計 (簡化版)
                    invested = cost_unit * qty * usd_twd_hist.iloc[-1]
                else:
                    daily_val = combined_df[sym] * qty
                    invested = cost_unit * qty
                
                daily_total_value_twd += daily_val
                total_invested_twd += invested

        trend_df = pd.DataFrame({
            "總市值": daily_total_value_twd,
            "累計損益": daily_total_value_twd - total_invested_twd
        })
        return trend_df
    except Exception as e:
        return None

# ==========================================
# 技術分析邏輯 (Tab 2) ... (省略重複的部分以保持簡潔，建議保留原有的 analyze_stock_technical 等函數)
# ==========================================
# [此處保留原本 app (1).py 的 calculate_rsi, analyze_stock_technical, perform_portfolio_analysis 函數]
# ... 

# ==========================================
# 介面顯示組件 ... (省略重複的 display_headers, display_stock_rows 等)
# ==========================================
# [此處保留原本 app (1).py 的介面函數]
# ...

# (以下為整合了新圖表的「主程式邏輯」部分)

# ==========================================
# 主程式邏輯
# ==========================================
# (由於內容較長，我重點展示整合趨勢圖的 Tab 1 部分)

# [中間程式碼同原檔，直到進入 Tab 1]

with tab1:
    # ... (側邊欄與數據載入邏輯同原檔) ...

    if not df_record.empty:
        # (總資產看板 metric 顯示同原檔)
        # ...
        
        st.markdown("---")
        
        # --- 新增：每日損益趨勢圖 ---
        st.subheader("📈 累計損益變動趨勢 (TWD)")
        period_choice = st.select_slider("選擇顯示區間", options=["1mo", "3mo", "6mo", "1y"], value="1y", label_visibility="collapsed")
        
        with st.spinner('正在分析歷史數據...'):
            trend_data = get_historical_pl_trend(portfolio, period=period_choice)
            
        if trend_data is not None:
            fig_trend = px.line(
                trend_data, 
                y="累計損益", 
                title=f"過去 {period_choice} 投資組合損益走勢",
                labels={"Date": "日期", "累計損益": "金額 (TWD)"},
                color_discrete_sequence=["#2ECC71"]
            )
            fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_trend.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("暫無足夠歷史數據生成趨勢圖。")

        st.markdown("---")

        # 圖表區 (原本的圓餅圖)
        st.subheader("📊 資產分佈分析")
        # ... (以下接原檔的 col_pie1, col_pie2 以及詳細庫存列表)
