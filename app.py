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

# --- 使用者切換設定 ---
with st.sidebar:
    st.header("👤 帳戶切換")
    user_list = ["主要帳戶", "投資帳戶B", "家人代操"] # 你可以在這裡增加更多使用者
    current_user = st.selectbox("請選擇使用者：", user_list)
    
    # 根據使用者名稱生成專屬檔案路徑
    DATA_FILE = f"portfolio_{current_user}.csv"
    st.info(f"當前檔案: `{DATA_FILE}`")

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
# 核心功能函數 (已修改以支援動態路徑)
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

# --- 以下獲取數據函數保持不變 ---
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

# ==========================================
# 技術分析與投資組合分析邏輯 (保持不變)
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
        high_6m = df_recent['High'].max()
        low_6m = df_recent['Low'].min()
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_series = calculate_rsi(df['Close'], 14)
        rsi_curr = rsi_series.iloc[-1]
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        entry_price = max(low_6m * 1.02, ma_20)
        exit_price = high_6m * 0.98
        
        if rsi_curr > 70: advice, color = "過熱，建議分批獲利", "red"
        elif rsi_curr < 30: advice, color = "超賣，可考慮分批佈局", "green"
        elif current_price > ma_20: advice, color = "趨勢向上，持股續抱", "orange"
        else: advice, color = "趨勢偏弱，觀望或區間操作", "gray"

        return {
            "current_price": current_price, "high_6m": high_6m, "low_6m": low_6m,
            "ma_20": ma_20, "rsi": rsi_curr, "trend": trend,
            "entry_target": entry_price, "exit_target": exit_price,
            "advice": advice, "advice_color": color, "history_df": df_recent
        }, None
    except Exception as e:
        return None, str(e)

def perform_portfolio_analysis(portfolio_df):
    symbols = portfolio_df["股票代號"].unique().tolist()
    if not symbols: return None, "無持股資料"
    try:
        tickers_str = " ".join(symbols)
        hist_data = yf.download(tickers_str, period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(hist_data, pd.Series):
            hist_data = hist_data.to_frame(name=symbols[0])
        hist_data = hist_data.dropna(how='all')
        returns = hist_data.pct_change().dropna()
        corr_matrix = returns.corr()
        performance_list = []
        for symbol in hist_data.columns:
            try:
                series = hist_data[symbol].dropna()
                if len(series) < 20: continue 
                daily_rets = series.pct_change().dropna()
                days_diff = (series.index[-1] - series.index[0]).days
                years = days_diff / 365.25
                total_return = (series.iloc[-1] / series.iloc[0]) - 1
                cagr = ((series.iloc[-1] / series.iloc[0]) ** (1/years)) - 1 if years > 0 else 0
                stdev = daily_rets.std() * np.sqrt(252)
                mean_ret = daily_rets.mean() * 252
                sharpe = mean_ret / stdev if stdev != 0 else 0
                negative_rets = daily_rets[daily_rets < 0]
                downside_std = negative_rets.std() * np.sqrt(252)
                sortino = mean_ret / downside_std if downside_std != 0 else 0
                annual_prices = series.resample('YE').last()
                if len(annual_prices) < 2:
                     best_year = total_return
                     worst_year = total_return
                else:
                    annual_rets = series.resample('YE').apply(lambda x: (x.iloc[-1]/x.iloc[0])-1)
                    best_year = annual_rets.max()
                    worst_year = annual_rets.min()
                performance_list.append({
                    "股票代號": symbol, "CAGR (%)": cagr * 100, "年化波動率 (%)": stdev * 100,
                    "Best Year (%)": best_year * 100, "Worst Year (%)": worst_year * 100,
                    "Sharpe Ratio": sharpe, "Sortino Ratio": sortino
                })
            except: pass 
        perf_df = pd.DataFrame(performance_list)
        suggestions = []
        total_val = portfolio_df["現值(TWD)"].sum()
        for idx, row in portfolio_df.iterrows():
            weight = row["現值(TWD)"] / total_val
            if weight > 0.3: suggestions.append(f"⚠️ **集中度風險**：{row['股票代號']} 佔比達 {weight*100:.1f}%，建議適度減碼。")
        cols = corr_matrix.columns
        high_corr_pairs = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if corr_matrix.iloc[i, j] > 0.8: high_corr_pairs.append(f"{cols[i]} & {cols[j]}")
        if high_corr_pairs: suggestions.append(f"🔗 **連動風險**：以下股票相關性高：" + ", ".join(high_corr_pairs))
        if not suggestions: suggestions.append("✅ 配置健康。")
        return {"corr_matrix": corr_matrix, "suggestions": suggestions, "perf_df": perf_df}, None
    except Exception as e: return None, str(e)

# ==========================================
# 介面顯示組件 (保持不變)
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
    headers_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本(原幣)"), ("現值", "現值(原幣)"), ("獲利", "獲利(原幣)"), ("報酬率%", "獲利率(%)")]
    for col, (label, col_name) in zip(cols[:-1], headers_map):
        if col.button(f"{label} {'▲' if st.session_state.sort_asc and st.session_state.sort_col == col_name else '▼' if st.session_state.sort_col == col_name else ''}", key=f"btn_{col_name}_{key_suffix}"):
            update_sort(col_name); st.rerun()
    cols[-1].markdown("**管理**")

def display_stock_rows(df, currency_type):
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for index, row in df_sorted.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        symbol, price, prof, roi = row["股票代號"], row["最新股價"], row["獲利(原幣)"], row["獲利率(%)"]
        color = "red" if prof > 0 else "green"
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        c1.write(f"**{symbol}**"); c2.write(f"{row['股數']:.3f}"); c3.write(f"{row['平均持有單價']:.2f}"); c4.write(f"{price:.2f}"); c5.write(fmt.format(row['總投入成本(原幣)'])); c6.write(fmt.format(row['現值(原幣)'])); c7.markdown(f":{color}[{fmt.format(prof)}]"); c8.markdown(f":{color}[{roi:.2f}%]")
        if c9.button("🗑️", key=f"del_{symbol}_{current_user}"): remove_stock(symbol); st.rerun()

def display_subtotal_row(df, currency_type):
    tc, tv, tp = df["總投入成本(原幣)"].sum(), df["現值(原幣)"].sum(), df["獲利(原幣)"].sum()
    roi = (tp / tc * 100) if tc > 0 else 0
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
    fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
    color = "red" if tp > 0 else "green"
    c1.markdown("**🔹 小計**"); c5.markdown(f"**{fmt.format(tc)}**"); c6.markdown(f"**{fmt.format(tv)}**"); c7.markdown(f":{color}[**{fmt.format(tp)}**]"); c8.markdown(f":{color}[**{roi:.2f}%**]")
    return tv, tp

# ==========================================
# 主程式邏輯
# ==========================================
col_refresh, col_time = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 刷新數據"):
        st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
with col_time:
    st.markdown(f"<div style='padding-top: 10px; color: gray;'>最後更新時間: {st.session_state.last_updated} | 使用者: {current_user}</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 庫存與資產配置", "🧠 AI 技術分析", "⚖️ 投資組合分析"])

df_record = load_data()

with tab1:
    with st.sidebar:
        st.markdown("---")
        st.header(f"📝 新增投資 ({current_user})")
        with st.form("add_stock_form"):
            symbol_input = st.text_input("股票代號", value="2330.TW").upper().strip()
            qty_input = st.number_input("股數", min_value=0.0, value=1000.0, step=0.001, format="%.3f")
            cost_input = st.number_input("單價 (原幣)", min_value=0.0, value=500.0)
            if st.form_submit_button("新增"):
                df = load_data()
                new_data = pd.DataFrame({"股票代號": [symbol_input], "股數": [qty_input], "持有成本單價": [cost_input]})
                save_data(pd.concat([df, new_data], ignore_index=True))
                st.success(f"已新增至 {current_user}"); st.rerun()
        if st.button("🚨 清空當前使用者"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()

    if df_record.empty:
        st.info(f"帳戶 [{current_user}] 目前無投資紀錄。")
    else:
        usd_rate = get_exchange_rate()
        df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
        portfolio = df_record.groupby(["股票代號", "幣別"]).agg({"股數": "sum", "持有成本單價": "mean"}).reset_index()
        portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["持有成本單價"]
        portfolio["最新股價"] = portfolio["股票代號"].map(get_current_prices(portfolio["股票代號"].tolist()))
        portfolio = portfolio.dropna(subset=["最新股價"])
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
        portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
        
        # 總結看板
        total_val = portfolio["現值(TWD)"].sum()
        total_profit_twd = (portfolio["獲利(原幣)"] * portfolio["匯率因子"]).sum()
        st.metric(f"💰 {current_user} 總資產 (TWD)", f"${total_val:,.0f}", f"總獲利估計: ${total_profit_twd:,.0f}")
        
        # 圖表
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            st.plotly_chart(px.pie(portfolio, values="現值(TWD)", names="幣別", title="資產幣別分佈"), use_container_width=True)
        with col_pie2:
            st.plotly_chart(px.pie(portfolio, values="現值(TWD)", names="股票代號", title="個股佔比"), use_container_width=True)

        # 詳細列表
        st.subheader("📦 詳細庫存列表")
        for lang, cur in [("台股", "TWD"), ("美股", "USD")]:
            sub_df = portfolio[portfolio["幣別"] == cur]
            if not sub_df.empty:
                st.caption(f"🔹 {lang}")
                display_headers(cur.lower())
                display_stock_rows(sub_df, cur)
                display_subtotal_row(sub_df, cur)

# --- Tab 2 & 3 (主要邏輯與之前相同，僅確保使用對應的 portfolio 資料) ---
with tab2:
    if not df_record.empty:
        selected_stock = st.selectbox("分析股票：", portfolio["股票代號"].tolist())
        if selected_stock:
            res, err = analyze_stock_technical(selected_stock)
            if not err:
                st.success(f"建議：{res['advice']}")
                st.line_chart(res['history_df']['Close'])

with tab3:
    if not df_record.empty:
        if st.button("🚀 啟動深度分析", type="primary"):
            res, err = perform_portfolio_analysis(portfolio)
            if not err:
                st.plotly_chart(px.imshow(res['corr_matrix'], text_auto=".2f"), use_container_width=True)
                for s in res['suggestions']: st.info(s)
