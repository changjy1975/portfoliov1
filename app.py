import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
import shutil
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 設定與路徑初始化
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數
# ==========================================

def manage_backups(user, max_backups=10):
    backups = sorted([
        os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) 
        if f.startswith(f"backup_{user}_")
    ], key=os.path.getmtime)
    while len(backups) > max_backups:
        os.remove(backups.pop(0))

def create_backup(user):
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv")
        shutil.copy2(source_path, backup_path)
        manage_backups(user)

def load_data(user):
    path = f"portfolio_{user}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    create_backup(user)
    df.to_csv(f"portfolio_{user}.csv", index=False)

def remove_stock(symbol, user):
    df = load_data(user)
    df = df[df["股票代號"] != symbol]
    save_data(df, user)

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
# 3. 介面組件 (表格與小計)
# ==========================================
COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def update_sort(column_name):
    if st.session_state.sort_col == column_name:
        st.session_state.sort_asc = not st.session_state.sort_asc
    else:
        st.session_state.sort_col = column_name
        st.session_state.sort_asc = False

def display_headers(key_suffix, current_user):
    cols = st.columns(COLS_RATIO)
    headers = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本(原幣)"), ("現值", "現值(原幣)"), ("獲利", "獲利(原幣)"), ("報酬率%", "獲利率(%)")]
    for col, (label, col_name) in zip(cols[:-1], headers):
        arrow = "▲" if st.session_state.sort_asc and st.session_state.sort_col == col_name else "▼" if st.session_state.sort_col == col_name else ""
        if col.button(f"{label} {arrow}", key=f"h_{col_name}_{key_suffix}_{current_user}"):
            update_sort(col_name); st.rerun()
    cols[-1].write("管理")

def display_stock_rows(df, currency_type, current_user):
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        color = "red" if row["獲利(原幣)"] > 0 else "green"
        c1.write(f"**{row['股票代號']}**")
        c2.write(f"{row['股數']:.2f}"); c3.write(f"{row['平均持有單價']:.2f}"); c4.write(f"{row['最新股價']:.2f}")
        c5.write(fmt.format(row['總投入成本(原幣)'])); c6.write(fmt.format(row['現值(原幣)']))
        c7.markdown(f":{color}[{fmt.format(row['獲利(原幣)'])}]"); c8.markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if current_user == "All": c9.write("🔒")
        else:
            if c9.button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
                remove_stock(row['股票代號'], current_user); st.rerun()

def display_subtotal_row(df, currency_type):
    """計算並顯示特定幣別的小計列"""
    t_cost = df["總投入成本(原幣)"].sum()
    t_val = df["現值(原幣)"].sum()
    t_profit = df["獲利(原幣)"].sum()
    roi = (t_profit / t_cost * 100) if t_cost > 0 else 0
    fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
    color = "red" if t_profit > 0 else "green"
    
    st.markdown("<hr style='margin: 5px 0; border-top: 2px solid #666;'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
    c1.markdown(f"**🔹 {currency_type} 小計**")
    c5.markdown(f"**{fmt.format(t_cost)}**")
    c6.markdown(f"**{fmt.format(t_val)}**")
    c7.markdown(f":{color}[**{fmt.format(t_profit)}**]")
    c8.markdown(f":{color}[**{roi:.2f}%**]")
    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. 主程式邏輯
# ==========================================
if "sort_col" not in st.session_state: st.session_state.sort_col = "獲利(原幣)"
if "sort_asc" not in st.session_state: st.session_state.sort_asc = False
if "last_updated" not in st.session_state: st.session_state.last_updated = "尚未更新"

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        st.subheader(f"📝 新增 {current_user} 持股")
        with st.form("add_form"):
            s_in = st.text_input("股票代號", "2330.TW").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=100.0)
            c_in = st.number_input("成本單價", min_value=0.0, value=600.0)
            if st.form_submit_button("新增並備份"):
                df = load_data(current_user)
                save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                st.rerun()
        if st.button(f"🚨 清空 {current_user}"):
            if os.path.exists(f"portfolio_{current_user}.csv"):
                create_backup(current_user); os.remove(f"portfolio_{current_user}.csv"); st.rerun()

# 資料載入
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

st.title(f"📈 {current_user} 投資組合戰情室")
if st.button("🔄 刷新即時行情"):
    st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
    st.rerun()

tab1, tab2 = st.tabs(["📊 庫存與配置分析", "🧠 AI 技術健診"])

with tab1:
    if df_record.empty:
        st.info("尚無數據。")
    else:
        # 計算邏輯
        usd_rate = get_exchange_rate()
        df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
        
        # 加權平均計算
        def weighted_avg(g):
            t_q = g['股數'].sum()
            avg_c = (g['股數'] * g['持有成本單價']).sum() / t_q if t_q > 0 else 0
            return pd.Series({'股數': t_q, '平均持有單價': avg_c})
        
        portfolio = df_record.groupby(["股票代號", "幣別"]).apply(weighted_avg, include_groups=False).reset_index()
        prices = get_current_prices(portfolio["股票代號"].tolist())
        portfolio["最新股價"] = portfolio["股票代號"].map(prices)
        portfolio = portfolio.dropna(subset=["最新股價"])
        
        portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["平均持有單價"]
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)

        # 看板
        t_val = portfolio["現值(TWD)"].sum()
        t_profit_twd = (portfolio["獲利(原幣)"] * portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("總獲利 (TWD)", f"${t_profit_twd:,.0f}")
        c3.metric("總報酬率", f"{(t_profit_twd/(t_val-t_profit_twd)*100):.2f}%" if t_val!=t_profit_twd else "0%")

        st.divider()

        # --- 配置圓餅圖與下拉選單 ---
        st.subheader("🎯 投資組合配置圖解")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            currency_dist = portfolio.groupby("幣別")["現值(TWD)"].sum().reset_index()
            fig_cur = px.pie(currency_dist, values="現值(TWD)", names="幣別", title="市場資金比例 (TWD計價)", hole=0.5)
            st.plotly_chart(fig_cur, use_container_width=True)

        with chart_col2:
            # 下拉選單切換組合分佈
            view_option = st.selectbox("選擇配置視圖：", ["全部組合", "台股組合", "美股組合"], key="pie_view")
            if view_option == "台股組合":
                plot_df = portfolio[portfolio["幣別"] == "TWD"]
            elif view_option == "美股組合":
                plot_df = portfolio[portfolio["幣別"] == "USD"]
            else:
                plot_df = portfolio
            
            if not plot_df.empty:
                fig_stock = px.pie(plot_df, values="現值(TWD)", names="股票代號", title=f"{view_option}分佈", hole=0.5)
                fig_stock.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_stock, use_container_width=True)
            else:
                st.write("目前無相關持股可顯示圖表。")

        st.divider()

        # --- 庫存清單與小計 ---
        for label, cur in [("🇹🇼 台股列表", "TWD"), ("🇺🇸 美股列表", "USD")]:
            sub = portfolio[portfolio["幣別"] == cur]
            if not sub.empty:
                st.subheader(label)
                display_headers(cur.lower(), current_user)
                display_stock_rows(sub, cur, current_user)
                # 這裡調用小計函數
                display_subtotal_row(sub, cur)

with tab2:
    if not df_record.empty:
        target = st.selectbox("分析標的", portfolio["股票代號"].tolist())
        st.write(f"正在分析 {target}...")
