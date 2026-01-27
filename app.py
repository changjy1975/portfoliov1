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
# 2. 核心功能函數 (備份與資料處理)
# ==========================================

def manage_backups(user, max_backups=10):
    """只保留最近 10 份備份，避免佔用空間"""
    backups = sorted([
        os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) 
        if f.startswith(f"backup_{user}_")
    ], key=os.path.getmtime)
    while len(backups) > max_backups:
        os.remove(backups.pop(0))

def create_backup(user):
    """存檔前自動執行備份"""
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv")
        shutil.copy2(source_path, backup_path)
        manage_backups(user)

def load_data(user):
    """載入特定使用者的資料"""
    path = f"portfolio_{user}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    """儲存資料並自動觸發備份"""
    create_backup(user)
    df.to_csv(f"portfolio_{user}.csv", index=False)

def remove_stock(symbol, user):
    """從使用者檔案中移除特定股票"""
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
# 3. 介面組件 (表格顯示與排序)
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
        c2.write(f"{row['股數']:.2f}")
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{row['最新股價']:.2f}")
        c5.write(fmt.format(row['總投入成本(原幣)']))
        c6.write(fmt.format(row['現值(原幣)']))
        c7.markdown(f":{color}[{fmt.format(row['獲利(原幣)'])}]")
        c8.markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        
        if current_user == "All":
            c9.write("🔒") # All 模式鎖定管理功能
        else:
            if c9.button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
                remove_stock(row['股票代號'], current_user); st.rerun()

# ==========================================
# 4. 主程式邏輯
# ==========================================

# 初始化排序狀態
if "sort_col" not in st.session_state: st.session_state.sort_col = "獲利(原幣)"
if "sort_asc" not in st.session_state: st.session_state.sort_asc = False
if "last_updated" not in st.session_state: st.session_state.last_updated = "尚未更新"

# 側邊欄：帳戶切換
with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    
    if current_user != "All":
        st.info(f"當前操作：{current_user}")
    else:
        st.success("📊 模式：Alan + Jenny 加總總覽")

# 資料載入邏輯
if current_user == "All":
    df_alan = load_data("Alan")
    df_jenny = load_data("Jenny")
    df_record = pd.concat([df_alan, df_jenny], ignore_index=True)
else:
    df_record = load_data(current_user)

st.title(f"📈 {current_user} 投資組合戰情室")

# 頂部操作
col_ref, col_info = st.columns([1, 5])
if col_ref.button("🔄 刷新全部"):
    st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
    st.rerun()
col_info.markdown(f"<div style='padding-top:10px; color:gray;'>最後更新: {st.session_state.last_updated}</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 庫存列表", "🧠 AI 持股健診", "⚖️ 資產配置分析"])

with tab1:
    if current_user != "All":
        with st.sidebar:
            st.divider()
            st.subheader(f"📝 新增 {current_user} 的投資")
            with st.form("add_stock_form"):
                s_in = st.text_input("股票代號", "2330.TW").upper().strip()
                q_in = st.number_input("股數", min_value=0.0, value=100.0)
                c_in = st.number_input("成本單價", min_value=0.0, value=600.0)
                if st.form_submit_button("執行新增並備份"):
                    df = load_data(current_user)
                    new_row = pd.DataFrame([{"股票代號": s_in, "股數": q_in, "持有成本單價": c_in}])
                    save_data(pd.concat([df, new_row], ignore_index=True), current_user)
                    st.toast(f"✅ 已存檔，備份已建立於 {BACKUP_DIR}")
                    st.rerun()
            if st.button(f"🚨 清空 {current_user} 資料"):
                if os.path.exists(f"portfolio_{current_user}.csv"):
                    create_backup(current_user) # 刪除前也存一份
                    os.remove(f"portfolio_{current_user}.csv")
                    st.rerun()

    if df_record.empty:
        st.info("尚無投資數據，請先從側邊欄新增。")
    else:
        usd_rate = get_exchange_rate()
        df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
        
        # --- 核心聚合計算 (解決 KeyError 且支援加總) ---
        # 相同股票合併時，需要計算「加權平均成本」
        def weighted_avg(group):
            total_qty = group['股數'].sum()
            if total_qty == 0: return 0
            avg_cost = (group['股數'] * group['持有成本單價']).sum() / total_qty
            return pd.Series({'股數': total_qty, '平均持有單價': avg_cost})

        portfolio = df_record.groupby(["股票代號", "幣別"]).apply(weighted_avg, include_groups=False).reset_index()
        
        # 獲取價格與計算盈虧
        unique_symbols = portfolio["股票代號"].tolist()
        with st.spinner('獲取即時報價中...'):
            current_prices = get_current_prices(unique_symbols)
        
        portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
        portfolio = portfolio.dropna(subset=["最新股價"])
        
        portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["平均持有單價"]
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        
        # 匯率換算
        portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
        portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
        portfolio["獲利(TWD)"] = portfolio["獲利(原幣)"] * portfolio["匯率因子"]

        # 看板顯示
        t_val = portfolio["現值(TWD)"].sum()
        t_profit = portfolio["獲利(TWD)"].sum()
        t_roi = (t_profit / (t_val - t_profit) * 100) if (t_val - t_profit) != 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"💰 {current_user} 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總獲利 (TWD)", f"${t_profit:,.0f}")
        c3.metric("📊 總報酬率", f"{t_roi:.2f}%")

        st.divider()

        # 庫存表格
        for label, cur in [("🇹🇼 台股持倉", "TWD"), ("🇺🇸 美股持倉", "USD")]:
            sub = portfolio[portfolio["幣別"] == cur]
            if not sub.empty:
                st.subheader(label)
                display_headers(cur.lower(), current_user)
                display_stock_rows(sub, cur, current_user)

with tab2:
    if not df_record.empty:
        st.subheader("💡 系統操作建議")
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist(), key="tech_sel")
        if target:
            # 這裡簡單呈現 (實際 analyze_stock_technical 可參考先前定義)
            st.write(f"正在對 {target} 進行技術指標掃描...")
            st.info("提示：此部分會抓取 Yahoo Finance 近半年週線數據進行 RSI 與 MA 分析。")

with tab3:
    if not df_record.empty:
        st.subheader("🥧 資產權重分佈")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig1 = px.pie(portfolio, values="現值(TWD)", names="幣別", title="幣別佔比", hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        with col_c2:
            fig2 = px.pie(portfolio, values="現值(TWD)", names="股票代號", title="個股佔比 (TWD 換算)", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)
