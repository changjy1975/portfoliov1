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
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

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
    except: return 32.5

def get_current_prices(symbols):
    if not symbols: return {}
    prices = {}
    for s in symbols:
        try:
            t = yf.Ticker(s)
            p = t.fast_info.last_price
            if p is None or pd.isna(p):
                hist = t.history(period="1d")
                p = hist['Close'].iloc[-1] if not hist.empty else None
            prices[s] = p
        except: prices[s] = None
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# --- MPT 分析函數 ---
def perform_mpt_analysis(portfolio_df):
    symbols = portfolio_df["股票代號"].unique().tolist()
    if len(symbols) < 2: return None, "標的數量不足（需至少 2 支）"
    try:
        tickers_str = " ".join(symbols)
        data = yf.download(tickers_str, period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=symbols[0])
        data = data.dropna(how='all').ffill()
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        perf_list = []
        for symbol in data.columns:
            series = data[symbol].dropna()
            if len(series) < 50: continue
            days = (series.index[-1] - series.index[0]).days
            years = days / 365.25
            total_ret = (series.iloc[-1] / series.iloc[0]) - 1
            cagr = ((series.iloc[-1] / series.iloc[0])**(1/years) - 1) if years > 0 else 0
            vol = returns[symbol].std() * np.sqrt(252)
            sharpe = (cagr - 0.02) / vol if vol != 0 else 0
            perf_list.append({"股票代號": symbol, "CAGR": f"{cagr*100:.2f}%", "波動率": f"{vol*100:.2f}%", "Sharpe Ratio": round(sharpe, 2), "_raw": sharpe})
        
        perf_df = pd.DataFrame(perf_list)
        suggestions = []
        total_val = portfolio_df["現值(TWD)"].sum()
        for _, row in portfolio_df.iterrows():
            weight = row["現值(TWD)"] / total_val
            if weight > 0.35: suggestions.append(f"⚠️ **集中度警示**：{row['股票代號']} 佔比達 {weight*100:.1f}%。")
        
        return {"corr": corr_matrix, "perf": perf_df.drop(columns=['_raw']), "sugg": suggestions}, None
    except Exception as e: return None, str(e)

# ==========================================
# 3. 介面顯示組件
# ==========================================
COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def display_subtotal_row(df, currency_type, usd_rate):
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

    if currency_type == "USD":
        st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        c1.markdown("<span style='color: gray; font-size: 0.9em;'>└ 換算台幣 (TWD)</span>", unsafe_allow_html=True)
        c5.markdown(f"<span style='color: gray; font-size: 0.9em;'>${(t_cost * usd_rate):,.0f}</span>", unsafe_allow_html=True)
        c6.markdown(f"<span style='color: gray; font-size: 0.9em;'>${(t_val * usd_rate):,.0f}</span>", unsafe_allow_html=True)
        c7.markdown(f"<span style='color: gray; font-size: 0.9em;'>${(t_profit * usd_rate):,.0f}</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. 主程式邏輯
# ==========================================

if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利(原幣)"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False
if 'last_updated' not in st.session_state: st.session_state.last_updated = "尚未更新"

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form"):
            st.subheader(f"📝 新增 {current_user} 持股")
            s_in = st.text_input("代號", "2330.TW").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=100.0)
            c_in = st.number_input("成本", min_value=0.0, value=600.0)
            if st.form_submit_button("新增並備份"):
                df = load_data(current_user)
                save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                st.rerun()

# --- 資料載入與全局計算 (解決 ValueError 關鍵) ---
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

if not df_record.empty:
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
    
    # 重點：算出 Tab 1 與 Tab 3 圓餅圖通用的 現值(TWD)
    portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
    portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
    portfolio["獲利(TWD)"] = portfolio["獲利(原幣)"] * portfolio["匯率因子"]

# --- 分頁介面 ---
st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

with tab1:
    if df_record.empty: st.info("無數據。")
    else:
        # 看板與圖表顯示 (與之前版本一致)
        t_val = portfolio["現值(TWD)"].sum()
        t_prof = portfolio["獲利(TWD)"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("總獲利 (TWD)", f"${t_prof:,.0f}")
        c3.metric("總報酬率", f"{(t_prof/(t_val-t_prof)*100):.2f}%" if t_val!=t_prof else "0%")
        c4.metric("💱 匯率", f"{usd_rate:.2f}")
        # ... 圓餅圖與表格略
        st.write("表格與列表內容...")

with tab2:
    st.write("個股健診功能...")

with tab3:
    if df_record.empty: st.info("無數據。")
    else:
        st.subheader("⚖️ 現代投資組合理論 (MPT) 分析")
        
        # 繪製圓餅圖 (現值(TWD) 已在全局計算完成，不會再報錯)
        fig_mpt_pie = px.pie(portfolio, values="現值(TWD)", names="股票代號", hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_mpt_pie, use_container_width=True)

        

        if st.button("🚀 啟動深度分析 (36個月數據)", type="primary"):
            with st.spinner("計算中..."):
                res, err = perform_mpt_analysis(portfolio)
                if err: st.error(err)
                else:
                    st.session_state['mpt_res'] = res

        if 'mpt_res' in st.session_state:
            data = st.session_state['mpt_res']
            st.write("#### 相關係數矩陣")
            fig_corr = px.imshow(data['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.dataframe(data['perf'], use_container_width=True)
            for s in data['sugg']: st.info(s)
