import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
import shutil
from datetime import datetime, timedelta
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
# 2. 深度分析核心邏輯
# ==========================================

def perform_portfolio_analysis(portfolio_df):
    """執行深度風險與報酬分析"""
    symbols = portfolio_df["股票代號"].unique().tolist()
    if len(symbols) < 2: return None, "標的數量不足，無法進行相關性分析。"

    try:
        # 抓取 3 年歷史數據
        tickers_str = " ".join(symbols)
        data = yf.download(tickers_str, period="3y", interval="1d", auto_adjust=True)['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=symbols[0])
            
        data = data.dropna(how='all').ffill()
        returns = data.pct_change().dropna()
        
        # 1. 相關係數矩陣
        corr_matrix = returns.corr()
        
        # 2. 風險報酬指標計算
        perf_list = []
        for symbol in data.columns:
            series = data[symbol].dropna()
            if len(series) < 50: continue
            
            # 年化報酬率 (CAGR)
            years = (series.index[-1] - series.index[0]).days / 365.25
            total_ret = (series.iloc[-1] / series.iloc[0]) - 1
            cagr = ((series.iloc[-1] / series.iloc[0])**(1/years) - 1) if years > 0 else 0
            
            # 波動率 (Vol)
            vol = returns[symbol].std() * np.sqrt(252)
            
            # 夏普比率 (Sharpe) - 假設無風險利率 2%
            sharpe = (cagr - 0.02) / vol if vol != 0 else 0
            
            perf_list.append({
                "股票代號": symbol,
                "CAGR (%)": round(cagr * 100, 2),
                "年化波動率 (%)": round(vol * 100, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "3年總報酬 (%)": round(total_ret * 100, 2)
            })
        
        perf_df = pd.DataFrame(perf_list)

        # 3. 再平衡建議邏輯
        suggestions = []
        total_val = portfolio_df["現值(TWD)"].sum()
        for _, row in portfolio_df.iterrows():
            weight = row["現值(TWD)"] / total_val
            if weight > 0.3:
                suggestions.append(f"⚠️ **集中度警示**：{row['股票代號']} 佔比達 {weight*100:.1f}%，建議考慮分批減碼以分散風險。")
        
        # 尋找高相關性對
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.85:
                    high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
        
        if high_corr:
            suggestions.append(f"🔗 **連動風險**：{', '.join(high_corr)} 走勢高度相關，無法有效避險。")
        
        if not suggestions:
            suggestions.append("✅ 組合配置目前相當健康，無明顯風險集中情況。")

        return {"corr": corr_matrix, "perf": perf_df, "sugg": suggestions}, None
    except Exception as e:
        return None, str(e)

# (其餘 load_data, save_data, get_exchange_rate 等函數維持不變)
def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    if os.path.exists(f"portfolio_{user}.csv"):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(f"portfolio_{user}.csv", os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv"))
    df.to_csv(f"portfolio_{user}.csv", index=False)

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

def get_exchange_rate():
    try:
        return yf.Ticker("USDTWD=X").fast_info.last_price
    except: return 32.5

def get_current_prices(symbols):
    if not symbols: return {}
    prices = {}
    for s in symbols:
        try: prices[s] = yf.Ticker(s).fast_info.last_price
        except: prices[s] = None
    return prices

# ==========================================
# 3. UI 與主程式
# ==========================================
with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form"):
            st.subheader(f"新增 {current_user} 持股")
            s_in = st.text_input("股票代號", "2330.TW").upper()
            q_in = st.number_input("股數", min_value=0.0, value=100.0)
            c_in = st.number_input("成本單價", min_value=0.0, value=600.0)
            if st.form_submit_button("新增並備份"):
                df = load_data(current_user)
                save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                st.rerun()

# 加載資料
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

tab1, tab2 = st.tabs(["📊 庫存配置", "🧠 深度診斷與分析"])

with tab1:
    if not df_record.empty:
        # (這裡放原本的庫存計算與顯示邏輯，包含圓餅圖與小計)
        # 為了簡潔，此處省略重複代碼，確保與先前版本一致即可
        st.write("請參照前一版本顯示庫存列表與小計。")
    else:
        st.info("尚無數據。")

with tab2:
    if df_record.empty:
        st.info("請先新增持股以進行深度診斷。")
    else:
        st.subheader("📋 投資組合深度健康報告")
        st.caption("本分析採用近 36 個月歷史數據進行回測計算。")
        
        # 準備資料
        usd_rate = get_exchange_rate()
        df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
        portfolio = df_record.groupby(["股票代號", "幣別"]).agg({"股數":"sum", "持有成本單價":"mean"}).reset_index()
        # 預先計算 TWD 現值供分析使用
        current_prices = get_current_prices(portfolio["股票代號"].tolist())
        portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
        portfolio = portfolio.dropna(subset=["最新股價"])
        portfolio["現值(TWD)"] = portfolio["股數"] * portfolio["最新股價"] * portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)

        if st.button("🚀 啟動深度分析 (抓取3年數據)", type="primary"):
            with st.spinner("正在下載數據並運算風險指標..."):
                res, err = perform_portfolio_analysis(portfolio)
                if err:
                    st.error(f"分析失敗: {err}")
                else:
                    st.session_state['adv_analysis'] = res

        if 'adv_analysis' in st.session_state:
            data = st.session_state['adv_analysis']
            
            # 1. 績效指標表格
            st.markdown("### 1️⃣ 個股風險報酬指標")
            st.dataframe(data['perf'], use_container_width=True, hide_index=True)
            st.caption("💡 點擊標題可進行排序。Sharpe Ratio > 1 代表風險調整後的報酬優異。")
            
            # 相關係數公式說明
            st.write("夏普比率公式：")
            st.latex(r"Sharpe Ratio = \frac{R_p - R_f}{\sigma_p}")

            st.divider()

            # 2. 相關係數矩陣
            st.markdown("### 2️⃣ 個股相關係數矩陣 (走勢同步分析)")
            
            fig_corr = px.imshow(data['corr'], text_auto=".2f", aspect="auto", 
                                 color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("⚠️ 數值越接近 1 代表走勢越同步；接近 0 代表無關；接近 -1 代表走勢相反（具備極佳避險效果）。")

            st.divider()

            # 3. 優劣分析與建議
            st.markdown("### 3️⃣ 優劣分析與再平衡建議")
            for sugg in data['sugg']:
                if "⚠️" in sugg:
                    st.warning(sugg)
                elif "🔗" in sugg:
                    st.error(sugg)
                else:
                    st.success(sugg)
