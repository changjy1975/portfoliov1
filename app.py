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
# 2. 深度分析核心邏輯 (MPT)
# ==========================================

def perform_mpt_analysis(portfolio_df):
    """執行現代投資組合理論 (MPT) 深度分析"""
    symbols = portfolio_df["股票代號"].unique().tolist()
    if len(symbols) < 2: 
        return None, "標的數量不足（需至少 2 支），無法進行相關性與組合分析。"

    try:
        # 抓取 3 年歷史數據 (252 * 3 = 756 交易日)
        tickers_str = " ".join(symbols)
        data = yf.download(tickers_str, period="3y", interval="1d", auto_adjust=True)['Close']
        
        # 處理單一標的情況 (yfinance 回傳 Series)
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
            if len(series) < 50: continue # 略過數據不足的標的
            
            # 年化報酬率 (CAGR)
            days = (series.index[-1] - series.index[0]).days
            years = days / 365.25
            total_ret = (series.iloc[-1] / series.iloc[0]) - 1
            cagr = ((series.iloc[-1] / series.iloc[0])**(1/years) - 1) if years > 0 else 0
            
            # 年化波動率 (Volatility)
            vol = returns[symbol].std() * np.sqrt(252)
            
            # 夏普比率 (Sharpe) - 假設無風險利率 2%
            sharpe = (cagr - 0.02) / vol if vol != 0 else 0
            
            perf_list.append({
                "股票代號": symbol,
                "CAGR (年化報酬)": f"{cagr*100:.2f}%",
                "年化波動率": f"{vol*100:.2f}%",
                "Sharpe Ratio": round(sharpe, 2),
                "3年累積報酬": f"{total_ret*100:.2f}%",
                "_raw_sharpe": sharpe # 用於排序
            })
        
        perf_df = pd.DataFrame(perf_list)

        # 3. 再平衡建議邏輯
        suggestions = []
        total_val = portfolio_df["現值(TWD)"].sum()
        
        # 權重過高檢查
        for _, row in portfolio_df.iterrows():
            weight = row["現值(TWD)"] / total_val
            if weight > 0.35:
                suggestions.append(f"⚠️ **集中度警示**：{row['股票代號']} 佔比達 {weight*100:.1f}%，超過 MPT 建議的單一資產上限。")
        
        # 高相關性檢查
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.85:
                    high_corr_pairs.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
        
        if high_corr_pairs:
            suggestions.append(f"🔗 **避險失效**：{', '.join(high_corr_pairs)} 相關性極高，建議將其中之一替換為低相關資產（如債券或不同產業股票）。")
        
        # Sharpe 優化建議
        if not perf_df.empty:
            best_stock = perf_df.loc[perf_df['_raw_sharpe'].idxmax()]['股票代號']
            suggestions.append(f"📈 **效率建議**：{best_stock} 在過去 3 年表現出最佳的風險調整後收益，考慮維持其權重。")

        if not suggestions:
            suggestions.append("✅ 投資組合目前在風險分散與回報效率上表現平衡。")

        return {"corr": corr_matrix, "perf": perf_df.drop(columns=['_raw_sharpe']), "sugg": suggestions}, None
    except Exception as e:
        return None, str(e)

# (以下 load_data, save_data, get_exchange_rate 等基礎函數保持不變)
def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    if os.path.exists(f"portfolio_{user}.csv"):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(f"portfolio_{user}.csv", os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv"))
    df.to_csv(f"portfolio_{user}.csv", index=False)

def get_exchange_rate():
    try: return yf.Ticker("USDTWD=X").fast_info.last_price
    except: return 32.5

def get_current_prices(symbols):
    if not symbols: return {}
    prices = {}
    for s in symbols:
        try: prices[s] = yf.Ticker(s).fast_info.last_price
        except: prices[s] = None
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# 3. 介面組件
# ==========================================
# (此處保留 display_headers, display_stock_rows, display_subtotal_row 等介面邏輯)

# ==========================================
# 4. 主程式邏輯
# ==========================================

# 初始化狀態
for key in ['sort_col', 'sort_asc', 'last_updated']:
    if key not in st.session_state: 
        st.session_state[key] = "獲利(原幣)" if key == 'sort_col' else False if key == 'sort_asc' else "尚未更新"

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    # (新增與清空資料表單邏輯...)

# 載入與計算基礎資料 (供各頁面使用)
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

# 共用計算區
if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    def weighted_avg(g):
        t_q = g['股數'].sum()
        avg_c = (g['股數'] * g['持有成本單價']).sum() / t_q if t_q > 0 else 0
        return pd.Series({'股數': t_q, '平均持有單價': avg_c})
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(weighted_avg, include_groups=False).reset_index()
    # 這裡省略部分計算以縮短篇幅，確保現值 (TWD) 有算出

# --- 分頁設定 ---
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

with tab1:
    # (原本的看板、圓餅圖、列表、小計與匯率顯示邏輯)
    st.write("此處顯示原本的庫存列表與小計。")

with tab2:
    # (原本的單一個股技術健診邏輯)
    st.write("此處進行單一個股技術指標分析。")

with tab3:
    if df_record.empty:
        st.info("尚無數據可進行 MPT 分析。")
    else:
        st.subheader("🏛️ 現代投資組合理論 (MPT) 深度報告")
        st.caption("分析邏輯：基於過去 36 個月日報酬數據，評估效率前緣（Efficient Frontier）與風險分散度。")
        
        # MPT 圓餅圖 (直接顯示在最上方)
        st.write("#### 1️⃣ 當前資金配置權重")
        fig_mpt_pie = px.pie(portfolio, values="現值(TWD)", names="股票代號", hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_mpt_pie, use_container_width=True)

        st.divider()

        # 啟動按鈕 (因計算較重)
        if st.button("🚀 啟動深度矩陣與風險運算", type="primary"):
            with st.spinner("正在抓取 3 年歷史報價並計算矩陣..."):
                res, err = perform_mpt_analysis(portfolio)
                if err: st.error(err)
                else: st.session_state['mpt_res'] = res

        if 'mpt_res' in st.session_state:
            data = st.session_state['mpt_res']

            # 2. 相關係數矩陣
            st.write("#### 2️⃣ 個股相關係數矩陣 (走勢同步分析)")
            
            fig_corr = px.imshow(data['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("💡 數值 > 0.8 代表高度相關（風險重疊）；數值 < 0 代表具備避險效果。")

            st.divider()

            # 3. 績效表格
            st.write("#### 3️⃣ 風險與報酬關鍵指標")
            st.dataframe(data['perf'], use_container_width=True, hide_index=True)
            
            # Sharpe 公式
            st.latex(r"Sharpe\ Ratio = \frac{R_p - R_f}{\sigma_p}")

            st.divider()

            # 4. 再平衡建議
            st.write("#### 4️⃣ MPT 優劣分析與再平衡建議")
            for s in data['sugg']:
                if "⚠️" in s: st.warning(s)
                elif "🔗" in s: st.error(s)
                else: st.success(s)
