# 日米株６００銘柄モメンタムスクリーニング PURE Edition (No A.U.R.A Fusion)

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================================
# 1. データ取得：広域ユニバース (Robust Edition)
# ==========================================================
def get_tickers_lumus():
    print("🌌 L.U.M.U.S. ユニバース（S&P500 + 日本株精鋭）を構築中...")
    us_tickers = []

    # Plan A: Wikipedia
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        us_tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
    except: pass

    # Plan B: GitHub CSV
    if len(us_tickers) < 100:
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            s = requests.get(url).content
            df_csv = pd.read_csv(io.StringIO(s.decode('utf-8')))
            us_tickers = df_csv["Symbol"].tolist()
        except: pass

    # Plan C: Static List (Backup)
    if len(us_tickers) < 100:
        us_tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "JPM", "V", "WMT", "XOM", "CAT", "COST"]

    # 日本株（新リスト）
    jp_tickers = [
        "7203.T", "6758.T", "8306.T", "8035.T", "9984.T", "9432.T", "6861.T", "6098.T",
        "4063.T", "6954.T", "7974.T", "6301.T", "4568.T", "6501.T", "7741.T", "7267.T",
        "6273.T", "4543.T", "8058.T", "8001.T", "8031.T", "8053.T", "8002.T", "8316.T",
        "8411.T", "8766.T", "8801.T", "8802.T", "8591.T", "8725.T", "8750.T",
        "6857.T", "6146.T", "6723.T", "6920.T", "7735.T", "6981.T", "6503.T", "6702.T",
        "6752.T", "6506.T", "6965.T", "7729.T", "6869.T", "6971.T", "6315.T", "4062.T", "7701.T",
        "7011.T", "7012.T", "7013.T", "6367.T", "6113.T", "6481.T", "1801.T", "1802.T", "1803.T",
        "1812.T", "1925.T", "1928.T", "1808.T", "1721.T", "5803.T", "5802.T",
        "7201.T", "7269.T", "7270.T", "5401.T", "5713.T", "1605.T", "5020.T", "9101.T",
        "9104.T", "9107.T", "3407.T", "4188.T", "4452.T", "4911.T", "4183.T",
        "9983.T", "3382.T", "7453.T", "3092.T", "4661.T", "4385.T", "2413.T", "4689.T",
        "4755.T", "9735.T", "3659.T", "4307.T", "3088.T", "3064.T", "2802.T", "2502.T",
        "2503.T", "4502.T", "4519.T", "4503.T", "4523.T", "9020.T", "9021.T", "9022.T",
        "9201.T", "9202.T", "9501.T", "9502.T", "9503.T"
    ]
    return us_tickers, jp_tickers

# ==========================================================
# 2. 市場レジーム判定 (防御システム)
# ==========================================================
def check_regime():
    indices = {"US": "^GSPC", "JP": "^N225"}
    regimes = {}
    print("📡 市場天気図（200日線）を観測中...")
    try:
        data = yf.download(list(indices.values()), period="2y", progress=False)["Close"].ffill()
        for region, ticker in indices.items():
            if ticker in data.columns:
                series = data[ticker].dropna()
                current = series.iloc[-1]
                ma200 = series.rolling(200).mean().iloc[-1]
                regimes[region] = "BULL" if current > ma200 else "BEAR"
            else: regimes[region] = "UNKNOWN"
    except: regimes = {"US": "BULL", "JP": "BULL"}
    return regimes

# ==========================================================
# 3. コアロジック: 3因子モデル + マルチ期間 + 高速化
# ==========================================================
def analyze_lumus_engine(tickers, region, regime):
    print(f"🔍 {region}市場: {len(tickers)} 銘柄を解析中...")
    end = datetime.today()
    start = end - timedelta(days=400) # 1年以上確保

    try:
        # 価格データ取得
        data = yf.download(tickers, start=start, end=end, progress=False)["Close"].ffill().bfill()
        daily_ret = data.pct_change(fill_method=None)
    except: return pd.DataFrame()

    metrics = {}
    for t in data.columns:
        if data[t].count() < 250: continue
        series = data[t].dropna()
        d_r = daily_ret[t].dropna()

        # --- A. Efficiency (マルチ期間モメンタム / ボラティリティ) ---
        p_now = series.iloc[-1]
        p_12m = series.iloc[-252] if len(series) >= 252 else series.iloc[0]
        p_6m = series.iloc[-126] if len(series) >= 126 else series.iloc[0]
        p_3m = series.iloc[-63] if len(series) >= 63 else series.iloc[0]

        r_12m, r_6m, r_3m = (p_now/p_12m)-1, (p_now/p_6m)-1, (p_now/p_3m)-1
        # 合成リターン (12M重視)
        composite_ret = (r_12m * 3 + r_6m * 2 + r_3m * 1) / 6

        vol = d_r.std() * np.sqrt(252)
        efficiency = composite_ret / vol if vol > 0 else 0

        # --- B. Quality (非対称性: 上がりやすく下がりにくい) ---
        avg_pos = d_r[d_r > 0].mean()
        avg_neg = abs(d_r[d_r < 0].mean())
        quality = avg_pos / avg_neg if avg_neg > 0 else 1.0

        # --- C. Valuation (簡易PBR代替: 高値からの距離 & ボラティリティ逆数) ---
        max_52w = series.max()
        prox_high = p_now / max_52w
        valuation_score = (1 / prox_high) * (1 / vol)

        metrics[t] = {
            "Efficiency": efficiency,
            "Quality": quality,
            "Valuation_Alt": valuation_score,
            "Volatility": vol,
            "Composite_Ret": composite_ret
        }

    df = pd.DataFrame(metrics).T
    if df.empty: return pd.DataFrame()

    # --- スコアリング (3因子モデル) ---
    def z(s): return (s - s.mean()) / s.std()

    # 初代の哲学: 40:40:20
    df["Total_Score"] = z(df["Efficiency"])*0.4 + z(df["Quality"])*0.4 + z(df["Valuation_Alt"])*0.2

    # レジームフィルター (弱気相場なら厳格化)
    if regime == "BEAR":
        df["Total_Score"] -= 3.0
        print(f"⚠️ {region}市場は弱気です。選定基準を引き上げました。")

    return df.sort_values("Total_Score", ascending=False)

# ==========================================================
# 4. ポートフォリオ構築 (Risk Parity)
# ==========================================================
def build_lumus_portfolio(df_us, df_jp):
    print("\n" + "="*80)
    print("🏰 L.U.M.U.S.-8 Alpha Portfolio (Pure Risk Parity & 3-Factor)")
    print("="*80)

    top_us = df_us.head(6)
    top_jp = df_jp.head(6)
    portfolio = pd.concat([top_us, top_jp])

    # リスクパリティ・ウェイト
    inv_vol = 1 / portfolio["Volatility"]
    weights = inv_vol / inv_vol.sum()
    portfolio["Weight"] = (weights * 100).round(1)

    # UI用ダミーカテゴリー設定
    portfolio["Category"] = "🛡️ CORE (Pure)"

    cols = ["Weight", "Total_Score", "Composite_Ret", "Volatility", "Efficiency", "Quality"]
    print(portfolio[cols].sort_values("Weight", ascending=False))
    return portfolio

# ==========================================================
# 5. レジーム連動型ポジションサイズ決定 (Final Safety Valve)
# ==========================================================
def determine_exposure(regimes):
    print("\n" + "="*60)
    print("🛡️ ポートフォリオ稼働率 (Market Exposure)")
    print("="*60)

    us_status = regimes.get("US", "UNKNOWN")
    jp_status = regimes.get("JP", "UNKNOWN")

    if us_status == "BULL" and jp_status == "BULL":
        exposure = 1.0
        msg = "🌞 快晴 (Full Throttle): 株式 100% / 現金 0%"
    elif us_status == "BULL" or jp_status == "BULL":
        exposure = 0.6
        msg = "⛅ 曇り (Caution): 株式 60% / 現金 40% (分散投資)"
    else:
        exposure = 0.2
        msg = "⛈️ 嵐 (Defense Mode): 株式 20% / 現金 80% (シェルター退避)"

    print(f"市場環境: US={us_status} | JP={jp_status}")
    print(f"👉 推奨稼働率: {msg}")
    return exposure

# ==========================================================
# 6. L.U.M.U.S.-8 Auto Order Generator (自動発注リスト生成)
# ==========================================================
def generate_trade_orders(df_core, total_budget_jpy):
    print("\n" + "="*70)
    print(" 🛒 L.U.M.U.S.-8 自動オーダー生成 (発注ロット計算 - PURE版)")
    print("="*70)

    try:
        usdjpy = yf.download("JPY=X", period="1d", progress=False)["Close"].iloc[-1]
        if isinstance(usdjpy, pd.Series): usdjpy = usdjpy.iloc[0]
        print(f"🔄 適用為替レート: 1 USD = {usdjpy:.2f} JPY\n")
    except:
        usdjpy = 150.0 
        print(f"⚠️ 為替取得失敗。仮レート(1 USD = 150 JPY)を適用します。\n")

    orders = []
    tickers = df_core.index.tolist()

    try:
        prices = yf.download(tickers, period="5d", progress=False)["Close"].ffill().iloc[-1]
    except Exception as e:
        print(f"価格データの取得に失敗しました: {e}")
        return

    for ticker in tickers:
        alloc_jpy = df_core.loc[ticker, "Allocation_Amt(JPY)"]
        if alloc_jpy <= 0: continue

        try:
            price_local = prices[ticker]
            if isinstance(price_local, pd.Series): price_local = price_local.iloc[0]
            if pd.isna(price_local): continue

            if str(ticker).endswith(".T"):
                price_jpy = price_local
                currency = "JPY"
            else:
                price_jpy = price_local * usdjpy
                currency = "USD"

            shares = int(alloc_jpy // price_jpy)
            actual_cost_jpy = shares * price_jpy

            orders.append({
                "Ticker": ticker,
                "Cat": df_core.loc[ticker, "Category"][:4],
                "Price_Local": f"{price_local:>7.2f} {currency}",
                "Shares(株)": shares,
                "Target_Amt(¥)": int(alloc_jpy),
                "Actual_Cost(¥)": int(actual_cost_jpy)
            })
        except: pass

    df_orders = pd.DataFrame(orders).set_index("Ticker")
    df_executable = df_orders[df_orders["Shares(株)"] > 0]
    print(df_executable[["Cat", "Price_Local", "Shares(株)", "Target_Amt(¥)", "Actual_Cost(¥)"]])

    total_actual = df_executable["Actual_Cost(¥)"].sum()
    cash_remainder = total_budget_jpy - total_actual

    print("\n" + "-"*70)
    print(f"✅ 発注予定総額: {total_actual:,.0f} 円")
    print(f"📦 残存キャッシュ (端数調整後): {cash_remainder:,.0f} 円")
    print("-"*70)

    return df_executable

# ==========================================================
# 7. メイン実行セクション
# ==========================================================
if __name__ == "__main__":
    regimes = check_regime()
    us_list, jp_list = get_tickers_lumus()
    
    # 3因子コア分析
    df_us = analyze_lumus_engine(us_list, "US", regimes["US"])
    df_jp = analyze_lumus_engine(jp_list, "JP", regimes["JP"])
    
    # ポートフォリオ生成
    final_port = build_lumus_portfolio(df_us, df_jp)
    print("\n✅ 完了。これが三体戦略における『泥に染まらない蓮』のPURE候補です。")

    # 稼働率と予算の決定
    exposure_ratio = determine_exposure(regimes)
    total_budget = 4500000 * exposure_ratio # 例: 投資資金450万円ベース
    
    # 最終アロケーションの計算
    final_port["Allocation_Amt(JPY)"] = (final_port["Weight"] / 100 * total_budget).round(0)
    
    print("\n💰 推奨アロケーション (投資資金 450万円ベース)")
    display_cols = ["Weight", "Allocation_Amt(JPY)", "Total_Score", "Efficiency"]
    print(final_port[display_cols].sort_values("Weight", ascending=False))
    print(f"\n📦 防衛的現金退避(CASH): {4500000 * (1 - exposure_ratio):,.0f} 円")

    # オーダー生成（PUREポートフォリオを直接流し込む）
    generate_trade_orders(final_port, total_budget)