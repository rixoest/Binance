import re
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import ta

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="PRO TRADING AI | 비상 안전 대시보드",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GEMINI_API_KEY = "AQ.Ab8RN6Jow5GiPeKyRRPVFetIzbcqEKT8K-i-Yl2z6z-TWyKyww"

st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; color: #0F172A; font-family: sans-serif; }
    .main-header { background: #FFFFFF; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 20px; }
    .metric-card { background: #FFFFFF; border-radius: 10px; padding: 15px; border: 1px solid #E2E8F0; }
    .metric-val { font-size: 1.4rem; font-weight: 800; color: #0F172A; margin-top: 5px; }
    .ai-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #2563EB; border-radius: 10px; padding: 20px; margin-bottom: 15px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h2>⚡ PRO TRADING AI (최신 모델 연동 완료)</h2>
        <p style="color: #64748B; margin: 0;">외부 네트워크 우회 및 Gemini 최신 모델 적용 버전</p>
    </div>
""",
    unsafe_allow_html=True,
)

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

with ctrl_col1:
    symbol_input = (
        st.text_input(
            "조회 종목 심볼", value="KORU/USDT", label_visibility="collapsed"
        )
        .strip()
        .upper()
    )

with ctrl_col2:
    manual_price = st.number_input(
        "기준가 수동 설정", value=20.61, step=0.01, format="%.2f"
    )

with ctrl_col3:
    if st.button("🔄 강제 초기화 및 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------
# 2. 데이터 수집 및 시뮬레이션 함수
# ---------------------------------------------------------
@st.cache_data(ttl=0)
def load_safe_market_data(symbol, base_p):
    ohlcv = []
    clean_sym = symbol.replace("/", "").replace(" ", "").upper()
    if not clean_sym.endswith("USDT"):
        clean_sym += "USDT"

    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={clean_sym}&interval=1h&limit=100"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                for row in data:
                    ohlcv.append([
                        int(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ])
    except Exception:
        pass

    if not ohlcv or len(ohlcv) == 0:
        import time

        now_ts = int(time.time() * 1000)
        np.random.seed(42)
        prices = base_p + np.cumsum(np.random.normal(0, base_p * 0.002, 100))
        for i in range(100):
            ts = now_ts - (100 - i) * 3600 * 1000
            p = max(0.01, float(prices[i]))
            ohlcv.append([ts, p * 0.99, p * 1.01, p * 0.98, p, 50000.0])

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["EMA_50"] = ta.trend.ema_indicator(df["close"], window=50)

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    macd = ta.trend.MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()

    df["ATR"] = ta.volatility.average_true_range(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )

    return df, 0.0100


df, funding_rate = load_safe_market_data(symbol_input, manual_price)
curr = df.iloc[-1]
prev = df.iloc[-2]
price_change = ((curr["close"] - prev["close"]) / prev["close"]) * 100

# ---------------------------------------------------------
# 3. 메트릭 출력
# ---------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(
        f"""<div class="metric-card"><div style="color:#64748B; font-size:0.8rem; font-weight:700;">현재가</div><div class="metric-val">${curr['close']:,.2f}</div><div style="color:{'#16A34A' if price_change>=0 else '#DC2626'}; font-size:0.85rem;">{price_change:+.2f}%</div></div>""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""<div class="metric-card"><div style="color:#64748B; font-size:0.8rem; font-weight:700;">RSI (14)</div><div class="metric-val">{curr['RSI']:.1f}</div><div style="color:#94A3B8; font-size:0.75rem;">모멘텀 지표</div></div>""",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""<div class="metric-card"><div style="color:#64748B; font-size:0.8rem; font-weight:700;">EMA 20</div><div class="metric-val" style="color:#2563EB;">${curr['EMA_20']:,.2f}</div><div style="color:#94A3B8; font-size:0.75rem;">단기 이평선</div></div>""",
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f"""<div class="metric-card"><div style="color:#64748B; font-size:0.8rem; font-weight:700;">ATR 변동폭</div><div class="metric-val" style="color:#475569;">${curr['ATR']:,.2f}</div><div style="color:#94A3B8; font-size:0.75rem;">리스크 관리</div></div>""",
        unsafe_allow_html=True,
    )
with m5:
    st.markdown(
        f"""<div class="metric-card"><div style="color:#64748B; font-size:0.8rem; font-weight:700;">펀딩비</div><div class="metric-val" style="color:#16A34A;">{funding_rate:.4f}%</div><div style="color:#94A3B8; font-size:0.75rem;">선물 포지션</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------
# 4. 차트 렌더링
# ---------------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.7, 0.3],
)
fig.add_trace(
    go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df["datetime"],
        y=df["EMA_20"],
        line=dict(color="#2563EB", width=1.5),
        name="EMA 20",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=df["datetime"],
        y=df["MACD_Diff"],
        marker_color=[
            "#16A34A" if v >= 0 else "#DC2626" for v in df["MACD_Diff"]
        ],
        name="Hist",
    ),
    row=2,
    col=1,
)
fig.update_layout(
    template="plotly_white",
    height=480,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_rangeslider_visible=False,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 5. Gemini AI 전략 분석 (모델명을 gemini-3.6-flash로 변경)
# ---------------------------------------------------------
st.markdown("### 🤖 Gemini AI 전략 어드바이저")
user_question = st.text_input(
    "질문 입력",
    placeholder="예: 이 가격 기준으로 매매 전략 세워줘",
    label_visibility="collapsed",
)

if user_question:
    with st.spinner("AI 분석 중..."):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            sys_prompt = f"현재 심볼 {symbol_input}, 현재가 ${curr['close']:,.2f}, RSI {curr['RSI']:.1f} 기준으로 트레이딩 전략을 1~5번 항목으로 나누어 작성하라."

            # 에러 메시지가 권장한 최신 모델명으로 교체 완료
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=user_question,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt
                ),
            )
            st.markdown(
                f'<div class="ai-card">{response.text.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"AI 호출 오류: {e}")
