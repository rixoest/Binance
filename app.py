import re
import ccxt
from google import genai
from google.genai import types
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta

# ---------------------------------------------------------
# 1. 페이지 및 라이트(White) 테마 커스텀 CSS 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="PRO TRADING AI | 바이낸스 선물 대시보드",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GEMINI_API_KEY = "AQ.Ab8RN6Jow5GiPeKyRRPVFetIzbcqEKT8K-i-Yl2z6z-TWyKyww"

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        background: #FFFFFF;
        padding: 24px 28px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .main-title {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #64748B;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #2563EB;
        transform: translateY(-2px);
    }
    .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .metric-sub {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .ai-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .ai-card-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 10px;
        letter-spacing: -0.3px;
    }
    .ai-card-body {
        font-size: 0.98rem;
        color: #334155;
        line-height: 1.7;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. 헤더 및 검색 컨트롤
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">⚡ PRO TRADING AI DASHBOARD</div>
        <div class="sub-title">바이낸스 선물 실시간 정밀 시장 데이터 & Gemini AI 어드바이저</div>
    </div>
""",
    unsafe_allow_html=True,
)

ctrl_col1, ctrl_col2 = st.columns([3, 1])

with ctrl_col1:
    raw_symbol = st.text_input(
        "조회 종목 심볼", value="BTC/USDT", label_visibility="collapsed"
    )
    symbol_input = raw_symbol.strip().upper()

with ctrl_col2:
    refresh_btn = st.button("🔄 실시간 데이터 갱신", use_container_width=True)
    if refresh_btn:
        st.cache_data.clear()


# ---------------------------------------------------------
# 3. 데이터 수집 함수 (CCXT 기반 안정적 연동)
# ---------------------------------------------------------
@st.cache_data(ttl=20)
def load_market_data(symbol):
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # 심볼 포맷 표준화 (예: KORUUSDT -> KORU/USDT)
        clean_sym = symbol.replace('/', '').upper()
        if clean_sym.endswith('USDT'):
            base_coin = clean_sym[:-4]
            formatted_symbol = f"{base_coin}/USDT"
        else:
            formatted_symbol = f"{symbol}/USDT" if '/' not in symbol else symbol

        ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe='1h', limit=100)
        if not ohlcv or len(ohlcv) == 0:
            return None, 0.0

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 보조지표 계산
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()

        macd = ta.trend.MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()

        df['ATR'] = ta.volatility.average_true_range(
            high=df['high'], low=df['low'], close=df['close'], window=14
        )

        funding_rate = 0.0100
        try:
            funding_info = exchange.fetch_funding_rate(formatted_symbol)
            if funding_info and 'fundingRate' in funding_info:
                funding_rate = float(funding_info['fundingRate']) * 100
        except Exception:
            pass

        return df, funding_rate

    except Exception as e:
        return None, 0.0


# Market 데이터 로드
df, funding_rate = load_market_data(symbol_input)

if df is None:
    st.error(
        f"🚨 '{symbol_input}' 선물의 데이터를 불러오지 못했습니다.\n\n"
        f"**확인 사항**: Streamlit Cloud 앱 설정의 `requirements.txt`에 `ccxt` 패키지가 포함되어 있는지, "
        f"그리고 종목명이 올바른지 확인해 주세요."
    )
else:
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = ((curr['close'] - prev['close']) / prev['close']) * 100

    # ---------------------------------------------------------
    # 4. 카드형 대시보드
    # ---------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    color_change = "#16A34A" if price_change >= 0 else "#DC2626"

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{symbol_input} 현재가</div>
                <div class="metric-val">${curr['close']:,.2f}</div>
                <div style="color:{color_change}; font-size:0.9rem; font-weight:700;">{price_change:+.2f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with m2:
        rsi_val = curr['RSI'] if pd.notnull(curr['RSI']) else 50.0
        rsi_color = (
            "#DC2626"
            if rsi_val >= 70
            else ("#16A34A" if rsi_val <= 30 else "#D97706")
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">RSI (14)</div>
                <div class="metric-val" style="color:{rsi_color};">{rsi_val:.1f}</div>
                <div class="metric-sub">과매수(70) / 과매도(30)</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with m3:
        ema_val = curr['EMA_20'] if pd.notnull(curr['EMA_20']) else curr['close']
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EMA (20)</div>
                <div class="metric-val" style="color: #2563EB;">${ema_val:,.2f}</div>
                <div class="metric-sub">단기 핵심 지지선</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with m4:
        atr_val = curr['ATR'] if pd.notnull(curr['ATR']) else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ATR (1H 변동폭)</div>
                <div class="metric-val" style="color: #475569;">${atr_val:,.2f}</div>
                <div class="metric-sub">손절/목표가 산정 지표</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with m5:
        funding_color = "#16A34A" if funding_rate >= 0 else "#DC2626"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">선물 펀딩비</div>
                <div class="metric-val" style="color:{funding_color};">{funding_rate:.4f}%</div>
                <div class="metric-sub">8시간 간격 정산</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ---------------------------------------------------------
    # 5. 차트 시각화
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
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#16A34A',
            decreasing_line_color='#DC2626',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['EMA_20'],
            line=dict(color='#2563EB', width=1.5),
            name='EMA 20',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['EMA_50'],
            line=dict(color='#9333EA', width=1.5),
            name='EMA 50',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['BB_High'],
            line=dict(color='rgba(100,116,139,0.5)', dash='dot'),
            name='BB Upper',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['BB_Low'],
            line=dict(color='rgba(100,116,139,0.5)', dash='dot'),
            name='BB Lower',
        ),
        row=1,
        col=1,
    )

    colors_macd = [
        '#16A34A' if (val is not None and val >= 0) else '#DC2626'
        for val in df['MACD_Diff']
    ]
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['MACD_Diff'],
            marker_color=colors_macd,
            name='Histogram',
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['MACD'],
            line=dict(color='#2563EB', width=1.5),
            name='MACD',
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['MACD_Signal'],
            line=dict(color='#D97706', width=1.5),
            name='Signal',
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color='#334155'),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 6. Gemini AI 질의응답
    # ---------------------------------------------------------
    st.markdown("### 🤖 Gemini AI 선물 전략 어드바이저")

    user_question = st.text_input(
        "질문 입력",
        placeholder="예: 전략 세워줘 / 지금 Short으로 들어갈까? Long으로 들어갈까?",
        label_visibility="collapsed",
    )

    if user_question:
        with st.spinner("Gemini AI가 정량·정성 시장 분석을 진행 중입니다..."):
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)

                system_instruction = f"""
너는 바이낸스 선물 트레이딩 전문 AI 수석 전략가이다.
제시된 실시간 시장 데이터와 질문을 분석하여 완벽히 계산된 트레이딩 보고서를 제공하라.

[실시간 {symbol_input} 선물 시장 종합 데이터]
- 현재가: ${curr['close']:,.2f} USDT (전봉 대비 {price_change:+.2f}%)
- RSI(14): {curr['RSI']:.2f}
- EMA(20): ${curr['EMA_20']:,.2f} | EMA(50): ${curr['EMA_50']:,.2f}
- 볼린저 밴드: 상한선=${curr['BB_High']:,.2f}, 하한선=${curr['BB_Low']:,.2f}
- MACD: Line={curr['MACD']:.4f}, Signal={curr['MACD_Signal']:.4f}, Hist={curr['MACD_Diff']:.4f}
- ATR (1시간 평균 변동폭): ${curr['ATR']:,.2f}
- 선물 펀딩비: {funding_rate:.4f}%

[핵심 전략 지침]
1. 단기 하락 모멘텀을 역으로 활용하는 '평균 회귀(Mean Reversion)' 전략을 기본으로 작성하라.
2. 진입 타점은 볼린저 밴드 하한선 및 ATR 하단 범주를 활용해 3단계 분할 매수로 수치를 직접 계산하여 제시하라.
3. 예상 손익비(Risk/Reward Ratio)가 최소 1:2 이상 나오도록 진입가, 목표가, 손절가를 설정하라.
4. 마크다운 해시태그(###)를 사용하지 말고, 각 번호(1., 2., 3., 4., 5.)로 시작하도록 작성하라.

[필수 답변 구성 요소]
1. 🎯 **포지션 추천**: Long / Short / 관망 중 명확한 판정 및 핵심 유효 전략 명시
2. ⚡ **추천 레버리지**: 변동성을 고려한 최적 레버리지 배율 지정 및 사유
3. 💰 **진입 비중 & 분할 매매 전략**:
   - 총 추천 비중(%)
   - 볼린저 밴드 하한선 및 ATR을 고려한 1차/2차/3차 분할 진입가(예상 평단 포함) 및 익절 목표가
4. 🛑 **손절가 (Stop-Loss)**: ATR 변동성을 감안한 구체적 손절가 및 설정 이유
5. 📊 **기술적 분석 및 고도화 전략 평가**:
   - 평균 회귀 전략의 원리 및 손익비(Risk/Reward Ratio) 성과 분석 상세 서술
"""

                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=user_question,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                )

                raw_text = response.text.strip()
                sections = re.split(r'\n(?=\d+\.\s)', raw_text)

                for section in sections:
                    if not section.strip():
                        continue

                    lines = section.strip().split('\n')
                    title_line = lines[0]
                    body_lines = lines[1:]

                    clean_title = re.sub(
                        r'^(###|\*\*|\d+\.\s*)', '', title_line
                    ).replace('**', '')
                    clean_title = f"{section.strip()[:2]} {clean_title}"

                    body_html = '<br>'.join(body_lines)
                    body_html = re.sub(
                        r'\*\*(.*?)\*\*', r'<b>\1</b>', body_html
                    )

                    st.markdown(
                        f"""
                        <div class="ai-card">
                            <div class="ai-card-title">{clean_title}</div>
                            <div class="ai-card-body">{body_html}</div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
