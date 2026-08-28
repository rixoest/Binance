from concurrent.futures import ThreadPoolExecutor, as_completed
import time
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
    page_title="PRO TRADING AI | 바이낸스 실시간 대시보드",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "⚠️ GEMINI_API_KEY가 설정되지 않았습니다. "
        ".streamlit/secrets.toml 파일에 GEMINI_API_KEY를 추가해주세요."
    )
    st.stop()

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
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 12px;
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
# 2. 종목 리스트 및 세션 상태 초기화
# ---------------------------------------------------------
RECOMMENDED_ASSETS = {
    "1. KORU/USDT (선물 상장 종목)": "KORU/USDT",
    "2. BTC/USDT (디지털 자산의 닻 - 안정적 스윙)": "BTC/USDT",
    "3. ETH/USDT (스마트컨트랙트 코어 - 트렌드 추종)": "ETH/USDT",
    "4. SOL/USDT (고성능 생태계 리더 - 모멘텀 매매)": "SOL/USDT",
    "5. BNB/USDT (거래소 유틸리티 - 박스권 대응)": "BNB/USDT",
    "6. FET/USDT (AI 테마 주도주 - 단기 순환매 스캘핑)": "FET/USDT",
}

if "target_symbol" not in st.session_state:
    st.session_state.target_symbol = "KORU/USDT"
if "auto_strategy_trigger" not in st.session_state:
    st.session_state.auto_strategy_trigger = True
if "previous_symbol" not in st.session_state:
    st.session_state.previous_symbol = st.session_state.target_symbol

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">⚡ PRO TRADING AI DASHBOARD</div>
        <div class="sub-title">멀티팩터·멀티타임프레임 분석 + 리스크 기반 포지션 사이징 + AI 자동 브리핑</div>
    </div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. 데이터 수집 함수 (KORU/USDT 등 선물 전용 종목 순정 로직 유지)[cite: 1]
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def load_market_data(symbol, timeframe="1h", limit=100):
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
    ohlcv = None

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.BadSymbol:
        try:
            exchange_f = ccxt.binance(
                {"enableRateLimit": True, "timeout": 10000, "options": {"defaultType": "future"}}
            )
            ohlcv = exchange_f.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except ccxt.BadSymbol:
            return None, None, f"'{symbol}'은(는) 바이낸스 현물/선물 어디에도 존재하지 않는 심볼입니다."
        except Exception as e:
            return None, None, f"선물 데이터 조회 중 오류: {e}"
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        return None, None, f"바이낸스 API 오류: {e}"
    except Exception as e:
        return None, None, f"알 수 없는 오류: {e}"

    if not ohlcv or len(ohlcv) < 55:
        return None, None, f"'{symbol}'의 캔들 데이터가 부족합니다(최소 55개 필요)."

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
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
    df["ATR_MA50"] = df["ATR"].rolling(window=50, min_periods=20).mean()
    df["Volume_MA20"] = df["volume"].rolling(window=20, min_periods=10).mean()

    funding_rate = None
    try:
        exchange_f = ccxt.binance(
            {"enableRateLimit": True, "timeout": 10000, "options": {"defaultType": "future"}}
        )
        fr = exchange_f.fetch_funding_rate(symbol)["fundingRate"]
        funding_rate = fr * 100 if fr is not None else None
    except Exception:
        funding_rate = None

    return df, funding_rate, None


@st.cache_data(ttl=60)
def get_trend_label(symbol, timeframe):
    df, _, err = load_market_data(symbol, timeframe=timeframe, limit=80)
    if err or df is None:
        return "unknown"
    curr = df.iloc[-1]
    return "up" if curr["close"] > curr["EMA_50"] else "down"


# ---------------------------------------------------------
# 4. 멀티팩터 + 멀티타임프레임 랭킹 함수
# ---------------------------------------------------------
def analyze_single_asset(args):
    label, sym, btc_trend = args
    df_temp, fr_temp, err = load_market_data(sym)
    time.sleep(0.02)

    if err or df_temp is None or len(df_temp) < 2:
        return None, f"{sym}: {err or '데이터 부족'}"

    curr = df_temp.iloc[-1]
    prev = df_temp.iloc[-2]
    price_change = ((curr["close"] - prev["close"]) / prev["close"]) * 100

    bb_width = curr["BB_High"] - curr["BB_Low"]
    bb_position = (curr["close"] - curr["BB_Low"]) / (bb_width if bb_width > 0 else 1.0)
    rr_score = (1.0 - bb_position) * 30.0

    rsi = curr["RSI"]
    if 30 <= rsi <= 45:
        rsi_score, rsi_desc = 30.0, "과매도 탈출 구간"
    elif 55 <= rsi <= 70:
        rsi_score, rsi_desc = 25.0, "상승 모멘텀 구간"
    else:
        rsi_score, rsi_desc = 10.0, "중립 또는 과열권"

    funding_score = 10.0 if fr_temp is None else max(0.0, min(20.0, 10.0 - fr_temp * 200))
    macd_score = 10.0 if curr["MACD_Diff"] >= 0 else 5.0
    trend_1h_up = curr["close"] > curr["EMA_50"]
    trend_score = 15.0 if trend_1h_up else 0.0

    trend_4h = get_trend_label(sym, "4h")
    trend_1h_label = "up" if trend_1h_up else "down"
    mtf_aligned = (trend_4h != "unknown") and (trend_4h == trend_1h_label)
    mtf_score = 15.0 if mtf_aligned else 0.0

    vol_ma = curr["Volume_MA20"]
    vol_ratio = (curr["volume"] / vol_ma) if vol_ma and vol_ma > 0 else 1.0
    volume_score = 10.0 if vol_ratio >= 1.3 else (5.0 if vol_ratio >= 1.0 else 0.0)

    btc_align_score = 10.0 if (sym == "BTC/USDT" or btc_trend == trend_1h_label) else 0.0

    atr_ma = curr["ATR_MA50"]
    atr_ratio = (curr["ATR"] / atr_ma) if atr_ma and atr_ma > 0 else 1.0
    vol_regime_penalty = -15.0 if atr_ratio >= 2.0 else (-7.0 if atr_ratio >= 1.5 else 0.0)

    total_score = (
        rr_score + rsi_score + funding_score + macd_score + trend_score
        + mtf_score + volume_score + btc_align_score + vol_regime_penalty
    )

    result_item = {
        "symbol": sym,
        "label": label,
        "score": total_score,
        "price": curr["close"],
        "change": price_change,
        "rsi": rsi,
        "direction": trend_1h_label,
        "funding": fr_temp,
        "atr": curr["ATR"],
        "atr_ratio": atr_ratio,
        "mtf_aligned": mtf_aligned,
        "reason": f"RSI({rsi:.1f}-{rsi_desc}) | 1H/4H 추세 정렬여부: {mtf_aligned}",
    }
    return result_item, None


def analyze_and_rank_assets(assets_dict):
    rankings = []
    errors = []
    btc_trend = get_trend_label("BTC/USDT", "1h")
    tasks = [(label, sym, btc_trend) for label, sym in assets_dict.items()]
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_asset = {executor.submit(analyze_single_asset, task): task[1] for task in tasks}
        for future in as_completed(future_to_asset):
            res, err = future.result()
            if err:
                errors.append(err)
            elif res:
                rankings.append(res)

    rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings, errors, None


# ---------------------------------------------------------
# 5. 포지션 사이징 계산기 및 Fragment (기본값 수정 반영)
# ---------------------------------------------------------
def calc_position_size(capital, risk_pct, entry_price, atr, atr_multiplier=1.5, leverage=1):
    risk_amount = capital * (risk_pct / 100.0)
    stop_distance = atr * atr_multiplier
    if stop_distance <= 0:
        return None
    qty = risk_amount / stop_distance
    position_value = qty * entry_price
    required_margin = position_value / max(leverage, 1)
    return {
        "risk_amount": risk_amount,
        "stop_distance": stop_distance,
        "qty": qty,
        "position_value": position_value,
        "required_margin": required_margin,
        "stop_loss_long": entry_price - stop_distance,
        "stop_loss_short": entry_price + stop_distance,
    }


@st.fragment
def render_position_sizing_calculator(current_close, current_atr, symbol_name):
    st.markdown("### 🧮 리스크 기반 포지션 사이징 계산기")
    ps_col1, ps_col2, ps_col3, ps_col4 = st.columns(4)
    with ps_col1:
        capital = st.number_input("계좌 자본 (USDT)", min_value=10.0, value=1000.0, step=50.0, key="ps_capital")
    with ps_col2:
        risk_pct = st.slider("이번 거래 리스크 (%)", min_value=0.25, max_value=10.0, value=5.0, step=0.25, key="ps_risk")
    with ps_col3:
        atr_multiplier = st.slider("손절 ATR 배수", min_value=0.5, max_value=3.0, value=1.5, step=0.25, key="ps_atr_mult")
    with ps_col4:
        leverage_input = st.number_input("레버리지 (배)", min_value=1, max_value=50, value=5, step=1, key="ps_lev")

    sizing = calc_position_size(capital, risk_pct, current_close, current_atr, atr_multiplier, leverage_input)

    if sizing:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("리스크 금액", f"${sizing['risk_amount']:,.2f}")
        sc2.metric("추천 수량", f"{sizing['qty']:.4f} {symbol_name.split('/')[0]}")
        sc3.metric("포지션 가치", f"${sizing['position_value']:,.2f}")
        sc4.metric("필요 증거금", f"${sizing['required_margin']:,.2f}")

        st.markdown(
            f"""
            - **롱 기준** 손절가: `${sizing['stop_loss_long']:,.2f}`
            - **숏 기준** 손절가: `${sizing['stop_loss_short']:,.2f}`
            """
        )
    else:
        st.info("ATR 데이터가 유효하지 않습니다.")

    st.session_state.current_sizing = sizing
    st.session_state.current_capital = capital
    st.session_state.current_risk_pct = risk_pct
    st.session_state.current_leverage = leverage_input


# ---------------------------------------------------------
# 6. 제어판 UI (종목 변경 감지 및 자동 트리거 처리)
# ---------------------------------------------------------
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 3, 1])

with ctrl_col1:
    selected_rec = st.selectbox("🔥 AI 추천 종목 선택 (총 6개)", list(RECOMMENDED_ASSETS.keys()))
    new_symbol_from_box = RECOMMENDED_ASSETS[selected_rec]
    if new_symbol_from_box != st.session_state.target_symbol:
        st.session_state.target_symbol = new_symbol_from_box
        st.session_state.auto_strategy_trigger = True

with ctrl_col2:
    symbol_input = st.text_input(
        "조회 종목 심볼 (직접 입력 가능)", value=st.session_state.target_symbol
    ).strip().upper()
    if symbol_input != st.session_state.target_symbol:
        st.session_state.target_symbol = symbol_input
        st.session_state.auto_strategy_trigger = True

with ctrl_col3:
    st.write("")
    if st.button("🔄 캐시 초기화", use_container_width=True):
        st.cache_data.clear()
        st.session_state.auto_strategy_trigger = True
        st.rerun()

if st.button("⚡ 멀티팩터·MTF 랭킹 스캔 및 1위 자동 선택", use_container_width=True):
    with st.spinner("고속 스캔 중..."):
        rankings, scan_errors, _ = analyze_and_rank_assets(RECOMMENDED_ASSETS)
        if rankings:
            st.session_state.target_symbol = rankings[0]["symbol"]
            st.session_state.rankings_cache = rankings
            st.session_state.auto_strategy_trigger = True
            st.success(f"🎯 1위 종목 자동 선정: **{rankings[0]['symbol']}**")
            st.rerun()

# ---------------------------------------------------------
# 7. 메인 데이터 로드 및 대시보드 렌더링
# ---------------------------------------------------------
df, funding_rate, load_error = load_market_data(st.session_state.target_symbol)

if load_error or df is None:
    st.error(f"'{st.session_state.target_symbol}' 데이터를 가져올 수 없습니다. 사유: {load_error}")
else:
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = ((curr["close"] - prev["close"]) / prev["close"]) * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    color_change = "#16A34A" if price_change >= 0 else "#DC2626"

    with m1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">{st.session_state.target_symbol} 현재가</div>
                <div class="metric-val">${curr['close']:,.2f}</div>
                <div style="color:{color_change}; font-size:0.9rem; font-weight:700;">{price_change:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        rsi_color = "#DC2626" if curr["RSI"] >= 70 else ("#16A34A" if curr["RSI"] <= 30 else "#D97706")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">RSI (14)</div>
                <div class="metric-val" style="color:{rsi_color};">{curr['RSI']:.1f}</div>
                <div class="metric-sub">과매수/과매도</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">EMA (20)</div>
                <div class="metric-val" style="color: #2563EB;">${curr['EMA_20']:,.2f}</div>
                <div class="metric-sub">단기 지지선</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">ATR (1H 변동폭)</div>
                <div class="metric-val" style="color: #475569;">${curr['ATR']:,.2f}</div>
                <div class="metric-sub">손절/목표가 산정</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m5:
        funding_color = "#16A34A" if (funding_rate or 0) >= 0 else "#DC2626"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">선물 펀딩비</div>
                <div class="metric-val" style="color:{funding_color};">{(f"{funding_rate:.4f}%" if funding_rate is not None else "N/A")}</div>
                <div class="metric-sub">8시간 정산</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # ---------------------------------------------------------
    # 8. 차트 시각화
    # ---------------------------------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#16A34A", decreasing_line_color="#DC2626",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["EMA_20"], line=dict(color="#2563EB", width=1.5), name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["EMA_50"], line=dict(color="#9333EA", width=1.5), name="EMA 50"), row=1, col=1)

    colors_macd = ["#16A34A" if val >= 0 else "#DC2626" for val in df["MACD_Diff"]]
    fig.add_trace(go.Bar(x=df["datetime"], y=df["MACD_Diff"], marker_color=colors_macd, name="Histogram"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["MACD"], line=dict(color="#2563EB", width=1.5), name="MACD"), row=2, col=1)

    fig.update_layout(
        template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=520,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#334155")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 9. 포지션 사이징 계산기 호출
    # ---------------------------------------------------------
    render_position_sizing_calculator(curr["close"], curr["ATR"], st.session_state.target_symbol)

    st.divider()

    # ---------------------------------------------------------
    # 10. Gemini AI 어드바이저 (지시사항 반영)
    # ---------------------------------------------------------
    st.markdown("### 🤖 Gemini AI 선물 전략 어드바이저")
    
    if st.session_state.auto_strategy_trigger:
        default_prompt = f"선택된 종목인 {st.session_state.target_symbol}에 대한 트레이딩 전략을 분석해줘."
    else:
        default_prompt = ""

    user_question = st.text_input("질문 입력", value=default_prompt, placeholder="종목 분석 및 전략 질문을 입력하세요...")

    if user_question or st.session_state.auto_strategy_trigger:
        prompt_to_run = user_question if user_question else f"선택된 종목인 {st.session_state.target_symbol}에 대한 트레이딩 전략을 분석해줘."
        
        with st.spinner("AI 분석 중..."):
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                trend_4h = get_trend_label(st.session_state.target_symbol, "4h")
                trend_1h = "up" if curr["close"] > curr["EMA_50"] else "down"
                funding_display = f"{funding_rate:.4f}%" if funding_rate is not None else "N/A"

                system_instruction = f"""
너는 전문 트레이딩 AI다. 현재 종목 {st.session_state.target_symbol}의 가격은 ${curr['close']:,.2f}, RSI는 {curr['RSI']:.1f}, 1H 추세는 {trend_1h}, 4H 추세는 {trend_4h}, 펀딩비는 {funding_display}이다.
설정된 리스크 조건: 거래 리스크 5%, 손절 ATR 배수 1.5, 레버리지 5배.
절대 인사말이나 서두 멘트("...분석 리포트입니다" 등)를 출력하지 말고 바로 본론부터 시작하라.
응답 맨 첫 줄에는 반드시 추천 포지션 방향을 [LONG] 또는 [SHORT] 중 하나만 정확히 대문자로 단독 기재하라.
이후 내용을 1., 2., 3. 형태의 번호 항목으로 나누고, 별도의 마크다운 볼드(**) 기호는 절대 사용하지 말고 숫자 기호 등을 활용하여 가독성 높게 작성하라.
"""

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt_to_run,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                )

                raw_text = response.text.strip()
                
                # 리포트 최상단 타이틀 렌더링
                st.markdown(f"""<div class="ai-card-title">💡 종합 전략 리포트 ({st.session_state.target_symbol})</div>""", unsafe_allow_html=True)
                
                # 번호 패턴을 기준으로 텍스트 분할
                import re
                sections = re.split(r'(?=\n\s*(?:\d+[\.\)]))', raw_text)
                
                for section in sections:
                    if section.strip():
                        st.markdown(
                            f"""<div class="ai-card" style="border-left-color: #3B82F6; margin-bottom: 12px;">
                                <div class="ai-card-body">{section.strip().replace(chr(10), '<br>')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Gemini API 호출 오류: {e}")
        
        st.session_state.auto_strategy_trigger = False
