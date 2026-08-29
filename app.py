from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time
import ccxt
import numpy as np
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
    .backtest-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #0EA5E9;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .rank-report-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #10B981;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
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
if "scan_rankings" not in st.session_state:
    st.session_state.scan_rankings = None

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">⚡ PRO TRADING AI DASHBOARD</div>
        <div class="sub-title">멀티팩터·멀티타임프레임 분석 + 리스크 기반 포지션 사이징 + 간이 백테스트 + AI 자동 브리핑</div>
    </div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. 데이터 수집 함수 (KORU/USDT 등 선물 전용 종목 순정 로직)
# ---------------------------------------------------------
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


@st.cache_data(ttl=30)
def load_market_data(symbol, timeframe="1h", limit=120, drop_unclosed=True):
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

    if not ohlcv:
        return None, None, f"'{symbol}'의 캔들 데이터가 없습니다."

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # 거래소가 반환하는 마지막 캔들은 아직 마감되지 않은(진행 중인) 캔들일 수 있다.
    # 이걸 그대로 쓰면 지표(RSI/MACD/볼린저 등)가 시간이 지날수록 계속 바뀌는
    # '리페인팅' 현상이 생겨 스캔 결과의 재현성이 떨어진다. 마감 여부를 확인해서
    # 아직 안 끝난 캔들이면 제거하고 가장 최근 '완성된' 캔들만 사용한다.
    if drop_unclosed and len(df) > 1:
        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
        candle_close_ms = int(df["timestamp"].iloc[-1]) + tf_minutes * 60 * 1000
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if now_ms < candle_close_ms:
            df = df.iloc[:-1].reset_index(drop=True)

    if len(df) < 55:
        return None, None, f"'{symbol}'의 마감된 캔들 데이터가 부족합니다(최소 55개 필요)."

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
# 4. 멀티팩터 + 멀티타임프레임 랭킹 함수 (교차단면 Z-score 기반 상대평가)
# ---------------------------------------------------------
# 설계 원칙(업그레이드 포인트):
#   1) 절대 기준(고정 점수/임계값) 대신, 이번 스캔 유니버스 내에서의 상대적
#      강도를 Z-score로 측정한다. 임의로 정한 "RSI 30~45=30점" 같은 계단식
#      기준을 없애고 연속값으로 비교하므로 정보 손실이 줄어든다.
#   2) 모든 팩터의 방향을 '추세추종(상승 지속에 우호적)' 하나로 통일했다.
#      기존 코드는 볼린저 하단 근접(역추세/저점매수)과 EMA 상단 이탈(추세추종)을
#      동시에 가점 요인으로 섞어써서, 하락추세에서 저점까지 밀린 종목이 오히려
#      1위로 뽑힐 수 있는 모순이 있었다. 지금은 밴드 위치를 '과열/과매도로 인한
#      되돌림 리스크'로만 사용해 트렌드 로직과 충돌하지 않게 했다.
#   3) 가중치(FACTOR_WEIGHTS)는 여전히 사람이 정한 값이라는 한계가 있다.
#      다만 핵심 추세/모멘텀 신호와 보조/리스크 신호를 명시적으로 구분해두었으니,
#      추후 과거 수익률 데이터로 팩터별 IC(정보계수)를 계산해 이 값을 교체하면
#      실제로 검증된 가중치로 업그레이드할 수 있다.
#   4) 점수는 0~100 스케일로 표시되지만, 이는 "매매 신뢰도의 절대치"가 아니라
#      '같은 시점·같은 6종목 유니버스 안에서의 상대 순위'라는 점을 화면에도 명시한다.

FACTOR_WEIGHTS = {
    # --- 핵심 신호: 추세/모멘텀 (동일 철학, 서로 보완) ---
    "trend_strength": 1.0,      # EMA50 대비 이격(ATR 정규화) - 추세 크기
    "mtf_alignment": 1.0,       # 1H/4H 추세 방향 일치 여부
    "momentum": 1.0,            # MACD 히스토그램(ATR 정규화) - 모멘텀 크기
    "rsi_momentum": 0.7,        # RSI-50, 추세 방향과의 정합성
    # --- 리스크/과열 보정 (추세 신호를 깎는 방향으로만 작동, 반대 전략 아님) ---
    "rsi_extreme_penalty": 0.5,   # RSI 극단(과매수/과매도) 되돌림 리스크
    "band_overextension": 0.5,    # 볼린저 밴드 끝단 근접(과확장) 리스크
    "vol_regime_penalty": 0.5,    # 평소 대비 변동성 급등 리스크
    # --- 보조/맥락 신호 ---
    "funding_signal": 0.6,      # 펀딩비 역이용(포지션 쏠림 반대 매매 유인)
    "volume_confirmation": 0.4, # 거래량 동반 여부
    "btc_alignment": 0.3,       # BTC와의 방향 동조화
}

FACTOR_LABELS = {
    "trend_strength": "추세 강도(EMA50 이격)",
    "mtf_alignment": "1H/4H 추세 정렬",
    "momentum": "MACD 모멘텀",
    "rsi_momentum": "RSI 방향성",
    "rsi_extreme_penalty": "RSI 과열·과매도 리스크",
    "band_overextension": "볼린저 밴드 과확장 리스크",
    "funding_signal": "펀딩비(포지션 쏠림)",
    "volume_confirmation": "거래량 확인",
    "btc_alignment": "BTC 동조화",
    "vol_regime_penalty": "변동성 급등 리스크",
}


def compute_raw_factors(args):
    """종목 1개의 '원시 팩터'만 계산한다. 유니버스 전체를 모아야 계산 가능한
    상대평가(Z-score)는 analyze_and_rank_assets에서 한 번에 수행한다."""
    label, sym, btc_trend = args
    df_temp, fr_temp, err = load_market_data(sym)
    time.sleep(0.02)

    if err or df_temp is None or len(df_temp) < 2:
        return None, f"{sym}: {err or '마감된 캔들 데이터 부족'}"

    curr = df_temp.iloc[-1]
    prev = df_temp.iloc[-2]
    price_change = ((curr["close"] - prev["close"]) / prev["close"]) * 100

    atr = curr["ATR"]
    if atr is None or pd.isna(atr) or atr <= 0:
        return None, f"{sym}: ATR 계산 불가(데이터 부족)"

    # 1) 추세 강도: EMA50 대비 이격을 ATR 단위로 정규화한 연속값
    trend_strength = (curr["close"] - curr["EMA_50"]) / atr
    trend_1h_up = curr["close"] > curr["EMA_50"]
    trend_1h_label = "up" if trend_1h_up else "down"

    # 2) 상위 타임프레임(4H) 정렬: +1(일치) / -1(불일치) / 0(4H 판단 불가)
    trend_4h = get_trend_label(sym, "4h")
    mtf_aligned = (trend_4h != "unknown") and (trend_4h == trend_1h_label)
    if trend_4h == "unknown":
        mtf_alignment = 0.0
    else:
        mtf_alignment = 1.0 if mtf_aligned else -1.0

    # 3) 모멘텀: MACD 히스토그램을 ATR로 정규화한 연속값
    momentum = curr["MACD_Diff"] / atr

    # 4) RSI 방향성: 50을 기준으로 추세와 같은 방향인지(연속값)
    rsi = curr["RSI"]
    rsi_momentum = rsi - 50.0
    # 극단(과매수 75+ / 과매도 25-)은 추세 지속 신뢰도가 떨어지는 리스크로 별도 취급
    rsi_extreme_penalty = -abs(rsi - 50.0) if (rsi >= 75 or rsi <= 25) else 0.0

    # 5) 볼린저 밴드: '저점 근접=매수'가 아니라 '밴드 끝단 근접=과확장 리스크'로 재정의
    #    (기존처럼 추세추종 팩터와 정반대 방향으로 작동하지 않도록 통일)
    bb_width = curr["BB_High"] - curr["BB_Low"]
    bb_position = (curr["close"] - curr["BB_Low"]) / (bb_width if bb_width > 0 else 1.0)
    band_overextension = -abs(bb_position - 0.5)

    # 6) 펀딩비: 부호만 반전(양(+)의 펀딩비=롱 과열=감점, 음(-)=숏 과열=가점).
    #    절대 배율(*200) 없이 원시값을 유니버스 Z-score로 정규화해 과민반응을 없앤다.
    funding_signal = -fr_temp if fr_temp is not None else 0.0

    # 7) 거래량 확인: 20기간 평균 대비 배율(연속값)
    vol_ma = curr["Volume_MA20"]
    vol_ratio = (curr["volume"] / vol_ma) if vol_ma and vol_ma > 0 else 1.0

    # 8) BTC 동조화: BTC 자신은 0(자기상관 방지), 그 외엔 방향 일치 +1 / 불일치 -1
    btc_alignment = 0.0 if sym == "BTC/USDT" else (1.0 if btc_trend == trend_1h_label else -1.0)

    # 9) 변동성 레짐: 평소(50기간 평균) 대비 변동성이 커질수록 감점되도록 부호 반전
    atr_ma = curr["ATR_MA50"]
    atr_ratio = (curr["ATR"] / atr_ma) if atr_ma and atr_ma > 0 else 1.0
    vol_regime_penalty = -atr_ratio

    raw_item = {
        "symbol": sym,
        "label": label,
        "price": curr["close"],
        "change": price_change,
        "rsi": rsi,
        "direction": trend_1h_label,
        "funding": fr_temp,
        "atr": atr,
        "atr_ratio": atr_ratio,
        "mtf_aligned": mtf_aligned,
        "bb_position": bb_position,
        "vol_ratio": vol_ratio,
        "macd_diff": curr["MACD_Diff"],
        "factors": {
            "trend_strength": trend_strength,
            "mtf_alignment": mtf_alignment,
            "momentum": momentum,
            "rsi_momentum": rsi_momentum,
            "rsi_extreme_penalty": rsi_extreme_penalty,
            "band_overextension": band_overextension,
            "funding_signal": funding_signal,
            "volume_confirmation": vol_ratio,
            "btc_alignment": btc_alignment,
            "vol_regime_penalty": vol_regime_penalty,
        },
    }
    return raw_item, None


def _zscore(values):
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _build_reason(item):
    # 실제 총점에 기여한 요인만, 기여도 절댓값 순으로 최대 4개까지 투명하게 노출한다.
    # (기존 코드는 총점 계산에 쓰인 펀딩비/BTC동조화/변동성 페널티가 화면 설명에는
    #  전혀 나타나지 않는 불일치가 있었다.)
    ranked = sorted(item["contrib"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    detail_lines = []
    for name, val in ranked:
        tag = "긍정적 기여" if val > 0.05 else ("부정적 기여" if val < -0.05 else "중립적")
        detail_lines.append(f"• {FACTOR_LABELS.get(name, name)}: {tag} ({val:+.2f})")

    header = (
        f"• RSI {item['rsi']:.1f} · MTF {'정렬' if item['mtf_aligned'] else '미정렬'} "
        f"· 거래량 {item['vol_ratio']:.1f}배 · 변동성비 {item['atr_ratio']:.1f}<br>"
    )
    return header + "<br>".join(detail_lines)


def analyze_and_rank_assets(assets_dict):
    errors = []
    btc_trend = get_trend_label("BTC/USDT", "1h")
    tasks = [(label, sym, btc_trend) for label, sym in assets_dict.items()]

    raw_list = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_asset = {executor.submit(compute_raw_factors, task): task[1] for task in tasks}
        for future in as_completed(future_to_asset):
            res, err = future.result()
            if err:
                errors.append(err)
            elif res:
                raw_list.append(res)

    if not raw_list:
        return [], errors, None

    # 유니버스 전체를 대상으로 팩터별 Z-score 정규화(교차단면 정규화).
    # 종목이 1개뿐이면 표준편차가 0이라 상대비교가 불가능하므로 그 경우엔
    # 모든 팩터가 0점(중립) 처리된다.
    factor_names = list(FACTOR_WEIGHTS.keys())
    zscores = {name: _zscore([item["factors"][name] for item in raw_list]) for name in factor_names}

    total_weight = sum(FACTOR_WEIGHTS.values())
    composite_values = []
    for i, item in enumerate(raw_list):
        contrib = {name: float(zscores[name][i]) * FACTOR_WEIGHTS[name] for name in factor_names}
        composite_z = sum(contrib.values()) / total_weight
        item["contrib"] = contrib
        item["composite_z"] = composite_z
        composite_values.append(composite_z)

    # 해석 편의를 위해 0~100 스케일로 변환한다. 단, 이는 절대적인 매매 신뢰도가
    # 아니라 '이번 스캔·이번 유니버스 내 상대 순위'라는 점에 유의해야 한다.
    lo, hi = min(composite_values), max(composite_values)
    span = (hi - lo) if (hi - lo) > 1e-9 else 1.0
    for item in raw_list:
        item["score"] = (item["composite_z"] - lo) / span * 100.0
        item["reason"] = _build_reason(item)

    raw_list.sort(key=lambda x: x["score"], reverse=True)
    return raw_list, errors, None


# ---------------------------------------------------------
# 5. 포지션 사이징 계산기 및 간이 백테스트 로직
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


def run_simple_backtest(df):
    if df is None or len(df) < 20:
        return 0.0, 0.0
    working_df = df.copy()
    working_df["signal"] = 0
    working_df.loc[working_df["EMA_20"] > working_df["EMA_50"], "signal"] = 1
    working_df.loc[working_df["EMA_20"] < working_df["EMA_50"], "signal"] = -1

    working_df["market_return"] = working_df["close"].pct_change()
    working_df["strategy_return"] = working_df["signal"].shift(1) * working_df["market_return"]
    
    cum_market = (1.0 + working_df["market_return"].fillna(0)).cumprod().iloc[-1] - 1.0
    cum_strategy = (1.0 + working_df["strategy_return"].fillna(0)).cumprod().iloc[-1] - 1.0
    
    return cum_market * 100, cum_strategy * 100


# ---------------------------------------------------------
# 6. 제어판 UI
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
        st.session_state.scan_rankings = None
        st.rerun()

if st.button("⚡ 멀티팩터·MTF 랭킹 스캔 및 1위 자동 선택", use_container_width=True):
    with st.spinner("고속 스캔 중..."):
        rankings, scan_errors, _ = analyze_and_rank_assets(RECOMMENDED_ASSETS)
        if scan_errors:
            # 기존 코드는 실패한 종목을 조용히 스캔 결과에서 제외해 사용자가
            # "6개 전체가 정상 비교됐다"고 오해할 수 있었다. 실패 종목을 명시한다.
            st.warning("⚠️ 다음 종목은 스캔에서 제외되었습니다: " + " / ".join(scan_errors))
        if rankings:
            st.session_state.scan_rankings = rankings
            st.session_state.target_symbol = rankings[0]["symbol"]
            st.session_state.auto_strategy_trigger = True
            st.success(f"🎯 1위 종목 자동 선정: **{rankings[0]['symbol']}** (스캔된 {len(rankings)}개 종목 중 상대 1위)")
            st.rerun()
        else:
            st.error("스캔 가능한 종목이 없습니다. 잠시 후 다시 시도해주세요.")

# ---------------------------------------------------------
# 6.1. 랭킹 결과 카드 박스 출력 영역 (들여쓰기 오류 수정 완료)
# ---------------------------------------------------------
if st.session_state.scan_rankings:
    import textwrap
    
    rank_html = textwrap.dedent(f"""
    <div class="rank-report-box">
        <div style="font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
            📊 실시간 멀티팩터 전 종목 스캔 랭킹 리포트
        </div>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 12px;">
            ※ 점수는 절대적인 매매 신뢰도가 아니라, 이번 스캔에 포함된 {len(st.session_state.scan_rankings)}개 종목
            사이의 상대적 강도(0~100, Z-score 기반 상대순위)입니다.
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
            <tr style="border-bottom: 2px solid #CBD5E1; color: #475569; text-align: left;">
                <th style="padding: 6px;">순위</th>
                <th style="padding: 6px;">종목명</th>
                <th style="padding: 6px;">상대 강도</th>
                <th style="padding: 6px;">현재가 / 등락</th>
                <th style="padding: 6px;">상세 상태 분석 및 선정 이유</th>
            </tr>
    """)
    
    for idx, item in enumerate(st.session_state.scan_rankings, 1):
        c_color = "#16A34A" if item["change"] >= 0 else "#DC2626"
        rank_html += textwrap.dedent(f"""\
        <tr style="border-bottom: 1px solid #E2E8F0; color: #1E293B; vertical-align: top;">
        <td style="padding: 8px; font-weight: 700;">{idx}위</td>
        <td style="padding: 8px; font-weight: 800;">{item['symbol']}</td>
        <td style="padding: 8px; font-weight: 800; color: #2563EB;">{item['score']:.1f}점</td>
        <td style="padding: 8px;">${item['price']:,.2f} <span style="color: {c_color}; font-weight: 700;">({item['change']:+.2f}%)</span></td>
        <td style="padding: 8px; color: #475569; font-size: 0.85rem; line-height: 1.5;">{item['reason']}</td>
        </tr>
        """)
    rank_html += "</table></div>"
    st.markdown(rank_html, unsafe_allow_html=True)

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
    # 8.1. 간이 백테스트 (성과 및 매매 인사이트 자동 도출)
    # ---------------------------------------------------------
    m_ret, s_ret = run_simple_backtest(df)
    
    if s_ret > m_ret and s_ret > 0:
        insight_text = "💡 <b>인사이트:</b> 이평선 추세 추종 전략이 시장 수익률을 압도했습니다. <b>방향성이 깔끔한 구간이므로 AI의 3단계 분할 진입 전략을 적극 신뢰할 만합니다.</b>"
    elif s_ret < m_ret and m_ret > 0:
        insight_text = "💡 <b>인사이트:</b> 코인 가격은 올랐으나 잔파동으로 인해 이평선 효율이 떨어졌습니다. <b>횡보/속임수 장세일 수 있으니 분할 진입 타점을 보수적으로 잡으세요.</b>"
    elif s_ret > m_ret and s_ret > 0 and m_ret <= 0:
        insight_text = "💡 <b>인사이트:</b> 하락 또는 횡보장 속에서도 시스템이 추세를 잘 방어했습니다. <b>숏 포지션이나 변동성 대응에 유리한 종목입니다.</b>"
    else:
        insight_text = "💡 <b>인사이트:</b> 전략 성과와 시장 수익률 모두 부진합니다. <b>차트 호흡이 더럽고 예측이 어려우므로 매매 비중을 낮추거나 관망을 권장합니다.</b>"

    st.markdown(
        f"""
        <div class="backtest-box">
            📊 <b>간이 백테스트 (참고용 EMA 20/50 크로스오버 성과 컴포넌트)</b><br>
            <span style="font-size: 0.92rem; color: #475569;">
            - 시장(Buy & Hold) 누적 수익률: <b style="color: {'#16A34A' if m_ret>=0 else '#DC2626'}">{m_ret:+.2f}%</b><br>
            - EMA 크로스오버 전략 누적 성과: <b style="color: {'#16A34A' if s_ret>=0 else '#DC2626'}">{s_ret:+.2f}%</b><br>
            <hr style="margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;">
            {insight_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # 9. 포지션 사이징 계산기 호출
    # ---------------------------------------------------------
    render_position_sizing_calculator(curr["close"], curr["ATR"], st.session_state.target_symbol)

    st.divider()

    # ---------------------------------------------------------
    # 10. Gemini AI 어드바이저 (5개 항목 + 3단계 분할 진입 강제)
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

아래의 명확한 5가지 핵심 항목으로 나누어 번호를 매겨 작성하라:
1. 시장 국면 및 변동성 체질 진단
2. 멀티타임프레임 추세 정렬 상태
3. 구체적인 진입 타점 및 [LONG] 또는 [SHORT] 방향 (무조건 1차, 2차, 3차 분할 진입 타점과 각 비중을 상세히 포함할 것)
4. 리스크 관리 및 확정 손절가
5. 기대 수익 및 목표가 셋업

별도의 마크다운 볼드(**) 기호는 제목에 절대 사용하지 말고 숫자 기호 등을 활용하여 가독성 높게 작성하라.
"""

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt_to_run,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                )

                raw_text = response.text.strip()
                
                st.markdown(f"""<div class="ai-card-title">💡 종합 전략 리포트 ({st.session_state.target_symbol})</div>""", unsafe_allow_html=True)
                
                import re
                sections = re.split(r'(?=\n\s*(?:\d+[\.\)]))', raw_text)
                
                for section in sections:
                    sec_text = section.strip()
                    if not sec_text:
                        continue
                    
                    lines = sec_text.split('\n', 1)
                    first_line = lines[0].strip()
                    body_text = lines[1].strip() if len(lines) > 1 else ""
                    body_html = body_text.replace(chr(10), '<br>')
                    
                    if re.match(r'^\d+[\.\)]', first_line):
                        st.markdown(
                            f"""<div class="ai-card" style="border-left-color: #3B82F6; margin-bottom: 12px;">
                                <div style="font-size: 1.18rem; font-weight: 800; color: #0F172A; margin-bottom: 8px; letter-spacing: -0.3px;">{first_line}</div>
                                {f'<div style="font-size: 0.98rem; color: #334155; line-height: 1.7;">{body_html}</div>' if body_html else ''}
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""<div class="ai-card" style="border-left-color: #94A3B8; margin-bottom: 12px;">
                                <div style="font-size: 0.98rem; color: #334155; line-height: 1.7;">{sec_text.replace(chr(10), '<br>')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Gemini API 호출 오류: {e}")
        
        st.session_state.auto_strategy_trigger = False
