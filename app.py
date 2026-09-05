from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import re
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


@st.cache_resource
def get_exchange(future=False):
    # ccxt 거래소 객체를 매 호출마다 새로 만들면, 심볼 조회 시 내부적으로
    # 수행되는 전체 마켓 목록 로딩(load_markets)이 캐싱되지 못하고 매번
    # 반복돼 API 응답이 크게 느려진다. st.cache_resource로 세션 동안
    # 인스턴스를 재사용해서 마켓 목록 로딩을 최초 1회로 줄인다.
    if future:
        return ccxt.binance({"enableRateLimit": True, "timeout": 10000, "options": {"defaultType": "future"}})
    return ccxt.binance({"enableRateLimit": True, "timeout": 10000})


@st.cache_data(ttl=30)
def load_market_data(symbol, timeframe="1h", limit=120, drop_unclosed=True, fetch_funding=True):
    exchange = get_exchange(future=False)
    ohlcv = None

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.BadSymbol:
        try:
            exchange_f = get_exchange(future=True)
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
    # 4H/1D 추세만 필요한 호출(get_trend_label)에서는 펀딩비가 전혀 쓰이지
    # 않는데도 기존 코드는 타임프레임을 부를 때마다 매번 별도 REST 호출로
    # 펀딩비를 가져오고 있었다. fetch_funding=False로 넘기면 이 불필요한
    # 네트워크 호출을 건너뛴다(실제 사용하는 1H 메인 호출만 True로 유지).
    if fetch_funding:
        try:
            exchange_f = get_exchange(future=True)
            fr = exchange_f.fetch_funding_rate(symbol)["fundingRate"]
            funding_rate = fr * 100 if fr is not None else None
        except Exception:
            funding_rate = None

    return df, funding_rate, None


@st.cache_data(ttl=60)
def get_trend_label(symbol, timeframe):
    df, _, err = load_market_data(symbol, timeframe=timeframe, limit=80, fetch_funding=False)
    if err or df is None:
        return "unknown"
    curr = df.iloc[-1]
    return "up" if curr["close"] > curr["EMA_50"] else "down"


# ---------------------------------------------------------
# 4. 멀티팩터 + 멀티타임프레임 랭킹 함수 (교차단면 Z-score + 방향 대칭 설계)
# ---------------------------------------------------------
# 설계 원칙 (핵심 목표: "차트가 오르는지"가 아니라 "어느 방향에 배팅해야
# 기대수익이 가장 큰지, 그걸 얼마나 오래 들고 가야 하는지"를 찾는 것):
#
#   1) 방향성 팩터는 전부 부호(+상승/-하락)를 갖도록 설계했다. 이전 버전은
#      "1H/4H 정렬", "BTC 동조", "거래량 동반"을 방향과 무관하게 무조건
#      가점 처리해서, 하락추세가 아주 뚜렷하게 정렬돼도 오히려 점수가
#      상승(LONG) 쪽으로 밀리는 버그가 있었다. 이제는 정렬된 방향이 하락이면
#      음수로, 상승이면 양수로 기여해서 롱/숏이 대칭적으로 나온다.
#   2) RSI 극단/밴드 과확장/변동성 급등은 "감점"이 아니라 "확신도 감쇠
#      (dampening)"로 재설계했다. 예전처럼 그냥 빼버리면, 이미 하락 쪽으로
#      기운 종목에서 극단 지표가 나올 때 오히려 더 큰 음수가 되어 "더 강한
#      숏 신호"로 왜곡됐다. 리스크는 방향을 바꾸는 게 아니라 확신의 크기만
#      줄여야 하므로, 최종 점수에 곱하는 감쇠 계수로 분리했다.
#   3) 1위~N위는 "가장 강하게 오르는 종목"이 아니라 "방향에 상관없이 가장
#      확신도가 높은 종목"이다. LONG 후보와 SHORT 후보가 같은 기준으로
#      경쟁하며, 방향은 부호로, 강도는 절댓값으로 나타난다.
#   4) 1H/4H에 더해 1D(일봉) 추세까지 확인해서, 상위 타임프레임까지 정렬된
#      정도로 권장 보유기간(스캘핑/데이트레이딩/스윙)을 자동 추정한다.
#
# 여전히 남는 한계: 가중치는 사람이 정한 값이고, 이 스코어카드 자체가
# "과거 수익률로 검증된 우위(edge)"를 보장하지 않는다. 실전 자금 투입 전에는
# 반드시 과거 데이터로 위 로직의 성과를 검증(백테스트/페이퍼 트레이딩)해야 한다.

DIRECTIONAL_WEIGHTS = {
    # --- 핵심 추세/모멘텀 (부호 = 방향, 크기 = 강도) ---
    "trend_strength": 1.0,      # EMA50 대비 이격(ATR 정규화)
    "momentum": 1.0,            # MACD 히스토그램(ATR 정규화)
    "rsi_momentum": 0.7,        # RSI-50
    # --- 멀티타임프레임 정렬 (정렬된 '방향'으로 부호 부여) ---
    "mtf_alignment": 1.0,       # 1H/4H 정렬
    "daily_alignment": 0.8,     # 1H/1D 정렬
    "btc_alignment": 0.4,       # BTC 동조(같은 방향일 때만 부호 부여)
    # --- 보조 신호 ---
    "funding_signal": 0.6,      # 펀딩비 역이용(포지션 쏠림 반대 매매 유인)
    "volume_confirmation": 0.4, # 거래량 동반(현재 추세 '방향'으로 부호 부여)
}

RISK_WEIGHTS = {
    # 방향을 바꾸지 않고, 최종 확신도(|z|)를 줄이는 감쇠 요인으로만 사용
    "rsi_extreme_penalty": 0.5,   # RSI 극단(과매수/과매도) 되돌림 리스크
    "band_overextension": 0.5,    # 볼린저 밴드 끝단 근접(과확장) 리스크
    "vol_regime_penalty": 0.5,    # 평소 대비 변동성 급등 리스크
}

RISK_DAMPEN_COEF = 0.5   # 리스크 강도 1단위(z 기준)당 확신도를 얼마나 깎을지
NEUTRAL_Z_THRESHOLD = 0.15  # 이 값보다 |최종 z|가 작으면 "중립/관망"으로 표시

FACTOR_LABELS = {
    "trend_strength": "추세 강도(EMA50 이격)",
    "momentum": "MACD 모멘텀",
    "rsi_momentum": "RSI 방향성",
    "mtf_alignment": "1H/4H 추세 정렬",
    "daily_alignment": "1H/1D 추세 정렬",
    "btc_alignment": "BTC 동조화",
    "funding_signal": "펀딩비(포지션 쏠림)",
    "volume_confirmation": "거래량 확인",
    "rsi_extreme_penalty": "RSI 과열·과매도 리스크",
    "band_overextension": "볼린저 밴드 과확장 리스크",
    "vol_regime_penalty": "변동성 급등 리스크",
}

HORIZON_LABELS = {
    3: "스윙 (수일~1~2주)",
    2: "데이트레이딩~단기 스윙 (수 시간~1~2일)",
    1: "단기 스캘핑 (수 시간 이내)",
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

    # 1) 추세 강도: EMA50 대비 이격을 ATR 단위로 정규화한 연속값 (부호=방향)
    trend_strength = (curr["close"] - curr["EMA_50"]) / atr
    trend_1h_label = "up" if curr["close"] > curr["EMA_50"] else "down"
    dir_sign = 1.0 if trend_1h_label == "up" else -1.0

    # 2) 상위 타임프레임 정렬: 정렬된 '방향'으로 부호를 부여한다(핵심 수정).
    #    예) 1H/4H가 둘 다 하락으로 정렬 -> -1 (하락 확신 강화, 롱 쪽으로 왜곡되지 않음)
    trend_4h = get_trend_label(sym, "4h")
    mtf_aligned = (trend_4h != "unknown") and (trend_4h == trend_1h_label)
    mtf_alignment = dir_sign if mtf_aligned else 0.0

    trend_1d = get_trend_label(sym, "1d")
    daily_aligned = (trend_1d != "unknown") and (trend_1d == trend_1h_label)
    daily_alignment = dir_sign if daily_aligned else 0.0

    # 3) 모멘텀: MACD 히스토그램을 ATR로 정규화한 연속값(부호=방향)
    momentum = curr["MACD_Diff"] / atr

    # 4) RSI 방향성: 50 기준 부호 있는 값 + 극단(과열/과매도) 리스크는 별도 분리
    rsi = curr["RSI"]
    rsi_momentum = rsi - 50.0
    rsi_extreme_penalty = -abs(rsi - 50.0) if (rsi >= 75 or rsi <= 25) else 0.0

    # 5) 볼린저 밴드 과확장 리스크(방향 무관, 감쇠 전용)
    bb_width = curr["BB_High"] - curr["BB_Low"]
    bb_position = (curr["close"] - curr["BB_Low"]) / (bb_width if bb_width > 0 else 1.0)
    band_overextension = -abs(bb_position - 0.5)

    # 6) 펀딩비: 부호만 반전(양(+) 펀딩비=롱 과열=감점, 음(-)=숏 과열=가점)
    funding_signal = -fr_temp if fr_temp is not None else 0.0

    # 7) 거래량 확인: 평소 대비 배율에 '현재 추세 방향'의 부호를 부여(핵심 수정).
    #    하락추세에서 거래량이 실리는 것도 그 방향의 확신을 높이는 신호여야
    #    하는데, 예전엔 방향과 무관하게 항상 가점이라 롱 쪽으로 왜곡됐었다.
    vol_ma = curr["Volume_MA20"]
    vol_ratio = (curr["volume"] / vol_ma) if vol_ma and vol_ma > 0 else 1.0
    volume_confirmation = vol_ratio * dir_sign

    # 8) BTC 동조화: 같은 방향일 때만 그 방향 부호를 부여, 엇갈리면 0(중립)
    if sym == "BTC/USDT":
        btc_alignment = 0.0
    elif trend_1h_label == btc_trend:
        btc_alignment = dir_sign
    else:
        btc_alignment = 0.0

    # 9) 변동성 레짐 리스크(방향 무관, 감쇠 전용)
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
        "daily_aligned": daily_aligned,
        "trend_4h": trend_4h,
        "trend_1d": trend_1d,
        "bb_position": bb_position,
        "vol_ratio": vol_ratio,
        "macd_diff": curr["MACD_Diff"],
        "factors": {
            "trend_strength": trend_strength,
            "momentum": momentum,
            "rsi_momentum": rsi_momentum,
            "mtf_alignment": mtf_alignment,
            "daily_alignment": daily_alignment,
            "btc_alignment": btc_alignment,
            "funding_signal": funding_signal,
            "volume_confirmation": volume_confirmation,
            "rsi_extreme_penalty": rsi_extreme_penalty,
            "band_overextension": band_overextension,
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


def _holding_period_label(item):
    confirmed = 1 + (1 if item["mtf_aligned"] else 0) + (1 if item["daily_aligned"] else 0)
    return HORIZON_LABELS[confirmed], confirmed


def _build_reason(item):
    # 실제 최종 점수에 기여한 방향성 요인만, 기여도 절댓값 순으로 최대 4개 노출.
    ranked = sorted(item["dir_contrib"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    detail_lines = []
    for name, val in ranked:
        tag = "상승(LONG) 쪽 기여" if val > 0.05 else ("하락(SHORT) 쪽 기여" if val < -0.05 else "중립적")
        detail_lines.append(f"• {FACTOR_LABELS.get(name, name)}: {tag} ({val:+.2f})")

    risk_note = ""
    if item["risk_intensity"] > 0.3:
        pct_cut = (1 - 1 / (1 + item["risk_intensity"] * RISK_DAMPEN_COEF)) * 100
        risk_note = f"<br>• ⚠️ 과열/변동성 리스크로 확신도 약 {pct_cut:.0f}% 감쇠 적용됨"

    header = (
        f"• 방향: <b>{item['final_direction']}</b> (권장 보유기간: {item['horizon']})<br>"
        f"• RSI {item['rsi']:.1f} · MTF(1H/4H) {'정렬' if item['mtf_aligned'] else '미정렬'} "
        f"· 일봉 정렬 {'예' if item['daily_aligned'] else '아니오'} · 거래량 {item['vol_ratio']:.1f}배<br>"
    )
    return header + "<br>".join(detail_lines) + risk_note


@st.cache_data(ttl=30)
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

    # 유니버스 전체를 대상으로 팩터별 Z-score 정규화(교차단면 정규화)
    all_factor_names = list(DIRECTIONAL_WEIGHTS.keys()) + list(RISK_WEIGHTS.keys())
    zscores = {name: _zscore([item["factors"][name] for item in raw_list]) for name in all_factor_names}

    total_dir_weight = sum(DIRECTIONAL_WEIGHTS.values())
    total_risk_weight = sum(RISK_WEIGHTS.values())

    for i, item in enumerate(raw_list):
        dir_contrib = {name: float(zscores[name][i]) * w for name, w in DIRECTIONAL_WEIGHTS.items()}
        raw_direction_z = sum(dir_contrib.values()) / total_dir_weight

        risk_contrib = {name: float(zscores[name][i]) * w for name, w in RISK_WEIGHTS.items()}
        risk_raw_avg = sum(risk_contrib.values()) / total_risk_weight  # 리스크 클수록 더 음수
        risk_intensity = max(0.0, -risk_raw_avg)  # 항상 0 이상, 클수록 위험

        dampen_factor = 1.0 / (1.0 + risk_intensity * RISK_DAMPEN_COEF)
        final_z = raw_direction_z * dampen_factor

        horizon, confirmed_tf = _holding_period_label(item)

        item["dir_contrib"] = dir_contrib
        item["raw_direction_z"] = raw_direction_z
        item["risk_intensity"] = risk_intensity
        item["final_z"] = final_z
        item["horizon"] = horizon
        item["confirmed_tf_count"] = confirmed_tf
        item["final_direction"] = (
            "LONG" if final_z > NEUTRAL_Z_THRESHOLD
            else ("SHORT" if final_z < -NEUTRAL_Z_THRESHOLD else "중립/관망")
        )

    # 방향(부호)과 무관하게 '확신도의 크기(|z|)'로 순위를 매긴다.
    # -> "가장 강하게 오르는 종목"이 아니라 "롱이든 숏이든 가장 확신이 큰 종목"이 1위.
    abs_values = [abs(item["final_z"]) for item in raw_list]
    lo, hi = min(abs_values), max(abs_values)
    span = (hi - lo) if (hi - lo) > 1e-9 else 1.0
    for item in raw_list:
        item["score"] = (abs(item["final_z"]) - lo) / span * 100.0
        item["reason"] = _build_reason(item)

    raw_list.sort(key=lambda x: x["score"], reverse=True)
    return raw_list, errors, None


def get_quant_signal_for_symbol(symbol):
    """랭킹 스캔과 동일한 로직(교차단면 Z-score)으로 특정 심볼 하나의 방향성
    신호를 계산한다. AI 어드바이저가 항상 이 로직과 일치된 신호를 참고하도록
    보장하기 위해, 6개 추천 종목 유니버스에 대상 심볼을 포함시켜 재계산한다."""
    universe = dict(RECOMMENDED_ASSETS)
    if symbol not in universe.values():
        universe[f"custom. {symbol}"] = symbol
    rankings, errors, _ = analyze_and_rank_assets(universe)
    for item in rankings:
        if item["symbol"] == symbol:
            return item, errors
    return None, errors


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


def suggested_risk_pct(abs_final_z):
    """확신도(|final_z|)가 클수록 베팅 비중을 키우는 게 기대값(EV) 관점에서
    유리하다는 켈리 기준의 단순화된 근사치. 실제 최적 비율이 아니라 방향성
    참고값이며, 절대 이 값 그대로 실전에 쓰라는 뜻이 아니다."""
    return float(np.clip(1.5 + 2.5 * min(abs_final_z, 3.0), 1.0, 9.0))


@st.fragment
def render_position_sizing_calculator(current_close, current_atr, symbol_name, quant_item=None):
    st.markdown("### 🧮 리스크 기반 포지션 사이징 계산기")

    if quant_item:
        rec_risk = suggested_risk_pct(abs(quant_item["final_z"]))
        st.caption(
            f"💡 정량 신호 기준: **{quant_item['final_direction']}**, 확신도(z)={quant_item['final_z']:+.2f} "
            f"→ 참고용 권장 리스크 비중 약 **{rec_risk:.1f}%** (확신도가 낮을수록 비중을 줄이고, "
            f"높을수록 늘리는 게 장기 기대값에 유리하다는 원칙에 따른 참고치입니다. 슬라이더 기본값과 별개입니다.)"
        )

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


def render_ai_report_cards(raw_text, symbol):
    """Gemini 응답을 번호(0~5) 섹션별 색상 카드로 렌더링한다.
    기존(채팅 도입 전) 레이아웃을 그대로 재사용 — 최초 자동 브리핑뿐 아니라
    채팅의 모든 AI 응답에 동일하게 적용해서 시각적 형태를 유지한다."""
    sections = re.split(r'(?=\n\s*(?:\d+[\.\)]))', raw_text)

    # "0. 가정 보유기간"으로 시작하는 정식 6항목 브리핑일 때만 큰 제목을 표시
    if re.match(r'^\s*0[\.\)]', raw_text.strip()):
        st.markdown(
            f"""<div class="ai-card-title">💡 종합 전략 리포트 ({symbol})</div>""",
            unsafe_allow_html=True,
        )

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
            top = rankings[0]
            st.success(
                f"🎯 1위 종목 자동 선정: **{top['symbol']}** — 방향: **{top['final_direction']}** "
                f"(권장 보유기간: {top['horizon']}, 스캔된 {len(rankings)}개 종목 중 확신도 1위)"
            )
            st.rerun()
        else:
            st.error("스캔 가능한 종목이 없습니다. 잠시 후 다시 시도해주세요.")

# ---------------------------------------------------------
# 6.1. 랭킹 결과 카드 박스 출력 영역
# ---------------------------------------------------------
if st.session_state.scan_rankings:
    import textwrap

    rank_html = textwrap.dedent(f"""
    <div class="rank-report-box">
        <div style="font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
            📊 실시간 멀티팩터 전 종목 스캔 랭킹 리포트 (LONG·SHORT 대칭 평가)
        </div>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 12px;">
            ※ 순위는 '가장 강한 상승세'가 아니라, 이번 스캔에 포함된 {len(st.session_state.scan_rankings)}개 종목 중
            <b>방향(LONG/SHORT)에 상관없이 확신도가 가장 큰 종목</b> 순입니다. 점수는 절대 신뢰도가 아닌 상대 순위이며,
            권장 보유기간은 1H/4H/1D 추세 정렬 정도로 추정한 참고값입니다.
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
            <tr style="border-bottom: 2px solid #CBD5E1; color: #475569; text-align: left;">
                <th style="padding: 6px;">순위</th>
                <th style="padding: 6px;">종목명</th>
                <th style="padding: 6px;">방향</th>
                <th style="padding: 6px;">확신도</th>
                <th style="padding: 6px;">권장 보유기간</th>
                <th style="padding: 6px;">현재가 / 등락</th>
                <th style="padding: 6px;">상세 상태 분석 및 선정 이유</th>
            </tr>
    """)

    for idx, item in enumerate(st.session_state.scan_rankings, 1):
        c_color = "#16A34A" if item["change"] >= 0 else "#DC2626"
        if item["final_direction"] == "LONG":
            dir_color, dir_bg = "#16A34A", "#DCFCE7"
        elif item["final_direction"] == "SHORT":
            dir_color, dir_bg = "#DC2626", "#FEE2E2"
        else:
            dir_color, dir_bg = "#64748B", "#F1F5F9"
        rank_html += textwrap.dedent(f"""\
        <tr style="border-bottom: 1px solid #E2E8F0; color: #1E293B; vertical-align: top;">
        <td style="padding: 8px; font-weight: 700;">{idx}위</td>
        <td style="padding: 8px; font-weight: 800;">{item['symbol']}</td>
        <td style="padding: 8px;"><span style="background:{dir_bg}; color:{dir_color}; font-weight: 800; padding: 2px 8px; border-radius: 6px;">{item['final_direction']}</span></td>
        <td style="padding: 8px; font-weight: 800; color: #2563EB;">{item['score']:.1f}점</td>
        <td style="padding: 8px; font-size: 0.85rem;">{item['horizon']}</td>
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
    # 9. 정량 신호 계산 + 포지션 사이징 계산기 호출
    # ---------------------------------------------------------
    # 랭킹 스캔과 동일한 로직(교차단면 Z-score, 방향 대칭)으로 지금 보고 있는
    # 종목의 정량 신호(LONG/SHORT, 확신도, 권장 보유기간)를 계산해서, 포지션
    # 사이징 힌트와 AI 어드바이저 프롬프트가 항상 같은 신호를 참고하게 한다.
    quant_item, quant_errors = get_quant_signal_for_symbol(st.session_state.target_symbol)

    render_position_sizing_calculator(curr["close"], curr["ATR"], st.session_state.target_symbol, quant_item)

    st.divider()

    # ---------------------------------------------------------
    # 10. Gemini AI 어드바이저 (실시간 멀티턴 채팅)
    # ---------------------------------------------------------
    st.markdown("### 🤖 Gemini AI 선물 전략 어드바이저 — 실시간 채팅")

    if quant_item:
        badge_color = "#16A34A" if quant_item["final_direction"] == "LONG" else (
            "#DC2626" if quant_item["final_direction"] == "SHORT" else "#64748B"
        )
        st.markdown(
            f"""<div style="font-size:0.85rem; color:#475569; margin-bottom:8px;">
            정량 모델 판단: <b style="color:{badge_color};">{quant_item['final_direction']}</b>
            (확신도 z={quant_item['final_z']:+.2f}, 상대강도 {quant_item['score']:.0f}/100) ·
            자동 추정 권장 보유기간: <b>{quant_item['horizon']}</b>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("⚠️ 정량 신호를 계산하지 못했습니다(데이터 부족). AI는 아래 기본 데이터만으로 판단합니다.")

    horizon_choice = st.selectbox(
        "보유기간 관점 (전략에 반영됩니다)",
        ["자동 (퀀트 신호 기반)", "스캘핑 (수 시간 이내)", "데이트레이딩 (당일)", "스윙 (수일~1~2주)"],
        key="horizon_choice",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 종목이 바뀌거나 재스캔되면(=auto_strategy_trigger) 이전 종목 대화를
    # 새 종목에 이어붙이지 않도록 대화창을 초기화한다.
    if st.session_state.auto_strategy_trigger:
        st.session_state.chat_history = []

    def _build_system_instruction():
        trend_4h = get_trend_label(st.session_state.target_symbol, "4h")
        trend_1h = "up" if curr["close"] > curr["EMA_50"] else "down"
        trend_1d = get_trend_label(st.session_state.target_symbol, "1d")
        funding_display = f"{funding_rate:.4f}%" if funding_rate is not None else "N/A"

        if quant_item:
            quant_summary = (
                f"방향={quant_item['final_direction']}, 확신도(z-score)={quant_item['final_z']:+.2f}, "
                f"상대강도={quant_item['score']:.0f}/100, 자동 추정 권장 보유기간={quant_item['horizon']}"
            )
        else:
            quant_summary = "계산 실패(데이터 부족) - 아래 원시 지표만으로 판단할 것"

        if horizon_choice.startswith("자동"):
            horizon_instruction = "위 '정량 모델 판단'의 권장 보유기간을 그대로 채택하라."
        else:
            horizon_instruction = f"사용자가 보유기간을 '{horizon_choice}'로 직접 지정했다. 이 보유기간을 최우선으로 반영해서 전략을 다시 설계하라(정량 모델의 자동 추정 보유기간과 다르면 그 차이를 짧게 언급할 것)."

        return f"""
너는 리스크 관리를 최우선으로 하는 정량 기반 트레이딩 AI다.
너의 목표는 "차트가 상승 중인지"를 설명하는 게 아니라, "지금 이 종목에서 LONG과 SHORT 중 어느 방향에 배팅해야
같은 리스크 대비 기대수익(EV)이 가장 큰지, 그리고 그 배팅을 얼마나 오래 들고 가야 하는지"를 정하는 것이다.
LONG과 SHORT은 완전히 동등한 후보다. 데이터가 하락을 가리키면 절대 망설이지 말고 SHORT을 추천하라.
'상승장이니까 일단 롱' 같은 관성적 판단은 절대 하지 마라.

너는 지금 사용자와 실시간으로 대화하는 채팅형 AI 어드바이저다. 첫 메시지에서는 아래 6가지 항목으로
전체 전략을 제시하고, 이후 사용자의 후속 질문(예: "손절가 더 타이트하게", "숏 관점은?", "왜 그렇게 판단했어?")에는
이전 대화 맥락을 유지하면서 해당 질문에 맞게 자연스럽게 답하라. 매번 6개 항목을 전부 반복할 필요는 없다.

[현재 데이터]
- 종목: {st.session_state.target_symbol}, 현재가: ${curr['close']:,.2f}
- RSI(14): {curr['RSI']:.1f}
- 추세: 1H={trend_1h}, 4H={trend_4h}, 1D={trend_1d}
- 펀딩비: {funding_display}
- 정량 모델 판단(교차단면 Z-score 기반, 6개 종목 비교 결과): {quant_summary}
- 리스크 조건: 거래 리스크 5%, 손절 ATR 배수 1.5, 레버리지 5배

[보유기간 지침]
{horizon_instruction}

[최초 전략 브리핑 작성 규칙 - 대화의 첫 메시지에만 적용]
- 절대 인사말이나 서두 멘트("...분석 리포트입니다" 등)를 출력하지 말고 바로 본론부터 시작하라.
- 항목 3의 방향이 위 "정량 모델 판단"의 방향과 다르다면, 항목 1에서 왜 정량 신호와 다르게 판단했는지
  구체적 근거(예: 정량 모델이 못 보는 최근 뉴스성 변수, 극단적 리스크 등)를 반드시 명시하라.
  근거 없이 임의로 방향을 뒤집지 마라.
- 아래 6가지 항목을 번호로 나누어 작성하라:
0. 가정 보유기간 (구체적 시간 단위로 명시. 예: "약 6~12시간" 또는 "약 3~7일")
1. 시장 국면 및 변동성 체질 진단 (정량 모델 판단과 일치/불일치 여부와 근거 포함)
2. 멀티타임프레임(1H/4H/1D) 추세 정렬 상태
3. 구체적인 진입 타점 및 [LONG] 또는 [SHORT] 방향 (1차, 2차, 3차 분할 진입 타점과 각 비중을 상세히 포함할 것)
4. 리스크 관리 및 확정 손절가
5. 기대 수익 및 목표가 셋업 (0번에서 밝힌 보유기간 안에 현실적으로 도달 가능한 목표로 설정)

별도의 마크다운 볼드(**) 기호는 제목에 절대 사용하지 말고 숫자 기호 등을 활용하여 가독성 높게 작성하라.
"""

    def _stream_gemini_reply(prompt_text):
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_instruction = _build_system_instruction()
        # 이전 대화 턴을 그대로 실어 보내 멀티턴 맥락(이전 질문/답변)을 유지한다.
        contents = [
            types.Content(
                role=("user" if m["role"] == "user" else "model"),
                parts=[types.Part(text=m["content"])],
            )
            for m in st.session_state.chat_history[:-1]
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt_text)]))

        stream = client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    # 지금까지의 대화 이력 렌더링 (AI 응답은 기존 색상 카드 레이아웃 그대로 유지)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                render_ai_report_cards(msg["content"], st.session_state.target_symbol)

    def _is_transient_gemini_error(e):
        msg = str(e)
        return any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"))

    def _run_and_render_ai_reply(prompt_text):
        """스트리밍 응답을 시도하고, 503/429 같은 일시적 오류면 자동 재시도한다.
        실패한 응답은 대화 이력(chat_history)에 넣지 않는다 - 그대로 넣으면
        다음 턴에 '모델이 방금 에러 메시지를 말했다'는 잘못된 맥락이 그대로
        Gemini에게 다시 전달돼 대화가 꼬인다."""
        max_retries = 2
        retry_delays = [2, 4]

        with st.chat_message("assistant", avatar="🤖"):
            stream_slot = st.empty()
            full_reply = ""
            last_err = None
            for attempt in range(max_retries + 1):
                full_reply = ""
                try:
                    for chunk in _stream_gemini_reply(prompt_text):
                        full_reply += chunk
                        stream_slot.markdown(full_reply + "▌")  # 실시간 타이핑 효과
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_retries and _is_transient_gemini_error(e):
                        wait_s = retry_delays[attempt]
                        stream_slot.info(
                            f"⏳ Gemini 서버가 일시적으로 혼잡합니다. {wait_s}초 후 자동 재시도합니다 "
                            f"({attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_s)
                        continue
                    break

            stream_slot.empty()
            if last_err is None:
                render_ai_report_cards(full_reply, st.session_state.target_symbol)
                st.session_state.chat_history.append({"role": "assistant", "content": full_reply})
            else:
                st.error(f"⚠️ Gemini API 호출 오류(자동 재시도 {max_retries}회 모두 실패): {last_err}")
                if st.button("🔄 다시 시도", key=f"retry_{len(st.session_state.chat_history)}"):
                    st.session_state.pending_retry_prompt = prompt_text
                    st.rerun()

        st.session_state.auto_strategy_trigger = False

    user_prompt = st.chat_input("종목 분석, 진입가 조정, 리스크 관리 등 무엇이든 물어보세요...")
    retry_prompt = st.session_state.pop("pending_retry_prompt", None)

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_prompt)
        _run_and_render_ai_reply(user_prompt)
    elif retry_prompt:
        # 재시도는 이미 chat_history에 있는 사용자 질문을 그대로 재사용하므로
        # 사용자 말풍선을 다시 추가하지 않는다(중복 표시 방지).
        _run_and_render_ai_reply(retry_prompt)
    elif st.session_state.auto_strategy_trigger and not st.session_state.chat_history:
        auto_prompt = f"선택된 종목인 {st.session_state.target_symbol}에 대한 트레이딩 전략을 분석해줘."
        st.session_state.chat_history.append({"role": "user", "content": auto_prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(auto_prompt)
        _run_and_render_ai_reply(auto_prompt)

    st.caption(
        "⚠️ 이 채팅은 규칙 기반 정량 신호 + LLM 해설을 결합한 참고 자료이며, "
        "수익을 보장하지 않습니다. 실제 자금 투입 전 반드시 과거 데이터로 이 로직의 성과를 검증하고, "
        "소액/페이퍼 트레이딩으로 먼저 확인하는 것을 권장합니다."
    )
