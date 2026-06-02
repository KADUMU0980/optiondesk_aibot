from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import pandas as pd
import threading
import time
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.')
app.secret_key = os.urandom(24)
CORS(app)

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
state = {
    "token": None,
    "last_update": None,
    "option_chain": {},      # { "NIFTY": [...], "SENSEX": [...] }
    "spot_price": {},        # { "NIFTY": 24500.0 }
    "expiries": {},          # { "NIFTY": ["2025-06-05", ...] }
    "active_expiry": {},     # { "NIFTY": "2025-06-05" }
    "vix": None,
    "alerts": {},            # { "NIFTY": [...], "SENSEX": [...] }  per-index
    "sentiment": {},
    "trade_suggestions": {},
    "support_resistance": {},
    "greeks_summary": {},
    "insights": {},
    "error": None,
}

INSTRUMENT_KEYS = {
    "NIFTY":  "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
}
VIX_KEY = "NSE_INDEX|India VIX"
UPSTOX_BASE = "https://api.upstox.com/v2"
UPDATE_INTERVAL = 10

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

def hdrs():
    return {
        "Authorization": f"Bearer {state['token']}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def api_get(path, params=None):
    """Generic GET with error logging. Returns parsed JSON or None."""
    try:
        r = requests.get(f"{UPSTOX_BASE}{path}", headers=hdrs(), params=params, timeout=12)
        d = r.json()
        if d.get("status") == "success":
            return d.get("data")
        else:
            logger.warning(f"API {path} error: {d}")
    except Exception as e:
        logger.error(f"API {path} exception: {e}")
    return None


def fetch_ltp(instrument_key):
    data = api_get("/market-quote/ltp", {"instrument_key": instrument_key})
    if data:
        key = list(data.keys())[0]
        return float(data[key].get("last_price", 0) or 0)
    return None


def fetch_expiries(index_name):
    """Returns sorted list of upcoming expiry date strings."""
    data = api_get("/option/contract", {"instrument_key": INSTRUMENT_KEYS[index_name]})
    if data:
        today = datetime.now().date()
        expiries = sorted(set(
            item["expiry"] for item in data
            if item.get("expiry") and item["expiry"] >= str(today)
        ))
        return expiries
    return []


def fetch_option_chain(index_name, expiry_date):
    """Fetch raw option chain for given index + expiry. Returns list of strike rows."""
    data = api_get("/option/chain", {
        "instrument_key": INSTRUMENT_KEYS[index_name],
        "expiry_date": expiry_date,
    })
    return data or []


# ─────────────────────────────────────────────
# PARSE OPTION CHAIN ROW
# ─────────────────────────────────────────────

def parse_row(item):
    """Convert raw Upstox option chain item into flat dict."""
    def safe(d, *keys, default=0):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
        return float(d) if d is not None else default

    ce = item.get("call_options", {})
    pe = item.get("put_options", {})
    ce_md = ce.get("market_data", {})
    pe_md = pe.get("market_data", {})
    ce_gr = ce.get("option_greeks", {})
    pe_gr = pe.get("option_greeks", {})

    ce_oi    = safe(ce_md, "oi")
    pe_oi    = safe(pe_md, "oi")
    ce_prev  = safe(ce_md, "prev_oi")
    pe_prev  = safe(pe_md, "prev_oi")

    spot = float(item.get("underlying_spot_price") or 0)
    strike = float(item.get("strike_price") or 0)

    return {
        "strike":      strike,
        "expiry":      item.get("expiry", ""),
        "spot":        spot,
        # CE
        "ce_ltp":      safe(ce_md, "ltp"),
        "ce_oi":       ce_oi,
        "ce_prev_oi":  ce_prev,
        "ce_chg_oi":   ce_oi - ce_prev,
        "ce_vol":      safe(ce_md, "volume"),
        "ce_delta":    safe(ce_gr, "delta"),
        "ce_gamma":    safe(ce_gr, "gamma"),
        "ce_theta":    safe(ce_gr, "theta"),
        "ce_vega":     safe(ce_gr, "vega"),
        "ce_iv":       safe(ce_gr, "iv"),
        # PE
        "pe_ltp":      safe(pe_md, "ltp"),
        "pe_oi":       pe_oi,
        "pe_prev_oi":  pe_prev,
        "pe_chg_oi":   pe_oi - pe_prev,
        "pe_vol":      safe(pe_md, "volume"),
        "pe_delta":    safe(pe_gr, "delta"),
        "pe_gamma":    safe(pe_gr, "gamma"),
        "pe_theta":    safe(pe_gr, "theta"),
        "pe_vega":     safe(pe_gr, "vega"),
        "pe_iv":       safe(pe_gr, "iv"),
        # Computed
        "pcr":         round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0,
        "distance":    round(((strike - spot) / spot) * 100, 2) if spot > 0 else 0,
    }


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────

def detect_support_resistance(chain, spot):
    if not chain:
        return {"support": [], "resistance": []}

    df = pd.DataFrame(chain)
    ce_q75 = df["ce_oi"].quantile(0.75) if len(df) > 4 else 0
    pe_q75 = df["pe_oi"].quantile(0.75) if len(df) > 4 else 0

    resistances, supports = [], []

    for row in chain:
        s = row["strike"]
        if s > spot and row["ce_oi"] > 0:
            strength = "hard" if row["ce_oi"] >= ce_q75 else "soft"
            resistances.append({
                "strike": s, "oi": int(row["ce_oi"]),
                "chg_oi": int(row["ce_chg_oi"]), "vol": int(row["ce_vol"]),
                "strength": strength,
                "weakening":     bool(row["ce_chg_oi"] < -row["ce_oi"] * 0.03),
                "strengthening": bool(row["ce_chg_oi"] >  row["ce_oi"] * 0.03),
            })
        if s < spot and row["pe_oi"] > 0:
            strength = "hard" if row["pe_oi"] >= pe_q75 else "soft"
            supports.append({
                "strike": s, "oi": int(row["pe_oi"]),
                "chg_oi": int(row["pe_chg_oi"]), "vol": int(row["pe_vol"]),
                "strength": strength,
                "weakening":     bool(row["pe_chg_oi"] < -row["pe_oi"] * 0.03),
                "strengthening": bool(row["pe_chg_oi"] >  row["pe_oi"] * 0.03),
            })

    return {
        "resistance": sorted(resistances, key=lambda x: x["oi"], reverse=True)[:6],
        "support":    sorted(supports,    key=lambda x: x["oi"], reverse=True)[:6],
    }


def compute_sentiment(chain, spot, vix, pcr):
    if not chain:
        return {"bullish": 33, "bearish": 33, "neutral": 34, "label": "NEUTRAL", "score": 50, "pcr": pcr}

    signals = []

    # PCR
    if   pcr > 1.4: signals += [("bullish", 3)]
    elif pcr > 1.1: signals += [("bullish", 1)]
    elif pcr < 0.7: signals += [("bearish", 3)]
    elif pcr < 0.9: signals += [("bearish", 1)]
    else:           signals += [("neutral", 1)]

    # OI change direction
    ce_chg = sum(r["ce_chg_oi"] for r in chain)
    pe_chg = sum(r["pe_chg_oi"] for r in chain)
    if pe_chg > 0 and ce_chg <= 0:  signals += [("bullish", 2)]
    elif ce_chg > 0 and pe_chg <= 0: signals += [("bearish", 2)]
    else: signals += [("neutral", 1)]

    # VIX
    if vix:
        if   vix < 14: signals += [("bullish", 1)]
        elif vix > 22: signals += [("bearish", 2)]
        elif vix > 18: signals += [("bearish", 1)]

    # ATM delta
    atm = min(chain, key=lambda x: abs(x["strike"] - spot))
    nd = atm["ce_delta"] + atm["pe_delta"]
    if   nd >  0.1: signals += [("bullish", 1)]
    elif nd < -0.1: signals += [("bearish", 1)]

    bull = sum(w for d,w in signals if d=="bullish")
    bear = sum(w for d,w in signals if d=="bearish")
    neut = sum(w for d,w in signals if d=="neutral")
    total = max(bull+bear+neut, 1)

    bull_p = round(bull/total*100)
    bear_p = round(bear/total*100)
    neut_p = 100 - bull_p - bear_p
    score  = round(50 + (bull_p - bear_p)/2)
    label  = "BULLISH" if score>65 else ("BEARISH" if score<35 else "NEUTRAL")

    return {"bullish": bull_p, "bearish": bear_p, "neutral": neut_p,
            "label": label, "score": score, "pcr": round(pcr, 2)}


def generate_alerts(chain, spot, vix, pcr, sr, sentiment):
    alerts = []
    now = datetime.now().strftime("%H:%M:%S")

    def add(msg, level="info"):
        alerts.append({"msg": msg, "level": level, "time": now})

    if pcr > 1.5:  add("🟢 STRONG BULLISH: PCR extremely high — Put writers dominating", "success")
    elif pcr > 1.2: add("📈 PCR turning bullish — Positive market bias", "success")
    elif pcr < 0.7: add("🔴 WARNING: PCR turning bearish — Call writers dominant", "danger")
    elif pcr < 0.9: add("⚠️ PCR weakening — Monitor for reversal", "warning")

    if vix:
        if   vix > 22: add(f"⚡ VOLATILITY SPIKE — VIX at {vix:.1f}", "danger")
        elif vix < 12: add(f"🔇 LOW VOLATILITY — VIX {vix:.1f} — Theta decay environment", "info")
        elif 14 <= vix <= 17: add(f"✅ VIX supportive for trends — {vix:.1f}", "success")

    for r in (sr.get("resistance") or [])[:2]:
        if r["weakening"]:    add(f"⚠️ {int(r['strike'])} resistance WEAKENING — Breakout possible", "warning")
        elif r["strength"]=="hard": add(f"🔴 HARD RESISTANCE at {int(r['strike'])} — {r['oi']//1000}K OI", "danger")

    for s in (sr.get("support") or [])[:2]:
        if s["weakening"]:    add(f"⚠️ {int(s['strike'])} support WEAKENING", "warning")
        elif s["strength"]=="hard": add(f"🟢 STRONG SUPPORT at {int(s['strike'])} — {s['oi']//1000}K OI", "success")

    ce_chg = sum(r["ce_chg_oi"] for r in chain)
    pe_chg = sum(r["pe_chg_oi"] for r in chain)
    if pe_chg > 0 and ce_chg < 0:  add("📊 Put writing increasing — Call unwinding detected", "success")
    elif ce_chg > 0 and pe_chg < 0: add("📊 Call writing increasing — Bearish pressure building", "warning")

    atm = min(chain, key=lambda x: abs(x["strike"]-spot))
    if atm["ce_gamma"] > 0.002:
        add(f"⚡ GAMMA EXPANSION at {int(atm['strike'])} ATM — Large move imminent", "warning")

    if sentiment["label"]=="BULLISH" and sentiment["score"] > 70:
        add("🚀 BREAKOUT POSSIBILITY HIGH — Institutional bull positioning", "success")
    elif sentiment["label"]=="BEARISH" and sentiment["score"] < 30:
        add("🔴 BEARISH PRESSURE BUILDING — Short accumulation detected", "danger")

    return alerts[:8]


def generate_insights(chain, spot, vix, pcr, sr):
    ins = []
    if not chain: return ins

    for r in (sr.get("resistance") or [])[:2]:
        s = int(r["strike"])
        if r["strengthening"]: ins.append(f"Strong resistance forming at {s} — heavy call writing")
        elif r["weakening"]:   ins.append(f"Resistance weakening at {s} — possible breakout loading")
        else:                  ins.append(f"Resistance zone active at {s} — {r['oi']//1000}K CE OI")

    for s in (sr.get("support") or [])[:2]:
        st = int(s["strike"])
        if s["strengthening"]: ins.append(f"Support strengthening at {st} — put writers absorbing")
        elif s["weakening"]:   ins.append(f"Weak support likely to fail at {st}")
        else:                  ins.append(f"Support holding at {st} — {s['oi']//1000}K PE OI")

    if   pcr > 1.3: ins.append("Put writers aggressively defending downside")
    elif pcr < 0.8: ins.append("Call writers dominant — bearish bias intact")

    if vix:
        if   vix < 14: ins.append("Volatility compression — range-bound market likely")
        elif vix > 20: ins.append("Momentum expansion likely — high volatility regime")
        else:          ins.append("VIX supportive for directional moves")

    atm = min(chain, key=lambda x: abs(x["strike"]-spot))
    if atm["ce_gamma"] > 0.002:  ins.append("Gamma expansion near ATM — intraday range may widen")
    if atm["ce_theta"] < -50:    ins.append("Theta decay dominant — option sellers advantaged")

    total_vol = sum(r["ce_vol"]+r["pe_vol"] for r in chain)
    if total_vol > 5_000_000: ins.append("High volume detected — institutional activity elevated")

    return ins[:8]


def generate_trade_suggestions(chain, spot, vix, pcr, sr, sentiment):
    """
    Institutional-grade trade suggestion engine.
    Uses: Max Pain, OI writing/unwinding, ITM/OTM skew, CE/PE delta balance,
    gamma positioning, IV skew, volume confirmation, S/R confluence.
    Always returns 3 suggestions regardless of sentiment.
    """
    sugg = []
    if not chain or spot <= 0: return sugg

    import statistics

    strikes   = sorted(set(r["strike"] for r in chain))
    if len(strikes) < 4: return sugg
    step      = strikes[1] - strikes[0]
    atm_row   = min(chain, key=lambda x: abs(x["strike"] - spot))
    atm       = atm_row["strike"]

    # ── MAX PAIN CALCULATION ──
    def max_pain(chain):
        pain = {}
        for exp_s in strikes:
            total = 0
            for row in chain:
                ce_loss = max(0, exp_s - row["strike"]) * row["ce_oi"]
                pe_loss = max(0, row["strike"] - exp_s) * row["pe_oi"]
                total  += ce_loss + pe_loss
            pain[exp_s] = total
        return min(pain, key=pain.get)

    mp = max_pain(chain)

    # ── OI ANALYSIS ──
    total_ce_oi  = sum(r["ce_oi"]     for r in chain)
    total_pe_oi  = sum(r["pe_oi"]     for r in chain)
    total_ce_chg = sum(r["ce_chg_oi"] for r in chain)
    total_pe_chg = sum(r["pe_chg_oi"] for r in chain)
    total_ce_vol = sum(r["ce_vol"]    for r in chain)
    total_pe_vol = sum(r["pe_vol"]    for r in chain)

    # OTM CE writing (bearish pressure above spot)
    otm_ce_writing = sum(r["ce_chg_oi"] for r in chain if r["strike"] > spot and r["ce_chg_oi"] > 0)
    # OTM PE writing (bullish support below spot)
    otm_pe_writing = sum(r["pe_chg_oi"] for r in chain if r["strike"] < spot and r["pe_chg_oi"] > 0)
    # CE unwinding above spot (shorts covering = bullish)
    ce_unwinding   = sum(abs(r["ce_chg_oi"]) for r in chain if r["strike"] > spot and r["ce_chg_oi"] < 0)
    # PE unwinding below spot (longs exiting = bearish)
    pe_unwinding   = sum(abs(r["pe_chg_oi"]) for r in chain if r["strike"] < spot and r["pe_chg_oi"] < 0)

    # ── IV SKEW ──
    otm_ce_rows = [r for r in chain if r["strike"] > spot and r["ce_iv"] > 0]
    otm_pe_rows = [r for r in chain if r["strike"] < spot and r["pe_iv"] > 0]
    avg_ce_iv   = statistics.mean([r["ce_iv"] for r in otm_ce_rows]) if otm_ce_rows else 20
    avg_pe_iv   = statistics.mean([r["pe_iv"] for r in otm_pe_rows]) if otm_pe_rows else 20
    iv_skew     = avg_pe_iv - avg_ce_iv   # positive = put skew = bearish fear

    # ── ATM GREEKS ──
    atm_ce_delta  = atm_row["ce_delta"]
    atm_pe_delta  = atm_row["pe_delta"]
    atm_gamma     = atm_row["ce_gamma"]
    atm_theta     = atm_row["ce_theta"]
    atm_ce_ltp    = atm_row["ce_ltp"]
    atm_pe_ltp    = atm_row["pe_ltp"]
    atm_straddle  = atm_ce_ltp + atm_pe_ltp

    # ── NEAREST S/R ──
    res_list = sr.get("resistance", [])
    sup_list = sr.get("support", [])
    nearest_res = res_list[0]["strike"] if res_list else spot + step * 4
    nearest_sup = sup_list[0]["strike"] if sup_list else spot - step * 4
    hard_res    = next((r for r in res_list if r["strength"]=="hard"), None)
    hard_sup    = next((r for r in sup_list if r["strength"]=="hard"), None)

    # ── SCORING SYSTEM ──
    # Each signal adds to bull_score or bear_score
    bull_score = 0
    bear_score = 0
    bull_reasons = []
    bear_reasons = []

    # 1. PCR
    if pcr > 1.5:   bull_score += 3; bull_reasons.append(f"PCR extremely bullish ({pcr:.2f}) — put writers dominant")
    elif pcr > 1.2: bull_score += 2; bull_reasons.append(f"PCR bullish ({pcr:.2f}) — positive bias")
    elif pcr > 1.0: bull_score += 1; bull_reasons.append(f"PCR mildly bullish ({pcr:.2f})")
    elif pcr < 0.6: bear_score += 3; bear_reasons.append(f"PCR extremely bearish ({pcr:.2f}) — call writers dominant")
    elif pcr < 0.8: bear_score += 2; bear_reasons.append(f"PCR bearish ({pcr:.2f}) — negative bias")
    elif pcr < 1.0: bear_score += 1; bear_reasons.append(f"PCR mildly bearish ({pcr:.2f})")

    # 2. OI writing direction
    if otm_pe_writing > otm_ce_writing * 1.3:
        bull_score += 2; bull_reasons.append("Aggressive PE writing below spot — support building")
    elif otm_ce_writing > otm_pe_writing * 1.3:
        bear_score += 2; bear_reasons.append("Aggressive CE writing above spot — resistance building")

    # 3. Unwinding (short covering)
    if ce_unwinding > total_ce_oi * 0.02:
        bull_score += 2; bull_reasons.append("CE (call) short covering — bears exiting above spot")
    if pe_unwinding > total_pe_oi * 0.02:
        bear_score += 2; bear_reasons.append("PE (put) long unwinding — support eroding below spot")

    # 4. Volume confirmation
    if total_pe_vol > total_ce_vol * 1.3:
        bull_score += 1; bull_reasons.append("Put volume elevated — hedging/protection buying")
    elif total_ce_vol > total_pe_vol * 1.3:
        bear_score += 1; bear_reasons.append("Call volume elevated — bearish speculation")

    # 5. VIX
    if vix:
        if vix < 14:   bull_score += 1; bull_reasons.append(f"VIX low ({vix:.1f}) — complacent, trend-following")
        elif vix < 18: bull_score += 1; bull_reasons.append(f"VIX normal ({vix:.1f}) — supportive for directional move")
        elif vix > 22: bear_score += 2; bear_reasons.append(f"VIX elevated ({vix:.1f}) — fear rising")

    # 6. Max Pain vs Spot
    mp_dist = mp - spot
    if mp_dist > step * 2:
        bull_score += 1; bull_reasons.append(f"Max pain at {int(mp)} — spot likely pulled up")
    elif mp_dist < -step * 2:
        bear_score += 1; bear_reasons.append(f"Max pain at {int(mp)} — spot likely pulled down")

    # 7. IV Skew
    if iv_skew > 3:
        bear_score += 1; bear_reasons.append(f"Put IV skew elevated ({iv_skew:.1f}%) — downside fear hedging")
    elif iv_skew < -2:
        bull_score += 1; bull_reasons.append(f"Call IV premium ({-iv_skew:.1f}%) — upside demand")

    # 8. Hard S/R
    if hard_res and hard_res["weakening"]:
        bull_score += 2; bull_reasons.append(f"Hard resistance at {int(hard_res['strike'])} weakening — breakout imminent")
    if hard_sup and hard_sup["strengthening"]:
        bull_score += 1; bull_reasons.append(f"Hard support at {int(hard_sup['strike'])} absorbing — floor firm")
    if hard_sup and hard_sup["weakening"]:
        bear_score += 2; bear_reasons.append(f"Hard support at {int(hard_sup['strike'])} weakening — breakdown risk")
    if hard_res and hard_res["strengthening"]:
        bear_score += 1; bear_reasons.append(f"Hard resistance at {int(hard_res['strike'])} strengthening — ceiling firm")

    # ── DETERMINE PRIMARY BIAS ──
    total_score  = bull_score + bear_score
    bull_pct     = round(bull_score / max(total_score, 1) * 100)
    bear_pct     = round(bear_score / max(total_score, 1) * 100)

    # Nearest OTM call for entry (1 step above ATM)
    atm_idx     = strikes.index(atm) if atm in strikes else len(strikes)//2
    call_strike = strikes[atm_idx + 1] if atm_idx + 1 < len(strikes) else int(atm + step)
    put_strike  = strikes[atm_idx - 1] if atm_idx - 1 >= 0 else int(atm - step)

    # ── SUGGESTION 1: PRIMARY DIRECTIONAL ──
    if bull_score >= bear_score:
        # CALL BUY / BULLISH PLAY
        conf        = min(92, 45 + bull_pct // 2 + (5 if pcr > 1.3 else 0) + (3 if vix and vix < 17 else 0))
        target_s    = hard_res["strike"] if hard_res and not hard_res["weakening"] else nearest_res
        sl_strike   = hard_sup["strike"] if hard_sup else nearest_sup
        top4_bull   = bull_reasons[:4] if bull_reasons else ["PCR bullish", "OI structure supports upside"]
        sugg.append({
            "type": "CALL BUY",
            "color": "bullish",
            "strike": int(call_strike),
            "entry": f"Buy {int(call_strike)} CE above {int(spot)}",
            "sl":    f"Close below {int(sl_strike)} (prev. support)",
            "target": f"{int(target_s)} (CE wall / resistance)",
            "confidence": conf,
            "reasons": top4_bull,
            "max_pain": int(mp),
            "bias_score": f"Bull {bull_pct}% vs Bear {bear_pct}%",
        })
    else:
        # PUT BUY / BEARISH PLAY
        conf        = min(92, 45 + bear_pct // 2 + (5 if pcr < 0.8 else 0))
        target_s    = hard_sup["strike"] if hard_sup and not hard_sup["weakening"] else nearest_sup
        sl_strike   = hard_res["strike"] if hard_res else nearest_res
        top4_bear   = bear_reasons[:4] if bear_reasons else ["PCR bearish", "OI structure supports downside"]
        sugg.append({
            "type": "PUT BUY",
            "color": "bearish",
            "strike": int(put_strike),
            "entry": f"Buy {int(put_strike)} PE below {int(spot)}",
            "sl":    f"Close above {int(sl_strike)} (prev. resistance)",
            "target": f"{int(target_s)} (PE wall / support)",
            "confidence": conf,
            "reasons": top4_bear,
            "max_pain": int(mp),
            "bias_score": f"Bear {bear_pct}% vs Bull {bull_pct}%",
        })

    # ── SUGGESTION 2: INTRADAY SCALP / SHORT COVERING ──
    # Detect short covering: CE unwinding + put writing together
    short_covering = ce_unwinding > 0 and otm_pe_writing > 0
    long_unwinding = pe_unwinding > 0 and otm_ce_writing > 0

    if short_covering and bull_score >= bear_score:
        scalp_target = nearest_res
        sugg.append({
            "type": "INTRADAY SCALP — LONG",
            "color": "bullish",
            "strike": int(call_strike),
            "entry": f"Buy {int(call_strike)} CE on dip near {int(atm)}",
            "sl":    f"Spot breaks below {int(atm - step)}",
            "target": f"{int(scalp_target)} intraday",
            "confidence": min(85, 55 + bull_score * 3),
            "reasons": [
                "CE short covering above spot",
                "PE writing confirming support",
                "Momentum buy setup",
                f"Max pain magnet at {int(mp)}",
            ],
            "max_pain": int(mp),
            "bias_score": f"Bull {bull_pct}% vs Bear {bear_pct}%",
        })
    elif long_unwinding and bear_score > bull_score:
        scalp_target = nearest_sup
        sugg.append({
            "type": "INTRADAY SCALP — SHORT",
            "color": "bearish",
            "strike": int(put_strike),
            "entry": f"Buy {int(put_strike)} PE on bounce near {int(atm)}",
            "sl":    f"Spot breaks above {int(atm + step)}",
            "target": f"{int(scalp_target)} intraday",
            "confidence": min(85, 55 + bear_score * 3),
            "reasons": [
                "PE long unwinding below spot",
                "CE writing confirming resistance",
                "Momentum sell setup",
                f"Max pain drag to {int(mp)}",
            ],
            "max_pain": int(mp),
            "bias_score": f"Bear {bear_pct}% vs Bull {bull_pct}%",
        })
    else:
        # Fallback: Iron Condor / Range if balanced
        if abs(bull_score - bear_score) <= 2:
            upper = nearest_res
            lower = nearest_sup
            sugg.append({
                "type": "IRON CONDOR — RANGE",
                "color": "neutral",
                "strike": int(atm),
                "entry": f"Sell {int(lower)} PE + {int(upper)} CE (collect {int(atm_straddle*0.4):.0f} pts)",
                "sl":    f"Spot < {int(lower - step)} or > {int(upper + step)}",
                "target": f"Keep {int(atm_straddle * 0.35):.0f}+ pts premium (35% of straddle)",
                "confidence": 70,
                "reasons": [
                    f"OI range locked {int(lower)}–{int(upper)}",
                    "Balanced bull/bear OI — market indecision",
                    f"Theta decay — ATM straddle ₹{atm_straddle:.0f}",
                    f"Max pain anchor at {int(mp)}",
                ],
                "max_pain": int(mp),
                "bias_score": f"Bull {bull_pct}% vs Bear {bear_pct}%",
            })
        else:
            sugg.append({
                "type": "AVOID — CONFLICTING SIGNALS",
                "color": "warning",
                "strike": int(atm),
                "entry": "No clear institutional edge — wait",
                "sl": "—",
                "target": "—",
                "confidence": 45,
                "reasons": [
                    "OI change direction unclear",
                    "Wait for CE/PE writing confirmation",
                    "Let VIX settle before entry",
                    f"Watch max pain level: {int(mp)}",
                ],
                "max_pain": int(mp),
                "bias_score": f"Bull {bull_pct}% vs Bear {bear_pct}%",
            })

    # ── SUGGESTION 3: GAMMA / PREMIUM STRATEGY ──
    high_gamma   = atm_gamma > 0.001
    low_vix_env  = vix and vix < 14
    theta_heavy  = atm_theta < -30

    if high_gamma and not low_vix_env:
        # Straddle buy — explosive move expected
        sugg.append({
            "type": "STRADDLE BUY — GAMMA PLAY",
            "color": "gamma",
            "strike": int(atm),
            "entry": f"Buy {int(atm)} CE + {int(atm)} PE (cost ≈ ₹{atm_straddle:.0f})",
            "sl":    f"25% stop (exit below ₹{atm_straddle*0.75:.0f} combined)",
            "target": f"Exit at 40–80% gain (≥ ₹{atm_straddle*1.5:.0f})",
            "confidence": 65,
            "reasons": [
                f"High ATM gamma ({atm_gamma:.5f}) — intraday expansion imminent",
                "Premium will move fast on directional break",
                f"IV breakout likely — current skew: {iv_skew:.1f}%",
                "Best used near open or event catalyst",
            ],
            "max_pain": int(mp),
            "bias_score": f"Gamma play — direction agnostic",
        })
    elif low_vix_env and theta_heavy:
        # Short strangle — low vol, sell premium
        strangle_upper = strikes[min(atm_idx + 2, len(strikes)-1)]
        strangle_lower = strikes[max(atm_idx - 2, 0)]
        sugg.append({
            "type": "SHORT STRANGLE — SELL PREMIUM",
            "color": "neutral",
            "strike": int(atm),
            "entry": f"Sell {int(strangle_lower)} PE + {int(strangle_upper)} CE",
            "sl":    f"Spot closes outside {int(strangle_lower - step)}–{int(strangle_upper + step)}",
            "target": f"Collect full premium decay in {3 if vix and vix<12 else 5} days",
            "confidence": 74,
            "reasons": [
                f"VIX extremely low ({vix:.1f}) — ideal sell environment",
                f"ATM theta ₹{atm_theta:.0f}/day accelerating",
                "OI confirms range-bound structure",
                f"Max pain at {int(mp)} supports range",
            ],
            "max_pain": int(mp),
            "bias_score": f"Theta/Vega — vol seller",
        })
    else:
        # Bull/Bear spread based on direction
        if bull_score >= bear_score:
            spread_sell = strikes[min(atm_idx + 2, len(strikes)-1)]
            sugg.append({
                "type": "BULL CALL SPREAD",
                "color": "bullish",
                "strike": int(call_strike),
                "entry": f"Buy {int(call_strike)} CE, Sell {int(spread_sell)} CE",
                "sl":    f"Spot < {int(atm - step)} (debit lost)",
                "target": f"Full spread width at {int(spread_sell)} expiry",
                "confidence": min(80, 50 + bull_pct // 3),
                "reasons": [
                    "Defined risk bullish spread",
                    f"Reduces cost vs naked call",
                    f"Target: {int(spread_sell)} resistance",
                    bull_reasons[0] if bull_reasons else "PCR supports upside",
                ],
                "max_pain": int(mp),
                "bias_score": f"Bull {bull_pct}% vs Bear {bear_pct}%",
            })
        else:
            spread_sell = strikes[max(atm_idx - 2, 0)]
            sugg.append({
                "type": "BEAR PUT SPREAD",
                "color": "bearish",
                "strike": int(put_strike),
                "entry": f"Buy {int(put_strike)} PE, Sell {int(spread_sell)} PE",
                "sl":    f"Spot > {int(atm + step)} (debit lost)",
                "target": f"Full spread width at {int(spread_sell)} support",
                "confidence": min(80, 50 + bear_pct // 3),
                "reasons": [
                    "Defined risk bearish spread",
                    f"Reduces cost vs naked put",
                    f"Target: {int(spread_sell)} support",
                    bear_reasons[0] if bear_reasons else "PCR supports downside",
                ],
                "max_pain": int(mp),
                "bias_score": f"Bear {bear_pct}% vs Bull {bull_pct}%",
            })

    return sugg[:3]


def compute_greeks_summary(chain, spot):
    if not chain: return {}
    atm = min(chain, key=lambda x: abs(x["strike"]-spot))
    mgr = max(chain, key=lambda x: x["ce_gamma"])
    df  = pd.DataFrame(chain)
    return {
        "atm_strike":     atm["strike"],
        "atm_ce_delta":   atm["ce_delta"],
        "atm_pe_delta":   atm["pe_delta"],
        "atm_gamma":      atm["ce_gamma"],
        "atm_theta":      atm["ce_theta"],
        "atm_vega":       atm["ce_vega"],
        "max_gamma_strike": mgr["strike"],
        "max_gamma":      mgr["ce_gamma"],
        "total_ce_oi":    int(df["ce_oi"].sum()),
        "total_pe_oi":    int(df["pe_oi"].sum()),
        "total_ce_vol":   int(df["ce_vol"].sum()),
        "total_pe_vol":   int(df["pe_vol"].sum()),
        "avg_ce_iv":      round(df["ce_iv"].mean(), 2),
        "avg_pe_iv":      round(df["pe_iv"].mean(), 2),
    }


# ─────────────────────────────────────────────
# BACKGROUND THREAD
# ─────────────────────────────────────────────

def update_index(index_name):
    try:
        # 1. Fetch expiries if needed
        if not state["expiries"].get(index_name):
            exps = fetch_expiries(index_name)
            if exps:
                state["expiries"][index_name] = exps
                state["active_expiry"][index_name] = exps[0]  # nearest expiry
                logger.info(f"{index_name} expiries: {exps[:3]}")

        expiry = state["active_expiry"].get(index_name)
        if not expiry:
            logger.warning(f"No expiry for {index_name}")
            return

        # 2. Fetch raw chain
        raw = fetch_option_chain(index_name, expiry)
        if not raw:
            logger.warning(f"Empty chain for {index_name} {expiry}")
            return

        # 3. Parse rows
        chain = [parse_row(item) for item in raw if item.get("strike_price")]
        chain = sorted(chain, key=lambda x: x["strike"])

        if not chain:
            return

        # 4. Spot price (from chain data or LTP endpoint)
        spot = chain[0]["spot"] if chain[0]["spot"] > 0 else fetch_ltp(INSTRUMENT_KEYS[index_name])
        if spot:
            state["spot_price"][index_name] = spot

        state["option_chain"][index_name] = chain

        # 5. Analysis
        vix = state.get("vix")
        pcr_overall = sum(r["pe_oi"] for r in chain) / max(sum(r["ce_oi"] for r in chain), 1)
        sr          = detect_support_resistance(chain, spot)
        sentiment   = compute_sentiment(chain, spot, vix, pcr_overall)
        alerts      = generate_alerts(chain, spot, vix, pcr_overall, sr, sentiment)
        insights    = generate_insights(chain, spot, vix, pcr_overall, sr)
        trades      = generate_trade_suggestions(chain, spot, vix, pcr_overall, sr, sentiment)
        greeks      = compute_greeks_summary(chain, spot)

        state["support_resistance"][index_name] = sr
        state["sentiment"][index_name]          = sentiment
        state["alerts"][index_name]             = alerts
        state["insights"][index_name]           = insights
        state["trade_suggestions"][index_name]  = trades
        state["greeks_summary"][index_name]     = greeks

        logger.info(f"{index_name} updated: {len(chain)} strikes, spot={spot}")

    except Exception as e:
        logger.error(f"update_index({index_name}) error: {e}", exc_info=True)


def update_loop():
    while True:
        if state["token"]:
            try:
                # VIX
                v = fetch_ltp(VIX_KEY)
                if v: state["vix"] = round(v, 2)
            except Exception as e:
                logger.error(f"VIX error: {e}")

            for idx in ["NIFTY", "SENSEX"]:
                update_index(idx)

            state["last_update"] = datetime.now().isoformat()
            state["error"] = None
        time.sleep(UPDATE_INTERVAL)


threading.Thread(target=update_loop, daemon=True).start()


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/auth", methods=["POST"])
def auth():
    token = (request.json or {}).get("token", "").strip()
    if not token:
        return jsonify({"success": False, "error": "Token required"}), 400

    state["token"] = token
    try:
        r = requests.get(f"{UPSTOX_BASE}/user/profile", headers=hdrs(), timeout=8)
        d = r.json()
        if d.get("status") == "success":
            prof = d.get("data", {})
            return jsonify({"success": True, "user": prof.get("user_name","User")})
        state["token"] = None
        return jsonify({"success": False, "error": d.get("errors","Invalid token")}), 401
    except Exception as e:
        state["token"] = None
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/status")
def status():
    return jsonify({
        "connected": state["token"] is not None,
        "last_update": state["last_update"],
        "vix": state.get("vix"),
        "error": state.get("error"),
        "expiries": state.get("expiries", {}),
        "active_expiry": state.get("active_expiry", {}),
    })


@app.route("/api/chain/<index_name>")
def get_chain(index_name):
    idx = index_name.upper()
    if idx not in INSTRUMENT_KEYS:
        return jsonify({"error": "Invalid index"}), 400

    return jsonify({
        "index":              idx,
        "spot":               state["spot_price"].get(idx),
        "vix":                state.get("vix"),
        "expiry":             state["active_expiry"].get(idx),
        "expiries":           state["expiries"].get(idx, []),
        "chain":              state["option_chain"].get(idx, []),
        "support_resistance": state["support_resistance"].get(idx, {}),
        "sentiment":          state["sentiment"].get(idx, {}),
        "trade_suggestions":  state["trade_suggestions"].get(idx, []),
        "greeks_summary":     state["greeks_summary"].get(idx, {}),
        "insights":           state["insights"].get(idx, []),
        "alerts":             state.get("alerts", {}).get(idx, []),
        "last_update":        state["last_update"],
    })


@app.route("/api/set_expiry/<index_name>", methods=["POST"])
def set_expiry(index_name):
    idx = index_name.upper()
    expiry = (request.json or {}).get("expiry", "")
    if expiry and idx in INSTRUMENT_KEYS:
        state["active_expiry"][idx] = expiry
        # Clear cached chain to force fresh fetch
        state["option_chain"].pop(idx, None)
        return jsonify({"success": True, "expiry": expiry})
    return jsonify({"success": False}), 400


@app.route("/api/logout", methods=["POST"])
def logout():
    state["token"] = None
    state["option_chain"] = {}
    state["spot_price"] = {}
    state["expiries"] = {}
    state["active_expiry"] = {}
    state["alerts"] = {}
    return jsonify({"success": True})


# ─────────────────────────────────────────────
# AI CHAT ENDPOINT  (key lives only on server)
# ─────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

AI_SYSTEM_PROMPT = """You are OptionDesk AI — an elite institutional options trading analyst for Indian F&O markets, specialising in NIFTY and SENSEX index options.

You have access to real-time live option chain data, OI analysis, PCR, India VIX, support/resistance zones, greeks, and AI-generated trade suggestions.

Your role:
- Interpret the live market data and provide sharp, actionable insights
- Identify support/resistance strength (writing vs unwinding, institutional activity)
- Detect market structure: BOS, CHOCH, traps (bull trap, bear trap, fake breakout)
- Explain PCR signals: what options writers are doing, who is dominant
- Give clear trade recommendations with entry, SL, targets when asked
- Use SMC (Smart Money Concepts) language where relevant
- Always ground your analysis in the LIVE DATA provided — not generic advice

Style:
- Be concise, direct, institutional — no fluff
- Use bullet points for multi-part analysis
- Highlight key numbers and strikes
- Use ✅ for bullish signals, ❌ for bearish, ⚠️ for caution/trap, 🎯 for key levels
- Format trade setups clearly: Entry / SL / Target / Confidence
- Keep responses under 350 words unless analysis demands more"""


@app.route("/api/ai_chat", methods=["POST"])
def ai_chat():
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set on server. Add it to your environment and restart."}), 500

    body = request.json or {}
    user_message = body.get("message", "").strip()
    index_name   = body.get("index", "NIFTY").upper()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # ── Build live market context from server state ──
    idx = index_name if index_name in INSTRUMENT_KEYS else "NIFTY"
    chain     = state["option_chain"].get(idx, [])
    spot      = state["spot_price"].get(idx)
    vix       = state.get("vix")
    sentiment = state["sentiment"].get(idx, {})
    sr        = state["support_resistance"].get(idx, {})
    greeks    = state["greeks_summary"].get(idx, {})
    trades    = state["trade_suggestions"].get(idx, [])
    alerts    = state["alerts"].get(idx, [])
    insights  = state["insights"].get(idx, [])
    expiry    = state["active_expiry"].get(idx, "")

    def fmt(n):
        try:
            n = float(n)
            if abs(n) >= 1e7:  return f"{n/1e7:.1f}Cr"
            if abs(n) >= 1e5:  return f"{n/1e5:.1f}L"
            if abs(n) >= 1000: return f"{n/1000:.1f}K"
            return f"{n:.0f}"
        except Exception:
            return str(n)

    total_ce_oi = sum(r.get("ce_oi", 0) for r in chain)
    total_pe_oi = sum(r.get("pe_oi", 0) for r in chain)
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else None

    top_ce = sorted(chain, key=lambda x: x.get("ce_oi", 0), reverse=True)[:3]
    top_pe = sorted(chain, key=lambda x: x.get("pe_oi", 0), reverse=True)[:3]

    top_ce_str = ", ".join(f"{int(r['strike'])} CE (OI:{fmt(r['ce_oi'])}, ΔOI:{fmt(r['ce_chg_oi'])})" for r in top_ce) or "—"
    top_pe_str = ", ".join(f"{int(r['strike'])} PE (OI:{fmt(r['pe_oi'])}, ΔOI:{fmt(r['pe_chg_oi'])})" for r in top_pe) or "—"

    res_str = "; ".join(
        f"{int(z['strike'])} [{z['strength'].upper()}{'  WEAKENING' if z.get('weakening') else ' BUILDING' if z.get('strengthening') else ''}] OI:{fmt(z['oi'])}"
        for z in (sr.get("resistance") or [])[:4]
    ) or "None detected"

    sup_str = "; ".join(
        f"{int(z['strike'])} [{z['strength'].upper()}{'  WEAKENING' if z.get('weakening') else ' ABSORBING' if z.get('strengthening') else ''}] OI:{fmt(z['oi'])}"
        for z in (sr.get("support") or [])[:4]
    ) or "None detected"

    trade_str = "\n".join(
        f"{t.get('type','')}: Entry={t.get('entry','')}, SL={t.get('sl','')}, Target={t.get('target','')}, "
        f"Confidence={t.get('confidence','')}%, Reasons=[{'; '.join(t.get('reasons', []))}]"
        for t in trades[:2]
    ) or "No suggestions available"

    alert_str   = "\n".join(f"[{a.get('level','').upper()}] {a.get('msg','')}" for a in alerts[:5])  or "No alerts"
    insight_str = "\n".join(insights[:5]) or "No insights"

    market_ctx = f"""
=== LIVE {idx} OPTION CHAIN CONTEXT (as of {state.get('last_update','now')}) ===

SPOT PRICE: {spot}
INDIA VIX:  {vix}
EXPIRY:     {expiry}

SENTIMENT:
- Label: {sentiment.get('label','—')}
- Bullish: {sentiment.get('bullish',0)}% | Bearish: {sentiment.get('bearish',0)}% | Neutral: {sentiment.get('neutral',0)}%
- Market Score: {sentiment.get('score',50)}/100
- PCR (Overall): {pcr}

OI SUMMARY:
- Total CE OI: {fmt(total_ce_oi)} | Total PE OI: {fmt(total_pe_oi)}
- ATM Strike: {greeks.get('atm_strike','—')}
- ATM CE Delta: {greeks.get('atm_ce_delta','—')} | ATM PE Delta: {greeks.get('atm_pe_delta','—')}
- ATM Gamma: {greeks.get('atm_gamma','—')} | ATM Theta: {greeks.get('atm_theta','—')} | ATM Vega: {greeks.get('atm_vega','—')}
- Avg CE IV: {greeks.get('avg_ce_iv','—')}% | Avg PE IV: {greeks.get('avg_pe_iv','—')}%
- Max Gamma Strike: {greeks.get('max_gamma_strike','—')}
- Total CE Volume: {fmt(greeks.get('total_ce_vol',0))} | Total PE Volume: {fmt(greeks.get('total_pe_vol',0))}

TOP CE OI STRIKES (Resistance walls): {top_ce_str}
TOP PE OI STRIKES (Support walls):    {top_pe_str}

RESISTANCE ZONES: {res_str}
SUPPORT ZONES:    {sup_str}

TRADE SUGGESTIONS FROM ENGINE:
{trade_str}

LIVE ALERTS:
{alert_str}

AI INSIGHTS FROM ENGINE:
{insight_str}
=== END CONTEXT ===
""".strip()

    # ── Call Anthropic API from server ──
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 1000,
                "system": AI_SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Here is the current live market context:\n\n{market_ctx}\n\n---\n\nUser question: {user_message}"
                    }
                ],
            },
            timeout=30,
        )
        result = r.json()
        if r.status_code == 200:
            reply = "".join(b.get("text", "") for b in result.get("content", []))
            return jsonify({"reply": reply})
        else:
            err = result.get("error", {}).get("message", r.text)
            logger.error(f"Anthropic API error: {err}")
            return jsonify({"error": f"Anthropic error: {err}"}), 502
    except Exception as e:
        logger.error(f"ai_chat exception: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  OptionDesk Pro — NIFTY/SENSEX Option Chain Analyzer")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
