"""
Shared helpers for Vercel serverless functions.
Each function is stateless — token is passed via request header or body.
"""
import requests

UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENT_KEYS = {
    "NIFTY":  "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
}
VIX_KEY = "NSE_INDEX|India VIX"


def hdrs(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def api_get(token, path, params=None):
    try:
        r = requests.get(f"{UPSTOX_BASE}{path}", headers=hdrs(token), params=params, timeout=12)
        d = r.json()
        if d.get("status") == "success":
            return d.get("data")
    except Exception:
        pass
    return None


def fetch_ltp(token, instrument_key):
    data = api_get(token, "/market-quote/ltp", {"instrument_key": instrument_key})
    if data:
        key = list(data.keys())[0]
        return float(data[key].get("last_price", 0) or 0)
    return None


def parse_row(item):
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

    ce_oi   = safe(ce_md, "oi")
    pe_oi   = safe(pe_md, "oi")
    ce_prev = safe(ce_md, "prev_oi")
    pe_prev = safe(pe_md, "prev_oi")
    spot    = float(item.get("underlying_spot_price") or 0)
    strike  = float(item.get("strike_price") or 0)

    return {
        "strike": strike, "expiry": item.get("expiry", ""), "spot": spot,
        "ce_ltp": safe(ce_md, "ltp"), "ce_oi": ce_oi, "ce_prev_oi": ce_prev,
        "ce_chg_oi": ce_oi - ce_prev, "ce_vol": safe(ce_md, "volume"),
        "ce_delta": safe(ce_gr, "delta"), "ce_gamma": safe(ce_gr, "gamma"),
        "ce_theta": safe(ce_gr, "theta"), "ce_vega": safe(ce_gr, "vega"),
        "ce_iv": safe(ce_gr, "iv"),
        "pe_ltp": safe(pe_md, "ltp"), "pe_oi": pe_oi, "pe_prev_oi": pe_prev,
        "pe_chg_oi": pe_oi - pe_prev, "pe_vol": safe(pe_md, "volume"),
        "pe_delta": safe(pe_gr, "delta"), "pe_gamma": safe(pe_gr, "gamma"),
        "pe_theta": safe(pe_gr, "theta"), "pe_vega": safe(pe_gr, "vega"),
        "pe_iv": safe(pe_gr, "iv"),
        "pcr": round(pe_oi / ce_oi, 2) if ce_oi > 0 else 0,
        "distance": round(((strike - spot) / spot) * 100, 2) if spot > 0 else 0,
    }
