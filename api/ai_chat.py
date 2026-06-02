"""
/api/ai_chat — POST
Stateless AI chat using Anthropic API. Token sent via X-Upstox-Token header.
Market data is re-fetched server-side for the AI context.
Requires ANTHROPIC_API_KEY env var set in Vercel project settings.
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENT_KEYS = {"NIFTY": "NSE_INDEX|Nifty 50", "SENSEX": "BSE_INDEX|SENSEX"}
VIX_KEY = "NSE_INDEX|India VIX"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

AI_SYSTEM_PROMPT = """You are OptionDesk AI — an elite institutional options trading analyst for Indian F&O markets, specialising in NIFTY and SENSEX index options.

You have access to real-time live option chain data, OI analysis, PCR, India VIX, support/resistance zones, greeks, and AI-generated trade suggestions.

Your role:
- Interpret the live market data and provide sharp, actionable insights
- Identify support/resistance strength (writing vs unwinding, institutional activity)
- Detect market structure: BOS, CHOCH, traps (bull trap, bear trap, fake breakout)
- Give clear trade recommendations with entry, SL, targets when asked
- Always ground your analysis in the LIVE DATA provided

Style:
- Be concise, direct, institutional — no fluff
- Use bullet points for multi-part analysis
- Use ✅ for bullish, ❌ for bearish, ⚠️ for caution, 🎯 for key levels
- Keep responses under 350 words unless analysis demands more"""


def _hdrs(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get(token, path, params=None):
    try:
        r = requests.get(f"{UPSTOX_BASE}{path}", headers=_hdrs(token), params=params, timeout=12)
        d = r.json()
        if d.get("status") == "success":
            return d.get("data")
    except Exception:
        pass
    return None


def fetch_ltp(token, key):
    d = _get(token, "/market-quote/ltp", {"instrument_key": key})
    if d:
        k = list(d.keys())[0]
        return float(d[k].get("last_price", 0) or 0)
    return None


def fmt(n):
    try:
        n = float(n)
        if abs(n) >= 1e7: return f"{n/1e7:.1f}Cr"
        if abs(n) >= 1e5: return f"{n/1e5:.1f}L"
        if abs(n) >= 1000: return f"{n/1000:.1f}K"
        return f"{n:.0f}"
    except Exception:
        return str(n)


@app.route("/api/ai_chat", methods=["POST", "OPTIONS"])
def ai_chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured on server. Add it in Vercel project settings → Environment Variables."}), 500

    body = request.get_json(silent=True) or {}
    user_message = body.get("message", "").strip()
    index_name = body.get("index", "NIFTY").upper()
    # Accept market snapshot from client (sent by frontend along with the message)
    snapshot = body.get("snapshot", {})

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Build context from client-provided snapshot (avoids re-fetching on Vercel)
    spot = snapshot.get("spot", "—")
    vix = snapshot.get("vix", "—")
    expiry = snapshot.get("expiry", "—")
    sentiment = snapshot.get("sentiment", {})
    sr = snapshot.get("support_resistance", {})
    greeks = snapshot.get("greeks_summary", {})
    trades = snapshot.get("trade_suggestions", [])
    alerts = snapshot.get("alerts", [])
    insights = snapshot.get("insights", [])
    chain = snapshot.get("chain", [])

    total_ce_oi = sum(r.get("ce_oi", 0) for r in chain)
    total_pe_oi = sum(r.get("pe_oi", 0) for r in chain)
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else None

    top_ce = sorted(chain, key=lambda x: x.get("ce_oi", 0), reverse=True)[:3]
    top_pe = sorted(chain, key=lambda x: x.get("pe_oi", 0), reverse=True)[:3]
    top_ce_str = ", ".join(f"{int(r['strike'])} CE (OI:{fmt(r['ce_oi'])}, ΔOI:{fmt(r['ce_chg_oi'])})" for r in top_ce) or "—"
    top_pe_str = ", ".join(f"{int(r['strike'])} PE (OI:{fmt(r['pe_oi'])}, ΔOI:{fmt(r['pe_chg_oi'])})" for r in top_pe) or "—"

    res_str = "; ".join(f"{int(z['strike'])} [{z['strength'].upper()}] OI:{fmt(z['oi'])}" for z in (sr.get("resistance") or [])[:4]) or "None"
    sup_str = "; ".join(f"{int(z['strike'])} [{z['strength'].upper()}] OI:{fmt(z['oi'])}" for z in (sr.get("support") or [])[:4]) or "None"
    trade_str = "\n".join(f"{t.get('type','')}: {t.get('entry','')} | SL:{t.get('sl','')} | Target:{t.get('target','')} | {t.get('confidence','')}%" for t in trades[:2]) or "No suggestions"
    alert_str = "\n".join(f"[{a.get('level','').upper()}] {a.get('msg','')}" for a in alerts[:5]) or "No alerts"
    insight_str = "\n".join(insights[:5]) or "No insights"

    market_ctx = f"""=== LIVE {index_name} OPTION CHAIN CONTEXT ===
SPOT: {spot} | VIX: {vix} | EXPIRY: {expiry}
SENTIMENT: {sentiment.get('label','—')} | Bull:{sentiment.get('bullish',0)}% Bear:{sentiment.get('bearish',0)}% | Score:{sentiment.get('score',50)}/100 | PCR:{pcr}
OI: CE={fmt(total_ce_oi)} PE={fmt(total_pe_oi)} | ATM:{greeks.get('atm_strike','—')} | Gamma:{greeks.get('atm_gamma','—')} | Theta:{greeks.get('atm_theta','—')}
TOP CE WALLS: {top_ce_str}
TOP PE WALLS: {top_pe_str}
RESISTANCE: {res_str}
SUPPORT: {sup_str}
TRADE SETUPS:\n{trade_str}
ALERTS:\n{alert_str}
INSIGHTS:\n{insight_str}
=== END ==="""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 1000,
                "system": AI_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": f"Live market context:\n\n{market_ctx}\n\n---\n\nUser: {user_message}"}],
            },
            timeout=30,
        )
        result = r.json()
        if r.status_code == 200:
            reply = "".join(b.get("text", "") for b in result.get("content", []))
            return jsonify({"reply": reply})
        err = result.get("error", {}).get("message", r.text)
        return jsonify({"error": f"Anthropic error: {err}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


handler = app
