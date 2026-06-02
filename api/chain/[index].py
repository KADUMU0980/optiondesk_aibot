"""
/api/chain/[index] — GET
Stateless: reads token from Authorization header or ?token= query param.
Fetches live option chain, computes analysis, returns full JSON payload.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import statistics
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENT_KEYS = {
    "NIFTY":  "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
}
VIX_KEY = "NSE_INDEX|India VIX"


# ── helpers ──────────────────────────────────────────

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


def fetch_expiries(token, idx):
    d = _get(token, "/option/contract", {"instrument_key": INSTRUMENT_KEYS[idx]})
    if d:
        today = datetime.now().date()
        return sorted(set(
            item["expiry"] for item in d
            if item.get("expiry") and item["expiry"] >= str(today)
        ))
    return []


def parse_row(item):
    def safe(d, *keys, default=0):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
        return float(d) if d is not None else default

    ce = item.get("call_options", {}); pe = item.get("put_options", {})
    cm = ce.get("market_data", {}); pm = pe.get("market_data", {})
    cg = ce.get("option_greeks", {}); pg = pe.get("option_greeks", {})
    co = safe(cm, "oi"); po = safe(pm, "oi")
    cp = safe(cm, "prev_oi"); pp = safe(pm, "prev_oi")
    spot = float(item.get("underlying_spot_price") or 0)
    strike = float(item.get("strike_price") or 0)
    return {
        "strike": strike, "expiry": item.get("expiry",""), "spot": spot,
        "ce_ltp": safe(cm,"ltp"), "ce_oi": co, "ce_prev_oi": cp,
        "ce_chg_oi": co-cp, "ce_vol": safe(cm,"volume"),
        "ce_delta": safe(cg,"delta"), "ce_gamma": safe(cg,"gamma"),
        "ce_theta": safe(cg,"theta"), "ce_vega": safe(cg,"vega"), "ce_iv": safe(cg,"iv"),
        "pe_ltp": safe(pm,"ltp"), "pe_oi": po, "pe_prev_oi": pp,
        "pe_chg_oi": po-pp, "pe_vol": safe(pm,"volume"),
        "pe_delta": safe(pg,"delta"), "pe_gamma": safe(pg,"gamma"),
        "pe_theta": safe(pg,"theta"), "pe_vega": safe(pg,"vega"), "pe_iv": safe(pg,"iv"),
        "pcr": round(po/co,2) if co>0 else 0,
        "distance": round(((strike-spot)/spot)*100,2) if spot>0 else 0,
    }


def detect_sr(chain, spot):
    if not chain: return {"support":[],"resistance":[]}
    df = pd.DataFrame(chain)
    ceq = df["ce_oi"].quantile(0.75) if len(df)>4 else 0
    peq = df["pe_oi"].quantile(0.75) if len(df)>4 else 0
    res,sup = [],[]
    for r in chain:
        s = r["strike"]
        if s>spot and r["ce_oi"]>0:
            st = "hard" if r["ce_oi"]>=ceq else "soft"
            res.append({"strike":s,"oi":int(r["ce_oi"]),"chg_oi":int(r["ce_chg_oi"]),"vol":int(r["ce_vol"]),
                        "strength":st,"weakening":bool(r["ce_chg_oi"]<-r["ce_oi"]*0.03),
                        "strengthening":bool(r["ce_chg_oi"]>r["ce_oi"]*0.03)})
        if s<spot and r["pe_oi"]>0:
            st = "hard" if r["pe_oi"]>=peq else "soft"
            sup.append({"strike":s,"oi":int(r["pe_oi"]),"chg_oi":int(r["pe_chg_oi"]),"vol":int(r["pe_vol"]),
                        "strength":st,"weakening":bool(r["pe_chg_oi"]<-r["pe_oi"]*0.03),
                        "strengthening":bool(r["pe_chg_oi"]>r["pe_oi"]*0.03)})
    return {
        "resistance": sorted(res,key=lambda x:x["oi"],reverse=True)[:6],
        "support":    sorted(sup,key=lambda x:x["oi"],reverse=True)[:6],
    }


def compute_sentiment(chain, spot, vix, pcr):
    if not chain:
        return {"bullish":33,"bearish":33,"neutral":34,"label":"NEUTRAL","score":50,"pcr":pcr}
    sig=[]
    if pcr>1.4: sig+=[ ("bullish",3)]
    elif pcr>1.1: sig+=[("bullish",1)]
    elif pcr<0.7: sig+=[("bearish",3)]
    elif pcr<0.9: sig+=[("bearish",1)]
    else: sig+=[("neutral",1)]
    ce_chg=sum(r["ce_chg_oi"] for r in chain)
    pe_chg=sum(r["pe_chg_oi"] for r in chain)
    if pe_chg>0 and ce_chg<=0: sig+=[("bullish",2)]
    elif ce_chg>0 and pe_chg<=0: sig+=[("bearish",2)]
    else: sig+=[("neutral",1)]
    if vix:
        if vix<14: sig+=[("bullish",1)]
        elif vix>22: sig+=[("bearish",2)]
        elif vix>18: sig+=[("bearish",1)]
    atm=min(chain,key=lambda x:abs(x["strike"]-spot))
    nd=atm["ce_delta"]+atm["pe_delta"]
    if nd>0.1: sig+=[("bullish",1)]
    elif nd<-0.1: sig+=[("bearish",1)]
    bull=sum(w for d,w in sig if d=="bullish")
    bear=sum(w for d,w in sig if d=="bearish")
    neut=sum(w for d,w in sig if d=="neutral")
    tot=max(bull+bear+neut,1)
    bp=round(bull/tot*100); brp=round(bear/tot*100)
    sc=round(50+(bp-brp)/2)
    lbl="BULLISH" if sc>65 else ("BEARISH" if sc<35 else "NEUTRAL")
    return {"bullish":bp,"bearish":brp,"neutral":100-bp-brp,"label":lbl,"score":sc,"pcr":round(pcr,2)}


def generate_alerts(chain, spot, vix, pcr, sr, sentiment):
    alerts=[]; now=datetime.now().strftime("%H:%M:%S")
    def add(msg,lv="info"): alerts.append({"msg":msg,"level":lv,"time":now})
    if pcr>1.5: add("🟢 STRONG BULLISH: PCR extremely high","success")
    elif pcr>1.2: add("📈 PCR turning bullish","success")
    elif pcr<0.7: add("🔴 WARNING: PCR turning bearish","danger")
    elif pcr<0.9: add("⚠️ PCR weakening — Monitor for reversal","warning")
    if vix:
        if vix>22: add(f"⚡ VOLATILITY SPIKE — VIX at {vix:.1f}","danger")
        elif vix<12: add(f"🔇 LOW VOLATILITY — VIX {vix:.1f}","info")
        elif 14<=vix<=17: add(f"✅ VIX supportive for trends — {vix:.1f}","success")
    for r in (sr.get("resistance") or [])[:2]:
        if r["weakening"]: add(f"⚠️ {int(r['strike'])} resistance WEAKENING","warning")
        elif r["strength"]=="hard": add(f"🔴 HARD RESISTANCE at {int(r['strike'])} — {r['oi']//1000}K OI","danger")
    for s in (sr.get("support") or [])[:2]:
        if s["weakening"]: add(f"⚠️ {int(s['strike'])} support WEAKENING","warning")
        elif s["strength"]=="hard": add(f"🟢 STRONG SUPPORT at {int(s['strike'])} — {s['oi']//1000}K OI","success")
    ce_chg=sum(r["ce_chg_oi"] for r in chain)
    pe_chg=sum(r["pe_chg_oi"] for r in chain)
    if pe_chg>0 and ce_chg<0: add("📊 Put writing increasing — Call unwinding detected","success")
    elif ce_chg>0 and pe_chg<0: add("📊 Call writing increasing — Bearish pressure building","warning")
    atm=min(chain,key=lambda x:abs(x["strike"]-spot))
    if atm["ce_gamma"]>0.002: add(f"⚡ GAMMA EXPANSION at {int(atm['strike'])} ATM — Large move imminent","warning")
    if sentiment["label"]=="BULLISH" and sentiment["score"]>70:
        add("🚀 BREAKOUT POSSIBILITY HIGH — Institutional bull positioning","success")
    elif sentiment["label"]=="BEARISH" and sentiment["score"]<30:
        add("🔴 BEARISH PRESSURE BUILDING — Short accumulation detected","danger")
    return alerts[:8]


def generate_insights(chain, spot, vix, pcr, sr):
    ins=[]
    if not chain: return ins
    for r in (sr.get("resistance") or [])[:2]:
        s=int(r["strike"])
        if r["strengthening"]: ins.append(f"Strong resistance forming at {s} — heavy call writing")
        elif r["weakening"]: ins.append(f"Resistance weakening at {s} — possible breakout loading")
        else: ins.append(f"Resistance zone active at {s} — {r['oi']//1000}K CE OI")
    for s in (sr.get("support") or [])[:2]:
        st=int(s["strike"])
        if s["strengthening"]: ins.append(f"Support strengthening at {st} — put writers absorbing")
        elif s["weakening"]: ins.append(f"Weak support likely to fail at {st}")
        else: ins.append(f"Support holding at {st} — {s['oi']//1000}K PE OI")
    if pcr>1.3: ins.append("Put writers aggressively defending downside")
    elif pcr<0.8: ins.append("Call writers dominant — bearish bias intact")
    if vix:
        if vix<14: ins.append("Volatility compression — range-bound market likely")
        elif vix>20: ins.append("Momentum expansion likely — high volatility regime")
        else: ins.append("VIX supportive for directional moves")
    atm=min(chain,key=lambda x:abs(x["strike"]-spot))
    if atm["ce_gamma"]>0.002: ins.append("Gamma expansion near ATM — intraday range may widen")
    if atm["ce_theta"]<-50: ins.append("Theta decay dominant — option sellers advantaged")
    total_vol=sum(r["ce_vol"]+r["pe_vol"] for r in chain)
    if total_vol>5_000_000: ins.append("High volume detected — institutional activity elevated")
    return ins[:8]


def generate_trades(chain, spot, vix, pcr, sr, sentiment):
    sugg=[]
    if not chain or spot<=0: return sugg
    strikes=sorted(set(r["strike"] for r in chain))
    if len(strikes)<4: return sugg
    step=strikes[1]-strikes[0]
    atm_row=min(chain,key=lambda x:abs(x["strike"]-spot))
    atm=atm_row["strike"]
    def max_pain(chain):
        pain={}
        for exp_s in strikes:
            t=sum(max(0,exp_s-r["strike"])*r["ce_oi"]+max(0,r["strike"]-exp_s)*r["pe_oi"] for r in chain)
            pain[exp_s]=t
        return min(pain,key=pain.get)
    mp=max_pain(chain)
    total_ce_oi=sum(r["ce_oi"] for r in chain); total_pe_oi=sum(r["pe_oi"] for r in chain)
    total_ce_vol=sum(r["ce_vol"] for r in chain); total_pe_vol=sum(r["pe_vol"] for r in chain)
    otm_ce_wr=sum(r["ce_chg_oi"] for r in chain if r["strike"]>spot and r["ce_chg_oi"]>0)
    otm_pe_wr=sum(r["pe_chg_oi"] for r in chain if r["strike"]<spot and r["pe_chg_oi"]>0)
    ce_unw=sum(abs(r["ce_chg_oi"]) for r in chain if r["strike"]>spot and r["ce_chg_oi"]<0)
    pe_unw=sum(abs(r["pe_chg_oi"]) for r in chain if r["strike"]<spot and r["pe_chg_oi"]<0)
    atm_idx=strikes.index(atm) if atm in strikes else len(strikes)//2
    call_s=strikes[atm_idx+1] if atm_idx+1<len(strikes) else int(atm+step)
    put_s=strikes[atm_idx-1] if atm_idx-1>=0 else int(atm-step)
    res_list=sr.get("resistance",[]); sup_list=sr.get("support",[])
    near_res=res_list[0]["strike"] if res_list else spot+step*4
    near_sup=sup_list[0]["strike"] if sup_list else spot-step*4
    bull_score=0; bear_score=0; bull_r=[]; bear_r=[]
    if pcr>1.5: bull_score+=3; bull_r.append(f"PCR extremely bullish ({pcr:.2f})")
    elif pcr>1.2: bull_score+=2; bull_r.append(f"PCR bullish ({pcr:.2f})")
    elif pcr<0.6: bear_score+=3; bear_r.append(f"PCR extremely bearish ({pcr:.2f})")
    elif pcr<0.8: bear_score+=2; bear_r.append(f"PCR bearish ({pcr:.2f})")
    if otm_pe_wr>otm_ce_wr*1.3: bull_score+=2; bull_r.append("PE writing below spot — support building")
    elif otm_ce_wr>otm_pe_wr*1.3: bear_score+=2; bear_r.append("CE writing above spot — resistance building")
    if ce_unw>total_ce_oi*0.02: bull_score+=2; bull_r.append("CE short covering — bears exiting")
    if pe_unw>total_pe_oi*0.02: bear_score+=2; bear_r.append("PE unwinding — support eroding")
    if total_pe_vol>total_ce_vol*1.3: bull_score+=1; bull_r.append("Put volume elevated — hedging")
    elif total_ce_vol>total_pe_vol*1.3: bear_score+=1; bear_r.append("Call volume elevated — bearish spec")
    if vix:
        if vix<14: bull_score+=1; bull_r.append(f"VIX low ({vix:.1f})")
        elif vix<18: bull_score+=1; bull_r.append(f"VIX normal ({vix:.1f})")
        elif vix>22: bear_score+=2; bear_r.append(f"VIX elevated ({vix:.1f})")
    tot=bull_score+bear_score; bp=round(bull_score/max(tot,1)*100); brp=round(bear_score/max(tot,1)*100)
    atm_ltp=atm_row["ce_ltp"]+atm_row["pe_ltp"]
    if bull_score>=bear_score:
        conf=min(92,45+bp//2+(5 if pcr>1.3 else 0))
        sugg.append({"type":"CALL BUY","color":"bullish","strike":int(call_s),
            "entry":f"Buy {int(call_s)} CE above {int(spot)}","sl":f"Close below {int(near_sup)}",
            "target":f"{int(near_res)} (CE wall)","confidence":conf,
            "reasons":bull_r[:4] or ["PCR bullish","OI supports upside"],
            "max_pain":int(mp),"bias_score":f"Bull {bp}% vs Bear {brp}%"})
    else:
        conf=min(92,45+brp//2+(5 if pcr<0.8 else 0))
        sugg.append({"type":"PUT BUY","color":"bearish","strike":int(put_s),
            "entry":f"Buy {int(put_s)} PE below {int(spot)}","sl":f"Close above {int(near_res)}",
            "target":f"{int(near_sup)} (PE wall)","confidence":conf,
            "reasons":bear_r[:4] or ["PCR bearish","OI supports downside"],
            "max_pain":int(mp),"bias_score":f"Bear {brp}% vs Bull {bp}%"})
    if bull_score>=bear_score and ce_unw>0 and otm_pe_wr>0:
        sugg.append({"type":"INTRADAY SCALP — LONG","color":"bullish","strike":int(call_s),
            "entry":f"Buy {int(call_s)} CE on dip near {int(atm)}","sl":f"Spot breaks below {int(atm-step)}",
            "target":f"{int(near_res)} intraday","confidence":min(85,55+bull_score*3),
            "reasons":["CE short covering","PE writing confirms support","Momentum buy",f"Max pain at {int(mp)}"],
            "max_pain":int(mp),"bias_score":f"Bull {bp}% vs Bear {brp}%"})
    else:
        upper=int(near_res); lower=int(near_sup)
        sugg.append({"type":"IRON CONDOR — RANGE","color":"neutral","strike":int(atm),
            "entry":f"Sell {lower} PE + {upper} CE","sl":f"Spot < {lower-int(step)} or > {upper+int(step)}",
            "target":f"Keep {int(atm_ltp*0.35):.0f}+ pts premium","confidence":68,
            "reasons":[f"OI range {lower}–{upper}","Balanced OI","Theta decay",f"Max pain {int(mp)}"],
            "max_pain":int(mp),"bias_score":f"Bull {bp}% vs Bear {brp}%"})
    spread_sell=strikes[min(atm_idx+2,len(strikes)-1)]
    sugg.append({"type":"BULL CALL SPREAD" if bull_score>=bear_score else "BEAR PUT SPREAD",
        "color":"bullish" if bull_score>=bear_score else "bearish",
        "strike":int(call_s) if bull_score>=bear_score else int(put_s),
        "entry":f"Buy {int(call_s)} CE, Sell {int(spread_sell)} CE" if bull_score>=bear_score else f"Buy {int(put_s)} PE",
        "sl":f"Spot < {int(atm-step)}","target":f"Full spread at {int(spread_sell)}",
        "confidence":min(80,50+bp//3),"reasons":["Defined risk spread","Reduces cost",
        f"Target {int(spread_sell)}",bull_r[0] if bull_r else "OI supports direction"],
        "max_pain":int(mp),"bias_score":f"Bull {bp}% vs Bear {brp}%"})
    return sugg[:3]


def compute_greeks(chain, spot):
    if not chain: return {}
    atm=min(chain,key=lambda x:abs(x["strike"]-spot))
    mgr=max(chain,key=lambda x:x["ce_gamma"])
    df=pd.DataFrame(chain)
    return {"atm_strike":atm["strike"],"atm_ce_delta":atm["ce_delta"],"atm_pe_delta":atm["pe_delta"],
            "atm_gamma":atm["ce_gamma"],"atm_theta":atm["ce_theta"],"atm_vega":atm["ce_vega"],
            "max_gamma_strike":mgr["strike"],"max_gamma":mgr["ce_gamma"],
            "total_ce_oi":int(df["ce_oi"].sum()),"total_pe_oi":int(df["pe_oi"].sum()),
            "total_ce_vol":int(df["ce_vol"].sum()),"total_pe_vol":int(df["pe_vol"].sum()),
            "avg_ce_iv":round(df["ce_iv"].mean(),2),"avg_pe_iv":round(df["pe_iv"].mean(),2)}


# ── Route ────────────────────────────────────────────

@app.route("/api/chain/<index_name>", methods=["GET", "OPTIONS"])
def get_chain(index_name):
    if request.method == "OPTIONS":
        return jsonify({}), 200

    idx = index_name.upper()
    if idx not in INSTRUMENT_KEYS:
        return jsonify({"error": "Invalid index"}), 400

    # Get token from Authorization header (set by frontend)
    token = request.headers.get("X-Upstox-Token", "").strip()
    if not token:
        token = request.args.get("token", "").strip()
    if not token:
        return jsonify({"error": "Token required — please log in again"}), 401

    try:
        # Expiries
        expiries = fetch_expiries(token, idx)
        if not expiries:
            return jsonify({"index": idx, "chain": [], "expiries": [], "error": "Could not fetch expiries"}), 200

        expiry = expiries[0]

        # Client may specify a different expiry via query param
        client_expiry = request.args.get("expiry", "").strip()
        if client_expiry and client_expiry in expiries:
            expiry = client_expiry

        # Spot + VIX
        spot = fetch_ltp(token, INSTRUMENT_KEYS[idx]) or 0
        vix_val = fetch_ltp(token, VIX_KEY)
        vix = round(vix_val, 2) if vix_val else None

        # Option chain
        raw = _get(token, "/option/chain", {"instrument_key": INSTRUMENT_KEYS[idx], "expiry_date": expiry}) or []
        chain = sorted([parse_row(item) for item in raw if item.get("strike_price")], key=lambda x: x["strike"])

        if chain and chain[0]["spot"] > 0:
            spot = chain[0]["spot"]

        # Analysis
        pcr = sum(r["pe_oi"] for r in chain) / max(sum(r["ce_oi"] for r in chain), 1) if chain else 1
        sr = detect_sr(chain, spot)
        sent = compute_sentiment(chain, spot, vix, pcr)
        alerts = generate_alerts(chain, spot, vix, pcr, sr, sent)
        insights = generate_insights(chain, spot, vix, pcr, sr)
        trades = generate_trades(chain, spot, vix, pcr, sr, sent)
        greeks = compute_greeks(chain, spot)

        return jsonify({
            "index": idx, "spot": spot, "vix": vix,
            "expiry": expiry, "expiries": expiries[:5],
            "chain": chain,
            "support_resistance": sr, "sentiment": sent,
            "trade_suggestions": trades, "greeks_summary": greeks,
            "insights": insights, "alerts": alerts,
            "last_update": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e), "chain": [], "expiries": []}), 500


handler = app
