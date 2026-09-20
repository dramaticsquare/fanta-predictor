#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FANTA PREDICTOR - Serie A
=========================
Stima per ogni giocatore della tua rosa un "fantavoto" atteso e un voto 1-10
per la prossima giornata, in base a:
  - xG, xA, xGBuildup del giocatore (Understat), pesati sulle partite recenti
  - forza offensiva/difensiva dell'avversario (xG fatti/subiti, con casa/trasferta)
  - probabilita' di clean sheet (difensori/portieri)
  - probabilita' di giocare (minuti nelle ultime partite) e cartellini
Poi sceglie il modulo e la formazione migliore.

Uso:
  pip install -r requirements.txt
  python fanta_predictor.py                 # usa rosa.csv, scrive docs/index.html
  python fanta_predictor.py --check         # verifica solo il riconoscimento dei nomi
  python fanta_predictor.py --refresh       # ignora la cache e riscarica tutto
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ============================================================== CONFIG ======
# Tutti i pesi sono modificabili. Quelli marcati [EURISTICA] NON sono calibrati
# su fantavoti reali: sono scelte ragionevoli da tarare con un backtest.
CFG = {
    "league": "Serie_A",
    # regolamento fantacalcio classico
    "base_vote": 6.0,
    "goal": 3.0,
    "assist": 1.0,
    "conceded_gk": -1.0,                              # per gol subito dal portiere
    "yellow": -0.5,
    "red": -1.0,
    "clean_sheet": {"P": 0.5, "D": 0.5, "C": 0.0, "A": 0.0},   # [EURISTICA]
    "involvement_weight": 0.10,                       # [EURISTICA] bonus per xGBuildup (D, C)
    # statistica
    "player_half_life": 10,      # partite: mezza vita del peso sulle partite del giocatore
    "team_half_life": 8,         # partite: idem per le squadre
    "player_shrink_90s": 6.0,    # "partite virtuali" di media di ruolo mescolate al giocatore
    "team_shrink_matches": 4.0,  # idem per le squadre (verso la media di lega)
    "avail_window": 6,           # ultime N partite di squadra per stimare la titolarita'
    "avail_half_life": 3,
    "starter_minutes": 60,
    "sub_weight": 0.4,           # quanto conta una presenza da subentrato
    "min_play_prob": 0.25,       # sotto questa soglia il giocatore non entra in formazione (se ce ne sono altri)
    # voto previsto = center + slope * (fantavoto - fantavoto di un giocatore medio del ruolo in partita neutra)
    # center 6.0: come i voti veri del fantacalcio (nei tuoi screenshot la media dei voti e' ~6,1)
    "rating_center": 6.0,
    "rating_slope": 1.5,
    # quote dei bookmaker (The Odds API): peso dei gol attesi impliciti nelle quote rispetto al solo xG
    "odds_weight": 0.6,          # [EURISTICA]
    "odds_cache_hours": 3,       # risparmia crediti del piano gratuito
    # calibrazione sui tuoi voti reali
    "calib_min_obs": 5,          # osservazioni minime per ruolo prima di applicare una correzione
    "calib_shrink": 20,          # piu' alto = correzione piu' prudente
    "tz": "Europe/Rome",
    "cache_hours": 12,
}

MODULES = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]

# nome squadra come lo scrivi in rosa.csv -> nome Understat
TEAM_ALIAS = {
    "milan": "AC Milan",
    "parma": "Parma Calcio 1913",
    "hellas verona": "Verona",
    "verona": "Verona",
    "ac milan": "AC Milan",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "as roma": "Roma",
}

POS_MAP = {"GK": "P", "D": "D", "M": "C", "F": "A"}


# ============================================================== UTILS =======
def strip_accents(s: str) -> str:
    s = s.translate(str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
                                   "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                                   "ß": "ss", "ı": "i"}))
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z ]", " ", strip_accents(str(s)).lower()).split()


def num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def decay(n: int, half_life: float) -> np.ndarray:
    """Pesi per n osservazioni in ordine cronologico: l'ultima pesa 1."""
    return 0.5 ** (np.arange(n)[::-1] / half_life)


# ============================================================== DATI ========
class UnderstatProvider:
    """Scarica i dati da Understat (via understatapi) con cache su disco."""

    def __init__(self, cache_dir=".cache_fanta", hours=12, refresh=False):
        from understatapi import UnderstatClient  # import qui: i test non lo richiedono
        self.client = UnderstatClient()
        self.dir = Path(cache_dir)
        self.dir.mkdir(exist_ok=True)
        self.hours = hours
        self.refresh = refresh

    def _get(self, key, fn):
        f = self.dir / f"{key}.json"
        fresh = f.exists() and (time.time() - f.stat().st_mtime) < self.hours * 3600
        if fresh and not self.refresh:
            return json.loads(f.read_text(encoding="utf-8"))
        err = None
        for i in range(3):
            try:
                data = fn()
                f.write_text(json.dumps(data), encoding="utf-8")
                time.sleep(0.4)  # gentilezza verso il server
                return data
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 * (i + 1))
        if f.exists():
            print(f"  ! rete non disponibile per '{key}': uso la cache vecchia", file=sys.stderr)
            return json.loads(f.read_text(encoding="utf-8"))
        raise RuntimeError(f"Impossibile scaricare '{key}': {err}")

    def league(self):
        return self.client.league(league=CFG["league"])

    def players(self, season):
        return self._get(f"players_{season}", lambda: self.league().get_player_data(season=str(season)))

    def matches(self, season):
        return self._get(f"matches_{season}", lambda: self.league().get_match_data(season=str(season)))

    def teams(self, season):
        return self._get(f"teams_{season}", lambda: self.league().get_team_data(season=str(season)))

    def player_matches(self, pid):
        return self._get(f"pm_{pid}", lambda: self.client.player(player=str(pid)).get_match_data())


# ============================================================== SQUADRE =====
def build_team_history(prov, season, cutoff=None):
    """dict titolo -> DataFrame (date, h_a, xG, xGA) ordinato per data, ultime 2 stagioni.
    cutoff (datetime): per il backtest, tiene solo le partite PRECEDENTI."""
    frames, current = {}, set()
    for s in (season - 1, season):
        try:
            teams = prov.teams(s)
        except Exception as e:  # noqa: BLE001
            print(f"  ! stagione {s} non disponibile ({e})", file=sys.stderr)
            continue
        if s == season:
            current = {t["title"] for t in teams.values()}
        for t in teams.values():
            df = pd.DataFrame(t["history"])
            if df.empty:
                continue
            df = num(df, ["xG", "xGA"])
            df["date"] = pd.to_datetime(df["date"])
            if cutoff is not None:
                df = df[df["date"] < cutoff]
            if df.empty:
                continue
            frames.setdefault(t["title"], []).append(df[["date", "h_a", "xG", "xGA"]])
    return {k: pd.concat(v).sort_values("date").reset_index(drop=True)
            for k, v in frames.items() if not current or k in current}


def team_strengths(hist):
    allm = pd.concat(hist.values())
    lg = allm["xG"].mean()
    hf = allm.loc[allm.h_a == "h", "xG"].mean() / lg   # xG casa / media
    af = allm.loc[allm.h_a == "a", "xG"].mean() / lg   # xG trasferta / media
    k = CFG["team_shrink_matches"]
    rows = {}
    for team, df in hist.items():
        w = decay(len(df), CFG["team_half_life"])
        ws = w.sum()
        xg = ((w * df.xG).sum() + k * lg) / (ws + k)
        xga = ((w * df.xGA).sum() + k * lg) / (ws + k)
        rows[team] = {"A": xg / lg, "D": xga / lg, "n": len(df)}   # D>1 = difesa perforabile
    return rows, lg, hf, af


def team_ctx(team, opp, is_home, S, lg, hf, af, odds=None):
    a_own, d_own = (S.get(team) or {"A": 1, "D": 1})["A"], (S.get(team) or {"A": 1, "D": 1})["D"]
    a_opp, d_opp = (S.get(opp) or {"A": 1, "D": 1})["A"], (S.get(opp) or {"A": 1, "D": 1})["D"]
    venue_att = hf if is_home else af
    venue_def = af if is_home else hf            # chi subisce in casa incassa "xG trasferta"
    att_factor = d_opp * venue_att               # moltiplica le stat offensive del giocatore
    lam_for = lg * a_own * d_opp * venue_att     # xG attesi della sua squadra
    lam_conc = lg * a_opp * d_own * venue_def    # xG attesi subiti
    src = "xG"
    if odds:                                     # mescola con i gol attesi impliciti nelle quote
        o_for, o_conc = (odds["lh"], odds["la"]) if is_home else (odds["la"], odds["lh"])
        w = CFG["odds_weight"]
        new_for, new_conc = w * o_for + (1 - w) * lam_for, w * o_conc + (1 - w) * lam_conc
        att_factor *= new_for / max(lam_for, 0.05)
        lam_for, lam_conc, src = new_for, new_conc, "xG+quote"
    return {"att_factor": att_factor, "lam_for": lam_for, "lam_conc": lam_conc, "src": src}


# ------------------------------------------------------------- GIORNATE -----
def split_rounds(matches):
    """Divide il calendario in giornate: ordinate per data, una nuova giornata inizia quando una
    squadra compare per la seconda volta. Ritorna [(numero, [partite])]; i gruppi con meno di 5
    partite (recuperi) hanno numero 0."""
    ms = sorted(matches, key=lambda m: m["datetime"])
    groups, cur, seen = [], [], set()
    for m in ms:
        t = {m["h"]["title"], m["a"]["title"]}
        if t & seen:
            groups.append(cur)
            cur, seen = [], set()
        cur.append(m)
        seen |= t
    if cur:
        groups.append(cur)
    out, n = [], 0
    for g in groups:
        if len(g) >= 5:
            n += 1
            out.append((n, g))
        else:
            out.append((0, g))
    return out


def current_round(matches, now):
    """Giornata in corso (o la prossima), COMPLETA: include anche le partite gia' giocate."""
    for no, g in split_rounds(matches):
        if any((not m.get("isResult")) and datetime.fromisoformat(m["datetime"]) >= now - timedelta(days=2)
               for m in g):
            return no, g
    return 0, []


def get_round(matches, no):
    for n, g in split_rounds(matches):
        if n == no:
            return g
    return []


# ------------------------------------------------------------- QUOTE --------
FACT = np.array([math.factorial(i) for i in range(16)], dtype=float)


def _pmf(lam, n=12):
    k = np.arange(n + 1)
    return np.exp(-lam) * lam ** k / FACT[: n + 1]


def fit_lambdas(p1, px, p2, p_over=None, line=2.5):
    """Trova i gol attesi (casa, trasferta) di due Poisson indipendenti che riproducono le
    probabilita' 1X2 (e, se c'e', la probabilita' di Over) ricavate dalle quote."""
    k = int(math.floor(line))

    def loss(lh, la):
        M = np.outer(_pmf(lh), _pmf(la))
        v = (np.tril(M, -1).sum() - p1) ** 2 + (np.trace(M) - px) ** 2 + (np.triu(M, 1).sum() - p2) ** 2
        if p_over is not None:
            v += (1 - _pmf(lh + la, 14)[: k + 1].sum() - p_over) ** 2
        return v

    best = (9.0, 1.4, 1.1)
    for lh in np.arange(0.2, 3.61, 0.1):
        for la in np.arange(0.2, 3.61, 0.1):
            v = loss(lh, la)
            if v < best[0]:
                best = (v, lh, la)
    _, bh, ba = best
    for lh in np.arange(max(bh - 0.1, 0.05), bh + 0.101, 0.02):
        for la in np.arange(max(ba - 0.1, 0.05), ba + 0.101, 0.02):
            v = loss(lh, la)
            if v < best[0]:
                best = (v, lh, la)
    return float(best[1]), float(best[2])


def odds_to_lambdas(ev):
    """Da un evento The Odds API a gol attesi. Media dei bookmaker dopo aver tolto il margine."""
    h, a = ev["home_team"], ev["away_team"]
    P, over = [], []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] == "h2h":
                d = {o["name"]: o["price"] for o in mk["outcomes"]}
                if h in d and a in d and "Draw" in d:
                    inv = np.array([1 / d[h], 1 / d["Draw"], 1 / d[a]])
                    P.append(inv / inv.sum())
            elif mk["key"] == "totals":
                o = [x for x in mk["outcomes"] if x["name"] == "Over" and abs(float(x.get("point", 0)) - 2.5) < 1e-9]
                u = [x for x in mk["outcomes"] if x["name"] == "Under" and abs(float(x.get("point", 0)) - 2.5) < 1e-9]
                if o and u:
                    io, iu = 1 / o[0]["price"], 1 / u[0]["price"]
                    over.append(io / (io + iu))
    if not P:
        return None
    p1, px, p2 = np.mean(P, axis=0)
    lh, la = fit_lambdas(p1, px, p2, float(np.mean(over)) if over else None)
    return {"lh": lh, "la": la, "p1": float(p1), "px": float(px), "p2": float(p2), "n_book": len(P)}


def _http_json(url, timeout=25, headers_out=None):
    req = urllib.request.Request(url, headers={"User-Agent": "fanta-predictor"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if headers_out is not None:
            headers_out.update({k.lower(): v for k, v in r.headers.items()})
        return json.loads(r.read().decode("utf-8"))


def fetch_odds(api_key, cache_dir=".cache_fanta"):
    """Quote Serie A da The Odds API (piano gratuito). Cache di poche ore per risparmiare crediti."""
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    f = cache / "odds.json"
    if f.exists() and (time.time() - f.stat().st_mtime) < CFG["odds_cache_hours"] * 3600:
        return json.loads(f.read_text(encoding="utf-8")), None
    base = "https://api.the-odds-api.com/v4/sports"
    sports = _http_json(f"{base}/?apiKey={urllib.parse.quote(api_key)}")          # chiamata gratuita
    keys = [s["key"] for s in sports]
    key = "soccer_italy_serie_a" if "soccer_italy_serie_a" in keys else next(
        (k for k in keys if "italy" in k and "serie_a" in k), None)
    if not key:
        raise RuntimeError("campionato Serie A non presente nell'elenco dell'API quote")
    last = None
    for markets in ("h2h,totals", "h2h"):                                        # totals: se non disponibile, solo 1X2
        hdr = {}
        try:
            data = _http_json(f"{base}/{key}/odds/?regions=eu&markets={markets}&oddsFormat=decimal"
                              f"&apiKey={urllib.parse.quote(api_key)}", headers_out=hdr)
            f.write_text(json.dumps(data), encoding="utf-8")
            return data, hdr.get("x-requests-remaining")
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def attach_odds(events, titles):
    """(casa, trasferta) con nomi Understat -> gol attesi dalle quote."""
    out, unknown = {}, set()
    for ev in events or []:
        h, a = resolve_team(ev["home_team"], titles), resolve_team(ev["away_team"], titles)
        for nm, t in ((ev["home_team"], h), (ev["away_team"], a)):
            if t is None:
                unknown.add(nm)
        lam = odds_to_lambdas(ev) if h and a else None
        if lam:
            out[(h, a)] = lam
    if unknown:
        print("  ! quote: squadre non riconosciute:", ", ".join(sorted(unknown)))
    return out


# ============================================================== GIOCATORI ===
def role_priors(players_df):
    df = players_df.copy()
    df["role"] = df["position"].astype(str).str.split().str[0].map(POS_MAP)
    df = df.dropna(subset=["role"])
    df = df[df.time > 0]
    pri = {}
    for role, g in df.groupby("role"):
        tot = g.time.sum()
        p = {c: g[c].sum() / tot * 90 for c in ["xG", "xA", "xGBuildup", "yellow_cards", "red_cards"]}
        gg = g[g.time >= 600]
        bu = gg.xGBuildup / gg.time * 90
        p["bu_mean"] = bu.mean() if len(bu) > 3 else p["xGBuildup"]
        p["bu_std"] = max(bu.std(), 1e-3) if len(bu) > 3 else 0.1
        pri[role] = p
    return pri


def find_player(name, override, team, pool):
    """Riconosce il giocatore in Understat. pool: id, player_name, team_title, time, tag(0=stagione corrente)."""
    tg = tokens(override or name)
    longs = [t for t in tg if len(t) > 1]
    inits = {t for t in tg if len(t) == 1}
    best, best_s = None, 0.0
    for r in pool.itertuples():
        ct = tokens(r.player_name)
        if not ct or not longs:
            continue
        if all(t in ct for t in longs):
            s = 1.0 + (0.05 if inits and any(c[0] in inits for c in ct if c not in longs) else 0)
        else:
            r1 = difflib.SequenceMatcher(None, " ".join(tg), " ".join(ct)).ratio()
            r2 = max(difflib.SequenceMatcher(None, t, c).ratio() for t in longs for c in ct)
            s = max(r1, 0.9 * r2)
            if s < 0.78:
                continue
        s += 0.25 * (team in str(r.team_title).split(",")) + 0.05 * (r.tag == 0) + 1e-6 * r.time
        if s > best_s:
            best, best_s = r, s
    return best


def resolve_team(name, titles):
    key = " ".join(tokens(name))
    if key in TEAM_ALIAS and TEAM_ALIAS[key] in titles:
        return TEAM_ALIAS[key]
    for t in titles:
        if " ".join(tokens(t)) == key:
            return t
    for t in titles:
        if key and (key in " ".join(tokens(t)) or " ".join(tokens(t)) in key):
            return t
    m = difflib.get_close_matches(name, titles, n=1, cutoff=0.6)
    return m[0] if m else None


def player_profile(pm, team, team_hist, role, priors, cards):
    """Ritorna i tassi per 90' (ristretti verso la media di ruolo) e la disponibilita'."""
    pr = priors[role]
    K = CFG["player_shrink_90s"]
    out = {"n_matches": 0}
    pm = num(pd.DataFrame(pm), ["time", "xG", "xA", "xGBuildup"]) if len(pm) else pd.DataFrame()
    if len(pm):
        pm["date"] = pd.to_datetime(pm["date"])
        pm = pm[pm.time > 0].sort_values("date").tail(50).reset_index(drop=True)
    if len(pm):
        w = decay(len(pm), CFG["player_half_life"])
        s90 = (w * pm.time / 90).sum()
        out["xg90"] = ((w * pm.xG).sum() + K * pr["xG"]) / (s90 + K)
        out["xa90"] = ((w * pm.xA).sum() + K * pr["xA"]) / (s90 + K)
        out["bu90"] = ((w * pm.xGBuildup).sum() + K * pr["xGBuildup"]) / (s90 + K)
        out["n_matches"] = len(pm)
    else:
        out["xg90"], out["xa90"], out["bu90"] = pr["xG"], pr["xA"], pr["xGBuildup"]
    out["z_bu"] = float(np.clip((out["bu90"] - pr["bu_mean"]) / pr["bu_std"], -2, 2))

    # cartellini per 90' (2 stagioni), ristretti verso il ruolo
    c = cards or {"time": 0, "y": 0, "r": 0}
    out["yc90"] = (c["y"] + 10 * pr["yellow_cards"]) / (c["time"] / 90 + 10)
    out["rc90"] = (c["r"] + 10 * pr["red_cards"]) / (c["time"] / 90 + 10)

    # disponibilita': ultime N partite della SQUADRA ATTUALE dopo il primo impiego con lei
    p_start = p_sub = 0.0
    exp_min = 60.0
    if len(pm):
        mine = pm[(pm.h_team == team) | (pm.a_team == team)]
        if len(mine):
            first = mine.date.min().normalize()
            th = team_hist.get(team)
            if th is not None:
                recent = th[th.date.dt.normalize() >= first].tail(CFG["avail_window"])
                if len(recent):
                    mins = {d.normalize(): t for d, t in zip(mine.date, mine.time)}
                    m = np.array([mins.get(d.normalize(), 0.0) for d in recent.date])
                    w = decay(len(m), CFG["avail_half_life"])
                    p_start = float((w * (m >= CFG["starter_minutes"])).sum() / w.sum())
                    p_sub = float((w * ((m > 0) & (m < CFG["starter_minutes"]))).sum() / w.sum())
                    played = m > 0
                    if played.any():
                        exp_min = float((w[played] * m[played]).sum() / w[played].sum())
    out.update(p_start=p_start, p_sub=p_sub, exp_min=min(exp_min, 90.0))
    return out


def fv_ref(role, pr, lg):
    """Fantavoto di un giocatore MEDIO del ruolo in una partita neutra (definisce il '6' del voto previsto)."""
    fv = CFG["base_vote"] + CFG["goal"] * pr["xG"] + CFG["assist"] * pr["xA"]
    fv += CFG["yellow"] * pr["yellow_cards"] + CFG["red"] * pr["red_cards"]
    fv += CFG["clean_sheet"][role] * float(np.exp(-lg))
    if role == "P":
        fv += CFG["conceded_gk"] * lg
    return fv


def project(role, prof, ctx, ref, off=0.0):
    """off = correzione per ruolo ricavata dalla calibrazione sui voti reali (0 se non disponibile)."""
    m = prof["exp_min"] / 90
    e_g = prof["xg90"] * ctx["att_factor"] * m
    e_a = prof["xa90"] * ctx["att_factor"] * m
    p_cs = float(np.exp(-ctx["lam_conc"]))
    fv_raw = CFG["base_vote"]
    fv_raw += CFG["goal"] * e_g + CFG["assist"] * e_a
    fv_raw += CFG["yellow"] * prof["yc90"] * m + CFG["red"] * prof["rc90"] * m
    fv_raw += CFG["clean_sheet"][role] * p_cs
    if role == "P":
        fv_raw += CFG["conceded_gk"] * ctx["lam_conc"]
    if role in ("D", "C"):
        fv_raw += CFG["involvement_weight"] * prof["z_bu"]
    fv = fv_raw + off
    # voto previsto: 6 = giocatore medio del ruolo in partita neutra (come i voti veri del fantacalcio)
    rating = float(np.clip(CFG["rating_center"] + CFG["rating_slope"] * (fv_raw - ref), 1, 10))
    p_eff = min(1.0, prof["p_start"] + CFG["sub_weight"] * prof["p_sub"])
    ev = p_eff * fv + (1 - p_eff) * (ref + off)     # se non gioca, entra in media un sostituto "medio"
    return {"e_g": e_g, "e_a": e_a, "p_cs": p_cs, "fv": fv, "fv_raw": fv_raw, "rating": rating,
            "p_eff": p_eff, "ev": ev}


# ============================================================== FORMAZIONE ==
def best_lineup(df, modules=None):
    best = None
    for mod in (modules or MODULES):
        d, c, a = map(int, mod.split("-"))
        need = {"P": 1, "D": d, "C": c, "A": a}
        pick, tot, ok = [], 0.0, True
        for role, n in need.items():
            g = df[(df.Ruolo == role) & (df.Disp)]
            g_ok = g[g["Titolare%"] >= 100 * CFG["min_play_prob"]]
            g = (g_ok if len(g_ok) >= n else g).sort_values("EV", ascending=False)
            if len(g) < n:
                ok = False
                break
            pick.append(g.head(n))
            tot += g.head(n).EV.sum()
        if ok and (best is None or tot > best[1]):
            best = (mod, tot, pd.concat(pick))
    if best is None:
        return None
    mod, tot, xi = best
    bench = df[(~df.index.isin(xi.index)) & (df.Disp)].sort_values("EV", ascending=False)
    return mod, tot, xi, bench


# ============================================================== PIPELINE ====
def run(prov, roster, season, now, check_only=False, round_no=None, backtest=False,
        odds_events=None, calib=None, odds_note=None):
    print(f"Stagione Understat: {season}/{str(season + 1)[-2:]}   -   {now:%d/%m/%Y %H:%M}")
    print("Scarico dati squadre e giocatori...")
    matches_all = prov.matches(season)
    if round_no:
        rno, fixtures = round_no, get_round(matches_all, round_no)
    else:
        rno, fixtures = current_round(matches_all, now)
    if not fixtures:
        sys.exit("Nessuna partita trovata su Understat per la giornata richiesta "
                 "(stagione finita, non ancora pubblicata o numero errato).")
    cutoff = datetime.fromisoformat(fixtures[0]["datetime"]) if backtest else None
    if backtest:
        now = cutoff

    hist = build_team_history(prov, season, cutoff)
    S, lg, hf, af = team_strengths(hist)
    pcur = pd.DataFrame(prov.players(season))
    try:
        pold = pd.DataFrame(prov.players(season - 1))
    except Exception:  # noqa: BLE001
        pold = pd.DataFrame(columns=pcur.columns)
    numcols = ["time", "xG", "xA", "xGBuildup", "yellow_cards", "red_cards"]
    pcur, pold = num(pcur, numcols), num(pold, numcols)
    priors = role_priors(pd.concat([pcur, pold]))
    refs = {r: fv_ref(r, priors[r], lg) for r in priors}
    pool = pd.concat([pcur.assign(tag=0), pold.assign(tag=1)])[["id", "player_name", "team_title", "time", "tag"]]
    cards_all = pd.concat([pcur, pold]).groupby("id").agg(time=("time", "sum"), y=("yellow_cards", "sum"),
                                                          r=("red_cards", "sum"))
    titles = list(hist.keys())
    off = (calib or {}).get("offset", {})

    fx = {}
    for m in fixtures:
        h, a = m["h"]["title"], m["a"]["title"]
        played = (not backtest) and (bool(m.get("isResult")) or datetime.fromisoformat(m["datetime"]) < now)
        fx[h] = (a, True, m["datetime"], played)
        fx[a] = (h, False, m["datetime"], played)
    n_played = sum(1 for v in fx.values() if v[3]) // 2
    odds_map = {} if backtest else attach_odds(odds_events, titles)
    tag = " [BACKTEST: solo dati precedenti]" if backtest else ""
    print(f"Giornata {rno}: {len(fixtures)} partite dal {fixtures[0]['datetime'][:16]} "
          f"({n_played} gia' iniziate/giocate){tag}")
    if odds_events is not None and not backtest:
        print(f"Quote bookmaker trovate per {len(odds_map)}/{len(fixtures)} partite")
    if any(abs(v) > 1e-9 for v in off.values()):
        print("Correzione calibrazione per ruolo:", {k: round(v, 2) for k, v in off.items()})

    rows, problems = [], []
    for r in roster.itertuples():
        team = resolve_team(r.squadra, titles)
        role = str(r.ruolo).strip().upper()[0]
        override = getattr(r, "understat", None)
        override = override if isinstance(override, str) and override.strip() else None
        avail = int(getattr(r, "disponibile", 1)) == 1
        if team is None:
            problems.append(f"{r.nome}: squadra '{r.squadra}' non riconosciuta")
            continue
        hit = find_player(r.nome, override, team, pool)
        if hit is not None:
            print(f"  {r.nome:<14} -> {hit.player_name:<28} [{hit.team_title}] (id {hit.id})")
        else:
            print(f"  {r.nome:<14} -> NON TROVATO su Understat")
        if check_only:
            continue
        if team not in fx:
            problems.append(f"{r.nome}: {team} non gioca in questa giornata")
            continue
        opp, home, when, played = fx[team]
        if hit is None:
            # nessun dato (esordiente/mai in campo): media del ruolo, titolarita' 0 -> la imposti tu con lo slider
            pr = priors[role]
            prof = {"xg90": pr["xG"], "xa90": pr["xA"], "bu90": pr["xGBuildup"], "z_bu": 0.0,
                    "yc90": pr["yellow_cards"], "rc90": pr["red_cards"],
                    "p_start": 0.0, "p_sub": 0.0, "exp_min": 60.0, "n_matches": 0}
            nota = "nessun dato Understat: usa lo slider"
        else:
            pm = prov.player_matches(hit.id)
            if backtest:
                pm = [x for x in pm if str(x["date"])[:10] < cutoff.strftime("%Y-%m-%d")]
            prof = player_profile(pm, team, hist, role, priors, cards_all.loc[hit.id].to_dict()
                                  if hit.id in cards_all.index else None)
            nota = ("nessuna presenza recente" if prof["p_start"] + prof["p_sub"] == 0 else
                    "pochi dati" if prof["n_matches"] < 5 else "")
        if played:
            nota = (nota + " - " if nota else "") + "partita gia' iniziata/giocata"
        odds = odds_map.get((team, opp) if home else (opp, team))
        ctx = team_ctx(team, opp, home, S, lg, hf, af, odds)
        pj = project(role, prof, ctx, refs[role], off.get(role, 0.0))
        rows.append({
            "Giocatore": r.nome, "Squadra": r.squadra, "Ruolo": role, "Disp": avail,
            "Avversario": f"{'vs' if home else '@'} {opp}", "Data": f"{when[8:10]}/{when[5:7]}",
            "xG": round(pj["e_g"], 2), "xA": round(pj["e_a"], 2),
            "P_gol": round(100 * (1 - math.exp(-pj["e_g"]))), "P_ass": round(100 * (1 - math.exp(-pj["e_a"]))),
            "CS%": round(100 * pj["p_cs"]) if role in "PD" else None,
            "Titolare%": round(100 * pj["p_eff"]),
            "Fantavoto": round(pj["fv"], 2), "FV_raw": round(pj["fv_raw"], 3),
            "Voto": round(pj["rating"], 1), "EV": round(pj["ev"], 2), "Rif": round(refs[role] + off.get(role, 0.0), 3),
            "GolSq": round(ctx["lam_for"], 2), "GolOpp": round(ctx["lam_conc"], 2), "Fonte": ctx["src"],
            "Nota": nota,
        })
    if problems:
        print("\nATTENZIONE:")
        for p in problems:
            print("  -", p)
    if backtest:
        odds_msg = "non usate (backtest)"
    elif odds_events is None:
        odds_msg = odds_note or "non attive"
    else:
        odds_msg = f"{len(odds_map)}/{len(fixtures)} partite"
        if len(odds_map) < len(fixtures):
            odds_msg += f" ({len(odds_events)} eventi ricevuti; le partite gia' iniziate non hanno quote)"
        if odds_note:
            odds_msg += f" - {odds_note}"
    info = {"rno": rno, "odds": len(odds_map), "partite": len(fixtures), "odds_msg": odds_msg}
    if check_only:
        return None, None, problems, info
    return pd.DataFrame(rows), fixtures, problems, info


def save_predictions(df, info, fixtures, now, storico, force=False):
    """Salva le previsioni GREZZE (senza correzione) della giornata, finche' non inizia: servono alla calibrazione."""
    rno = info["rno"]
    if rno <= 0:
        return None
    kickoff = datetime.fromisoformat(fixtures[0]["datetime"])
    if not force and now >= kickoff:
        return None                      # giornata gia' iniziata: non sovrascrivo le previsioni pre-partita
    d = Path(storico)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"G{rno:02d}.csv"
    df[["Giocatore", "Ruolo", "Squadra", "Avversario", "FV_raw", "Fantavoto", "Titolare%", "EV"]].to_csv(f, index=False)
    return f


# ============================================================== CALIBRAZIONE =
def read_voti(path):
    """voti_reali.csv: blocchi '# giornata: N' seguiti da righe 'giocatore,voto,fantavoto'.
    '-' o 'sv' = senza voto (ignorato)."""
    rows, g, bad = [], None, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("giocatore"):
            continue
        m = re.match(r"^#\s*giornata\s*[:=]?\s*(\d+)", line, re.I)
        if m:
            g = int(m.group(1))
            continue
        if line.startswith("#"):
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        try:
            fv = float(p[2].replace(",", "."))
        except ValueError:
            continue
        if not g:
            bad += 1
            continue
        rows.append({"giornata": g, "Giocatore": p[0], "Reale": fv})
    if bad:
        print(f"  ! {bad} righe di {path} ignorate: manca il numero di giornata ('# giornata: N')")
    return pd.DataFrame(rows, columns=["giornata", "Giocatore", "Reale"])


def calibra(storico_dir, voti_path, out_json):
    out = {"aggiornato": datetime.now().strftime("%Y-%m-%d %H:%M"), "offset": {r: 0.0 for r in "PDCA"}, "report": {"n": 0}}
    frames = []
    for f in sorted(Path(storico_dir).glob("G*.csv")):
        d = pd.read_csv(f)
        d["giornata"] = int(re.findall(r"\d+", f.stem)[0])
        frames.append(d)
    if not frames or not Path(voti_path).exists():
        print("Calibrazione: mancano previsioni salvate o voti reali, nessuna correzione applicata.")
        Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out
    pred, real = pd.concat(frames), read_voti(voti_path)
    df = pred.merge(real, on=["giornata", "Giocatore"], how="inner")
    if df.empty:
        print("Calibrazione: nessun giocatore in comune fra previsioni e voti reali.")
        Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out
    df["err"] = df["FV_raw"] - df["Reale"]
    rep = {"n": int(len(df)), "giornate": sorted(int(x) for x in df.giornata.unique()),
           "mae_modello": float(df.err.abs().mean()), "bias": {}, "n_ruolo": {}}
    # baseline 1: media reale del ruolo
    rep["mae_media_ruolo"] = float((df.Reale - df.groupby("Ruolo").Reale.transform("mean")).abs().mean())
    # baseline 2: media dei suoi altri fantavoti reali (leave-one-out), solo chi ha >= 2 osservazioni
    g = df.groupby("Giocatore").Reale
    cnt, tot = g.transform("count"), g.transform("sum")
    sub = df[cnt >= 2].copy()
    if len(sub):
        loo = (tot[cnt >= 2] - sub.Reale) / (cnt[cnt >= 2] - 1)
        rep["n_sub"] = int(len(sub))
        rep["mae_modello_sub"] = float(sub.err.abs().mean())
        rep["mae_media_giocatore"] = float((sub.Reale - loo).abs().mean())
    cs = []
    for role, gr in df.groupby("Ruolo"):
        n = len(gr)
        bias = float(gr.err.mean())
        rep["bias"][role], rep["n_ruolo"][role] = round(bias, 3), int(n)
        if n >= CFG["calib_min_obs"]:
            out["offset"][role] = round(-bias * n / (n + CFG["calib_shrink"]), 3)
        if n >= 5:
            cs.append(gr["FV_raw"].rank().corr(gr["Reale"].rank()))
    rep["spearman_ruolo"] = float(np.nanmean(cs)) if cs else None
    out["report"] = rep
    Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Calibrazione su {rep['n']} prestazioni reali, giornate {rep['giornate']}")
    print(f"  errore medio modello (MAE): {rep['mae_modello']:.2f}   media del ruolo: {rep['mae_media_ruolo']:.2f}")
    if "mae_media_giocatore" in rep:
        print(f"  (su {rep['n_sub']} righe con storico) modello {rep['mae_modello_sub']:.2f} "
              f"vs media dei suoi fantavoti {rep['mae_media_giocatore']:.2f}")
    print(f"  scarto medio per ruolo (previsto - reale): {rep['bias']}")
    print(f"  correzione applicata: {out['offset']}   (prudente: si attenua con pochi dati)")
    return out


# ============================================================== OUTPUT ======
HTML_TEMPLATE = r"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fanta Predictor</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#777;--card:#f3f4f6;--line:#0002;--acc:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#111318;--fg:#eee;--mut:#999;--card:#1c1f27;--line:#fff2;--acc:#60a5fa}}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:0 12px 24px;background:var(--bg);color:var(--fg);max-width:720px;margin:auto}
h1{font-size:20px;margin:12px 0 2px}h2{font-size:15px;margin:18px 0 8px}.s{color:var(--mut);font-size:12px}
#bar{position:sticky;top:0;z-index:10;background:var(--bg);padding:8px 0 7px;border-bottom:1px solid var(--line);margin-top:8px}
.bh{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}.bh b{font-size:14px;flex:1;min-width:120px}
.mini div{font-size:12.5px;line-height:1.4}.mini i{font-style:normal;font-weight:700;color:var(--acc);margin-right:4px}
.card{background:var(--card);border-radius:12px;padding:10px 12px;margin-bottom:10px}
.ln{display:flex;gap:8px;align-items:baseline;padding:4px 0;font-size:14px;flex-wrap:wrap}
.ln b{min-width:16px}.tag{font-size:12px;color:var(--mut)}
.row{background:var(--card);border-radius:12px;padding:9px 12px;margin-bottom:8px;border-left:4px solid transparent}
.row.in{border-left-color:#16a34a}.row.out{opacity:.5}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.nm{font-weight:600}.r{font-size:11px;padding:1px 6px;border-radius:8px;background:var(--line);margin-right:6px}
.badge{color:#fff;font-weight:700;border-radius:8px;padding:3px 9px;font-size:14px;min-width:40px;text-align:center}
.info{font-size:12px;color:var(--mut);margin:3px 0 0}.info2{font-size:12px;color:var(--mut);margin:0 0 6px}
.ctl{display:flex;align-items:center;gap:10px}
input[type=range]{flex:1;accent-color:var(--acc)}
.pv{font-size:13px;min-width:74px;text-align:right}.chg{color:var(--acc);font-weight:600}
label.o{font-size:12px;white-space:nowrap}
button,select{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:7px 11px;font-size:13px}
select{background:var(--card);color:var(--fg);border:1px solid var(--line)}
</style></head><body>
<h1>Fanta Predictor</h1>
<div class="s" id="meta"></div>
<div id="bar">
  <div class="bh"><b id="lt">Formazione</b>
    <select id="mod" aria-label="Modulo"></select><button id="copy">Copia</button></div>
  <div class="mini" id="mini"></div>
</div>
<h2>Dettaglio formazione</h2><div class="card" id="lineup"></div>
<h2>Panchina (in ordine)</h2><div class="card" id="bench"></div>
<div id="probs" class="card" style="display:none"></div>
<div id="cal" class="card" style="display:none"></div>
<h2>Titolarit&agrave; e infortuni <button id="reset" style="float:right">Azzera</button></h2>
<div class="s" style="margin-bottom:8px">Muovi lo slider con le percentuali delle probabili formazioni: la formazione si ricalcola subito. Le modifiche restano salvate su questo dispositivo fino alla giornata successiva.</div>
<div id="roster"></div>
<p class="s">Voto previsto: 6 = giocatore medio del suo ruolo in una partita neutra (come i voti veri); +1 fantavoto rispetto alla media = +1,5 voti. Vale SE il giocatore scende in campo. Gol/assist % = probabilit&agrave; di almeno un gol/assist. Valore atteso = P(gioca) x fantavoto + (1 - P) x sostituto medio. Modello statistico, non una garanzia.</p>
<script>
const D = __DATA__;
const KEY = "fanta_v2_" + D.giornata;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let modSel = saved._mod || "auto", cur = null, inSet = new Set();
const S = D.players.map(p => Object.assign({}, p, {
  p0: p["Titolare%"], p: (saved[p.Giocatore] && saved[p.Giocatore].p != null) ? saved[p.Giocatore].p : p["Titolare%"],
  disp: (saved[p.Giocatore] && saved[p.Giocatore].disp != null) ? saved[p.Giocatore].disp : !!p.Disp }));
const ev = s => (s.p/100)*s.Fantavoto + (1 - s.p/100)*s.Rif;
const col = v => "hsl(" + (120*(Math.max(3,Math.min(9,v))-3)/6).toFixed(0) + ",62%,40%)";
function save(){ const o = {}; S.forEach(s => { if (s.p !== s.p0 || s.disp !== !!s.Disp) o[s.Giocatore] = {p:s.p, disp:s.disp}; });
  if (modSel !== "auto") o._mod = modSel; try { localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) {} }
function calc(mods){
  let best = null;
  for (const mod of mods){
    const [d,c,a] = mod.split("-").map(Number), need = {P:1, D:d, C:c, A:a};
    let tot = 0, pick = [], ok = true;
    for (const r of ["P","D","C","A"]){
      let g = S.filter(s => s.Ruolo === r && s.disp);
      const gok = g.filter(s => s.p >= D.minp*100);
      g = (gok.length >= need[r] ? gok : g).sort((x,y) => ev(y) - ev(x));
      if (g.length < need[r]) { ok = false; break; }
      const t = g.slice(0, need[r]); pick.push(...t); tot += t.reduce((q,s) => q + ev(s), 0);
    }
    if (ok && (!best || tot > best.tot)) best = {mod, tot, pick};
  }
  return best;
}
function bestLineup(){
  if (modSel !== "auto") { const b = calc([modSel]); if (b) return b; }
  return calc(D.modules);
}
function mk(tag, cls, txt){ const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }
function line(s){
  const l = mk("div","ln"); l.appendChild(mk("b",null,s.Ruolo)); l.appendChild(mk("span","nm",s.Giocatore));
  l.appendChild(mk("span","tag", s.Avversario + " \u00b7 voto " + s.Voto.toFixed(1) + " \u00b7 gioca " + s.p + "%")); return l;
}
const byRole = (b, r) => b.pick.filter(s => s.Ruolo === r).sort((x,y) => ev(y)-ev(x));
function update(){
  const b = bestLineup(); cur = b;
  const L = document.getElementById("lineup"), B = document.getElementById("bench"), M = document.getElementById("mini");
  L.textContent = ""; B.textContent = ""; M.textContent = ""; inSet = new Set();
  if (!b) { L.textContent = "Nessun modulo valido con i giocatori disponibili."; document.getElementById("lt").textContent = "Formazione non disponibile"; }
  else {
    document.getElementById("lt").textContent = b.mod + " \u00b7 atteso " + b.tot.toFixed(1);
    ["P","D","C","A"].forEach(r => { const g = byRole(b, r), d = mk("div"); d.appendChild(mk("i",null,r));
      d.appendChild(document.createTextNode(g.map(s => s.Giocatore).join(" \u00b7 "))); M.appendChild(d);
      g.forEach(s => { inSet.add(s.Giocatore); L.appendChild(line(s)); }); });
  }
  S.filter(s => s.disp && !inSet.has(s.Giocatore)).sort((x,y) => ev(y)-ev(x)).forEach(s => B.appendChild(line(s)));
  document.querySelectorAll(".row").forEach(r => { const s = S[+r.dataset.i];
    r.classList.toggle("in", inSet.has(s.Giocatore)); r.classList.toggle("out", !s.disp);
    const pv = r.querySelector(".pv"); pv.textContent = s.p + "%" + (s.p !== s.p0 ? " (modello " + s.p0 + "%)" : "");
    pv.classList.toggle("chg", s.p !== s.p0); });
}
function lineupText(){
  if (!cur) return "";
  const nm = r => byRole(cur, r).map(s => s.Giocatore).join(", ");
  const bench = S.filter(s => s.disp && !inSet.has(s.Giocatore)).sort((x,y) => ev(y)-ev(x)).map(s => s.Giocatore).join(", ");
  return "Giornata " + D.gno + " - modulo " + cur.mod + "\nP: " + nm("P") + "\nD: " + nm("D") + "\nC: " + nm("C") + "\nA: " + nm("A") + "\nPanchina: " + bench;
}
function copyText(t, btn){
  const ok = () => { btn.textContent = "Copiato \u2713"; setTimeout(() => btn.textContent = "Copia", 1500); };
  const fb = () => { try { const ta = mk("textarea"); ta.value = t; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); ok(); } catch (e) { btn.textContent = "Non riuscito"; } };
  try { if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(ok, fb); else fb(); } catch (e) { fb(); }
}
function build(){
  const R = document.getElementById("roster"); R.textContent = "";
  D.order.forEach(i => { const s = S[i], r = mk("div","row"); r.dataset.i = i;
    const t = mk("div","top"), left = mk("div"); left.appendChild(mk("span","r",s.Ruolo)); left.appendChild(mk("span","nm",s.Giocatore));
    const bd = mk("div","badge",s.Voto.toFixed(1)); bd.style.background = col(s.Voto); t.appendChild(left); t.appendChild(bd); r.appendChild(t);
    r.appendChild(mk("div","info", s.Avversario + " (" + s.Data + ") \u00b7 gol " + s.P_gol + "% \u00b7 assist " + s.P_ass + "%" +
      (s["CS%"] != null ? " \u00b7 clean sheet " + s["CS%"] + "%" : "")));
    r.appendChild(mk("div","info2", "fantavoto " + s.Fantavoto.toFixed(2) + " \u00b7 squadra " + s.GolSq.toFixed(1) + " gol attesi, avversario " + s.GolOpp.toFixed(1) +
      " (" + s.Fonte + ")" + (s.Nota ? " \u00b7 " + s.Nota : "")));
    const c = mk("div","ctl"), sl = mk("input"); sl.type = "range"; sl.min = 0; sl.max = 100; sl.step = 5; sl.value = s.p;
    sl.addEventListener("input", () => { s.p = +sl.value; save(); update(); });
    const pv = mk("span","pv"); const lb = mk("label","o"), cb = mk("input"); cb.type = "checkbox"; cb.checked = !s.disp;
    cb.addEventListener("change", () => { s.disp = !cb.checked; save(); update(); });
    lb.appendChild(cb); lb.appendChild(document.createTextNode(" fuori"));
    c.appendChild(sl); c.appendChild(pv); c.appendChild(lb); r.appendChild(c); R.appendChild(r); });
}
const sel = document.getElementById("mod");
[["auto","Auto"]].concat(D.modules.map(m => [m, m])).forEach(o => { const e = mk("option", null, o[1]); e.value = o[0]; sel.appendChild(e); });
sel.value = modSel;
sel.addEventListener("change", () => { modSel = sel.value; save(); update(); });
document.getElementById("copy").addEventListener("click", e => copyText(lineupText(), e.target));
if (D.problems && D.problems.length) { const P = document.getElementById("probs"); P.style.display = "block";
  P.appendChild(mk("b", null, "Giocatori non calcolati")); D.problems.forEach(t => P.appendChild(mk("div", "info", t))); }
if (D.calib && D.calib.n > 0) { const C = document.getElementById("cal"), c = D.calib; C.style.display = "block";
  C.appendChild(mk("b", null, "Affidabilit\u00e0 del modello"));
  C.appendChild(mk("div", "info", "Su " + c.n + " prestazioni reali (giornate " + c.giornate.join(", ") + ") l'errore medio sul fantavoto \u00e8 " + c.mae_modello.toFixed(2) +
    "; prevedere la media del ruolo darebbe " + c.mae_media_ruolo.toFixed(2) + "." + (c.mae_media_giocatore != null ?
    " Su chi ha storico: modello " + c.mae_modello_sub.toFixed(2) + " contro " + c.mae_media_giocatore.toFixed(2) + " usando la media dei suoi ultimi fantavoti." : "")));
  C.appendChild(mk("div", "info", c.n < 60 ? "Campione ancora piccolo: prendi questi numeri come indicativi." : "")); }
document.getElementById("meta").textContent = "Giornata " + D.gno + " \u00b7 aggiornato " + D.agg + " \u00b7 quote bookmaker: " + D.odds_msg;
document.getElementById("reset").addEventListener("click", () => { S.forEach(s => { s.p = s.p0; s.disp = !!s.Disp; }); modSel = "auto"; sel.value = "auto";
  try { localStorage.removeItem(KEY); } catch (e) {} build(); update(); });
build(); update();
</script></body></html>"""


def to_html(df, fixtures, now, modules, problems=(), info=None, calib=None):
    d = df.reset_index(drop=True)
    order = d.assign(_o=d.Ruolo.map({"P": 0, "D": 1, "C": 2, "A": 3})).sort_values(
        ["_o", "EV"], ascending=[True, False])
    info = info or {}
    rep = (calib or {}).get("report") or None
    payload = {
        "players": json.loads(d.to_json(orient="records", force_ascii=False)),
        "order": [int(i) for i in order.index],
        "modules": modules,
        "minp": CFG["min_play_prob"],
        "problems": list(problems),
        "giornata": fixtures[0]["datetime"][:10],
        "gno": info.get("rno", 0),
        "odds": info.get("odds", 0),
        "odds_msg": info.get("odds_msg", "non attive"),
        "partite": info.get("partite", len(fixtures)),
        "agg": f"{now:%d/%m/%Y %H:%M}",
        "calib": rep if rep and rep.get("n", 0) > 0 else None,
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", data)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Previsione voti fantacalcio Serie A (xG/xA + avversario + quote)")
    ap.add_argument("--rosa", default="rosa.csv")
    ap.add_argument("--out", default="docs", help="cartella di output (index.html + previsioni.csv)")
    ap.add_argument("--storico", default="storico", help="cartella delle previsioni salvate per la calibrazione")
    ap.add_argument("--voti", default="voti_reali.csv", help="file con i tuoi voti reali")
    ap.add_argument("--calib", default="calibrazione.json")
    ap.add_argument("--season", type=int, help="anno di inizio stagione (default: automatico)")
    ap.add_argument("--moduli", default=",".join(MODULES), help="moduli ammessi dalla tua lega, es. 4-3-3,3-5-2")
    ap.add_argument("--check", action="store_true", help="solo verifica riconoscimento nomi")
    ap.add_argument("--refresh", action="store_true", help="ignora la cache")
    ap.add_argument("--calibra", action="store_true", help="ricalcola la calibrazione da storico + voti reali ed esci")
    ap.add_argument("--backtest", action="store_true",
                    help="con --giornata: ricostruisce la previsione usando solo dati precedenti a quella giornata")
    ap.add_argument("--giornata", default="", help="numero/i di giornata, es. 3 oppure 2,3 (con --backtest)")
    ap.add_argument("--no-odds", action="store_true", help="non usare le quote dei bookmaker")
    a = ap.parse_args(argv)

    if a.calibra:
        calibra(a.storico, a.voti, a.calib)
        return

    now = datetime.now(ZoneInfo(CFG["tz"])).replace(tzinfo=None)      # ora italiana
    season = a.season or (now.year if now.month >= 7 else now.year - 1)
    roster = pd.read_csv(a.rosa)
    prov = UnderstatProvider(hours=CFG["cache_hours"], refresh=a.refresh)
    modules = [m.strip() for m in a.moduli.split(",")]
    calib = None
    if Path(a.calib).exists():
        try:
            calib = json.loads(Path(a.calib).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            calib = None

    if a.backtest:
        rounds = [int(x) for x in a.giornata.split(",") if x.strip()]
        if not rounds:
            sys.exit("Con --backtest indica le giornate: --giornata 2,3")
        for g in rounds:
            df, fixtures, problems, info = run(prov, roster, season, now, round_no=g, backtest=True, calib=None)
            f = save_predictions(df, info, fixtures, now, a.storico, force=True)
            print(f"-> previsioni della giornata {g} salvate in {f}\n")
        return

    odds_events, odds_note, key = None, None, os.environ.get("ODDS_API_KEY", "").strip()
    if a.no_odds:
        odds_note = "disattivate (--no-odds)"
    elif key and not a.check:
        print(f"ODDS_API_KEY trovata ({len(key)} caratteri)")
        try:
            odds_events, remaining = fetch_odds(key)
            odds_note = f"crediti rimasti {remaining}" if remaining is not None else None
            if remaining is not None:
                print(f"Crediti The Odds API rimasti: {remaining}")
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace(key, "***")
            hint = " - chiave rifiutata: incolla nel secret solo il codice ricevuto per email, senza spazi" \
                if ("401" in msg or "403" in msg) else ""
            odds_note = f"non attive: errore API ({msg[:80]}){hint}"
            print(f"  ! quote bookmaker non disponibili, uso solo xG ({msg})")
    elif not a.check:
        odds_note = "non attive: manca il secret ODDS_API_KEY"
        print("Quote bookmaker: nessuna ODDS_API_KEY impostata, uso solo xG.")

    df, fixtures, problems, info = run(prov, roster, season, now, check_only=a.check, calib=calib,
                                       odds_events=odds_events, odds_note=odds_note)
    if a.check or df is None:
        return
    Path(a.storico).mkdir(parents=True, exist_ok=True)
    saved = save_predictions(df, info, fixtures, now, a.storico)
    lineup = best_lineup(df, modules)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values("EV", ascending=False).to_csv(out / "previsioni.csv", index=False)
    (out / "index.html").write_text(to_html(df, fixtures, now, modules, problems, info, calib), encoding="utf-8")

    pd.set_option("display.width", 200)
    show = ["Giocatore", "Ruolo", "Avversario", "Voto", "Fantavoto", "P_gol", "P_ass", "CS%", "Titolare%", "Nota"]
    if lineup:
        mod, tot, xi, bench = lineup
        print(f"\n=== FORMAZIONE CONSIGLIATA {mod} ===")
        print(xi.sort_values("Ruolo", key=lambda s: s.map({"P": 0, "D": 1, "C": 2, "A": 3}))[show].to_string(index=False))
        print("\n=== PANCHINA ===")
        print(bench[show].to_string(index=False))
    print(f"\nFile scritti in {out}/ (index.html, previsioni.csv)")
    if saved:
        print(f"Previsioni pre-partita salvate in {saved}")


if __name__ == "__main__":
    main()
