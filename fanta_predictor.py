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
    # infortuni/squalifiche (API-Football, facoltativo)
    "injuries_cache_hours": 3,
    "doubt_factor": 0.5,         # [EURISTICA] titolarita' moltiplicata per questo se il giocatore e' "in dubbio"
    # calibrazione sui tuoi voti reali
    "calib_min_obs": 5,          # osservazioni minime per ruolo prima di applicare una correzione
    "calib_min_round": 3,        # le giornate precedenti (rodaggio: pochi dati della stagione) non entrano nella correzione
    "calib_shrink": 20,          # piu' alto = correzione piu' prudente
    "fc_shrink": 20,             # idem per la correzione delle percentuali di Fantacalcio.it
    "mercato_min_minutes": 450,   # tetto massimo di minuti richiesti (si applica da meta' stagione in poi)
    "mercato_min_frac": 0.55,     # ...prima, la soglia e' questa frazione dei minuti disponibili fino ad oggi
    "mercato_search_min": 45,     # minuti minimi per comparire nella ricerca (piu' basso: solo per escludere chi non ha mai giocato)
    "mercato_luck_soglia": 0.30,  # scarto minimo (in fantavoto/90) tra reale e atteso per segnalare un giocatore
    # effetto risultato: chi vince prende voti un po' piu' alti. beta = fantavoto per unita' di (P(vittoria) - P(sconfitta)).
    # Parte da un valore prudente [EURISTICA] e viene stimato sui tuoi voti reali (con shrink verso questo valore)
    "result_beta_prior": 0.15,
    "result_beta_shrink": 60,
    "tz": "Europe/Rome",
    "version": "22/09 - forma recente e taratura",
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


def to_local(s: str) -> datetime:
    """Understat scrive gli orari in UTC (es. il venerdi' delle 20:45 italiane e' 18:45): converto in ora italiana."""
    return datetime.fromisoformat(s).replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(CFG["tz"])).replace(tzinfo=None)


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

    def match_roster(self, mid):
        return self._get(f"roster_{mid}", lambda: self.client.match(match=str(mid)).get_roster_data())


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


def outcome_probs(lf, lc):
    """P(vittoria), P(pareggio), P(sconfitta) da due Poisson indipendenti con i gol attesi della squadra e dell'avversario."""
    M = np.outer(_pmf(max(lf, 0.05)), _pmf(max(lc, 0.05)))
    M = M / M.sum()
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


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
    pv, pn, ps = outcome_probs(lam_for, lam_conc)
    return {"att_factor": att_factor, "lam_for": lam_for, "lam_conc": lam_conc, "src": src, "pv": pv, "pn": pn, "ps": ps}


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


def season_progress(matches):
    """Ultima giornata con almeno una partita gia' giocata, per sapere quanti minuti un giocatore puo' avere accumulato."""
    played = [no for no, g in split_rounds(matches) if no and any(m.get("isResult") for m in g)]
    return max(played) if played else 0


def current_round(matches, now):
    """Giornata in corso (o la prossima), COMPLETA: include anche le partite gia' giocate."""
    for no, g in split_rounds(matches):
        if any((not m.get("isResult")) and to_local(m["datetime"]) >= now - timedelta(days=2) for m in g):
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


def _http_json(url, timeout=25, headers_out=None, headers=None):
    hdrs = {"User-Agent": "fanta-predictor"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
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


# ------------------------------------------------------------- INFORTUNI ---
API_FOOTBALL = "https://v3.football.api-sports.io"


def _injuries_apisports(api_key, season, dates, cache_dir=".cache_fanta"):
    """API-Sports (sito api-football.com): infortunati, squalificati e in dubbio, una chiamata per data di gioco."""
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    f = cache / f"injuries_{season}.json"
    store = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    out = []
    for d in sorted(set(dates)):
        ent = store.get(d)
        if ent and (time.time() - ent["t"]) < CFG["injuries_cache_hours"] * 3600:
            out += ent["r"]
            continue
        data = _http_json(f"{API_FOOTBALL}/injuries?league=135&season={season}&date={d}",
                          headers={"x-apisports-key": api_key})
        errs = data.get("errors")
        if errs:                                   # API-Sports risponde 200 anche in caso di errore
            msg = "; ".join(f"{k}: {v}" for k, v in errs.items()) if isinstance(errs, dict) else "; ".join(map(str, errs))
            raise RuntimeError(msg)
        store[d] = {"t": time.time(), "r": data.get("response", [])}
        out += store[d]["r"]
    f.write_text(json.dumps(store), encoding="utf-8")
    return out


APIFOOTBALL_COM = "https://apiv3.apifootball.com/"


def _injuries_apifootball_com(api_key, cache_dir=".cache_fanta"):
    """apifootball.com (servizio DIVERSO da api-football.com): usa il flag 'player_injured' della rosa di ogni squadra.
    Solo infortunati: niente squalificati ne' in dubbio."""
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    f = cache / "injuries_apifootballcom.json"
    if f.exists() and (time.time() - f.stat().st_mtime) < CFG["injuries_cache_hours"] * 3600:
        return json.loads(f.read_text(encoding="utf-8"))
    k = urllib.parse.quote(api_key)

    def call(q):
        d = _http_json(f"{APIFOOTBALL_COM}?{q}&APIkey={k}", timeout=60)
        if isinstance(d, dict):                    # errore: {"error": 404, "message": "..."}
            raise RuntimeError(f"apifootball.com: {d.get('message') or d}")
        return d

    leagues = call("action=get_leagues&country_id=5")
    lid = next((l["league_id"] for l in leagues if str(l.get("league_name", "")).strip().lower() == "serie a"), None)
    if not lid:
        raise RuntimeError("apifootball.com: Serie A non inclusa nel tuo piano")
    out = []
    for t in call(f"action=get_teams&league_id={lid}"):
        for p in t.get("players", []):
            if str(p.get("player_injured", "")).strip().lower() == "yes":
                out.append({"player": {"name": p.get("player_name"), "type": "Missing Fixture", "reason": "infortunato"},
                            "team": {"name": t.get("team_name")}})
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


def is_apifootball_com(site):
    """'apifootball.com' (senza trattino) e' un servizio diverso da 'api-football.com' (con trattino, API-Sports)."""
    return str(site).strip().lower().replace("https://", "").replace("www.", "").startswith("apifootball")


def fetch_injuries(api_key, season, dates, site="api-football.com", cache_dir=".cache_fanta"):
    if is_apifootball_com(site):
        return _injuries_apifootball_com(api_key, cache_dir)
    return _injuries_apisports(api_key, season, dates, cache_dir)


def normalize_injuries(raw, titles):
    """Lista di {team, tokens, type, reason} con squadra in nome Understat."""
    out = []
    for e in raw or []:
        pl, tm = e.get("player") or {}, e.get("team") or {}
        team = resolve_team(tm.get("name", ""), titles) if tm.get("name") else None
        if not team or not pl.get("name"):
            continue
        out.append({"team": team, "tokens": set(tokens(pl["name"])), "type": str(pl.get("type") or ""),
                    "reason": str(pl.get("reason") or "").strip()})
    return out


def find_injury(names, team, inj):
    for name in names:
        longs = [t for t in tokens(name) if len(t) > 1]
        if not longs:
            continue
        for e in inj:
            if e["team"] == team and all(t in e["tokens"] for t in longs):
                return e
    return None


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


DEFAULT_S2 = {"A": 0.22, "C": 0.12, "D": 0.06}     # [EURISTICA] varianza per 90' dell'xGI partita per partita, se manca la taratura


def _trend(pm, role, tune):
    """Forma: xGI per 90 delle ultime 5 presenze contro le 30 precedenti, con z-score (rumore stimato dalla taratura)."""
    if role == "P" or len(pm) < 8:
        return None
    n = len(pm)
    xg, xa, mn = pm.xG.to_numpy(float), pm.xA.to_numpy(float), pm.time.to_numpy(float)
    r_i, b_i = slice(n - 5, n), slice(max(0, n - 35), n - 5)
    nr, nb = mn[r_i].sum() / 90, mn[b_i].sum() / 90
    if nr < 1.5 or nb < 3.0:
        return None
    r, b = (xg[r_i] + xa[r_i]).sum() / nr, (xg[b_i] + xa[b_i]).sum() / nb
    s2 = ((tune or {}).get(role) or {}).get("sigma2_90") or DEFAULT_S2.get(role, 0.12)
    z = (r - b) / math.sqrt(s2 / nr + s2 / nb)
    arrow = "\u2191" if z >= 1.5 else "\u2197" if z >= 0.75 else "\u2193" if z <= -1.5 else "\u2198" if z <= -0.75 else "\u2192"
    m_r, m_b = mn[-3:].mean(), mn[max(0, n - 18):n - 3].mean()
    mnote = ("minuti in calo" if (m_r < 0.75 * m_b and m_b - m_r >= 15) else
             "minuti in aumento" if (m_r > 1.25 * m_b and m_r - m_b >= 15) else "")
    return {"r": float(r), "b": float(b), "z": float(z), "arrow": arrow, "mnote": mnote, "mr": float(m_r), "mb": float(m_b),
            "r_xg": float(xg[r_i].sum() / nr), "r_xa": float(xa[r_i].sum() / nr)}


def player_profile(pm, team, team_hist, role, priors, cards, tune=None):
    """Ritorna i tassi per 90' (ristretti verso la media di ruolo) e la disponibilita'."""
    pr = priors[role]
    th = (tune or {}).get(role) or {}                     # parametri misurati dalla taratura (se presente)
    hl = th.get("half_life", CFG["player_half_life"])
    K = th.get("shrink", CFG["player_shrink_90s"])
    out = {"n_matches": 0, "trend": None}
    pm = num(pd.DataFrame(pm), ["time", "xG", "xA", "xGBuildup"]) if len(pm) else pd.DataFrame()
    if len(pm):
        pm["date"] = pd.to_datetime(pm["date"])
        pm = pm[pm.time > 0].sort_values("date").tail(50).reset_index(drop=True)
    if len(pm):
        w = decay(len(pm), hl)
        s90 = (w * pm.time / 90).sum()
        out["xg90"] = ((w * pm.xG).sum() + K * pr["xG"]) / (s90 + K)
        out["xa90"] = ((w * pm.xA).sum() + K * pr["xA"]) / (s90 + K)
        out["bu90"] = ((w * pm.xGBuildup).sum() + K * pr["xGBuildup"]) / (s90 + K)
        out["n_matches"] = len(pm)
        out["trend"] = _trend(pm, role, tune)
        tb, tr = th.get("trend_beta", 0.0), out["trend"]
        if tb > 0 and tr:                                 # la forma recente pesa solo se la taratura ha dimostrato che serve
            out["xg90"] = max(0.0, out["xg90"] + tb * (tr["r_xg"] - out["xg90"]))
            out["xa90"] = max(0.0, out["xa90"] + tb * (tr["r_xa"] - out["xa90"]))
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


# ============================================================== TARATURA =====
# Misura, su migliaia di partite di giocatori di Serie A, quanto pesare il passato (decadimento e ristringimento)
# e se la "forma recente" aggiunge informazione oltre alla media pesata. Serve la rete: si lancia da GitHub Actions.
TUNE_HL = [2, 3, 4, 6, 8, 10, 12, 16, 24, 36, 60, 1e9]   # mezza vita in partite (1e9 = nessun decadimento)
TUNE_K = [0.5, 1, 2, 4, 6, 10, 16, 30]                # peso della media di ruolo, in "90 minuti"
GROUP = {"F": "A", "M": "C", "D": "D"}                 # ruolo Understat -> ruolo fantacalcio
_FEATS = ("xgi", "shots", "kp")


def _series(pm, first_season):
    df = pd.DataFrame(pm)
    if df.empty:
        return None
    df = num(df, ["time", "xG", "xA", "shots", "key_passes"])
    df["date"] = pd.to_datetime(df["date"])
    seas = pd.to_numeric(df["season"], errors="coerce").fillna(0) if "season" in df.columns else 0
    df = df[(df.time > 0) & (seas >= first_season)].sort_values("date")
    if len(df) < 12:
        return None
    return (df.time.to_numpy(float), (df.xG + df.xA).to_numpy(float), df.shots.to_numpy(float), df.key_passes.to_numpy(float))


def _mse_grid(series, prior, min_hist=8, min_min=20):
    """Errore quadratico medio (pesato sui minuti) della stima 'xGI per 90' in funzione di (mezza vita, ristringimento)."""
    out = {}
    for h in TUNE_HL:
        lam = 0.5 ** (1.0 / h)
        for K in TUNE_K:
            sq = wt = 0.0
            for m, x, _s, _k in series:
                S = W = 0.0
                for t in range(len(m)):
                    if t >= min_hist and m[t] >= min_min:
                        e = x[t] * 90.0 / m[t] - (S + K * prior) / (W + K)
                        w = m[t] / 90.0
                        sq += w * e * e
                        wt += w
                    S = lam * S + x[t]
                    W = lam * W + m[t] / 90.0
            out[(h, K)] = sq / wt if wt else float("nan")
    return out


def _feature_tests(series, priors, h, K, min_hist=8, min_min=20):
    """La differenza 'ultime 5 partite - media pesata' (di xGI, tiri, passaggi chiave) spiega la partita successiva
    oltre alla media pesata? Coefficiente e z-score con errori standard raggruppati per giocatore."""
    lam = 0.5 ** (1.0 / h)
    acc = {f: [] for f in _FEATS}
    s2_num, s2_cnt = 0.0, 0
    for m, x, sh, kp in series:
        vals = {"xgi": x, "shots": sh, "kp": kp}
        S = {f: 0.0 for f in _FEATS}
        W = 0.0
        A, B = {f: 0.0 for f in _FEATS}, {f: 0.0 for f in _FEATS}
        for t in range(len(m)):
            n90 = m[t] / 90.0
            if t >= max(min_hist, 5) and m[t] >= min_min:
                lo = t - 5
                n_r = m[lo:t].sum() / 90.0
                if n_r >= 2.0:
                    base = (S["xgi"] + K * priors["xgi"]) / (W + K)
                    resid = x[t] / n90 - base
                    for f in _FEATS:
                        d = vals[f][lo:t].sum() / n_r - (S[f] + K * priors[f]) / (W + K)
                        A[f] += n90 * d * resid
                        B[f] += n90 * d * d
                    s2_num += resid * resid * n90
                    s2_cnt += 1
            for f in _FEATS:
                S[f] = lam * S[f] + vals[f][t]
            W = lam * W + n90
        for f in _FEATS:
            acc[f].append((A[f], B[f]))
    res = {}
    for f in _FEATS:
        ab = np.array(acc[f])
        bs = ab[:, 1].sum()
        if bs <= 0:
            continue
        c = ab[:, 0].sum() / bs
        se = math.sqrt(((ab[:, 0] - c * ab[:, 1]) ** 2).sum()) / bs
        res[f] = {"coef": float(c), "z": float(c / se) if se > 0 else 0.0}
    return res, (s2_num / s2_cnt if s2_cnt else None), s2_cnt


def run_tuning(prov, season, out_json, n_back=3):
    seasons = list(range(season - n_back, season + 1))
    ids = {}
    for s in seasons:
        try:
            players = prov.players(s)
        except Exception:  # noqa: BLE001
            continue
        for r in players:
            grp = str(r.get("position", "")).split()[:1]
            if grp and grp[0] in GROUP and float(r.get("time") or 0) >= 900:
                ids[str(r["id"])] = grp[0]
    print(f"Taratura: {len(ids)} giocatori di campo con almeno 900 minuti in una delle stagioni {seasons[0]}-{seasons[-1]}")
    data = {g: [] for g in GROUP}
    for i, (pid, g) in enumerate(ids.items()):
        try:
            ser = _series(prov.player_matches(pid), season - n_back)
        except Exception:  # noqa: BLE001
            continue
        if ser:
            data[g].append(ser)
        if (i + 1) % 100 == 0:
            print(f"  scaricati {i + 1}/{len(ids)}")
    out = {"aggiornato": datetime.now().strftime("%Y-%m-%d %H:%M"), "stagioni": seasons, "roles": {}}
    for g, series in data.items():
        if len(series) < 20:
            print(f"  ruolo {GROUP[g]}: pochi giocatori ({len(series)}), salto")
            continue
        tot_n = sum(m.sum() / 90.0 for m, *_ in series)
        priors = {"xgi": sum(x.sum() for _m, x, *_ in series) / tot_n, "shots": sum(s.sum() for _m, _x, s, _k in series) / tot_n,
                  "kp": sum(k.sum() for _m, _x, _s, k in series) / tot_n}
        grid = _mse_grid(series, priors["xgi"])
        bh, bK = min(grid, key=lambda k: grid[k])
        ft, s2, n_obs = _feature_tests(series, priors, bh, bK)
        c, z = ft.get("xgi", {}).get("coef", 0.0), ft.get("xgi", {}).get("z", 0.0)
        beta = round(min(c, 0.5), 3) if (c > 0 and z >= 2) else 0.0
        role = GROUP[g]
        out["roles"][role] = {
            "half_life": float(bh), "shrink": float(bK), "trend_beta": beta, "sigma2_90": round(s2, 4) if s2 else None,
            "mse_best": round(grid[(bh, bK)], 5), "mse_default": round(grid[(10, 6)], 5),
            "miglioramento_pct": round(100 * (grid[(10, 6)] - grid[(bh, bK)]) / grid[(10, 6)], 2),
            "mse_per_mezza_vita": {str(h): round(min(grid[(h, k)] for k in TUNE_K), 5) for h in TUNE_HL},
            "forma_recente": {f: {k: round(v, 3) for k, v in d.items()} for f, d in ft.items()},
            "n_giocatori": len(series), "n_osservazioni": int(n_obs)}
        print(f"  ruolo {role}: mezza vita {bh:g} partite, ristringimento {bK:g} (errore {grid[(bh, bK)]:.4f} contro "
              f"{grid[(10, 6)]:.4f} con i valori attuali, {out['roles'][role]['miglioramento_pct']:+.1f}%)")
        print(f"     forma recente (ultime 5 - media pesata): xGI coefficiente {c:+.2f} (z={z:+.1f}) -> "
              f"{'uso ' + str(beta) if beta else 'nessun peso aggiuntivo'}; tiri z={ft.get('shots', {}).get('z', 0):+.1f}, "
              f"passaggi chiave z={ft.get('kp', {}).get('z', 0):+.1f}")
    Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _trend_text(t):
    if not t:
        return ""
    txt = f"forma {t['arrow']}: xGI/90 {t['r']:.2f} (ultime 5) vs {t['b']:.2f} (30 prima), z {t['z']:+.1f}"
    return txt + (f", {t['mnote']}" if t["mnote"] else "")


def fv_ref(role, pr, lg):
    """Fantavoto di un giocatore MEDIO del ruolo in una partita neutra (definisce il '6' del voto previsto)."""
    fv = CFG["base_vote"] + CFG["goal"] * pr["xG"] + CFG["assist"] * pr["xA"]
    fv += CFG["yellow"] * pr["yellow_cards"] + CFG["red"] * pr["red_cards"]
    fv += CFG["clean_sheet"][role] * float(np.exp(-lg))
    if role == "P":
        fv += CFG["conceded_gk"] * lg
    return fv


def project(role, prof, ctx, ref, off=0.0, beta=0.0):
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
    res = ctx.get("pv", 0.0) - ctx.get("ps", 0.0)          # da -1 (sconfitta certa) a +1 (vittoria certa)
    res_eff = beta * res                                     # effetto risultato sul voto
    fv = fv_raw + off + res_eff
    # voto previsto: 6 = giocatore medio del ruolo in partita neutra (come i voti veri del fantacalcio)
    rating = float(np.clip(CFG["rating_center"] + CFG["rating_slope"] * (fv_raw + res_eff - ref), 1, 10))
    p_eff = min(1.0, prof["p_start"] + CFG["sub_weight"] * prof["p_sub"])
    ev = p_eff * fv + (1 - p_eff) * (ref + off)     # se non gioca, entra in media un sostituto "medio"
    return {"e_g": e_g, "e_a": e_a, "p_cs": p_cs, "fv": fv, "fv_raw": fv_raw, "rating": rating,
            "p_eff": p_eff, "ev": ev, "res": res, "res_eff": res_eff}


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
        odds_events=None, calib=None, odds_note=None, inj_key=None,
        inj_site="api-football.com", tune=None):
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
    beta = float((calib or {}).get("result_beta", CFG["result_beta_prior"]))

    fx = {}
    for m in fixtures:
        h, a = m["h"]["title"], m["a"]["title"]
        played = (not backtest) and (bool(m.get("isResult")) or to_local(m["datetime"]) < now)
        when_local = to_local(m["datetime"]).strftime("%Y-%m-%d %H:%M")
        fx[h] = (a, True, when_local, played, m["datetime"])
        fx[a] = (h, False, when_local, played, m["datetime"])
    n_played = sum(1 for v in fx.values() if v[3]) // 2
    odds_map = {} if backtest else attach_odds(odds_events, titles)
    inj_list, inj_msg, n_inj = [], ("non usati (backtest)" if backtest else "non attivi: manca il secret API_FOOTBALL_KEY"), 0
    if inj_key and not backtest and not check_only:
        try:
            raw_inj = fetch_injuries(inj_key, season, {m["datetime"][:10] for m in fixtures}, inj_site)
            inj_list = normalize_injuries(raw_inj, titles)
            inj_msg = "attivi"
            if is_apifootball_com(inj_site):
                inj_msg = "attivi via apifootball.com (solo infortunati, non squalificati)"
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace(inj_key, "***")
            low = msg.lower()
            if any(w in low for w in ("key", "token", "401", "403", "invalid", "unauthor")):
                hint = " - chiave non riconosciuta: controlla API_FOOTBALL_SITE (apifootball.com e api-football.com sono due servizi diversi)"
            elif "plan" in low or "season" in low:
                hint = " (il tuo piano potrebbe non includere questi dati: usa gli slider)"
            else:
                hint = ""
            inj_msg = f"non attivi: {msg[:110]}{hint}"
            print(f"  ! infortuni non disponibili ({msg})")
    tag = " [BACKTEST: solo dati precedenti]" if backtest else ""
    print(f"Giornata {rno}: {len(fixtures)} partite dal {to_local(fixtures[0]['datetime']):%Y-%m-%d %H:%M} (ora italiana) "
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
        opp, home, when, played, raw_dt = fx[team]
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
                                  if hit.id in cards_all.index else None, tune)
            nota = ("nessuna presenza recente" if prof["p_start"] + prof["p_sub"] == 0 else
                    "pochi dati" if prof["n_matches"] < 5 else "")
        if inj_list:
            e = find_injury([r.nome] + ([override] if override else []), team, inj_list)
            if e:
                n_inj += 1
                why = e["reason"] or e["type"]
                if "suspen" in why.lower() or "red card" in why.lower():
                    why = "squalificato"
                if e["type"].lower().startswith("missing"):
                    avail = False
                    nota = (nota + " - " if nota else "") + f"ASSENTE: {why}"
                else:
                    prof["p_start"] *= CFG["doubt_factor"]
                    prof["p_sub"] *= CFG["doubt_factor"]
                    nota = (nota + " - " if nota else "") + f"in dubbio: {why}"
        if played:
            nota = (nota + " - " if nota else "") + "partita gia' iniziata/giocata"
        odds = odds_map.get((team, opp) if home else (opp, team))
        ctx = team_ctx(team, opp, home, S, lg, hf, af, odds)
        pj = project(role, prof, ctx, refs[role], off.get(role, 0.0), beta)
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
            "Res": round(pj["res"], 3), "P_V": round(100 * ctx["pv"]), "P_N": round(100 * ctx["pn"]), "P_S": round(100 * ctx["ps"]),
            "KO": raw_dt.replace(" ", "T") + "Z",
            "Trend": (prof.get("trend") or {}).get("arrow", ""),
            "TrendTxt": _trend_text(prof.get("trend")),
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
    if inj_msg.startswith("attivi"):
        inj_msg += f" - {n_inj} segnalazioni sulla tua rosa" if inj_msg != "attivi" else f" ({n_inj} segnalazioni sulla tua rosa)"
    kicks = [{"t": m["datetime"].replace(" ", "T") + "Z", "m": f"{m['h']['title']}-{m['a']['title']}"} for m in fixtures]
    info = {"rno": rno, "odds": len(odds_map), "partite": len(fixtures), "odds_msg": odds_msg, "inj_msg": inj_msg, "kicks": kicks}
    if check_only:
        return None, None, problems, info
    return pd.DataFrame(rows), fixtures, problems, info


def save_predictions(df, info, fixtures, now, storico, force=False):
    """Salva le previsioni GREZZE (senza correzione) della giornata, finche' non inizia: servono alla calibrazione."""
    rno = info["rno"]
    if rno <= 0:
        return None
    kickoff = to_local(fixtures[0]["datetime"])
    if not force and now >= kickoff:
        return None                      # giornata gia' iniziata: non sovrascrivo le previsioni pre-partita
    d = Path(storico)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"G{rno:02d}.csv"
    df[["Giocatore", "Ruolo", "Squadra", "Avversario", "FV_raw", "Fantavoto", "Titolare%", "EV", "Disp", "Res"]].to_csv(f, index=False)
    return f


# ============================================================== FANTACALCIO.IT =
# Traccia l'affidabilita' delle probabili formazioni di Fantacalcio.it: i testi copiati (cartella formazioni/)
# vengono confrontati con i minuti realmente giocati (Understat). Il risultato corregge le percentuali incollate.
_MOD = re.compile(r"^\d(?:-\d){1,3}$")
_PCT = re.compile(r"^\d{1,3}\s*%$")
_UPD = re.compile(r"ultimo aggiornamento\s+(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{1,2}):(\d{2})", re.I)


def parse_fc_text(text):
    """Legge il testo copiato da Fantacalcio.it (stesso formato del lettore nella pagina).
    Ritorna una lista di partite: home, away, updated (ora italiana), entries per squadra, out, doubt."""
    L = [x.strip() for x in text.splitlines()]
    n, k, pending, upd, out = len(L), 0, [], None, []
    while k < n:
        if k + 1 < n and _MOD.match(L[k + 1]) and L[k] and not _PCT.match(L[k]) and not _MOD.match(L[k]):
            entries, j, bench = [], k + 2, False
            while j < n:
                cur = L[j]
                if re.match(r"(?i)ultimo aggiornamento", cur):
                    break
                if j + 1 < n and _MOD.match(L[j + 1]) and not _PCT.match(cur):
                    break
                if cur.lower() == "panchina":
                    bench = True
                    j += 1
                    continue
                if j + 1 < n and _PCT.match(L[j + 1]) and cur and not _PCT.match(cur):
                    entries.append({"name": cur, "pct": int(L[j + 1].replace("%", "").strip()), "bench": bench})
                    j += 2
                    continue
                j += 1
            pending.append({"team": L[k], "entries": entries})
            k = j
            continue
        m = _UPD.search(L[k])
        if m:
            d, mo, y, hh, mi = map(int, m.groups())
            upd = datetime(y, mo, d, hh, mi)
        if re.match(r"(?i)^dettaglio calciatori", L[k]):
            j, cur, o_names, d_names = k + 1, None, [], []
            while j < n and not re.match(r"(?i)^(stemma|campioncino)\b", L[j]) and not (j + 1 < n and _MOD.match(L[j + 1])):
                t = L[j]
                hm = re.match(r"(?i)^(ballottaggi|squalificati|diffidati|infortunati|in dubbio)$", t)
                if hm:
                    cur = hm.group(1).lower()
                elif cur and t and not re.match(r"(?i)^nessun", t) and not re.search(r"[,\d%]", t) and len(t) <= 32 and t[0].isupper():
                    if cur in ("squalificati", "infortunati"):
                        o_names.append(t)
                    elif cur == "in dubbio":
                        d_names.append(t)
                j += 1
            if len(pending) >= 2:
                a, b = pending[-2], pending[-1]
                out.append({"home": a["team"], "away": b["team"], "updated": upd, "home_entries": a["entries"],
                            "away_entries": b["entries"], "out": o_names, "doubt": d_names})
            pending, upd, k = [], None, j
            continue
        k += 1
    return out


def name_ok(fc_name, full_name):
    """'Miranda J.' / 'Esposito F.P.' / 'Milinkovic-Savic V.' contro il nome completo Understat."""
    parts, abbr = [], []
    for tok in strip_accents(fc_name).lower().split():
        if "." in tok:
            abbr += [x for x in re.sub(r"[^a-z.]", "", tok).split(".") if x]
        else:
            parts += tokens(tok)
    ct = tokens(full_name)
    if not parts or not all(p in ct for p in parts):
        return False
    rest = [t for t in ct if t not in parts]
    return all(any(t.startswith(a) for t in rest) for a in abbr)


def _roster_players(raw):
    """Understat rostersData -> {'h': [(nome, minuti)], 'a': [...]}"""
    out = {"h": [], "a": []}
    for side in ("h", "a"):
        d = (raw or {}).get(side) or {}
        for v in (d.values() if isinstance(d, dict) else d):
            try:
                out[side].append((str(v.get("player", "")), float(v.get("time") or 0)))
            except (TypeError, ValueError):
                continue
    return out


def _bucket(p):
    for lo, hi in ((100, 100), (60, 99), (50, 59), (30, 49), (10, 29), (0, 9)):
        if lo <= p <= hi:
            return lo, hi
    return 0, 9


def evaluate_fc(prov, season, fc_dir, storico_dir, roster):
    """Confronta le percentuali di Fantacalcio.it con chi ha davvero giocato. None se non ci sono dati utili."""
    files = sorted(Path(fc_dir).glob("*.txt"))
    if not files:
        return None
    matches_all = prov.matches(season)
    titles = sorted({m[s]["title"] for m in matches_all for s in ("h", "a")})
    idx = {(m["h"]["title"], m["a"]["title"]): m for m in matches_all}
    rounds = {id(m): no for no, g in split_rounds(matches_all) for m in g}
    best = {}                                            # id partita -> (aggiornamento, snapshot)
    for f in files:
        for sm in parse_fc_text(f.read_text(encoding="utf-8", errors="ignore")):
            h, a = resolve_team(sm["home"], titles), resolve_team(sm["away"], titles)
            m = idx.get((h, a))
            if not m or not m.get("isResult"):
                continue
            kick = to_local(m["datetime"])
            if sm["updated"] and sm["updated"] > kick:      # copiato a partita iniziata: non e' una previsione
                continue
            key = m["id"]
            if key not in best or (sm["updated"] or datetime.min) > (best[key][0] or datetime.min):
                best[key] = (sm["updated"], sm, m, h, a, kick)
    if not best:
        return None
    pool = pd.DataFrame(prov.players(season))[["player_name", "team_title"]]
    rows, out_n, out_wrong, leads = [], 0, 0, []
    my = {r.nome: r.squadra for r in roster.itertuples()}
    stor = {}
    for f in Path(storico_dir).glob("G*.csv"):
        d = pd.read_csv(f)
        stor[int(re.findall(r"\d+", f.stem)[0])] = {row["Giocatore"]: row["Titolare%"] for _, row in d.iterrows()}
    for key, (upd, sm, m, h, a, kick) in best.items():
        ros = _roster_players(prov.match_roster(key))
        if upd:
            leads.append((kick - upd).total_seconds() / 3600)
        for side, team, entries in (("h", h, sm["home_entries"]), ("a", a, sm["away_entries"])):
            known = pool[pool.team_title.astype(str).str.contains(re.escape(team), regex=True)].player_name.tolist()
            for e in entries:
                hit = next(((nm, mn) for nm, mn in ros[side] if name_ok(e["name"], nm)), None)
                if hit is not None:
                    minutes = hit[1]
                elif any(name_ok(e["name"], nm) for nm in known):
                    minutes = 0.0                            # lo conosce Understat ma non e' in campo: non ha giocato
                else:
                    continue                                 # sconosciuto (terzo portiere, nome diverso): escluso
                mine = next((nome for nome, sq in my.items() if resolve_team(sq, titles) == team
                             and (name_ok(nome, e["name"]) or name_ok(e["name"], nome))), None)
                mp = None
                if mine is not None and rounds.get(id(m)) in stor:
                    v = stor[rounds[id(m)]].get(mine)
                    mp = None if v is None else float(v)
                rows.append({"pct": e["pct"], "play": float(minutes >= 20), "start": float(minutes >= 60), "model": mp})
        for nm in sm["out"]:
            for side in ("h", "a"):
                hit = next(((x, mn) for x, mn in ros[side] if name_ok(nm, x)), None)
                if hit is not None:
                    out_n += 1
                    out_wrong += int(hit[1] >= 20)
                elif any(name_ok(nm, x) for x in pool.player_name):
                    out_n += 1
                    break
    if not rows:
        return None
    df = pd.DataFrame(rows)
    rep = {"n": int(len(df)), "matches": len(best), "brier_fc": float(((df.pct / 100 - df.play) ** 2).mean()),
           "brier_const": float(((df.play.mean() - df.play) ** 2).mean()), "buckets": [],
           "out_n": out_n, "out_wrong": out_wrong, "lead_h": float(np.mean(leads)) if leads else None, "roster": None}
    df["lo"], df["hi"] = zip(*df.pct.map(_bucket))
    for (lo, hi), g in df.groupby(["lo", "hi"]):
        rep["buckets"].append({"lo": int(lo), "hi": int(hi), "n": int(len(g)), "pct": float(g.pct.mean()),
                               "play": float(g.play.mean()), "start": float(g.start.mean())})
    sub = df.dropna(subset=["model"])
    if len(sub) >= 10:
        rep["roster"] = {"n": int(len(sub)), "brier_fc": float(((sub.pct / 100 - sub.play) ** 2).mean()),
                         "brier_model": float(((sub.model / 100 - sub.play) ** 2).mean())}
    fcmap = []
    for b in rep["buckets"]:
        if b["n"] >= 8:                                      # correzione prudente: si fida dei dati solo se abbastanza
            p = (b["n"] * b["play"] + CFG["fc_shrink"] * b["pct"] / 100) / (b["n"] + CFG["fc_shrink"])
            fcmap.append({"lo": b["lo"], "hi": b["hi"], "p": round(p, 3), "n": b["n"]})
    return rep, fcmap


# ============================================================== CALIBRAZIONE =
def read_voti(path):
    """voti_reali.csv: blocchi '# giornata: N' seguiti da righe 'giocatore,voto,fantavoto[,T|P]'.
    '-' o 'sv' = senza voto. Quarta colonna facoltativa: T = schierato titolare, P = in panchina."""
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
            fv = float("nan")                          # senza voto: lo tengo per sapere se era titolare
        if not g:
            bad += 1
            continue
        rows.append({"giornata": g, "Giocatore": p[0], "Reale": fv, "Sched": p[3].strip().upper() if len(p) > 3 else ""})
    if bad:
        print(f"  ! {bad} righe di {path} ignorate: manca il numero di giornata ('# giornata: N')")
    return pd.DataFrame(rows, columns=["giornata", "Giocatore", "Reale", "Sched"])


def _team_points(xi, bench, fv, role):
    """Somma dei fantavoti dei titolari; chi non ha voto viene sostituito (max 3) dal primo panchinaro dello stesso ruolo con voto."""
    tot, used, subs = 0.0, set(), 0
    for n in xi:
        if n in fv:
            tot += fv[n]
        elif subs < 3:
            for b in bench:
                if b not in used and role.get(b) == role.get(n) and b in fv:
                    tot += fv[b]
                    used.add(b)
                    subs += 1
                    break
    return tot


def pagella(pred, real):
    """Per ogni giornata: punti della formazione consigliata dal modello, di quella che hai schierato e del massimo possibile."""
    rounds = []
    for g, pg in pred.groupby("giornata"):
        rg = real[real.giornata == g]
        fv = {r.Giocatore: r.Reale for r in rg.itertuples() if pd.notna(r.Reale)}
        if not fv:
            continue
        role = dict(zip(pg.Giocatore, pg.Ruolo))
        dfm = pg.copy().reset_index(drop=True)
        dfm["Disp"] = dfm["Disp"].fillna(True).astype(bool) if "Disp" in dfm.columns else True
        item = {"giornata": int(g)}
        ln = best_lineup(dfm, MODULES)
        if ln:
            mod, _, xi, bench = ln
            item["modello"] = {"modulo": mod, "punti": round(_team_points(list(xi.Giocatore), list(bench.Giocatore), fv, role), 1)}
        t = [r.Giocatore for r in rg.itertuples() if r.Sched == "T" and r.Giocatore in role]
        if len(t) == 11:
            order = dfm.sort_values("EV", ascending=False).Giocatore.tolist()
            cnt = {k: sum(1 for x in t if role[x] == k) for k in "PDCA"}
            item["tua"] = {"modulo": f"{cnt['D']}-{cnt['C']}-{cnt['A']}",
                           "punti": round(_team_points(t, [x for x in order if x not in t], fv, role), 1)}
        best = None
        for mod in MODULES:
            d, c, a = map(int, mod.split("-"))
            tot, ok = 0.0, True
            for r, nn in (("P", 1), ("D", d), ("C", c), ("A", a)):
                vals = sorted((v for x, v in fv.items() if role.get(x) == r), reverse=True)
                if len(vals) < nn:
                    ok = False
                    break
                tot += sum(vals[:nn])
            if ok and (best is None or tot > best[1]):
                best = (mod, tot)
        if best:
            item["massimo"] = {"modulo": best[0], "punti": round(best[1], 1)}
        rounds.append(item)
    if not rounds:
        return None
    media = {k: round(float(np.mean([x[k]["punti"] for x in rounds if k in x])), 1)
             for k in ("modello", "massimo") if any(k in x for x in rounds)}
    media["n"] = len(rounds)
    common = [x for x in rounds if all(k in x for k in ("modello", "tua", "massimo"))]      # confronto alla pari: solo le giornate con la tua formazione
    comune = {k: round(float(np.mean([x[k]["punti"] for x in common])), 1) for k in ("modello", "tua", "massimo")} if common else None
    if comune:
        comune["n"] = len(common)
    return {"rounds": rounds, "media": media, "comune": comune}


def _result_lookup(prov, season):
    """(giornata, squadra della rosa) -> +1 vittoria, 0 pareggio, -1 sconfitta, dai risultati Understat."""
    ms = prov.matches(season)
    titles = sorted({m[s]["title"] for m in ms for s in ("h", "a")})
    res = {}
    for no, grp in split_rounds(ms):
        if not no:
            continue
        for m in grp:
            gl = m.get("goals") or {}
            if not m.get("isResult") or gl.get("h") in (None, "") or gl.get("a") in (None, ""):
                continue
            gh, ga = int(float(gl["h"])), int(float(gl["a"]))
            res[(no, m["h"]["title"])] = float(np.sign(gh - ga))
            res[(no, m["a"]["title"])] = float(np.sign(ga - gh))
    return lambda g, sq: res.get((int(g), resolve_team(sq, titles)))


def _calibra_votes(storico_dir, voti_path, out_json, lookup=None):
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
    pred, real_all = pd.concat(frames), read_voti(voti_path)
    minr = CFG["calib_min_round"]
    out["pagella"] = pagella(pred[pred.giornata >= minr], real_all)          # le prime giornate (rodaggio) non entrano nel giudizio
    real = real_all.dropna(subset=["Reale"])
    df_all = pred.merge(real, on=["giornata", "Giocatore"], how="inner")
    if df_all.empty:
        print("Calibrazione: nessun giocatore in comune fra previsioni e voti reali.")
        Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out
    # Rodaggio: nelle prime giornate il modello ha pochissime partite della stagione in corso (nuovi acquisti e neopromosse
    # sconosciuti). I suoi errori li' sono di natura diversa da quelli di oggi: non li uso per correggere il modello
    # ne' per giudicarlo. Le uso solo per l'effetto risultato (sotto), che non dipende da questo problema.
    df = df_all[df_all.giornata >= minr].copy()
    excl = sorted(int(x) for x in set(df_all.giornata) - set(df.giornata))
    rep = {"n": int(len(df)), "escluse": excl, "giornate": sorted(int(x) for x in df.giornata.unique()) if len(df) else [],
           "bias": {}, "n_ruolo": {}}
    if len(df):
        df["err"] = df["FV_raw"] - df["Reale"]
        rep["mae_modello"] = float(df.err.abs().mean())
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
    # effetto risultato: quanto il voto residuo (reale - previsto) dipende da vittoria/pareggio/sconfitta effettivi.
    # Usa tutte le giornate, con gli scarti calcolati rispetto alla media del ruolo (quindi il rodaggio non lo distorce).
    if lookup is not None and "Squadra" in df_all.columns:
        d2 = df_all.copy()
        d2["x"] = [lookup(g_, sq) for g_, sq in zip(d2.giornata, d2.Squadra)]
        d2 = d2.dropna(subset=["x"])
        if len(d2) >= 8:
            d2["r"] = d2.Reale - d2.FV_raw
            rdm = d2.r - d2.groupby("Ruolo").r.transform("mean")
            xdm = d2.x - d2.groupby("Ruolo").x.transform("mean")
            den = float((xdm ** 2).sum())
            ols = float((xdm * rdm).sum() / den) if den > 0 else 0.0
            n_ = len(d2)
            beta = (n_ * ols + CFG["result_beta_shrink"] * CFG["result_beta_prior"]) / (n_ + CFG["result_beta_shrink"])
            out["result_beta"] = round(beta, 3)
            rep["result_beta_ols"], rep["result_n"] = round(ols, 3), int(n_)
            print(f"  effetto risultato: dai dati {ols:+.2f} fantavoto per vittoria (vs sconfitta = doppio) su {n_} prestazioni; "
                  f"uso {beta:.2f} (prudente, mescolato al valore di partenza {CFG['result_beta_prior']})")
    out["report"] = rep
    Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if excl:
        print(f"  giornate {excl} in rodaggio: escluse dalla correzione e dal giudizio del modello (usate solo per l'effetto risultato)")
    if len(df):
        print(f"Calibrazione su {rep['n']} prestazioni reali, giornate {rep['giornate']}")
        print(f"  errore medio modello (MAE): {rep['mae_modello']:.2f}   media del ruolo: {rep['mae_media_ruolo']:.2f}")
        if "mae_media_giocatore" in rep:
            print(f"  (su {rep['n_sub']} righe con storico) modello {rep['mae_modello_sub']:.2f} "
                  f"vs media dei suoi fantavoti {rep['mae_media_giocatore']:.2f}")
        print(f"  scarto medio per ruolo (previsto - reale): {rep['bias']}")
        print(f"  correzione applicata: {out['offset']}   (prudente: si attenua con pochi dati)")
    else:
        print("Calibrazione: nessuna giornata oltre il rodaggio, nessuna correzione applicata.")
    return out


def calibra(storico_dir, voti_path, out_json, fc_dir=None, prov=None, season=None, roster=None):
    lookup = None
    if prov is not None and season:
        try:
            lookup = _result_lookup(prov, season)
        except Exception as e:  # noqa: BLE001
            print(f"  ! risultati delle partite non disponibili, salto l'effetto risultato ({type(e).__name__})")
    out = _calibra_votes(storico_dir, voti_path, out_json, lookup)
    if out.get("pagella"):
        pg = out["pagella"]
        print("Pagella della formazione (somma dei fantavoti dei titolari):")
        for r in pg["rounds"]:
            print("  G%d: " % r["giornata"] + " | ".join(f"{k} {r[k]['punti']} ({r[k]['modulo']})" for k in ("modello", "tua", "massimo") if k in r))
        print("  media su tutte le giornate:", pg["media"], "| dove c'e' anche la tua formazione:", pg.get("comune"))
    if fc_dir and prov is not None and Path(fc_dir).exists() and list(Path(fc_dir).glob("*.txt")):
        try:
            res = evaluate_fc(prov, season, fc_dir, storico_dir, roster)
        except Exception as e:  # noqa: BLE001
            res = None
            print(f"  ! valutazione Fantacalcio.it non riuscita ({type(e).__name__}: {e})")
        if res:
            rep, fcmap = res
            out["fc"], out["fc_map"] = rep, fcmap
            print(f"Fantacalcio.it: {rep['n']} giocatori in {rep['matches']} partite, errore (Brier) {rep['brier_fc']:.3f} "
                  f"contro {rep['brier_const']:.3f} di chi prevedesse sempre la media")
            for b in rep["buckets"]:
                print(f"   {b['lo']}-{b['hi']}%: {b['n']} giocatori, ha giocato il {b['play']*100:.0f}% (dato medio {b['pct']:.0f}%)")
            if rep["roster"]:
                print(f"   sui tuoi giocatori: Fantacalcio.it {rep['roster']['brier_fc']:.3f} vs modello {rep['roster']['brier_model']:.3f}")
        else:
            print("Fantacalcio.it: nessuna partita conclusa con un testo salvato in formazioni/ (copiato prima del calcio d'inizio).")
        Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ============================================================== STATISTICHE.XLSX ===
# Le statistiche ufficiali di Fantacalcio.it (foglio "Tutti" del file scaricabile dal sito) coprono TUTTA la Serie A,
# con voto medio, fantamedia e - soprattutto - i rigori calciati (Rc) e segnati (R+): l'unico modo per sapere chi
# tira davvero i rigori oggi, cosa che Understat da solo non dice.
FC_COLS = {"Id": "id_fc", "R": "ruolo_fc", "Nome": "nome_fc", "Squadra": "squadra_fc", "Pv": "Pv", "Mv": "Mv",
           "Fm": "Fm", "Gf": "Gf", "Gs": "Gs", "Rp": "Rp", "Rc": "Rc", "R+": "Rpiu", "R-": "Rmeno",
           "Ass": "AssReali", "Amm": "AmmReali", "Esp": "EspReali", "Au": "Au"}


def read_fc_stats(path):
    """Legge il foglio 'Tutti' del file statistiche di Fantacalcio.it. None se il file non ha il formato atteso."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    sheet = "Tutti" if "Tutti" in wb.sheetnames else wb.sheetnames[0]
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header_row = next((i for i, r in enumerate(rows) if r and r[0] == "Id"), None)
    if header_row is None:
        return None
    header = rows[header_row]
    df = pd.DataFrame(rows[header_row + 1:], columns=header)
    df = df.dropna(subset=["Nome"])
    keep = [c for c in FC_COLS if c in df.columns]
    df = df[keep].rename(columns=FC_COLS)
    for c in ("Pv", "Gf", "Gs", "Rp", "Rc", "Rpiu", "Rmeno", "AssReali", "AmmReali", "EspReali", "Au", "Mv", "Fm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def match_fc_stats(fc_df, pcur, titles):
    """Abbina ogni riga del file Fantacalcio.it a un giocatore Understat (per nome e squadra). Ritorna
    id Understat -> statistiche reali. I non abbinati (nomi molto diversi tra i due siti) restano fuori."""
    pool = pcur[["id", "player_name", "team_title", "time"]].assign(tag=0)
    out = {}
    for r in fc_df.itertuples():
        team = resolve_team(str(r.squadra_fc), titles)
        hit = find_player(str(r.nome_fc), None, team or "", pool)
        if hit is not None:
            out[str(hit.id)] = r._asdict()
    return out


def latest_fc_stats(folder):
    files = sorted(Path(folder).glob("*.xlsx"), key=lambda f: f.stat().st_mtime)
    if not files:
        return None, None
    f = files[-1]
    try:
        return read_fc_stats(f), f.name
    except Exception as e:  # noqa: BLE001
        print(f"  ! non riesco a leggere {f.name}: {e}")
        return None, f.name


# ============================================================== MERCATO =====
MKT_GROUP = {"F": "A", "M": "C", "D": "D"}


def _bucket3(values, value, higher_is_better=True):
    """Colloca 'value' tra i terzili di 'values' (tutte le squadre) -> 'Facile'/'Nella media'/'Difficile' e simili."""
    vs = sorted(v for v in values if v is not None)
    if len(vs) < 6 or value is None:
        return "n/d"
    lo, hi = vs[len(vs) // 3], vs[2 * len(vs) // 3]
    good, mid, bad = ("Facile", "Nella media", "Difficile") if higher_is_better else ("Difficile", "Nella media", "Facile")
    return good if value >= hi else bad if value < lo else mid


def _fixture_comfort(opp, home, S, hf, af):
    """Quanto e' comoda una partita: per un attaccante (att) e per un difensore (def), dato l'avversario e il campo."""
    d_opp, a_opp = (S.get(opp) or {"A": 1, "D": 1})["D"], (S.get(opp) or {"A": 1, "D": 1})["A"]
    return d_opp * (hf if home else af), 1.0 / max(a_opp * (af if home else hf), 0.05)


def team_outlook(matches_all, S, lg, hf, af, now, n=5):
    """Per ogni squadra: qualita' offensiva propria, difficolta' del calendario nelle prossime n partite E
    di quelle gia' giocate in questa stagione, sia per attaccare (alto = comodo per attaccanti/centrocampisti)
    sia per difendere (alto = comodo per difensori/portieri, cioe' avversari deboli in attacco)."""
    teams = sorted(S.keys())
    fut, past = {t: [] for t in teams}, {t: [] for t in teams}
    for m in matches_all:
        h, a = m["h"]["title"], m["a"]["title"]
        played = bool(m.get("isResult"))
        bucket = past if played else (fut if to_local(m["datetime"]) >= now else None)
        if bucket is None:
            continue
        for t, opp, home in ((h, a, True), (a, h, False)):
            if t in bucket:
                bucket[t].append((opp, home))
    raw, raw_past = {}, {}
    for t in teams:
        nxt = fut[t][:n]
        if nxt:
            att, dfc = zip(*(_fixture_comfort(o, h, S, hf, af) for o, h in nxt))
            raw[t] = {"att": float(np.mean(att)), "def": float(np.mean(dfc)), "opps": [o for o, _ in nxt]}
        else:
            raw[t] = {"att": None, "def": None, "opps": []}
        gia = past[t]                                        # tutte le partite gia' giocate in stagione
        if gia:
            att, dfc = zip(*(_fixture_comfort(o, h, S, hf, af) for o, h in gia))
            raw_past[t] = {"att": float(np.mean(att)), "def": float(np.mean(dfc)), "n": len(gia)}
        else:
            raw_past[t] = {"att": None, "def": None, "n": 0}
    all_att = [v["att"] for v in raw.values()]
    all_def = [v["def"] for v in raw.values()]
    all_A = [s["A"] for s in S.values()]
    med_A = float(np.median(all_A)) if all_A else 1.0
    all_att_p = [v["att"] for v in raw_past.values()]
    all_def_p = [v["def"] for v in raw_past.values()]
    out = {}
    for t in teams:
        out[t] = {"attacco_label": "Forte" if S[t]["A"] >= med_A * 1.08 else "Debole" if S[t]["A"] < med_A * 0.92 else "Nella media",
                  "cal_att": _bucket3(all_att, raw[t]["att"]), "cal_def": _bucket3(all_def, raw[t]["def"]),
                  "prossimi": raw[t]["opps"],
                  "cal_att_finora": _bucket3(all_att_p, raw_past[t]["att"]) if raw_past[t]["n"] >= 3 else "n/d",
                  "cal_def_finora": _bucket3(all_def_p, raw_past[t]["def"]) if raw_past[t]["n"] >= 3 else "n/d",
                  "n_giocate": raw_past[t]["n"]}
    return out


def market_report(pcur, priors, roster, titles, S=None, outlook=None, fc_stats=None, min_minutes=None):
    if "goals" not in pcur.columns or "assists" not in pcur.columns:
        return None, "Understat non fornisce i gol/assist reali in questo formato: analisi di mercato non disponibile."
    min_minutes = min_minutes or CFG["mercato_min_minutes"]
    search_min = min(CFG["mercato_search_min"], min_minutes)     # per la ricerca includo anche chi ha giocato poco
    has_np = "npg" in pcur.columns and "npxG" in pcur.columns   # gol/xG al netto dei rigori, se Understat li fornisce
    cols = ["goals", "assists"] + (["npg", "npxG"] if has_np else [])
    df = num(pcur.copy(), cols)
    df["role"] = df["position"].astype(str).str.split().str[0].map(MKT_GROUP)
    df = df.dropna(subset=["role"])
    df = df[df.time >= search_min].reset_index(drop=True)
    if df.empty:
        return None, "Nessun giocatore con abbastanza minuti ancora in questa stagione."
    if has_np:
        df["rigori_segnati"], df["fonte_np"] = (df.goals - df.npg).clip(lower=0), "Understat"
        g_col, xg_col = "npg", "npxG"       # esclude i rigori dal confronto: sono affidabili, non "fortuna"
    else:
        df["rigori_segnati"], df["fonte_np"] = 0, None
        g_col, xg_col = "goals", "xG"
    K = CFG["player_shrink_90s"]
    pool = pcur[["id", "player_name", "team_title", "time"]].assign(tag=0)
    mine = set()
    for r in roster.itertuples():
        team = resolve_team(r.squadra, titles)
        override = getattr(r, "understat", None)
        override = override if isinstance(override, str) and override.strip() else None
        hit = find_player(r.nome, override, team or "", pool)
        if hit is not None:
            mine.add(str(hit.id))
    n_fc_np = 0
    rows = []
    for r in df.itertuples():
        pr = priors.get(r.role)
        if pr is None:
            continue
        gol_reali, xg_ref, rigorista, fonte = getattr(r, g_col), getattr(r, xg_col), False, r.fonte_np
        fc = (fc_stats or {}).get(str(r.id))
        if fc:                                        # dati ufficiali Fantacalcio.it: piu' precisi e piu' aggiornati di Understat
            gol_reali = max(0, fc["Gf"] - fc["Rpiu"])   # gol al netto dei rigori davvero segnati (non stimati)
            rigorista = fc["Rc"] > 0
            fonte = "Fantacalcio.it"
            n_fc_np += 1
        atteso = 3 * ((r.xG * 90 + K * pr["xG"]) / (r.time + K * 90)) + 1 * ((r.xA * 90 + K * pr["xA"]) / (r.time + K * 90))
        reale = 3 * ((gol_reali * 90 + K * pr["xG"]) / (r.time + K * 90)) + 1 * ((r.assists * 90 + K * pr["xA"]) / (r.time + K * 90))
        o = (outlook or {}).get(r.team_title, {})
        rows.append({"id": str(r.id), "Giocatore": r.player_name, "Squadra": r.team_title, "Ruolo": r.role,
                     "Minuti": int(r.time), "Gol": int(r.goals), "Rigori": int(r.rigori_segnati), "xG": round(xg_ref, 1),
                     "Assist": int(r.assists), "xA": round(r.xA, 1), "Atteso90": round(atteso, 3), "Reale90": round(reale, 3),
                     "Fortuna": round(reale - atteso, 3), "Tua": str(r.id) in mine, "Rigorista": rigorista, "FonteNP": fonte,
                     "MvReale": round(fc["Mv"], 2) if fc and fc["Mv"] else None, "FmReale": round(fc["Fm"], 2) if fc and fc["Fm"] else None,
                     "AttaccoSquadra": o.get("attacco_label", "n/d"),
                     "Calendario": o.get("cal_def" if r.role == "D" else "cal_att", "n/d"),
                     "CalendarioFinora": o.get("cal_def_finora" if r.role == "D" else "cal_att_finora", "n/d"),
                     "Affidabile": r.time >= min_minutes})
    rep = pd.DataFrame(rows)
    rep["_med"] = rep.groupby("Ruolo")["Atteso90"].transform("median")

    def _verdetto(r):
        bits = []
        if not r.Affidabile:
            bits.append("Pochi minuti finora: giudizio ancora incerto.")
        if r.Rigorista:
            bits.append("Rigorista designato.")
        if r.Fortuna <= -CFG["mercato_luck_soglia"]:
            bits.append("Sfortunato: le sue occasioni valgono pi\u00f9 dei suoi gol/assist reali, probabile miglioramento." +
                        (" Se \u00e8 tuo, non cederlo ora." if r.Tua else " Possibile obiettivo di mercato."))
        elif r.Fortuna >= CFG["mercato_luck_soglia"] and r.Atteso90 <= r._med:
            bits.append("Sta rendendo sopra le sue occasioni: rischio di calo." +
                        (" Se \u00e8 tuo, valuta di cederlo adesso." if r.Tua else " Occhio se te lo offrono in cambio."))
        else:
            bits.append("Rendimento in linea con le sue occasioni.")
        return " ".join(bits)

    rep["Verdetto"] = rep.apply(_verdetto, axis=1)
    out = {}
    for role, g_all in rep.groupby("Ruolo"):
        g = g_all[g_all.Affidabile].sort_values("Atteso90", ascending=False)
        med = g["Atteso90"].median() if len(g) else 0.0
        obiettivi = g[(~g.Tua) & (g.Fortuna <= -CFG["mercato_luck_soglia"])].sort_values("Fortuna").head(8)
        sfortunati_tuoi = g[g.Tua & (g.Fortuna <= -CFG["mercato_luck_soglia"] / 2)].sort_values("Fortuna")
        fortunati_tuoi = g[g.Tua & (g.Fortuna >= CFG["mercato_luck_soglia"] / 2) & (g.Atteso90 <= med)].sort_values("Fortuna", ascending=False)
        top_processo = g.head(10)
        out[role] = {"top": top_processo.to_dict("records"), "obiettivi": obiettivi.to_dict("records"),
                     "sfortunati_tuoi": sfortunati_tuoi.to_dict("records"), "fortunati_tuoi": fortunati_tuoi.to_dict("records"),
                     "n": len(g), "mediana_atteso": round(float(med), 3)}
    out["_penalty_note"] = has_np
    out["_fc_n"] = n_fc_np
    out["_all"] = rep.drop(columns=["_med"]).sort_values("Giocatore").to_dict("records")
    return out, None


def to_html_mercato(report, err, season, now):
    ROLE_NAME = {"D": "Difensori", "C": "Centrocampisti", "A": "Attaccanti"}
    css = ("body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:0 12px 24px;background:#fff;color:#111;max-width:760px;margin:auto}"
           "@media(prefers-color-scheme:dark){body{background:#111318;color:#eee}input,select{background:#1c1f27;color:#eee;border-color:#fff3}}"
           "h1{font-size:20px;margin:14px 0 2px}h2{font-size:16px;margin:22px 0 6px}h3{font-size:14px;margin:14px 0 4px;opacity:.85}"
           ".s{opacity:.6;font-size:12px}table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:6px;display:block;overflow-x:auto}"
           "th,td{padding:5px 7px;border-bottom:1px solid #8884;text-align:left;white-space:nowrap}"
           ".pos{color:#16a34a;font-weight:600}.neg{color:#dc2626;font-weight:600}.tua{background:#2563eb22}"
           "#search{width:100%;font-size:15px;padding:9px 10px;border-radius:8px;border:1px solid #8886;box-sizing:border-box}"
           "#card{background:#8881 22;border-radius:10px;padding:10px 12px;margin-top:8px}"
           "#card{background:#2563eb14;border-radius:10px;padding:10px 12px;margin-top:8px;display:none}"
           "#card b{font-size:15px}.badge{display:inline-block;border-radius:8px;padding:2px 8px;font-size:12px;margin-left:6px;color:#fff}"
           ".row2{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin:6px 0}.row2 div{min-width:90px}"
           "#matches{font-size:13px;margin-top:4px}#matches div{padding:3px 0;cursor:pointer;color:#2563eb}")
    body = (f"<h1>Mercato</h1><p class=s>Stagione {season}/{str(season + 1)[-2:]} \u00b7 aggiornato {now:%d/%m/%Y %H:%M} \u00b7 "
            "confronta occasioni (xG/xA, senza rigori) e gol/assist reali di tutta la Serie A, con la forza offensiva della "
            "squadra e il calendario delle prossime 5 partite. Non conosco le rose degli altri della tua lega: controlla tu "
            "chi \u00e8 davvero libero prima di fare un\u2019offerta.</p>"
            "<h2>Cerca un giocatore</h2>"
            "<input id=search list=plist autocomplete=off placeholder=\"Scrivi un nome, es. Hojlund...\">"
            "<datalist id=plist></datalist><div id=matches></div><div id=card></div>")
    if err:
        body += f"<p>{html.escape(err)}</p>"
    else:
        if report.get("_fc_n"):
            body += (f"<p class=s>Rigori e rendimento reale di {report['_fc_n']} giocatori presi dal file statistiche di "
                     "Fantacalcio.it (pi\u00f9 precisi di Understat); per gli altri, dati Understat.</p>")
        elif not report.get("_penalty_note"):
            body += "<p class=s>Understat non fornisce qui il dato senza rigori: il confronto include anche i rigori segnati.</p>"
        else:
            body += ("<p class=s>Carica il file statistiche di Fantacalcio.it in statistiche/ per identificare i rigoristi "
                     "attuali e usare il voto reale invece di una stima. Vedi le istruzioni che Claude ti ha dato.</p>")
        for role, d in report.items():
            if role.startswith("_"):
                continue
            body += f"<h2>{ROLE_NAME.get(role, role)} ({d['n']} giocatori con abbastanza minuti)</h2>"

            def table(rows, cols):
                if not rows:
                    return "<p class=s>Nessuno al momento.</p>"
                h = "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
                for r in rows:
                    cls = " class=tua" if r.get("Tua") else ""
                    h += f"<tr{cls}>"
                    for c in cols:
                        v = r.get(c, "")
                        style = ""
                        if c == "Fortuna" and isinstance(v, (int, float)):
                            style = ' class="pos"' if v > 0 else ' class="neg"' if v < 0 else ""
                            v = f"{v:+.2f}"
                        if c == "Giocatore" and r.get("Rigorista"):
                            v = str(v) + " \U0001F3AF"
                        h += f"<td{style}>{html.escape(str(v))}</td>"
                    h += "</tr>"
                return h + "</table>"

            cols = ["Giocatore", "Squadra", "AttaccoSquadra", "Calendario", "Minuti", "Gol", "Rigori", "xG", "Assist", "xA", "Fortuna"]
            if any(x.get("MvReale") is not None for x in d["top"] + d["obiettivi"] + d["sfortunati_tuoi"] + d["fortunati_tuoi"]):
                cols = cols[:6] + ["MvReale", "FmReale"] + cols[6:]
            body += "<h3>Da valutare in acquisto (occasioni buone, sotto-rendimento reale)</h3>" + table(d["obiettivi"], cols)
            body += "<h3>Nella tua rosa, sfortunati: non cederli</h3>" + table(d["sfortunati_tuoi"], cols)
            body += "<h3>Nella tua rosa, da valutare in cessione (sopra le loro occasioni)</h3>" + table(d["fortunati_tuoi"], cols)
            body += "<h3>I migliori per occasioni create (indipendentemente dalla fortuna)</h3>" + table(d["top"], cols)
        body += ('<p class="s">"Fortuna" = (gol reali/90 - gol attesi/90, senza rigori quando disponibile)\u00d73 + '
                 '(assist reali/90 - assist attesi/90), ristretti verso la media del ruolo con pochi minuti. Positiva = sta '
                 'segnando pi\u00f9 di quanto meriti (rischio di calo); negativa = meno (probabile miglioramento). '
                 '"AttaccoSquadra" = forza offensiva della sua squadra nella stagione. "Calendario" = difficolt\u00e0 delle '
                 'prossime 5 partite (per i difensori conta la fase difensiva, per gli altri quella offensiva). '
                 'Righe evidenziate = giocatori della tua rosa. I rigoristi attuali non sono identificati: la colonna "Rigori" '
                 'mostra solo quanti ne ha gi\u00e0 segnati in stagione.</p>')
    js = ""
    if not err:
        payload = json.dumps(report.get("_all", []), ensure_ascii=False).replace("</", "<\\/")
        js = ("<script>const ALL=" + payload + ";"
              "const dl=document.getElementById('plist');ALL.forEach(p=>{const o=document.createElement('option');o.value=p.Giocatore;dl.appendChild(o);});"
              "const colf=v=>v>0?'#16a34a':v<0?'#dc2626':'#6b7280';"
              "function showCard(p){const c=document.getElementById('card');c.style.display='block';"
              "const rig=p.Rigorista?' <span class=badge style=\"background:#f59e0b\">rigorista</span>':'';"
              "const voti=(p.MvReale!=null)?(' \u00b7 voto medio reale '+p.MvReale+' \u00b7 fantamedia '+p.FmReale):'';"
              "c.innerHTML='<b>'+p.Giocatore+rig+'</b><div class=s>'+p.Squadra+' ('+p.Ruolo+') \u00b7 '+p.Minuti+' minuti'+voti+'</div>"
              "<div class=row2><div>Gol: <b>'+p.Gol+'</b></div><div>xG: <b>'+p.xG+'</b></div><div>Assist: <b>'+p.Assist+'</b></div>"
              "<div>xA: <b>'+p.xA+'</b></div><div>Rigori segnati: <b>'+p.Rigori+'</b></div></div>"
              "<div class=row2><div>Attacco squadra: <b>'+p.AttaccoSquadra+'</b></div><div>Prossime 5: <b>'+p.Calendario+'</b></div>"
              "<div>Fortuna: <b style=\"color:'+colf(p.Fortuna)+'\">'+(p.Fortuna>0?'+':'')+p.Fortuna.toFixed(2)+'</b></div></div>"
              "<div class=s style=\"margin-top:2px\">Avversari affrontati finora: <b>'+p.CalendarioFinora+'</b> (statistiche ottenute contro un calendario '+"
              "(p.CalendarioFinora==='Difficile'?'difficile: probabilmente vale anche di pi\u00f9':p.CalendarioFinora==='Facile'?'comodo: valutalo con un po\u2019 di cautela':'nella media')+')</div>"
              "<div style=\"margin-top:6px\">'+p.Verdetto+'</div>';}"
              "function search(){const q=document.getElementById('search').value.trim().toLowerCase();const M=document.getElementById('matches');"
              "M.innerHTML='';document.getElementById('card').style.display='none';if(!q)return;"
              "const norm=s=>s.normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/\u00f8/gi,'o').replace(/\u00e6/gi,'ae').replace(/\u0142/gi,'l').replace(/\u0111/gi,'d').replace(/\u00df/g,'ss').toLowerCase();const nq=norm(q);"
              "const hits=ALL.filter(p=>norm(p.Giocatore).includes(nq));"
              "if(hits.length===1){showCard(hits[0]);return;}"
              "hits.slice(0,8).forEach(p=>{const d=document.createElement('div');d.textContent=p.Giocatore+' \u2013 '+p.Squadra+' ('+p.Ruolo+')';"
              "d.onclick=()=>{document.getElementById('search').value=p.Giocatore;showCard(p);M.innerHTML='';};M.appendChild(d);});"
              "if(!hits.length)M.innerHTML='<div class=s>Nessun giocatore trovato (prova con solo il cognome).</div>';}"
              "document.getElementById('search').addEventListener('input',search);"
              "document.getElementById('search').addEventListener('change',search);</script>")
    return (f"<!doctype html><html lang=it><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'><title>Mercato</title>"
            f"<style>{css}</style></head><body>{body}{js}</body></html>")


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
.ln b{min-width:16px}.tap{cursor:pointer;border-radius:8px;padding:6px 4px}.tap:active{background:var(--line)}.tag{font-size:12px;color:var(--mut)}
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
textarea{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;font-size:13px;font-family:inherit}
details summary{cursor:pointer;font-weight:600;font-size:14px}
</style></head><body>
<h1>Fanta Predictor</h1>
<div class="s" id="meta"></div>
<div id="bar">
  <div class="bh"><b id="lt">Formazione</b>
    <select id="mod" aria-label="Modulo"></select><button id="pbtn">Incolla elenco</button><button id="copy">Copia</button></div>
  <div class="mini" id="mini"></div>
  <div class="s" id="dl" style="margin-top:3px"></div>
</div>
<h2>Dettaglio formazione</h2>
<div class="s" style="margin-bottom:6px">Tocca un titolare per mandarlo in panchina, o un panchinaro per farlo giocare: al suo posto entra il migliore disponibile. &#128274; = scelta tua, tocca di nuovo per tornare in automatico.</div>
<div class="info" id="warn" style="display:none;color:#d97706;margin-bottom:6px"></div>
<div class="card" id="lineup"></div>
<h2>Panchina (in ordine)</h2><div class="card" id="bench"></div>
<div id="probs" class="card" style="display:none"></div>
<div id="cal" class="card" style="display:none"></div>
<div id="pag" class="card" style="display:none"></div>
<h2>Titolarit&agrave; e infortuni <button id="reset" style="float:right">Azzera</button></h2>
<div class="s" style="margin-bottom:8px">Muovi lo slider con le percentuali delle probabili formazioni: la formazione si ricalcola subito. Le modifiche restano salvate su questo dispositivo fino alla giornata successiva.</div>
<details class="card" id="pdet"><summary>Incolla un elenco (infortunati, squalificati, probabili formazioni)</summary>
  <div class="s" style="margin:6px 0">Incolla il testo di un sito con le probabili formazioni, anche pi&ugrave; partite insieme: riconosco i blocchi tipo &quot;MILAN (3-4-2-1): Maignan; Gila, ...&quot; (titolari 90%, alternative con &quot;/&quot; 50%, gli altri della tua squadra 15%), i &quot;Ballottaggi ... 55%-45%&quot; e gli &quot;Indisponibili/Squalificati: ...&quot;. Vanno bene anche righe singole come &quot;Maignan 100%&quot; o &quot;Pulisic infortunato&quot;. Con &quot;Salva su GitHub&quot; conservi il testo di Fantacalcio.it per misurare, giornata dopo giornata, quanto &egrave; affidabile.</div>
  <textarea id="paste" rows="6" placeholder="Maignan 100%&#10;Pulisic infortunato&#10;Lucum&igrave; squalificato"></textarea>
  <div class="bh" style="margin-top:6px"><select id="pmode"><option value="auto">Riconosci dalla riga</option><option value="out">Sono tutti indisponibili</option><option value="start">Sono tutti probabili titolari</option></select>
  <button id="apply">Applica</button><button id="gh">Salva su GitHub</button></div><div class="info" id="pout"></div></details>
<div id="roster"></div>
<p class="s" id="ver"></p>
<p class="s" id="beta"></p>
<p class="s">Voto previsto: 6 = giocatore medio del suo ruolo in una partita neutra (come i voti veri); +1 fantavoto rispetto alla media = +1,5 voti. Vale SE il giocatore scende in campo. Gol/assist % = probabilit&agrave; di almeno un gol/assist. Valore atteso = P(gioca) x fantavoto + (1 - P) x sostituto medio. Modello statistico, non una garanzia.</p>
<script>
const D = __DATA__;
const KEY = "fanta_v2_" + D.giornata;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let modSel = saved._mod || "auto", cur = null, inSet = new Set();
const S = D.players.map(p => Object.assign({}, p, {
  p0: p["Titolare%"], p: (saved[p.Giocatore] && saved[p.Giocatore].p != null) ? saved[p.Giocatore].p : p["Titolare%"],
  disp: (saved[p.Giocatore] && saved[p.Giocatore].disp != null) ? saved[p.Giocatore].disp : !!p.Disp,
  force: (saved[p.Giocatore] && saved[p.Giocatore].f) || 0 }));            // 1 = fisso titolare, -1 = fisso in panchina
const ev = s => (s.p/100)*s.Fantavoto + (1 - s.p/100)*s.Rif;
const col = v => "hsl(" + (120*(Math.max(3,Math.min(9,v))-3)/6).toFixed(0) + ",62%,40%)";
function save(){ const o = {}; S.forEach(s => { if (s.p !== s.p0 || s.disp !== !!s.Disp || s.force) o[s.Giocatore] = {p:s.p, disp:s.disp, f:s.force}; });
  if (modSel !== "auto") o._mod = modSel; try { localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) {} }
function calc(mods, useForce){
  let best = null;
  for (const mod of mods){
    const [d,c,a] = mod.split("-").map(Number), need = {P:1, D:d, C:c, A:a};
    let tot = 0, pick = [], ok = true;
    for (const r of ["P","D","C","A"]){
      const fin = useForce ? S.filter(s => s.Ruolo === r && s.force === 1) : [];          // scelte fisse: giocano comunque
      if (fin.length > need[r]) { ok = false; break; }
      let g = S.filter(s => s.Ruolo === r && s.disp && !fin.includes(s) && !(useForce && s.force === -1));
      const gok = g.filter(s => s.p >= D.minp*100), left = need[r] - fin.length;
      g = (gok.length >= left ? gok : g).sort((x,y) => ev(y) - ev(x));
      if (g.length < left) { ok = false; break; }
      const t = fin.concat(g.slice(0, left)); pick.push(...t); tot += t.reduce((q,s) => q + ev(s), 0);
    }
    if (ok && (!best || tot > best.tot)) best = {mod, tot, pick};
  }
  return best;
}
let warnMsg = "";
function bestLineup(){
  warnMsg = "";
  if (modSel !== "auto") { const b = calc([modSel], true); if (b) return b; warnMsg = "Il modulo " + modSel + " non \u00e8 compatibile con le tue scelte fisse: uso il migliore possibile."; }
  const b2 = calc(D.modules, true); if (b2) return b2;
  warnMsg = "Le scelte fisse (\ud83d\udd12) non entrano in nessun modulo: le ignoro. Togline qualcuna.";
  return calc(D.modules, false);
}
function mk(tag, cls, txt){ const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }
function line(s, where){
  const l = mk("div","ln tap"); l.appendChild(mk("b",null,s.Ruolo)); l.appendChild(mk("span","nm",(s.force ? "\ud83d\udd12 " : "") + s.Giocatore + (s.Trend ? " " + s.Trend : "")));
  l.appendChild(mk("span","tag", s.Avversario + " \u00b7 voto " + s.Voto.toFixed(1) + " \u00b7 gioca " + s.p + "%" + (s.KO && Date.parse(s.KO) <= Date.now() ? " \u00b7 partita iniziata" : "") + (s.force === -1 ? " \u00b7 tua scelta: panchina" : s.force === 1 ? " \u00b7 tua scelta: titolare" : "")));
  l.addEventListener("click", () => {
    if (s.force) s.force = 0; else s.force = (where === "xi") ? -1 : 1;              // titolare -> panchina, panchinaro -> titolare, tocca ancora = automatico
    save(); update(); });
  return l;
}
const byRole = (b, r) => b.pick.filter(s => s.Ruolo === r).sort((x,y) => ev(y)-ev(x));
function update(){
  const b = bestLineup(); cur = b;
  const L = document.getElementById("lineup"), B = document.getElementById("bench"), M = document.getElementById("mini"), W = document.getElementById("warn");
  L.textContent = ""; B.textContent = ""; M.textContent = ""; inSet = new Set();
  W.textContent = warnMsg; W.style.display = warnMsg ? "block" : "none";
  if (!b) { L.textContent = "Nessun modulo valido con i giocatori disponibili."; document.getElementById("lt").textContent = "Formazione non disponibile"; }
  else {
    document.getElementById("lt").textContent = b.mod + " \u00b7 atteso " + b.tot.toFixed(1);
    ["P","D","C","A"].forEach(r => { const g = byRole(b, r), d = mk("div"); d.appendChild(mk("i",null,r));
      d.appendChild(document.createTextNode(g.map(s => s.Giocatore + (s.force === 1 ? "*" : "")).join(" \u00b7 "))); M.appendChild(d);
      g.forEach(s => { inSet.add(s.Giocatore); L.appendChild(line(s, "xi")); }); });
  }
  S.filter(s => s.disp && !inSet.has(s.Giocatore)).sort((x,y) => ev(y)-ev(x)).forEach(s => B.appendChild(line(s, "bn")));
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
  const orig = btn.textContent, ok = () => { btn.textContent = "Copiato \u2713"; setTimeout(() => btn.textContent = orig, 1500); };
  const fb = () => { try { const ta = mk("textarea"); ta.value = t; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); ok(); } catch (e) { btn.textContent = "Non riuscito"; } };
  try { if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(ok, fb); else fb(); } catch (e) { fb(); }
}
function build(){
  const R = document.getElementById("roster"); R.textContent = "";
  D.order.forEach(i => { const s = S[i], r = mk("div","row"); r.dataset.i = i;
    const t = mk("div","top"), left = mk("div"); left.appendChild(mk("span","r",s.Ruolo)); left.appendChild(mk("span","nm",s.Giocatore + (s.Trend ? " " + s.Trend : "")));
    const bd = mk("div","badge",s.Voto.toFixed(1)); bd.style.background = col(s.Voto); t.appendChild(left); t.appendChild(bd); r.appendChild(t);
    r.appendChild(mk("div","info", s.Avversario + " (" + s.Data + ") \u00b7 gol " + s.P_gol + "% \u00b7 assist " + s.P_ass + "%" +
      (s["CS%"] != null ? " \u00b7 clean sheet " + s["CS%"] + "%" : "") + (s.TrendTxt ? " \u00b7 " + s.TrendTxt : "")));
    r.appendChild(mk("div","info2", "fantavoto " + s.Fantavoto.toFixed(2) + " \u00b7 squadra " + s.GolSq.toFixed(1) + " gol attesi, avversario " + s.GolOpp.toFixed(1) +
      " (" + s.Fonte + ") \u00b7 risultato: V " + s.P_V + "% N " + s.P_N + "% P " + s.P_S + "%" + (s.Nota ? " \u00b7 " + s.Nota : "")));
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
  C.appendChild(mk("div", "info", (c.escluse && c.escluse.length ? "Giornate " + c.escluse.join(", ") + " escluse dal giudizio (rodaggio: il modello aveva pochi dati della stagione). " : "") + (c.n < 60 ? "Campione ancora piccolo: prendi questi numeri come indicativi." : ""))); }
if (D.fc) { const C = document.getElementById("cal"), f = D.fc; C.style.display = "block";
  C.appendChild(mk("b", null, "Fantacalcio.it: quanto ci si pu\u00f2 fidare"));
  C.appendChild(mk("div", "info", "Su " + f.n + " giocatori in " + f.matches + " partite" + (f.lead_h != null ? " (testo copiato in media " + f.lead_h.toFixed(1) + " ore prima)" : "") +
    ": errore delle percentuali " + f.brier_fc.toFixed(3) + ", contro " + f.brier_const.toFixed(3) + " di chi prevedesse per tutti la stessa probabilit\u00e0."));
  f.buckets.filter(b => b.n >= 5).forEach(b => C.appendChild(mk("div", "info", "\u2022 dato " + b.lo + (b.hi !== b.lo ? "-" + b.hi : "") + "%: ha giocato il " + Math.round(b.play * 100) + "% (" + b.n + " casi)")));
  if (f.roster) C.appendChild(mk("div", "info", "Sui tuoi giocatori (" + f.roster.n + "): Fantacalcio.it " + f.roster.brier_fc.toFixed(3) + " contro il mio modello " + f.roster.brier_model.toFixed(3) + " (pi\u00f9 basso = meglio)."));
  if (f.out_n) C.appendChild(mk("div", "info", "Tra gli infortunati/squalificati indicati, ne hanno giocato " + f.out_wrong + " su " + f.out_n + "."));
}
function fmtDur(ms){ const m = Math.max(0, Math.floor(ms / 60000)), h = Math.floor(m / 60), g = Math.floor(h / 24);
  return g > 0 ? g + " g " + (h % 24) + " h" : h > 0 ? h + " h " + (m % 60) + " min" : m + " min"; }
function tick(){
  const el = document.getElementById("dl"), ks = (D.kicks || []).map(k => ({t: Date.parse(k.t), m: k.m})).sort((a, b) => a.t - b.t), now = Date.now();
  if (!ks.length) { el.textContent = ""; return; }
  const started = ks.filter(k => k.t <= now).length, next = ks.find(k => k.t > now);
  const fmt = t => new Date(t).toLocaleString("it-IT", {weekday: "long", hour: "2-digit", minute: "2-digit"});
  el.style.color = "";
  if (!started) { const left = ks[0].t - now; el.textContent = "\u23f1 Scadenza (primo calcio d'inizio): " + fmt(ks[0].t) + ", " + ks[0].m + " \u00b7 mancano " + fmtDur(left) + " (dipende dalla tua lega)";
    if (left < 2 * 3600000) el.style.color = "#d97706"; }
  else if (next) { el.textContent = "\u23f1 " + started + " partite su " + ks.length + " gi\u00e0 iniziate \u00b7 prossima: " + next.m + " tra " + fmtDur(next.t - now); el.style.color = "#d97706"; }
  else el.textContent = "\u23f1 Tutte le partite della giornata sono iniziate.";
}
if (D.pagella && D.pagella.rounds && D.pagella.rounds.length) { const P = document.getElementById("pag"), g = D.pagella; P.style.display = "block";
  P.appendChild(mk("b", null, "Pagella della formazione"));
  g.rounds.forEach(r => { const f = k => r[k] ? r[k].punti.toFixed(1) + " (" + r[k].modulo + ")" : "\u2013";
    P.appendChild(mk("div", "info", "Giornata " + r.giornata + ": modello " + f("modello") + " \u00b7 tu " + f("tua") + " \u00b7 massimo possibile " + f("massimo"))); });
  const m = g.media || {}, v = (o, k) => o && o[k] != null ? o[k].toFixed(1) : "\u2013";
  P.appendChild(mk("div", "info", "Media su " + (m.n || g.rounds.length) + " giornate: modello " + v(m, "modello") + " \u00b7 massimo " + v(m, "massimo") + (m.modello != null && m.massimo ? ". Il modello coglie il " + Math.round(100 * m.modello / m.massimo) + "% del massimo." : "")));
  if (g.comune) P.appendChild(mk("div", "info", "Solo le " + g.comune.n + " giornate con la tua formazione: modello " + v(g.comune, "modello") + " \u00b7 tu " + v(g.comune, "tua") + " \u00b7 massimo " + v(g.comune, "massimo") + "."));
  P.appendChild(mk("div", "info", "Somma dei fantavoti dei titolari con le sostituzioni, senza modificatore difesa. \"Modello\" = sola previsione automatica, senza le tue correzioni di titolarit\u00e0."));
}
document.getElementById("beta").textContent = "Effetto risultato: la vittoria attesa della squadra sposta il fantavoto di circa " + (D.beta != null ? D.beta.toFixed(2) : "0.15") + " per ogni punto di (probabilit\u00e0 di vittoria - probabilit\u00e0 di sconfitta).";
tick(); setInterval(() => { tick(); update(); }, 60000);
document.getElementById("meta").textContent = "Giornata " + D.gno + " \u00b7 aggiornato " + D.agg + " \u00b7 quote bookmaker: " + D.odds_msg + " \u00b7 infortuni: " + D.inj_msg;
document.getElementById("reset").addEventListener("click", () => { S.forEach(s => { s.p = s.p0; s.disp = !!s.Disp; s.force = 0; }); modSel = "auto"; sel.value = "auto";
  try { localStorage.removeItem(KEY); } catch (e) {} build(); update(); });

const norm = t => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\u00f8/gi, "o").replace(/\u00e6/gi, "ae").replace(/\u0142/gi, "l").replace(/\u0111/gi, "d").replace(/\u00df/g, "ss").toLowerCase();
const toks = t => norm(t).replace(/[^a-z ]/g, " ").split(/\s+/).filter(Boolean);
const mapPct = p => { const b = (D.fcmap || []).find(x => p >= x.lo && p <= x.hi); return b ? Math.round(b.p * 100) : p; };
const TW = s => norm(s).replace(/[^a-z ]/g, " ").replace(/\s+/g, " ").trim();
const nameMatches = (s, txt) => { const lt = new Set(toks(txt)), l = toks(s.Giocatore).filter(t => t.length > 1); return l.length > 0 && l.every(t => lt.has(t)); };
function teamIn(header){
  const h = " " + TW(header) + " "; let best = null, bi = -1;
  [...new Set(S.map(s => s.Squadra))].forEach(t => { const i = h.lastIndexOf(" " + TW(t) + " "); if (i > bi) { bi = i; best = t; } });
  return best;
}
const SEC = "Ballottagg|Squalificat|Indisponibil|Infortunat|Diffidat|Panchina|Arbitro|Allenator|All\\.|Probabili formazioni";
function parseBlocks(text){
  const re = /\((\d(?:-\d){1,3})\)\s*:/g, marks = []; let m;
  while ((m = re.exec(text))) marks.push({start: m.index, end: re.lastIndex});
  const acts = new Map(), teams = [];
  const put = (s, p, out) => { const a = acts.get(s.Giocatore) || {s}; if (out) a.out = true; else if (p != null) a.p = p; acts.set(s.Giocatore, a); };
  marks.forEach((mk, i) => {
    const team = teamIn(text.slice(Math.max(0, mk.start - 45), mk.start)); if (!team) return;
    if (!teams.includes(team)) teams.push(team);
    let B = text.slice(mk.end, i + 1 < marks.length ? marks[i+1].start : text.length);
    const cut = B.search(/probabili formazioni/i); if (cut >= 0) B = B.slice(0, cut);
    const xiEnd = B.search(new RegExp("\\n|" + SEC, "i")), XI = xiEnd < 0 ? B : B.slice(0, xiEnd), rest = xiEnd < 0 ? "" : B.slice(xiEnd);
    const groups = XI.split(/[;,]/).map(x => x.split("/").map(y => y.trim()).filter(Boolean)).filter(g => g.length);
    const R = S.filter(s => s.Squadra === team);
    if (groups.length >= 8) R.forEach(s => { const g = groups.find(g => g.some(a => nameMatches(s, a))); put(s, g ? (g.length > 1 ? 50 : 90) : 15, false); });
    const bm = rest.match(new RegExp("Ballottagg\\w*\\s*:?\\s*([\\s\\S]*?)(?=Squalificat|Indisponibil|Infortunat|Diffidat|Panchina|Arbitro|Probabili|$)", "i"));
    if (bm){
      const T = bm[1], items = [];
      const rb = /([^,;%\d]+?)\s+(\d{1,3})\s*%\s*[-\u2013]\s*([^,;%\d]+?)\s+(\d{1,3})\s*%/g; let x;
      while ((x = rb.exec(T))) items.push([x[1], +x[2], x[3], +x[4]]);
      if (!items.length){
        const ra = /(\d{1,3})\s*%\s*[-\u2013]\s*(\d{1,3})\s*%/g; let last = 0;
        while ((x = ra.exec(T))) { const seg = T.slice(last, x.index).replace(/^[\s,;:.\u00b7]+/, ""); last = ra.lastIndex;
          const pr = seg.split(/\s[\u2013\u2014-]\s/); if (pr.length >= 2) items.push([pr[0], +x[1], pr[1], +x[2]]); }
      }
      items.forEach(it => R.forEach(s => { if (nameMatches(s, it[0])) put(s, it[1], false); if (nameMatches(s, it[2])) put(s, it[3], false); }));
    }
    const ro = new RegExp("(Squalificat\\w*|Indisponibil\\w*|Infortunat\\w*|Assent\\w*)\\s*:\\s*([^|\\n]*?)(?=Squalificat|Indisponibil|Infortunat|Diffidat|Ballottagg|Panchina|Probabili|\\||\\n|$)", "gi");
    let o; while ((o = ro.exec(rest))) o[2].split(/[,;]|\se\s|\//).forEach(n => { if (n.trim()) R.forEach(s => { if (nameMatches(s, n)) put(s, null, true); }); });
  });
  return {acts, teams};
}
function applyLines(text, mode){
  const done = [], seen = new Set(); let skipped = 0;
  text.split(/\n/).map(x => x.trim()).filter(Boolean).forEach(line => {
    const lt = new Set(toks(line)), has = pre => [...lt].some(t => t.startsWith(pre));
    const found = S.filter(s => nameMatches(s, line));
    if (!found.length) { skipped++; return; }
    const m = line.match(/(\d{1,3})\s*%/);
    found.forEach(s => {
      let what = null;
      if (m) { s.p = Math.max(0, Math.min(100, Math.round(+m[1] / 5) * 5)); s.disp = true; what = s.p + "%"; }
      else if (mode === "out" || has("infortun") || has("squalific") || has("indisponibil") || has("assent") || has("lesion") || lt.has("out") || lt.has("fuori") || lt.has("stop")) { s.disp = false; what = "fuori"; }
      else if (mode === "start" || has("titolar")) { s.p = 90; s.disp = true; what = "90%"; }
      else if (has("dubbio") || has("ballottagg") || has("incerto")) { s.p = 50; s.disp = true; what = "50%"; }
      if (what && !seen.has(s.Giocatore)) { seen.add(s.Giocatore); done.push(s.Giocatore + " " + what); }
    });
  });
  return {done, skipped};
}

const isMod = l => /^\d(?:-\d){1,3}$/.test(l || "");
const isPct = l => /^\d{1,3}\s*%$/.test(l || "");
function parseFantacalcio(text){
  const L = text.split(/\r?\n/).map(x => x.trim()), n = L.length, acts = new Map(), read = [];
  let pending = [], matches = 0, k = 0;
  const put = (s, p, out) => { const a = acts.get(s.Giocatore) || {s}; if (out) a.out = true; else a.p = p; acts.set(s.Giocatore, a); };
  while (k < n){
    if (k + 1 < n && isMod(L[k+1]) && L[k] && !isPct(L[k]) && !isMod(L[k])) {          // "Bologna" + "3-4-2-1"
      const name = L[k], entries = []; let j = k + 2, bench = false;
      while (j < n){
        const cur = L[j];
        if (/^ultimo aggiornamento/i.test(cur)) break;
        if (j + 1 < n && isMod(L[j+1]) && !isPct(cur)) break;                          // inizia la squadra dopo
        if (/^panchina$/i.test(cur)) { bench = true; j++; continue; }
        if (j + 1 < n && isPct(L[j+1]) && cur && !isPct(cur)) { entries.push({name: cur, pct: parseInt(L[j+1]), bench}); j += 2; continue; }
        j++;
      }
      const rt = teamIn(name); read.push(rt || name); pending.push(rt);
      if (rt) S.filter(s => s.Squadra === rt).forEach(s => { const e = entries.find(e => nameMatches(s, e.name)); put(s, e ? mapPct(e.pct) : 5, false); });
      k = j; continue;
    }
    if (/^dettaglio calciatori/i.test(L[k])) {
      const keys = pending.slice(-2).filter(Boolean), cand = S.filter(s => keys.includes(s.Squadra)); let j = k + 1, cur = null;
      while (j < n && !/^(stemma|campioncino)\b/i.test(L[j]) && !(j + 1 < n && isMod(L[j+1]))) {
        const t = L[j], hm = t.match(/^(ballottaggi|squalificati|diffidati|infortunati|in dubbio)$/i);
        if (hm) cur = hm[1].toLowerCase();
        else if (cur && t && !/^nessun/i.test(t) && !/[,\d%]/.test(t) && t.length <= 32 && /^[A-Z\u00c0-\u00dd]/.test(t)) {
          if (cur === "squalificati" || cur === "infortunati") cand.forEach(s => { if (nameMatches(s, t)) put(s, null, true); });
          else if (cur === "in dubbio") cand.forEach(s => { if (nameMatches(s, t)) put(s, 50, false); });
        }
        j++;
      }
      matches++; pending = []; k = j; continue;
    }
    k++;
  }
  return {acts, teams: [...new Set(read.filter(t => S.some(s => s.Squadra === t)))], nread: read.length, matches};
}

function applyPaste(){
  const text = document.getElementById("paste").value, mode = document.getElementById("pmode").value, out = document.getElementById("pout");
  const fc = parseFantacalcio(text); let res = fc, msg;
  if (!fc.teams.length) res = parseBlocks(text);
  if (res.teams.length){
    const done = [];
    res.acts.forEach(a => { const s = a.s;
      if (a.out) { s.disp = false; done.push(s.Giocatore + " fuori"); }
      else if (a.p != null) { const was = !s.disp; s.p = Math.max(0, Math.min(100, Math.round(a.p / 5) * 5)); s.disp = true; done.push(s.Giocatore + " " + s.p + "%" + (was ? " (era fuori: rimesso in gioco)" : "")); } });
    const missing = [...new Set(S.map(s => s.Squadra))].filter(t => !res.teams.includes(t));
    msg = (fc.teams.length ? "Fantacalcio.it: lette " + fc.nread + " squadre (" + fc.matches + " partite). " : "") + "Squadre della tua rosa trovate: " + res.teams.join(", ") + ". " + (done.length ? "Applicato: " + done.join(", ") + ". " : "Nessuna modifica ai tuoi giocatori. ") +
      (missing.length ? "Non trovate nel testo: " + missing.join(", ") + ". " : "") + (fc.teams.length && (D.fcmap || []).length ? "Percentuali corrette con lo storico di Fantacalcio.it." : "");
  } else {
    const r = applyLines(text, mode); msg = r.done.length ? "Applicato: " + r.done.join(", ") + ". " + (r.skipped ? r.skipped + " righe senza giocatori della tua rosa." : "") :
      "Non ho trovato blocchi tipo 'MILAN (3-4-2-1): Maignan; ...' n\u00e9 righe con nome e %/parola chiave. Se hai incollato una lista di soli nomi, scegli dal menu 'indisponibili' o 'titolari'.";
  }
  save(); build(); update(); out.textContent = msg;
}
document.getElementById("apply").addEventListener("click", applyPaste);
document.getElementById("gh").addEventListener("click", e => {
  const t = document.getElementById("paste").value, o = document.getElementById("pout");
  if (!t.trim()) { o.textContent = "Incolla prima il testo di Fantacalcio.it, poi tocca Salva su GitHub."; return; }
  copyText(t, e.target);
  const parts = location.pathname.split("/").filter(Boolean), host = location.hostname, stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  if (host.endsWith("github.io") && parts.length) {
    window.open("https://github.com/" + host.split(".")[0] + "/" + parts[0] + "/new/main?filename=formazioni/" + stamp + ".txt", "_blank");
    o.textContent = "Testo copiato. Nella pagina GitHub incollalo nel riquadro e premi Commit changes.";
  } else o.textContent = "Testo copiato. Nel tuo repository crea un file dentro la cartella formazioni/ e incollalo.";
});

document.getElementById("pbtn").addEventListener("click", () => { const d = document.getElementById("pdet"); d.open = true;
  if (d.scrollIntoView) d.scrollIntoView({block: "center"}); document.getElementById("paste").focus(); });
document.getElementById("ver").textContent = "versione pagina: " + D.ver;
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
        "inj_msg": info.get("inj_msg", "non attivi"),
        "ver": CFG["version"],
        "partite": info.get("partite", len(fixtures)),
        "agg": f"{now:%d/%m/%Y %H:%M}",
        "calib": rep if rep and rep.get("n", 0) > 0 else None,
        "fc": (calib or {}).get("fc"),
        "pagella": (calib or {}).get("pagella"),
        "kicks": info.get("kicks", []),
        "beta": (calib or {}).get("result_beta", CFG["result_beta_prior"]),
        "fcmap": (calib or {}).get("fc_map") or [],
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
    ap.add_argument("--tuning", default="tuning.json", help="parametri misurati dalla taratura")
    ap.add_argument("--mercato", action="store_true", help="analisi di mercato (occasioni vs gol reali) ed esci")
    ap.add_argument("--taratura", action="store_true", help="misura sui dati di tutta la Serie A quanto pesare il passato ed esci")
    ap.add_argument("--statistiche", default="statistiche", help="cartella col file statistiche di Fantacalcio.it (xlsx)")
    ap.add_argument("--formazioni", default="formazioni", help="cartella con i testi copiati da Fantacalcio.it")
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

    if a.mercato:
        n_ = datetime.now(ZoneInfo(CFG["tz"]))
        season = a.season or (n_.year if n_.month >= 7 else n_.year - 1)
        roster = pd.read_csv(a.rosa)
        prov = UnderstatProvider(hours=CFG["cache_hours"], refresh=a.refresh)
        pcur = num(pd.DataFrame(prov.players(season)), ["time", "xG", "xA", "xGBuildup", "yellow_cards", "red_cards", "goals", "assists", "npg", "npxG"])
        pold = num(pd.DataFrame(prov.players(season - 1)), ["time", "xG", "xA", "xGBuildup", "yellow_cards", "red_cards"])
        priors = role_priors(pd.concat([pcur, pold], ignore_index=True))
        hist = build_team_history(prov, season)
        S, lg, hf, af = team_strengths(hist)
        titles = list(hist.keys())
        outlook = team_outlook(prov.matches(season), S, lg, hf, af, n_.replace(tzinfo=None))
        rounds_played = max(season_progress(prov.matches(season)), 1)
        min_minutes = min(CFG["mercato_min_minutes"], round(rounds_played * 90 * CFG["mercato_min_frac"]))
        print(f"Mercato: {rounds_played} giornate giocate finora, minuti minimi richiesti {min_minutes}")
        fc_raw, fc_name = latest_fc_stats(a.statistiche)
        fc_stats = match_fc_stats(fc_raw, pcur, titles) if fc_raw is not None else None
        if fc_stats is not None:
            print(f"Statistiche Fantacalcio.it: {fc_name}, {len(fc_stats)}/{len(fc_raw)} giocatori riconosciuti")
        elif fc_name:
            print(f"Statistiche Fantacalcio.it: {fc_name} trovato ma non leggibile, uso solo Understat")
        else:
            print("Statistiche Fantacalcio.it: nessun file in statistiche/, uso solo Understat (rigoristi non identificati)")
        report, err = market_report(pcur, priors, roster, titles, S, outlook, fc_stats, min_minutes)
        if err:
            print(err)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "mercato.html").write_text(to_html_mercato(report, err, season, n_.replace(tzinfo=None)), encoding="utf-8")
        print(f"Analisi di mercato scritta in {out}/mercato.html")
        return
    if a.taratura:
        n_ = datetime.now(ZoneInfo(CFG["tz"]))
        season = a.season or (n_.year if n_.month >= 7 else n_.year - 1)
        run_tuning(UnderstatProvider(hours=24 * 30, refresh=a.refresh), season, a.tuning)
        print(f"Parametri salvati in {a.tuning}")
        return
    if a.calibra:
        prov, season, roster = None, None, None
        try:
            n_ = datetime.now(ZoneInfo(CFG["tz"]))
            season = a.season or (n_.year if n_.month >= 7 else n_.year - 1)
            roster = pd.read_csv(a.rosa)
            prov = UnderstatProvider(hours=CFG["cache_hours"], refresh=a.refresh)
        except Exception as e:  # noqa: BLE001
            prov = None
            print(f"  ! Understat non raggiungibile ({type(e).__name__}): calibrazione solo sui voti")
        calibra(a.storico, a.voti, a.calib, a.formazioni, prov, season, roster)
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

    tune = None
    if Path(a.tuning).exists():
        try:
            tune = json.loads(Path(a.tuning).read_text(encoding="utf-8")).get("roles")
        except Exception:  # noqa: BLE001
            tune = None
    if a.backtest:
        rounds = [int(x) for x in a.giornata.split(",") if x.strip()]
        if not rounds:
            sys.exit("Con --backtest indica le giornate: --giornata 2,3")
        for g in rounds:
            df, fixtures, problems, info = run(prov, roster, season, now, round_no=g, backtest=True, calib=None, tune=tune)
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

    inj_key = os.environ.get("API_FOOTBALL_KEY", "").strip() or None
    inj_site = os.environ.get("API_FOOTBALL_SITE", "api-football.com").strip().lower() or "api-football.com"
    if inj_key:
        print(f"API_FOOTBALL_KEY trovata ({len(inj_key)} caratteri), servizio: {inj_site}")
    df, fixtures, problems, info = run(prov, roster, season, now, check_only=a.check, calib=calib,
                                       odds_events=odds_events, odds_note=odds_note, inj_key=inj_key, inj_site=inj_site, tune=tune)
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
