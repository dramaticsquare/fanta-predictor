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
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

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
    # voto 1-10:  5 + slope * (fantavoto - fantavoto di un giocatore medio del ruolo in partita neutra)
    "rating_slope": 2.0,
    "cache_hours": 12,
}

MODULES = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]

# nome squadra come lo scrivi in rosa.csv -> nome Understat
TEAM_ALIAS = {
    "milan": "AC Milan",
    "parma": "Parma Calcio 1913",
    "hellas verona": "Verona",
    "verona": "Verona",
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
def build_team_history(prov, season):
    """dict titolo -> DataFrame (date, h_a, xG, xGA) ordinato per data, ultime 2 stagioni."""
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


def team_ctx(team, opp, is_home, S, lg, hf, af):
    a_own, d_own = (S.get(team) or {"A": 1, "D": 1})["A"], (S.get(team) or {"A": 1, "D": 1})["D"]
    a_opp, d_opp = (S.get(opp) or {"A": 1, "D": 1})["A"], (S.get(opp) or {"A": 1, "D": 1})["D"]
    venue_att = hf if is_home else af
    venue_def = af if is_home else hf            # chi subisce in casa incassa "xG trasferta"
    return {
        "att_factor": d_opp * venue_att,                  # moltiplica le stat offensive del giocatore
        "lam_for": lg * a_own * d_opp * venue_att,        # xG attesi della sua squadra
        "lam_conc": lg * a_opp * d_own * venue_def,       # xG attesi subiti
    }


def next_matchday(matches, now):
    up = [m for m in matches if not m.get("isResult")
          and datetime.fromisoformat(m["datetime"]) >= now - timedelta(hours=3)]
    up.sort(key=lambda m: m["datetime"])
    seen, chosen = set(), []
    for m in up:
        h, a = m["h"]["title"], m["a"]["title"]
        if h in seen or a in seen:
            break
        chosen.append(m)
        seen |= {h, a}
    return chosen


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
    """Fantavoto di un giocatore MEDIO del ruolo in una partita neutra (definisce il '5' del voto 1-10)."""
    fv = CFG["base_vote"] + CFG["goal"] * pr["xG"] + CFG["assist"] * pr["xA"]
    fv += CFG["yellow"] * pr["yellow_cards"] + CFG["red"] * pr["red_cards"]
    fv += CFG["clean_sheet"][role] * float(np.exp(-lg))
    if role == "P":
        fv += CFG["conceded_gk"] * lg
    return fv


def project(role, prof, ctx, ref):
    m = prof["exp_min"] / 90
    e_g = prof["xg90"] * ctx["att_factor"] * m
    e_a = prof["xa90"] * ctx["att_factor"] * m
    p_cs = float(np.exp(-ctx["lam_conc"]))
    fv = CFG["base_vote"]
    fv += CFG["goal"] * e_g + CFG["assist"] * e_a
    fv += CFG["yellow"] * prof["yc90"] * m + CFG["red"] * prof["rc90"] * m
    fv += CFG["clean_sheet"][role] * p_cs
    if role == "P":
        fv += CFG["conceded_gk"] * ctx["lam_conc"]
    if role in ("D", "C"):
        fv += CFG["involvement_weight"] * prof["z_bu"]
    rating = float(np.clip(5 + CFG["rating_slope"] * (fv - ref), 1, 10))
    p_eff = min(1.0, prof["p_start"] + CFG["sub_weight"] * prof["p_sub"])
    ev = p_eff * fv + (1 - p_eff) * ref            # se non gioca, entra in media un sostituto "medio"
    return {"e_g": e_g, "e_a": e_a, "p_cs": p_cs, "fv": fv, "rating": rating, "p_eff": p_eff, "ev": ev}


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
def run(prov, roster, season, now, check_only=False):
    print(f"Stagione Understat: {season}/{str(season + 1)[-2:]}   -   {now:%d/%m/%Y %H:%M}")
    print("Scarico dati squadre e giocatori...")
    hist = build_team_history(prov, season)
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

    fixtures = next_matchday(prov.matches(season), now)
    if not fixtures:
        sys.exit("Nessuna partita in programma trovata su Understat (stagione finita o non ancora pubblicata).")
    fx = {}
    for m in fixtures:
        h, a = m["h"]["title"], m["a"]["title"]
        fx[h] = (a, True, m["datetime"])
        fx[a] = (h, False, m["datetime"])
    print(f"Prossima giornata: {len(fixtures)} partite, dal {fixtures[0]['datetime'][:16]}")

    rows, problems = [], []
    for r in roster.itertuples():
        team = resolve_team(r.squadra, titles)
        role = str(r.ruolo).strip().upper()[0]
        override = getattr(r, "understat", None)
        override = override if isinstance(override, str) and override.strip() else None
        avail = int(getattr(r, "disponibile", 1)) == 1
        if team is None:
            problems.append(f"{r.nome}: squadra '{r.squadra}' non riconosciuta (titoli: {', '.join(sorted(titles))})")
            continue
        hit = find_player(r.nome, override, team, pool)
        if hit is None:
            problems.append(f"{r.nome} ({r.squadra}): non trovato su Understat. "
                            f"Aggiungi il nome esatto nella colonna 'understat' di rosa.csv")
            continue
        print(f"  {r.nome:<14} -> {hit.player_name:<28} [{hit.team_title}] (id {hit.id})")
        if check_only:
            continue
        if team not in fx:
            problems.append(f"{r.nome}: {team} non gioca in questa giornata")
            continue
        opp, home, when = fx[team]
        pm = prov.player_matches(hit.id)
        prof = player_profile(pm, team, hist, role, priors, cards_all.loc[hit.id].to_dict()
                              if hit.id in cards_all.index else None)
        ctx = team_ctx(team, opp, home, S, lg, hf, af)
        pj = project(role, prof, ctx, refs[role])
        rows.append({
            "Giocatore": r.nome, "Squadra": r.squadra, "Ruolo": role, "Disp": avail,
            "Avversario": f"{'vs' if home else '@'} {opp}", "Data": f"{when[8:10]}/{when[5:7]}",
            "xG": round(pj["e_g"], 2), "xA": round(pj["e_a"], 2),
            "CS%": round(100 * pj["p_cs"]) if role in "PD" else None,
            "Titolare%": round(100 * pj["p_eff"]),
            "Fantavoto": round(pj["fv"], 2), "Voto": round(pj["rating"], 1), "EV": round(pj["ev"], 2),
            "Nota": ("nessuna presenza recente" if pj["p_eff"] == 0 else
                     "pochi dati" if prof["n_matches"] < 5 else ""),
        })
    if problems:
        print("\nATTENZIONE:")
        for p in problems:
            print("  -", p)
    if check_only:
        return None, None
    return pd.DataFrame(rows), fixtures


# ============================================================== OUTPUT ======
def color(v):
    v = max(1, min(10, v))
    hue = 120 * (v - 1) / 9          # rosso -> verde
    return f"hsl({hue:.0f},60%,42%)"


def to_html(df, lineup, fixtures, now):
    def table(d, cols):
        h = "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        for _, r in d.iterrows():
            h += "<tr>"
            for c in cols:
                v = "" if pd.isna(r[c]) else r[c]
                style = f' style="background:{color(r[c])};color:#fff;font-weight:600"' if c == "Voto" else ""
                h += f"<td{style}>{html.escape(str(v))}</td>"
            h += "</tr>"
        return h + "</table>"

    cols = ["Giocatore", "Ruolo", "Avversario", "Data", "Voto", "Fantavoto", "xG", "xA", "CS%", "Titolare%", "Nota"]
    body = f"<h1>Fanta Predictor</h1><p class=s>Aggiornato {now:%d/%m/%Y %H:%M} - " \
           f"giornata dal {fixtures[0]['datetime'][:10]}</p>"
    if lineup:
        mod, tot, xi, bench = lineup
        body += f"<h2>Formazione consigliata: {mod}</h2>"
        order = {"P": 0, "D": 1, "C": 2, "A": 3}
        xi = xi.assign(_o=xi.Ruolo.map(order)).sort_values(["_o", "EV"], ascending=[True, False])
        body += table(xi, cols)
        body += "<h2>Panchina (in ordine)</h2>" + table(bench, cols)
    body += "<h2>Tutta la rosa</h2>" + table(df.sort_values("EV", ascending=False), cols + ["Disp"])
    body += ("<p class=s>Voto 1-10 = 5 + 2 x (fantavoto atteso - 6), calcolato SE il giocatore scende in campo. "
             "Titolare% = probabilita' di partire titolare/entrare. Ordinamento formazione: valore atteso "
             "che include la probabilita' di giocare. Modello statistico, non una garanzia.</p>")
    css = ("body{font-family:system-ui,sans-serif;margin:12px;background:#fff;color:#111}"
           "table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:8px;display:block;overflow-x:auto}"
           "th,td{padding:5px 7px;border-bottom:1px solid #8884;text-align:left;white-space:nowrap}"
           ".s{color:#888;font-size:12px}"
           "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}}")
    return (f"<!doctype html><html lang=it><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Fanta Predictor</title><style>{css}</style></head><body>{body}</body></html>")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Previsione voti fantacalcio Serie A (xG/xA + avversario)")
    ap.add_argument("--rosa", default="rosa.csv")
    ap.add_argument("--out", default="docs", help="cartella di output (index.html + previsioni.csv)")
    ap.add_argument("--season", type=int, help="anno di inizio stagione (default: automatico)")
    ap.add_argument("--moduli", default=",".join(MODULES), help="moduli ammessi dalla tua lega, es. 4-3-3,3-5-2")
    ap.add_argument("--check", action="store_true", help="solo verifica riconoscimento nomi")
    ap.add_argument("--refresh", action="store_true", help="ignora la cache")
    a = ap.parse_args(argv)

    now = datetime.now()
    season = a.season or (now.year if now.month >= 7 else now.year - 1)
    roster = pd.read_csv(a.rosa)
    prov = UnderstatProvider(hours=CFG["cache_hours"], refresh=a.refresh)
    df, fixtures = run(prov, roster, season, now, check_only=a.check)
    if a.check or df is None:
        return
    lineup = best_lineup(df, [m.strip() for m in a.moduli.split(",")])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values("EV", ascending=False).to_csv(out / "previsioni.csv", index=False)
    (out / "index.html").write_text(to_html(df, lineup, fixtures, now), encoding="utf-8")

    pd.set_option("display.width", 200)
    show = ["Giocatore", "Ruolo", "Avversario", "Voto", "Fantavoto", "xG", "xA", "CS%", "Titolare%", "Nota"]
    if lineup:
        mod, tot, xi, bench = lineup
        print(f"\n=== FORMAZIONE CONSIGLIATA {mod} ===")
        print(xi.sort_values("Ruolo", key=lambda s: s.map({"P": 0, "D": 1, "C": 2, "A": 3}))[show].to_string(index=False))
        print("\n=== PANCHINA ===")
        print(bench[show].to_string(index=False))
    print(f"\nFile scritti in {out}/ (index.html, previsioni.csv)")


if __name__ == "__main__":
    main()
