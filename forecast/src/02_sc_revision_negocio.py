#!/usr/bin/env python3
"""
02_sc_revision_negocio.py
=========================
Revision de supply chain / demand planning del dataset `reto1_dataset.xlsx`.

Prueba, con un backtesting de origen rodante semanal y horizonte de 7 dias (D-01):
  A. Estructura comercial: bloques de promo / exhibicion / promo competidor; quiebres vs promo.
  B. Tratamiento del quiebre de stock (D-03: demanda total).
  C. Tratamiento de la promo del competidor (D-02: no se conoce de antemano).
  D. Rezagos de venta validos con h = 7 (y(t-7) crudo, des-censurado, correccion de nivel).
  E. Efectos de negocio: post-promo, pre-promo, dia dentro de la promo, rendimientos
     decrecientes del descuento, promo x exhibicion, quincena, feriados.
  F. Metrica asimetrica (D-04): pinball / costo sub:sobre, cobertura, error semanal.
  G. Conjuntos de variables de negocio combinados (objetivo des-censurado + competidor en curso).

Evaluacion: dias SIN quiebre (alli la venta observada = demanda), pronostico ex ante con
quiebre = 0, distribucion = media de los ultimos 7 dias al origen y competidor segun escenario.

Uso:
    python3 -I forecast/src/02_sc_revision_negocio.py [--data ruta.xlsx] [--out carpeta]

Salidas: stdout + CSV en forecast/reportes/02_salidas/.
Determinista (no usa procesos aleatorios).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "forecast" / "data" / "reto1_dataset.xlsx"
DEFAULT_OUT = REPO / "forecast" / "reportes" / "02_salidas"
Y = "unidades_vendidas"
DOW = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
H = 7                 # horizonte (D-01)
FIRST_ORIGIN = 62     # primer origen: se entrena con 63 dias (9 semanas)
CU, CO = 3.0, 1.0     # costo sub-pronostico : sobre-pronostico (propuesta D-04)


def titulo(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# --------------------------------------------------------------------------------------
# carga y variables derivadas (todas conocidas ex ante salvo las marcadas)
# --------------------------------------------------------------------------------------
def cargar(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Datos_completos")
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d.sort_values("fecha").reset_index(drop=True)
    d["y"] = d[Y].astype(float)
    d["ly"] = np.log(d["y"])
    d["q"] = d["quiebre_stock_pct"]
    d["promo_on"] = (d["promo_pct"] > 0).astype(int)
    # calendario de promo (plan comercial): dias desde fin de promo, dias hasta inicio
    n = len(d)
    post = np.full(n, 99)
    last_end = None
    for i in range(n):
        if d.promo_on[i]:
            last_end = i
        elif last_end is not None:
            post[i] = i - last_end
    pre = np.full(n, 99)
    next_start = None
    for i in range(n - 1, -1, -1):
        if d.promo_on[i]:
            next_start = i
        elif next_start is not None:
            pre[i] = next_start - i
    d["dias_post_promo"] = post
    d["dias_pre_promo"] = pre
    d["post_1_3"] = ((post >= 1) & (post <= 3)).astype(int)
    d["post_4_7"] = ((post >= 4) & (post <= 7)).astype(int)
    d["pre_1_3"] = ((pre >= 1) & (pre <= 3)).astype(int)
    # descuento de la promo anterior (intensidad del "pantry loading")
    prev_disc = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if d.promo_on[i]:
            cur = d.promo_pct[i]
        prev_disc[i] = cur if post[i] <= 7 else 0.0
    d["post_1_3_x_desc"] = d["post_1_3"] * prev_disc
    # dia dentro del bloque de promo
    g = (d["promo_pct"] != d["promo_pct"].shift()).cumsum()
    d["dia_en_promo"] = np.where(d.promo_on == 1, d.groupby(g).cumcount() + 1, 0)
    d["promo_dia1"] = (d["dia_en_promo"] == 1).astype(int)
    d["promo_x_exhib"] = d["promo_pct"] * d["exhibicion"]
    d["promo_sq"] = d["promo_pct"] ** 2
    d["quinc_x_nopromo"] = d["dia_quincena"] * (1 - d.promo_on)
    d["fin_mes"] = (d["fecha"].dt.day >= 29).astype(int) | (d["fecha"].dt.day == 1).astype(int)
    d["quinc_15"] = d["fecha"].dt.day.isin([14, 15, 16]).astype(int)
    for k in DOW[:-1]:            # domingo = referencia
        d[f"dow_{k}"] = (d["dia_semana"] == k).astype(int)
    # dias del bloque de competidor ya iniciado al origen (se usa en backtest)
    gc = (d["promo_competidor"] != d["promo_competidor"].shift()).cumsum()
    d["comp_bloque"] = np.where(d.promo_competidor == 1, gc, -1)
    return d


DOWC = [f"dow_{k}" for k in DOW[:-1]]
BASE = ["promo_pct", "exhibicion", "distribucion_numerica", "dia_quincena"] + DOWC


# --------------------------------------------------------------------------------------
# modelo log-lineal (OLS) con retransformacion por smearing
# --------------------------------------------------------------------------------------
class LogOLS:
    def __init__(self, cols, target="ly"):
        self.cols, self.target = cols, target

    def fit(self, d):
        X = sm.add_constant(d[self.cols].astype(float), has_constant="add")
        self.m = sm.OLS(d[self.target], X).fit()
        r = self.m.resid
        self.smear = float(np.mean(np.exp(r)))
        self.resid = r
        return self

    def predict_log(self, d):
        X = sm.add_constant(d[self.cols].astype(float), has_constant="add")
        return np.asarray(self.m.predict(X))

    def predict(self, d):
        return np.exp(self.predict_log(d)) * self.smear


def metricas(y, p, cu=CU, co=CO):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    under = np.clip(y - p, 0, None)
    over = np.clip(p - y, 0, None)
    tau = cu / (cu + co)
    return {
        "n": len(y),
        "WAPE_%": 100 * np.abs(e).sum() / y.sum(),
        "Bias_%": 100 * e.sum() / y.sum(),
        "Faltante_%": 100 * under.sum() / y.sum(),
        "Sobrante_%": 100 * over.sum() / y.sum(),
        f"Costo{int(cu)}:{int(co)}_%": 100 * (cu * under + co * over).sum() / y.sum(),
        f"Pinball{tau:.2f}_%": 100 * (tau * under + (1 - tau) * over).sum() / y.sum(),
        "Cobertura_%": 100 * np.mean(p >= y),
    }


# --------------------------------------------------------------------------------------
# backtest de origen rodante: origen o = ultimo dia con venta observada; pronostica o+1..o+7
# --------------------------------------------------------------------------------------
def backtest(d, make_model, prep_future, origins, train_filter=None, post=None):
    rows = []
    for o in origins:
        tr = d.iloc[: o + 1].copy()
        if train_filter is not None:
            tr = train_filter(tr)
        fut = d.iloc[o + 1: o + 1 + H].copy()
        if fut.empty:
            continue
        m = make_model().fit(tr)
        fx = prep_future(fut.copy(), d.iloc[: o + 1])
        p = m.predict(fx)
        if post is not None:
            p = post(p, m, tr, fx)
        for i, (idx, r) in enumerate(fut.iterrows()):
            rows.append({"origen": o, "h": i + 1, "idx": idx, "fecha": r.fecha, "y": r.y,
                         "pred": p[i], "q": r.q, "promo_on": r.promo_on,
                         "periodo": r.periodo})
    return pd.DataFrame(rows)


def fut_base(fut, hist):
    """Ex ante: quiebre = 0 (D-03); distribucion = media ultimos 7 dias (D-05);
    promo competidor = 0 (D-02)."""
    fut["q"] = 0.0
    fut["log1mq"] = 0.0
    fut["distribucion_numerica"] = hist["distribucion_numerica"].iloc[-7:].mean()
    fut["promo_competidor"] = 0
    return fut


def resumen(bt, nombre, solo_sin_quiebre=True):
    out = {"escenario": nombre}
    b = bt[bt.q == 0] if solo_sin_quiebre else bt
    out.update(metricas(b.y, b.pred))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = cargar(a.data)
    tr = d[d.periodo == "Entrenamiento"].copy()
    origins = list(range(FIRST_ORIGIN, len(d) - 1, H))
    print(f"Filas {len(d)}; origenes del backtest: {len(origins)} (h=1..7), "
          f"primer origen {d.fecha[origins[0]].date()}")

    # ------------------------------------------------------------------ A. estructura
    titulo("A. Estructura comercial y quiebres")
    qd = d[d.q > 0]
    rel = pd.DataFrame({
        "fecha": qd.fecha.dt.date, "q": qd.q, "promo_pct": qd.promo_pct,
        "dia_en_promo": qd.dia_en_promo, "dias_post_promo": qd.dias_post_promo.replace(99, np.nan),
        "exhib": qd.exhibicion, "y": qd.y})
    print(rel.to_string(index=False))
    ev = (d.q > 0).astype(int)
    eventos = d[(ev == 1) & (ev.shift(fill_value=0) == 0)]
    lig = ((eventos.promo_on == 1) | (eventos.dias_post_promo <= 2)).sum()
    print(f"Eventos de quiebre: {len(eventos)}; ligados a promo (durante o <=2 dias despues): {lig}")
    # prueba: log(1-q) como regresor -> coef ~1 => venta = demanda * (1 - q)
    m = sm.OLS(tr.ly, sm.add_constant(tr[BASE + ["promo_competidor"]].assign(
        log1mq=np.log(1 - tr.q)))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"Coef log(1-q) en Entrenamiento: {m.params['log1mq']:.3f} "
          f"(IC95 {m.conf_int().loc['log1mq', 0]:.2f} a {m.conf_int().loc['log1mq', 1]:.2f}) "
          "-> ~1 implica venta perdida proporcional a tiendas sin stock")
    mall = sm.OLS(d.ly, sm.add_constant(d[BASE + ["promo_competidor"]].assign(
        log1mq=np.log(1 - d.q)))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"Coef log(1-q) en 181 dias: {mall.params['log1mq']:.3f} "
          f"(IC95 {mall.conf_int().loc['log1mq', 0]:.2f} a {mall.conf_int().loc['log1mq', 1]:.2f})")
    d["y_desc"] = d.y / (1 - d.q)
    d["ly_desc"] = np.log(d.y_desc)
    perdida = (d.y_desc - d.y).sum()
    print(f"Venta perdida estimada por quiebres (y/(1-q) - y): {perdida:,.0f} u. "
          f"= {100 * perdida / d.y.sum():.2f} % de la venta del semestre")

    # ------------------------------------------------------------------ B. quiebre
    titulo("B. Tratamiento del quiebre (evaluado ex ante, quiebre=0, en dias SIN quiebre)")
    cols_q = BASE + ["q"]
    esc = []
    esc.append(resumen(backtest(d, lambda: LogOLS(cols_q), fut_base, origins), "B1 control q, pron. q=0"))
    esc.append(resumen(backtest(d, lambda: LogOLS(BASE + ["log1mq"]), fut_base, origins,
                                train_filter=lambda t: t.assign(log1mq=np.log(1 - t.q))),
                       "B2 control log(1-q), pron. 0"))
    esc.append(resumen(backtest(d, lambda: LogOLS(BASE, target="ly_desc"), fut_base, origins),
                       "B3 des-censurar y/(1-q)"))
    esc.append(resumen(backtest(d, lambda: LogOLS(BASE), fut_base, origins,
                                train_filter=lambda t: t[t.q == 0]), "B4 excluir dias con quiebre"))
    esc.append(resumen(backtest(d, lambda: LogOLS(BASE), fut_base, origins), "B5 ignorar quiebre"))
    tb = pd.DataFrame(esc)
    print(tb.round(2).to_string(index=False))
    tb.to_csv(a.out / "B_quiebre.csv", index=False)

    # ------------------------------------------------------------------ C. competidor
    titulo("C. Promo del competidor (D-02 no se conoce). Backtest, dias sin quiebre")
    def fut_comp_esperada(fut, hist):
        fut = fut_base(fut, hist)
        fut["promo_competidor"] = hist["promo_competidor"].mean()
        return fut

    def fut_comp_en_curso(fut, hist):
        """Se observa al origen si el competidor esta en promo; sus bloques duran 7 dias,
        asi que se conocen los dias restantes del bloque ya iniciado."""
        real = fut["promo_competidor"].values.copy()
        fut = fut_base(fut, hist)
        b_last = hist["comp_bloque"].iloc[-1]
        if b_last >= 0:
            fut["promo_competidor"] = ((fut["comp_bloque"] == b_last) & (real == 1)).astype(int)
        return fut

    cols_c = BASE + ["q", "promo_competidor"]
    escc = [
        resumen(backtest(d, lambda: LogOLS(BASE + ["q"]), fut_base, origins), "C1 excluir variable"),
        resumen(backtest(d, lambda: LogOLS(cols_c), fut_base, origins), "C2 control, pron. 0"),
        resumen(backtest(d, lambda: LogOLS(cols_c), fut_comp_esperada, origins), "C3 control, pron. tasa hist."),
        resumen(backtest(d, lambda: LogOLS(cols_c), fut_comp_en_curso, origins), "C4 control, bloque en curso"),
        resumen(backtest(d, lambda: LogOLS(cols_c), lambda f, h: f.assign(
            q=0.0, distribucion_numerica=h["distribucion_numerica"].iloc[-7:].mean()), origins),
            "C5 oraculo (competidor real)"),
    ]
    tc = pd.DataFrame(escc)
    print(tc.round(2).to_string(index=False))
    tc.to_csv(a.out / "C_competidor.csv", index=False)

    # ------------------------------------------------------------------ D. rezagos
    titulo("D. Rezagos de venta validos con horizonte 7")
    # ACF de residuos del modelo (Entrenamiento) en rezagos >= 7
    mres = LogOLS(BASE + ["q", "promo_competidor"]).fit(tr)
    r = mres.resid.values
    acf = {k: np.corrcoef(r[k:], r[:-k])[0, 1] for k in [1, 2, 7, 8, 14]}
    print("ACF residuos log (Entr.):", {k: round(v, 3) for k, v in acf.items()})
    # correlacion de y(t-7) con la promo de t-7 (lo que realmente transporta el rezago)
    d["lag7"] = d.y.shift(7)
    d["lag7_promo"] = d.promo_on.shift(7)
    d["lag7_q"] = d.q.shift(7)
    sub = d.dropna(subset=["lag7"])
    print(f"Corr(y(t-7), promo(t-7)) = {sub.lag7.corr(sub.lag7_promo):.2f}; "
          f"Corr(y(t-7), promo(t)) = {sub.lag7.corr(sub.promo_on):.2f}; "
          f"dias en que t-7 tuvo quiebre: {(sub.lag7_q > 0).sum()}; "
          f"dias con estado de promo distinto en t y t-7: {(sub.lag7_promo != sub.promo_on).sum()} de {len(sub)}")

    # D1 y(t-7) crudo como regresor (log), D2 des-censurado, D3 baseline "naive ajustado por promo"
    d["l_lag7"] = np.log(d.y.shift(7))
    d["l_lag7_desc"] = np.log((d.y / (1 - d.q)).shift(7))
    origins_l = [o for o in origins if o >= 14]
    cols_d = BASE + ["q", "promo_competidor"]
    escd = [resumen(backtest(d, lambda: LogOLS(cols_d), fut_base, origins_l), "D0 sin rezago")]
    escd.append(resumen(backtest(d, lambda: LogOLS(cols_d + ["l_lag7"]), fut_base, origins_l,
                                 train_filter=lambda t: t.dropna(subset=["l_lag7"])),
                        "D1 + log y(t-7) crudo"))
    escd.append(resumen(backtest(d, lambda: LogOLS(cols_d + ["l_lag7_desc"]), fut_base, origins_l,
                                 train_filter=lambda t: t.dropna(subset=["l_lag7_desc"])),
                        "D2 + log y(t-7) des-censurado"))

    # D3/D4: correccion de nivel = media de residuos log de los ultimos k dias observados
    def post_nivel(k, peso=1.0):
        def f(p, m, trn, fx):
            return p * np.exp(peso * m.resid.iloc[-k:].mean())
        return f
    for k in (7, 14, 28):
        escd.append(resumen(backtest(d, lambda: LogOLS(cols_d), fut_base, origins_l,
                                     post=post_nivel(k)), f"D3 corr. nivel residuos {k}d"))
    escd.append(resumen(backtest(d, lambda: LogOLS(cols_d), fut_base, origins_l,
                                 post=post_nivel(14, 0.5)), "D4 corr. nivel 14d amortiguada x0,5"))
    # D5 naive t-7 ajustado por promo (practica comun de planeadores)
    tmp = []
    for o in origins_l:
        mm = LogOLS(cols_d).fit(d.iloc[: o + 1])
        fut = d.iloc[o + 1: o + 1 + H]
        pf = mm.predict_log(fut_base(fut.copy(), d.iloc[: o + 1]))
        past = d.iloc[[i - 7 for i in fut.index]].copy()
        pp = mm.predict_log(past)
        p = past.y.values * np.exp(pf - pp)          # y(t-7) * uplift(t)/uplift(t-7) (incluye q de t-7)
        tmp.append(pd.DataFrame({"y": fut.y.values, "pred": p, "q": fut.q.values}))
    escd.append(resumen(pd.concat(tmp), "D5 naive t-7 ajustado por modelo"))
    tmp = []
    for o in origins_l:
        fut = d.iloc[o + 1: o + 1 + H]
        tmp.append(pd.DataFrame({"y": fut.y.values, "pred": d.y.shift(7).iloc[fut.index].values,
                                 "q": fut.q.values}))
    escd.append(resumen(pd.concat(tmp), "D6 naive t-7 crudo"))
    td = pd.DataFrame(escd)
    print(td.round(2).to_string(index=False))
    td.to_csv(a.out / "D_rezagos.csv", index=False)

    # ------------------------------------------------------------------ E. efectos de negocio
    titulo("E. Efectos de negocio (OLS log y, HAC(7)); base = promo, exhib, distr, quincena, dow, q, promo_comp")
    ctrl = BASE + ["q", "promo_competidor"]
    pruebas = {
        "post_promo_1_3d": ["post_1_3"],
        "post_promo_4_7d": ["post_4_7"],
        "post_1_3 + post_4_7": ["post_1_3", "post_4_7"],
        "post_1_3 x descuento previo": ["post_1_3_x_desc"],
        "pre_promo_1_3d": ["pre_1_3"],
        "promo_dia1": ["promo_dia1"],
        "dia_en_promo (fatiga)": ["dia_en_promo"],
        "promo^2 (rend. decrecientes)": ["promo_sq"],
        "promo x exhibicion": ["promo_x_exhib"],
        "quincena x sin_promo": ["quinc_x_nopromo"],
    }
    filas = []
    for per, dd in (("Entr", d[d.periodo == "Entrenamiento"]), ("Todo", d)):
        for nom, extra in pruebas.items():
            mm = sm.OLS(dd.ly, sm.add_constant(dd[ctrl + extra].astype(float))).fit(
                cov_type="HAC", cov_kwds={"maxlags": 7})
            for v in extra:
                filas.append({"periodo": per, "prueba": nom, "var": v, "coef": mm.params[v],
                              "efecto_%": 100 * (np.exp(mm.params[v]) - 1), "p": mm.pvalues[v],
                              "R2adj": mm.rsquared_adj})
    te = pd.DataFrame(filas)
    print(te.round(4).to_string(index=False))
    te.to_csv(a.out / "E_efectos_negocio.csv", index=False)

    # quincena descompuesta
    mm = sm.OLS(d.ly, sm.add_constant(d[[c for c in ctrl if c != "dia_quincena"] + ["fin_mes", "quinc_15"]]
                                       .astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"Quincena descompuesta (181 d): fin de mes/dia 1 {100*(np.exp(mm.params['fin_mes'])-1):+.1f} % "
          f"(p {mm.pvalues['fin_mes']:.3f}); dias 14-16 {100*(np.exp(mm.params['quinc_15'])-1):+.1f} % "
          f"(p {mm.pvalues['quinc_15']:.3f})")

    # rendimientos decrecientes: uplift por nivel (dummies) vs exp(b*promo)
    lv = sorted(d.promo_pct.unique())
    dm = pd.get_dummies(d.promo_pct, prefix="p").astype(float).drop(columns="p_0.0")
    xx = pd.concat([d[[c for c in ctrl if c != "promo_pct"]], dm], axis=1).astype(float)
    mm = sm.OLS(d.ly, sm.add_constant(xx)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    ml = sm.OLS(d.ly, sm.add_constant(d[ctrl].astype(float))).fit()
    b = ml.params["promo_pct"]
    print("Uplift por nivel de descuento (181 d): nivel | dummies | log-lineal | elasticidad arco")
    for l in lv[1:]:
        c = mm.params[f"p_{l}"]
        print(f"  {l:.2f} | x{np.exp(c):.2f} (IC95 x{np.exp(mm.conf_int().loc[f'p_{l}',0]):.2f}-"
              f"x{np.exp(mm.conf_int().loc[f'p_{l}',1]):.2f}) | x{np.exp(b*l):.2f} | "
              f"{(np.exp(c)-1)/l:.1f} % por pp/100")

    # feriados / fechas comerciales: residuo del modelo en fechas clave
    mres_all = LogOLS(ctrl).fit(d)
    d["res"] = mres_all.resid.values
    fechas = {"Año nuevo": "2026-01-01", "Jueves Santo": "2026-04-02", "Viernes Santo": "2026-04-03",
              "Sabado Santo": "2026-04-04", "Dia del trabajo": "2026-05-01",
              "Dia de la madre (10-may)": "2026-05-10", "Dia del padre (21-jun)": "2026-06-21"}
    sd = d.res.std()
    print(f"Residuo log del modelo en fechas especiales (sd residuo = {sd:.3f}):")
    for k, f in fechas.items():
        rr = d.loc[d.fecha == f, "res"]
        if len(rr):
            print(f"  {k:28s} {f}: {100*(np.exp(rr.iloc[0])-1):+.1f} % (z {rr.iloc[0]/sd:+.1f})")
    # semana previa a dia de la madre
    w = d[(d.fecha >= "2026-05-04") & (d.fecha <= "2026-05-10")].res
    print(f"  Semana 4-10 may (pre dia de la madre): residuo medio {100*(np.exp(w.mean())-1):+.1f} %")

    # perfil de residuos alrededor de las promos
    prof = d.groupby(d.dias_post_promo.clip(upper=15)).res.agg(["mean", "count"])
    prof["efecto_%"] = 100 * (np.exp(prof["mean"]) - 1)
    print("Residuo medio (sin variables post-promo) por dias desde fin de promo (99 = sin promo previa):")
    print(prof.round(3).to_string())
    perf = d[d.promo_on == 1].groupby("dia_en_promo").res.agg(["mean", "count"])
    perf["efecto_%"] = 100 * (np.exp(perf["mean"]) - 1)
    print("Residuo medio por dia dentro de la promo:")
    print(perf.round(3).to_string())

    # ------------------------------------------------------------------ F. metrica asimetrica
    titulo(f"F. Metrica asimetrica: costo sub:sobre = {CU:.0f}:{CO:.0f} -> cuantil objetivo {CU/(CU+CO):.2f}")
    cols_f = BASE + ["q"]   # D-02: sin promo competidor como entrada
    cols_fc = BASE + ["q", "promo_competidor"]

    def post_cuantil(tau):
        def f(p, m, trn, fx):
            # cuantil empirico del residuo log in-sample (solo dias sin quiebre) -> factor
            rr = m.resid[trn.q.values == 0]
            return np.exp(m.predict_log(fx)) * np.exp(np.quantile(rr, tau))
        return f

    class QReg:
        def __init__(self, cols, tau):
            self.cols, self.tau = cols, tau

        def fit(self, dd):
            X = sm.add_constant(dd[self.cols].astype(float), has_constant="add")
            self.m = sm.QuantReg(dd["ly"], X).fit(q=self.tau, max_iter=5000)
            return self

        def predict(self, dd):
            X = sm.add_constant(dd[self.cols].astype(float), has_constant="add")
            return np.exp(np.asarray(self.m.predict(X)))

    tau = CU / (CU + CO)
    escf = []
    variantes = {
        "F1 media (smearing), sin comp": (lambda: LogOLS(cols_f), fut_base, None),
        "F2 media, comp control=0": (lambda: LogOLS(cols_fc), fut_base, None),
        "F3 mediana (exp log), comp=0": (lambda: LogOLS(cols_fc), fut_base,
                                         lambda p, m, t, fx: np.exp(m.predict_log(fx))),
        f"F4 cuantil emp. {tau:.2f}, comp=0": (lambda: LogOLS(cols_fc), fut_base, post_cuantil(tau)),
        "F5 cuantil emp. 0.67, comp=0": (lambda: LogOLS(cols_fc), fut_base, post_cuantil(2 / 3)),
        "F6 cuantil emp. 0.90, comp=0": (lambda: LogOLS(cols_fc), fut_base, post_cuantil(0.90)),
        f"F7 QuantReg log tau={tau:.2f}, comp=0": (lambda: QReg(cols_fc, tau), fut_base, None),
    }
    bts = {}
    for nom, (mk, fp, po) in variantes.items():
        bt = backtest(d, mk, fp, origins, post=po)
        bts[nom] = bt
        for seg, b_ in (("sin quiebre", bt[bt.q == 0]), ("promo", bt[(bt.q == 0) & (bt.promo_on == 1)]),
                        ("sin promo", bt[(bt.q == 0) & (bt.promo_on == 0)]),
                        ("Validación", bt[(bt.q == 0) & (bt.periodo == "Validación")])):
            row = {"escenario": nom, "segmento": seg}
            row.update(metricas(b_.y, b_.pred))
            escf.append(row)
        # error acumulado semanal (lo que mueve el inventario en un ciclo de reposicion)
        wk = bt[bt.q == 0].groupby("origen")[["y", "pred"]].sum()
        row = {"escenario": nom, "segmento": "suma 7 dias"}
        row.update(metricas(wk.y, wk.pred))
        escf.append(row)
    tf = pd.DataFrame(escf)
    print(tf.round(2).to_string(index=False))
    tf.to_csv(a.out / "F_metrica_asimetrica.csv", index=False)

    # sensibilidad: cuantil optimo vs ratio de costos
    print("\nCosto (sub*cu + sobre*co, % de la demanda) segun cuantil y ratio, backtest sin quiebre:")
    filas = []
    for tq in (0.5, 0.6, 0.67, 0.75, 0.8, 0.9):
        bt = backtest(d, lambda: LogOLS(cols_fc), fut_base, origins, post=post_cuantil(tq))
        b_ = bt[bt.q == 0]
        fila = {"cuantil": tq}
        for ratio in (1, 2, 3, 4):
            under = np.clip(b_.y - b_.pred, 0, None).sum()
            over = np.clip(b_.pred - b_.y, 0, None).sum()
            fila[f"costo_{ratio}:1"] = 100 * (ratio * under + over) / b_.y.sum()
        fila["WAPE_%"] = 100 * np.abs(b_.pred - b_.y).sum() / b_.y.sum()
        fila["Bias_%"] = 100 * (b_.pred - b_.y).sum() / b_.y.sum()
        fila["Cobertura_%"] = 100 * np.mean(b_.pred >= b_.y)
        filas.append(fila)
    ts = pd.DataFrame(filas)
    print(ts.round(2).to_string(index=False))
    ts.to_csv(a.out / "F_sensibilidad_cuantil.csv", index=False)

    # error por horizonte
    bt = bts["F2 media, comp control=0"]
    hh = bt[bt.q == 0].groupby("h").apply(lambda b_: pd.Series(metricas(b_.y, b_.pred))[["WAPE_%", "Bias_%"]])
    print("\nError por horizonte (F2):")
    print(hh.round(2).T.to_string())

    # ------------------------------------------------------------------ G. conjuntos de negocio
    titulo("G. Conjuntos de variables de negocio (objetivo des-censurado y/(1-q), competidor = "
           "bloque en curso). Mediana (P50) y cuantil 0.75")
    d["p10"] = np.isclose(d.promo_pct, 0.10).astype(int)
    d["p15"] = np.isclose(d.promo_pct, 0.15).astype(int)
    d["p25"] = np.isclose(d.promo_pct, 0.25).astype(int)
    d["p30"] = np.isclose(d.promo_pct, 0.30).astype(int)
    d["ly_desc_dn"] = d.ly_desc - np.log(d.distribucion_numerica)   # elasticidad 1 a la distribucion
    sinq = [c for c in BASE if c != "dia_quincena"]
    conjuntos = {
        "G0 BASE (promo lineal, quincena)": (BASE, "ly_desc"),
        "G1 BASE sin quincena": (sinq, "ly_desc"),
        "G2 fin_mes en vez de quincena": (sinq + ["fin_mes"], "ly_desc"),
        "G3 promo + promo^2": (BASE + ["promo_sq"], "ly_desc"),
        "G4 promo por niveles (dummies)": ([c for c in BASE if c != "promo_pct"] + ["p10", "p15", "p25", "p30"], "ly_desc"),
        "G5 BASE + post_4_7": (BASE + ["post_4_7"], "ly_desc"),
        "G6 BASE + promo x exhib": (BASE + ["promo_x_exhib"], "ly_desc"),
        "G7 distribucion como offset (elast. 1)": ([c for c in BASE if c != "distribucion_numerica"], "ly_desc_dn"),
        "G8 G2 + promo^2": (sinq + ["fin_mes", "promo_sq"], "ly_desc"),
    }

    class LogOLSComp(LogOLS):
        """Objetivo des-censurado; promo_competidor como control. Si target es ly_desc_dn,
        se suma log(DN) del futuro."""
        def predict_log(self, dd):
            pl = super().predict_log(dd)
            if self.target == "ly_desc_dn":
                pl = pl + np.log(dd["distribucion_numerica"].values)
            return pl

        def predict(self, dd):
            return np.exp(self.predict_log(dd)) * self.smear

    def post_med(p, m, t, fx):
        return np.exp(m.predict_log(fx))

    def post_q75(p, m, t, fx):
        return np.exp(m.predict_log(fx)) * np.exp(np.quantile(m.resid, 0.75))

    filas = []
    for nom, (cols, tgt) in conjuntos.items():
        for etiqueta, po in (("P50", post_med), ("P75", post_q75)):
            bt = backtest(d, lambda: LogOLSComp(cols + ["promo_competidor"], target=tgt),
                          fut_comp_en_curso, origins, post=po)
            b_ = bt[bt.q == 0]
            fila = {"conjunto": nom, "salida": etiqueta}
            fila.update(metricas(b_.y, b_.pred))
            bp = b_[b_.promo_on == 1]
            fila["WAPE_promo_%"] = 100 * np.abs(bp.pred - bp.y).sum() / bp.y.sum()
            fila["Bias_promo_%"] = 100 * (bp.pred - bp.y).sum() / bp.y.sum()
            filas.append(fila)
    tg = pd.DataFrame(filas)
    print(tg.round(2).to_string(index=False))
    tg.to_csv(a.out / "G_conjuntos_negocio.csv", index=False)
    # coeficiente de distribucion libre en 181 d (elasticidad implicita)
    mm = sm.OLS(d.ly_desc, sm.add_constant(d[BASE + ["promo_competidor"]].assign(
        l_dn=np.log(d.distribucion_numerica)).drop(columns="distribucion_numerica").astype(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"Elasticidad ventas a distribucion numerica (log-log, 181 d): {mm.params['l_dn']:.2f} "
          f"(IC95 {mm.conf_int().loc['l_dn', 0]:.2f} a {mm.conf_int().loc['l_dn', 1]:.2f}); 1 = proporcional")


if __name__ == "__main__":
    main()
