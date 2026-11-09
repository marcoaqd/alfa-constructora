#!/usr/bin/env python3
"""
04_torneo_modelos.py
====================
Torneo de modelos de demanda con el MISMO protocolo para todos (harness `qa_backtest`):
origen rodante, ventana expansiva, horizonte 7, paso 7, escenario ex ante (quiebre = 0,
distribución = último valor, promo recortada a 30 %), sin variables del competidor en el futuro.

- Selección: filas 1-153 (hasta 2-jun), 12 folds / 84 días (mismos folds que QA).
- Hold-out final: 3-30 jun, 4 folds / 28 días; se evalúa UNA sola vez, solo para finalistas.
- Métrica principal (D-04, supuesto 3:1): pérdida asimétrica 3:1 del P75 en días sin quiebre
  (equivale a pinball tau = 0,75 × 4). También: 3:1 en todos los días, 2:1, WAPE/MAE/bias del
  P50 (sobre venta observada y sin quiebre), WAPE en promo / sin promo, cobertura del P75,
  unidades faltantes.
- Regla de decisión (QA): entre los modelos oficiales cuya diferencia con el mejor NO es
  significativa (IC95 bootstrap de folds contiene 0), gana el más simple. Un modelo más complejo
  que el núcleo QA debe además no empeorar el hold-out.

Uso:  python3 -I forecast/src/04_torneo_modelos.py [--data ruta.xlsx] [--out carpeta] [--rapido]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_backtest import DEFAULT_DATA, FECHA, REPO, SEED, backtest, cargar_excel  # noqa: E402
from modelos_torneo import (Spec, escenario, fabrica_p50, fabrica_p75, features,  # noqa: E402
                            preparar)

N_SEL = 153          # filas de selección (hasta 2026-06-02)
CS, CO = 3, 1        # D-04: supuesto por defecto 3:1
N_BOOT = 5000
OUT_DEF = REPO / "forecast" / "reportes" / "04_salidas"


# ======================================================================================
# Métricas y pruebas pareadas
# ======================================================================================
def _pa(y, p, cs, co):
    return 100 * np.sum(cs * np.clip(y - p, 0, None) + co * np.clip(p - y, 0, None)) / np.sum(y)


def evaluar(pr: pd.DataFrame) -> dict:
    y, p50, p75 = pr["y"].to_numpy(), pr["p50"].to_numpy(), pr["p75"].to_numpy()
    s = (pr["quiebre_real"] == 0).to_numpy()
    pro = (pr["promo_pct_real"] > 0).to_numpy()
    ys, p5s, p7s = y[s], p50[s], p75[s]
    return {
        "PA31_P75_%": _pa(ys, p7s, 3, 1),                       # PRINCIPAL (sin quiebre)
        "PA31_P75_todos_%": _pa(y, p75, 3, 1),
        "PA21_P75_%": _pa(ys, p7s, 2, 1),
        "PA31_P50_%": _pa(ys, p5s, 3, 1),
        "Cob_P75_%": 100 * np.mean(ys <= p7s),
        "Falt_P75_%": 100 * np.clip(ys - p7s, 0, None).sum() / ys.sum(),
        "Falt_P75_u": float(np.clip(ys - p7s, 0, None).sum()),
        "Sobr_P75_%": 100 * np.clip(p7s - ys, 0, None).sum() / ys.sum(),
        "WAPE_P50_%": 100 * np.abs(p50 - y).sum() / y.sum(),    # sobre venta observada
        "WAPE_P50_sq_%": 100 * np.abs(p5s - ys).sum() / ys.sum(),
        "MAE_P50_sq": float(np.mean(np.abs(p5s - ys))),
        "Bias_P50_%": 100 * (p50 - y).sum() / y.sum(),
        "Bias_P50_sq_%": 100 * (p5s - ys).sum() / ys.sum(),
        "WAPE_promo_sq_%": 100 * np.abs(p50[s & pro] - y[s & pro]).sum() / y[s & pro].sum(),
        "WAPE_nopromo_sq_%": 100 * np.abs(p50[s & ~pro] - y[s & ~pro]).sum() / y[s & ~pro].sum(),
        "n": len(y), "n_sq": int(s.sum()),
    }


METRICAS_PAR = {  # nombre: (columna, costo_sub, costo_sobre, solo_sin_quiebre)
    "PA31_P75": ("p75", 3, 1, True),
    "PA21_P75": ("p75", 2, 1, True),
    "WAPE_P50_sq": ("p50", 1, 1, True),
    "WAPE_P50": ("p50", 1, 1, False),
}


def pareada(pa: pd.DataFrame, pb: pd.DataFrame, metrica: str = "PA31_P75",
            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Diferencia (a - b) en puntos porcentuales con IC95 por bootstrap de folds (bloques de 7
    días), igual que `qa_backtest.diferencia_pareada` pero restringible a días sin quiebre."""
    if not pa[FECHA].reset_index(drop=True).equals(pb[FECHA].reset_index(drop=True)):
        raise ValueError("predicciones no alineadas")
    col, cs, co, sq = METRICAS_PAR[metrica]
    m = (pa["quiebre_real"] == 0).to_numpy() if sq else np.ones(len(pa), bool)
    y = pa["y"].to_numpy()
    la = cs * np.clip(y - pa[col].to_numpy(), 0, None) + co * np.clip(pa[col].to_numpy() - y, 0, None)
    lb = cs * np.clip(y - pb[col].to_numpy(), 0, None) + co * np.clip(pb[col].to_numpy() - y, 0, None)
    g = pd.DataFrame({"f": pa["fold"].to_numpy(), "a": la * m, "b": lb * m, "y": y * m}).groupby("f").sum()
    A, B, Y = g["a"].to_numpy(), g["b"].to_numpy(), g["y"].to_numpy()
    obs = 100 * (A.sum() - B.sum()) / Y.sum()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = 100 * (A[ix].sum(1) - B[ix].sum(1)) / Y[ix].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metrica": metrica, "dif_pp": float(obs), "IC95_lo": float(lo), "IC95_hi": float(hi),
            "significativa": bool(lo > 0 or hi < 0), "n_folds": len(g)}


# ======================================================================================
# Ejecución de un candidato
# ======================================================================================
def correr(spec: Spec, datos: pd.DataFrame, primer_origen: int | None = None,
           cs: float = CS, co: float = CO) -> tuple[pd.DataFrame, dict]:
    feats, esc = features(spec), escenario(spec)
    kw = dict(escenario=esc, primer_origen=primer_origen, costo_sub=cs, costo_sobre=co,
              permitir_no_ex_ante=spec.competidor is not None or any(
                  Spec.from_dict(m).competidor for m in spec.miembros))
    r50 = backtest(datos, fabrica_p50(spec), feats, nombre=spec.nombre, **kw)
    r75 = backtest(datos, fabrica_p75(spec, cs, co), feats, nombre=spec.nombre, **kw)
    pr = r50.predicciones.rename(columns={"pred": "p50"})
    pr["p75"] = r75.predicciones["pred"].to_numpy()
    pr.insert(0, "modelo", spec.nombre)
    extra = {"WAPE_train_%": r50.agregado["WAPE_train_media_%"]}
    if r75.coeficientes is not None and "factor_P75" in r75.coeficientes:
        extra["factor_P75_med"] = float(r75.coeficientes["factor_P75"].median())
        extra["_factores"] = r75.coeficientes["factor_P75"].tolist()
    return pr, extra


def md(df: pd.DataFrame, dec: int = 2) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cel = [f"{v:.{dec}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cel) + " |")
    return "\n".join(out)


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=OUT_DEF)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    t0 = time.time()

    df = preparar(cargar_excel(a.data))
    sel = df.iloc[:N_SEL]
    print(f"Selección: {sel[FECHA].iloc[0].date()}..{sel[FECHA].iloc[-1].date()} | "
          f"hold-out: {df[FECHA].iloc[N_SEL].date()}..{df[FECHA].iloc[-1].date()}")

    preds: dict[str, pd.DataFrame] = {}
    info: dict[str, dict] = {}
    specs: dict[str, Spec] = {}

    def run(s: Spec, etapa: str):
        t = time.time()
        pr, ex = correr(s, sel)
        preds[s.nombre], specs[s.nombre] = pr, s
        info[s.nombre] = {"etapa": etapa, "complejidad": s.complejidad, "oficial": s.oficial, **ex}
        e = evaluar(pr)
        print(f"  [{etapa}] {s.nombre:<46} PA31_P75 {e['PA31_P75_%']:6.2f}  WAPE_P50_sq "
              f"{e['WAPE_P50_sq_%']:6.2f}  ({time.time() - t:4.1f}s)")

    # ---------------------------------------------------------------- Etapa A: baselines + lineales
    print("\n== Etapa A: baselines y especificaciones log-lineales ==")
    A = [
        Spec("B1 Naive t-7", "naive", complejidad=1, descripcion="y(t-7) del último bloque"),
        Spec("B2 Media móvil 28d", "mm28", complejidad=2),
        Spec("B3 Baseline × uplift promo", "uplift", complejidad=6,
             descripcion="nivel sin promo 28 d × uplift por nivel de descuento (método del planeador)"),
        Spec("L0 Ridge log núcleo QA (q control)", "ridge_qa", complejidad=12,
             descripcion="spec congelada por QA: promo, exhib, dow, distr, quiebre como control"),
        Spec("L0o OLS log núcleo (q control)", "loglineal", quiebre="control", complejidad=12),
        Spec("L1 OLS log des-censurado y/(1-q)", "loglineal", quiebre="descensurar", complejidad=11),
        Spec("L1r Ridge log des-censurado", "loglineal", quiebre="descensurar", metodo="ridge",
             complejidad=11, descripcion="diagnóstico: inestabilidad del alpha de RidgeCV"),
        Spec("L2 OLS log + promo² (q control)", "loglineal", promo2=True, complejidad=13),
        Spec("L3 OLS log + fin_mes (q control)", "loglineal", extras=["fin_mes"], complejidad=13),
        Spec("L4 OLS log + dia_quincena (q control)", "loglineal", extras=["dia_quincena"], complejidad=13),
        Spec("L5 OLS log des-censurado + promo²", "loglineal", quiebre="descensurar", promo2=True,
             complejidad=12),
    ]
    for s in A:
        run(s, "A")

    # ---------------------------------------------------------------- Decisiones de especificación
    pruebas = []

    def par(a_, b_, met="PA31_P75", tag=""):
        r = {"comparacion": f"{a_} - {b_}", "a": a_, "b": b_, "contexto": tag,
             **pareada(preds[a_], preds[b_], met)}
        pruebas.append(r)
        return r

    dec = []
    n = {s.nombre[:3].strip(): s.nombre for s in A}
    L0o, L1, L2, L3, L4, L5 = n["L0o"], n["L1"], n["L2"], n["L3"], n["L4"], n["L5"]
    # (b) quiebre: des-censurar tiene 1 parámetro menos -> se adopta salvo que sea signif. peor
    rb = par(L1, L0o, tag="(b) quiebre"); rbw = par(L1, L0o, "WAPE_P50_sq", "(b) quiebre")
    desc = not (rb["significativa"] and rb["dif_pp"] > 0)
    dec.append({"id": "(b) tratamiento del quiebre", "prueba": rb["comparacion"],
                "dif_PA31_pp": rb["dif_pp"], "IC95": f"[{rb['IC95_lo']:+.2f}; {rb['IC95_hi']:+.2f}]",
                "dif_WAPE_sq_pp": rbw["dif_pp"],
                "decision": "des-censurar y/(1-q)" if desc else "quiebre como control",
                "regla": "más simple (1 parámetro menos) salvo que sea significativamente peor"})
    # (d) promo²: entra solo si es significativamente mejor
    a2, b2 = (L5, L1) if desc else (L2, L0o)
    rd = par(a2, b2, tag="(d) promo²"); rdw = par(a2, b2, "WAPE_P50_sq", "(d) promo²")
    p2 = rd["significativa"] and rd["dif_pp"] < 0
    dec.append({"id": "(d) promo² (convexidad)", "prueba": rd["comparacion"], "dif_PA31_pp": rd["dif_pp"],
                "IC95": f"[{rd['IC95_lo']:+.2f}; {rd['IC95_hi']:+.2f}]", "dif_WAPE_sq_pp": rdw["dif_pp"],
                "decision": "incluir promo²" if p2 else "no incluir",
                "regla": "entra solo si el IC95 excluye 0 a su favor"})
    # (c) calendario
    extras = []
    for cand, nom in [(L3, "fin_mes"), (L4, "dia_quincena")]:
        rc = par(cand, L0o, tag="(c) calendario"); rcw = par(cand, L0o, "WAPE_P50_sq", "(c) calendario")
        ok = rc["significativa"] and rc["dif_pp"] < 0
        dec.append({"id": f"(c) {nom}", "prueba": rc["comparacion"], "dif_PA31_pp": rc["dif_pp"],
                    "IC95": f"[{rc['IC95_lo']:+.2f}; {rc['IC95_hi']:+.2f}]", "dif_WAPE_sq_pp": rcw["dif_pp"],
                    "decision": f"incluir {nom}" if ok else "no incluir",
                    "regla": "entra solo si el IC95 excluye 0 a su favor"})
        if ok:
            extras.append(nom)
    extras = extras[:1]  # nunca ambas (fin_mes es un subconjunto de quincena)
    par(L3, L4, tag="(c) fin_mes vs quincena")
    q_mode = "descensurar" if desc else "control"
    n_lin = (11 if desc else 12) + int(p2) + len(extras)
    lin = Spec(f"LIN* OLS log ({'des-censurado' if desc else 'q control'}"
               f"{' + promo²' if p2 else ''}{' + ' + extras[0] if extras else ''})", "loglineal",
               quiebre=q_mode, promo2=p2, extras=extras, complejidad=n_lin,
               descripcion="mejor especificación lineal según (b), (c), (d)")
    existente = [k for k, s in specs.items() if s.tipo == "loglineal" and s.metodo == "ols"
                 and s.quiebre == q_mode and s.promo2 == p2 and s.extras == extras and not s.competidor]
    LIN = existente[0] if existente else None
    if LIN is None:
        run(lin, "A"); LIN = lin.nombre
    print(f"\n  Especificación lineal elegida: {LIN}")
    pd.DataFrame(dec).to_csv(a.out / "decisiones_especificacion.csv", index=False)

    # ---------------------------------------------------------------- Etapa B: familias
    print("\n== Etapa B: familias de modelos sobre la especificación elegida ==")
    sL = specs[LIN]
    B = [
        Spec("M6 Cuantílica log (tau 0,5 / 0,75)", "quantreg", quiebre=sL.quiebre, promo2=sL.promo2,
             extras=sL.extras, complejidad=2 * (sL.complejidad - 1), descripcion="P75 directo"),
        Spec("M5a GLM Poisson offset log(1-q)", "glm", familia="poisson", promo2=sL.promo2,
             extras=sL.extras, complejidad=11 + int(sL.promo2) + len(sL.extras)),
        Spec("M5b GLM BinNeg offset log(1-q)", "glm", familia="negbin", promo2=sL.promo2,
             extras=sL.extras, complejidad=12 + int(sL.promo2) + len(sL.extras)),
        Spec("M7 Híbrido LIN* + GBM residuos", "hibrido", quiebre=sL.quiebre, promo2=sL.promo2,
             extras=sL.extras, complejidad=sL.complejidad + 100),
        Spec("M8a GBM directo (log y)", "gbm", complejidad=300),
        Spec("M8b RandomForest directo", "rf", complejidad=400),
    ]
    for s in B:
        run(s, "B")

    # ---------------------------------------------------------------- Etapa C: competidor
    print("\n== Etapa C: promo del competidor como control (entrenamiento) ==")
    base_c = dict(tipo="loglineal", quiebre=sL.quiebre, promo2=sL.promo2, extras=sL.extras,
                  complejidad=sL.complejidad + 1)
    C = [
        Spec("C2 LIN* + competidor control, futuro 0", competidor="cero", oficial=True,
             descripcion="compatible con D-02 (supuesto 'sin promo del competidor')", **base_c),
        Spec("C3 LIN* + competidor control, futuro tasa hist.", competidor="tasa", oficial=True,
             descripcion="compatible con D-02 (valor esperado histórico)", **base_c),
        Spec("C4 LIN* + competidor bloque en curso (SC-08)", competidor="bloque", oficial=False,
             descripcion="requiere observar a diario la promo del competidor al origen", **base_c),
    ]
    for s in C:
        run(s, "C")

    # ---------------------------------------------------------------- Etapa D: ensambles
    print("\n== Etapa D: ensambles (promedio de P50; P75 calibrado sobre el promedio) ==")
    tab = pd.DataFrame({k: evaluar(v) for k, v in preds.items()}).T
    elig = [k for k in tab.sort_values("PA31_P75_%").index
            if specs[k].oficial and specs[k].tipo not in ("naive", "mm28", "uplift", "ensamble")]
    top, modos = [], set()
    for k in elig:   # no mezclar modos de competidor; no repetir L0/L0o (son el mismo modelo)
        m = specs[k].competidor
        if (modos - {None}) and m and m not in modos:
            continue
        if specs[k].tipo == "ridge_qa" and any(specs[t].tipo == "loglineal" and specs[t].quiebre == "control"
                                               and not specs[t].promo2 and not specs[t].extras for t in top):
            continue
        top.append(k); modos.add(m)
        if len(top) == 3:
            break
    for kk in (2, 3):
        mi = [specs[k].to_dict() for k in top[:kk]]
        run(Spec(f"E{kk} Ensamble top-{kk}", "ensamble", miembros=mi,
                 complejidad=sum(specs[k].complejidad for k in top[:kk]) + 1,
                 descripcion=" + ".join(top[:kk])), "D")

    # ---------------------------------------------------------------- Ranking y ganador
    tab = pd.DataFrame({k: {**info[k], **evaluar(v)} for k, v in preds.items()}).T
    tab = tab.drop(columns=["_factores"], errors="ignore").sort_values("PA31_P75_%")
    tab.index.name = "modelo"
    of = tab[tab["oficial"] == True]  # noqa: E712
    mejor = of.index[0]
    seg = of.index[1]
    L0 = n["L0"]
    # (1) Regla mecánica "el más simple entre los empatados con el mejor" (solo informativa)
    empates = []
    for k in of.index:
        r = pareada(preds[k], preds[mejor]) if k != mejor else {"dif_pp": 0, "significativa": False}
        if not (r["significativa"] and r["dif_pp"] > 0):
            empates.append(k)
    alt_parsimonia = of.loc[empates].sort_values(["complejidad", "PA31_P75_%"]).index[0]
    # (2) Regla de QA (oficial), secuencial: la referencia es LIN*, que ya resultó de aplicar la
    #     regla variable por variable en la etapa A. Un modelo de las etapas B-D (otra familia,
    #     competidor, ensamble) la reemplaza solo si es SIGNIFICATIVAMENTE mejor en selección
    #     (IC95 excluye 0) y luego no empeora el hold-out.
    par(LIN, L0, tag="LIN* vs núcleo QA"); par(LIN, L0, "WAPE_P50_sq", "LIN* vs núcleo QA")
    retadores = []
    for k in of.index:
        if info[k]["etapa"] in ("B", "C", "D"):
            r = par(k, LIN, tag="retador vs LIN*")
            if r["significativa"] and r["dif_pp"] < 0:
                retadores.append(k)
    ganador = min(retadores, key=lambda k: of.loc[k, "PA31_P75_%"]) if retadores else LIN
    print(f"\nMejor oficial por métrica: {mejor} | empatados con el mejor: {len(empates)} modelos")
    print(f"Alternativa por parsimonia mecánica (informativa): {alt_parsimonia}")
    print(f"Retadores significativamente mejores que LIN*: {retadores or 'ninguno'} -> ganador: {ganador}")

    par(mejor, seg, tag="mejor vs segundo")
    par(mejor, seg, "WAPE_P50_sq", "mejor vs segundo")
    par(mejor, ganador, tag="mejor por métrica vs ganador")
    par(mejor, ganador, "WAPE_P50_sq", "mejor por métrica vs ganador")
    for k in dict.fromkeys([mejor, ganador]):
        for met in ("PA31_P75", "WAPE_P50_sq", "WAPE_P50"):
            par(k, L0, met, f"{'ganador' if k == ganador else 'mejor'} vs núcleo QA")
    for b_ in (n["B3"], n["B2"]):
        par(ganador, b_, tag="FVA ganador vs baseline")
        par(ganador, b_, "WAPE_P50_sq", "FVA ganador vs baseline")
    C4 = C[2].nombre
    par(C4, ganador, tag="SC-08 competidor bloque vs ganador")
    par(C4, ganador, "WAPE_P50_sq", "SC-08 competidor bloque vs ganador")

    # ---------------------------------------------------------------- Extrapolación (árboles)
    ext = []
    ymax = {}
    for f, g in preds[LIN].groupby("fold"):
        ymax[f] = df.loc[df[FECHA] < g[FECHA].iloc[0], "unidades_vendidas"].max()
    for k in dict.fromkeys([LIN, L0, n["L2"], B[0].nombre, B[3].nombre, B[4].nombre, B[5].nombre]):
        p = preds[k].copy()
        p["ymax_train"] = p["fold"].map(ymax)
        sup = p[(p["y"] > p["ymax_train"]) & (p["quiebre_real"] == 0)]
        ext.append({"modelo": k, "dias_y>max_train": len(sup),
                    "P50/y_en_esos_dias": float((sup["p50"] / sup["y"]).mean()) if len(sup) else np.nan,
                    "max_P50_vs_max_train": float((p["p50"] / p["ymax_train"]).max()),
                    "WAPE_P50_promo30_sq_%": float(100 * np.abs(p["p50"] - p["y"])[(p["promo_pct_real"] >= 0.3) & (p["quiebre_real"] == 0)].sum()
                                                   / p["y"][(p["promo_pct_real"] >= 0.3) & (p["quiebre_real"] == 0)].sum()),
                    "Bias_P50_promo30_sq_%": float(100 * (p["p50"] - p["y"])[(p["promo_pct_real"] >= 0.3) & (p["quiebre_real"] == 0)].sum()
                                                   / p["y"][(p["promo_pct_real"] >= 0.3) & (p["quiebre_real"] == 0)].sum())})
    ext = pd.DataFrame(ext)

    # ---------------------------------------------------------------- Hold-out (una sola vez)
    print("\n== HOLD-OUT final 3-30 jun (una sola evaluación, finalistas) ==")
    finalistas = list(dict.fromkeys([ganador, mejor, seg, alt_parsimonia, L0, LIN, C4, n["B3"]]))
    ho_pred, ho = {}, []
    for k in finalistas:
        pr, ex = correr(specs[k], df, primer_origen=N_SEL)
        ho_pred[k] = pr
        ho.append({"modelo": k, **evaluar(pr), "factor_P75_med": ex.get("factor_P75_med", np.nan)})
    ho = pd.DataFrame(ho).set_index("modelo")
    print(ho[["PA31_P75_%", "PA31_P75_todos_%", "Cob_P75_%", "Falt_P75_%", "WAPE_P50_%",
              "WAPE_P50_sq_%", "Bias_P50_%"]].round(2).to_string())
    ho_pr = []
    for k in finalistas:
        if k != L0:
            for met in ("PA31_P75", "WAPE_P50_sq"):
                ho_pr.append({"comparacion": f"{k} - {L0}", "contexto": "hold-out", **pareada(ho_pred[k], ho_pred[L0], met)})
    for k in finalistas:
        if k not in (ganador, L0):
            ho_pr.append({"comparacion": f"{k} - {ganador}", "contexto": "hold-out vs ganador",
                          **pareada(ho_pred[k], ho_pred[ganador])})
    ho_pr = pd.DataFrame(ho_pr)
    # Veto QA: un retador no debe empeorar el hold-out frente a LIN*, y un modelo más complejo
    # que el núcleo QA no debe empeorarlo frente al núcleo QA.
    ganador_final, veto = ganador, False
    if ganador != LIN and ho.loc[ganador, "PA31_P75_%"] > ho.loc[LIN, "PA31_P75_%"]:
        ganador_final, veto = LIN, True
    if (specs[ganador_final].complejidad > specs[L0].complejidad
            and ho.loc[ganador_final, "PA31_P75_%"] > ho.loc[L0, "PA31_P75_%"]):
        ganador_final, veto = L0, True
    print(f"\nVeto QA por hold-out: {veto} -> ganador final: {ganador_final}")

    # ---------------------------------------------------------------- Sensibilidad a la razón de costos
    print("\n== Sensibilidad del ganador a la razón de costos (selección) ==")
    sens = []
    for cs in (1, 2, 3, 4, 5):
        pr, ex = correr(specs[ganador_final], sel, cs=cs, co=1)
        y = pr["y"].to_numpy(); s_ = (pr["quiebre_real"] == 0).to_numpy()
        fila = {"costo_sub:sobre": f"{cs}:1", "cuantil_objetivo": cs / (cs + 1),
                "factor_med": ex.get("factor_P75_med", np.nan),
                "Cob_%": 100 * np.mean(y[s_] <= pr["p75"].to_numpy()[s_]),
                "Falt_%": 100 * np.clip(y[s_] - pr["p75"].to_numpy()[s_], 0, None).sum() / y[s_].sum(),
                "Sobr_%": 100 * np.clip(pr["p75"].to_numpy()[s_] - y[s_], 0, None).sum() / y[s_].sum()}
        for c2 in (1, 2, 3, 4, 5):
            fila[f"PA{c2}1_%"] = _pa(y[s_], pr["p75"].to_numpy()[s_], c2, 1)
        sens.append(fila)
    sens = pd.DataFrame(sens)
    print(sens.round(2).to_string(index=False))

    # ---------------------------------------------------------------- Guardar
    pruebas = pd.DataFrame(pruebas)
    tab.round(4).to_csv(a.out / "ranking_seleccion.csv")
    pd.concat(preds.values()).to_csv(a.out / "predicciones_seleccion.csv", index=False)
    pd.concat(ho_pred.values()).to_csv(a.out / "predicciones_holdout.csv", index=False)
    pruebas.round(4).to_csv(a.out / "pruebas_pareadas.csv", index=False)
    ho.round(4).to_csv(a.out / "holdout.csv")
    ho_pr.round(4).to_csv(a.out / "pruebas_pareadas_holdout.csv", index=False)
    ext.round(4).to_csv(a.out / "extrapolacion.csv", index=False)
    sens.round(4).to_csv(a.out / "sensibilidad_costos.csv", index=False)
    fac = pd.DataFrame({k: info[k].get("_factores") for k in [ganador_final, L0, LIN] if info[k].get("_factores")})
    fac.index.name = "fold"
    fac.to_csv(a.out / "factores_p75_por_fold.csv")
    json.dump({"ganador": ganador_final, "spec": specs[ganador_final].to_dict(),
               "mejor_por_metrica": mejor, "veto_holdout": bool(veto),
               "alternativa_parsimonia_mecanica": alt_parsimonia, "retadores_significativos": retadores,
               "costo_sub": CS, "costo_sobre": CO, "features": features(specs[ganador_final]),
               "seleccion": {k: float(v) for k, v in evaluar(preds[ganador_final]).items()},
               "holdout": {k: float(v) for k, v in ho.loc[ganador_final].items()},
               "generado": pd.Timestamp.now().isoformat(timespec="seconds")},
              open(a.out / "ganador.json", "w"), ensure_ascii=False, indent=2)

    cols_r = ["etapa", "complejidad", "PA31_P75_%", "PA31_P75_todos_%", "PA21_P75_%", "Cob_P75_%",
              "Falt_P75_%", "Falt_P75_u", "WAPE_P50_%", "WAPE_P50_sq_%", "MAE_P50_sq", "Bias_P50_%",
              "WAPE_promo_sq_%", "WAPE_nopromo_sq_%", "WAPE_train_%", "factor_P75_med"]
    with open(a.out / "tablas_torneo.md", "w") as f:
        f.write("## Ranking (selección, 12 folds, 84 días; P75 3:1 en días sin quiebre)\n\n")
        f.write(md(tab[cols_r].reset_index()) + "\n\n## Decisiones de especificación\n\n")
        f.write(md(pd.DataFrame(dec)) + "\n\n## Pruebas pareadas (selección)\n\n")
        f.write(md(pruebas[["comparacion", "contexto", "metrica", "dif_pp", "IC95_lo", "IC95_hi", "significativa"]]) + "\n\n")
        f.write("## Hold-out (3-30 jun, 4 folds)\n\n" + md(ho.drop(columns=["n", "n_sq"]).reset_index()) + "\n\n")
        f.write(md(ho_pr[["comparacion", "contexto", "metrica", "dif_pp", "IC95_lo", "IC95_hi", "significativa"]]) + "\n\n")
        f.write("## Extrapolación\n\n" + md(ext) + "\n\n## Sensibilidad de costos\n\n" + md(sens) + "\n")
    print("\n== RANKING (selección) ==")
    print(tab[cols_r].round(2).to_string())
    print(f"\nGanador final: {ganador_final}. Salidas en {a.out}. Tiempo total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
