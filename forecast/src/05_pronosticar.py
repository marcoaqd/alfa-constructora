#!/usr/bin/env python3
"""
05_pronosticar.py
=================
Pipeline de producción: entrena el modelo ganador del torneo (04_salidas/ganador.json) con la
historia disponible y pronostica un archivo externo con el mismo esquema.

Pasos: (1) validar_entrada (esquema, rangos, drift; ERROR detiene), (2) escenario ex ante
(quiebre = 0 [D-03], distribución = último valor observado [D-05], promo recortada a 30 % [D-07],
sin promo del competidor futura [D-02]), (3) P50 y P75 (P75 = P50 × factor calibrado fuera de
muestra, costo sub:sobre 3:1 [D-04]), (4) alertas por fila, (5) métricas si el archivo trae
`unidades_vendidas`.

Modos
-----
una-vez  (defecto) entrena con la historia y pronostica TODO el archivo de una vez (no usa las
         ventas del archivo; si el archivo cubre > 7 días, el horizonte supera D-01 y se avisa).
rodante  simula la operación semanal: para cada bloque de 7 filas del archivo, re-entrena con la
         historia + las filas anteriores del archivo (requiere unidades_vendidas y quiebre en el
         archivo) y pronostica el bloque.

Uso
---
python3 -I forecast/src/05_pronosticar.py --data-nuevo externo.xlsx [--modo una-vez|rodante]
    [--data historia.xlsx] [--entrenar-hasta AAAA-MM-DD] [--ganador ganador.json] [--out carpeta]
Código de salida 2 si la validación encuentra ERRORES (no se pronostica).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_backtest import (DEFAULT_DATA, FECHA, OBJETIVO, REPO, hallazgos_df,  # noqa: E402
                         validar_entrada)
from modelos_torneo import (COMP, Q, CalibradoCuantil, QuantLog, Spec, escenario,  # noqa: E402
                            fabrica_p50, features, preparar)

GANADOR_DEF = REPO / "forecast" / "reportes" / "04_salidas" / "ganador.json"
OUT_DEF = REPO / "forecast" / "reportes" / "05_salidas"
SPEC_RESPALDO = Spec("L1 OLS log des-censurado y/(1-q)", "loglineal", quiebre="descensurar", complejidad=11)
CS, CO = 3, 1
PROMO_MAX = 0.30
H = 7


def leer(path: Path, hoja: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    hojas = pd.ExcelFile(path).sheet_names
    return pd.read_excel(path, sheet_name=hoja if hoja in hojas else hojas[0])


class Modelo:
    """Envuelve el ganador: P50 y P75 con la misma interfaz para cualquier Spec."""

    def __init__(self, spec: Spec):
        self.spec = spec

    def fit(self, X, y):
        if self.spec.tipo == "quantreg":
            self.m50_ = QuantLog(0.5, self.spec.quiebre, self.spec.promo2).fit(X, y)
            self.m75_ = QuantLog(CS / (CS + CO), self.spec.quiebre, self.spec.promo2).fit(X, y)
            self.factor_ = np.nan
        else:
            self.cal_ = CalibradoCuantil(fabrica_p50(self.spec), escenario(self.spec), CS, CO).fit(X, y)
            self.factor_ = self.cal_.factor_
        return self

    def predecir(self, X):
        if self.spec.tipo == "quantreg":
            return self.m50_.predict(X), self.m75_.predict(X)
        p50 = self.cal_.predict_p50(X)
        return p50, p50 * self.factor_

    def coeficientes(self) -> pd.DataFrame:
        base = self.m50_ if self.spec.tipo == "quantreg" else self.cal_.m_
        if hasattr(base, "coef_series"):
            s = base.coef_series
            escala = np.array([0.10 if p in ("promo_pct", "distribucion_numerica") else 1.0 for p in s.index])
            ef = np.where(np.isin(s.index, ["intercepto_log", "alpha"]), np.nan, 100 * (np.exp(s.values * escala) - 1))
            return pd.DataFrame({"parametro": s.index, "valor_log": s.values,
                                 "efecto_%": ef, "por": np.where(escala < 1, "+0,10", "+1 / dummy")})
        return pd.DataFrame(columns=["parametro", "valor"])


def metricas_eval(out: pd.DataFrame) -> pd.DataFrame:
    y, p50, p75 = out["unidades_vendidas_real"].to_numpy(float), out["P50"].to_numpy(), out["P75"].to_numpy()
    filas = []
    masks = {"todos los días (venta observada)": np.ones(len(y), bool)}
    if "quiebre_real" in out:
        masks["días sin quiebre (venta = demanda)"] = (out["quiebre_real"] == 0).to_numpy()
    for nom, m in masks.items():
        if m.sum() == 0:
            continue
        yy, a, b = y[m], p50[m], p75[m]
        filas.append({"conjunto": nom, "n": int(m.sum()),
                      "WAPE_P50_%": 100 * np.abs(a - yy).sum() / yy.sum(),
                      "Bias_P50_%": 100 * (a - yy).sum() / yy.sum(),
                      "MAE_P50": float(np.mean(np.abs(a - yy))),
                      "PerdAsim_3a1_P75_%": 100 * (3 * np.clip(yy - b, 0, None) + np.clip(b - yy, 0, None)).sum() / yy.sum(),
                      "Cobertura_P75_%": 100 * np.mean(yy <= b),
                      "Faltante_P75_%": 100 * np.clip(yy - b, 0, None).sum() / yy.sum(),
                      "Faltante_P75_u": float(np.clip(yy - b, 0, None).sum())})
    return pd.DataFrame(filas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-nuevo", type=Path, required=True, help="Excel/CSV externo (mismo esquema)")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="historia para entrenar")
    ap.add_argument("--hoja", default="Datos_completos")
    ap.add_argument("--modo", choices=["una-vez", "rodante"], default="una-vez")
    ap.add_argument("--entrenar-hasta", default=None, help="corta la historia en esa fecha (pruebas)")
    ap.add_argument("--ganador", type=Path, default=GANADOR_DEF)
    ap.add_argument("--out", type=Path, default=OUT_DEF)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.data_nuevo.stem + ("_rodante" if a.modo == "rodante" else "")

    # ---------------------------------------------------------------- modelo y datos
    if a.ganador.exists():
        g = json.load(open(a.ganador))
        spec = Spec.from_dict(g["spec"])
    else:
        spec = SPEC_RESPALDO
        print(f"AVISO: no existe {a.ganador}; se usa el modelo de respaldo {spec.nombre}")
    feats = features(spec)
    esc = escenario(spec)
    hist = preparar(leer(a.data, a.hoja))
    if a.entrenar_hasta:
        hist = hist[hist[FECHA] <= pd.Timestamp(a.entrenar_hasta)].reset_index(drop=True)
    crudo = leer(a.data_nuevo, a.hoja)
    print(f"Modelo: {spec.nombre} | features: {feats} | modo: {a.modo}")
    print(f"Historia: {len(hist)} días ({hist[FECHA].min().date()}..{hist[FECHA].max().date()}) | "
          f"archivo nuevo: {len(crudo)} filas")

    # ---------------------------------------------------------------- validación
    tiene_y = OBJETIVO in crudo and pd.to_numeric(crudo[OBJETIVO], errors="coerce").notna().all()
    requeridas = ["promo_pct", "exhibicion"]
    if a.modo == "rodante":
        requeridas += ["distribucion_numerica", Q] + ([COMP] if spec.competidor else [])
    elif spec.competidor == "bloque":
        requeridas += [COMP]
    opcionales = [c for c in feats if c not in requeridas and c != "dia_semana" and c in crudo.columns
                  and c not in ("fin_mes", "dia_quincena")]
    H_ = validar_entrada(crudo, requeridas + opcionales, referencia=hist,
                         con_objetivo=(a.modo == "rodante") or tiene_y)
    hal = hallazgos_df(H_)
    if FECHA in crudo:
        f = pd.to_datetime(crudo[FECHA], errors="coerce")
        solap = f.isin(hist[FECHA]).sum()
        if solap:
            nivel = "ERROR" if a.modo == "rodante" else "ALERTA"
            hal.loc[len(hal)] = [nivel, FECHA, f"{solap} fechas ya están en la historia de entrenamiento "
                                 "(pronóstico in-sample; usar --entrenar-hasta para evaluar fuera de muestra)"]
        gap = (f.min() - hist[FECHA].max()).days if f.notna().any() else 1
        if gap > 1:
            hal.loc[len(hal)] = ["ALERTA", FECHA, f"el archivo empieza {gap} días después del último dato "
                                 "de entrenamiento: el horizonte supera 1..7 días (D-01)"]
    hal.to_csv(a.out / f"hallazgos_{stem}.csv", index=False)
    if len(hal):
        print("\nHallazgos de validación:\n" + hal.to_string(index=False))
    if (hal["nivel"] == "ERROR").any():
        print(f"\nERROR de validación: no se pronostica. Detalle en {a.out / f'hallazgos_{stem}.csv'}")
        return 2

    # ---------------------------------------------------------------- preparación
    nuevo = preparar(crudo)
    real = nuevo.copy()
    if Q not in nuevo:
        nuevo[Q] = 0.0
    if "distribucion_numerica" not in nuevo:
        nuevo["distribucion_numerica"] = np.nan   # el escenario la reemplaza por el último valor (D-05)
    if COMP in feats and COMP not in nuevo:
        nuevo[COMP] = 0.0                          # el escenario la reemplaza (no se usa el futuro)
    niveles_vistos = set(np.round(hist["promo_pct"].unique(), 4))

    # ---------------------------------------------------------------- pronóstico
    bloques = [(0, len(nuevo))] if a.modo == "una-vez" else [(i, min(i + H, len(nuevo))) for i in range(0, len(nuevo), H)]
    filas, modelo = [], None
    for ini, fin in bloques:
        train = pd.concat([hist, nuevo.iloc[:ini]], ignore_index=True) if ini else hist
        modelo = Modelo(spec).fit(train[feats], train[OBJETIVO].to_numpy(float))
        fut = esc.aplicar(train[feats], nuevo.iloc[ini:fin][feats])
        p50, p75 = modelo.predecir(fut)
        ult = train[FECHA].max()
        for j, i in enumerate(range(ini, fin)):
            r = nuevo.iloc[i]
            al = []
            pr = float(r["promo_pct"])
            if pr > PROMO_MAX:
                al.append(f"promo {pr:.0%} recortada a {PROMO_MAX:.0%} (D-07, extrapolación)")
            if round(min(pr, PROMO_MAX), 4) not in niveles_vistos:
                al.append(f"nivel de promo {pr:.0%} no visto en entrenamiento (interpolado)")
            dist_arch = r["distribucion_numerica"]
            dist_uso = float(fut["distribucion_numerica"].iloc[j])
            if pd.notna(dist_arch) and abs(float(dist_arch) - dist_uso) > 0.02:
                al.append(f"distribución del archivo {float(dist_arch):.3f} ≠ último valor usado {dist_uso:.3f} (D-05)")
            if not (hist["distribucion_numerica"].min() <= dist_uso <= hist["distribucion_numerica"].max()):
                al.append(f"distribución {dist_uso:.3f} fuera del rango de entrenamiento (extrapolación)")
            dias = (r[FECHA] - ult).days
            if dias > H:
                al.append(f"horizonte {dias} días > 7 (D-01)")
            if dias <= 0:
                al.append("fecha dentro del entrenamiento (in-sample)")
            fila = {"fecha": r[FECHA].date(), "dia_semana": r["dia_semana"], "origen": ult.date(),
                    "dias_adelante": dias, "promo_pct_usada": float(fut["promo_pct"].iloc[j]),
                    "exhibicion": int(r["exhibicion"]), "distribucion_usada": dist_uso,
                    "P50": round(float(p50[j]), 1), "P75": round(float(p75[j]), 1),
                    "factor_P75": modelo.factor_, "alertas": "; ".join(al)}
            if tiene_y:
                fila["unidades_vendidas_real"] = float(real[OBJETIVO].iloc[i])
                if Q in real:
                    fila["quiebre_real"] = float(real[Q].iloc[i])
            filas.append(fila)
    out = pd.DataFrame(filas)

    # ---------------------------------------------------------------- salida
    met = metricas_eval(out) if tiene_y else pd.DataFrame()
    info = pd.DataFrame({"clave": ["modelo", "modo", "costo_sub:sobre", "entrenamiento_hasta", "n_entrenamiento",
                                   "factor_P75 (último ajuste)", "features", "escenario"],
                         "valor": [spec.nombre, a.modo, f"{CS}:{CO}", str(hist[FECHA].max().date()), len(hist),
                                   modelo.factor_, ", ".join(feats), repr(esc)]})
    xlsx = a.out / f"pronostico_{stem}.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        out.to_excel(w, sheet_name="pronostico", index=False)
        hal.to_excel(w, sheet_name="hallazgos", index=False)
        if len(met):
            met.to_excel(w, sheet_name="metricas", index=False)
        info.to_excel(w, sheet_name="modelo", index=False)
        modelo.coeficientes().to_excel(w, sheet_name="coeficientes", index=False)
    out.to_csv(a.out / f"pronostico_{stem}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nPronóstico ({len(out)} filas). Primeras filas:")
    print(out.head(8).to_string(index=False))
    print(f"Filas con alertas: {(out['alertas'] != '').sum()}")
    if len(met):
        print("\nMétricas contra la venta real del archivo:\n" + met.round(2).to_string(index=False))
    print(f"\nEscrito: {xlsx} y {a.out / f'pronostico_{stem}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
