#!/usr/bin/env python3
"""
qa_backtest.py
==============
Harness de backtesting con origen rodante (rolling origin) para pronosticar `unidades_vendidas`
diarias, más modelos de referencia y validación de esquema para datasets externos.

Pensado para importarse desde otros scripts:

    import sys; sys.path.insert(0, "forecast/src")
    from qa_backtest import (backtest, RidgeLog, NaiveEstacional, MediaMovil,
                             EscenarioExAnte, comparar, diferencia_pareada,
                             validar_entrada, cargar_excel)

    df = cargar_excel("forecast/data/reto1_dataset.xlsx")
    res = backtest(df, RidgeLog, ["promo_pct", "exhibicion", "dia_semana",
                                  "distribucion_numerica", "quiebre_stock_pct"])
    print(res.agregado)        # métricas agregadas (dict)
    print(res.folds)           # métricas por fold (DataFrame)
    print(res.por_segmento)    # métricas por día de semana, promo, quiebre, horizonte...

Contrato del modelo
-------------------
`modelo` puede ser una clase/función fábrica sin argumentos (se llama una vez por fold) o una
instancia con `fit`/`predict` (se copia con deepcopy en cada fold). Recibe `X` = DataFrame con
las columnas de `features` (sin codificar; el modelo hace su propio encoding) e `y` en unidades
originales. `predict(X_futuro)` devuelve un array de unidades.
Las filas futuras siempre son las `horizonte` filas consecutivas que siguen a la última fila de
entrenamiento (los baselines dependen de ello).

Protocolo por defecto (decisiones D-01..D-07 de forecast/DECISIONES.md)
-----------------------------------------------------------------------
- Ventana expansiva, re-entrenamiento en cada origen, horizonte 7 días, paso 7 días.
- Los folds se alinean desde el final de la serie (todos completos de 7 días).
- Escenario ex ante: `quiebre_stock_pct` = 0 en el futuro (D-03), `distribucion_numerica` =
  último valor observado en el origen (D-05), `promo_pct` recortado a [0, 0,30] (D-07).
- Pérdida asimétrica: costo_sub por unidad faltante y costo_sobre por unidad sobrante (D-04).

Ejecución directa (demo con baselines y el Ridge log del estadístico):
    python3 -I forecast/src/qa_backtest.py [--data ruta.xlsx]
"""
from __future__ import annotations

import argparse
import copy
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

SEED = 42
OBJETIVO = "unidades_vendidas"
FECHA = "fecha"
DOW = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
# Variables que nunca pueden ser entrada del modelo (no existen al pronosticar).
PROHIBIDAS = {"unidades_facturadas_mismo_dia", OBJETIVO}
# Variables que, según D-02, no se conocen de antemano.
NO_CONOCIDAS_EX_ANTE = {"promo_competidor", "precio_competidor", "indice_precio_relativo"}

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "forecast" / "data" / "reto1_dataset.xlsx"


# ======================================================================================
# Carga
# ======================================================================================
def cargar_excel(path: str | Path, hoja: str = "Datos_completos") -> pd.DataFrame:
    """Lee la hoja de datos, parsea la fecha y ordena. No modifica valores."""
    df = pd.read_excel(path, sheet_name=hoja)
    df[FECHA] = pd.to_datetime(df[FECHA])
    return df.sort_values(FECHA).reset_index(drop=True)


def _norm_txt(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


_DOW_NORM = {_norm_txt(d): d for d in DOW}


# ======================================================================================
# Modelos
# ======================================================================================
class RidgeLog:
    """Ridge sobre log(y) con corrección de retransformación de Duan (smearing).

    Replica el `LogLinear(ridge=True)` del estadístico (01_analisis_variables.py), pero hace
    su propio encoding: `dia_semana` -> 6 dummies con categorías FIJAS (referencia domingo).
    Una categoría desconocida lanza ValueError en vez de convertirse silenciosamente en la
    referencia. El escalado y el alpha se ajustan solo con los datos de `fit`.
    """

    def __init__(self, alphas: Sequence[float] | None = None, n_splits_cv: int = 5):
        self.alphas = np.logspace(-3, 3, 25) if alphas is None else np.asarray(alphas)
        self.n_splits_cv = n_splits_cv

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "dia_semana" in X.columns:
            d = X.pop("dia_semana").map(lambda v: _DOW_NORM.get(_norm_txt(v)))
            if d.isna().any():
                raise ValueError("dia_semana con valores no reconocidos")
            for nom in DOW[:-1]:
                X[f"dow_{nom}"] = (d == nom).astype(float)
        no_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if no_num:
            raise ValueError(f"RidgeLog solo acepta numéricas + dia_semana; recibió {no_num}")
        return X.astype(float)

    def fit(self, X: pd.DataFrame, y):
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(y, float)
        if (y <= 0).any():
            raise ValueError("RidgeLog requiere y > 0 (usa log)")
        Xe = self._encode(X)
        self.columnas_ = list(Xe.columns)
        ly = np.log(y)
        cv = TimeSeriesSplit(min(self.n_splits_cv, max(2, len(y) // 10)))
        self.m_ = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas, cv=cv)).fit(Xe, ly)
        self.smear_ = float(np.mean(np.exp(ly - self.m_.predict(Xe))))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xe = self._encode(X)[self.columnas_]
        return np.exp(self.m_.predict(Xe)) * self.smear_

    @property
    def coef_series(self) -> pd.Series:
        """Coeficientes en escala original de las variables (efecto sobre log y por unidad)."""
        sc, rg = self.m_.named_steps["standardscaler"], self.m_.named_steps["ridgecv"]
        coef = rg.coef_ / np.where(sc.scale_ == 0, 1.0, sc.scale_)
        s = pd.Series(coef, index=self.columnas_)
        s["alpha"] = rg.alpha_
        return s


class NaiveEstacional:
    """Naive estacional t-7: el valor de hace 7 días (repite la última semana observada)."""

    def __init__(self, periodo: int = 7):
        self.periodo = periodo

    def fit(self, X, y):
        self.ult_ = np.asarray(y, float)[-self.periodo:]
        return self

    def predict(self, X) -> np.ndarray:
        return np.resize(self.ult_, len(X))


class MediaMovil:
    """Promedio de los últimos `k` días observados."""

    def __init__(self, k: int = 28):
        self.k = k

    def fit(self, X, y):
        self.nivel_ = float(np.mean(np.asarray(y, float)[-self.k:]))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.nivel_)


class Ajuste:
    """Envuelve un modelo y multiplica su pronóstico por `factor` (sesgo deliberado, D-04)."""

    def __init__(self, base, factor: float = 1.0):
        self.base, self.factor = base, factor

    def fit(self, X, y):
        self.m_ = (self.base() if not hasattr(self.base, "fit") else copy.deepcopy(self.base)).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X) * self.factor


# ======================================================================================
# Escenario ex ante
# ======================================================================================
@dataclass
class EscenarioExAnte:
    """Transforma las filas futuras para que solo contengan información disponible en el origen.

    fijar:        columna -> valor constante en el futuro (p. ej. quiebre = 0, D-03).
    ultimo_valor: columnas que se sustituyen por su último valor observado en el origen (D-05).
    recortar:     columna -> (min, max) para recortar valores fuera de rango (D-07).
    """
    fijar: dict = field(default_factory=lambda: {"quiebre_stock_pct": 0.0})
    ultimo_valor: list = field(default_factory=lambda: ["distribucion_numerica"])
    recortar: dict = field(default_factory=lambda: {"promo_pct": (0.0, 0.30)})

    def aplicar(self, hist: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
        fut = fut.copy()
        for c, v in self.fijar.items():
            if c in fut.columns:
                fut[c] = v
        for c in self.ultimo_valor:
            if c in fut.columns:
                fut[c] = hist[c].iloc[-1]
        for c, (lo, hi) in self.recortar.items():
            if c in fut.columns:
                fut[c] = fut[c].clip(lo, hi)
        return fut


ORACULO = EscenarioExAnte(fijar={}, ultimo_valor=[], recortar={})  # usa valores reales (solo diagnóstico)


# ======================================================================================
# Métricas
# ======================================================================================
def metricas(y, p, costo_sub: float = 2.0, costo_sobre: float = 1.0) -> dict:
    """WAPE, MAE, RMSE, bias y pérdida asimétrica.

    Bias_% = sum(p - y) / sum(y): negativo = sub-pronóstico.
    PerdAsim_% = sum(costo_sub·max(y-p,0) + costo_sobre·max(p-y,0)) / sum(y) · 100.
      Con costo_sub = costo_sobre = 1 coincide con el WAPE.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {}
    e = p - y
    sub, sobre = np.clip(-e, 0, None), np.clip(e, 0, None)
    sy = y.sum()
    return {
        "n": len(y),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "WAPE_%": float(100 * np.abs(e).sum() / sy),
        "Bias_%": float(100 * e.sum() / sy),
        "PerdAsim_%": float(100 * (costo_sub * sub.sum() + costo_sobre * sobre.sum()) / sy),
        "Sub_%": float(100 * sub.sum() / sy),          # unidades faltantes / demanda
        "DiasSub_%": float(100 * np.mean(e < 0)),      # % de días sub-pronosticados
    }


# ======================================================================================
# Backtest
# ======================================================================================
@dataclass
class ResultadoBacktest:
    nombre: str
    folds: pd.DataFrame
    predicciones: pd.DataFrame
    agregado: dict
    por_segmento: pd.DataFrame
    coeficientes: pd.DataFrame | None
    config: dict

    def resumen(self) -> pd.Series:
        s = pd.Series(self.agregado, name=self.nombre)
        return s


def _instanciar(modelo):
    if hasattr(modelo, "fit") and hasattr(modelo, "predict") and not isinstance(modelo, type):
        return copy.deepcopy(modelo)
    if callable(modelo):
        return modelo()
    raise TypeError("modelo debe ser fábrica/clase o instancia con fit/predict")


def origenes(n: int, min_train: int, horizonte: int, paso: int,
             primer_origen: int | None = None) -> list[int]:
    """Índices de origen (= nº de filas de entrenamiento), alineados desde el final para que el
    último fold termine exactamente en la última fila."""
    ult = n - horizonte
    out = list(range(ult, min_train - 1, -paso))[::-1]
    if primer_origen is not None:
        out = [o for o in out if o >= primer_origen]
    return out


def _segmentos(pred: pd.DataFrame, cs: float, co: float) -> pd.DataFrame:
    seg = {
        "dia_semana": pred["dia_semana"] if "dia_semana" in pred else None,
        "promo": np.where(pred["promo_pct_real"] > 0, "con_promo", "sin_promo") if "promo_pct_real" in pred else None,
        "exhibicion": pred["exhibicion_real"].astype(str) if "exhibicion_real" in pred else None,
        "quiebre_real": np.where(pred["quiebre_real"] > 0, "con_quiebre", "sin_quiebre") if "quiebre_real" in pred else None,
        "horizonte": pred["h"],
        "mes": pred[FECHA].dt.month,
    }
    rows = []
    for nom, s in seg.items():
        if s is None:
            continue
        for val, g in pred.groupby(np.asarray(s)):
            rows.append({"segmento": nom, "valor": val, **metricas(g["y"], g["pred"], cs, co)})
    return pd.DataFrame(rows)


def backtest(df: pd.DataFrame, modelo, features: Sequence[str], *,
             objetivo: str = OBJETIVO, min_train: int = 63, horizonte: int = 7, paso: int = 7,
             ventana_max: int | None = None, primer_origen: int | None = None,
             escenario: EscenarioExAnte | None = None,
             costo_sub: float = 2.0, costo_sobre: float = 1.0,
             permitir_no_ex_ante: bool = False, nombre: str | None = None) -> ResultadoBacktest:
    """Backtesting con origen rodante.

    En cada origen t (nº de filas de entrenamiento): ajusta con filas [t-ventana_max, t),
    pronostica las filas [t, t+horizonte) tras aplicar `escenario`, y avanza `paso` filas.

    Parámetros
    ----------
    df            DataFrame diario, consecutivo, con `fecha`, `objetivo` y `features`.
    modelo        Fábrica/clase o instancia con fit(X, y) / predict(X).
    features      Columnas de entrada (crudas). Se rechazan las de PROHIBIDAS.
    min_train     Mínimo de filas de entrenamiento del primer fold (63 = 9 semanas).
    ventana_max   None = ventana expansiva; int = ventana deslizante de ese tamaño.
    primer_origen Índice mínimo de origen (útil para evaluar solo un hold-out final).
    escenario     EscenarioExAnte (por defecto D-03/D-05/D-07). Usa ORACULO para diagnóstico.
    costo_sub / costo_sobre  Pesos de la pérdida asimétrica (D-04).
    permitir_no_ex_ante      Si False, rechaza variables no conocidas ex ante (D-02).
    """
    features = list(features)
    malas = PROHIBIDAS.intersection(features)
    if malas:
        raise ValueError(f"Variables prohibidas (leakage) en features: {sorted(malas)}")
    no_ex = NO_CONOCIDAS_EX_ANTE.intersection(features)
    if no_ex and not permitir_no_ex_ante:
        raise ValueError(f"Variables no conocidas ex ante (D-02): {sorted(no_ex)}. "
                         "Usa permitir_no_ex_ante=True solo para diagnóstico.")
    escenario = EscenarioExAnte() if escenario is None else escenario
    df = df.sort_values(FECHA).reset_index(drop=True)
    saltos = (df[FECHA].diff().dt.days.dropna() != 1).sum()
    if saltos:
        warnings.warn(f"La serie tiene {saltos} saltos de fecha != 1 día; los baselines t-7 se desalinean")
    n = len(df)
    ors = origenes(n, min_train, horizonte, paso, primer_origen)
    if not ors:
        raise ValueError("No hay folds: revisa min_train/horizonte/primer_origen")

    fold_rows, preds, coefs = [], [], []
    for k, t in enumerate(ors):
        ini = 0 if ventana_max is None else max(0, t - ventana_max)
        hist = df.iloc[ini:t]
        fut_real = df.iloc[t:t + horizonte]
        fut = escenario.aplicar(hist, fut_real)
        m = _instanciar(modelo).fit(hist[features], hist[objetivo].values)
        p = np.asarray(m.predict(fut[features]), float)
        p_tr = np.asarray(m.predict(hist[features]), float)
        y = fut_real[objetivo].values.astype(float)
        mt = metricas(hist[objetivo].values, p_tr, costo_sub, costo_sobre)
        fold_rows.append({"fold": k, "origen": hist[FECHA].iloc[-1].date(), "n_train": len(hist),
                          "desde": fut_real[FECHA].iloc[0].date(), "hasta": fut_real[FECHA].iloc[-1].date(),
                          **metricas(y, p, costo_sub, costo_sobre),
                          "WAPE_train_%": mt["WAPE_%"]})
        pr = pd.DataFrame({"fold": k, FECHA: fut_real[FECHA].values, "h": np.arange(1, len(y) + 1),
                           "y": y, "pred": p})
        for c, alias in [("dia_semana", "dia_semana"), ("promo_pct", "promo_pct_real"),
                         ("exhibicion", "exhibicion_real"), ("quiebre_stock_pct", "quiebre_real")]:
            if c in fut_real.columns:
                pr[alias] = fut_real[c].values
        preds.append(pr)
        if hasattr(m, "coef_series"):
            coefs.append(m.coef_series.rename(k))

    folds = pd.DataFrame(fold_rows)
    pred = pd.concat(preds, ignore_index=True)
    agg = metricas(pred["y"], pred["pred"], costo_sub, costo_sobre)
    agg["WAPE_fold_media_%"] = float(folds["WAPE_%"].mean())
    agg["WAPE_fold_sd_%"] = float(folds["WAPE_%"].std(ddof=1)) if len(folds) > 1 else np.nan
    agg["WAPE_train_media_%"] = float(folds["WAPE_train_%"].mean())
    agg["n_folds"] = len(folds)
    if "quiebre_real" in pred:
        sq = pred["quiebre_real"] == 0
        agg["WAPE_sin_quiebre_%"] = metricas(pred.loc[sq, "y"], pred.loc[sq, "pred"])["WAPE_%"]
    if "promo_pct_real" in pred:
        for lab, msk in [("promo", pred["promo_pct_real"] > 0), ("sin_promo", pred["promo_pct_real"] == 0)]:
            if msk.any():
                agg[f"WAPE_{lab}_%"] = metricas(pred.loc[msk, "y"], pred.loc[msk, "pred"])["WAPE_%"]
    return ResultadoBacktest(
        nombre=nombre or getattr(modelo, "__name__", type(modelo).__name__),
        folds=folds, predicciones=pred, agregado=agg,
        por_segmento=_segmentos(pred, costo_sub, costo_sobre),
        coeficientes=pd.DataFrame(coefs) if coefs else None,
        config={"features": features, "min_train": min_train, "horizonte": horizonte, "paso": paso,
                "ventana_max": ventana_max, "primer_origen": primer_origen, "escenario": escenario,
                "costo_sub": costo_sub, "costo_sobre": costo_sobre})


def comparar(resultados: Iterable[ResultadoBacktest],
             cols: Sequence[str] = ("WAPE_%", "MAE", "Bias_%", "PerdAsim_%", "Sub_%",
                                    "WAPE_fold_sd_%", "WAPE_train_media_%", "WAPE_sin_quiebre_%",
                                    "WAPE_promo_%", "WAPE_sin_promo_%", "n_folds", "n")) -> pd.DataFrame:
    """Tabla comparativa de varios backtests (una fila por modelo)."""
    t = pd.DataFrame([r.resumen() for r in resultados])
    return t[[c for c in cols if c in t.columns]]


def diferencia_pareada(a: ResultadoBacktest, b: ResultadoBacktest, metrica: str = "WAPE_%",
                       n_boot: int = 5000, seed: int = SEED) -> dict:
    """Diferencia (a - b) de una métrica agregada con IC 95 % por bootstrap de folds (bloques de
    7 días, conserva la dependencia intra-semana). Ambos backtests deben compartir folds.
    Si el IC contiene 0, la diferencia no es significativa -> preferir el modelo más simple."""
    pa, pb = a.predicciones, b.predicciones
    if not (pa[FECHA].equals(pb[FECHA])):
        raise ValueError("Los backtests no tienen los mismos días/folds")
    cs, co = a.config["costo_sub"], a.config["costo_sobre"]
    folds = pa["fold"].unique()
    rng = np.random.default_rng(seed)
    idx_f = {f: np.where(pa["fold"].values == f)[0] for f in folds}

    def met(ix):
        return (metricas(pa["y"].values[ix], pa["pred"].values[ix], cs, co)[metrica]
                - metricas(pb["y"].values[ix], pb["pred"].values[ix], cs, co)[metrica])

    obs = met(np.arange(len(pa)))
    boots = [met(np.concatenate([idx_f[f] for f in rng.choice(folds, len(folds))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metrica": metrica, "dif": float(obs), "IC95_lo": float(lo), "IC95_hi": float(hi),
            "significativa": bool(lo > 0 or hi < 0)}


def factor_asimetrico(res: ResultadoBacktest, costo_sub: float, costo_sobre: float,
                      grilla: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """Factor multiplicativo k que minimiza la pérdida asimétrica sobre las predicciones de un
    backtest. Estimarlo en el periodo de selección y evaluarlo en el hold-out (no en el mismo)."""
    grilla = np.round(np.arange(0.95, 1.21, 0.01), 2) if grilla is None else grilla
    y, p = res.predicciones["y"].values, res.predicciones["pred"].values
    t = pd.DataFrame([{"factor": k, **metricas(y, p * k, costo_sub, costo_sobre)} for k in grilla])
    return float(t.loc[t["PerdAsim_%"].idxmin(), "factor"]), t


# ======================================================================================
# Validación de esquema / rangos del dataset externo
# ======================================================================================
ESQUEMA = {  # columna: (tipo, min, max)  -> rangos FÍSICOS (violarlos = ERROR)
    "fecha": ("fecha", None, None),
    "dia_semana": ("texto", None, None),
    "unidades_vendidas": ("num", 0, None),
    "promo_pct": ("num", 0.0, 1.0),
    "exhibicion": ("bin", 0, 1),
    "distribucion_numerica": ("num", 0.0, 1.0),
    "quiebre_stock_pct": ("num", 0.0, 1.0),
    "dia_quincena": ("bin", 0, 1),
    "promo_competidor": ("bin", 0, 1),
    "precio": ("num", 0.0, None),
    "ventas_hace_7_dias": ("num", 0, None),
    "codigo_region": ("texto", None, None),
}


@dataclass
class Hallazgo:
    nivel: str      # ERROR (no predecir) / ALERTA (predecir pero avisar)
    columna: str
    mensaje: str


def _psi(a, b, bins=5) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        vals = np.unique(np.concatenate([a, b]))
        pa = np.array([(a == v).mean() for v in vals]); pb = np.array([(b == v).mean() for v in vals])
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        pa = np.histogram(a, qs)[0] / len(a); pb = np.histogram(b, qs)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def validar_entrada(nuevo: pd.DataFrame, features: Sequence[str],
                    referencia: pd.DataFrame | None = None, con_objetivo: bool = True,
                    psi_alerta: float = 0.25, holgura_rango: float = 0.0) -> list[Hallazgo]:
    """Valida un dataset nuevo ANTES de predecir. Devuelve la lista de hallazgos.

    ERROR  -> no se debe predecir (columna faltante, tipo inválido, nulos, rango físico, fechas
              duplicadas, dia_semana irreconocible o incoherente con la fecha, y <= 0).
    ALERTA -> se puede predecir pero el pronóstico extrapola o el dato es sospechoso (valor fuera
              del rango de entrenamiento, categoría nueva, drift PSI alto, saltos de fecha,
              incoherencias precio-promo / t-7 / quincena).
    """
    H: list[Hallazgo] = []
    req = {FECHA, "dia_semana", *features} | ({OBJETIVO} if con_objetivo else set())
    faltan = sorted(req - set(nuevo.columns))
    for c in faltan:
        H.append(Hallazgo("ERROR", c, "columna requerida ausente"))
    df = nuevo.copy()
    # --- tipos y nulos
    for c in sorted(req - set(faltan)):
        tipo = ESQUEMA.get(c, ("num", None, None))[0]
        if tipo == "fecha":
            f = pd.to_datetime(df[c], errors="coerce")
            if f.isna().any():
                H.append(Hallazgo("ERROR", c, f"{int(f.isna().sum())} fechas no parseables/nulas"))
            df[c] = f
        elif tipo in ("num", "bin"):
            v = pd.to_numeric(df[c], errors="coerce")
            if v.isna().any():
                H.append(Hallazgo("ERROR", c, f"{int(v.isna().sum())} valores nulos o no numéricos"))
            df[c] = v
        elif df[c].isna().any():
            H.append(Hallazgo("ERROR", c, f"{int(df[c].isna().sum())} nulos"))
    # --- fechas
    if FECHA in df and df[FECHA].notna().all():
        if df[FECHA].duplicated().any():
            H.append(Hallazgo("ERROR", FECHA, f"{int(df[FECHA].duplicated().sum())} fechas duplicadas "
                              "(¿varias regiones/series por día? el modelo es de una sola serie)"))
        d = df[FECHA].sort_values().diff().dt.days.dropna()
        if (d != 1).any():
            H.append(Hallazgo("ALERTA", FECHA, f"{int((d != 1).sum())} saltos de fecha != 1 día"))
        if not df[FECHA].is_monotonic_increasing:
            H.append(Hallazgo("ALERTA", FECHA, "filas no ordenadas por fecha (se ordenarán)"))
        if "dia_semana" in df:
            norm = df["dia_semana"].map(lambda v: _DOW_NORM.get(_norm_txt(v)))
            if norm.isna().any():
                H.append(Hallazgo("ERROR", "dia_semana", f"valores no reconocidos: "
                                  f"{sorted(set(df.loc[norm.isna(), 'dia_semana'].astype(str)))}"))
            esperado = df[FECHA].dt.dayofweek.map(dict(enumerate(DOW)))
            inc = (norm.notna() & (norm != esperado)).sum()
            if inc:
                H.append(Hallazgo("ERROR", "dia_semana", f"{int(inc)} filas incoherentes con la fecha"))
        if "dia_quincena" in df and df["dia_quincena"].notna().all():
            regla = (df[FECHA].dt.day.isin([1, 14, 15, 16]) | (df[FECHA].dt.day >= 29)).astype(int)
            inc = int((regla != df["dia_quincena"]).sum())
            if inc:
                H.append(Hallazgo("ALERTA", "dia_quincena", f"{inc} filas no siguen la regla de calendario"))
    # --- rangos físicos
    for c, (tipo, lo, hi) in ESQUEMA.items():
        if c not in df or tipo in ("fecha", "texto") or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna()
        if tipo == "bin" and not v.isin([0, 1]).all():
            H.append(Hallazgo("ERROR", c, f"valores fuera de {{0,1}}: {sorted(set(v) - {0, 1})[:5]}"))
        if lo is not None and (v < lo).any():
            H.append(Hallazgo("ERROR", c, f"{int((v < lo).sum())} valores < {lo}"))
        if hi is not None and (v > hi).any():
            H.append(Hallazgo("ERROR", c, f"{int((v > hi).sum())} valores > {hi} (¿porcentaje en 0-100?)"))
    if con_objetivo and OBJETIVO in df and pd.api.types.is_numeric_dtype(df[OBJETIVO]):
        if (df[OBJETIVO] <= 0).any():
            H.append(Hallazgo("ERROR", OBJETIVO, f"{int((df[OBJETIVO] <= 0).sum())} valores <= 0 (log(y) indefinido)"))
    # --- coherencias internas
    if {"precio", "promo_pct"} <= set(df.columns) and pd.api.types.is_numeric_dtype(df["precio"]):
        base = df["precio"] / (1 - df["promo_pct"].clip(upper=0.99))
        if referencia is not None:
            b0 = (referencia["precio"] / (1 - referencia["promo_pct"])).median()
            desv = (base / b0 - 1).abs()
            if (desv > 0.05).any():
                H.append(Hallazgo("ALERTA", "precio", f"{int((desv > 0.05).sum())} filas con precio de lista "
                                  f"distinto en >5 % al de referencia ({b0:.2f}): cambió el precio base o la "
                                  "definición de promo_pct"))
    if {"ventas_hace_7_dias", OBJETIVO} <= set(df.columns) and len(df) > 7:
        s = df.sort_values(FECHA)
        ok = (s["ventas_hace_7_dias"].iloc[7:].values == s[OBJETIVO].shift(7).iloc[7:].values).mean()
        if ok < 0.99:
            H.append(Hallazgo("ALERTA", "ventas_hace_7_dias", f"solo {100 * ok:.0f} % coincide con y(t-7)"))
    for c in PROHIBIDAS & set(features):
        H.append(Hallazgo("ERROR", c, "variable prohibida (leakage) entre las features"))
    # --- comparación con la referencia (entrenamiento)
    if referencia is not None:
        for c in features:
            if c not in df or c not in referencia:
                continue
            if not pd.api.types.is_numeric_dtype(referencia[c]):
                nuevas = sorted(set(df[c].dropna().map(_norm_txt)) - set(referencia[c].dropna().map(_norm_txt)))
                if nuevas:
                    H.append(Hallazgo("ALERTA", c, f"categorías no vistas en entrenamiento: {nuevas}"))
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            lo, hi = referencia[c].min(), referencia[c].max()
            marg = holgura_rango * (hi - lo)
            fuera = ((df[c] < lo - marg) | (df[c] > hi + marg)).sum()
            if fuera:
                H.append(Hallazgo("ALERTA", c, f"{int(fuera)} filas fuera del rango de entrenamiento "
                                  f"[{lo:g}, {hi:g}] -> extrapolación"))
            nv = set(df[c].dropna().unique()) - set(referencia[c].unique())
            if referencia[c].nunique() <= 6 and nv:
                H.append(Hallazgo("ALERTA", c, f"niveles no vistos: {sorted(float(v) for v in nv)}"))
            # PSI solo para variables no dispersas (si un valor domina >= 80 %, p. ej. quiebre = 0,
            # el PSI se dispara con pocos eventos y da falsas alarmas)
            dominante = referencia[c].value_counts(normalize=True).iloc[0]
            if df[c].nunique() > 1 and len(df) >= 20 and dominante < 0.8:
                ps = _psi(referencia[c], df[c])
                if ps > psi_alerta:
                    H.append(Hallazgo("ALERTA", c, f"drift PSI={ps:.2f} (> {psi_alerta})"))
        if "codigo_region" in df and "codigo_region" in referencia:
            nuevas = sorted(set(df["codigo_region"].dropna().astype(str)) - set(referencia["codigo_region"].astype(str)))
            if nuevas:
                H.append(Hallazgo("ALERTA", "codigo_region", f"regiones no vistas {nuevas} (no se usa en el modelo; "
                                  "confirmar que la granularidad sigue siendo 1 fila por día)"))
    return H


def hallazgos_df(H: list[Hallazgo]) -> pd.DataFrame:
    return pd.DataFrame([h.__dict__ for h in H], columns=["nivel", "columna", "mensaje"])


# ======================================================================================
# Demo
# ======================================================================================
NUCLEO_EST = ["promo_pct", "exhibicion", "promo_competidor", "quiebre_stock_pct", "dia_quincena",
              "dia_semana", "distribucion_numerica"]                  # núcleo del estadístico
NUCLEO_EX_ANTE = [c for c in NUCLEO_EST if c != "promo_competidor"]  # compatible con D-02


def _demo(path: Path) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    df = cargar_excel(path)
    n_sel = int((df[FECHA] <= "2026-06-02").sum())        # 153 filas: selección
    sel = df.iloc[:n_sel]
    print(f"Filas={len(df)}; selección = filas 0..{n_sel - 1} (hasta {sel[FECHA].iloc[-1].date()}); "
          f"hold-out = {df[FECHA].iloc[n_sel].date()}..{df[FECHA].iloc[-1].date()}")

    print("\n== Validación de esquema: Validación (may-jun) contra Entrenamiento ==")
    tr = df[df["periodo"] == "Entrenamiento"]
    va = df[df["periodo"] != "Entrenamiento"]
    print(hallazgos_df(validar_entrada(va, NUCLEO_EX_ANTE, referencia=tr)).to_string(index=False))

    modelos = {
        "Naive t-7": NaiveEstacional,
        "Media móvil 28d": MediaMovil,
        "RidgeLog núcleo ex ante": RidgeLog,
    }
    for etiqueta, datos, prim in [("SELECCIÓN (backtest)", sel, None), ("HOLD-OUT final", df, n_sel)]:
        res = [backtest(datos, m, NUCLEO_EX_ANTE, nombre=k, primer_origen=prim) for k, m in modelos.items()]
        print(f"\n== {etiqueta}: h=7, paso=7, ventana expansiva, min_train=63, costo_sub=2 ==")
        print(comparar(res).round(2).to_string())
        print("RidgeLog vs Media móvil:", diferencia_pareada(res[2], res[1]))
    r = backtest(sel, RidgeLog, NUCLEO_EX_ANTE, nombre="RidgeLog")
    print("\nPor fold (RidgeLog, selección):")
    print(r.folds.round(2).to_string(index=False))
    print("\nCoeficientes por fold (RidgeLog):")
    print(r.coeficientes.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    _demo(ap.parse_args().data)
