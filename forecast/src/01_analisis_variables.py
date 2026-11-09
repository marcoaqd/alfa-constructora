#!/usr/bin/env python3
"""
01_analisis_variables.py
========================
Analisis estadistico de variables para pronosticar `unidades_vendidas`.

Cubre: calidad de datos, distribucion del objetivo, varianza ~0, fuga de informacion
(leakage), redundancias/VIF, relacion univariada con el objetivo, modelos comparados en
validacion (sin shuffle, ajustados solo con Entrenamiento), importancia por permutacion,
ablacion, escenario con/sin leakage y estabilidad (drift) Entrenamiento vs Validacion.

Uso (desde la raiz del repo o cualquier lugar):
    python3 -I forecast/src/01_analisis_variables.py
    python3 -I forecast/src/01_analisis_variables.py --data ruta/al/archivo.xlsx --out carpeta_salida

Salidas: texto por stdout + CSVs en forecast/reportes/01_salidas/ (por defecto).
Todos los procesos aleatorios usan semilla fija (SEED=42) para ser reproducibles.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

SEED = 42
REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "forecast" / "data" / "reto1_dataset.xlsx"
DEFAULT_OUT = REPO / "forecast" / "reportes" / "01_salidas"
Y = "unidades_vendidas"
TRAIN, VALID = "Entrenamiento", "Validación"
DOW_ORDER = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


# --------------------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------------------
def titulo(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "WAPE_%": 100 * np.sum(np.abs(e)) / np.sum(y),
        "MAPE_%": 100 * np.mean(np.abs(e) / y),
        "Bias": np.mean(e),
        "Bias_%": 100 * np.sum(e) / np.sum(y),
    }


def cargar(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hojas = pd.read_excel(path, sheet_name=None)
    df = hojas["Datos_completos"].copy()
    dic = hojas.get("Diccionario")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df, dic


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features (numericas + dummies). No usa informacion futura."""
    X = df.copy()
    for d in DOW_ORDER[:-1]:  # referencia: domingo
        X[f"dow_{d}"] = (X["dia_semana"] == d).astype(int)
    for r in ["R2", "R3", "R4"]:  # referencia: R1
        X[f"reg_{r}"] = (X["codigo_region"] == r).astype(int)
    X["medios_on"] = (X["inversion_medios_usd"] > 0).astype(int)
    X["promo_x_exhib"] = X["promo_pct"] * X["exhibicion"]
    return X


DOW_COLS = [f"dow_{d}" for d in DOW_ORDER[:-1]]
REG_COLS = ["reg_R2", "reg_R3", "reg_R4"]
# grupos de columnas que se permutan / quitan juntos
GRUPOS = {"dia_semana": DOW_COLS, "codigo_region": REG_COLS}


def expand(cols):
    out = []
    for c in cols:
        out += GRUPOS.get(c, [c])
    return out


NUM_ORIG = [
    "mes", "ventas_hace_7_dias", "ventas_mismo_dia_anio_anterior", "precio", "promo_pct",
    "exhibicion", "precio_competidor", "indice_precio_relativo", "promo_competidor",
    "distribucion_numerica", "quiebre_stock_pct", "inversion_medios_usd", "grp_tv",
    "dia_quincena", "inflacion_mensual_pct", "confianza_consumidor", "temperatura_c",
    "unidades_facturadas_mismo_dia",
]
BINARIAS = ["exhibicion", "promo_competidor", "dia_quincena", "medios_on"]

# Conjuntos de variables (nombres de grupo; se expanden a dummies)
NUCLEO = ["promo_pct", "exhibicion", "promo_competidor", "quiebre_stock_pct", "dia_quincena",
          "dia_semana", "distribucion_numerica"]
NUCLEO_SIN_QUIEBRE = [c for c in NUCLEO if c != "quiebre_stock_pct"]
TODO_SIN_FACT = ["ventas_hace_7_dias", "ventas_mismo_dia_anio_anterior", "precio", "promo_pct",
                 "exhibicion", "precio_competidor", "indice_precio_relativo", "promo_competidor",
                 "distribucion_numerica", "quiebre_stock_pct", "inversion_medios_usd", "grp_tv",
                 "dia_quincena", "inflacion_mensual_pct", "confianza_consumidor", "temperatura_c",
                 "codigo_region", "dia_semana"]
CON_LEAK = TODO_SIN_FACT + ["unidades_facturadas_mismo_dia"]
CANDIDATOS_EXTRA = ["ventas_hace_7_dias", "ventas_mismo_dia_anio_anterior", "precio",
                    "precio_competidor", "indice_precio_relativo", "inversion_medios_usd",
                    "grp_tv", "temperatura_c", "inflacion_mensual_pct", "confianza_consumidor",
                    "codigo_region", "promo_x_exhib"]


# --------------------------------------------------------------------------------------
# modelos
# --------------------------------------------------------------------------------------
class LogLinear:
    """Regresion lineal/Ridge sobre log(y) con correccion de retransformacion (smearing de Duan)."""

    def __init__(self, ridge=True):
        self.ridge = ridge

    def fit(self, X, y):
        ly = np.log(np.asarray(y, float))
        if self.ridge:
            self.m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25),
                                                              cv=TimeSeriesSplit(5)))
        else:
            self.m = LinearRegression()
        self.m.fit(X, ly)
        self.smear = np.mean(np.exp(ly - self.m.predict(X)))
        return self

    def predict(self, X):
        return np.exp(self.m.predict(X)) * self.smear


def nuevo_modelo(nombre):
    if nombre == "OLS_y":
        return LinearRegression()
    if nombre == "OLS_logy":
        return LogLinear(ridge=False)
    if nombre == "Ridge_logy":
        return LogLinear(ridge=True)
    if nombre == "RandomForest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                     random_state=SEED, n_jobs=-1)
    if nombre == "GradBoosting":
        return GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, min_samples_leaf=3, random_state=SEED)
    raise ValueError(nombre)


def ajustar_predecir(nombre, Xtr, ytr, Xte):
    m = nuevo_modelo(nombre).fit(Xtr, ytr)
    return m, m.predict(Xte)


def cv_temporal(nombre, X, y, cols, inicios=(60, 75, 90, 105), h=15):
    """Origen rodante dentro de Entrenamiento (ventana expansiva)."""
    errs, ys = [], []
    for n in inicios:
        _, p = ajustar_predecir(nombre, X.iloc[:n][cols], y.iloc[:n], X.iloc[n:n + h][cols])
        errs.append(np.abs(p - y.iloc[n:n + h].values))
        ys.append(y.iloc[n:n + h].values)
    e, yy = np.concatenate(errs), np.concatenate(ys)
    return {"CV_MAE": e.mean(), "CV_WAPE_%": 100 * e.sum() / yy.sum()}


def perm_importance(model, Xva, yva, grupos, n_rep=30):
    rng = np.random.default_rng(SEED)
    base = np.mean(np.abs(model.predict(Xva) - yva))
    res = []
    for g in grupos:
        cols = [c for c in expand([g]) if c in Xva.columns]
        if not cols:
            continue
        deltas = []
        for _ in range(n_rep):
            Xp = Xva.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[idx]
            deltas.append(np.mean(np.abs(model.predict(Xp) - yva)) - base)
        res.append({"variable": g, "dMAE_media": np.mean(deltas), "dMAE_sd": np.std(deltas),
                    "dMAE_%": 100 * np.mean(deltas) / base})
    return pd.DataFrame(res).sort_values("dMAE_media", ascending=False)


def psi(a, b, bins=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:  # variable discreta: usar valores
        vals = np.unique(np.concatenate([a, b]))
        pa = np.array([(a == v).mean() for v in vals])
        pb = np.array([(b == v).mean() for v in vals])
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        pa = np.histogram(a, qs)[0] / len(a)
        pb = np.histogram(b, qs)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, dic = cargar(args.data)
    X = features(df)
    tr, va = X[X.periodo == TRAIN].reset_index(drop=True), X[X.periodo == VALID].reset_index(drop=True)
    ytr, yva = tr[Y].astype(float), va[Y].astype(float)

    # ---------------------------------------------------------------- 1. calidad
    titulo("1. CALIDAD DE DATOS")
    print(f"Filas={len(df)}, columnas={df.shape[1]}, Entrenamiento={len(tr)} "
          f"({tr.fecha.min().date()}..{tr.fecha.max().date()}), Validación={len(va)} "
          f"({va.fecha.min().date()}..{va.fecha.max().date()})")
    print(f"Nulos totales={int(df.isna().sum().sum())}; filas duplicadas={int(df.duplicated().sum())}; "
          f"fechas duplicadas={int(df.fecha.duplicated().sum())}; "
          f"saltos de fecha !=1 día={int((df.fecha.diff().dt.days.dropna() != 1).sum())}")
    dow_map = {0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves", 4: "viernes", 5: "sábado", 6: "domingo"}
    print("dia_semana coherente con fecha:", (df.fecha.dt.dayofweek.map(dow_map) == df.dia_semana).mean())
    print("mes coherente con fecha:", (df.fecha.dt.month == df.mes).mean())
    regla_q = df.fecha.dt.day.isin([1, 14, 15, 16]) | (df.fecha.dt.day >= 29)
    print("dia_quincena == regla calendario (día ∈ {1,14,15,16} o día ≥ 29):",
          (regla_q.astype(int) == df.dia_quincena).mean())
    lag7_ok = (df.ventas_hace_7_dias.iloc[7:].values == df[Y].shift(7).iloc[7:].values).mean()
    print(f"ventas_hace_7_dias == unidades_vendidas.shift(7) (filas 8..181): {lag7_ok:.3f}")
    ipr = (df.precio / df.precio_competidor).round(3)
    print("indice_precio_relativo == round(precio/precio_competidor,3): max |dif| =",
          float((df.indice_precio_relativo - ipr).abs().max()))
    for c in ["promo_pct", "distribucion_numerica", "quiebre_stock_pct"]:
        print(f"  rango {c}: [{df[c].min()}, {df[c].max()}] (debe estar en [0,1])")
    neg = {c: int((df[c] < 0).sum()) for c in NUM_ORIG if c != "mes"}
    print("Valores negativos por columna:", {k: v for k, v in neg.items() if v} or "ninguno")

    calidad = []
    for c in df.columns:
        s = df[c]
        top = s.value_counts(normalize=True).iloc[0]
        row = {"variable": c, "dtype": str(s.dtype), "n_unicos": s.nunique(), "frec_valor_top": round(top, 3)}
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            row.update({"media": s.mean(), "sd": s.std(), "cv": s.std() / s.mean() if s.mean() else np.nan,
                        "min": s.min(), "max": s.max(),
                        "outliers_IQR": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0})
        calidad.append(row)
    calidad = pd.DataFrame(calidad)
    print(calidad.round(3).to_string(index=False))
    calidad.to_csv(args.out / "calidad.csv", index=False)

    print("\nVariables mensuales (valores únicos por mes):")
    print(df.groupby("mes")[["inflacion_mensual_pct", "confianza_consumidor"]].nunique().T.to_string())
    print("distribucion_numerica por mes (media, sd):")
    print(df.groupby("mes").distribucion_numerica.agg(["mean", "std"]).round(4).T.to_string())

    # ---------------------------------------------------------------- 2. objetivo
    titulo("2. DISTRIBUCIÓN DEL OBJETIVO")
    for nom, s in [("Total", df[Y]), (TRAIN, ytr), (VALID, yva)]:
        print(f"{nom:13s} n={len(s):3d} media={s.mean():7.1f} mediana={s.median():6.0f} sd={s.std():6.1f} "
              f"min={s.min():5.0f} max={s.max():5.0f} asimetría={stats.skew(s):.2f} curtosis={stats.kurtosis(s):.2f}")
    print(f"asimetría log(y) total = {stats.skew(np.log(df[Y])):.2f}")
    q1, q3 = df[Y].quantile([.25, .75])
    out_y = df[df[Y] > q3 + 1.5 * (q3 - q1)]
    print(f"Outliers altos (IQR) = {len(out_y)}; de ellos con promo_pct>0: {(out_y.promo_pct > 0).sum()}")
    print("Media del objetivo por nivel de promo_pct:")
    print(df.groupby("promo_pct")[Y].agg(["count", "mean", "std"]).round(0).to_string())
    a = acf(ytr, nlags=14, fft=False)
    print("ACF(y) Entrenamiento: " + ", ".join(f"lag{k}={a[k]:.2f}" for k in [1, 2, 3, 7, 14]))

    # ---------------------------------------------------------------- 3. leakage
    titulo("3. FUGA DE INFORMACIÓN (LEAKAGE)")
    fac = df.unidades_facturadas_mismo_dia
    d = fac - df[Y]
    print(f"unidades_facturadas_mismo_dia: r Pearson={np.corrcoef(fac, df[Y])[0, 1]:.4f}; "
          f"dif (fact - venta): media={d.mean():.1f}, sd={d.std():.1f}, |dif| media={d.abs().mean():.1f}, "
          f"|dif|/y medio={100 * (d.abs() / df[Y]).mean():.2f}%")
    m_fac = metricas(yva, va.unidades_facturadas_mismo_dia)
    print("  'Pronóstico' = facturado mismo día, en Validación:", {k: round(v, 2) for k, v in m_fac.items()})
    # lag 7: errores en transiciones de promo
    prev = pd.concat([tr, va]).reset_index(drop=True)
    cambio = (prev.promo_pct != prev.promo_pct.shift(7)).iloc[len(tr):].values
    e7 = np.abs(va.ventas_hace_7_dias - yva)
    print(f"ventas_hace_7_dias (naive t-7) en Validación: WAPE días con cambio de promo vs t-7 = "
          f"{100 * e7[cambio].sum() / yva[cambio].sum():.1f}% (n={cambio.sum()}), sin cambio = "
          f"{100 * e7[~cambio].sum() / yva[~cambio].sum():.1f}% (n={(~cambio).sum()})")
    # quiebre: ¿consecuencia de demanda previa?
    full = X.copy()
    full["y_lag1"] = full[Y].shift(1)
    full["promo_lag1"] = full.promo_pct.shift(1)
    q = full.quiebre_stock_pct > 0
    print(f"quiebre_stock_pct>0 en {q.sum()} días (Entr={int((tr.quiebre_stock_pct > 0).sum())}, "
          f"Val={int((va.quiebre_stock_pct > 0).sum())}); y medio con quiebre={full.loc[q, Y].mean():.0f} "
          f"vs sin quiebre={full.loc[~q, Y].mean():.0f}")
    print(f"  y del día anterior: con quiebre hoy={full.loc[q, 'y_lag1'].mean():.0f} vs sin={full.loc[~q, 'y_lag1'].mean():.0f}; "
          f"% días de quiebre justo después de una promo propia: "
          f"{100 * ((full.promo_lag1 > 0) & (full.promo_pct == 0) & q).sum() / q.sum():.0f}%")
    qb = (full.quiebre_stock_pct > 0).astype(int)
    print(f"  persistencia: P(quiebre hoy | quiebre ayer) = {qb[qb.shift(1) == 1].mean():.2f}; "
          f"P(quiebre hoy | sin quiebre ayer) = {qb[qb.shift(1) == 0].mean():.2f}")
    print(f"  Spearman(quiebre, y) en días sin promo propia: "
          f"{stats.spearmanr(full.loc[full.promo_pct == 0, 'quiebre_stock_pct'], full.loc[full.promo_pct == 0, Y])[0]:.3f}")

    # ---------------------------------------------------------------- 4. redundancia / VIF
    titulo("4. REDUNDANCIAS Y MULTICOLINEALIDAD")
    pares = [("precio", "promo_pct"), ("inversion_medios_usd", "grp_tv"),
             ("precio_competidor", "promo_competidor"), ("indice_precio_relativo", "precio"),
             ("indice_precio_relativo", "precio_competidor"), ("distribucion_numerica", "mes"),
             ("temperatura_c", "mes"), ("ventas_hace_7_dias", "unidades_vendidas"),
             ("unidades_facturadas_mismo_dia", "unidades_vendidas")]
    red = []
    for a_, b_ in pares:
        red.append({"par": f"{a_} ~ {b_}", "pearson": np.corrcoef(df[a_], df[b_])[0, 1],
                    "spearman": stats.spearmanr(df[a_], df[b_])[0]})
    red = pd.DataFrame(red)
    print(red.round(4).to_string(index=False))
    # precio explicado por promo_pct (precio = base*(1-promo))
    base = df.precio / (1 - df.promo_pct)
    print(f"precio/(1-promo_pct): media={base.mean():.3f}, sd={base.std():.3f}, CV={100 * base.std() / base.mean():.1f}% "
          f"-> precio ≈ {base.mean():.2f}·(1-promo_pct)")
    r2 = LinearRegression().fit(df[["promo_pct"]], df.precio).score(df[["promo_pct"]], df.precio)
    print(f"R² precio ~ promo_pct = {r2:.4f}")
    r2 = LinearRegression().fit(df[["grp_tv"]], df.inversion_medios_usd).score(df[["grp_tv"]], df.inversion_medios_usd)
    print(f"R² inversion_medios_usd ~ grp_tv = {r2:.4f}")
    r2 = LinearRegression().fit(df[["promo_competidor"]], df.precio_competidor).score(df[["promo_competidor"]], df.precio_competidor)
    print(f"R² precio_competidor ~ promo_competidor = {r2:.4f}")

    def vif(cols, data):
        A = data[cols].astype(float).copy()
        A.insert(0, "const", 1.0)
        return pd.DataFrame({"variable": cols,
                             "VIF": [variance_inflation_factor(A.values, i + 1) for i in range(len(cols))]})

    num_pred = [c for c in NUM_ORIG if c not in ("unidades_facturadas_mismo_dia",)]
    v_all = vif(num_pred, tr).sort_values("VIF", ascending=False)
    print("\nVIF (Entrenamiento) con TODAS las numéricas (sin facturado):")
    print(v_all.round(1).to_string(index=False))
    v_nuc = vif(expand(NUCLEO), tr).sort_values("VIF", ascending=False)
    print("\nVIF (Entrenamiento) del conjunto NÚCLEO:")
    print(v_nuc.round(2).to_string(index=False))
    v_all.to_csv(args.out / "vif_todas.csv", index=False)
    v_nuc.to_csv(args.out / "vif_nucleo.csv", index=False)

    # ---------------------------------------------------------------- 5. relacion con objetivo
    titulo("5. RELACIÓN UNIVARIADA CON EL OBJETIVO")
    rel_cols = [c for c in NUM_ORIG if c != "mes"] + ["medios_on", "mes"]
    rows = []
    for c in rel_cols:
        row = {"variable": c}
        for nom, data in [("tr", tr), ("va", va)]:
            if data[c].nunique() < 2:
                row[f"pearson_{nom}"] = row[f"spearman_{nom}"] = np.nan
                continue
            row[f"pearson_{nom}"] = np.corrcoef(data[c], data[Y])[0, 1]
            row[f"spearman_{nom}"] = stats.spearmanr(data[c], data[Y])[0]
        row["p_spearman_tr"] = stats.spearmanr(tr[c], tr[Y])[1]
        disc = tr[c].nunique() <= 8
        mi = np.mean([mutual_info_regression(tr[[c]], ytr, discrete_features=[disc], random_state=s)[0]
                      for s in range(10)])
        row["MI_tr"] = mi
        rows.append(row)
    # categoricas como grupo: MI con codigo ordinal
    for c in ["dia_semana", "codigo_region"]:
        codes = tr[c].astype("category").cat.codes.to_frame()
        mi = np.mean([mutual_info_regression(codes, ytr, discrete_features=[True], random_state=s)[0]
                      for s in range(10)])
        rows.append({"variable": c, "MI_tr": mi})
    rel = pd.DataFrame(rows)
    rel["dif_spearman_va_tr"] = rel["spearman_va"] - rel["spearman_tr"]
    print(rel.round(3).to_string(index=False))
    rel.to_csv(args.out / "relacion_univariada.csv", index=False)

    print("\nBinarias: diferencia de medias, Mann-Whitney y tamaño de efecto (todo / sin promo propia ni quiebre)")
    sub_np = X[(X.promo_pct == 0) & (X.quiebre_stock_pct == 0)]
    bins = []
    for c in BINARIAS + ["promo_on", "quiebre_on"]:
        for nom, data in [("todo", X), ("sin_promo_sin_quiebre", sub_np)]:
            data = data.copy()
            data["promo_on"] = (data.promo_pct > 0).astype(int)
            data["quiebre_on"] = (data.quiebre_stock_pct > 0).astype(int)
            if data[c].nunique() < 2:
                continue
            g1, g0 = data.loc[data[c] == 1, Y], data.loc[data[c] == 0, Y]
            U, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
            sp = np.sqrt(((len(g1) - 1) * g1.var() + (len(g0) - 1) * g0.var()) / (len(g1) + len(g0) - 2))
            bins.append({"variable": c, "subconjunto": nom, "n1": len(g1), "n0": len(g0),
                         "media1": g1.mean(), "media0": g0.mean(), "dif": g1.mean() - g0.mean(),
                         "dif_%": 100 * (g1.mean() / g0.mean() - 1), "cohen_d": (g1.mean() - g0.mean()) / sp,
                         "rank_biserial": 1 - 2 * U / (len(g1) * len(g0)) * 1, "p_MW": p,
                         "p_t_welch": stats.ttest_ind(g1, g0, equal_var=False)[1]})
    bins = pd.DataFrame(bins)
    # rank-biserial con signo: positivo si grupo 1 > grupo 0
    bins["rank_biserial"] = -bins["rank_biserial"]
    print(bins.round(4).to_string(index=False))
    bins.to_csv(args.out / "tests_binarias.csv", index=False)

    print("\nCategóricas: ANOVA / Kruskal-Wallis y eta² (todo / sin promo propia ni quiebre)")
    cats = []
    for c in ["dia_semana", "codigo_region", "mes"]:
        for nom, data in [("todo", X), ("sin_promo_sin_quiebre", sub_np)]:
            grupos = [g[Y].values for _, g in data.groupby(c)]
            F, pF = stats.f_oneway(*grupos)
            H, pH = stats.kruskal(*grupos)
            ss_b = sum(len(g) * (g.mean() - data[Y].mean()) ** 2 for g in grupos)
            ss_t = ((data[Y] - data[Y].mean()) ** 2).sum()
            cats.append({"variable": c, "subconjunto": nom, "k": len(grupos), "F": F, "p_ANOVA": pF,
                         "H": H, "p_Kruskal": pH, "eta2": ss_b / ss_t})
    cats = pd.DataFrame(cats)
    print(cats.round(4).to_string(index=False))
    cats.to_csv(args.out / "tests_categoricas.csv", index=False)
    print("\nMedia de y por día de semana (sin promo propia ni quiebre):")
    print(sub_np.groupby("dia_semana")[Y].mean().reindex(DOW_ORDER).round(0).to_string())

    # efectos condicionales (OLS log y, Entrenamiento) -> efecto ceteris paribus
    import statsmodels.api as sm
    cols_ols = expand(NUCLEO) + ["inversion_medios_usd", "temperatura_c", "ventas_mismo_dia_anio_anterior"] + REG_COLS
    ols = sm.OLS(np.log(ytr), sm.add_constant(tr[cols_ols].astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    tab = pd.DataFrame({"coef_logy": ols.params, "efecto_%": 100 * (np.exp(ols.params) - 1), "p_HAC": ols.pvalues})
    print(f"\nOLS log(y) en Entrenamiento con errores HAC(7) — efectos condicionales (R²={ols.rsquared:.3f}):")
    print(tab.round(4).to_string())
    tab.to_csv(args.out / "ols_condicional_train.csv")
    resid = ols.resid
    ra = acf(resid, nlags=7, fft=False)
    print("ACF residuos OLS: " + ", ".join(f"lag{k}={ra[k]:.2f}" for k in [1, 2, 7]))
    for c in ["ventas_hace_7_dias", "inversion_medios_usd", "grp_tv", "inflacion_mensual_pct",
              "confianza_consumidor", "precio_competidor", "temperatura_c"]:
        print(f"  corr(residuo, {c}) = {np.corrcoef(resid, tr[c])[0, 1]:.3f}")
    # efecto retardado de medios: rezagos y adstock (calculado sobre la serie completa, sin futuro)
    inv = X.inversion_medios_usd.astype(float)
    for lag in [1, 2, 3, 7]:
        z = inv.shift(lag).iloc[:len(tr)]
        ok = z.notna().values
        print(f"  corr(residuo, inversion_medios lag{lag}) = {np.corrcoef(resid[ok], z[ok])[0, 1]:.3f}")
    for dec in [0.3, 0.5, 0.7, 0.9]:
        ad, acc = [], 0.0
        for v in inv:
            acc = v + dec * acc
            ad.append(acc)
        print(f"  corr(residuo, adstock medios decay={dec}) = {np.corrcoef(resid, np.array(ad[:len(tr)]))[0, 1]:.3f}")

    # ---------------------------------------------------------------- 6. modelos
    titulo("6. MODELOS EN VALIDACIÓN (ajuste solo con Entrenamiento, sin shuffle)")
    res = []

    def reg(nombre, conjunto, pred, extra=None):
        r = {"modelo": nombre, "conjunto": conjunto, **metricas(yva, pred)}
        if extra:
            r.update(extra)
        res.append(r)

    reg("Naive t-7 (ventas_hace_7_dias)", "baseline, solo h≤7", va.ventas_hace_7_dias)
    ult = ytr.iloc[-7:].values
    reg("Naive estacional fijo (última semana Entr. repetida)", "baseline, h=1..61",
        np.resize(ult, len(va)))
    reg("Año anterior (crudo)", "baseline", va.ventas_mismo_dia_anio_anterior)
    k = ytr.sum() / tr.ventas_mismo_dia_anio_anterior.sum()
    reg(f"Año anterior × {k:.3f} (escala Entr.)", "baseline", va.ventas_mismo_dia_anio_anterior * k)
    reg("Media de Entrenamiento", "baseline", np.full(len(va), ytr.mean()))

    conjuntos = {
        "NÚCLEO (quiebre real)": expand(NUCLEO),
        "NÚCLEO sin quiebre": expand(NUCLEO_SIN_QUIEBRE),
        "TODO sin facturado (incl. t-7)": expand(TODO_SIN_FACT),
        "CON LEAKAGE (todo + facturado)": expand(CON_LEAK),
    }
    modelos = ["OLS_y", "OLS_logy", "Ridge_logy", "RandomForest", "GradBoosting"]
    fitted = {}
    for cn, cols in conjuntos.items():
        for mn in modelos:
            m, p = ajustar_predecir(mn, tr[cols], ytr, va[cols])
            fitted[(cn, mn)] = m
            cvm = cv_temporal(mn, tr, ytr, cols)
            reg(mn, cn, p, cvm)
    # escenario realista: entrenar con quiebre como control y pronosticar con quiebre=0
    for mn in ["Ridge_logy", "GradBoosting"]:
        m = fitted[("NÚCLEO (quiebre real)", mn)]
        va0 = va.copy()
        va0["quiebre_stock_pct"] = 0.0
        reg(mn, "NÚCLEO, quiebre=0 al pronosticar (ex ante)", m.predict(va0[expand(NUCLEO)]))
        va0["promo_competidor"] = 0
        reg(mn, "NÚCLEO, quiebre=0 y promo_comp=0 (sin info competidor)", m.predict(va0[expand(NUCLEO)]))
    res = pd.DataFrame(res)
    print(res.round(2).to_string(index=False))
    res.to_csv(args.out / "metricas_modelos.csv", index=False)

    # error con/sin quiebre por subconjunto de dias
    m = fitted[("NÚCLEO (quiebre real)", "Ridge_logy")]
    va0 = va.copy()
    va0["quiebre_stock_pct"] = 0.0
    p0 = m.predict(va0[expand(NUCLEO)])
    qd = va.quiebre_stock_pct > 0
    print(f"\nRidge_logy NÚCLEO con quiebre=0: MAE días con quiebre real (n={qd.sum()}) = "
          f"{np.abs(p0 - yva)[qd].mean():.0f}; días sin quiebre = {np.abs(p0 - yva)[~qd].mean():.0f}")

    # ---------------------------------------------------------------- 7. importancia / ablacion
    titulo("7. IMPORTANCIA POR PERMUTACIÓN (Validación) Y ABLACIÓN")
    imp_all = []
    for (cn, grupos) in [("NÚCLEO (quiebre real)", NUCLEO), ("TODO sin facturado (incl. t-7)", TODO_SIN_FACT),
                         ("CON LEAKAGE (todo + facturado)", CON_LEAK)]:
        for mn in ["Ridge_logy", "GradBoosting"]:
            pi = perm_importance(fitted[(cn, mn)], va[expand(grupos)], yva.values, grupos)
            pi.insert(0, "modelo", mn)
            pi.insert(0, "conjunto", cn)
            imp_all.append(pi)
            print(f"\n[{cn} | {mn}]")
            print(pi.round(2).to_string(index=False))
    pd.concat(imp_all).to_csv(args.out / "importancia_permutacion.csv", index=False)

    print("\nAblación sobre NÚCLEO (quitar variable, reajustar) — ΔMAE vs NÚCLEO completo:")
    abl = []
    for mn in ["Ridge_logy", "GradBoosting"]:
        _, pb = ajustar_predecir(mn, tr[expand(NUCLEO)], ytr, va[expand(NUCLEO)])
        b_val = metricas(yva, pb)["MAE"]
        b_cv = cv_temporal(mn, tr, ytr, expand(NUCLEO))["CV_MAE"]
        for g in NUCLEO:
            cols = expand([c for c in NUCLEO if c != g])
            _, p = ajustar_predecir(mn, tr[cols], ytr, va[cols])
            abl.append({"modelo": mn, "accion": f"- {g}", "MAE_val": metricas(yva, p)["MAE"],
                        "dMAE_val": metricas(yva, p)["MAE"] - b_val,
                        "dMAE_cv": cv_temporal(mn, tr, ytr, cols)["CV_MAE"] - b_cv})
        for g in CANDIDATOS_EXTRA:
            cols = expand(NUCLEO + [g])
            _, p = ajustar_predecir(mn, tr[cols], ytr, va[cols])
            abl.append({"modelo": mn, "accion": f"+ {g}", "MAE_val": metricas(yva, p)["MAE"],
                        "dMAE_val": metricas(yva, p)["MAE"] - b_val,
                        "dMAE_cv": cv_temporal(mn, tr, ytr, cols)["CV_MAE"] - b_cv})
        # sustituciones de redundantes
        for a_, b_ in [("promo_pct", "precio"), ("promo_competidor", "precio_competidor"),
                       ("promo_competidor", "indice_precio_relativo")]:
            cols = expand([b_ if c == a_ else c for c in NUCLEO])
            _, p = ajustar_predecir(mn, tr[cols], ytr, va[cols])
            abl.append({"modelo": mn, "accion": f"{a_} -> {b_}", "MAE_val": metricas(yva, p)["MAE"],
                        "dMAE_val": metricas(yva, p)["MAE"] - b_val,
                        "dMAE_cv": cv_temporal(mn, tr, ytr, cols)["CV_MAE"] - b_cv})
        print(f"  {mn}: MAE_val base = {b_val:.1f}, CV_MAE base = {b_cv:.1f}")
    abl = pd.DataFrame(abl)
    print(abl.round(1).to_string(index=False))
    abl.to_csv(args.out / "ablacion.csv", index=False)

    # ---------------------------------------------------------------- 8. drift
    titulo("8. ESTABILIDAD Entrenamiento vs Validación (drift)")
    dr = []
    for c in [Y] + [c for c in NUM_ORIG if c != "mes"]:
        ks, pks = stats.ks_2samp(tr[c], va[c])
        dr.append({"variable": c, "media_tr": tr[c].mean(), "media_va": va[c].mean(),
                   "dif_media_%": 100 * (va[c].mean() / tr[c].mean() - 1) if tr[c].mean() else np.nan,
                   "sd_tr": tr[c].std(), "sd_va": va[c].std(), "KS": ks, "p_KS": pks,
                   "PSI": psi(tr[c], va[c])})
    dr = pd.DataFrame(dr)
    print(dr.round(3).to_string(index=False))
    dr.to_csv(args.out / "drift.csv", index=False)
    print("\nNiveles de promo_pct: Entr", sorted(tr.promo_pct.unique()), "| Val", sorted(va.promo_pct.unique()))
    for c in ["inflacion_mensual_pct", "confianza_consumidor"]:
        print(f"{c}: valores Val no vistos en Entr = {sorted(set(va[c]) - set(tr[c]))}")

    # estabilidad de la relacion: OLS log(y) ajustado por separado en cada periodo
    cols_s = ["promo_pct", "exhibicion", "promo_competidor", "quiebre_stock_pct", "dia_quincena"] + DOW_COLS
    est = {}
    for nom, data in [("Entr", tr), ("Val", va)]:
        o = sm.OLS(np.log(data[Y]), sm.add_constant(data[cols_s].astype(float))).fit()
        est[f"coef_{nom}"] = o.params
        est[f"se_{nom}"] = o.bse
    est = pd.DataFrame(est)
    est["z_dif"] = (est.coef_Val - est.coef_Entr) / np.sqrt(est.se_Val ** 2 + est.se_Entr ** 2)
    print("\nCoeficientes OLS log(y) ajustados por separado (Entr vs Val) y z de la diferencia:")
    print(est.round(3).to_string())
    est.to_csv(args.out / "estabilidad_coeficientes.csv")

    # sesgo por mes del mejor modelo limpio
    m = fitted[("NÚCLEO (quiebre real)", "Ridge_logy")]
    p = m.predict(va[expand(NUCLEO)])
    print("\nRidge_logy NÚCLEO (quiebre real) — bias % por mes en Validación:")
    for mes_, g in va.assign(p=p).groupby("mes"):
        print(f"  mes {mes_}: bias={100 * (g.p - g[Y]).sum() / g[Y].sum():.1f}%  WAPE={100 * (g.p - g[Y]).abs().sum() / g[Y].sum():.1f}%")
    print(f"\nSalidas CSV en: {args.out}")


if __name__ == "__main__":
    main()
