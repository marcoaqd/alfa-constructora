#!/usr/bin/env python3
"""
modelos_torneo.py
=================
Modelos candidatos del torneo (04_torneo_modelos.py) y del pipeline de producción
(05_pronosticar.py). Todos cumplen el contrato del harness `qa_backtest.backtest`:
`fit(X, y)` con X = DataFrame de features crudas (sin codificar) e y en unidades;
`predict(X_futuro)` devuelve unidades. Todo encoding/escalado/ajuste usa solo los datos de
`fit` (los de entrenamiento de cada fold).

Convenciones
------------
- P50: pronóstico de mediana / media de demanda (S&OP).
- P75: cuantil 0,75 de la demanda (reposición; D-04 con costo sub:sobre = 3:1).
  Se obtiene con `CalibradoCuantil` (P50 × factor calibrado con residuos FUERA de muestra de las
  últimas 8 semanas anteriores al origen, backtest interno h = 7 dentro de la ventana de
  entrenamiento) o directo con `QuantLog(tau=0,75)`.
- Quiebre (`quiebre_stock_pct`): 'control' (regresor, 0 al pronosticar), 'descensurar'
  (objetivo y/(1-q)), 'offset' (GLM con offset log(1-q)).
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from qa_backtest import SEED, EscenarioExAnte, MediaMovil, NaiveEstacional, RidgeLog, origenes

Q = "quiebre_stock_pct"
COMP = "promo_competidor"
BASE = ["promo_pct", "exhibicion", "dia_semana", "distribucion_numerica", Q]
Q_MAX = 0.95  # tope para y/(1-q) (máximo observado 0,57)


# ======================================================================================
# Preparación (solo calendario determinístico: se deriva de la fecha, nunca del archivo)
# ======================================================================================
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d.sort_values("fecha").reset_index(drop=True)
    dia = d["fecha"].dt.day
    d["fin_mes"] = ((dia >= 29) | (dia == 1)).astype(int)
    d["dia_quincena"] = (dia.isin([1, 14, 15, 16]) | (dia >= 29)).astype(int)
    return d


def codificar(X: pd.DataFrame, promo2: bool = False, quitar: tuple = ()) -> pd.DataFrame:
    """dia_semana -> 6 dummies fijas (ref. domingo; error ante valores desconocidos) y,
    opcionalmente, promo_pct^2. Se aplica DESPUÉS del escenario (promo ya recortada)."""
    X = X.drop(columns=[c for c in quitar if c in X.columns])
    Xe = RidgeLog._encode(X)
    if promo2 and "promo_pct" in Xe:
        Xe["promo_pct2"] = Xe["promo_pct"] ** 2
    return Xe


def _q(X: pd.DataFrame) -> np.ndarray:
    return np.clip(X[Q].to_numpy(float), 0, Q_MAX) if Q in X else np.zeros(len(X))


def _inst(m):
    if hasattr(m, "fit") and not isinstance(m, type):
        return copy.deepcopy(m)
    return m()


class _Estandar:
    """Estandarización + constante para los modelos de statsmodels (solo con datos de fit)."""

    def fit(self, Xe: pd.DataFrame):
        self.cols_ = list(Xe.columns)
        self.mu_ = Xe.mean()
        sd = Xe.std(ddof=0)
        self.sd_ = sd.where(sd > 1e-12, 1.0)
        return self

    def transform(self, Xe: pd.DataFrame) -> np.ndarray:
        Z = ((Xe[self.cols_] - self.mu_) / self.sd_).to_numpy(float)
        return np.column_stack([np.ones(len(Z)), Z])


# ======================================================================================
# Modelos
# ======================================================================================
class LogLineal:
    """Ridge (o OLS) sobre log(y) con smearing de Duan. Variantes: promo^2, tratamiento del
    quiebre ('control' | 'descensurar'). Con quiebre='control', promo2=False y metodo='ridge'
    equivale exactamente a `qa_backtest.RidgeLog`. Nota: el alpha de RidgeCV (TimeSeriesSplit
    interno con pocas filas) es inestable entre folds; OLS es el método por defecto del torneo."""

    def __init__(self, quiebre: str = "control", promo2: bool = False, metodo: str = "ridge"):
        self.quiebre, self.promo2, self.metodo = quiebre, promo2, metodo

    def _X(self, X):
        return codificar(X, self.promo2, quitar=() if self.quiebre == "control" else (Q,))

    def _y(self, X, y):
        y = np.asarray(y, float)
        return y / (1 - _q(X)) if self.quiebre == "descensurar" else y

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression, RidgeCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        ly = np.log(self._y(X, y))
        Xe = self._X(X)
        self.columnas_ = list(Xe.columns)
        if self.metodo == "ridge":
            cv = TimeSeriesSplit(min(5, max(2, len(ly) // 10)))
            reg = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)
        else:
            reg = LinearRegression()
        self.m_ = make_pipeline(StandardScaler(), reg).fit(Xe, ly)
        self.smear_ = float(np.mean(np.exp(ly - self.m_.predict(Xe))))
        return self

    def predict_log(self, X):
        return self.m_.predict(self._X(X)[self.columnas_])

    def predict(self, X):
        return np.exp(self.predict_log(X)) * self.smear_

    @property
    def coef_series(self) -> pd.Series:
        sc, rg = self.m_.steps[0][1], self.m_.steps[-1][1]
        s = pd.Series(rg.coef_ / np.where(sc.scale_ == 0, 1.0, sc.scale_), index=self.columnas_)
        s["intercepto_log"] = rg.intercept_ - float(np.sum(rg.coef_ * sc.mean_ / np.where(sc.scale_ == 0, 1.0, sc.scale_)))
        if hasattr(rg, "alpha_"):
            s["alpha"] = rg.alpha_
        return s


class QuantLog:
    """Regresión cuantílica sobre log(demanda). Los cuantiles son equivariantes ante exp(), así
    que exp(q_tau(log y)) = q_tau(y): da el P50 (tau=0,5) o el P75 (tau=0,75) directo."""

    def __init__(self, tau: float = 0.5, quiebre: str = "descensurar", promo2: bool = False):
        self.tau, self.quiebre, self.promo2 = tau, quiebre, promo2

    def _X(self, X):
        return codificar(X, self.promo2, quitar=() if self.quiebre == "control" else (Q,))

    def fit(self, X, y):
        import statsmodels.api as sm
        y = np.asarray(y, float)
        if self.quiebre == "descensurar":
            y = y / (1 - _q(X))
        Xe = self._X(X)
        self.est_ = _Estandar().fit(Xe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_ = sm.QuantReg(np.log(y), self.est_.transform(Xe)).fit(q=self.tau, max_iter=10000)
        return self

    def predict(self, X):
        return np.exp(self.est_.transform(self._X(X)) @ np.asarray(self.res_.params))


class GLMConteo:
    """GLM log-link Poisson o Binomial Negativa (NB2, alpha por momentos sobre el Poisson) con
    offset log(1-q): venta = demanda × (1-q). Pronostica la media de la demanda (offset 0)."""

    def __init__(self, familia: str = "poisson", promo2: bool = False):
        self.familia, self.promo2 = familia, promo2

    def fit(self, X, y):
        import statsmodels.api as sm
        y = np.asarray(y, float)
        off = np.log(1 - _q(X))
        Xe = codificar(X, self.promo2, quitar=(Q,))
        self.est_ = _Estandar().fit(Xe)
        Z = self.est_.transform(Xe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sm.GLM(y, Z, family=sm.families.Poisson(), offset=off).fit()
            if self.familia == "negbin":
                mu = np.asarray(r.fittedvalues)
                self.alpha_ = float(max(1e-8, np.mean(((y - mu) ** 2 - mu) / mu ** 2)))
                r = sm.GLM(y, Z, family=sm.families.NegativeBinomial(alpha=self.alpha_), offset=off).fit()
        self.res_ = r
        return self

    def predict(self, X):
        Z = self.est_.transform(codificar(X, self.promo2, quitar=(Q,)))
        return np.exp(Z @ np.asarray(self.res_.params))


class ArbolDirecto:
    """GradientBoosting (sobre log y, con smearing) o RandomForest (sobre y). El quiebre entra
    como variable (0 al pronosticar). Se incluyen para documentar que no extrapolan."""

    def __init__(self, tipo: str = "gbm"):
        self.tipo = tipo

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        y = np.asarray(y, float)
        Xe = codificar(X)
        self.cols_ = list(Xe.columns)
        if self.tipo == "gbm":
            self.m_ = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05,
                                                subsample=0.8, random_state=SEED).fit(Xe, np.log(y))
            self.smear_ = float(np.mean(np.exp(np.log(y) - self.m_.predict(Xe))))
        else:
            self.m_ = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED,
                                            n_jobs=1).fit(Xe, y)
        return self

    def predict(self, X):
        Xe = codificar(X)[self.cols_]
        if self.tipo == "gbm":
            return np.exp(self.m_.predict(Xe)) * self.smear_
        return self.m_.predict(Xe)


class Hibrido:
    """Log-lineal + GradientBoosting pequeño sobre los residuos log del lineal."""

    def __init__(self, quiebre: str = "control", promo2: bool = False, metodo: str = "ols"):
        self.quiebre, self.promo2, self.metodo = quiebre, promo2, metodo

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingRegressor
        self.lin_ = LogLineal(self.quiebre, self.promo2, self.metodo).fit(X, y)
        ly = np.log(self.lin_._y(X, y))
        r = ly - self.lin_.predict_log(X)
        Xe = self.lin_._X(X)
        self.gb_ = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                             subsample=0.8, random_state=SEED).fit(Xe, r)
        self.smear_ = float(np.mean(np.exp(ly - self.lin_.predict_log(X) - self.gb_.predict(Xe))))
        return self

    def predict(self, X):
        Xe = self.lin_._X(X)[self.lin_.columnas_]
        return np.exp(self.lin_.predict_log(X) + self.gb_.predict(Xe)) * self.smear_


class BaselineUplift:
    """Método del planeador: nivel base × uplift por nivel de descuento.
    - Demanda d = y/(1-q). Base de cada día = media de d en días SIN promo de los 28 días previos.
    - Uplift de un nivel visto = mediana de d/base en los días con ese nivel; nivel no visto =
      exp(b·promo) con b ajustado por origen sobre los niveles vistos.
    - Pronóstico = base al origen (media de d sin promo de los últimos 28 días) × uplift."""

    def fit(self, X, y):
        d = np.asarray(y, float) / (1 - _q(X))
        promo = X["promo_pct"].to_numpy(float)
        n = len(d)
        base = np.full(n, np.nan)
        for t in range(n):
            ok = (np.arange(n) < t) & (np.arange(n) >= t - 28) & (promo == 0)
            if ok.sum() >= 5:
                base[t] = d[ok].mean()
        rat = d / base
        self.uplift_ = {}
        for lv in np.unique(promo[promo > 0]):
            r = rat[(promo == lv) & np.isfinite(rat)]
            if len(r):
                self.uplift_[round(float(lv), 4)] = float(np.median(r))
        if self.uplift_:
            lv = np.array(list(self.uplift_)); lu = np.log(np.array(list(self.uplift_.values())))
            self.b_ = float(np.sum(lv * lu) / np.sum(lv ** 2))
        else:
            self.b_ = 2.5
        nopromo = np.where(promo[-28:] == 0)[0]
        self.nivel_ = float(d[-28:][nopromo].mean()) if len(nopromo) else float(np.median(d))
        return self

    def predict(self, X):
        out = []
        for lv in X["promo_pct"].to_numpy(float):
            k = round(float(lv), 4)
            out.append(1.0 if k == 0 else self.uplift_.get(k, float(np.exp(self.b_ * k))))
        return self.nivel_ * np.array(out)


class Ensamble:
    """Promedio simple de los P50 de varios modelos (cada uno con sus propias columnas)."""

    def __init__(self, miembros: list):
        self.miembros = miembros  # lista de (fabrica, features)

    def fit(self, X, y):
        self.m_ = [(_inst(f).fit(X[c], y), c) for f, c in self.miembros]
        return self

    def predict(self, X):
        return np.mean([m.predict(X[c]) for m, c in self.m_], axis=0)


class CalibradoCuantil:
    """P75 = P50 × factor. El factor minimiza la pérdida costo_sub:costo_sobre sobre las
    predicciones FUERA de muestra de las últimas `n_inner` semanas anteriores al origen
    (backtest interno con origen rodante, h = 7, mismo escenario ex ante; solo días sin
    quiebre, donde la venta = demanda). Usa únicamente datos de `fit`."""

    def __init__(self, base, escenario: EscenarioExAnte, costo_sub: float = 3, costo_sobre: float = 1,
                 n_inner: int = 8, min_inner: int = 42, horizonte: int = 7):
        self.base, self.escenario = base, escenario
        self.cs, self.co = costo_sub, costo_sobre
        self.n_inner, self.min_inner, self.h = n_inner, min_inner, horizonte

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.m_ = _inst(self.base).fit(X, y)
        ys, ps = [], []
        for t in origenes(len(X), self.min_inner, self.h, self.h)[-self.n_inner:]:
            hist, fut = X.iloc[:t], X.iloc[t:t + self.h]
            p = _inst(self.base).fit(hist, y[:t]).predict(self.escenario.aplicar(hist, fut))
            ok = _q(fut) == 0
            ys.append(y[t:t + self.h][ok]); ps.append(np.asarray(p, float)[ok])
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        self.n_calib_ = len(yy)
        grid = np.round(np.arange(0.85, 1.60, 0.005), 3)
        perd = [np.sum(self.cs * np.clip(yy - k * pp, 0, None) + self.co * np.clip(k * pp - yy, 0, None))
                for k in grid]
        self.factor_ = float(grid[int(np.argmin(perd))]) if len(yy) else 1.0
        return self

    def predict_p50(self, X):
        return np.asarray(self.m_.predict(X), float)

    def predict(self, X):
        return self.predict_p50(X) * self.factor_

    @property
    def coef_series(self) -> pd.Series:
        s = pd.Series({"factor_P75": self.factor_, "n_calib": self.n_calib_})
        if hasattr(self.m_, "coef_series"):
            s = pd.concat([self.m_.coef_series, s])
        return s


# ======================================================================================
# Escenario con promo del competidor
# ======================================================================================
@dataclass
class EscenarioCompetidor(EscenarioExAnte):
    """Escenario ex ante (D-03/D-05/D-07) + valor futuro de promo_competidor SIN usar el futuro:
    'cero'   -> 0 en todo el horizonte (supuesto conservador; compatible con D-02).
    'tasa'   -> frecuencia histórica de días con promo del competidor (compatible con D-02).
    'bloque' -> SC-08: si al origen el competidor está en promo y lleva r días, se marcan los
                (largo_bloque - r) días restantes (bloques observados siempre de 7 días).
                Requiere observar a diario al competidor; NO es la configuración oficial."""
    modo: str = "cero"
    largo_bloque: int = 7

    def aplicar(self, hist, fut):
        fut = super().aplicar(hist, fut)
        if COMP not in fut.columns:
            return fut
        h = hist[COMP].to_numpy(float)
        if self.modo == "cero":
            fut[COMP] = 0.0
        elif self.modo == "tasa":
            fut[COMP] = float(h.mean())
        elif self.modo == "bloque":
            r = 0
            for v in h[::-1]:
                if v != 1:
                    break
                r += 1
            rest = max(0, self.largo_bloque - r) if r > 0 else 0
            fut[COMP] = (np.arange(len(fut)) < rest).astype(float)
        else:
            raise ValueError(f"modo de competidor desconocido: {self.modo}")
        return fut


# ======================================================================================
# Especificaciones (registro común para el torneo y producción)
# ======================================================================================
@dataclass
class Spec:
    nombre: str
    tipo: str                      # naive | mm28 | uplift | ridge_qa | loglineal | quantreg | glm |
                                   # hibrido | gbm | rf | ensamble
    quiebre: str = "control"
    promo2: bool = False
    extras: list = field(default_factory=list)   # fin_mes, dia_quincena
    competidor: str | None = None                # None | cero | tasa | bloque
    familia: str = "poisson"
    metodo: str = "ols"                          # loglineal/hibrido: ols | ridge
    miembros: list = field(default_factory=list)  # Spec en dict (ensamble)
    complejidad: float = 0.0                       # nº aprox. de parámetros estimados
    oficial: bool = True                           # cumple D-01..D-07 sin insumos nuevos
    descripcion: str = ""

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Spec":
        return Spec(**d)


def features(s: Spec) -> list:
    if s.tipo == "ensamble":
        out = []
        for m in s.miembros:
            out += [c for c in features(Spec.from_dict(m)) if c not in out]
        return out
    return BASE + list(s.extras) + ([COMP] if s.competidor else [])


def escenario(s: Spec) -> EscenarioExAnte:
    modo = s.competidor
    if s.tipo == "ensamble":
        modos = {Spec.from_dict(m).competidor for m in s.miembros} - {None}
        if len(modos) > 1:
            raise ValueError("Ensamble con modos de competidor distintos")
        modo = modos.pop() if modos else None
    return EscenarioCompetidor(modo=modo) if modo else EscenarioExAnte()


def fabrica_p50(s: Spec):
    t = s.tipo
    if t == "naive":
        return NaiveEstacional
    if t == "mm28":
        return lambda: MediaMovil(28)
    if t == "uplift":
        return BaselineUplift
    if t == "ridge_qa":
        return RidgeLog
    if t == "loglineal":
        return lambda: LogLineal(s.quiebre, s.promo2, s.metodo)
    if t == "quantreg":
        return lambda: QuantLog(0.5, s.quiebre, s.promo2)
    if t == "glm":
        return lambda: GLMConteo(s.familia, s.promo2)
    if t == "hibrido":
        return lambda: Hibrido(s.quiebre, s.promo2, s.metodo)
    if t in ("gbm", "rf"):
        return lambda: ArbolDirecto(t)
    if t == "ensamble":
        mi = [(fabrica_p50(Spec.from_dict(m)), features(Spec.from_dict(m))) for m in s.miembros]
        return lambda: Ensamble(mi)
    raise ValueError(t)


def fabrica_p75(s: Spec, costo_sub: float = 3, costo_sobre: float = 1):
    if s.tipo == "quantreg":
        tau = costo_sub / (costo_sub + costo_sobre)
        return lambda: QuantLog(tau, s.quiebre, s.promo2)
    base, esc = fabrica_p50(s), escenario(s)
    return lambda: CalibradoCuantil(base, esc, costo_sub, costo_sobre)
