#!/usr/bin/env python3
"""Evalúa modelos finalistas en may–jun (ventas reales del dataset original) y llena la hoja
Pronostico del archivo de validación externo con el modelo elegido.
Entrena SOLO con Entrenamiento (ene–abr). Uso: python3 -I forecast/src/06_validacion_externa.py"""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, openpyxl
R = Path(__file__).resolve().parents[2] / "forecast"
d = pd.read_excel(R/"data/reto1_dataset.xlsx", "Datos_completos")
v = pd.read_excel(R/"data/reto1_validacion_equipos.xlsx", "Datos_validacion")
tr = d[d.periodo == "Entrenamiento"].reset_index(drop=True)
real = d[d.periodo != "Entrenamiento"].unidades_vendidas.values
DOW = ["martes","miércoles","jueves","viernes","sábado","domingo"]

def X(df, comp=False, sq=False, q=None):
    o = pd.DataFrame({"c":1.0,"promo":df.promo_pct.clip(0,.3),"exh":df.exhibicion,
                      "dist":df.distribucion_numerica.clip(.83,.92)})
    if sq: o["promo2"] = o.promo**2
    for k in DOW: o[k] = (df.dia_semana==k).astype(float)
    if comp: o["comp"] = df.promo_competidor
    if q is not None: o["q"] = q
    return o.astype(float)

def fit(name, comp=False, sq=False, descens=True, qctl=False):
    y = tr.unidades_vendidas/(1-tr.quiebre_stock_pct) if descens else tr.unidades_vendidas
    m = sm.OLS(np.log(y), X(tr,comp,sq, tr.quiebre_stock_pct if qctl else None)).fit()
    smear = np.mean(np.exp(m.resid))
    return lambda df, comp_fut=None, q_fut=None: np.exp(m.predict(X(df.assign(promo_competidor=comp_fut if comp_fut is not None else 0) if comp else df, comp, sq, (q_fut if q_fut is not None else 0*df.promo_pct) if qctl else None)))*smear

def met(p):
    e = real-p; return dict(WAPE=abs(e).sum()/real.sum()*100, bias=e.sum()/-real.sum()*100,
        costo31=(3*np.clip(e,0,None)+np.clip(-e,0,None)).sum()/real.sum()*100)
M = {"L1 des-censurado (ganador torneo)": fit("L1"),
     "L2 + promo² ": fit("L2", sq=True, descens=False, qctl=True),
     "L5 des-censurado + promo²": fit("L5", sq=True)}
rows=[]
for k,f in M.items():
    p=f(v); rows.append({"modelo":k, **met(p)})
    rows.append({"modelo":k+" ×1,065 (P75)", **met(p*1.065)})
# referencia: si se aceptara usar variables grises (solo informativo)
f=M["L1 des-censurado (ganador torneo)"]; rows.append({"modelo":"L1 × (1−quiebre real) [usa var. gris]", **met(f(v)*(1-v.quiebre_stock_pct))})
t=pd.DataFrame(rows).round(2); print(t.to_string(index=False))
t.to_csv(R/"reportes/06_validacion_externa.csv", index=False)
p50 = M["L1 des-censurado (ganador torneo)"](v)
out = R/"reportes/Reto1_validacion_pronostico_L1.xlsx"
wb = openpyxl.load_workbook(R/"data/reto1_validacion_equipos.xlsx"); ws = wb["Pronostico"]
ws["B2"] = "promo_pct, exhibicion, dia_semana, distribucion_numerica"; ws["B3"]="no"; ws["B4"]=4
r0 = next(r for r in range(1,ws.max_row+1) if ws.cell(r,1).value=="fecha")+1
for i,val in enumerate(p50): ws.cell(r0+i,3).value = int(round(val))
wb.save(out); print("guardado", out, "filas", len(p50))
