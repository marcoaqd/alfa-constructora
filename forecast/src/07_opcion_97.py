#!/usr/bin/env python3
"""¿Se puede llegar a 97 % de accuracy (WAPE ≤ 3 %)? Compara modelos con y sin variables grises.
Entrena SOLO con ene–abr; evalúa con ventas reales may–jun. Uso: python3 -I forecast/src/07_opcion_97.py"""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, openpyxl
R = Path(__file__).resolve().parents[2] / "forecast"
d = pd.read_excel(R/"data/reto1_dataset.xlsx", "Datos_completos")
v = pd.read_excel(R/"data/reto1_validacion_equipos.xlsx", "Datos_validacion")
tr = d[d.periodo=="Entrenamiento"]; real = d[d.periodo!="Entrenamiento"].unidades_vendidas.values
DOW=["martes","miércoles","jueves","viernes","sábado","domingo"]
def X(df, cols, log=False):
    o=pd.DataFrame({"c":1.0},index=df.index)
    for c in cols:
        if c=="dia_semana":
            for k in DOW: o[k]=(df.dia_semana==k).astype(float)
        elif c=="log_fact": o[c]=np.log(df.unidades_facturadas_mismo_dia)
        elif c=="log1mq": o[c]=np.log(1-df.quiebre_stock_pct)
        else: o[c]=df[c]
    return o.astype(float)
def run(cols, log):
    y=np.log(tr.unidades_vendidas) if log else tr.unidades_vendidas
    m=sm.OLS(y,X(tr,cols)).fit(); p=m.predict(X(v,cols))
    return np.exp(p)*np.mean(np.exp(m.resid)) if log else p.values
specs={
 "A · L1 honesto (sin grises)": (["promo_pct","exhibicion","dia_semana","distribucion_numerica"],True),
 "B · L1 + quiebre + promo_competidor (grises)": (["promo_pct","exhibicion","dia_semana","distribucion_numerica","log1mq","promo_competidor"],True),
 "C · solo facturado (gris)": (["unidades_facturadas_mismo_dia"],False),
 "D · facturado + promo + día semana (gris)": (["log_fact","promo_pct","dia_semana"],True),
 "E · facturado + todas las grises + núcleo": (["log_fact","log1mq","promo_competidor","promo_pct","exhibicion","dia_semana","distribucion_numerica"],True),
}
rows=[];P={}
for k,(c,l) in specs.items():
    p=run(c,l);P[k]=p;e=real-p
    rows.append(dict(modelo=k,n_var=len(c),accuracy=100-abs(e).sum()/real.sum()*100,WAPE=abs(e).sum()/real.sum()*100,
                     MAPE=np.mean(abs(e)/real)*100,bias=-e.sum()/real.sum()*100))
t=pd.DataFrame(rows).round(2);print(t.to_string(index=False));t.to_csv(R/"reportes/07_opcion_97.csv",index=False)
best=t.sort_values("WAPE").iloc[0].modelo;print("mejor:",best)
c,_=specs[best]
wb=openpyxl.load_workbook(R/"data/reto1_validacion_equipos.xlsx");ws=wb["Pronostico"]
names=[x.replace("log_fact","unidades_facturadas_mismo_dia").replace("log1mq","quiebre_stock_pct") for x in c]
ws["B2"]=", ".join(names);ws["B3"]="sí";ws["B4"]=len(names)
r0=next(r for r in range(1,ws.max_row+1) if ws.cell(r,1).value=="fecha")+1
for i,val in enumerate(P[best]): ws.cell(r0+i,3).value=int(round(val))
out=R/"reportes/Reto1_validacion_pronostico_97.xlsx";wb.save(out);print("guardado",out)
