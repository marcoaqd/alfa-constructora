# 03 · Revisión QA: leakage, validación y generalización

- **Revisa:** `reportes/01_analisis_estadistico.md`, `src/01_analisis_variables.py`, `reportes/01_salidas/*.csv` y `DECISIONES.md` (D-01 a D-07).
- **Harness nuevo:** `forecast/src/qa_backtest.py`. Se ejecuta con `python3 -I forecast/src/qa_backtest.py` (≈ 9 s) y se importa desde otros scripts.
- **Entorno de ejecución:** python3 -I; scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, statsmodels 0.15.0, scipy 1.18.1. Semilla 42. Dos corridas idénticas producen predicciones idénticas.
- **Nomenclatura:**
  - *NÚCLEO-EST* = núcleo del estadístico: promo_pct, exhibicion, promo_competidor, quiebre_stock_pct, dia_quincena, dia_semana, distribucion_numerica.
  - *NÚCLEO-EXA* = NÚCLEO-EST sin promo_competidor, para cumplir D-02.
  - *Escenario ex ante* = al pronosticar, quiebre = 0 (D-03), distribución = último valor observado (D-05) y promo recortada a [0; 0,30] (D-07).

---

## 1. Veredicto: **APROBADO CON CAMBIOS**

El enfoque (log-lineal con pocas variables planificadas) se sostiene en un backtesting rodante honesto y supera con margen amplio y significativo a los baselines. El 6,7 % de WAPE no es alcanzable con las decisiones del usuario: la cifra honesta es **≈ 9–10 % de WAPE sobre la venta observada** y **≈ 7,5–7,8 % sobre los días sin quiebre**, que es la mejor medida de la demanda.

La selección de variables se hizo en parte mirando el periodo de validación, así que may–jun ya no es un hold-out limpio. La única prueba limpia será el dataset externo.

### Cifras verificadas (re-ejecutadas)

| escenario | WAPE % | Bias % | WAPE sin días con quiebre % | Pérdida asimétrica (2:1) % |
|---|---|---|---|---|
| Estadístico, oráculo (quiebre y promo_comp reales), split fijo 120→61 | **6,71** (reproducido) | −0,7 | 6,58 | 10,42 |
| quiebre = 0 | 8,29 (reproducido) | +1,2 | 6,58 | 11,82 |
| quiebre = 0, promo_comp = 0 sin reentrenar | 9,46 (reproducido) | +2,9 | 7,77 | 12,73 |
| quiebre = 0, NÚCLEO-EXA reentrenado | 9,09 | +0,1 | 7,50 | 13,56 |
| **Honesto, split fijo: NÚCLEO-EXA + escenario ex ante completo** | **9,20** | −0,4 | **7,59** | 13,98 |
| **Honesto, rodante h = 7 sobre may–jun (8 re-entrenamientos)** | **9,40** | +1,3 | **7,73** | 13,44 |
| Backtest de selección (12 folds, 11-mar → 2-jun, 84 días) | **10,05** | +0,9 | **7,75** | 14,61 |
| Hold-out final (4 folds, 3 → 30-jun, 28 días) | **8,13** | +0,4 | **7,08** | 12,01 |
| Backtest de los 181 días (16 folds, 112 días) | 9,57 | — | — | — |
| *Baselines en selección:* naive t-7 / media móvil 28 d | 50,95 / 34,04 | −3,1 / −4,9 | 48,6 / 32,3 | 78,0 / 53,5 |

Cómo se reparten las 2,5 pp que separan el 6,7 % del 9,2 %:
- **≈ 1,6 pp por el quiebre**: los 3 días con quiebre en validación, que el modelo de demanda no debe "acertar".
- **≈ 0,8 pp por la ausencia de promo_competidor.**
- **≈ 0,1 pp por usar la distribución como último valor**, que aquí no cuesta casi nada porque la distribución es estable desde marzo.

En el backtest, el oráculo de quiebre y distribución da 7,86 % frente a 10,05 % ex ante; sin los días con quiebre la diferencia desaparece (7,69 % frente a 7,75 %).

---

## 2. Tabla de hallazgos

| # | severidad | hallazgo | evidencia | acción recomendada |
|---|---|---|---|---|
| 1 | **Mayor** | El 6,7 % de WAPE es una cifra con oráculo, incompatible con D-02, D-03 y D-05 | Re-ejecución: 6,71 %. Con las decisiones del usuario: 9,20 % (split fijo), 9,40 % (rodante h = 7), 10,05 % (backtest de selección). | Reemplazar el 6,7 % en `DECISIONES.md` y en cualquier comunicación. Reportar ~9–10 % sobre la venta y ~7,5–7,8 % en días sin quiebre. |
| 2 | **Mayor** | *Selection leakage*: el periodo de validación se usó para elegir variables y modelo | En `01_analisis_variables.py`, la ablación (`dMAE_val`), la permutación y la estabilidad de coeficientes (OLS ajustado **sobre Validación**) usan may–jun. "Quincena dudosa" se decidió por su coeficiente en Validación. Se probaron ~28 configuraciones contra los mismos 61 días. Una selección *forward* hecha **solo con los 120 días de entrenamiento** (backtest h = 7) elige {precio, distribución, quiebre, dow, quincena}, **no** elige exhibición, y en may–jun rinde 8,83 % frente a 7,85 % del núcleo. | Congelar la especificación con este reporte. Tratar jun 3–30 como hold-out semi-limpio (evaluar una sola vez) y el dataset externo como la única prueba limpia. No volver a elegir variables mirando may–jun. Atenuante: las variables del núcleo también son significativas solo con entrenamiento (OLS HAC, p < 0,001), así que el sesgo afecta más a la cifra que a la lista. |
| 3 | **Mayor** | `promo_competidor` sigue en el núcleo pese a D-02 | Fijarla en 0 sin reentrenar sesga al alza (+2,9 %), porque el modelo descuenta −9,5 % cuando hay promo del competidor. Reentrenar sin ella deja el bias en +0,1 % (WAPE 9,09 % frente a 9,46 %). | Usar NÚCLEO-EXA. El harness **rechaza** `promo_competidor`, `precio_competidor` e `indice_precio_relativo` salvo `permitir_no_ex_ante=True` (solo para diagnóstico). |
| 4 | **Mayor** | D-04 (quedarse corto es peor) no está implementado: el modelo está centrado (bias ≈ 0) y sub-pronostica los picos | El máximo pronosticado en validación es 3.481 frente a 4.122 real, porque el modelo no supera el máximo de entrenamiento (3.521). Bias −3,4 % con promo 30 % y −4,3 % con exhibición (paso = 1, sin quiebre). El 55 % de los días queda sub-pronosticado. Un factor estimado en selección y aplicado al hold-out baja la pérdida asimétrica: 2:1 → factor 1,05, de 12,0 a 11,5; 3:1 → 1,07, de 15,9 a 13,4; 5:1 → 1,10, de 23,7 a 14,8 (sube el WAPE de 8,1 a 9,4 / 10,4 / 11,9). | El usuario debe fijar la **razón de costos** sub:sobre. Estimar el factor (o un modelo cuantílico) **solo** en el periodo de selección (`factor_asimetrico`) y reportar WAPE y pérdida asimétrica por separado. |
| 5 | **Mayor** | No está definido cómo se usará el dataset externo | Si trae historia con `unidades_vendidas`, se puede re-entrenar cada 7 días (protocolo h = 7). Si hay que pronosticarlo entero desde el modelo de 181 días, el horizonte deja de ser 7 días, la distribución queda congelada en el valor del 30-jun y no hay corrección de nivel. Hoy el error no crece con h = 1..7 (WAPE 7,7–8,1 %, paso = 1, sin quiebre) porque el modelo no usa rezagos, pero un cambio de distribución o de precio de lista no se detectaría. | Decisión del usuario (ver §4). El pipeline debe correr `validar_entrada` antes de predecir en ambos casos. |
| 6 | **Mayor** | El efecto de `distribucion_numerica` se identifica con un solo escalón (0,84 → 0,91 el 2-mar) y su elasticidad implícita es ≈ 2 | Coeficiente entre folds 1,89–2,56 (CV 8,5 %). Quitarla suma +3,78 pp de WAPE (IC95 [+1,4; +6,2], significativo). Con distribución 0,95 el pronóstico sube +10 %; con 0,80 baja −22 %. Validación tiene PSI 3,35 frente a entrenamiento. | Mantenerla, pero **recortarla al rango de entrenamiento [0,83; 0,92] y emitir alerta** fuera de él. Pedir al negocio el plan de distribución (D-05). |
| 7 | Menor | Árboles y conjuntos amplios sobreajustan | GBM: WAPE de entrenamiento 3,7 % frente a 12,4 % en backtest (brecha 8,7 pp). TODO ex ante (13 grupos): 6,3 % frente a 12,3 %, peor que el núcleo por +2,29 pp (IC [+0,06; +4,86], significativo). Núcleo: 6,5 % frente a 10,1 % (7,75 % sin quiebre). | Descartar árboles como modelo principal (coincide con R-03). |
| 8 | Menor | La ventana deslizante es peor que la expansiva | Deslizante de 56 días: 13,85 % frente a 10,05 % con ventana expansiva. | Ventana expansiva. |
| 9 | Menor | Las dummies de `features()` del estadístico tratan categorías nuevas como la referencia sin avisar | `(X.dia_semana == d)` con "Miercoles" o "MIÉRCOLES" da domingo. Una región R5 se convierte en R1. | Corregido en `RidgeLog`: normaliza tildes y mayúsculas y lanza error ante valores desconocidos. Probado: mayúsculas sin tilde → predicción idéntica. |
| 10 | Menor | Con paso = 7 el análisis por horizonte y por día de semana está confundido (todos los orígenes caen en martes) | En la tabla por segmentos, h = 1 coincide con miércoles, etc. Los "errores de lunes y martes" (14–20 %) eran en realidad los quiebres del 16–17 mar y del 4–5 may. | Para diagnóstico por segmento usar `paso=1` (84 orígenes). Sin quiebre: martes 12,4 %, sábado 10,1 %, lunes 3,3 %, resto 6–9 %. |
| 11 | Menor | `dia_quincena` no aporta | Quitarla cambia el WAPE en −0,22 pp (IC [−0,80; +0,31]). | Descartar por parsimonia. |
| 12 | Menor | Hay 6 dummies de día de semana con coeficientes poco estables en jue/vie/sáb | CV entre folds de 30–40 %. Agrupar en `lun_mie` (1 dummy) cambia +0,32 pp (IC [−0,04; +0,64], no significativo). | Opcional: usar `lun_mie` y ahorrar 5 parámetros. Mantener 6 dummies es aceptable. |
| 13 | Menor | Los outliers son demanda legítima | Excluir los 13 outliers IQR de entrenamiento sube el WAPE de 9,2 % a 11,2 % y baja el coeficiente de promo de 2,70 a 2,30. | No winsorizar ni recortar y. |
| 14 | Menor | El baseline "naive estacional fijo" (65 %) del estadístico no corresponde a D-01 | Con h = 7 los baselines justos son t-7 rodante (51 %) y media móvil de 28 d (34 %). | Usar estos baselines; ya están en el harness. |
| 15 | Menor | Reproducibilidad: las versiones no están fijadas | No hay `requirements.txt`. El alpha de RidgeCV salta entre 0,001 y 1 entre folds, pero los coeficientes son estables y OLS ≈ Ridge (Δ 0,01 pp). | Añadir `forecast/requirements.txt` con las versiones de arriba. |
| 16 | Menor | La validación no contiene promo 10 % ni 15 %; el primer fold aún no vio promo 25 % | Niveles de validación: 0, 25 y 30 %. Bias con promo 10 %: +6 % (35 días). | El log-lineal interpola razonablemente. Vigilar los niveles no vistos con `validar_entrada`. |
| — | OK | Transformación log, escalado y encoding **no** usan validación | `StandardScaler` y `RidgeCV(TimeSeriesSplit)` van dentro del pipeline ajustado solo con entrenamiento. El smearing de Duan se calcula en entrenamiento. Las dummies son fijas, el log no tiene parámetros y los cuantiles del PSI salen de entrenamiento. | Sin acción. |
| — | OK | El leakage de `unidades_facturadas_mismo_dia` está bien detectado | r = 0,9993. Ahora el harness lanza error si se incluye. | Sin acción. |

---

## 3. Protocolo de validación definitivo

**Datos:** 181 días consecutivos. Todos los folds tienen 7 días y se alinean desde el final de la serie (`origenes()`).

| elemento | definición |
|---|---|
| Esquema | Origen rodante con **ventana expansiva**: en cada origen t se re-entrena con [0, t), se pronostica t+1..t+7 y se avanza 7 días. Sin shuffle. |
| Ventana mínima | 63 días. En la práctica el primer origen es el 10-mar (69 días). A esa altura ya hay 9 días del nuevo régimen de distribución, los niveles de promo 0/10/15/30 %, 21 días con exhibición y 9 con quiebre. |
| Selección de modelo | Filas 1–153 (hasta 2-jun): **12 folds, 84 días** (11-mar → 2-jun). Toda comparación de modelos y variables se hace aquí. |
| Hold-out final | 3 → 30-jun: **4 folds, 28 días**, con re-entrenamiento en cada origen (`primer_origen=153`). Se evalúa **una sola vez**, con la especificación congelada. Es semi-limpio, por el hallazgo 2. |
| Prueba limpia | El dataset externo. |
| Escenario ex ante | quiebre = 0, distribución = último valor en el origen, promo recortada a [0; 0,30], sin variables del competidor. `ORACULO` solo para diagnóstico. |
| Métricas | **WAPE** (agregado sobre todos los días), **WAPE sin días con quiebre** (métrica principal de demanda, D-03), MAE, RMSE, **Bias %** (negativo = corto), **pérdida asimétrica** = Σ(c_sub·faltante + c_sobre·sobrante)/Σy, **Sub %** (unidades faltantes / demanda) y % de días sub-pronosticados. También la media y la sd del WAPE por fold, y la brecha WAPE de entrenamiento frente a backtest (sobreajuste). |
| Segmentos | Con/sin promo y por nivel de promo, exhibición, con/sin quiebre real y mes. Día de semana y horizonte **con `paso=1`**. |
| Baselines obligatorios | Naive t-7 rodante y media móvil de 28 días. |
| Regla de decisión | `diferencia_pareada` (bootstrap de folds, IC95). Una variable o modelo más complejo solo entra si el IC95 excluye 0 a su favor en selección **y** no empeora el hold-out. Si no, gana el más simple. |

---

## 4. Riesgos de generalización y validaciones previas a predecir

`validar_entrada(nuevo, features, referencia=entrenamiento)` está implementada y probada con un dataset sintético "roto" (columna faltante, fecha duplicada, región R7, promo 40 %, distribución 0,96, quiebre en escala 0–100, día de semana incoherente). Detectó todos los problemas.

| riesgo | evidencia actual | validación | nivel |
|---|---|---|---|
| Columna requerida ausente o tipo no numérico o nulos | — | Presencia de fecha, dia_semana, features (y el objetivo si se evalúa); coerción numérica | ERROR |
| Fechas duplicadas (¿panel por región?) o no parseables | 1 fila por día | Duplicados → ERROR; saltos o desorden → ALERTA | ERROR / ALERTA |
| `dia_semana` irreconocible o incoherente con la fecha | 100 % coherente | Se normalizan tildes y mayúsculas; se contrasta con `fecha.dayofweek` | ERROR |
| Escala de porcentajes 0–100 en lugar de 0–1 | Proporciones en [0, 1] | promo, distribución y quiebre en [0, 1]; binarias en {0, 1}; y > 0 (por el log) | ERROR |
| Descuento > 30 % o nivel no visto | Niveles 0/10/15/25/30 % | Fuera de rango → ALERTA, y el pronóstico se recorta a 30 % (D-07). Con 40 % el modelo daría ×1,32 frente a 30 %, sin validar | ALERTA |
| Drift o extrapolación de `distribucion_numerica` | PSI 3,35 entre periodos; rango [0,831; 0,923] | Rango de entrenamiento + PSI > 0,25 → ALERTA. Recomiendo recortar al rango | ALERTA |
| Cambio de precio de lista o de la definición de promo | precio ≈ 6,50·(1 − promo), CV 1,1 % | Precio implícito de lista con desviación > 5 % → ALERTA | ALERTA |
| Inflación y temperatura | Inflación fuera de rango en validación (KS 1,0); temperatura +2,5 °C | No son entradas del modelo, así que no afectan. Si se incorporaran, la inflación y la confianza del mes no se conocen 7 días antes | — |
| Región nueva | R1–R4 rotan día a día | Región no vista → ALERTA (no se usa). Fechas duplicadas → ERROR por cambio de granularidad | ALERTA |
| Integridad de `ventas_hace_7_dias` | Igual a y(t−7) al 100 % | Coincidencia < 99 % → ALERTA (indica filas faltantes o desordenadas) | ALERTA |
| `dia_quincena` incoherente con el calendario | Regla al 100 % | ALERTA | ALERTA |
| Variable de leakage entre las features | — | ERROR (también en `backtest`) | ERROR |
| Outliers de y | Los 24 outliers son días de promo | No filtrar; revisar solo los y ≤ 0 o > 3× el máximo histórico | — |

El PSI se omite cuando un solo valor domina ≥ 80 % de la referencia (p. ej. quiebre = 0), porque daba falsas alarmas.

---

## 5. Sobreajuste: cuántos parámetros son razonables

- Regla práctica: 10–15 observaciones por parámetro libre en OLS.
  - Con la ventana mínima (63–69 días) eso permite ~5–7 parámetros.
  - Con 153–181 días permite ~10–15.
  - Ridge amplía un poco ese margen, pero aquí el alpha óptimo es casi 0, así que no está regularizando de forma apreciable.
- **NÚCLEO-EXA sin quincena = 10 coeficientes + intercepto.** Está en el límite superior razonable. Agrupar el día de semana (`lun_mie`) lo reduce a 5 sin pérdida significativa.
- Los coeficientes del núcleo son estables entre los 12 folds: CV de promo 2,3 %, quiebre 2,1 %, exhibición 8,1 % y distribución 8,5 %. Los únicos inestables son los de jue/vie/sáb.
- **Demasiado complejo para estos datos:**
  - GBM y RF: brecha de 8,7 pp entre entrenamiento y backtest.
  - El conjunto de 13 grupos.
  - Cualquier interacción o efecto por región o por mes.
  - Variables mensuales (inflación, confianza): solo 4–6 valores distintos, es decir, 4–6 observaciones efectivas.
  - Prophet con *changepoints* o SARIMAX con muchas exógenas: absorberían el escalón de distribución como tendencia y la extrapolarían.

---

## 6. Opinión variable por variable (backtest de selección, NÚCLEO-EXA = 10,05 %)

| variable | ΔWAPE en backtest (IC95) | QA | estadístico | ¿discrepo? |
|---|---|---|---|---|
| `promo_pct` | promo + dow solo: +7,30 [+3,2; +11,3] frente al núcleo | **usar** | usar | No. Preferirla a `precio` (10,54 %), porque es invariante a cambios del precio de lista |
| `exhibicion` | quitarla: +1,17 [−0,32; +2,54], no significativo | **usar** | usar | Parcial: la evidencia fuera de muestra es débil y la selección hecha solo con entrenamiento no la elige. Se mantiene por la inferencia HAC en entrenamiento (p < 0,001), su coeficiente estable y el bias −4 % en días con exhibición. Confirmar que se conoce 7 días antes |
| `dia_semana` | promo + dow + distr + quiebre frente al núcleo: +1,46 [−0,03; +2,77] | **usar** (6 dummies o `lun_mie`) | usar | No |
| `distribucion_numerica` | quitarla: +3,78 [+1,4; +6,2] | **usar con recorte y alerta** | usar con cautela | Coincido; QA añade el recorte al rango |
| `quiebre_stock_pct` | sin control: 14,57 % (+4,5 pp) | **usar como control, 0 al pronosticar** | igual | No. QA añade que la métrica de demanda debe excluir los días con quiebre |
| `promo_competidor` | — | **descartar** (D-02) | "usar si se conoce" | **Sí**: el usuario ya respondió que no se conoce. Cuesta ~0,8 pp |
| `dia_quincena` | quitarla: −0,22 [−0,80; +0,31] | **descartar** | dudosa | Leve: QA la saca por parsimonia |
| `ventas_hace_7_dias` | +0,56 [−0,45; +1,81] | descartar | descartar | No (es válida con h = 7, pero no aporta) |
| `ventas_mismo_dia_anio_anterior` | +0,04 | descartar | descartar | No |
| `inversion_medios_usd` / `grp_tv` | +0,17 / +0,14 | descartar | descartar | No |
| `temperatura_c` | +0,08 | descartar | descartar | Añado que no se conoce exactamente 7 días antes |
| `inflacion_mensual_pct` / `confianza_consumidor` | +0,39 / +0,81 | descartar | descartar | **Añado**: tampoco son ex ante (se publican después del mes); usarlas sería leakage temporal |
| `precio` | sustituyendo a promo: +0,49 | alternativa, no usar | alternativa | No |
| `precio_competidor`, `indice_precio_relativo` | — | descartar (D-02) | descartar | No; el índice depende del precio del competidor |
| `codigo_region`, `mes` | — | descartar | descartar | No |
| `unidades_facturadas_mismo_dia` | — | **prohibida** | descartar | No |

**Especificación recomendada para congelar:** RidgeLog (o OLS log) con `promo_pct`, `exhibicion`, `dia_semana`, `distribucion_numerica` (recortada) y `quiebre_stock_pct` (control, 0 al pronosticar). Ventana expansiva, re-entrenamiento semanal y factor asimétrico según D-04.

En backtest da 9,83 % de WAPE y 7,57 % sin quiebre. Se evaluó **una sola vez** en el hold-out, ya congelada: **WAPE 7,90 %, 6,82 % sin quiebre, bias +0,6 %, pérdida asimétrica (2:1) 11,57 %**.

---

## 7. Harness `forecast/src/qa_backtest.py`

| API | qué hace |
|---|---|
| `backtest(df, modelo, features, min_train=63, horizonte=7, paso=7, ventana_max=None, primer_origen=None, escenario=None, costo_sub=2, costo_sobre=1)` | Devuelve `ResultadoBacktest` con `.folds`, `.predicciones`, `.agregado`, `.por_segmento` y `.coeficientes` (estabilidad entre folds). `modelo` puede ser una clase/fábrica o una instancia con fit/predict. |
| `EscenarioExAnte(fijar, ultimo_valor, recortar)` / `ORACULO` | Transforma las filas futuras según D-03/D-05/D-07 / usa los valores reales (solo diagnóstico). |
| `RidgeLog`, `NaiveEstacional`, `MediaMovil`, `Ajuste(base, factor)` | Modelo del estadístico con encoding robusto, baselines y sesgo deliberado. |
| `comparar`, `diferencia_pareada`, `factor_asimetrico` | Tabla comparativa, IC95 por bootstrap de folds y factor que minimiza la pérdida asimétrica. |
| `validar_entrada`, `hallazgos_df` | Validación de esquema, rangos y drift antes de predecir. |
| Constantes | `NUCLEO_EST`, `NUCLEO_EX_ANTE`, `PROHIBIDAS`, `NO_CONOCIDAS_EX_ANTE`. |

Uso:
```python
import sys; sys.path.insert(0, "forecast/src")
from qa_backtest import backtest, RidgeLog, cargar_excel
res = backtest(cargar_excel("forecast/data/reto1_dataset.xlsx").iloc[:153], RidgeLog,
               ["promo_pct", "exhibicion", "dia_semana", "distribucion_numerica", "quiebre_stock_pct"])
```

---

## 8. Checklist de generalización

| prueba | resultado |
|---|---|
| Re-ejecución de las cifras del estadístico (6,71 / 8,29 / 9,46) | Reproducidas exactamente |
| Leakage de variables (facturado, competidor, mensuales) | Facturado: OK. Competidor: corregido. Inflación y confianza no son ex ante (no se usan) |
| Leakage en el preprocesamiento (log, escalado, encoding, alpha, smearing) | OK, todo se ajusta con entrenamiento |
| Leakage de selección | **Presente** (hallazgo 2); cuantificado en ~1 pp en may–jun |
| Backtest rodante h = 7 frente a baselines | Núcleo 10,05 % frente a 34,0 % (MA28) y 51,0 % (t-7). Δ −24 pp, IC [−30,6; −17,1] |
| Hold-out jun (una evaluación) | NÚCLEO-EXA: 8,13 % (7,08 % sin quiebre). Especificación recomendada: 7,90 % (6,82 % sin quiebre), bias +0,6 % |
| Brecha entrenamiento/backtest | Núcleo 6,5 → 10,1 (7,75 sin quiebre); GBM 3,7 → 12,4; TODO 6,3 → 12,3 |
| Estabilidad de coeficientes (12 folds) | Estables, salvo las dummies de jue/vie/sáb |
| Error por horizonte 1..7 (paso = 1, sin quiebre) | Plano: 7,7–8,1 % |
| Por día de semana (paso = 1, sin quiebre) | Lunes 3,3 %, martes 12,4 %, sábado 10,1 %, resto 6–9 % |
| Con/sin promo (sin quiebre) | 8,75 % / 7,07 %; bias −1,3 % en ambos |
| Con/sin quiebre real | Con quiebre, WAPE 125 % (4 días; esperado en un modelo de demanda); sin quiebre, 7,75 % |
| Sensibilidad a outliers | Quitarlos empeora (9,2 → 11,2 %): son señal |
| Extrapolación (promo 40–50 %, distribución 0,80/0,95) | Respuesta monótona y multiplicativa (×1,32 / ×1,72 / ×0,78 / ×1,10), sin validar; queda cubierta por la alerta y el recorte |
| Categorías nuevas o formato de texto | Región nueva → ALERTA. "MIERCOLES" → OK. Valor desconocido → ERROR |
| Esquema roto (dataset sintético) | Detectados los 7 tipos de error inyectados |
| Reproducibilidad | Determinista con semilla 42; faltan las versiones fijadas |

---

## 9. Decisiones para el usuario (vía orquestador)

1. **Razón de costos sub:sobre (D-04).** Con 2:1 el factor es ≈ 1,05, con 3:1 ≈ 1,07 y con 5:1 ≈ 1,10. Cada opción sube el WAPE pero reduce el faltante: en el hold-out, Sub % pasa de 3,9 a 2,0 / 1,5 / 0,7.
2. **Uso del dataset externo.** ¿Traerá `unidades_vendidas` para re-entrenar semanalmente (h = 7), o hay que pronosticarlo entero con el modelo de 181 días? ¿Es el mismo cliente y un periodo posterior?
3. **Métrica oficial de demanda.** ¿Se acepta medir el WAPE excluyendo los días con quiebre, ya que esos días la venta está censurada?
4. **Plan de `distribucion_numerica`.** ¿Puede salir de [0,83; 0,92]? Si sí, el modelo extrapola con una elasticidad ≈ 2 estimada con un solo escalón.
5. **¿`exhibicion` se conoce 7 días antes?** Su aporte fuera de muestra es modesto (+1,2 pp, no significativo).
6. **D-07 sigue sin respuesta.** Mientras tanto se aplica el recorte a 30 % con alerta.
