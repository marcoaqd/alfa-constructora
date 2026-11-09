# 01 · Análisis estadístico de variables para pronosticar `unidades_vendidas`

- **Datos:** `forecast/data/reto1_dataset.xlsx`, hoja `Datos_completos` (181 filas × 23 columnas) + `Diccionario`.
- **Partición:** Entrenamiento 2026-01-01 a 2026-04-30 (120 días) y Validación 2026-05-01 a 2026-06-30 (61 días). Todos los modelos se ajustaron **solo** con Entrenamiento, sin shuffle.
- **Script reproducible:** `forecast/src/01_analisis_variables.py`. Se ejecuta con `python3 -I forecast/src/01_analisis_variables.py [--data ruta.xlsx] [--out carpeta]`, usa semilla 42 y escribe los CSV de detalle en `forecast/reportes/01_salidas/`.

---

## 1. Resumen ejecutivo

1. La demanda depende sobre todo de **`promo_pct`** (+1 pp de descuento ≈ +2,8 % de unidades; con 30 % de descuento las ventas son ×2,3), seguida de **`quiebre_stock_pct`**, **`dia_semana`**, **`exhibicion`** (+14 %), **`distribucion_numerica`** (cambio de nivel en marzo) y **`promo_competidor`** (−9 %).
2. **`unidades_facturadas_mismo_dia` es fuga de información** (r = 0,9993). Si se incluye, el WAPE de validación baja artificialmente de 6,7 % a 1,4 %, es decir, las métricas se ven unas 5 veces mejores de lo real. Hay que descartarla.
3. `ventas_hace_7_dias` no aporta: el naive t-7 tiene WAPE 51,9 % y añadirla al modelo empeora el MAE. Además, solo sería válida con un horizonte de 7 días o menos.
4. El mejor modelo limpio es una **regresión log-lineal (Ridge/OLS sobre log y)** con 7 variables: WAPE 6,7 % y bias −0,7 % en validación (CV temporal 7,7 %). El mejor baseline tiene WAPE 28,2 %. Los árboles quedan peor (GB 9,4 %, RF 11,7 %) porque no extrapolan por encima del máximo de entrenamiento.
5. Medios/GRP, temperatura, inflación, confianza, región y año anterior **no aportan** una vez que se controla por la promo. Sus correlaciones marginales vienen de la confusión con las promos o de la tendencia temporal.

---

## 2. Tabla de variables

Notación: ρ = Spearman, IM = información mutua (Entrenamiento), Δ% = efecto condicional del OLS log(y) con errores HAC(7) en Entrenamiento, ΔMAE = cambio de MAE en validación al quitar (−) o añadir (+) la variable sobre el NÚCLEO (Ridge log y; MAE base 114,9).

| variable | relación con objetivo | evidencia | riesgo (leakage / colinealidad / drift) | veredicto |
|---|---|---|---|---|
| `promo_pct` | Muy fuerte, multiplicativa | ρ_tr 0,68, ρ_va 0,78; IM 0,54; Δ% = e^(2,77·promo): ×1,32 con 10 %, ×2,30 con 30 %; −promo → ΔMAE +381; permutación +494 | Colineal con `precio` (R² 0,99). Coeficiente estable (z = −0,14). Validación solo tiene niveles 0, 0,25 y 0,30 | **usar** (núcleo) |
| `precio` | Muy fuerte (espejo de la promo) | precio ≈ 6,51·(1−promo_pct), CV 1,1 %; ρ −0,62; sustituir promo→precio: ΔMAE −0,2 | VIF 278. Redundante con promo_pct | **transformar / alternativa**: usar precio **o** promo_pct, nunca ambas |
| `exhibicion` | Moderada | +13,8 % (p_HAC < 0,001); sin promo +11,7 %, d = 0,86, p_MW 0,001; −exhib → ΔMAE +22 (GB +38) | VIF 1,07; estable (z −0,34) | **usar** (núcleo) |
| `promo_competidor` | Moderada, **enmascarada** | Marginal ≈ 0 (ρ −0,10); sin promo propia −14 % (d −1,13, p < 0,001); condicional −9,5 %; −promo_comp → ΔMAE +15,5 | ¿Se conoce ex ante? Pronosticar con 0 sube el WAPE de 8,3 % a 9,5 %. Estable (z −0,06) | **usar si se conoce de antemano** (si no, dudosa) |
| `precio_competidor` | Igual que promo_competidor | R² ~ promo_comp 0,99; +precio_comp ΔMAE +2,0 | VIF 240; misma incertidumbre ex ante | descartar (redundante) |
| `indice_precio_relativo` | Fuerte marginal | Es exactamente precio/precio_competidor (dif. máx 0,000); añadirla: Ridge −1,3, GB +19,6 | Función determinística, VIF 242, PSI 0,62 | **descartar** |
| `quiebre_stock_pct` | Fuerte negativa (censura de la venta) | Con quiebre 710 u. vs 1.607 sin quiebre; condicional e^(−1,40·q): −50 % con q = 0,5; −quiebre → ΔMAE +36 (CV +255) | **No se conoce ex ante** (se mide el mismo día). Persistente: P(q_t \| q_{t−1}) = 0,43. Solo 14 días con quiebre (3 en validación) | **usar como control en entrenamiento y pronosticar con 0** (o con un escenario de quiebre) |
| `dia_semana` | Moderada | Sin promo ni quiebre: η² 0,17, Kruskal p 0,005; lun–mié −10 a −13 %, sábado/domingo como referencia; −dow → ΔMAE +39 | Calendario, ex ante. Estable (\|z\| ≤ 1,8) | **usar** (núcleo) |
| `distribucion_numerica` | Cambio de nivel | Escalón de 0,84 (ene–feb) a 0,91 (mar–jun); +17 % en ese salto; −distr → ΔMAE +38 (CV +84); permutación en validación ≈ 0 (casi constante, sd 0,004) | Confundida con el tiempo (r con mes 0,82). PSI 3,35. Su coeficiente se identifica con **un solo** escalón | **usar con cautela** (núcleo; vigilar extrapolación) |
| `dia_quincena` | Débil / inestable | Condicional +4 % en Entr. (p_HAC 0,001), pero −0,6 % en Val. (z −1,4); marginal p_MW 0,68; −quincena → ΔMAE −2,6 (mejora) | Calendario, determinística (regla día ∈ {1,14,15,16} o ≥ 29, 100 % de acierto) | **dudosa** (opcional) |
| `ventas_hace_7_dias` | Nula | ρ_tr −0,08 (p 0,40); naive t-7 WAPE 51,9 % (73,6 % en días con cambio de promo); +lag7 ΔMAE +1,4 (GB +6,7) | **Solo es válida con h ≤ 7**. Es exactamente y.shift(7). Drift +21 % | **descartar** |
| `ventas_mismo_dia_anio_anterior` | Nula / proxy del día de semana | ρ_tr 0,04, ρ_va 0,37; nivel −24 % vs 2026; baseline WAPE 37 %; condicional p 0,30; +año ant. ΔMAE +1,6 (CV +18,6) | Ex ante, válida | **descartar** (dudosa si llega más historia) |
| `inversion_medios_usd` | Nula condicional | Marginal ρ 0,23 por confusión; condicional p 0,15; corr con residuo 0,000; rezagos 1–7 y adstock 0,3–0,9: \|r\| ≤ 0,11; +medios ΔMAE 0,0 (CV +20,8) | Colineal con grp_tv (R² 0,98, VIF 56). Drift +29 % | **descartar** |
| `grp_tv` | Nula condicional | corr con residuo −0,003; +grp ΔMAE −0,6 (CV +14,5) | Colineal con medios | **descartar** |
| `temperatura_c` | Nula condicional | Marginal ρ 0,25 (estacional); condicional p 0,76; corr con residuo 0,000; GB +21 | Drift fuerte (KS 0,56, PSI 2,5; validación +2,5 °C) | **descartar** |
| `inflacion_mensual_pct` | Nula | 6 valores (1 por mes); +inflación ΔMAE +11,1 | Validación fuera de rango (0,37 y 0,43 no vistos; KS 1,0, PSI 4,4) | **descartar** |
| `confianza_consumidor` | Nula | 5 valores; +confianza ΔMAE +1,5 (CV +24) | KS 0,49, PSI 4,5 | **descartar** |
| `codigo_region` | Nula | ANOVA p 0,80, η² 0,006; condicional p ≥ 0,39; +región ΔMAE +0,8 | Rota día a día en una serie única (semántica dudosa) | **descartar** |
| `mes` | Proxy de tiendas/tendencia | η² 0,13 marginal; los meses 5 y 6 no existen en entrenamiento | No extrapolable | **descartar** |
| `unidades_facturadas_mismo_dia` | Casi idéntica al objetivo | r 0,9993; \|fact−venta\| medio 19,6 u. (1,3 %); copiarla da WAPE 1,25 % | **LEAKAGE**: es consecuencia de la venta del mismo día | **descartar** |

---

## 3. Hallazgos detallados

### 3.1 Calidad de datos
- Sin nulos, sin filas ni fechas duplicadas y serie diaria continua (0 saltos). `dia_semana` y `mes` son 100 % coherentes con `fecha`.
- Consistencias determinísticas verificadas:
  - `ventas_hace_7_dias` = y.shift(7) en el 100 % de las filas 8 a 181.
  - `indice_precio_relativo` = round(precio/precio_competidor, 3), con diferencia máxima 0.
  - `dia_quincena` sigue la regla día ∈ {1,14,15,16} o día ≥ 29 en el 100 % de las filas.
- Rangos válidos: las proporciones están en [0, 1] y no hay negativos.
- **Variables casi constantes o de baja resolución:**
  - `inflacion_mensual_pct` (6 valores) y `confianza_consumidor` (5 valores) son constantes dentro de cada mes. A efectos prácticos, cada una aporta 6 observaciones, no 181.
  - `distribucion_numerica` tiene CV 3,8 % y en la práctica dos regímenes (media 0,840 en ene–feb y 0,907–0,911 en mar–jun). La variación dentro de cada mes (sd ≈ 0,004) es ruido.
  - `quiebre_stock_pct` vale 0 el 92 % de los días (14 días con quiebre).
- Outliers: `unidades_vendidas` tiene 24 outliers altos por IQR y **todos son días de promo**. Son demanda legítima, no errores.

### 3.2 Objetivo
- Distribución general: media 1.537, mediana 1.306, sd 662, rango 549–4.122, asimetría 1,45. En log la asimetría baja a 0,39, por eso se modela log(y).
- Media por nivel de promo: 0 % → 1.217; 10 % → 1.560; 15 % → 1.625; 25 % → 2.613; 30 % → 2.969.
- Autocorrelación en Entrenamiento: lag1 0,70, lag7 −0,05, lag14 −0,10. La persistencia de corto plazo se debe a los bloques de promo de varios días, no a una estacionalidad semanal fuerte. Los residuos del modelo núcleo tienen ACF lag1 de 0,07, así que no queda dependencia que explotar con rezagos.
- La validación tiene una media 18 % mayor (1.714 vs 1.448) porque incluye más días con descuentos altos: 17 de 61 días con 25–30 %, frente a 16 de 120 en Entrenamiento.

### 3.3 Fuga de información
| variable | ¿existe al momento de pronosticar? | evidencia | decisión |
|---|---|---|---|
| `unidades_facturadas_mismo_dia` | **No**: se factura el mismo día como consecuencia de la venta | r 0,9993. Usarla tal cual como pronóstico da MAE 21, WAPE 1,25 % | Descartar |
| `ventas_hace_7_dias` | Solo si el horizonte es h ≤ 7 | Es y(t−7). Para h > 7 habría que usar el pronóstico recursivo. Aun con h ≤ 7 no aporta (ver ablación) | Descartar |
| `quiebre_stock_pct` | **No** (se mide ese día). Persiste algo: P(q_t \| q_{t−1}) = 0,43 frente a 0,05 | Solo 2 de 14 quiebres siguen a una promo propia. En los días con quiebre, la venta del día anterior era menor (1.340 vs 1.556): parece un problema de abastecimiento, no de demanda | Usar como control en el ajuste y fijar en 0 al pronosticar (demanda no restringida) |
| `precio_competidor` / `promo_competidor` | Dudoso: depende de que exista inteligencia comercial | Pronosticar con promo_comp = 0 sube el WAPE de 8,3 % a 9,5 % (Ridge) | Pregunta de negocio |
| `promo_pct`, `precio`, `exhibicion`, `inversion_medios`, `grp_tv` | Sí, si hay plan comercial y de medios | — | Válidas |
| calendario (`dia_semana`, `dia_quincena`) | Sí, son determinísticas | — | Válidas |

**Escenario con leakage vs. sin leakage (Validación):**

| escenario | mejor modelo | MAE | WAPE | Bias % |
|---|---|---|---|---|
| CON LEAKAGE (todo + facturado) | OLS (y) | 23,7 | **1,38 %** | −0,5 |
| CON LEAKAGE | GradBoosting | 54,3 | 3,17 % | −1,8 |
| Sin facturado, con t-7 y quiebre real (solo h ≤ 7) | Ridge log y | 128,3 | 7,49 % | +1,7 |
| **NÚCLEO, quiebre real (oráculo)** | Ridge log y | 114,9 | **6,71 %** | −0,7 |
| **NÚCLEO, quiebre = 0 al pronosticar (ex ante real)** | Ridge log y | 142,1 | **8,29 %** | +1,2 |
| NÚCLEO, quiebre = 0 y promo_comp = 0 | Ridge log y | 162,1 | 9,46 % | +2,9 |

Conclusiones:
- El leakage infla el desempeño unas **5 veces** (WAPE 1,4 % frente a 6,7–8,3 %).
- Con quiebre = 0, casi todo el error extra está en los 3 días con quiebre real: MAE 647 en esos días frente a 116 en el resto.
- Entrenar sin quiebre es peor que entrenar con él como control y luego fijarlo en 0. En validación: WAPE 8,8 % vs 8,3 %. En el CV temporal de entrenamiento la diferencia es mucho mayor: 23,6 % vs 7,7 % (este último con quiebre real), porque los quiebres de enero y febrero contaminan los coeficientes.

### 3.4 Redundancias y multicolinealidad
| par | r Pearson | R² | comentario |
|---|---|---|---|
| precio ~ promo_pct | −0,995 | 0,990 | precio ≈ 6,51·(1−promo) |
| precio_competidor ~ promo_competidor | −0,994 | 0,988 | la promo del competidor es una baja de precio de ~6,2 a ~5,1 |
| inversion_medios_usd ~ grp_tv | 0,990 | 0,980 | miden lo mismo |
| indice_precio_relativo | — | — | ratio exacto (determinístico) |
| distribucion_numerica ~ mes | 0,82 | — | escalón temporal |

- VIF con todas las numéricas: precio 278, índice 242, precio_comp 240, promo_pct 137, promo_comp 126, grp 56, medios 56, mes 25, distribución 25.
- VIF del NÚCLEO: máximo 1,78 (las dummies de día de semana); las demás variables ≤ 1,15.

### 3.5 Relación con el objetivo: correlación marginal vs. efecto condicional
- Varias correlaciones marginales engañan por **confusión con las promos**:
  - `promo_competidor` tiene una diferencia marginal de +0,5 % (p 0,67), pero −14 % (p < 0,001) en días sin promo propia.
  - `inversion_medios` y `medios_on` muestran +11,6 % marginal (p 0,03), que cae a +1,9 % (p 0,28) sin promo y a un coeficiente condicional ≈ 0 (p 0,15).
  - `dia_semana` no es significativo con ANOVA sobre todos los días (p 0,17), pero sí sin promo (η² 0,17, p 0,002).
- Efectos condicionales (OLS log y, HAC, R² 0,958): promo_pct 2,77/unidad (p < 0,001); quiebre −1,40 (p < 0,001); exhibición +13,8 % (p < 0,001); promo_comp −9,5 % (p < 0,001); distribución 2,29/unidad (p < 0,001); quincena +4,0 % (p 0,001); lun/mar/mié −10/−13/−11 %.
- Sin efecto condicional: medios (p 0,15), temperatura (p 0,76), año anterior (p 0,30) y regiones (p ≥ 0,39).
- Los efectos de medios se probaron también con rezagos 1, 2, 3 y 7 y con adstock de decay 0,3–0,9: correlación con el residuo |r| ≤ 0,11 en todos los casos.
- **Interacción promo × exhibición:** no aporta en el modelo log (ΔMAE +0,6), porque el log ya la modela como multiplicativa. En GB mejora 4 unidades de MAE.

### 3.6 Modelos en validación (61 días)

| modelo | conjunto | MAE | RMSE | WAPE % | Bias | Bias % | CV-MAE (Entr.) |
|---|---|---|---|---|---|---|---|
| Naive t-7 | baseline (solo h ≤ 7) | 889,1 | 1191,7 | 51,9 | +3 | +0,2 | — |
| Naive estacional fijo (última semana repetida) | baseline h = 1..61 | 1119,3 | 1360,6 | 65,3 | +437 | +25,5 | — |
| Año anterior (crudo) | baseline | 636,7 | 936,3 | 37,2 | −612 | −35,7 | — |
| Año anterior × 1,321 | baseline | 483,2 | 750,0 | 28,2 | −258 | −15,0 | — |
| Media de Entrenamiento | baseline | 505,8 | 788,4 | 29,5 | −266 | −15,5 | — |
| OLS (y) | NÚCLEO | 148,5 | 230,2 | 8,66 | −32 | −1,9 | 154,1 |
| **OLS / Ridge (log y)** | **NÚCLEO** | **114,9** | **148,5** | **6,71** | **−12** | **−0,7** | **122,4** |
| RandomForest | NÚCLEO | 200,1 | 316,1 | 11,67 | −36 | −2,1 | 209,9 |
| GradBoosting | NÚCLEO | 160,2 | 213,4 | 9,35 | +8 | +0,5 | 190,5 |
| Ridge (log y) | NÚCLEO sin quiebre | 151,0 | 214,0 | 8,81 | −11 | −0,7 | 377,2 |
| GradBoosting | NÚCLEO sin quiebre | 183,8 | 243,2 | 10,73 | +2 | +0,1 | 241,7 |
| Ridge (log y) | TODO sin facturado | 128,3 | 162,3 | 7,49 | +30 | +1,7 | 176,6 |
| GradBoosting | TODO sin facturado | 175,2 | 236,5 | 10,22 | −50 | −2,9 | 211,2 |
| Ridge (log y) | NÚCLEO, quiebre = 0 (ex ante) | 142,1 | 206,8 | 8,29 | +21 | +1,2 | — |
| GradBoosting | NÚCLEO, quiebre = 0 (ex ante) | 186,9 | 252,2 | 10,91 | +36 | +2,1 | — |

Cómo leer la tabla:
- NÚCLEO = promo_pct, exhibicion, promo_competidor, quiebre_stock_pct, dia_quincena, dia_semana y distribucion_numerica.
- La columna CV-MAE viene de un origen rodante dentro de Entrenamiento (cortes en los días 60, 75, 90 y 105, con horizonte de 15 días).
- Añadir las 11 variables restantes al NÚCLEO **empeora** el resultado: Ridge pasa de 6,7 % a 7,5 % de WAPE y el CV-MAE sube de 122 a 177. Es una señal de sobreajuste.
- Los árboles no extrapolan. El máximo de Entrenamiento es 3.521 y Validación llega a 4.122 (promo 30 % + exhibición). Con solo 120 filas, el modelo log-lineal generaliza mejor.
- Bias por mes del Ridge NÚCLEO: mayo −2,3 % y junio +1,1 %. No hay deriva sistemática.

### 3.7 Importancia por permutación (Validación, 30 repeticiones) y ablación
- **Permutación, Ridge log y NÚCLEO (ΔMAE):** promo_pct +494, quiebre +62, dia_semana +55, exhibición +41, promo_comp +27, distribución +0,8, quincena −0,7.
- **Permutación, GB NÚCLEO:** promo +493, exhibición +49, quiebre +39, dow +26, promo_comp +9.
- En validación, la distribución numérica es casi constante, así que la permutación no puede medir su efecto de nivel. La ablación sí lo mide: quitarla suma +38 de MAE en validación y +84 en CV.
- **Ablación Ridge (quitar, ΔMAE val / CV):** promo +381/+279, dow +39/+32, distribución +38/+84, quiebre +36/+255, exhibición +22/+50, promo_comp +16/+21, quincena **−2,6/+3,2**.
- **Ablación Ridge (añadir, ΔMAE val / CV):** t-7 +1,4/+3,7; año anterior +1,6/+18,6; medios 0,0/+20,8; grp −0,6/+14,5; temperatura +0,1/−1,0; inflación +11,1/−4,4; confianza +1,5/+24,3; región +0,8/+2,9; índice −1,3/+4,8; precio_comp +2,0/+10,0.
  - Ninguna variable añadida mejora a la vez la validación y el CV en más de 3 unidades.
- **Con leakage**, `unidades_facturadas_mismo_dia` acapara toda la importancia: permutarla suma +474 de MAE en Ridge y +633 en GB. promo_pct pasa a ≈ 0, porque el modelo deja de "aprender" la demanda.

### 3.8 Estabilidad (drift) Entrenamiento → Validación
| variable | media Entr → Val | KS (p) | PSI | lectura |
|---|---|---|---|---|
| unidades_vendidas | 1.448 → 1.714 (+18 %) | 0,29 (0,001) | 0,35 | explicado por la mezcla de promos (bias del modelo −0,7 %) |
| distribucion_numerica | 0,875 → 0,910 | 0,51 (< 0,001) | 3,35 | régimen alto en toda la validación |
| inflacion_mensual_pct | 0,30 → 0,40 | 1,00 (< 0,001) | 4,43 | valores fuera de rango |
| confianza_consumidor | 45,7 → 46,2 | 0,49 (< 0,001) | 4,51 | — |
| temperatura_c | 23,9 → 26,4 | 0,56 (< 0,001) | 2,48 | estacional |
| inversion_medios_usd | 1.034 → 1.334 (+29 %) | 0,23 (0,025) | 0,22 | — |
| indice_precio_relativo | 1,04 → 0,99 | 0,24 (0,017) | 0,62 | — |
| promo_pct / exhibición / quincena / quiebre | estables | p ≥ 0,32 | ≤ 0,72* | *el PSI de quiebre es alto solo por los pocos días con quiebre (11 vs 3) |

**La relación con el objetivo es estable.** Se ajustó un OLS log(y) por separado en cada periodo y se comparó cada coeficiente:

| variable | coef. Entrenamiento | coef. Validación | z de la diferencia |
|---|---|---|---|
| promo_pct | 2,79 | 2,77 | −0,14 |
| exhibicion | 0,101 | 0,089 | −0,34 |
| promo_competidor | −0,116 | −0,118 | −0,06 |
| quiebre_stock_pct | −1,53 | −1,30 | +1,6 |
| dia_semana (dummies) | — | — | \|z\| ≤ 1,8 |
| dia_quincena | +0,050 | −0,006 | −1,4 |
| intercepto | — | — | +2,9 |

- La única inestabilidad notable es la quincena, que en validación pierde su efecto.
- El intercepto sube (z +2,9) porque ese modelo no incluye la distribución. Esto confirma que el escalón de `distribucion_numerica` es un cambio real de nivel (≈ +13–17 %).

### 3.9 Supuestos
- Cada fila es la demanda diaria de un único cliente. `codigo_region` parece un atributo del registro que rota día a día, no una serie distinta.
- La promo propia, la exhibición y el calendario se conocen con anticipación porque forman parte del plan comercial.
- El dataset externo tendrá el mismo esquema y generación de datos. El modelo log-lineal es multiplicativo, así que extrapola de forma razonable a niveles de promo vistos. **No** está garantizado para descuentos > 30 % ni para distribución numérica fuera de [0,83; 0,92].
- Con 61 días de validación las diferencias de MAE < 5 unidades no son concluyentes. Por eso se reporta también el CV temporal.

---

## 4. Lista final recomendada de variables

**Núcleo (usar):**
1. `promo_pct`, o `precio` en su lugar (nunca ambas). Modelar sobre log(y).
2. `exhibicion`
3. `dia_semana` (6 dummies)
4. `distribucion_numerica`: efecto de nivel. Vigilar su extrapolación.
5. `quiebre_stock_pct`: **como control en el entrenamiento**. Al pronosticar se fija en 0 (demanda no restringida) o en un escenario de quiebre informado por supply.
6. `promo_competidor`: condicionada a que se conozca de antemano. Si no se conoce, se fija en 0 o en su tasa histórica, y el WAPE pasa de 8,3 % a 9,5 %.

**Opcionales (probar en el dataset externo, sin prioridad):**
- `dia_quincena`: +4 % en Entrenamiento pero 0 en Validación; la ablación sale neutra.
- `promo_pct × exhibicion`: solo útil para árboles (−4 MAE en GB).
- `ventas_mismo_dia_anio_anterior`: solo si llega un histórico más largo que permita estimar el nivel interanual.

**Descartadas:**
- `unidades_facturadas_mismo_dia` (leakage).
- `ventas_hace_7_dias`: solo válida con h ≤ 7 y no aporta.
- `indice_precio_relativo` (ratio determinístico).
- `precio_competidor` (redundante con promo_competidor).
- `inversion_medios_usd` y `grp_tv`: sin efecto condicional, ni contemporáneo, ni rezagado, ni con adstock.
- `temperatura_c`, `inflacion_mensual_pct`, `confianza_consumidor` y `mes`: sin efecto y con drift fuerte o valores fuera de rango.
- `codigo_region`.

**Modelo recomendado como punto de partida:**
- Ridge/OLS sobre log(y) con el NÚCLEO y corrección de retransformación (smearing).
- Validación: WAPE 6,7 % con quiebre real y 8,3 % ex ante con quiebre = 0; bias en torno a ±1 %.
- Los árboles se pueden usar como contraste, no como modelo principal.

---

## 5. Preguntas abiertas para el orquestador (decisión de negocio)

1. **Horizonte y momento del pronóstico:** ¿con cuántos días de anticipación se pronostica (1, 7, 30, 61 días)? Esto decide la validez de cualquier rezago de ventas. Hoy ninguno aporta, pero condiciona el diseño de la evaluación externa.
2. **¿Se conocen de antemano la promo y el precio del competidor?** (inteligencia comercial, acuerdos con el cliente). Si no, ¿se pronostica con un escenario "sin promo del competidor"? Coste estimado: +1,2 pp de WAPE.
3. **Quiebre de stock:** ¿el objetivo es pronosticar la **demanda** (no restringida, quiebre = 0) o la **venta** realizada? Si es la venta, ¿supply puede dar un pronóstico de quiebre? Los 3 días con quiebre en validación concentran gran parte del error ex ante (MAE 647 frente a 116).
4. **Facturación del mismo día:** confirmar que `unidades_facturadas_mismo_dia` no estará disponible al pronosticar. Si existiera un dato anticipado real (p. ej. pedidos en firme del cliente), sería otra variable distinta y legítima.
5. **Distribución numérica:** ¿qué pasó en marzo (de 84 % a 91 % de tiendas)? ¿Hay un plan de distribución futuro? El modelo extrapola linealmente a partir de un solo escalón observado.
6. **Promociones futuras:** ¿habrá descuentos > 30 % o nuevas mecánicas? Están fuera del rango observado y el efecto multiplicativo no está validado allí.
7. **Medios/TV:** el dato no muestra efecto sobre las ventas diarias, ni inmediato ni rezagado. ¿Negocio espera un efecto (de marca, de largo plazo) que justifique conservarlas o medirlas de otra forma?
8. **`codigo_region`:** ¿qué significa en una serie de un solo cliente? Si el dataset externo trae varias regiones por día, habrá que replantear el análisis (panel en lugar de serie).
9. **Métrica oficial:** ¿WAPE, MAE o bias? ¿Penaliza más el sobre-pronóstico (inventario) o el sub-pronóstico (quiebre)? Esto define si conviene sesgar el pronóstico.
