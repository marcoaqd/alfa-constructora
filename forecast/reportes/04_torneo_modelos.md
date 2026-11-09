# 04 · Torneo de modelos de demanda

- **Autor:** orquestador. En esta sesión no había herramienta de delegación, así que lo implementó el orquestador directamente. **Queda pendiente** que QA revise `04_torneo_modelos.py`, `modelos_torneo.py` y `05_pronosticar.py`.
- **Scripts:**
  - `forecast/src/04_torneo_modelos.py`: torneo completo (≈ 105 s). Es determinista: dos corridas dan CSV idénticos.
  - `forecast/src/modelos_torneo.py`: modelos, escenarios y especificaciones. Lo usan 04 y 05.
  - `forecast/src/05_pronosticar.py`: pipeline de producción.
  - Ejecución: `python3 -I forecast/src/04_torneo_modelos.py`.
- **Salidas:** `forecast/reportes/04_salidas/`
  - `ranking_seleccion.csv`, `pruebas_pareadas.csv`, `decisiones_especificacion.csv`
  - `holdout.csv`, `pruebas_pareadas_holdout.csv`, `extrapolacion.csv`
  - `sensibilidad_costos.csv`, `factores_p75_por_fold.csv`
  - `predicciones_*.csv`, `ganador.json`, `tablas_torneo.md`

## 1. Resumen

1. **Ganador: L1, regresión OLS sobre log(demanda des-censurada y/(1−q))** con `promo_pct`, `exhibicion`, `dia_semana` (6 dummies) y `distribucion_numerica`. Tiene 9 coeficientes más el intercepto.
   - **P75** = P50 × factor. El factor se calibra con los errores fuera de muestra de las 8 semanas anteriores al origen. Su mediana es 1,05 en selección y 1,065 en el modelo final.
2. **Selección** (12 folds, 84 días, de los cuales 80 sin quiebre):
   - Pérdida 3:1 del P75 = **12,97 %**, con cobertura del 73,8 % y 2,16 % de faltante.
   - P50: WAPE **9,69 %** sobre la venta observada y **7,40 %** en días sin quiebre; bias +0,8 %.
3. **Hold-out** (3–30 jun, evaluado una sola vez): pérdida 3:1 del P75 = **11,98 %**, cobertura 77,8 %, faltante 1,04 % (479 u.). WAPE del P50 = **7,96 %** sobre la venta observada y **6,86 %** sin quiebre; bias +1,0 %.
4. **No hay ningún modelo significativamente mejor** que L1. El mejor por cifra puntual, L2 (con promo²), tiene −1,44 pp con IC95 [−3,61; +0,53], y en el hold-out es peor. Se aplica la regla de QA: gana el más simple.
5. **Valor agregado frente al método del planeador** (baseline × uplift): −15,6 pp de pérdida 3:1, IC95 [−26,4; −7,0], y −6,5 pp de WAPE sin quiebre. Las dos diferencias son significativas.

## 2. Protocolo (idéntico para todos)

- **Harness:** `qa_backtest.backtest`.
  - Origen rodante con ventana expansiva, horizonte 7 y paso 7; `min_train` 63.
  - Mismos folds que QA: la selección son las filas 1–153 y el hold-out las filas 154–181. El hold-out solo se usó para los finalistas.
- **Escenario ex ante:**
  - quiebre = 0 (D-03)
  - distribución = último valor (D-05)
  - promo recortada a 30 % (D-07)
  - sin promo del competidor futura (D-02). Las variantes C reemplazan siempre el valor futuro sin mirarlo.
- **Ajuste dentro de cada fold:** todo encoding, escalado, factor P75 y alpha se ajusta solo con los datos de entrenamiento del fold. El P75 por factor usa un backtest **interno** dentro de la ventana de entrenamiento.
- **Métrica principal (D-04, supuesto 3:1):** Σ(3·faltante + 1·sobrante)/Σy del P75, sobre los días **sin quiebre** (= 4 × pinball 0,75).
- **Métricas secundarias:** la misma pérdida en todos los días; 2:1; WAPE, MAE y bias del P50 sobre la venta observada y sin quiebre; WAPE con promo y sin promo; cobertura del P75; unidades faltantes.
- **Pruebas:** bootstrap pareado de folds (5.000 réplicas, IC95), igual que `diferencia_pareada`, pero restringido a los días sin quiebre.
- **Regla de decisión (QA), aplicada de forma secuencial:**
  1. Etapa A: cada cambio de especificación entra solo si el IC95 lo favorece. La excepción es la des-censura, que **quita** un parámetro: entra salvo que sea significativamente peor.
  2. Etapas B–D (otras familias, competidor, ensambles): reemplazan a la especificación lineal elegida solo si son significativamente mejores en selección y además no empeoran el hold-out.

## 3. Ranking en selección (12 folds; P75 3:1 en días sin quiebre)

| # | modelo | parám. | **3:1 P75** | 3:1 todos | 2:1 | cobert. P75 | falt. % | WAPE P50 obs. | WAPE P50 sin q. | bias P50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | L2 OLS log + promo² (q control) | 13 | **11,54** | 13,98 | 9,82 | 76,3 | 1,72 | 9,08 | 6,77 | +1,5 |
| 2 | E2 Ensamble L2 + L5 | 26 | 11,64 | 14,09 | 9,93 | 76,3 | 1,71 | 9,03 | 6,72 | +1,6 |
| 3 | L5 des-censurado + promo² | 12 | 11,69 | 14,14 | 9,99 | 75,0 | 1,70 | 9,02 | 6,70 | +1,7 |
| 4 | E3 Ensamble top-3 | 38 | 11,72 | 14,21 | 10,09 | 76,3 | 1,63 | 9,06 | 6,72 | +2,0 |
| 5 | C2 L1 + competidor control, futuro 0 | 12 | 12,43 | 14,89 | 10,56 | 77,5 | 1,87 | 9,53 | 7,15 | +2,9 |
| 6 | C3 L1 + competidor control, futuro tasa | 12 | 12,50 | 14,96 | 10,67 | 77,5 | 1,82 | 9,61 | 7,36 | +0,3 |
| 7 | M5a GLM Poisson, offset log(1−q) | 11 | 12,65 | 15,09 | 10,73 | 72,5 | 1,92 | 9,68 | 7,40 | +0,9 |
| 8 | C4 L1 + competidor bloque en curso (SC-08) | 12 | 12,83 | 15,39 | 10,94 | 80,0 | 1,89 | 9,26 | 6,88 | +1,8 |
| 9 | M5b GLM Binomial Negativa | 12 | 12,92 | 15,34 | 10,85 | 73,8 | 2,08 | 9,69 | 7,40 | +0,7 |
| **10** | **L1 OLS log des-censurado (GANADOR)** | **11** | **12,97** | 15,37 | 10,81 | 73,8 | 2,16 | 9,69 | 7,40 | +0,8 |
| 11 | L3 + fin_mes | 13 | 13,44 | 15,84 | 11,23 | 71,3 | 2,21 | 9,89 | 7,58 | +1,2 |
| 12 | M7 Híbrido L1 + GBM residuos | 111 | 13,48 | 15,83 | 10,98 | 73,8 | 2,50 | 9,62 | 7,32 | +0,6 |
| 13 | L4 + dia_quincena | 13 | 13,48 | 15,91 | 11,36 | 72,5 | 2,12 | 10,07 | 7,74 | +1,1 |
| 14 | L0 Ridge log núcleo QA (q control) | 12 | 13,59 | 15,92 | 11,28 | 70,0 | 2,31 | 9,83 | 7,57 | +0,4 |
| 15 | L0o OLS log núcleo (q control) | 12 | 13,80 | 16,14 | 11,45 | 68,8 | 2,35 | 9,85 | 7,57 | +0,5 |
| 16 | M8a GBM directo (log y) | ~300 | 13,83 | 16,27 | 11,72 | 75,0 | 2,10 | 10,72 | 8,53 | −1,4 |
| 17 | M6 Cuantílica log (τ 0,5 y 0,75 directos) | 20 | 14,47 | 16,78 | 11,45 | 65,0 | 3,01 | 9,70 | 7,47 | +0,6 |
| 18 | L1r Ridge log des-censurado | 11 | 16,16 | 18,19 | 13,78 | 76,3 | 2,38 | 11,08 | 9,34 | −2,6 |
| 19 | M8b RandomForest directo | ~400 | 22,73 | 25,15 | 19,82 | 78,8 | 2,90 | 13,20 | 11,20 | −3,3 |
| 20 | B3 Baseline × uplift de promo | 6 | 28,57 | 31,08 | 24,18 | 81,3 | 4,39 | 16,19 | 13,87 | +2,1 |
| 21 | B2 Media móvil 28 días | 2 | 73,39 | 75,15 | 60,83 | 71,3 | 12,56 | 34,04 | 32,30 | −4,9 |
| 22 | B1 Naive t-7 | 1 | 104,21 | 106,91 | 84,60 | 63,8 | 19,62 | 50,95 | 48,59 | −3,0 |

Notas de lectura:
- **WAPE con promo / sin promo de L1:** 7,79 % y 7,02 % (días sin quiebre).
- **Brecha de sobreajuste del GBM:** WAPE de entrenamiento 4,4 % frente a 10,7 % en backtest.
- **WAPE de entrenamiento de los des-censurados:** se mide contra la venta censurada, así que no es comparable.

## 4. Decisiones de especificación (etapa A)

| tema | prueba | Δ 3:1 P75 (pp) | IC95 | Δ WAPE sin q. | decisión |
|---|---|---|---|---|---|
| (b) quiebre | L1 des-censurado − L0o control (ambos OLS) | **−0,83** | **[−1,62; −0,07]** | −0,18 | **Des-censurar.** Es significativo y tiene un parámetro menos |
| (b) idem frente a la spec QA | L1 − L0 (Ridge QA) | −0,62 | [−1,21; 0,00] | −0,17 | Coherente, en el límite |
| (d) promo² | L5 − L1 | −1,28 | [−3,30; +0,62] | −0,70 | No entra (no significativo; empeora el hold-out, §6) |
| (c) fin_mes | L3 − L0o | −0,36 | [−1,37; +0,62] | +0,01 | No entra |
| (c) dia_quincena | L4 − L0o | −0,32 | [−1,25; +0,56] | +0,17 | No entra |
| (c) fin_mes frente a quincena | L3 − L4 | −0,05 | [−0,80; +0,53] | — | Indistinguibles |
| método | L1r (RidgeCV) frente a L1 (OLS) | +3,19 | — | +1,94 | **OLS.** El alpha de RidgeCV (TimeSeriesSplit interno) salta a 10–178 en varios folds y encoge los coeficientes (promo 0,74 en el fold 0). Con el objetivo del núcleo QA, Ridge ≈ OLS |

## 5. Ganador y por qué

- **Retadores de las etapas B–D frente a L1 (3:1 P75).** Ninguno es significativamente mejor.

  | retador | Δ (pp) | IC95 |
  |---|---|---|
  | E2 | −1,33 | [−3,40; +0,59] |
  | E3 | −1,25 | [−3,19; +0,50] |
  | C2 | −0,54 | [−1,33; +0,26] |
  | C3 | −0,47 | [−1,41; +0,46] |
  | GLM Poisson | −0,32 | [−1,30; +0,64] |
  | Binomial Negativa | −0,05 | [−0,25; +0,15] |
  | Híbrido | +0,51 | [−0,85; +1,91] |
  | GBM | +0,86 | [−1,27; +3,08] |
  | Cuantílica | +1,50 | [−0,49; +3,89] |
  | RF | **+9,76** | **[+5,53; +14,86]** (peor) |

- **Mejor por cifra puntual frente a L1:** L2 tiene −1,44 pp, IC95 [−3,61; +0,53]; en WAPE sin quiebre, −0,62 [−1,74; +0,36]. **Mejor frente a segundo:** L2 − E2 = −0,11 [−0,38; +0,11].
- **Por qué no gana promo².** Es la mejor cifra puntual en selección, porque corrige el sesgo en días con 30 % (L1 −5,3 %, L2 +1,0 %). Pero:
  - no es significativa;
  - en el hold-out empeora: L2 − L0 da WAPE sin quiebre **+0,51 pp, IC95 [+0,35; +0,77]**, que sí es significativo;
  - la curvatura se estima con solo 8 bloques de promo.

  Queda **en vigilancia** para el dataset externo.
- **Transparencia sobre la regla.** La regla mecánica "el más simple entre los empatados con el mejor" (16 modelos empatados) elige el **GLM Poisson**: tiene los mismos 11 parámetros que L1 y gana el desempate por 0,32 pp. Pero cambiar de familia no tiene evidencia (IC95 [−1,30; +0,64]) y el hold-out lo invierte (GLM − L1: +0,58 pp, IC95 [−0,09; +1,09]). El GLM queda como **alternativa equivalente** documentada.
- **Coeficientes del modelo final** (181 días; `05_pronosticar.py`, hoja `coeficientes`):

  | variable | efecto |
  |---|---|
  | promo | +30,8 % por cada +10 pp (×2,24 con 30 %) |
  | exhibición | +14,1 % |
  | distribución | +28,2 % por cada +0,10 |
  | lunes / martes / miércoles | −12,3 / −14,6 / −11,5 % |
  | jueves / viernes / sábado | −6,6 / −2,7 / +2,6 % (frente al domingo) |
  | factor P75 | 1,065 |

## 6. Hold-out final (3–30 jun, 28 días, 27 sin quiebre; una sola evaluación)

| modelo | 3:1 P75 | 3:1 todos | cobert. | falt. % (u.) | WAPE P50 obs. | WAPE sin q. | bias P50 |
|---|---|---|---|---|---|---|---|
| **L1 ganador** | **11,98** | 13,29 | 77,8 | 1,04 (479) | **7,96** | **6,86** | +1,0 |
| L0 Ridge núcleo QA | 11,81 | 13,12 | 77,8 | 0,84 (387) | 7,90 | 6,82 | +0,6 |
| C4 competidor bloque (SC-08) | 12,20 | 13,49 | 77,8 | 0,87 (401) | 8,47 | 7,36 | **+3,3** |
| E2 Ensamble | 12,39 | 13,67 | 70,4 | 1,58 | 8,52 | 7,37 | +1,3 |
| L2 promo² | 12,41 | 13,68 | 70,4 | 1,66 | 8,47 | 7,33 | +1,1 |
| M5a GLM Poisson | 12,56 | 13,85 | 74,1 | 1,21 | 8,00 | 6,91 | +1,3 |
| B3 Baseline × uplift | 16,98 | 17,82 | 70,4 | 2,05 | 11,62 | 11,06 | −1,8 |

Lectura:
- **L1 frente a L0:** +0,17 pp, IC95 [−0,33; +0,75], no significativo. L1 tiene un parámetro menos, así que el veto de QA (más complejo y peor en hold-out) no aplica.
- **B3 frente a L1:** +5,00 pp, IC95 [+1,22; +8,71], significativo.
- **Comparación con lo que vio QA:** QA reportó 7,90 % de WAPE para L0 en este mismo hold-out, y aquí se reproduce.

## 7. Árboles y extrapolación

- En selección hay **7 días sin quiebre con venta por encima del máximo de entrenamiento**. En esos días el cociente P50/real es:

  | modelo | P50/real |
  |---|---|
  | L1 | 0,92 |
  | L2 | 0,96 |
  | GBM | 0,90 |
  | RF | **0,80** |

- **Sesgo del P50 en días con 30 % de promo:** L1 −5,3 %, GBM **−10,5 %**, RF **−17,8 %**.
- **Máximo pronosticado frente al máximo de entrenamiento:** L1 ×1,10, GBM ×1,04, RF ×0,94.
- **Conclusión:**
  - El RF no extrapola y es significativamente peor.
  - El GBM sobre log se acerca en promedio, pero duplica el sesgo en los picos y tiene una brecha entrenamiento/backtest de 6,3 pp.
  - El híbrido no aporta (+0,51 pp).

## 8. Sensibilidad a la razón de costos (L1, selección)

| razón sub:sobre | cuantil | factor | cobertura | faltante % | sobrante % | pérdida a su propia razón |
|---|---|---|---|---|---|---|
| 1:1 (P50) | 0,50 | 1,005 | 50,0 | 4,22 | 3,50 | 7,72 (3:1: 16,17) |
| 2:1 | 0,67 | 1,048 | 67,5 | 2,63 | 5,58 | 10,84 |
| **3:1** | **0,75** | **1,053** | **73,8** | **2,16** | **6,48** | **12,97** |
| 4:1 | 0,80 | 1,063 | 75,0 | 1,93 | 7,22 | 14,95 |
| 5:1 | 0,83 | 1,075 | 75,0 | 1,66 | 7,83 | 16,11 |

- **Qué aporta el P75:** baja la pérdida 3:1 de 16,17 % (entregando solo el P50) a 12,97 %, es decir −20 %. A cambio, el sobrante pasa de 3,5 % a 6,5 %.
- **Estabilidad del factor:** se mantiene entre 1,035 y 1,095 entre folds (`factores_p75_por_fold.csv`).

## 9. Pipeline de producción (`05_pronosticar.py`)

- **Entrada:** `--data-nuevo` (Excel o CSV con el mismo esquema).
- **Modos:**
  - `--modo una-vez` (por defecto): entrena con los 181 días y pronostica todo el archivo.
  - `--modo rodante`: re-entrena cada 7 filas con los reales del archivo.
  - `--entrenar-hasta` corta la historia para hacer pruebas.
- **Pasos:**
  1. Corre `validar_entrada`. Un ERROR corta el proceso con código de salida 2.
  2. Aplica el escenario ex ante.
  3. Escribe `forecast/reportes/05_salidas/pronostico_<archivo>.xlsx` y `.csv`, con las hojas pronostico (fecha, P50, P75, alertas), hallazgos, métricas (si el archivo trae ventas), modelo y coeficientes.
- **Pruebas realizadas:**

  | prueba | resultado |
  |---|---|
  | Entrenar hasta 30-abr y pronosticar may–jun como archivo externo, **una vez** (h = 1..61) | WAPE P50 8,72 % observado / 7,01 % sin quiebre; 3:1 P75 12,51 %; cobertura 69 %; faltante 2,11 % |
  | Igual, **rodante** | WAPE 8,93 / 7,17 %; 3:1 12,58 %; cobertura 74 %; faltante **1,45 %** |
  | Plan sintético de 1–7 jul sin ventas, con promo del 35 % y región R9 | Recorta a 30 % y emite las alertas |
  | Archivo roto (promo en escala 0–100, falta `exhibicion`) | ERROR, código de salida 2 |
  | Dataset completo como archivo externo (in-sample) | Bias +9,6 %: D-05 aplica la distribución de jun (0,914) a ene–feb (0,84). Muestra el riesgo de D-05 si el periodo externo tiene otra distribución |

  Las salidas de las dos primeras pruebas están en `forecast/reportes/05_salidas/`.
- **Advertencia:** may–jun se usó en la selección de variables (hallazgo 2 de QA), así que estas cifras son algo optimistas. La única prueba limpia será el dataset externo.

## 10. Limitaciones

- **Potencia estadística limitada:** 12 folds de selección y 4 de hold-out. Las diferencias menores a ~1,3 pp de pérdida 3:1 no se pueden distinguir.
- **Sesgo en descuentos altos:** en días con 30 % de promo el sesgo del P50 es −5 %. El P75 lo compensa en parte, pero promo² es la mejora candidata si el dataset externo la confirma.
- **Distribución:** su efecto se identifica con un solo escalón.
