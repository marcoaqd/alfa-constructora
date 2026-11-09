## Ranking (selección, 12 folds, 84 días; P75 3:1 en días sin quiebre)

| modelo | etapa | complejidad | PA31_P75_% | PA31_P75_todos_% | PA21_P75_% | Cob_P75_% | Falt_P75_% | Falt_P75_u | WAPE_P50_% | WAPE_P50_sq_% | MAE_P50_sq | Bias_P50_% | WAPE_promo_sq_% | WAPE_nopromo_sq_% | WAPE_train_% | factor_P75_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L2 OLS log + promo² (q control) | A | 13 | 11.54 | 13.98 | 9.82 | 76.25 | 1.72 | 2395.45 | 9.08 | 6.77 | 117.99 | 1.47 | 6.67 | 6.88 | 6.50 | 1.06 |
| E2 Ensamble top-2 | D | 26 | 11.64 | 14.09 | 9.93 | 76.25 | 1.71 | 2385.63 | 9.03 | 6.72 | 116.98 | 1.58 | 6.55 | 6.87 | 8.16 | 1.06 |
| L5 OLS log des-censurado + promo² | A | 12 | 11.69 | 14.14 | 9.99 | 75.00 | 1.70 | 2374.62 | 9.02 | 6.70 | 116.71 | 1.69 | 6.50 | 6.90 | 10.36 | 1.06 |
| E3 Ensamble top-3 | D | 38 | 11.72 | 14.21 | 10.09 | 76.25 | 1.63 | 2270.88 | 9.06 | 6.72 | 117.10 | 2.03 | 6.67 | 6.78 | 8.58 | 1.06 |
| C2 LIN* + competidor control, futuro 0 | C | 12 | 12.43 | 14.89 | 10.56 | 77.50 | 1.87 | 2604.48 | 9.53 | 7.15 | 124.55 | 2.92 | 7.59 | 6.73 | 10.14 | 1.05 |
| C3 LIN* + competidor control, futuro tasa hist. | C | 12 | 12.50 | 14.96 | 10.67 | 77.50 | 1.82 | 2541.49 | 9.61 | 7.36 | 128.16 | 0.31 | 7.82 | 6.92 | 10.14 | 1.08 |
| M5a GLM Poisson offset log(1-q) | B | 11 | 12.65 | 15.09 | 10.73 | 72.50 | 1.92 | 2668.80 | 9.68 | 7.40 | 128.88 | 0.91 | 7.63 | 7.17 | 10.49 | 1.06 |
| C4 LIN* + competidor bloque en curso (SC-08) | C | 12 | 12.83 | 15.39 | 10.94 | 80.00 | 1.89 | 2631.70 | 9.26 | 6.88 | 119.78 | 1.80 | 7.10 | 6.67 | 10.14 | 1.06 |
| M5b GLM BinNeg offset log(1-q) | B | 12 | 12.92 | 15.34 | 10.85 | 73.75 | 2.08 | 2894.93 | 9.69 | 7.40 | 128.94 | 0.71 | 7.81 | 7.01 | 10.57 | 1.05 |
| L1 OLS log des-censurado y/(1-q) | A | 11 | 12.97 | 15.37 | 10.81 | 73.75 | 2.16 | 3014.57 | 9.69 | 7.40 | 128.87 | 0.83 | 7.79 | 7.02 | 10.57 | 1.05 |
| L3 OLS log + fin_mes (q control) | A | 13 | 13.44 | 15.84 | 11.23 | 71.25 | 2.21 | 3074.72 | 9.89 | 7.58 | 132.05 | 1.18 | 8.01 | 7.17 | 6.42 | 1.05 |
| M7 Híbrido LIN* + GBM residuos | B | 111 | 13.48 | 15.83 | 10.98 | 73.75 | 2.50 | 3479.17 | 9.62 | 7.32 | 127.44 | 0.63 | 7.94 | 6.72 | 9.45 | 1.05 |
| L4 OLS log + dia_quincena (q control) | A | 13 | 13.48 | 15.91 | 11.36 | 72.50 | 2.12 | 2959.53 | 10.07 | 7.74 | 134.82 | 1.07 | 8.29 | 7.21 | 6.49 | 1.06 |
| L0 Ridge log núcleo QA (q control) | A | 12 | 13.59 | 15.92 | 11.28 | 70.00 | 2.31 | 3220.45 | 9.83 | 7.57 | 131.84 | 0.38 | 8.07 | 7.09 | 6.63 | 1.06 |
| L0o OLS log núcleo (q control) | A | 12 | 13.80 | 16.14 | 11.45 | 68.75 | 2.35 | 3278.58 | 9.85 | 7.57 | 131.94 | 0.51 | 8.10 | 7.08 | 6.62 | 1.06 |
| M8a GBM directo (log y) | B | 300 | 13.83 | 16.27 | 11.72 | 75.00 | 2.10 | 2932.23 | 10.72 | 8.53 | 148.50 | -1.43 | 9.54 | 7.56 | 4.37 | 1.10 |
| M6 Cuantílica log (tau 0,5 / 0,75) | B | 20 | 14.47 | 16.78 | 11.45 | 65.00 | 3.01 | 4200.77 | 9.70 | 7.47 | 130.14 | 0.62 | 7.84 | 7.11 | 10.39 | nan |
| L1r Ridge log des-censurado | A | 11 | 16.16 | 18.19 | 13.78 | 76.25 | 2.38 | 3322.59 | 11.08 | 9.34 | 162.71 | -2.63 | 11.78 | 7.00 | 11.98 | 1.12 |
| M8b RandomForest directo | B | 400 | 22.73 | 25.15 | 19.82 | 78.75 | 2.90 | 4046.88 | 13.20 | 11.20 | 195.12 | -3.32 | 14.31 | 8.23 | 5.86 | 1.19 |
| B3 Baseline × uplift promo | A | 6 | 28.57 | 31.08 | 24.18 | 81.25 | 4.39 | 6110.54 | 16.19 | 13.87 | 241.63 | 2.09 | 19.04 | 8.93 | 20.21 | 1.11 |
| B2 Media móvil 28d | A | 2 | 73.39 | 75.15 | 60.83 | 71.25 | 12.56 | 17506.25 | 34.04 | 32.30 | 562.56 | -4.86 | 39.15 | 25.74 | 36.35 | 1.30 |
| B1 Naive t-7 | A | 1 | 104.21 | 106.91 | 84.60 | 63.75 | 19.62 | 27334.59 | 50.95 | 48.59 | 846.45 | -3.05 | 51.29 | 46.01 | 45.71 | 1.35 |

## Decisiones de especificación

| id | prueba | dif_PA31_pp | IC95 | dif_WAPE_sq_pp | decision | regla |
|---|---|---|---|---|---|---|
| (b) tratamiento del quiebre | L1 OLS log des-censurado y/(1-q) - L0o OLS log núcleo (q control) | -0.83 | [-1.62; -0.07] | -0.18 | des-censurar y/(1-q) | más simple (1 parámetro menos) salvo que sea significativamente peor |
| (d) promo² (convexidad) | L5 OLS log des-censurado + promo² - L1 OLS log des-censurado y/(1-q) | -1.28 | [-3.30; +0.62] | -0.70 | no incluir | entra solo si el IC95 excluye 0 a su favor |
| (c) fin_mes | L3 OLS log + fin_mes (q control) - L0o OLS log núcleo (q control) | -0.36 | [-1.37; +0.62] | 0.01 | no incluir | entra solo si el IC95 excluye 0 a su favor |
| (c) dia_quincena | L4 OLS log + dia_quincena (q control) - L0o OLS log núcleo (q control) | -0.32 | [-1.25; +0.56] | 0.17 | no incluir | entra solo si el IC95 excluye 0 a su favor |

## Pruebas pareadas (selección)

| comparacion | contexto | metrica | dif_pp | IC95_lo | IC95_hi | significativa |
|---|---|---|---|---|---|---|
| L1 OLS log des-censurado y/(1-q) - L0o OLS log núcleo (q control) | (b) quiebre | PA31_P75 | -0.83 | -1.62 | -0.07 | True |
| L1 OLS log des-censurado y/(1-q) - L0o OLS log núcleo (q control) | (b) quiebre | WAPE_P50_sq | -0.18 | -0.42 | 0.06 | False |
| L5 OLS log des-censurado + promo² - L1 OLS log des-censurado y/(1-q) | (d) promo² | PA31_P75 | -1.28 | -3.30 | 0.62 | False |
| L5 OLS log des-censurado + promo² - L1 OLS log des-censurado y/(1-q) | (d) promo² | WAPE_P50_sq | -0.70 | -1.77 | 0.32 | False |
| L3 OLS log + fin_mes (q control) - L0o OLS log núcleo (q control) | (c) calendario | PA31_P75 | -0.36 | -1.37 | 0.62 | False |
| L3 OLS log + fin_mes (q control) - L0o OLS log núcleo (q control) | (c) calendario | WAPE_P50_sq | 0.01 | -0.58 | 0.51 | False |
| L4 OLS log + dia_quincena (q control) - L0o OLS log núcleo (q control) | (c) calendario | PA31_P75 | -0.32 | -1.25 | 0.56 | False |
| L4 OLS log + dia_quincena (q control) - L0o OLS log núcleo (q control) | (c) calendario | WAPE_P50_sq | 0.17 | -0.30 | 0.63 | False |
| L3 OLS log + fin_mes (q control) - L4 OLS log + dia_quincena (q control) | (c) fin_mes vs quincena | PA31_P75 | -0.05 | -0.80 | 0.53 | False |
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | LIN* vs núcleo QA | PA31_P75 | -0.62 | -1.21 | 0.00 | False |
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | LIN* vs núcleo QA | WAPE_P50_sq | -0.17 | -0.41 | 0.07 | False |
| E2 Ensamble top-2 - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | -1.33 | -3.40 | 0.59 | False |
| E3 Ensamble top-3 - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | -1.25 | -3.19 | 0.50 | False |
| C2 LIN* + competidor control, futuro 0 - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | -0.54 | -1.33 | 0.26 | False |
| C3 LIN* + competidor control, futuro tasa hist. - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | -0.47 | -1.41 | 0.46 | False |
| M5a GLM Poisson offset log(1-q) - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | -0.32 | -1.30 | 0.64 | False |
| M5b GLM BinNeg offset log(1-q) - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | -0.05 | -0.25 | 0.15 | False |
| M7 Híbrido LIN* + GBM residuos - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | 0.51 | -0.85 | 1.91 | False |
| M8a GBM directo (log y) - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | 0.86 | -1.27 | 3.08 | False |
| M6 Cuantílica log (tau 0,5 / 0,75) - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | 1.50 | -0.49 | 3.89 | False |
| M8b RandomForest directo - L1 OLS log des-censurado y/(1-q) | retador vs LIN* | PA31_P75 | 9.76 | 5.53 | 14.86 | True |
| L2 OLS log + promo² (q control) - E2 Ensamble top-2 | mejor vs segundo | PA31_P75 | -0.11 | -0.38 | 0.11 | False |
| L2 OLS log + promo² (q control) - E2 Ensamble top-2 | mejor vs segundo | WAPE_P50_sq | 0.06 | -0.05 | 0.17 | False |
| L2 OLS log + promo² (q control) - L1 OLS log des-censurado y/(1-q) | mejor por métrica vs ganador | PA31_P75 | -1.44 | -3.61 | 0.53 | False |
| L2 OLS log + promo² (q control) - L1 OLS log des-censurado y/(1-q) | mejor por métrica vs ganador | WAPE_P50_sq | -0.62 | -1.74 | 0.36 | False |
| L2 OLS log + promo² (q control) - L0 Ridge log núcleo QA (q control) | mejor vs núcleo QA | PA31_P75 | -2.05 | -4.32 | 0.22 | False |
| L2 OLS log + promo² (q control) - L0 Ridge log núcleo QA (q control) | mejor vs núcleo QA | WAPE_P50_sq | -0.80 | -2.08 | 0.38 | False |
| L2 OLS log + promo² (q control) - L0 Ridge log núcleo QA (q control) | mejor vs núcleo QA | WAPE_P50 | -0.75 | -1.86 | 0.35 | False |
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | ganador vs núcleo QA | PA31_P75 | -0.62 | -1.21 | 0.00 | False |
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | ganador vs núcleo QA | WAPE_P50_sq | -0.17 | -0.41 | 0.07 | False |
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | ganador vs núcleo QA | WAPE_P50 | -0.14 | -0.37 | 0.08 | False |
| L1 OLS log des-censurado y/(1-q) - B3 Baseline × uplift promo | FVA ganador vs baseline | PA31_P75 | -15.60 | -26.38 | -7.03 | True |
| L1 OLS log des-censurado y/(1-q) - B3 Baseline × uplift promo | FVA ganador vs baseline | WAPE_P50_sq | -6.47 | -10.43 | -2.88 | True |
| L1 OLS log des-censurado y/(1-q) - B2 Media móvil 28d | FVA ganador vs baseline | PA31_P75 | -60.42 | -77.79 | -42.49 | True |
| L1 OLS log des-censurado y/(1-q) - B2 Media móvil 28d | FVA ganador vs baseline | WAPE_P50_sq | -24.90 | -30.53 | -18.86 | True |
| C4 LIN* + competidor bloque en curso (SC-08) - L1 OLS log des-censurado y/(1-q) | SC-08 competidor bloque vs ganador | PA31_P75 | -0.14 | -1.92 | 1.49 | False |
| C4 LIN* + competidor bloque en curso (SC-08) - L1 OLS log des-censurado y/(1-q) | SC-08 competidor bloque vs ganador | WAPE_P50_sq | -0.52 | -1.28 | 0.20 | False |

## Hold-out (3-30 jun, 4 folds)

| modelo | PA31_P75_% | PA31_P75_todos_% | PA21_P75_% | PA31_P50_% | Cob_P75_% | Falt_P75_% | Falt_P75_u | Sobr_P75_% | WAPE_P50_% | WAPE_P50_sq_% | MAE_P50_sq | Bias_P50_% | Bias_P50_sq_% | WAPE_promo_sq_% | WAPE_nopromo_sq_% | factor_P75_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 OLS log des-censurado y/(1-q) | 11.98 | 13.29 | 10.94 | 14.00 | 77.78 | 1.04 | 478.61 | 8.86 | 7.96 | 6.86 | 116.93 | 0.97 | -0.28 | 5.82 | 7.59 | 1.08 |
| L2 OLS log + promo² (q control) | 12.41 | 13.68 | 10.75 | 14.87 | 70.37 | 1.66 | 764.84 | 7.43 | 8.47 | 7.33 | 124.98 | 1.08 | -0.21 | 6.27 | 8.07 | 1.06 |
| E2 Ensamble top-2 | 12.39 | 13.67 | 10.81 | 14.79 | 70.37 | 1.58 | 727.00 | 7.65 | 8.52 | 7.37 | 125.70 | 1.25 | -0.05 | 6.18 | 8.20 | 1.06 |
| M5a GLM Poisson offset log(1-q) | 12.56 | 13.85 | 11.35 | 13.78 | 74.07 | 1.21 | 554.93 | 8.94 | 8.00 | 6.91 | 117.72 | 1.27 | 0.03 | 5.69 | 7.76 | 1.07 |
| L0 Ridge log núcleo QA (q control) | 11.81 | 13.12 | 10.97 | 14.31 | 77.78 | 0.84 | 386.90 | 9.28 | 7.90 | 6.82 | 116.33 | 0.57 | -0.66 | 5.97 | 7.42 | 1.09 |
| C4 LIN* + competidor bloque en curso (SC-08) | 12.20 | 13.49 | 11.33 | 12.63 | 77.78 | 0.87 | 400.69 | 9.59 | 8.47 | 7.36 | 125.43 | 3.31 | 2.09 | 6.59 | 7.89 | 1.06 |
| B3 Baseline × uplift promo | 16.98 | 17.82 | 14.93 | 24.76 | 70.37 | 2.05 | 943.66 | 10.83 | 11.62 | 11.06 | 188.48 | -1.81 | -2.65 | 11.00 | 11.09 | 1.12 |

| comparacion | contexto | metrica | dif_pp | IC95_lo | IC95_hi | significativa |
|---|---|---|---|---|---|---|
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | hold-out | PA31_P75 | 0.17 | -0.33 | 0.75 | False |
| L1 OLS log des-censurado y/(1-q) - L0 Ridge log núcleo QA (q control) | hold-out | WAPE_P50_sq | 0.03 | -0.06 | 0.17 | False |
| L2 OLS log + promo² (q control) - L0 Ridge log núcleo QA (q control) | hold-out | PA31_P75 | 0.61 | -0.69 | 2.01 | False |
| L2 OLS log + promo² (q control) - L0 Ridge log núcleo QA (q control) | hold-out | WAPE_P50_sq | 0.51 | 0.35 | 0.77 | True |
| E2 Ensamble top-2 - L0 Ridge log núcleo QA (q control) | hold-out | PA31_P75 | 0.59 | -0.63 | 1.90 | False |
| E2 Ensamble top-2 - L0 Ridge log núcleo QA (q control) | hold-out | WAPE_P50_sq | 0.55 | 0.30 | 0.91 | True |
| M5a GLM Poisson offset log(1-q) - L0 Ridge log núcleo QA (q control) | hold-out | PA31_P75 | 0.75 | -0.31 | 1.45 | False |
| M5a GLM Poisson offset log(1-q) - L0 Ridge log núcleo QA (q control) | hold-out | WAPE_P50_sq | 0.08 | -0.13 | 0.33 | False |
| C4 LIN* + competidor bloque en curso (SC-08) - L0 Ridge log núcleo QA (q control) | hold-out | PA31_P75 | 0.39 | -0.58 | 1.31 | False |
| C4 LIN* + competidor bloque en curso (SC-08) - L0 Ridge log núcleo QA (q control) | hold-out | WAPE_P50_sq | 0.53 | 0.33 | 0.75 | True |
| B3 Baseline × uplift promo - L0 Ridge log núcleo QA (q control) | hold-out | PA31_P75 | 5.17 | 1.97 | 8.52 | True |
| B3 Baseline × uplift promo - L0 Ridge log núcleo QA (q control) | hold-out | WAPE_P50_sq | 4.23 | 1.11 | 7.37 | True |
| L2 OLS log + promo² (q control) - L1 OLS log des-censurado y/(1-q) | hold-out vs ganador | PA31_P75 | 0.43 | -0.36 | 1.29 | False |
| E2 Ensamble top-2 - L1 OLS log des-censurado y/(1-q) | hold-out vs ganador | PA31_P75 | 0.41 | -0.30 | 1.18 | False |
| M5a GLM Poisson offset log(1-q) - L1 OLS log des-censurado y/(1-q) | hold-out vs ganador | PA31_P75 | 0.58 | -0.09 | 1.09 | False |
| C4 LIN* + competidor bloque en curso (SC-08) - L1 OLS log des-censurado y/(1-q) | hold-out vs ganador | PA31_P75 | 0.22 | -0.82 | 1.52 | False |
| B3 Baseline × uplift promo - L1 OLS log des-censurado y/(1-q) | hold-out vs ganador | PA31_P75 | 5.00 | 1.22 | 8.71 | True |

## Extrapolación

| modelo | dias_y>max_train | P50/y_en_esos_dias | max_P50_vs_max_train | WAPE_P50_promo30_sq_% | Bias_P50_promo30_sq_% |
|---|---|---|---|---|---|
| L1 OLS log des-censurado y/(1-q) | 7 | 0.92 | 1.10 | 9.06 | -5.31 |
| L0 Ridge log núcleo QA (q control) | 7 | 0.91 | 1.09 | 9.67 | -6.07 |
| L2 OLS log + promo² (q control) | 7 | 0.96 | 1.18 | 6.44 | 1.02 |
| M6 Cuantílica log (tau 0,5 / 0,75) | 7 | 0.92 | 1.10 | 8.89 | -5.32 |
| M7 Híbrido LIN* + GBM residuos | 7 | 0.92 | 1.08 | 10.61 | -5.13 |
| M8a GBM directo (log y) | 7 | 0.90 | 1.04 | 13.48 | -10.54 |
| M8b RandomForest directo | 7 | 0.80 | 0.94 | 21.40 | -17.83 |

## Sensibilidad de costos

| costo_sub:sobre | cuantil_objetivo | factor_med | Cob_% | Falt_% | Sobr_% | PA11_% | PA21_% | PA31_% | PA41_% | PA51_% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1:1 | 0.50 | 1.00 | 50.00 | 4.22 | 3.50 | 7.72 | 11.95 | 16.17 | 20.39 | 24.62 |
| 2:1 | 0.67 | 1.05 | 67.50 | 2.63 | 5.58 | 8.21 | 10.84 | 13.47 | 16.09 | 18.72 |
| 3:1 | 0.75 | 1.05 | 73.75 | 2.16 | 6.48 | 8.64 | 10.81 | 12.97 | 15.13 | 17.30 |
| 4:1 | 0.80 | 1.06 | 75.00 | 1.93 | 7.22 | 9.15 | 11.08 | 13.01 | 14.95 | 16.88 |
| 5:1 | 0.83 | 1.08 | 75.00 | 1.66 | 7.83 | 9.49 | 11.15 | 12.80 | 14.46 | 16.11 |
