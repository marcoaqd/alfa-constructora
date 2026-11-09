# Bitácora de decisiones — Pronóstico de demanda

Mantenida por el agente `orquestador-forecast`. Cada decisión tiene un **supuesto por defecto**
que el equipo usará mientras el usuario no decida.

## Iteración 1 — Selección de variables (2026-11-09)

**Fuente:** `reportes/01_analisis_estadistico.md` (agente `estadistico-data-science`).
Revisión de `experto-supply-chain-forecasting` y `qa-generalizacion`: pendiente (iteración 2).

### Resultado
- Modelo propuesto: regresión lineal (Ridge/OLS) sobre `log(unidades_vendidas)` con 7
  variables núcleo. El ~~WAPE 6,7 % en validación~~ era una cifra **con oráculo**: usaba el
  quiebre y la promo del competidor reales, que no se conocen al pronosticar.
  - **Cifra honesta** (corregida en la iteración 2), con las decisiones D-01 a D-05 y backtest
    rodante de 7 días: **WAPE ≈ 9,7 % sobre la venta observada y ≈ 7,4 % en días sin quiebre**.
  - En el hold-out de junio: **8,0 % y 6,9 %**.
  - Ver `reportes/03_revision_qa.md` y `reportes/04_torneo_modelos.md`.
  - Las cifras de R-01 y R-03 (más abajo) también se calcularon con oráculo; sus conclusiones
    se mantienen.
- Variables núcleo: `promo_pct`, `exhibicion`, `dia_semana`, `distribucion_numerica`,
  `quiebre_stock_pct` (solo como control), `promo_competidor`, y `dia_quincena` (dudosa).
- Descartadas: `unidades_facturadas_mismo_dia` (leakage, r = 0,9993), `ventas_hace_7_dias`,
  `ventas_mismo_dia_anio_anterior`, `indice_precio_relativo`, `precio_competidor`, `precio`
  (redundante con promo), medios/GRP, temperatura, inflación, confianza, región y mes.

### Decisiones resueltas con evidencia
| ID | Decisión | Evidencia |
|---|---|---|
| R-01 | Excluir `unidades_facturadas_mismo_dia` | Es consecuencia de la venta: infla el WAPE de 6,7 % a 1,4 %, un resultado que no se puede repetir en producción |
| R-02 | Usar `promo_pct` **o** `precio`, nunca ambos | R² = 0,99 entre ellos |
| R-03 | Modelo lineal en log en vez de árboles | Los árboles no extrapolan: el máximo de validación (4.122) supera al de entrenamiento (3.521). GBM 9,4 % y RF 11,7 % frente a 6,7 % |

### Decisiones pendientes del usuario

**D-01 · ¿Con cuántos días de anticipación se hace el pronóstico?**
- A) ≤ 7 días: los rezagos de venta serían válidos, aunque hoy no aportan (naive t-7: WAPE 52 %).
- B) 7–30 días: solo variables planificadas (promo, exhibición, calendario, distribución).
- *Recomendación:* B. *Por defecto:* B.

**D-02 · ¿Se conoce de antemano la promoción del competidor?**
- A) Sí → se incluye como variable (WAPE 8,3 %).
- B) No → se excluye (WAPE 9,5 %) o se modela por escenarios.
- *Por defecto:* se entrenan las dos versiones y se reportan ambas.

**D-03 · ¿Qué se pronostica: demanda sin restricción o venta realizada?**
- A) Demanda sin restricción: entrenar con `quiebre_stock_pct` como control y pronosticar con
  quiebre = 0 (WAPE 8,3 %).
- B) Venta realizada: requiere que supply anticipe los quiebres (WAPE 6,7 % solo si se
  conocen).
- *Recomendación:* A, porque es lo que sirve para planear inventario. *Por defecto:* A.

**D-04 · ¿Cuál es la métrica oficial y qué cuesta más: pasarse o quedarse corto?**
- *Por defecto:* WAPE como métrica principal, con el bias reportado aparte; costo simétrico.

**D-05 · ¿Cuál es el plan de distribución numérica a futuro?**
- La distribución pasó de 0,84 a 0,91 en marzo (+17 % de ventas). El modelo necesita el valor
  planificado como insumo.
- *Por defecto:* se mantiene el último valor observado.

**D-06 · ¿Qué representa `codigo_region`?**
- No tiene efecto en estos datos (p = 0,80). Si el dataset externo trae varias regiones por
  día, hay que cambiar la granularidad del modelo.
- *Por defecto:* se ignora.

**D-07 · ¿Habrá descuentos mayores al 30 % o un efecto esperado de medios/TV?**
- Un descuento mayor al 30 % queda fuera del rango observado, así que el pronóstico sería una
  extrapolación. Los medios no muestran efecto una vez que se controla por la promo.
- *Por defecto:* se limita el descuento al 30 % y se avisa si llega uno mayor; medios queda
  fuera del modelo.

### Respuestas del usuario (2026-11-09)
| ID | Decisión del usuario | Implicación para el modelo |
|---|---|---|
| D-01 | Horizonte de **7 días** | Solo valen variables conocidas 7 días antes. Un rezago de venta solo es válido si es ≥ 7 días (por revisar si y(t−7) ya está disponible al pronosticar). |
| D-02 | Promo del competidor **no** se conoce de antemano | `promo_competidor` y `precio_competidor` salen del modelo como variables de entrada. |
| D-03 | Pronosticar **demanda total** | `quiebre_stock_pct` se usa solo al entrenar como control y se pone en 0 al pronosticar. |
| D-04 | **Quedarse corto es peor**, porque genera quiebre de stock | Costo asimétrico: además del WAPE se penaliza el sub-pronóstico (bias negativo, pérdida asimétrica). |
| D-05 | Distribución numérica: se **mantiene el último valor** | Al pronosticar se usa el último valor observado de `distribucion_numerica`. |
| D-06 | `codigo_region`: **no se sabe** qué representa | Se excluye. El pipeline debe tolerar regiones nuevas en el dataset externo. |
| D-07 | Sin respuesta | Supuesto por defecto: descuento máximo 30 % con alerta si se supera; medios fuera del modelo. |

### Próximos pasos (iteración 2)
1. `experto-supply-chain-forecasting` revisa la lista de variables y D-01 a D-07.
2. `qa-generalizacion` hace el backtesting con ventana rodante, revisa la estabilidad y
   prepara la validación de esquema para el dataset externo.
3. Se construye el pipeline final (`forecast/src/02_modelo.py`) que reciba un Excel nuevo y
   genere el pronóstico.

---

## Iteración 2: Variables (revisión cruzada) y torneo de modelos (2026-11-09)

### Fuentes
- `reportes/02_revision_supply_chain.md` (supply chain, SC)
- `reportes/03_revision_qa.md` (QA)
- `reportes/04_torneo_modelos.md` (torneo)

Scripts: `src/04_torneo_modelos.py`, `src/modelos_torneo.py` y `src/05_pronosticar.py`.

> En esta sesión no había herramienta de delegación: el orquestador implementó el torneo y el
> pipeline directamente, con el harness de QA (`src/qa_backtest.py`). **Pendiente:** que QA
> revise los scripts 04 y 05.

### Resumen
- **Qué se hizo.** Se consolidaron las tres opiniones sobre las variables. Se compararon 22
  modelos con el mismo protocolo (origen rodante, h = 7, escenario ex ante, 12 folds de
  selección) y se evaluaron 7 finalistas **una sola vez** en el hold-out (3–30 jun). Además,
  se construyó el pipeline de producción.
- **Qué se encontró.**
  - Gana un modelo **log-lineal (OLS) sobre la demanda des-censurada y/(1−q)** con 4
    variables: promo, exhibición, día de semana y distribución.
  - Ningún modelo más complejo es significativamente mejor: GLM, cuantílica, híbrido, árboles,
    ensambles y variantes con el competidor.
  - **Cifras del ganador:**

    | periodo | pérdida 3:1 del P75 | WAPE P50 sobre venta observada | WAPE P50 sin quiebre |
    |---|---|---|---|
    | Selección | 12,97 % | 9,69 % | 7,40 % |
    | Hold-out | 11,98 % | 7,96 % | 6,86 % |

  - Frente al método del planeador (baseline × uplift): −15,6 pp de pérdida 3:1, IC95
    [−26,4; −7,0].
- **Qué hay que decidir.** D-08 a D-12, más abajo.

### Variables: postura de cada agente y decisión final

Notación de la evidencia: Δ = cambio en la pérdida 3:1 del P75, en pp (backtest de selección),
con su IC95 por bootstrap.

| variable | estadístico (01) | supply chain (02) | QA (03) | **decisión final** | justificación |
|---|---|---|---|---|---|
| `promo_pct` | usar (núcleo) | núcleo; probar promo² | usar | **USAR**, lineal en log | Principal palanca: +30,8 % por cada +10 pp en el modelo final. Las tres posturas coinciden |
| `promo_pct²` (convexidad) | no propuesta | candidata (convexa) | sin opinión; regla de significancia | **NO** (en vigilancia) | (d) Δ −1,28 [−3,30; +0,62], no significativo. En el hold-out L2 empeora el WAPE sin quiebre +0,51 [+0,35; +0,77], que sí es significativo. Se reevalúa con el dataset externo porque corrige el sesgo −5 % en días con 30 % |
| `precio` | alternativa a promo | descartar el observado (es ex post) | alternativa, no usar | **DESCARTAR** | R² 0,99 con promo; el observado no es ex ante |
| `exhibicion` | usar | núcleo (confirmar plan en firme) | usar; evidencia fuera de muestra débil (+1,17 pp, no significativo) | **USAR** | +14,1 %. Coeficiente estable. Riesgo: que no esté en firme a 7 días (D-12) |
| `dia_semana` | usar | núcleo | usar (6 dummies) | **USAR** (6 dummies) | Lun–mié entre −11 y −15 % frente al domingo |
| `distribucion_numerica` | usar con cautela | núcleo, media de 7 días | usar con recorte y alerta | **USAR**, con el último valor (D-05) y alerta fuera de rango | Quitarla: +3,78 pp de WAPE (QA, significativo). La media de 7 días de SC no se adopta: D-05 dice último valor y la diferencia es ruido (sd 0,004) |
| `quiebre_stock_pct` | control; 0 al pronosticar | **des-censurar y/(1−q)** | control; 0 al pronosticar | **DES-CENSURAR** (no es entrada) | (b) L1 − L0o: Δ **−0,83 [−1,62; −0,07]**, significativo, con un parámetro menos. Contra el Ridge de QA: −0,62 [−1,21; 0,00]. En el hold-out: +0,17 [−0,33; +0,75], no significativo. Coeficiente de log(1−q) = 1,00 (SC) |
| `promo_competidor` | usar si se conoce | control + bloque en curso al origen (C4) | descartar (D-02) | **FUERA del modelo oficial** (respeta D-02) | Ver el conflicto (a) |
| `precio_competidor`, `indice_precio_relativo` | descartar | descartar | descartar (D-02) | **DESCARTAR** | Redundantes y no ex ante |
| `dia_quincena` / `fin_mes` | dudosa | probar fin_mes | descartar | **NINGUNA** | (c) fin_mes −0,36 [−1,37; +0,62]; quincena −0,32 [−1,25; +0,56]; entre ellas −0,05 [−0,80; +0,53] |
| `ventas_hace_7_dias` | descartar | descartar (latencia; arrastra la promo de la semana anterior) | descartar | **DESCARTAR** | Naive t-7: 3:1 de 104 %, WAPE 51 % |
| `ventas_mismo_dia_anio_anterior` | descartar | descartar | descartar | **DESCARTAR** | Sin aporte |
| `inversion_medios_usd`, `grp_tv` | descartar | descartar (confundidas con promo) | descartar | **DESCARTAR** | Sin efecto condicional |
| `temperatura_c` | descartar | descartar | descartar | **DESCARTAR** | Sin efecto; drift |
| `inflacion_mensual_pct`, `confianza_consumidor` | descartar | descartar (no ex ante) | descartar (leakage temporal) | **DESCARTAR** | 5–6 valores; se publican después del mes |
| `codigo_region`, `mes` | descartar | descartar | descartar | **DESCARTAR** | El pipeline acepta regiones nuevas y emite una ALERTA |
| `unidades_facturadas_mismo_dia` | leakage | leakage | prohibida | **PROHIBIDA** | r 0,9993; el harness lanza error si aparece |

### Conflictos resueltos con evidencia

| ID | conflicto | posturas | resolución |
|---|---|---|---|
| R-04 (a) | Promo del competidor | **QA:** descartar por D-02. **SC:** mantenerla como control y usar el bloque en curso al origen (C4) | **Se respeta D-02 como configuración oficial** (ver el detalle debajo de la tabla). **SC-08 queda cerrada sin escalar**; se reabre si el dataset externo cambia el resultado |
| R-05 (b) | Tratamiento del quiebre | **Estadístico y QA:** control. **SC:** des-censurar | **Des-censurar.** Es significativo frente al control, más simple y deja una serie de demanda reutilizable |
| R-06 (c) | dia_quincena frente a fin_mes | **Estadístico:** dudosa. **SC:** fin_mes. **QA:** fuera | **Ninguna.** Ninguna de las dos es significativa |
| R-07 (d) | promo² | **SC:** a favor. **Estadístico y QA:** sin propuesta / regla de significancia | **No entra.** No es significativa y empeora el hold-out. Queda en vigilancia |
| R-08 (e) | Métrica de selección | **QA:** WAPE sin quiebre. **SC:** pinball 0,75 | **Principal:** pérdida 3:1 del P75 en días sin quiebre (= 4 × pinball 0,75; supuesto por defecto de D-04). Se reporta también el WAPE y el bias del P50 |
| R-09 (f) | WAPE sobre venta observada o sin quiebre | **SC:** sin quiebre. **QA:** ambos | **Se reportan ambos siempre.** La venta observada castiga al modelo de demanda en los días con quiebre |
| R-10 | Ridge o OLS | **QA:** Ridge ≈ OLS | **OLS.** El alpha de RidgeCV es inestable: con el objetivo des-censurado, Ridge da 9,34 % de WAPE sin quiebre frente a 7,40 % de OLS |
| R-11 | Árboles, híbrido y ensambles | **Todos:** solo como contraste | **Descartados** (ver el detalle debajo de la tabla) |
| R-12 | Ganador | — | **L1** (ver el detalle debajo de la tabla) |

Detalle de R-04 (promo del competidor):
- Las variantes compatibles con D-02 (control en el entrenamiento y futuro = 0 o = tasa
  histórica) no son significativas: −0,54 [−1,33; +0,26] y −0,47 [−1,41; +0,46].
- La variante de SC (C4) **no gana con claridad**:
  - En selección: −0,14 [−1,92; +1,49] y WAPE −0,52 [−1,28; +0,20].
  - En el hold-out: +0,22 [−0,82; +1,52], bias del P50 +3,3 % y WAPE sin quiebre +0,53
    [+0,33; +0,75] frente al núcleo QA, que es significativamente peor.

Detalle de R-11 (árboles, híbrido y ensambles):
- RF: +9,76 [+5,53; +14,86], significativamente peor. No supera el máximo de entrenamiento
  (0,94×).
- GBM: sesgo −10,5 % en días con 30 % de promo y brecha de 6,3 pp entre entrenamiento y
  backtest.
- Híbrido: +0,51. Ensambles: −1,3, no significativo.

Detalle de R-12 (ganador):
- Se aplicó la regla de QA de forma secuencial: un cambio entra solo si es significativo en
  selección y no empeora el hold-out.
- L1 es el mejor modelo lineal obtenido con esa regla, y ningún retador lo supera.
- El mejor por cifra puntual (L2, con promo²) queda a −1,44 [−3,61; +0,53].
- La regla mecánica "el más simple entre los empatados" elige el GLM Poisson, que es
  equivalente: mismos parámetros, −0,32 en selección y +0,58 en el hold-out. Queda
  documentado como alternativa.

### Modelo oficial vigente
- **Especificación:** `log(y/(1−q)) ~ promo_pct + exhibicion + dia_semana + distribucion_numerica`,
  ajustada por OLS con smearing.
- **Salidas:**
  - **P50**: S&OP.
  - **P75** = P50 × factor: reposición. El factor se calibra con los errores fuera de muestra
    de las últimas 8 semanas (1,065 con los 181 días).
- **Escenario:** quiebre = 0, distribución = último valor, promo ≤ 30 % con alerta, sin
  competidor.
- **Producción:** `python3 -I forecast/src/05_pronosticar.py --data-nuevo <archivo> [--modo una-vez|rodante]`.

### Decisiones pendientes del usuario

**D-08 · ¿Cuál es la razón de costo sub:sobre (quedarse corto frente a pasarse)?**
- Opciones (selección, modelo L1):

  | opción | faltante | sobrante | pérdida a su propia razón |
  |---|---|---|---|
  | A) 2:1 → P67 (factor 1,048) | 2,6 % | 5,6 % | 10,8 % |
  | B) 3:1 → P75 (factor 1,053) | 2,2 % | 6,5 % | 13,0 % |
  | C) 5:1 → P83 (factor 1,075) | 1,7 % | 7,8 % | 16,1 % |

  Entregar solo el P50 cuesta 16,2 % con 3:1.
- *Recomendación:* B. SC la justifica por consumo masivo no perecedero. QA no discrepa, pero
  pide que el número salga de finanzas: margen perdido + penalidades frente a costo de
  inventario. Riesgo bajo, porque entre 3:1 y 5:1 la curva es plana.
- *Por defecto:* 3:1 → P75.

**D-09 · ¿Cómo se usará el dataset externo?**
- A) **Pronóstico de una vez:** el modelo de 181 días pronostica todo el archivo.
  - Prueba hecha (may–jun, 61 días): 3:1 de 12,5 %, cobertura 69 %, faltante 2,1 %.
  - Riesgo: el horizonte supera 7 días y la distribución queda congelada al 30-jun.
- B) **Rodante semanal:** se re-entrena cada 7 días con los reales del archivo.
  - Misma prueba: 3:1 de 12,6 %, cobertura 74 %, faltante 1,45 %.
  - Es consistente con D-01, pero requiere que el archivo traiga `unidades_vendidas` y
    `quiebre_stock_pct`.
- *Recomendación:* B como cifra oficial y A como prueba de estrés. QA y SC coinciden.
  Confirmar también que el archivo es el mismo cliente/SKU y un periodo posterior.
- *Por defecto:* se corren los dos modos. La cifra oficial es la del modo rodante si el archivo
  trae ventas; si no, la de una vez.

**D-10 · ¿La `distribucion_numerica` del archivo externo es un plan conocido de antemano o un
dato observado después?**
- A) Observado después → se mantiene D-05 (último valor) y se emite una alerta si difiere en
  más de 0,02.
- B) Plan conocido → se usa el valor del archivo. Es ex ante legítimo y evita el sesgo.
- Impacto: +0,10 de distribución equivale a +28 % de demanda. En la prueba in-sample,
  congelar 0,914 sobre ene–feb (0,84) dio un bias de **+9,6 %**.
- *Recomendación:* B si existe un plan de listados (SC). QA exige que sea un plan y no un dato
  realizado.
- *Por defecto:* A (D-05).

**D-11 · ¿Con qué frecuencia se repone (diaria o semanal) y con qué lead time?**
- A) Diaria → P75 diario (lo actual).
- B) Semanal → el P75 se calibra sobre la suma de 7 días. Sumar 7 P75 diarios sobreprotege:
  cubre el 82 % de las semanas con 5,7 % de sobrante (SC, reporte 02).
- *Recomendación:* depende de la operación. Es un cambio menor en el pipeline.
- *Por defecto:* A.

**D-12 · ¿El plan de exhibición está en firme 7 días antes y se cumple?**
- Impacto: +14 % por día con exhibición. Si no se ejecuta, el P50 sobreestima y si se ejecuta
  sin aviso, lo subestima. Quitar la variable cuesta +1,2 pp (QA, no significativo).
- *Recomendación:* mantenerla si está en firme. Si no, usar el plan vigente al origen.
- *Por defecto:* plan = ejecución.

*Sigue pendiente D-07* (descuentos > 30 %): por defecto se recorta a 30 % y se emite una
alerta. El pipeline ya lo aplica.

### Respuestas del usuario (iteración 2)
| ID | Decisión del usuario | Estado |
|---|---|---|
| D-08 | — | pendiente (por defecto 3:1) |
| D-09 | — | pendiente (por defecto ambos modos; oficial = rodante si hay ventas) |
| D-10 | — | pendiente (por defecto D-05) |
| D-11 | — | pendiente (por defecto diaria) |
| D-12 | — | pendiente (por defecto plan = ejecución) |

### Próximos pasos
1. QA revisa `04_torneo_modelos.py`, `modelos_torneo.py` y `05_pronosticar.py`.
2. Al llegar el dataset externo:
   - correr `05_pronosticar.py` en los modos que defina D-09;
   - reportar la pérdida 3:1, el WAPE (observado y sin quiebre) y la cobertura;
   - reevaluar promo² y C4 **sin** re-seleccionar sobre ese mismo archivo.
3. Agregar `forecast/requirements.txt` con las versiones fijadas (hallazgo 15 de QA).
