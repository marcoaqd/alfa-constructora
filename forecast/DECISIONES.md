# Bitácora de decisiones — Pronóstico de demanda

Mantenida por el agente `orquestador-forecast`. Cada decisión tiene un **supuesto por defecto**
que el equipo usará mientras el usuario no decida.

## Iteración 1 — Selección de variables (2026-11-09)

**Fuente:** `reportes/01_analisis_estadistico.md` (agente `estadistico-data-science`).
Revisión de `experto-supply-chain-forecasting` y `qa-generalizacion`: pendiente (iteración 2).

### Resultado
- Modelo propuesto: regresión lineal (Ridge/OLS) sobre `log(unidades_vendidas)` con 7
  variables núcleo → **WAPE 6,7 % en validación** (8,3 % si el quiebre de stock no se conoce de
  antemano), frente a 28,2 % del mejor baseline.
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
