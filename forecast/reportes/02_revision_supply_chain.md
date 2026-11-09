# 02 · Revisión de supply chain y demand planning

- **Insumos:** `reportes/01_analisis_estadistico.md`, `DECISIONES.md` (D-01 a D-07) y `data/reto1_dataset.xlsx`.
- **Script:** `forecast/src/02_sc_revision_negocio.py`. Se ejecuta con `python3 -I forecast/src/02_sc_revision_negocio.py`, es determinista y escribe los CSV en `forecast/reportes/02_salidas/`. Tiene una sección por tema, de la A a la G, que coinciden con las referencias de este reporte.
- **Cómo se evaluó:**
  - **Origen rodante semanal con horizonte de 7 días (D-01):** 17 orígenes entre el 2026-03-04 y el 2026-06-24. En cada uno se entrena con todo lo anterior y se pronostican los 7 días siguientes, en total 119 días.
  - **Pronóstico ex ante:** quiebre = 0 (D-03), distribución = media de los últimos 7 días (D-05) y competidor según el escenario (D-02).
  - **Días evaluados:** solo los 113 días **sin quiebre**. Esos son los únicos días en que la venta observada es igual a la demanda. Evaluar una "demanda total" contra ventas censuradas castiga al modelo justo cuando acierta.
  - **Comparabilidad:** por este método de evaluación, mis cifras no son directamente comparables con el 8,3 % del estadístico, que evalúa un holdout fijo de 61 días e incluye días con quiebre.

---

## 1. Lectura de negocio

1. Es un SKU impulsado por un **calendario promocional planificado** que se repite en bloques fijos: promo propia de 7 días cada 2–3 semanas, exhibición en ciclos semanales de jueves a miércoles y promo del competidor también en bloques de 7 días. Lo que mueve la demanda se conoce de antemano casi en su totalidad.
2. La venta observada está **censurada** por los quiebres. Los datos encajan casi exactamente con *venta = demanda × (1 − quiebre)*: el coeficiente de log(1−q) es 1,00 (IC95 0,95–1,06). Se perdió aproximadamente el **3,0 % de la demanda del semestre** (8.393 u.).
3. **La mitad de los eventos de quiebre (4 de 8) ocurren durante una promo o en los 2 días siguientes.** Eso apunta a que la reposición no anticipa la promo, y es justo donde D-04 más duele.
4. Con D-04 (quedarse corto es peor), el modelo debe elegirse por **costo asimétrico (pinball τ = 0,75)** y entregar dos salidas: P50 para S&OP y P75 para reposición. Con eso el costo 3:1 baja **~19 %** frente a entregar solo la mediana.
5. `ventas_hace_7_dias` no se debe usar. Tiene una validez operativa dudosa, en 99 de 174 días arrastra el estado de promo equivocado y en el backtest empeora el WAPE (7,57 → 7,94 %).

---

## 2. Tabla de variables (horizonte 7 días, decisiones D-01 a D-07)

| variable | ¿conocida antes del pronóstico (t−7)? | lógica de negocio | recomendación |
|---|---|---|---|
| `promo_pct` | **Sí**: plan promocional acordado con el cliente | Principal palanca. El efecto es **convexo**: el uplift observado es ×1,24 / ×1,51 / ×1,99 / ×2,32 con 10/15/25/30 %. El log-lineal sobreestima la promo del 10 % (×1,32 vs ×1,24) | **Núcleo.** Agregar `promo_pct²` como candidata (P50: WAPE 7,07 → 6,95 %, costo 3:1 14,3 → 13,6 %). **No** usar dummies por nivel: fallan cuando aparece un nivel nuevo (WAPE 9,8 %). Limitar a 30 % (D-07) |
| `precio` | Parcial: el **precio planificado** sí; el observado no | El observado tiene ruido (CV 1,1 %) y parece un precio medio realizado, que se conoce ex post. Es igual a lista × (1 − promo) | **Descartar el observado**. Usar `promo_pct`. Si el dataset externo trae cambios de precio de lista, pedir el `precio_lista` planificado |
| `exhibicion` | **Sí, con riesgo de ejecución**: bloques de 7 días que siempre empiezan en jueves (ciclo de espacio del cliente) | +14 %. En 26 días hubo exhibición sin promo. Promo × exhibición no aporta (p 0,21) | **Núcleo.** Confirmar que el plan de exhibición está en firme 7 días antes y cuál es su cumplimiento |
| `dia_semana` | Sí (calendario) | Lun–mié entre −10 y −13 % frente al fin de semana | **Núcleo** |
| `dia_quincena` | Sí (calendario) | El efecto viene del **fin de mes / día 1** (+3,1 %, p 0,03), no de los días 14–16 (+1,2 %, p 0,48). En el backtest quitarla mejora un poco (7,07 → 6,96 %) | **Candidata**. Probar `fin_mes` en su lugar; si no mejora el pinball, dejarla fuera |
| `mes` | Sí | Con 6 meses no se puede estimar la estacionalidad anual, y mayo y junio no existen en entrenamiento | **Descartar** (reevaluar con ≥ 2 años) |
| `distribucion_numerica` | **Sí, como estado**: cambia despacio y se conoce al origen. D-05: último valor | Escalón de 0,84 a 0,91 el **2 de marzo**. La elasticidad es **2,06** (IC95 1,74–2,38), más que proporcional, lo que sugiere tiendas nuevas más grandes o un evento confundido. Forzar elasticidad 1 empeora (WAPE 7,9 %) | **Núcleo**, con la media de los últimos 7 días al origen. Pedir el **plan de listados**: si se gana distribución sin aviso, el error será un sub-pronóstico, que es el lado caro |
| `quiebre_stock_pct` | **No** (se mide el mismo día). D-03: demanda total | Censura la venta en proporción exacta a (1−q). Se concentra en promos y post-promo | **No es variable de entrada.** **Des-censurar el objetivo**: entrenar con `y/(1−q)` (sección B, WAPE 7,61 % vs 8,90 % si se ignora). El control con coeficiente libre da casi lo mismo (7,71 %); el des-censurado es más simple y robusto con solo 14 días de quiebre |
| `promo_competidor` | **No en el futuro (D-02), pero sí en el presente**: al origen se observa si el competidor está en promo, y sus bloques duran exactamente 7 días | −9,5 % en días sin promo propia | **Mantener en el entrenamiento como control** (no eliminarla). Al pronosticar, marcar los días restantes del bloque ya iniciado al origen y 0 en el resto (C4: WAPE 7,16 % vs 7,71 % si se excluye; el oráculo da 6,94 %). Si no hay observación diaria del competidor: control y pronóstico con 0 (C2: 7,57 %, bias +0,9 %, conservador y alineado con D-04) |
| `precio_competidor` | No (precio observado del competidor) | Espejo de `promo_competidor` (R² 0,99) | **Descartar** |
| `indice_precio_relativo` | No (depende del precio del competidor) | Ratio determinístico | **Descartar** |
| `inversion_medios_usd` | **Sí** (plan de medios) | Las pautas acompañan a las promos de 10, 15 y 30 % (≈ 1.580–2.000 USD/día, frente a ≈ 1.030 sin promo; en las de 25 % fueron ≈ 790), así que está confundida. Sin efecto condicional ni con adstock (estadístico) | **Descartar** por ahora. Pedir el plan de medios si negocio espera un efecto de marca de largo plazo, que no se puede medir con 6 meses |
| `grp_tv` | Solo el **GRP planificado**. El GRP observado (post-buy) se conoce después | Redundante con medios (R² 0,98) | **Descartar** |
| `temperatura_c` | Solo como **pronóstico del clima** (error típico de 2–3 °C a 7 días) | Sin efecto condicional (p 0,76), con drift estacional | **Descartar** |
| `inflacion_mensual_pct` | **No**: se publica después del cierre del mes; a t−7 solo se conoce la del mes anterior | 6 valores; fuera de rango en validación | **Descartar** |
| `confianza_consumidor` | No / con rezago de publicación | 5 valores | **Descartar** |
| `ventas_mismo_dia_anio_anterior` | Sí | Arrastra el calendario de promos del año pasado (que no se conoce) y su nivel es −24 % | **Descartar**. Reevaluar con histórico largo y el calendario promocional del año anterior para limpiarla |
| `ventas_hace_7_dias` | **Depende del momento de emisión y de la latencia del sell-out** (ver 2.1) | 99 de 174 días tienen un estado de promo distinto en t y t−7. Además está censurada en 13 días | **Descartar** como regresor (ver 2.1) |
| `codigo_region` | ? (D-06) | Rota día a día en una serie única | **Descartar**. El pipeline debe aceptar códigos nuevos sin fallar |
| `unidades_facturadas_mismo_dia` | **No** | Sell-in del mismo día: es consecuencia de la venta, no causa (r 0,9993) | **Descartar (leakage)**. Que se parezca tanto al sell-out sugiere reposición contra venta (VMI/consignación); a confirmar |
| `fecha` / `periodo` | — | Identificadores | No son variables |
| *Derivadas probadas* | Sí (calendario de promo planificado) | Post-promo días 1–3: −1,6 % (p 0,34), **sin pantry loading**. Pre-promo días 1–3: −1,3 % (p 0,50). Fatiga dentro de la promo: −0,1 %/día (p 0,75). Día 1 de promo: −2,6 % (p 0,41). Feriados (Año nuevo, Semana Santa, 1-may, día de la madre y del padre): \|z\| ≤ 1,7 | **No incluir.** Post-promo días 4–7 da +4,6 % (p 0,005) pero no tiene lógica de negocio clara y en el backtest es neutro (7,07 → 7,12 %): solo monitorear |

### 2.1 ¿Es legítimo `ventas_hace_7_dias` con horizonte de 7 días?
- **Validez temporal.** Sea *o* el último día con sell-out disponible al emitir el pronóstico, que cubre los días o+1 … o+7.
  - y(t−7) = y(o+h−7) es válido para h = 1..7 **solo si** el pronóstico se emite **después del cierre del día o** con su venta ya cargada.
  - Si se emite al inicio del día o, o si el sell-out del cliente llega con latencia de L días (lo típico es 1–2 días por portal o EDI, y a veces semanal), el rezago mínimo válido es **7 + L**. Para h = 7, y(t−7) ya no existiría.
  - Regla operativa: construir los rezagos como "último dato disponible al origen", nunca como un shift fijo.
- **Valor de negocio.**
  - En 57 % de los días (99/174) t y t−7 tienen un estado de promo distinto. Corr(y(t−7), promo(t−7)) = 0,73 y corr(y(t−7), promo(t)) = −0,28: el rezago transporta la promo **de la semana pasada**.
  - Está censurada en 13 días.
  - En el backtest, el log y(t−7) crudo empeora el WAPE de 7,57 a 7,94 % y el des-censurado lo deja en 7,97 %. El naive t-7 ajustado por uplift da 8,37 % y el crudo 44,9 %.
- **Promedios móviles de ≥ 7 días:** tienen el mismo problema (mezclan semanas con y sin promo). La única versión con lógica es una **corrección de nivel**: la media de los residuos log del modelo, sobre demanda des-censurada, de los últimos 7–14 días disponibles. Sale neutra en WAPE (7,56 %) y mejora un poco el costo 3:1 (14,2 → 13,6 %).
- **Recomendación:** no usar rezagos de venta como regresor. Usar la corrección de nivel solo como **tracking signal / alarma** de cambio estructural (por ejemplo, un cambio de distribución no informado), no en el núcleo. Nota: el ACF de los residuos en el rezago 7 es −0,28 en entrenamiento; ninguna variante lo convierte en ganancia fuera de muestra.

### 2.2 Tratamiento del quiebre (sección B; días sin quiebre; pronóstico con q = 0)
| opción | WAPE % | Bias % | Costo 3:1 % |
|---|---|---|---|
| B1 control q lineal (estadístico), pronóstico con q = 0 | 7,71 | −1,13 | 16,5 |
| B2 control log(1−q), pronóstico con 0 | 7,61 | −0,81 | 16,0 |
| **B3 des-censurar: objetivo y/(1−q)** | **7,61** | −0,82 | **16,0** |
| B4 excluir días con quiebre | 7,80 | −0,44 | 16,0 |
| B5 ignorar el quiebre | 8,90 | −2,06 | 19,9 |

- **Recomendación: B3.** No requiere estimar un coeficiente con solo 14 días, deja una serie de *demanda* reutilizable (para rezagos, reportes y la medición de venta perdida) y su supuesto es explícito.
- **Supuesto:** no hay recuperación de la venta perdida por sustitución ni por compra diferida. Si la hubiera, la demanda quedaría algo sobreestimada, que es el lado aceptable según D-04.
- **Evaluación:** medir siempre en días sin quiebre, o contra y/(1−q).

### 2.3 Métrica asimétrica (D-04; secciones F y G)
- **Propuesta:**
  - Costo **sub:sobre = 3:1** → cuantil objetivo τ = 3/(3+1) = **0,75**. Es la razón crítica del newsvendor, equivalente a un **nivel de servicio de ciclo diario del 75 %**.
  - **Métrica de selección:** pérdida pinball(0,75) del P75, en % de la demanda, sobre el backtest rodante de 7 días y en días sin quiebre.
- **Por qué 3:1:** en consumo masivo no perecedero, el costo de un faltante (margen perdido, multas OTIF del cliente, pérdida de exhibición) suele ser varias veces el costo de mantener una semana de inventario extra. Hay que confirmar el ratio con el usuario; la sensibilidad está abajo.
- **Sensibilidad** (backtest; cuantil empírico de los residuos in-sample):

| cuantil | costo 1:1 | costo 2:1 | costo 3:1 | costo 4:1 | Bias % | Cobertura % |
|---|---|---|---|---|---|---|
| 0,50 | **7,5** | 11,3 | 15,1 | 19,0 | −0,2 | 49 |
| 0,67 | 8,0 | **10,5** | 12,9 | 15,4 | +3,1 | 67 |
| 0,75 | 8,8 | 10,6 | **12,3** | 14,1 | +5,3 | 73 |
| 0,80 | 9,5 | 10,9 | 12,4 | **13,8** | +6,5 | 77 |
| 0,90 | 12,5 | 13,1 | 13,8 | 14,4 | +11,2 | 89 |

  - El óptimo empírico coincide con la teoría: 2:1 → 0,67; 3:1 → 0,75–0,80; 4:1 → 0,80.
  - Con el P75, el faltante baja de 3,6 % a 1,8 % de la demanda, es decir, un **fill rate implícito de 96,4 % → 98,2 %**, a cambio de un sobrante de 3,5 % → 6,2 %.
- **Dos salidas y no un pronóstico sesgado:**
  - **P50** (insesgado) para S&OP, finanzas y comunicación con el cliente.
  - **P75** para el pedido de reposición.
  - Sesgar el único pronóstico contamina el plan de demanda y genera efecto látigo.
- **Detalle operativo:**
  - Los cuantiles calculados con residuos in-sample **sub-cubren** (cobertura 69–73 % frente a 75 % esperado). El factor del P75 debe calibrarse con residuos **fuera de muestra** del backtest rodante.
  - Si la reposición es semanal, el riesgo relevante es el **error de la suma de 7 días** (WAPE 4,9 % frente a 7,6 % diario). Sumar 7 P75 diarios sobreprotege: cubre 82 % de las semanas con 5,7 % de sobrante. En ese caso hay que calcular el cuantil sobre la suma semanal.
  - QuantReg directo en log (τ = 0,75) es comparable (costo 13,2 vs 12,3) y es una alternativa válida.
- **Métricas secundarias** (reportar siempre):
  1. WAPE del P50.
  2. Bias del P50, con banda objetivo ±2 %.
  3. WAPE y bias **en días de promo**: ahí se concentran los quiebres y hoy es el segmento con más error (8,0 % vs 7,2–7,4 % sin promo).
  4. Faltante % (≈ 1 − fill rate) y sobrante %.
  5. Cobertura del P75 (objetivo 75 ± 5 %).
  6. Error del acumulado de 7 días.
  7. Error por horizonte h = 1..7. Hoy no se degrada con h: 5–11 % sin tendencia.

### 2.4 Configuración de negocio recomendada (sección G, backtest)
- **Configuración:** objetivo des-censurado + promo del competidor como control con el bloque en curso al origen + núcleo sin quincena.
  - **P50:** WAPE **6,96 %**, bias −0,3 %, costo 3:1 14,3 %.
  - **P75:** costo 3:1 **11,6 %**, faltante 1,8 %, cobertura 72 %.
- **Referencia:** núcleo del estadístico ajustado a D-02 (competidor excluido) y D-03: WAPE 7,71 %, bias −1,1 %, costo 3:1 16,5 %.

---

## 3. Riesgos operativos y mitigación

| riesgo | evidencia | mitigación |
|---|---|---|
| El plan promocional o de exhibición cambia dentro de la ventana de 7 días, o no se ejecuta | Promo y exhibición son las variables con más peso (permutación del estadístico: ΔMAE +494 y +41) | Volver a correr el pronóstico ante cada cambio del plan; medir el **cumplimiento** de exhibición; alimentar el modelo con el plan vigente al origen, no con lo ejecutado |
| Quiebres en promo y post-promo (reposición no anticipa la promo) | 4 de 8 eventos: 16-feb (día 1 de promo), 16–17-mar (días 5–6), 4–5-may y 28-jun (1–2 días después de promos de 30 % y 25 %) | Usar el P75 en promo; revisar con supply el **pre-build de inventario** antes de cada promo y la cobertura de los 2–3 días posteriores. Opcional: τ = 0,80 solo en días de promo |
| Descuentos > 30 % (D-07) | El efecto es convexo: la extrapolación con `promo²` crece rápido | Limitar a 30 % y emitir una alerta. Para mecánicas nuevas, usar juicio comercial (override documentado) |
| Distribución: el "último valor" (D-05) no anticipa nuevos listados | Escalón del 2-mar con elasticidad ≈ 2; un solo evento | Pedir el plan de listados; monitorear la **tracking signal** (residuos de 7–14 días) y alertar si \|sesgo\| > 5 % dos semanas seguidas |
| Latencia del sell-out | Desconocida | No usar rezagos de venta en el núcleo; si se usan, rezago ≥ 7 + latencia |
| Supuesto de venta perdida proporcional | Coef. log(1−q) = 1,00 | Si hay sustitución entre SKUs o tiendas, la demanda queda algo sobreestimada (lado aceptable según D-04). Revalidar el coeficiente en el dataset externo |
| Promo del competidor no observada al origen | C4 requiere observación diaria | Respaldo: C2 (pronóstico con 0), conservador (+0,9 % de bias) |
| Muestra chica | 8 bloques de promo; cada nivel de descuento con 1–3 bloques | No usar dummies por nivel; forma paramétrica (lineal + cuadrática); revisar coeficientes en cada reentrenamiento |
| Sell-in ≠ sell-out | Facturación ≈ venta del mismo día | Si la reposición se planifica con sell-in (pedidos), el pronóstico de sell-out se traduce a pedidos según la política de inventario, no se copia |

---

## 4. Decisiones que debe tomar el usuario (para el orquestador)

| ID | Decisión | Por defecto propuesto |
|---|---|---|
| SC-01 | **Ratio de costo sub:sobre** (margen perdido + penalidades del cliente vs. costo de mantener inventario; ¿producto perecedero?) | **3:1 → P75**, con sensibilidad 2:1 y 4:1 |
| SC-02 | **Momento de emisión y latencia del sell-out**: ¿se emite al cierre del día con la venta de ese día disponible? ¿Con cuántos días de retraso llega el dato del cliente? | Inicio del día con latencia de 1 día → no usar rezagos de venta |
| SC-03 | **¿Se observa a diario si el competidor está en promo** (fuerza de campo, scraping)? Las promos del competidor duran 7 días | Sí → C4 (bloque en curso). Si no → C2 (pronóstico con 0) |
| SC-04 | **Frecuencia de reposición y lead time** (diaria o semanal) | Semanal → calibrar el P75 sobre la suma de 7 días |
| SC-05 | **¿Qué pasó el 2 de marzo** con la distribución (0,84 → 0,91)? ¿Hay un plan de listados? | Último valor (D-05), con alarma de tracking signal |
| SC-06 | **¿El plan de exhibición está en firme 7 días antes** y cuál es su cumplimiento? | Plan = ejecución |
| SC-07 | **¿La venta perdida en quiebre se recupera** (sustitución o compra diferida)? | No se recupera → objetivo y/(1−q) |
| SC-08 | Confirmar que el orquestador cambia D-02 de "sale del modelo" a "**se mantiene como control en el entrenamiento**" | Mantener como control |

### Lista final
- **Núcleo:** `promo_pct`, `exhibicion`, `dia_semana`, `distribucion_numerica` (media de 7 días al origen) y `promo_competidor` como **control** (pronóstico: bloque en curso o 0). Objetivo: **log(y/(1−quiebre))**.
- **Candidatas a probar** (aceptar solo si mejoran el pinball 0,75 en backtest rodante):
  - `promo_pct²`
  - `fin_mes` en lugar de `dia_quincena`, o ninguna
  - corrección de nivel por residuos de 7–14 días
  - `post_4_7` (solo monitoreo)
- **Descartadas:**
  - Leakage o sin disponibilidad ex ante: `unidades_facturadas_mismo_dia`, `quiebre_stock_pct` como entrada, `precio_competidor`, `indice_precio_relativo`, `inflacion_mensual_pct`, `confianza_consumidor`.
  - Sin aporte o redundantes: `ventas_hace_7_dias`, `ventas_mismo_dia_anio_anterior`, `precio` observado, `inversion_medios_usd`, `grp_tv`, `temperatura_c`, `mes`, `codigo_region`.
  - Efectos de negocio sin evidencia: dummies por nivel de descuento, promo × exhibición, pre-promo, post-promo 1–3 días, fatiga dentro de la promo y feriados.

### Familias de modelos recomendadas (óptica de negocio)
1. **Regresión log-lineal** (OLS/Ridge) sobre la demanda des-censurada, con P75 por cuantil de residuos **fuera de muestra**. Es el modelo principal: interpretable (uplift por palanca) y extrapola de forma multiplicativa.
2. **Regresión cuantílica en log** (τ = 0,5 y 0,75): da el P75 directamente y se ajusta a D-04.
3. **GLM Poisson / binomial negativa** con enlace log y **offset log(1−q)**: el modelo natural para conteos censurados en proporción a tiendas sin stock; evita el sesgo de retransformación.
4. **Híbrido**: log-lineal + boosting sobre residuos, con restricción monótona en promo. Solo si el dataset externo trae mucha más historia; los árboles solos no extrapolan.
5. **Benchmarks obligatorios para FVA (Forecast Value Added):** "baseline × uplift de promo" (el método del planeador) y naive t-7 ajustado por promo (8,4 %).
- **No recomendados:** ARIMA, ETS o Prophet sin regresores. La demanda la dicta el calendario comercial, no la inercia.
