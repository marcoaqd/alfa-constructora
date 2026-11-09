# Pronóstico de demanda — Reto 1

Proyecto para pronosticar `unidades_vendidas` (demanda diaria en un cliente) con un equipo de
agentes de IA de Claude Code.

## Equipo de agentes (`.claude/agents/`)

| Agente | Rol |
|---|---|
| `orquestador-forecast` | Coordina al equipo, consolida opiniones y levanta decisiones al usuario |
| `estadistico-data-science` | EDA, selección de variables, modelado estadístico/ML |
| `experto-supply-chain-forecasting` | Lógica de negocio, horizonte, métricas operativas |
| `qa-generalizacion` | Leakage, validación temporal, robustez y generalización |

Para usarlos en Claude Code: *"Usa el agente orquestador-forecast para …"* (o cualquiera de los
otros por su nombre).

## Estructura

```
forecast/
├── data/reto1_dataset.xlsx   # dataset original (Datos_completos + Diccionario)
├── src/                      # scripts reproducibles
├── reportes/                 # reportes de cada agente
└── DECISIONES.md             # bitácora de decisiones del orquestador
```

## Flujo de trabajo

1. **Estadístico** → entiende los datos y propone variables. *(hecho: `reportes/01_analisis_estadistico.md`)*
2. **Supply chain** → valida que las variables existan al momento de pronosticar y tengan lógica de negocio.
3. **QA** → audita leakage, sobreajuste y diseña el backtesting.
4. **Orquestador** → consolida y presenta decisiones en `DECISIONES.md`.
5. Modelo final → se evalúa con el dataset de validación externo que entregará el usuario.
