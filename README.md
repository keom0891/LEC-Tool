# LEC Tool v3: Cálculo de curvas de excedencia de pérdidas, evaluación de estrategias de gestión de riesgo y análisis costo-beneficio

La herramienta **LEC Tool** consiste en una plataforma desarrollada por el **Banco Interamericano de Desarrollo** con el propósito de derivar curvas de excedencia de pérdidas (LEC) a partir de registros históricos y resultados de análisis probabilistas de desastres. Esta plataforma está diseñada para estimar la tasa de excedencia anual asociada a valores específicos de pérdidas económicas. La curva LEC resultante se utiliza posteriormente en análisis de riesgo y en la toma de decisiones para la gestión de desastres, particularmente para la selección de estrategias de transferencia y/o reducción de riesgo, y para su evaluación costo-beneficio.

![version](https://img.shields.io/badge/version-3.1.0-blue)

---

# ✨ Descripción

El motor de cálculo del LEC Tool está implementado en un conjunto de módulos Python desarrollados por el equipo de Gestión de Riesgos de Desastres del Banco Interamericano de Desarrollo. La versión 3 separa completamente los **insumos del usuario** (archivo `config.toml`) del código de cálculo, y agrupa todos los resultados de una corrida en una carpeta de salida identificada por el usuario.

## Novedades de la versión 3.1 (módulo CBA v8)

- **Umbrales por período de retorno**: la capa del seguro (y, opcionalmente, los umbrales del DDO y del PPO) se puede definir en años (`attachment_rp`, `exhaustion_rp`) en lugar de dólares. Se convierte con la curva del país que se corre, así que la misma estrategia sirve para cualquier país de `Country databases`.
- **Límite de cobertura en dólares**: el seguro se dimensiona con `coverage_limit`; la cesión se calcula.
- **Precio de la capa** (`[insurance_pricing]`): regla por tramos (`rol_rule`, por defecto), curva de reaseguro comercial (`market_curve`), cotización real del país (`gross_premium`) o ROL fijo (`fixed_rol`, comportamiento anterior).
- **Pago del seguro**: proporcional, un pago por año y pago mínimo igual a la prima bruta, como en las pólizas soberanas paramétricas.
- **Reporte del seguro**: capa en años y en dólares, fuente del precio, múltiplo implícito, B/C económico y fiscal (con descuento del donante) e indicadores de protección.
- Integra los cambios de la entrega v7 del módulo CBA (gracia en PPO y DDO, disparo propio del PPO, compatibilidad NumPy 1.x/2.x). Detalle en `docs/v8_insurance_layer.md`.

## Novedades de la versión 3

- **Base de datos de pérdidas por país** (`Country databases`): una base de datos pre-procesada por el equipo DRM del BID que contiene datos históricos y probabilistas para la mayoría de los países de Latinoamérica y el Caribe.
- **Archivo de configuración único** (`config.toml`): todos los parámetros de la corrida se definen ahí. No es necesario editar código.
- **Un solo punto de entrada**: `python main.py` ejecuta el flujo completo.
- **Módulo de análisis costo-beneficio** (`cba/`): indicadores B/C, brecha de financiamiento, eficiencia (CM, MV, OMV), Costo Neto Comparado (CNC) y costo-efectividad de la reducción de riesgo.
- **Etapas opcionales**: la reducción ex-ante del riesgo (DRR) y el CBA se ejecutan solo si se activan en la configuración. Se mejoró el flujo de trabajo para que una estrategia sin PPO no requiera del archivo de calendario del PPO.
- **Resultados persistentes**: todas las figuras, el reporte principal, el reporte CBA, las estadísticas y una copia de la configuración se guardan en `outputs/<id>/` con el identificador de la corrida como prefijo.
- **Reporte principal** con resumen de la curva LEC, PML por periodo de retorno, estadísticas de la estrategia, probabilidades de brecha de financiamiento (umbrales al 25 %, 50 %, 75 % y 90 % de la pérdida máxima de la curva LEC), resumen de la DRR y del CBA.

## Módulos

| Módulo | Descripción |
| --- | --- |
| `main.py` | Punto de entrada. Orquesta las etapas del flujo según `config.toml` |
| `config.toml` | Insumos del usuario (ver `docs/CONFIG.md`) |
| `config_loader.py` | Lectura y validación de la configuración; construcción de los objetos que usan los módulos |
| `lec_core.py` | Cálculo de la curva LEC empírica, intervalos de confianza por bootstrap, y curva híbrida |
| `hybrid_lec.py` | Construcción de curva híbrida mediante blending log-log entre curva empírica y cola probabilística |
| `simulation.py` | Generación de catálogos sintéticos de pérdidas (Poisson + muestreo inverso) con Common Random Numbers |
| `insurance_layer.py` | Ubica los umbrales en la curva del país (período de retorno → $MM) y pone precio a la capa del seguro |
| `risk_management.py` | Mecanismos de cobertura financiera: seguro paramétrico, PPO, CCF, DDO |
| `risk_reduction.py` | Reducción del riesgo ex-ante: calibración de desplazamiento de la curva LEC y catálogo reducido |
| `plots.py` | Generación de todas las figuras |
| `reporting.py` | Estadísticas, probabilidades de brecha, PML y reporte principal |
| `utils.py` | Funciones auxiliares compartidas |
| `cba/` | Paquete de análisis costo-beneficio (ver más abajo) |

---

# 📁 Estructura del repositorio

```
/
├── main.py                 punto de entrada
├── config.toml             configuración de la corrida (editar aquí)
├── config_loader.py
├── lec_core.py
├── hybrid_lec.py
├── simulation.py
├── insurance_layer.py
├── risk_management.py
├── risk_reduction.py
├── plots.py
├── reporting.py
├── utils.py
├── cba/
│   ├── config.py           parámetros por defecto del CBA (dataclasses)
│   ├── costs.py            funciones de costo y servicio de deuda a valor presente
│   ├── core.py             indicadores centrales (B/C, pérdida no pagada, B/UL)
│   ├── efficiency.py       Cost Multiple, Money Value, OMV
│   ├── cnc.py              Costo Neto Comparado (ex-ante vs deuda ex-post)
│   ├── diagnostics.py      costo-efectividad de la DRR
│   ├── discounting.py      descuento a valor presente
│   ├── engine.py           motor del CBA (run_cba)
│   └── reports.py          reporte de texto y figura del CBA
├── data/
│   ├── LEC_event_loss_example.csv
│   ├── ppo_example.csv
│   └── tail_curve_example.csv
├── Country databases/      catálogos y colas probabilísticas por país
├── docs/
│   └── CONFIG.md           referencia de todas las claves de config.toml
├── outputs/                resultados (una subcarpeta por corrida; no se versiona)
├── requirements.txt
└── .devcontainer/
```

---

# ⚙️ Prerrequisitos

Python 3.11 o superior (la configuración se lee con `tomllib`, incluido en la biblioteca estándar) con los siguientes paquetes:

```
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.13.0
matplotlib>=3.9.0
```

Instalación:

```bash
pip install -r requirements.txt
```

---

# 🚀 Uso

1. Editar `config.toml` (identificador de la corrida, archivos de entrada, parámetros de simulación, estrategia, DRR, CBA).
2. Ejecutar:

```bash
python main.py
```

Opcionalmente puede indicarse otro archivo de configuración: `python main.py mi_config.toml`.

El flujo completo es:

1. Carga del catálogo histórico de pérdidas (y de la cola probabilística y el calendario PPO si aplican)
2. Curva LEC empírica con intervalos de confianza; curva híbrida si `lec.hybrid_curve = true`
3. Ubicación de los umbrales en la curva del país y precio del seguro
4. Generación de catálogos sintéticos
5. Evaluación de la estrategia financiera sobre el catálogo base
6. Reducción del riesgo ex-ante y reevaluación de la estrategia sobre el catálogo reducido (si `risk_reduction.enabled = true`)
7. Análisis costo-beneficio (si `cba.enabled = true`), incluida la comparación DRR-CBA cuando la DRR está activa
8. Escritura de figuras, reporte principal, reporte CBA y estadísticas

Por defecto las figuras solo se guardan; con `run.show_figures = true` también se abren en pantalla.

## Archivos de entrada

| Archivo (clave en `[inputs]`) | Columnas / formato | Descripción |
| --- | --- | --- |
| `event_loss_file` | `year`, `econ_loss` | Catálogo histórico de pérdidas por evento ($MM). Obligatorio |
| `tail_curve_file` | `tail_loss`, `tail_aep` | Cola probabilística para la curva híbrida ($MM, tasa anual de excedencia). Solo si `lec.hybrid_curve = true` |
| `ppo_schedule_file` | fila sin encabezado | `catalogue_length` valores de cobertura PPO disponible por año ($MM). Solo si la estrategia incluye un instrumento `ppo` |

Los formatos de ejemplo están en la carpeta [`data/`](data/). La carpeta `Country databases/` contiene catálogos y colas probabilísticas por país que pueden referenciarse directamente desde `config.toml`.

## Archivos de salida (`outputs/<id>/`)

| Archivo | Contenido |
| --- | --- |
| `<id>_01_lec_curve.png` | Curva LEC empírica, banda bootstrap, curva híbrida y pérdidas históricas anuales |
| `<id>_02_lec_analytical_vs_simulated.png` | Curva LEC analítica vs. empírica de los eventos simulados |
| `<id>_03_catalogue_statistics.png` | Estadísticas de pérdidas anuales simuladas |
| `<id>_04_strategy_payouts_base.png` | Pagos de los instrumentos sobre el catálogo mostrado (base) |
| `<id>_05_drr_investment.png` | Inversión DRR por año (solo con DRR) |
| `<id>_06_reduced_lec_curves.png` | Familia de curvas LEC reducidas (solo con DRR) |
| `<id>_07_strategy_payouts_reduced.png` | Pagos de los instrumentos sobre el catálogo reducido (solo con DRR) |
| `<id>_08_horizon_loss_distribution.png` | Distribución de la pérdida acumulada en el horizonte |
| `<id>_09_cba_results.png` | Figura resumen del CBA (solo con CBA) |
| `<id>_report.txt` | Reporte principal de resultados |
| `<id>_statistics.csv` | Medianas de pérdida, cobertura, retención y no cubierto (total y fiscal) por escenario |
| `<id>_cba_report.txt` | Reporte completo del CBA (solo con CBA) |
| `<id>_config_used.toml` | Copia de la configuración utilizada |

---

# 📐 Descripción de módulos

## lec_core.py

**`compute_empirical_lec(event_loss_df, loss_scale_factor, freq_scale_factor, B, random_seed)`**
Calcula la curva LEC empírica a partir de un catálogo histórico de pérdidas. Aplica bootstrap (B réplicas) para estimar intervalos de confianza al 90 %. Devuelve un dict con la curva empírica, CIs (p05, p50, p95, mean), estadísticas globales (AAL, min, max, total) y la curva en formato `[[loss, rate], ...]`.

**`build_hybrid_lec(lec_curve, tail_loss, tail_aep)`**
Combina la curva empírica con una cola probabilística mediante blending log-log automático. Devuelve la curva híbrida y su AAL.

## simulation.py

**`generate_synthetic_catalogue(lec_curve, catalogue_length, simulation_number, random_seed)`**
Genera `simulation_number` catálogos sintéticos de longitud `catalogue_length` años mediante un proceso de Poisson homogéneo con muestreo inverso. Devuelve los catálogos, las pérdidas anuales agregadas y los streams de números aleatorios (CRN) necesarios para la reducción ex-ante.

## risk_management.py

**`apply_strategy(event_catalogue, drm_configs, catalogue_length)`**
Aplica una estrategia definida como lista de dicts sobre todos los catálogos sintéticos. Devuelve los DataFrames de pago por instrumento y la cobertura total. En la versión 3 la lista se construye automáticamente desde los bloques `[[strategy.instruments]]` de `config.toml`.

| Tipo | Instrumento | Parámetros de pago |
| --- | --- | --- |
| `insurance` | Seguro paramétrico (ej. CCRIF) | `attachment_point`, `exhaustion_point`, `ceding_percentage` |
| `ppo` | PPO de activación única por catálogo (se activa con el CCF) | calendario leído de `ppo_schedule_file` |
| `ccf` | CCF con techo acumulativo | `ccf_maximum`, `ccf_person`, `Pop_exposed` |
| `ddo` | DDO con umbral de activación y pago fijo | `ddo_threshold`, `ddo_available` |

> La cobertura total de todos los instrumentos está limitada automáticamente a no superar la pérdida bruta por evento (escala proporcional). Un `ppo` requiere un `ccf` en la misma estrategia.

## risk_reduction.py

**`compute_reduction_schedule(inv, rbc, hor, discount_rate)`**
Convierte los vectores anuales de inversión, relación beneficio-costo y horizonte de beneficios en la reducción acumulada de AAL vigente cada año.

**`generate_reduced_catalogue(base_lec_curve, red, N_events, U_times, U_loss, catalogue_length, simulation_number)`**
Calibra por bisección el desplazamiento de la curva LEC para cada valor de reducción y regenera el catálogo reutilizando los streams CRN del catálogo base, de modo que las diferencias se deban únicamente a la reducción del riesgo.

## cba/ (análisis costo-beneficio)

**`cba.engine.run_cba(losses_df, payout_dfs, drm_configs, cba_config, resp_fiscal, gap_thresholds, loss_basis)`**
Calcula, por simulación, los costos de cada instrumento (prima anual del seguro; comisiones y servicio de deuda a valor presente completo al desembolso para PPO, CCF y DDO), los beneficios (pagos × (1 + factor de beneficio indirecto)) y la pérdida no pagada, y los descuenta a la tasa social. Devuelve un objeto `CBAResults` con:

| Grupo | Indicadores |
| --- | --- |
| Centrales (`core`) | B/C esperado y su distribución, P(B/C > 1), pérdida no pagada esperada (VP) y sus percentiles, B/UL (complementariedad), B/C por instrumento |
| Eficiencia (`efficiency`) | Cost Multiple, Money Value y OMV por instrumento y agregados |
| CNC (`cnc`) | Ahorro neto esperado y mediano de la estrategia ex-ante frente a deuda comercial ex-post con spread endógeno por severidad del evento (% del PIB) |
| DRR (`drr`) | VP de la inversión, beneficio directo (reducción de pérdidas) e indirecto (reducción de la brecha), B/C directo e indirecto. Se calcula cuando DRR y CBA están activos |

La base de pérdidas del CBA (`cba.loss_basis`) puede ser la pérdida económica total o la pérdida fiscal (total × `resp_fiscal`). Los parámetros de costo por defecto de cada tipo de instrumento se definen en `[cba.defaults.*]` y pueden sobreescribirse instrumento por instrumento en su bloque `[[strategy.instruments]]`. Los parámetros del CNC (`[cba.cnc]`) están calibrados para Honduras y deben recalibrarse para otro país.

---

# 🧑‍🍳 Autores

El motor y la metodología de cálculo del LEC Tool es desarrollado por el **Disaster Risk Management Team** del **Banco Interamericano de Desarrollo**. La plataforma informática es desarrollada y mantenida por [GreenCode Software](https://www.greencodesoftware.com/).

Equipo de desarrolladores:
Andrés Abarca, Kenneth Otárola, Ginés Suárez, Víctor Cardenas

---

# 📑 Licencia

Copyright© 2025. Banco Interamericano de Desarrollo ("BID"). Uso autorizado [AM-331-A3](https://github.com/andresabarca-atlas/BID-LECTool/blob/main/LICENSE.md)

## Limitación de responsabilidades

El BID no será responsable, bajo circunstancia alguna, de daño ni indemnización, moral o patrimonial; directo o indirecto; accesorio o especial; o por vía de consecuencia, previsto o imprevisto, que pudiese surgir:

i. Bajo cualquier teoría de responsabilidad, ya sea por contrato, infracción de derechos de propiedad intelectual, negligencia o bajo cualquier otra teoría; y/o

ii. A raíz del uso de la Herramienta Digital, incluyendo, pero sin limitación de potenciales defectos en la Herramienta Digital, o la pérdida o inexactitud de los datos de cualquier tipo. Lo anterior incluye los gastos o daños asociados a fallas de comunicación y/o fallas de funcionamiento de computadoras, vinculados con la utilización de la Herramienta Digital.
