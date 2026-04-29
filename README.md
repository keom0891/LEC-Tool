# LEC Tool v2: Cálculo de curvas de excedencia de pérdidas y evaluación de estrategias de gestión de riesgo

La herramienta **LEC Tool** consiste en una plataforma desarrollada por el **Banco Interamericano de Desarrollo** con el propósito de derivar curvas de excedencia de pérdidas (LEC) a partir de registros históricos de desastres. Esta plataforma está diseñada para estimar la tasa de excedencia anual asociada a valores específicos de pérdidas económicas. La curva LEC resultante se utiliza posteriormente en análisis de riesgo y en la toma de decisiones para la gestión de desastres, particularmente para la selección de estrategias de transferencia y/o reducción de riesgo.

![version](https://img.shields.io/badge/version-2.0.1-blue)

---

# ✨ Descripción

El motor de cálculo del LEC Tool está implementado en un conjunto de módulos Python desarrollados por el equipo de Gestión de Riesgos de Desastres del Banco Interamericano de Desarrollo. La arquitectura de la versión 2 separa la lógica de cálculo de la interfaz de usuario, facilitando la integración con plataformas web externas.

## Módulos

| Módulo | Descripción |
| --- | --- |
| `lec_core.py` | Cálculo de la curva LEC empírica, intervalos de confianza por bootstrap, y curva híbrida |
| `simulation.py` | Generación de catálogos sintéticos de pérdidas mediante simulación estocástica (Poisson + muestreo inverso) con Common Random Numbers |
| `risk_management.py` | Mecanismos de cobertura financiera: seguro paramétrico, PPO, CCF, DDO |
| `risk_reduction.py` | Modelado de reducción del riesgo ex-ante: calibración de desplazamiento de curva LEC y generación de catálogo reducido |
| `utils.py` | Funciones auxiliares compartidas entre módulos |
| `hybrid_lec.py` | Construcción de curva híbrida mediante blending log-log entre curva empírica y cola probabilística |
| `main.py` | Script de demostración completo: carga datos, ejecuta todos los módulos y genera visualizaciones |

---

# 📁 Estructura del repositorio

```
/
├── lec_core.py
├── simulation.py
├── risk_management.py
├── risk_reduction.py
├── utils.py
├── hybrid_lec.py
├── main.py
├── requirements.txt
├── .devcontainer/
│   └── devcontainer.json
└── data/
    ├── LEC_event_loss_example.csv
    ├── ppo_example.csv
    └── tail_curve.csv
```

---

# ⚙️ Prerrequisitos

Python 3.11 o superior con los siguientes paquetes:

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

## Ejecutar el script de demostración

```bash
python main.py
```

Este script ejecuta el flujo completo utilizando los datos de ejemplo incluidos en la carpeta `data/`:

1. Carga el catálogo histórico de pérdidas
2. Calcula la curva LEC empírica con intervalos de confianza
3. Construye la curva híbrida (opcional)
4. Genera catálogos sintéticos mediante simulación estocástica
5. Evalúa la Estrategia 1 de gestión del riesgo (CCRIF + PPO + CCF + DDO)
6. Aplica reducción del riesgo ex-ante y evalúa la estrategia sobre el catálogo reducido
7. Genera todos los gráficos de resultados

## Archivos de entrada

| Archivo | Columnas / formato | Descripción |
| --- | --- | --- |
| `LEC_event_loss_example.csv` | `year`, `econ_loss` | Catálogo histórico de pérdidas por evento ($MM) |
| `ppo_example.csv` | fila sin encabezado | `catalogue_length` valores de cobertura PPO disponible por año ($MM) |
| `tail_curve.csv` | `tail_loss`, `tail_aep` | Cola probabilística para la curva híbrida: pérdidas ($MM) y tasas de excedencia anuales |

El formato del catálogo de pérdidas se puede descargar [en este link](data/LEC_event_loss_example.csv).

El formato del inventario de puntos probabilistas se puede descargar [en este link](data/tail_curve.csv).

El formato del archivo PPO se puede descargar [en este link](data/ppo_example.csv).

---

# 📐 Descripción de módulos

## lec_core.py

```python
from lec_core import compute_empirical_lec, build_hybrid_lec
```

**`compute_empirical_lec(event_loss_df, loss_scale_factor, freq_scale_factor, B, random_seed)`**
Calcula la curva LEC empírica a partir de un catálogo histórico de pérdidas. Aplica bootstrap (B réplicas) para estimar intervalos de confianza al 90%. Devuelve un dict con la curva empírica, CIs (p05, p50, p95, mean), estadísticas globales (AAL, min, max, total) y la curva en formato `[[loss, rate], ...]`.

**`build_hybrid_lec(lec_result, tail_loss, tail_aep)`**
Combina la curva empírica con una cola probabilística mediante blending log-log. Devuelve la curva híbrida y su AAL.

## simulation.py

```python
from simulation import build_inv_cdf, make_random_streams, generate_synthetic_catalogue
```

**`generate_synthetic_catalogue(LEC_curve_df, catalogue_length, simulation_number, seed)`**
Genera `simulation_number` catálogos sintéticos de pérdidas de longitud `catalogue_length` años mediante un proceso de Poisson homogéneo con muestreo inverso. Devuelve los catálogos, DataFrames de pérdidas anuales agregadas, y los streams de números aleatorios (CRN) necesarios para la reducción ex-ante.

## risk_management.py

```python
from risk_management import apply_strategy
```

**`apply_strategy(event_catalogue, drm_configs, catalogue_length)`**
Aplica una estrategia de gestión del riesgo definida como lista de dicts sobre todos los catálogos sintéticos. Devuelve los DataFrames de pago por instrumento y la cobertura total. Cada dict debe incluir `'name'` (etiqueta para gráficos) y `'type'` (instrumento), además de los parámetros específicos del instrumento.

Instrumentos disponibles (`'type'`):

| Tipo | Instrumento | Parámetros clave |
| --- | --- | --- |
| `'insurance'` | Seguro paramétrico (ej. CCRIF) | `attachment_point`, `exhaustion_point`, `ceding_percentage` |
| `'ppo'` | PPO de activación única por catálogo | `ppo_schedule`, `ppo_loss_trigger` |
| `'ccf'` | CCF con techo acumulativo | `ccf_maximum`, `ccf_person`, `Pop_exposed` |
| `'ddo'` | DDO con umbral de activación y pago fijo | `ddo_threshold`, `ddo_available` |

> La cobertura total de todos los instrumentos está limitada automáticamente a no superar la pérdida bruta por evento (mediante escala proporcional).

Ejemplo de configuración:

```python
drm_configs = [
    {
        'name': 'CCRIF',
        'type': 'insurance',
        'attachment_point': 50,
        'exhaustion_point': 190,
        'ceding_percentage': 0.066,
    },
    {
        'name': 'BID PPO',
        'type': 'ppo',
        'ppo_schedule': [0, 2, 10, 25, 35, 42, 46, 46, 46, 46],
        'ppo_loss_trigger': 120,
    },
    {
        'name': 'BID CCF',
        'type': 'ccf',
        'ccf_maximum': 300,
        'ccf_person': 1650,
        'Pop_exposed': 10.83e6,
    },
    {
        'name': 'BID DDO',
        'type': 'ddo',
        'ddo_threshold': 120,   # $MM — pérdida mínima para activación
        'ddo_available': 110,   # $MM — pago fijo al activarse
    },
]
```

## risk_reduction.py

```python
from risk_reduction import generate_reduced_catalogue
```

**`calibrate_LEC_AAL(Ct, L, lam)`**
Calibra por bisección el parámetro alpha que desplaza la curva LEC para lograr exactamente `Ct` de reducción en AAL.

**`generate_reduced_catalogue(base_lec_curve, red, N_events, U_times, U_loss, catalogue_length, simulation_number)`**
Genera el catálogo reducido reutilizando los streams CRN del catálogo base para garantizar comparabilidad. `red` es una lista de reducción acumulada de AAL por año ($MM), definida directamente por el usuario.

---

# 🧑‍🍳 Autores

El motor y la metodología de cálculo del LEC Tool es desarrollado por el **Disaster Risk Management Team** del **Banco Interamericano de Desarrollo**. La plataforma informática es desarrollada y mantenida por [GreenCode Software](https://www.greencodesoftware.com/).

Equipo de desarrolladores:
Andrés Abarca, Kenneth Otárola, Ginés Suárez

---

# 📑 Licencia

Copyright© 2025. Banco Interamericano de Desarrollo ("BID"). Uso autorizado [AM-331-A3](https://github.com/andresabarca-atlas/BID-LECTool/blob/main/LICENSE.md)

## Limitación de responsabilidades

El BID no será responsable, bajo circunstancia alguna, de daño ni indemnización, moral o patrimonial; directo o indirecto; accesorio o especial; o por vía de consecuencia, previsto o imprevisto, que pudiese surgir:

i. Bajo cualquier teoría de responsabilidad, ya sea por contrato, infracción de derechos de propiedad intelectual, negligencia o bajo cualquier otra teoría; y/o

ii. A raíz del uso de la Herramienta Digital, incluyendo, pero sin limitación de potenciales defectos en la Herramienta Digital, o la pérdida o inexactitud de los datos de cualquier tipo. Lo anterior incluye los gastos o daños asociados a fallas de comunicación y/o fallas de funcionamiento de computadoras, vinculados con la utilización de la Herramienta Digital.
