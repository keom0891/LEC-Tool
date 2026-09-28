# Referencia de `config.toml`

Todos los insumos del LEC Tool se definen en `config.toml`, en la carpeta raíz del repositorio. La herramienta se ejecuta con `python main.py`; opcionalmente `python main.py otro_archivo.toml`.

Convenciones:

- Valores monetarios en millones de dólares ($MM); tasas como decimales (`0.05` = 5 %).
- Las rutas de archivos son relativas a la carpeta donde está `config.toml`.
- "Obligatorio" indica que la corrida se detiene con un mensaje de error si la clave falta. Las claves con valor por defecto pueden omitirse.
- TOML admite `inf` como número (se usa en `spread_tiers`).

## `[run]`

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `id` | texto | obligatorio | Identificador de la corrida. Nombra la subcarpeta `outputs/<id>/` y es el prefijo de todos los archivos. No admite `< > : " / \ \| ? *` |
| `output_dir` | texto | `"outputs"` | Carpeta raíz de resultados |
| `show_figures` | booleano | `false` | `true` abre además las figuras en ventanas (la corrida se detiene hasta cerrarlas) |
| `figure_dpi` | entero ≥ 30 | `130` | Resolución de las figuras PNG |

## `[inputs]`

| Clave | Tipo | Requerida cuando | Descripción |
| --- | --- | --- | --- |
| `event_loss_file` | ruta | siempre | CSV con columnas `year` (entero) y `econ_loss` ($MM) |
| `tail_curve_file` | ruta | `lec.hybrid_curve = true` | CSV con columnas `tail_loss` ($MM) y `tail_aep` (tasa anual de excedencia) |
| `ppo_schedule_file` | ruta | existe un instrumento `ppo` | CSV de una fila sin encabezado con `catalogue_length` valores ($MM disponibles por año) |

## `[lec]`

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `loss_scale_factor` | número > 0 | `1.0` | Multiplica todas las pérdidas (catálogo histórico y cola probabilística) |
| `freq_scale_factor` | número > 0 | `1.0` | Multiplica todas las tasas de excedencia |
| `hybrid_curve` | booleano | `true` | Combina la curva empírica con la cola probabilística |
| `bootstrap_samples` | entero ≥ 1 | `1000` | Réplicas bootstrap para la banda de confianza |
| `pml_return_periods` | lista de números ≥ 1 | `[200]` | Periodos de retorno (años) para los que se reporta el PML |
| `gap_threshold_fractions` | lista de números > 0 | `[0.25, 0.50, 0.75, 1.00]` | Umbrales de brecha de financiamiento como fracción de la pérdida máxima de la curva LEC |

## `[simulation]`

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `catalogue_length` | entero ≥ 1 | obligatorio | Años por catálogo sintético (horizonte) |
| `simulation_number` | entero ≥ 1 | obligatorio | Número de catálogos independientes |
| `random_seed` | entero | (aleatorio) | Semilla de reproducibilidad. Omitir la línea para una corrida no reproducible |
| `displayed_catalogue` | entero | `0` | Índice (0 a `simulation_number - 1`) del catálogo mostrado en las figuras por catálogo |
| `first_year` | entero | año actual + 1 | Año calendario del primer año simulado (solo etiquetas de ejes y reporte) |

## `[fiscal]`

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `resp_fiscal` | número en [0, 1] | obligatorio | Proporción de las pérdidas que es responsabilidad fiscal del gobierno |

## `[[strategy.instruments]]`

Un bloque por instrumento; el orden define el orden de apilado en las figuras. Sin bloques, la estrategia no tiene cobertura y el CBA se omite. Un `ppo` que se dispara con el CCF exige un `ccf` en la misma estrategia; con disparo propio (`ppo_loss_trigger` o `ppo_loss_trigger_rp`) no lo necesita. Los nombres deben ser únicos.

**Umbrales en dólares o en período de retorno (v8).** Todo umbral puede darse en $MM o como período de retorno de evento en años (claves que terminan en `_rp`). El período de retorno se convierte a dólares con la curva LEC del país que se corre (la híbrida, si está activa, ya escalada), en una etapa previa a la simulación. Así la misma estrategia tiene sentido para cualquier país de `Country databases`. De cada par se da una sola clave, nunca las dos.

Claves comunes:

| Clave | Tipo | Descripción |
| --- | --- | --- |
| `name` | texto | Etiqueta del instrumento en figuras y reportes |
| `type` | `"insurance"`, `"ppo"`, `"ccf"`, `"ddo"` | Tipo de instrumento |

Parámetros de pago por tipo (obligatorios):

| Tipo | Claves | Descripción |
| --- | --- | --- |
| `insurance` | `attachment_rp` o `attachment_point`; `exhaustion_rp` o `exhaustion_point`; `coverage_limit` o `ceding_percentage` | Capa en años (período de retorno de evento) o en $MM, y el límite de cobertura en $MM (la cesión se calcula) o la proporción cedida (0-1) |
| `ppo` | (ninguna) | El calendario de disponibilidad se lee de `inputs.ppo_schedule_file` |
| `ccf` | `ccf_maximum`, `ccf_person`, `Pop_exposed` | Techo total ($MM), pago por persona afectada ($) y población expuesta. Los valores del ejemplo son de Honduras y deben cambiarse para otro país |
| `ddo` | `ddo_threshold` o `ddo_threshold_rp`; `ddo_available` | Pérdida (o período de retorno) que activa el desembolso y pago fijo por activación ($MM) |

Opciones del seguro (por instrumento; si se omiten se toman de `[insurance_pricing]`):

| Clave | Tipo | Descripción |
| --- | --- | --- |
| `pricing` | `"rol_rule"`, `"market_curve"`, `"quote"`, `"fixed_rol"` | Cómo se fija la prima (ver `[insurance_pricing]`) |
| `gross_premium` | número > 0 | Prima bruta anual de una cotización ($MM). Implica `pricing = "quote"` |
| `donor_discount` / `donor_discount_share` | número ≥ 0 / número en [0, 1] | Parte de la prima pagada por un donante, en $MM o como proporción. El reporte da el B/C económico (prima bruta) y el fiscal (prima neta) |
| `payout_mode` | `"proportional"`, `"binary"` | Proporcional: el pago crece dentro de la capa (mecánica de CCRIF). Binaria: paga el límite al cruzar el umbral (simplificación didáctica) |
| `one_payout_per_year` | booleano | La póliza paga como máximo una vez por año. Es una aproximación conservadora del límite agregado anual de las pólizas soberanas; `false` permite un pago por evento |
| `payout_floor` | booleano | Una póliza activada paga al menos la prima bruta (sin pasar del límite) |

Cómo capturar una cotización de CCRIF: `attachment_rp` = "Punto de activación – período de retorno"; `exhaustion_rp` = "Límite de responsabilidad – período de retorno"; `coverage_limit` = "Límite de cobertura"; `gross_premium` = "Prima bruta"; `donor_discount` = "Descuento". No copie los montos en dólares del punto de activación ni del límite de responsabilidad: están en la escala del modelo de CCRIF, no en la del catálogo del país.

Opciones del PPO: `ppo_trigger_mode` (`"ccf"` por defecto o `"loss"`), `ppo_loss_trigger` ($MM) o `ppo_loss_trigger_rp` (años; implica `"loss"`), `ppo_require_available_funds` (booleano: si es `true`, el PPO no gasta su única activación en un año sin fondos disponibles).

Parámetros de costo opcionales (solo usados por el CBA; sobreescriben `[cba.defaults.<type>]` para ese instrumento):

| Tipo | Claves admitidas |
| --- | --- |
| `insurance` | `rate_on_line` (solo con `pricing = "fixed_rol"`), `premium` (equivale a `gross_premium`) |
| `ppo` | `commitment_fee_rate`, `loan_interest_rate`, `repayment_years`, `front_end_fee_rate`, `credit_line`, `grace_period_years` |
| `ccf` | `drawdown_fee_rate`, `loan_interest_rate`, `repayment_years`, `grace_period_years` |
| `ddo` | `loan_interest_rate`, `repayment_years`, `grace_period_years` |

La clave antigua `interest_rate` no se acepta; la tasa contractual del préstamo es `loan_interest_rate` y la tasa social de descuento se define en `[cba]`.

## `[insurance_pricing]` (v8)

Cómo se fija la prima de cada seguro que no trae una cotización. Todas las claves son opcionales.

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `method` | `"rol_rule"`, `"market_curve"`, `"fixed_rol"` | `"rol_rule"` | Método para todos los seguros sin cotización |
| `cutoff_rp` | número ≥ 1 | `10` | Años. En `rol_rule`, las rebanadas de la capa que se tocan con más frecuencia que 1 en `cutoff_rp` años se cotizan con la curva de mercado |
| `flat_rol` | número en (0, 1] | `0.05` | En `rol_rule`, ROL de las rebanadas más remotas que el corte |
| `rate_on_line` | número en (0, 1] | `[cba.defaults.insurance].rate_on_line` o `0.05` | ROL para `fixed_rol` |
| `market_curve_rp`, `market_curve_rol` | listas de igual longitud | tabla del corredor | Curva de mercado: ROL de una capa delgada según el período de retorno de evento en que se toca. `market_curve_rp` estrictamente creciente |
| `market_curve_date` | texto | `"2026-08"` | Fecha de la curva; aparece en el reporte. Los precios de reaseguro son volátiles y estacionales |
| `payout_mode`, `one_payout_per_year`, `payout_floor` | ver arriba | `"proportional"`, `true`, `true` | Reglas de pago por defecto |

Métodos:

- **`rol_rule`**: la capa se parte en rebanadas delgadas; cada una se cotiza en su propio período de retorno sobre la curva del país. Por debajo del corte, con la curva de mercado; por encima, con la tasa plana. Los valores por defecto son consistentes con la estructura de precios de los mecanismos soberanos de mancomunación de riesgo; se recomienda sustituirlos por la cotización del país cuando esté disponible. Con la capa por defecto (1 en 15 a 1 en 50 años) todas las rebanadas quedan por encima del corte y la regla equivale a la tasa plana; en capas angostas cercanas al corte la prima puede quedar por debajo del pago esperado, y el reporte lo advierte.
- **`market_curve`**: todas las rebanadas con la curva de mercado (reaseguro comercial).
- **`fixed_rol`**: prima = ROL × límite de cobertura (comportamiento anterior a la v8).
- **`quote`** (solo por instrumento): la prima bruta de una cotización real. Siempre tiene prioridad.

El reporte muestra, para cada seguro, la capa en años y en dólares, la fuente de la prima, el múltiplo implícito (prima / pago esperado) y avisa si la prima queda por debajo del pago esperado. Un seguro con precio de mercado tiene B/C económico menor que 1; el reporte lo explica y encabeza el bloque del seguro con indicadores de protección.

## `[risk_reduction]`

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `enabled` | booleano | `false` | Activa la reducción ex-ante del riesgo (DRR). Con `false` no se ejecuta ninguna parte de la DRR |
| `discount_rate` | número en [0, 1] | obligatorio si activa | Tasa para anualizar los beneficios de la inversión |
| `investment` | lista de `catalogue_length` números ≥ 0 | obligatorio si activa | Inversión por año ($MM) |
| `benefit_cost_ratio` | lista de `catalogue_length` números ≥ 0 | obligatorio si activa | Relación beneficio-costo de la inversión de cada año |
| `benefit_horizon` | lista de `catalogue_length` números ≥ 0 | obligatorio si activa | Años durante los que cada inversión genera beneficios |

## `[cba]`

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `enabled` | booleano | `false` | Activa el análisis costo-beneficio |
| `loss_basis` | `"total"` o `"fiscal"` | `"total"` | Pérdidas usadas por el CBA: económicas totales o multiplicadas por `resp_fiscal` |
| `social_discount_rate` | número en [0, 1] | `0.05` | Tasa social de descuento para valores presentes |
| `indirect_benefit_factor` | número ≥ 0 | `0.10` | Multiplicador de beneficio indirecto sobre los pagos |
| `omv_lambda` | número ≥ 0 | `0.05` | Parámetro de ajuste por riesgo del OMV |
| `legacy_truncation` | booleano | `false` | `true` reproduce el servicio de deuda año a año truncado al horizonte (solo validación) |

### `[cba.defaults.insurance]`, `[cba.defaults.ppo]`, `[cba.defaults.ccf]`, `[cba.defaults.ddo]`

Valores por defecto de los parámetros de costo de cada tipo de instrumento (mismas claves que la tabla de parámetros de costo opcionales). Si una tabla se omite, se usan los valores del paquete `cba` (calibración Honduras). Para `ccf`, `grace_period_years` debe ser menor que `repayment_years`.

### `[cba.cnc]`

Costo Neto Comparado. Si la tabla se omite, el CNC no se calcula.

> **Advertencia: estos parámetros deben ajustarse al país que se corre.** Los valores de `config.toml` son los de Honduras (PIB del Banco Mundial 2024, spread soberano de calificación B). El CNC clasifica cada evento por su tamaño como fracción del PIB para fijar el spread de la deuda ex-post; si se corre otro país con el PIB de Honduras, los eventos parecen más o menos severos de lo que son y el ahorro del CNC queda distorsionado. Antes de correr un país, actualice `gdp` y `sovereign_base_spread` (y `base_rate` si corresponde). El resto de los resultados no depende de esta sección.

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `enabled` | booleano | `true` | Calcula el CNC |
| `gdp` | número > 0 | obligatorio | PIB de referencia ($MM), denominador de la severidad del evento |
| `base_rate` | número en [0, 1] | obligatorio | Tasa base internacional (SOFR) |
| `sovereign_base_spread` | número en [0, 1] | obligatorio | Spread soberano en condiciones ordinarias |
| `ex_post_term_years` | entero ≥ 1 | `10` | Plazo de la deuda comercial ex-post |
| `spread_tiers` | lista de pares `[umbral, spread]` | valores del paquete | Spread adicional según pérdida del evento / PIB; el umbral del último tramo suele ser `inf` |

## Validaciones

`config_loader.py` detiene la corrida con un mensaje que indica la sección y la clave cuando: falta una clave obligatoria; un valor está fuera de rango; los vectores de la DRR o el calendario PPO no tienen `catalogue_length` valores; `displayed_catalogue` no es menor que `simulation_number`; un instrumento tiene tipo desconocido, nombre repetido o le falta un parámetro de pago; se dan a la vez un umbral en dólares y en período de retorno; `exhaustion_rp` no es mayor que `attachment_rp`; `pricing = "quote"` sin `gross_premium`; el descuento del donante es mayor que la prima; hay un `ppo` que se dispara con el CCF sin `ccf`; un archivo de entrada requerido no existe. En la etapa de la capa (después de calcular la curva) se detiene si el límite de cobertura es mayor que el ancho de la capa o si la capa queda vacía en la curva del país, y avisa (sin detenerse) si un período de retorno cae fuera de la curva.
