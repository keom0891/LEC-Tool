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

Un bloque por instrumento; el orden define el orden de apilado en las figuras. Sin bloques, la estrategia no tiene cobertura y el CBA se omite. Un `ppo` exige un `ccf` en la misma estrategia. Los nombres deben ser únicos.

Claves comunes:

| Clave | Tipo | Descripción |
| --- | --- | --- |
| `name` | texto | Etiqueta del instrumento en figuras y reportes |
| `type` | `"insurance"`, `"ppo"`, `"ccf"`, `"ddo"`, `"loan"` | Tipo de instrumento |

Parámetros de pago por tipo (obligatorios):

| Tipo | Claves | Descripción |
| --- | --- | --- |
| `insurance` | `attachment_point`, `exhaustion_point`, `ceding_percentage` | Capa [attachment, exhaustion] en $MM y proporción cedida (0-1) |
| `ppo` | (ninguna) | El calendario de disponibilidad se lee de `inputs.ppo_schedule_file` |
| `ccf` | `ccf_threshold`, `ccf_person`, `Pop_exposed` + parámetros de saldo y renovación | Umbral de emergencia ($MM), pago por persona afectada ($) y población expuesta |
| `ddo` | `ddo_threshold` + parámetros de saldo y renovación | Pérdida que activa el desembolso ($MM); cobertura limitada al saldo disponible |

Parámetros de costo opcionales (solo usados por el CBA; sobreescriben `[cba.defaults.<type>]` para ese instrumento):

| Tipo | Claves admitidas |
| --- | --- |
| `insurance` | `rate_on_line`, `premium` |
| `ppo` | `commitment_fee_rate`, `loan_interest_rate`, `repayment_years`, `front_end_fee_rate`, `credit_line` |
| `ccf` | `drawdown_fee_rate`, `loan_interest_rate`, `repayment_years`, `grace_period_years` |
| `ddo` | `loan_interest_rate`, `repayment_years` |

La clave antigua `interest_rate` no se acepta; la tasa contractual del préstamo es `loan_interest_rate` y la tasa social de descuento se define en `[cba]`.

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

| Clave | Tipo | Por defecto | Descripción |
| --- | --- | --- | --- |
| `enabled` | booleano | `true` | Calcula el CNC |
| `gdp` | número > 0 | obligatorio | PIB de referencia ($MM), denominador de la severidad del evento |
| `base_rate` | número en [0, 1] | obligatorio | Tasa base internacional (SOFR) |
| `sovereign_base_spread` | número en [0, 1] | obligatorio | Spread soberano en condiciones ordinarias |
| `ex_post_term_years` | entero ≥ 1 | `10` | Plazo de la deuda comercial ex-post |
| `spread_tiers` | lista de pares `[umbral, spread]` | valores del paquete | Spread adicional según pérdida del evento / PIB; el umbral del último tramo suele ser `inf` |

## Validaciones

`config_loader.py` detiene la corrida con un mensaje que indica la sección y la clave cuando: falta una clave obligatoria; un valor está fuera de rango; los vectores de la DRR o el calendario PPO no tienen `catalogue_length` valores; `displayed_catalogue` no es menor que `simulation_number`; un instrumento tiene tipo desconocido, nombre repetido o le falta un parámetro de pago; hay un `ppo` sin `ccf`; un archivo de entrada requerido no existe.


### Saldos y renovaciones de CCF, DDO y loan

Los CCF y DDO requieren `initial_balance`, `contracted_amount`,
`first_renewal_year`, `renewal_interval_years` y `max_replenishments_per_period`.
`first_renewal_year` es un año calendario. Si es anterior al inicio de la
simulación, se usa para ubicar la siguiente renovación del calendario,
conservando el saldo inicial hasta esa fecha. La renovación ocurre al inicio de ese año, antes de los eventos,
y se repite cada `renewal_interval_years` (CCF: 5; DDO: 3). Restaura el saldo a
`contracted_amount`, incluso con saldo parcial y sin desastres, y reinicia el
contador de reposiciones. Antes de la primera renovación se usa `initial_balance`.

Con `max_replenishments_per_period = 1`, al agotarse el saldo se permite una
reposición al monto contratado por período, incluido el período inicial.
Cada desastre solo puede utilizar el saldo disponible al inicio del evento.
Si lo agota, la reposición se realiza después del pago y puede utilizarse
desde el siguiente evento, incluso en el mismo año. Con saldo inicial cero,
el primer evento no recibe desembolso y la reposición queda disponible para
el siguiente, siempre que quede una reposición permitida.
Un segundo agotamiento debe esperar a la renovación automática. La reposición
no mueve las fechas de renovación. Se admite 0 para desactivar la reposición.

CCF activa con pérdida estrictamente mayor que
`max(ccf_threshold, 0.01 * Pop_exposed * ccf_person / 1e6)` y mantiene la fórmula
por personas afectadas. `contracted_amount` reemplaza el antiguo `ccf_maximum`.
DDO activa con pérdida estrictamente mayor que `ddo_threshold` y solicita
`min(pérdida, saldo disponible al inicio del evento)`, sin límite adicional por desastre.

`loan` requiere `loan_threshold`, `initial_balance`, `start_year` y `end_year`.
No se repone ni renueva. Los años de vigencia son inclusivos. En cada evento
elegible solicita `min(pérdida, saldo)`. Su costo se calcula con el modelo de
servicio de deuda existente del DDO; `loan_interest_rate` y `repayment_years`
pueden definirse por instrumento o en `[cba.defaults.loan]`. Si se omiten,
son 0.035 y 5 respectivamente; deben ajustarse al contrato real.

Todos los saldos descuentan únicamente desembolsos efectivos después de aplicar
el límite conjunto por desastre. Cada instrumento y simulación tiene su propio
saldo. Los archivos `<run_id>_balances_base.csv` y, con DRR,
`<run_id>_balances_reduced.csv` guardan los saldos al cierre de cada año calendario.
El año inicial sigue definido por `simulation.first_year` o, si se omite, el
próximo año según la regla original del código.
