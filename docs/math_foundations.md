# Fundamentos matemáticos

## Módulo 1 — Pronóstico de demanda (Persona 2)

> Plan de trabajo y decisiones de diseño: `docs/plan_persona2.md`. Notación: `y` pasajeros observados, `μ` media pronosticada, `q` cuantil P90, `ℓ` nivel local (media de los 28 días previos al origen), `H = 30` días de horizonte.

### 1. Parametrización del modelo y pérdida para conteos

**Variable objetivo.** `y_{r,t}` es el número de pasajeros que abordan en el origen de la ruta `r` el día `t`: un conteo no negativo. Para un origen de pronóstico `o` se define el nivel local `ℓ_{r,o}` como la media de `y` en los 28 días anteriores a `o` (solo datos ya conocidos en `o`), que entra como *offset*:

$$\mu_{r,o,k} = \ell_{r,o}\,e^{f_k(x)}, \qquad k = 0,\dots,29$$

donde `f_k(x)` es la salida de la red (o la combinación lineal del GLM) para el día `o+k` y `x` reúne los rezagos y el calendario. El offset hace que la red aprenda la forma relativa a la escala de cada ruta, y deja de lado la recuperación pospandemia (2021 estuvo en 0,64 × el nivel de 2023).

**Pérdida de Poisson.** Si `y ∼ Poisson(μ)`, el logaritmo negativo de la verosimilitud es `μ − y·log μ + log y!`. Con `log μ = log ℓ + f` y descartando el término constante `log y!`, y aplicada sobre la escala relativa `y/ℓ`:

$$L_{\text{Poisson}} = \frac{1}{|\mathcal{V}|}\sum_{(n,k)\in\mathcal{V}}\Big(e^{f_{n,k}} - \tfrac{y_{n,k}}{\ell_n}\,f_{n,k}\Big)$$

que es exactamente `nn.PoissonNLLLoss(log_input=True)` con entrada `f` (el logaritmo de la media, no la media) y objetivo `y/ℓ`. Equivale a la verosimilitud de Poisson de `y` con cada ventana pesada por `1/ℓ`, para que las 10 rutas pesen parecido (el MASE promedia por ruta). `𝒱` son los días con máscara 1 (observados, no imputados y fuera del Paro Nacional).

**¿Por qué Poisson si hay sobredispersión?** La razón varianza/media por celda ruta × día de la semana × mes en entrenamiento tiene mediana **86** (mediana entre rutas del percentil 25 y del 75 de cada ruta: 43 y 164) y la dispersión de Pearson del GLM, que ya descuenta calendario y nivel, tiene mediana **39** (con 1 como Poisson puro; `reports/tables/m1_sobredispersion.csv`). Es decir, la varianza es de decenas de veces la media. Pero la pérdida de Poisson es una *cuasi-verosimilitud*: su condición de primer orden es `Σ (y − μ)·∂ log μ / ∂θ = 0`, que solo exige que la media esté bien especificada, no la varianza (Wedderburn, 1974; McCullagh y Nelder, 1989). Por eso la usamos para estimar la media aunque `Var(y) ≠ μ`.

**Consecuencia para el cuantil.** Con esa dispersión, derivar el P90 de una Poisson (cuya desviación estándar sería `√μ`, p. ej. ≈ 50 para `μ = 2 500`) lo subestimaría gravemente. Por eso el P90 se estima con una cabeza propia entrenada con pérdida cuantílica, sin suponer ninguna distribución:

$$q = \mu\,e^{\operatorname{softplus}(g)}\ \ge\ \mu, \qquad L_\tau(y,q)=\max\{\tau(y-q),\,(\tau-1)(y-q)\},\ \tau=0{,}9$$

$$L = L_{\text{Poisson}} + \lambda\,L_{\tau}$$

La parametrización `q = μ·exp(softplus(g))` garantiza `q ≥ μ` (no hay cruce entre media y cuantil). La pérdida cuantílica es consistente para el cuantil condicional (Koenker y Bassett, 1978): al minimizarla sobre una constante se obtiene el cuantil empírico (verificado en `tests/test_m1_perdidas.py`), y con `τ = 0,9` quedarse corto cuesta 9 veces más que pasarse en la misma cantidad. Si se invirtiera el signo de `(y − q)` se estimaría el P10, no el P90.

**Binomial negativa (ablación opcional).** La distribución NB2 tiene `Var(y) = μ + αμ²`, que captura la sobredispersión; su pérdida está en `m1_demanda/losses.py::nb2_nll` y coincide con `scipy.stats.nbinom` (`n = 1/α`, `p = 1/(1+αμ)`). En `torch.distributions.NegativeBinomial`, `probs` es la probabilidad de éxito y la media es `total_count·probs/(1−probs)` (comprobado en las pruebas). No se usa como pérdida principal porque exige una cabeza extra para `α`, mayor cuidado numérico y, para el cuantil, una distribución bien especificada.

**Referencias de esta sección** (por verificar su cita completa antes de la entrega; ver `docs/references.md`): Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica, 46*(1), 33–50. Wedderburn, R. W. M. (1974). Quasi-likelihood functions, generalized linear models, and the Gauss–Newton method. *Biometrika, 61*(3), 439–447. McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2.ª ed.). Chapman & Hall.

### 2. Métricas de evaluación

Sean `y_{n,k}` y `ŷ_{n,k}` el observado y el pronóstico del día `k = 1,…,30` de la ventana `n` (un origen y una ruta), y `𝒱` el conjunto de pares con máscara 1 (observados, no imputados, fuera del Paro Nacional). Todas las métricas se calculan solo sobre `𝒱` (nunca sobre valores imputados) y se reportan también por subconjunto: días de temporada alta y días base.

* **MAE** `= |𝒱|⁻¹ Σ |y − ŷ|` y **RMSE** `= √(|𝒱|⁻¹ Σ (y − ŷ)²)`, en pasajeros.
* **MASE** (Hyndman y Koehler, 2006). Para la ruta `r`, `MASE_r = MAE_r / s_r`, donde la escala `s_r` es el MAE del ingenuo estacional de periodo `m = 7` **dentro de la partición de entrenamiento**: `s_r = (T−m)⁻¹ Σ_t |y_{r,t} − y_{r,t−m}|`, sin días imputados ni del Paro Nacional. Un MASE de 1 significa "igual de bueno que el ingenuo un paso adelante en entrenamiento"; el MASE total es el **promedio de los MASE por ruta** (cada ruta pesa igual). Como la escala usa solo el entrenamiento, no depende de la partición que se evalúa.
* **Error por horizonte**: MAE y MASE para cada `k`.
* **Cobertura del P90** `= |𝒱|⁻¹ Σ 𝟙{y ≤ q}`; la meta es 0,90. Una cobertura menor indica un cuantil demasiado bajo.
* **Pérdida cuantílica** `L_τ` con `τ = 0,9` (sección 1), en pasajeros y escalada por `s_r` para promediar entre rutas.

Los 10 orígenes de prueba se solapan (un origen cada 7 días con horizonte de 30), así que **no son observaciones independientes**: no se deben usar como muestra independiente en pruebas estadísticas.

**Referencia**: Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688. (Cita por verificar contra la fuente antes de la entrega.)

### 3. Fórmulas de flota y conductores

Para la ruta `r` con regreso `r′` del corredor `c` y el día `t`, sea `D_{r,t}` la demanda de planeación (la media `μ` o el cuantil `q`):

$$C_r=\sum_k w_{r,k}\,\mathrm{cap}_k,\qquad N_{r,t}=\Big\lceil \frac{D_{r,t}}{\varphi\,C_r}\Big\rceil,\qquad R_{c,t}=\max\big(N_{r,t},\,N_{r',t}\big)$$

`C_r` es la capacidad media ponderada por la participación `w_{r,k}` de cada clase de vehículo en los despachos del entrenamiento; `φ` es la ocupación objetivo. **`R` es el máximo y no la suma de los dos sentidos**: un vehículo que sale debe volver, así que cada ciclo despacha un vehículo en cada sentido. Por eso los despachos ejecutados son `2R` y los del sentido más liviano que exceden lo requerido (`R − N`) se reportan como subutilizados y se cobran.

$$v_c=\Big\lfloor \frac{H_{op}}{T_c}\Big\rfloor,\quad T_c=2\,t_{viaje}+2\,t_{alist},\qquad V_{c,t}=\Big\lceil \frac{R_{c,t}}{v_c}\Big\rceil,\qquad F_c=\Big\lceil (1+\rho)\max_t V_{c,t}\Big\rceil$$

Conductores, con `H_{c,t}=R_{c,t}\cdot 2\,t_{viaje}` horas de conducción al día, `h_d` el máximo de conducción diaria, `h_s` la jornada semanal vigente el último día del horizonte (Ley 2101 de 2021), `n` los días del horizonte y `a` el ausentismo:

$$CD_{c,t}=\Big\lceil \frac{H_{c,t}}{h_d}\Big\rceil,\qquad K_c=\Big\lceil \frac{\max\!\big(\max_t CD_{c,t},\ \sum_t H_{c,t}/(h_s\,n/7)\big)}{1-a}\Big\rceil$$

El primer término de la plantilla cubre el día pico y el segundo que el total de horas del horizonte quepa en la jornada semanal. Pasajeros no atendidos contra la demanda real: `U_{r,t}=\max(0,\,y_{r,t}-R_{c,t}\,C_r)`. Limitaciones: la jornada semanal incluye tareas que aquí no se modelan, y no se programan turnos, descansos semanales ni pernocta.

### 4. Por qué un cuantil alto: el vendedor de periódicos

Si planear de menos cuesta `c_u` por pasajero sin atender y planear de más cuesta `c_o` por pasajero de capacidad ociosa, la decisión de capacidad que minimiza el costo esperado es el cuantil
$$\tau^*=\frac{c_u}{c_u+c_o}$$
de la distribución de la demanda (modelo del vendedor de periódicos; ver Hyndman y Athanasopoulos, 2021, para el uso de cuantiles en la toma de decisiones). Planear con el P90 es óptimo solo si `τ* = 0,9`, es decir si `c_u = 9 c_o`.

**Con los costos supuestos de `configs/m1_operacion.yaml`** el cálculo da `τ* ≈ 0,72`: con `c_u = 30 000` COP por pasajero no atendido y `c_o = c_despacho/(φ·C̄) = 180 000/(0,75 · 20,3) ≈ 11 800` COP por pasajero de capacidad extra (`C̄` = capacidad media entre rutas), `τ* = 30 000/(30 000 + 11 800) ≈ 0,72`. Eso es coherente con lo que mide la simulación del valor del pronóstico (`reports/tables/m1_valor_pronostico.csv`): con estos costos planear con la media del GLM cuesta menos que planear con su P90, aunque el P90 deja menos pasajeros sin atender. **La conclusión depende de los costos, que son supuestos**: si perder un pasajero costara cerca de 9 veces una silla ociosa (por ejemplo por pérdida de clientes o sanciones), el P90 sería la decisión correcta. Por eso el análisis incluye el barrido sobre `c_no_atendido` y la sensibilidad de ±10 %.

**Referencias de esta sección** (por verificar su cita completa antes de la entrega): Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/

### 5. Arquitecturas utilizadas: ecuaciones

Notación: `σ` sigmoide, `tanh` tangente hiperbólica, `⊙` producto elemento a elemento, `x_t ∈ ℝ^{1+F}` la entrada del día `t` (el valor relativo `log((y_t+1)/(ℓ+1))` y el calendario de ese día) y `h_t` el estado oculto.

**Perceptrón multicapa (MLP).** Con la entrada aplanada `u = [x_{o−L..o−1}, cal_{o..o+H−1}, e_r]` (historial, calendario futuro y embedding de la ruta `e_r = E[r]`), `h⁽⁰⁾ = u`, `h⁽ˡ⁾ = Dropout(ReLU(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾))` para `l = 1,…,m`, con `ReLU(a) = max(0, a)`. Las dos cabezas son lineales: `f = W_f h⁽ᵐ⁾ + b_f` y `g = W_g h⁽ᵐ⁾ + b_g`, ambas de dimensión `H = 30`.

**GRU** (Cho et al., 2014), tal como la implementa PyTorch (`nn.GRU`):

$$r_t=\sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}),\qquad z_t=\sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz})$$
$$n_t=\tanh\!\big(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})\big),\qquad h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}$$

La puerta de reinicio `r_t` multiplica **después** de la matriz recurrente `W_{hn}h_{t−1}+b_{hn}`. Algunas presentaciones la aplican antes, `W_{hn}(r_t⊙h_{t−1})`; esa variante **no** coincide con `nn.GRU`: se verificó numéricamente con un paso de ambas formas (la propia ecuación escrita aquí reproduce la salida de PyTorch con tolerancia 10⁻⁶ y la variante no).

**LSTM** (Hochreiter y Schmidhuber, 1997), orden de puertas de `nn.LSTM` (`i, f, g, o`):

$$i_t=\sigma(\cdot),\ f_t=\sigma(\cdot),\ g_t=\tanh(\cdot),\ o_t=\sigma(\cdot)\ \ \text{con}\ (\cdot)=W x_t+U h_{t-1}+b$$
$$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)$$

También verificado paso a paso contra `nn.LSTM`.

**Modelo recurrente completo.** El codificador recorre el historial `x_{o−L},…,x_{o−1}` con sus calendarios y se toma el estado final de la última capa, `s = h_{o−1}`. Se concatena con el calendario futuro y el embedding, `z = [s, cal_{o..o+H−1}, e_r]`, pasa por una capa densa con ReLU y las mismas dos cabezas `f` y `g`. El calendario futuro **no** entra al codificador, así que no puede alterar el estado del pasado (probado en `tests/test_m1_rnn.py`). Número de parámetros de una capa: GRU `3(d·h + h² + 2h)` y LSTM `4(d·h + h² + 2h)`, con `d = 1+F` y `h` el tamaño oculto (verificado contra `sum(p.numel())`).

### 6. Derivación de las pérdidas a partir de un modelo probabilístico

**Poisson.** Se supone `y_{n,k} | x ∼ Poisson(μ_{n,k})` con `log μ_{n,k} = log ℓ_n + f_{n,k}`. La verosimilitud de una observación es `μ^y e^{−μ} / y!`, así que
$$-\log p(y\mid\mu)=\mu-y\log\mu+\log y!=\ell\,e^{f}-y\,(\log\ell+f)+\log y!$$
Los términos `y log ℓ` y `log y!` no dependen de los parámetros de la red, de modo que minimizar la suma de `−log p` equivale a minimizar `Σ (ℓ e^{f} − y f)`. Dividiendo cada ventana por `ℓ_n` (peso `1/ℓ_n`) se obtiene `Σ (e^{f} − (y/ℓ) f)`, la pérdida usada. La condición de primer orden respecto a `f` es `ℓ e^{f} − y = 0`, esto es `μ = y`: el estimador iguala la media, y ese resultado solo exige que la media esté bien especificada (cuasi-verosimilitud), no que la varianza sea igual a la media.

**Pérdida cuantílica (pinball).** Sea `F` la función de distribución de `y` dado `x` y `q_τ = F⁻¹(τ)`. Se busca `q` que minimice `E[L_τ(y,q)]` con `L_τ(y,q) = τ(y−q)₊ + (1−τ)(q−y)₊`. Derivando bajo la esperanza,
$$\frac{d}{dq}E[L_\tau]=-\tau\,P(y>q)+(1-\tau)\,P(y\le q)=F(q)-\tau,$$
que se anula en `F(q) = τ`: el minimizador es el cuantil `τ`. Equivalentemente, es el estimador de máxima verosimilitud de `q` cuando los errores siguen una distribución de Laplace asimétrica con parámetro de asimetría `τ`, cuya log-verosimilitud negativa es proporcional a `L_τ(y,q)`. Con `τ = 0,9`, `L_τ` castiga 9 veces más quedarse corto (`y > q`) que pasarse en la misma cantidad. La forma de la pérdida no supone ninguna distribución de `y`, por eso sirve para estimar el P90 aunque el conteo esté sobredispersado.

**Referencias de esta sección** (por verificar su cita completa antes de la entrega): Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP 2014*. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. Koenker, R., & Bassett, G. (1978), ya citada arriba.
