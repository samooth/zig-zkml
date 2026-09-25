# Soundness del binding de operandos — nota de diseño

> Documento de diseño, en español como el resto de la especificación.
> Cubre `libs/stark/expr.zig`, `root.zig`, `gemm_air.zig`, `gemm_chunk.zig`,
> `scale_air.zig`, `quant_binding.zig` y `chunk_binding.zig`.
> La especificación de sistema sigue en [BLUE_PRINT.md](../BLUE_PRINT.md);
> aquí está el *por qué*, con las mediciones que lo justifican.

Tres huecos de soundness de F2, los tres cerrados. Todos son el mismo tipo de
error: **el prover tenía más libertad de la que el AIR parecía negar**.

| Hueco | Estado | Mecanismo |
|---|---|---|
| Padding / dominio elegido por el prover | cerrado | `System.trace_rows` |
| Fila de cierre como operando falsificable | cerrado | `System.transition_exemptions` |
| Escala de dequantización fabricada | cerrado | `scale_air` (procedencia fp16) |

---

## 1. El dominio cíclico no perdona

El dominio de la traza es cíclico: la constraint compuesta
`s' = s + a·b` se evalúa **también en la wrap**, de la última fila a la
primera. Sumando sobre todo el dominio:

```
Σ_r (s[r+1] - s[r] - a[r]·b[r]) = 0  ⟹  Σ_r a[r]·b[r] = 0
```

Los términos de `s` telescopes porque el dominio cierra. Los productos no.
Consecuencia directa: **la suma de productos de todo el dominio es cero**, y
como las filas reales aportan `C`, la última fila tiene que aportar `−C`.

El diseño original cumplía eso con dos pins de boundary:

```
a[last] = 1
b[last] = -c[last]        # con s[0] = 0, cierra: 0 = C + 1·(-C)
```

Eso hace que la fila de cierre sea un par de operandos **sintético**. Y un
operando sintético no es un operando del modelo, lo que arrastra dos
consecuencias: el dominio (§1.1) y la propia fila de cierre (§2).

### 1.1 Padding: elegir un dominio mayor

Un prover podía elegir un dominio más grande que el que la statement
declaraba y esconder un producto en las filas extra. Ninguna constraint lo
rechazaba, porque las filas extra no eran "incorrectas": simplemente nadie
miraba.

La forma obvia de arreglarlo — un contador de filas reales, con un flag
`is_real` — **no funciona aquí**, y conviene entender por qué antes de
intentar de nuevo. Un contador de filas reales telescopes:

```
r' = r + is_real  ⟹  Σ_r is_real = 0
```

y como el número de filas reales es `k` y el campo tiene `p ≫ k`, eso obliga
a `k = 0`. Por eso el layout no tiene relleno: 1 MAC/fila exige
`k = 2^m − 1`, y el chunked exige `k = 16·(2^m − 1)`, y el ragged se rechaza.

Lo que sí cierra el dominio es hacer su longitud parte de la statement:

```zig
pub const System = struct {
    trace_rows: ?usize = null,   // null = este AIR no fija la forma
    ...
};
```

`Trace.validate` (prover), `Config.validate` y `verify` comparan contra ello, y
`replaceConstraints` lo preserva para que un binder no lo pierda al añadir sus
propias constraints. Con eso, un dominio distinto se rechaza aunque las
constraints sean satisfechas.

Los dos conjuntos de `k` válidos son disjuntos, así que el bench no compara la
misma reducción: informa la forma válida más cercana de cada layout y dice
cuál es.

---

## 2. La fila de cierre: por qué un flag no la salvaba

Con el dominio cerrado, el ataque que quedaba es sobre la fila de cierre. Si
el binding de operandos es **por fila** — y tiene que serlo, porque si fuese
por bloque el prover elegiría la escala y el nibble que le conviene — entonces
esa fila sintética también queda atada. Y no puede.

### 2.1 Por qué `-C` no puede ser una escala válida

La imagen de `fp16ToFixedQ4_22` es

```
{ ±(1024 + m)·2^s : m < 2^10, s ∈ [0, 15] }
```

y el binding exige que `b[last] = (nib − 8)·u` con `u` en esa imagen. Como la
fila de cierre lleva `nib = 9`, el factor es 1 y `b[last] = u`. Para un `C`
genérico:

- `b[last] = 0` no vale: `0` no está en la imagen (haría falta `m = −1024`).
- `b[last] = 1` no vale: los mínimos son 1024, y de hecho el pin `a[last]=1`
  del diseño original tampoco era una escala válida.
- Para un `C` arbitrario **no existe** representación de `−C` como producto
  de dos elementos de la imagen. Eso no es anti-sonido, es **witness
  infactible**: ni siquiera un prover honesto podría construir la fila.

### 2.2 Por qué las herramientas habituales no aplican

La respuesta obvia — "marca la fila y no le apliques el binding" — choca con
tres límites del IR, y conviene tenerlos escritos:

- Una constraint `composed` es una identidad sobre **todo** el dominio. Si se
  cumple en todas las filas, se cumple en la de cierre.
- Los scopes `boundary_first` / `boundary_last` sólo se evalúan en las filas 0
  y `last`, sobre valores autentidados. Sirven para *fijar* esas dos filas,
  no para eximir el binding de las `k` reales.
- Un flag `is_last` necesita **one-hotness**, y sobre un dominio cíclico eso no
  es derivable: cualquier recurrencia (suma o producto) que "llegue" a un
  valor tiene que poder volver a su punto de partida, lo que sólo admite
  patrones periódicamente consistentes. Además, con `degree = 2`,
  `flag·(cuadrática)` da grado 3, así que ni siquiera se podría escribir.

Quedaban dos salidas: el polinomio de grado `N` que identifica la última fila
(inabordable: hunde el `log_residual_degree` del FRI), o un lookup contra una
traza preprocesada — que es exactamente el plumbing multi-traza de F3.

### 2.3 La solución: transition exemptions

La respuesta correcta ya existía en el framework de referencia. Winterfell
expone `AirContext::set_num_transition_exemptions(n)`, con **k = 1 por
defecto**: el divisor de las constraints de transición es

```
z(x) = (x^n − 1) / Π_{i=1..k} (x − g^(n−i))
```

es decir, las compuestas **no se exigen en las últimas `k` filas del dominio**.
El ejemplo `vdf/exempt` de Winterfell existe justo para esto, y su comentario
lo dice: "the last two rows are excluded from transition constraints as we
populate values in the last row with garbage". Aquí `k = 1` basta porque la
fila de cierre es una.

El coste es de una línea en el cociente. El prover multiplica `P` por el
factor de exención antes de dividir:

```zig
// P·E = Q·Z_H  con  E(X) = Π (X − g^(n−1−i))
fn multiplyByExemptionFactor(coeffs: []Fp2, c: Fp2) void {
    var prev: Fp2 = Fp2.zero;
    for (coeffs) |*slot| {
        const orig = slot.*;              // leer ANTES de sobrescribir
        slot.* = prev.sub(c.mul(orig));  // out[i] = orig[i-1] − c·orig[i]
        prev = orig;
    }
}
```

y el verifier comprueba la misma identidad por el otro lado:

```zig
const e_value = exemptionFactorAt(system, config.log_trace, n, x);
if (!p_value.mul(e_value).eql(q_value.mul(z_h))) return false;
```

El buffer de la LDE ya tiene `lde_n` coeficientes con los de arriba a cero, y
multiplicar por `(X − c)` sólo desplaza datos hacia arriba, así que ninguna
fila de la LDE se sale y el cociente sigue cabiendo en el tamaño comprometido.

### 2.4 Lo que aporta, y lo que cuesta

`transition_exemptions` es un campo de `System` con **0 por defecto**: ningún
AIR existente cambia de comportamiento, y la exemption se pide explícitamente.
`gemm_air` y `gemm_chunk` la ponen a 1.

`Config.validate` comprueba que el bound del residual FRI cubra el grado extra.
Para grado 2 y `k = 1` el cociente queda en `rows − 1`, dentro del bound ya
existente, así que **las configs del GEMM no se mueven**. Con `k = 2` habría
que subir `log_residual_degree` *y* `log_final` (si no, el residual queda en
rate 1), y el test lo asserta.

Con la exención, los pins de la fila de cierre se reducen a uno:

```
s[0]     = 0
s[last]  = c[last]     # el claim se lee de la suma telescópica
```

`s[last]` es la suma de las filas reales —todas ellas compositionalmente
vigiladas, porque la wrap es justo la que está exenta— y el boundary ata el
claim a ella. Un prover que quiera mover la salida tiene que mover `s[last]`
con ella, y eso rompe la telescopia en una fila real.

Un efecto secundario que salió gratis: `gemm_chunk` pasó de **34 constraints a
3**. Los 30 pins `aᵢ[last] = 0`, `bᵢ[last] = 0` existían porque la fila de
cierre estaba en la ecuación; exenta, sus slots no tienen nada que satisfacer.

Y un efecto que **no** es gratis: con la wrap exenta, un claim falso ya no lo
detecta ninguna constraint compuesta. Por eso `prove` ahora evalúa sus propios
boundaries antes de emitir la prueba. Un prover no debería producir una prueba
que él mismo sabe inválida.

---

## 3. Procedencia de la escala

Con la fila de cierre resuelta, el binding por fila es total. Quedaba el
segundo hueco: la escala seguía siendo witness libre.

`quant_binding` probaba `a = (nib − 8)·u`, así que `u = 1234567` — un elemento
del campo que ningún fp16 produce — pasaba todas las constraints y demostraba
un output que ningún tensor Q4_K real genera. El test lo marcaba como
`KNOWN GAP` para que no se olvidara.

### 3.1 Por qué no el barrel shifter

El plan proponía un barrel shifter, y es la codificación natural: `·2^s` es un
shift variable, y `out = sel·a + (1−sel)·b` es grado 2. Pero medido antes de
decidir, con `width = 26` (el ancho real del significando desplazado) y
`amount_bits = 4`:

| | composed | columnas por escala |
|---|---|---|
| `barrel.build` (26 bits, 4 etapas) | **131** | **157** |
| mux one-hot | **21** | **29** |

El barrel paga 26 constraints de ancho por cada una de sus 4 etapas, y además
necesita 26 columnas de entrada y 4 del amount que debe colocar quien llama.
Contra el trace de 16 columnas que tenía el layout de 1 MAC/fila antes de esto,
sólo las columnas del gadget ya lo multiplicaban por 10; con las dos escalas,
por 20.

El plan estimaba "~100 constraints" para el barrel; la cifra real medida es
131, y la alternativa barata son 21.

El razonamiento de fondo: **el desplazamiento son 4 bits, así que seleccionar
sale más barato que desplazar**. No hace falta mover 26 bits a lo largo de 4
etapas; basta con elegir cuál de los 16 desplazamientos se aplica.

### 3.2 El gadget

`scale_air.zig` demuestra `u = ±(2^10 + m)·2^s` con `m < 2^10` y
`s ∈ [0, 15]`:

```
sel_s booleano,  Σ_s sel_s = 1        exactamente un s, luego M = 2^s
M = Σ_s 2^s·sel_s                    lineal
out = (2^10 + m)·M                   una cuadrática, 11 términos
scale = ±out                         una cuadrática
```

Son 1 + 16 + 1 + 1 + 1 = **21 constraints** y 10 + 16 + 1 + 1 + 1 = **29
columnas** por escala, sin gadget nuevo y sin traza preprocesada. El
`2^10` implícito viaja como constante en la ecuación de `out`, así que no
necesita columna propia.

El witness se calcula desde el **patrón fp16**, no desde el valor de campo. Por
eso `bindOperands` ahora toma `[]const u16` (los bits) en vez de
`[]const Goldilocks`: el AIR necesita mantisa, exponente y signo por separado,
y un `Goldilocks` tiraría esa procedencia antes de construir el witness.
`scaleFromFp16` sigue siendo el único sitio donde un patrón se convierte en el
elemento de campo que consume la ecuación de dequant.

Efecto sobre los tests: el `KNOWN GAP` pasó a negativo, y `bindOperands` ahora
rechaza con `BadScale` los fp16 que no son escala q4.22 utilizable —
subnormales, demasiado pequeños, demasiado grandes, inf y NaN.

### 3.3 Qué cierra y qué no

- **Cierra**: la escala está en la imagen de `fp16ToFixedQ4_22`. Ningún
  elemento de campo fabricado pasa, y todo fp16 válido tiene witness.
- **No cierra**: *cuál* de los 65536 fp16 posibles usó el modelo. Atar el
  patrón a los bytes del fichero de pesos es el commitment de pesos, que es
  trabajo de F3 (public inputs). El gadget hace que la escala sea un valor
  *dequantizable*, no uno arbitrario.

### 3.4 El coste, medido

`zig build bench -- --k 256 --repeat 1`:

| layout | columnas | composed | prove | verify | proof |
|---|---|---|---|---|---|
| 1 MAC/fila (k=255) | 16 → **74** | 13 → **55** | 10.8 → **64.7 ms** | 0.37 → **0.85 ms** | 12.0 → **28.3 KiB** |
| 16 MACs/fila (k=240) | 226 → **1346** → **254** | 193 → **865** → **235** | 16.6 → **356** → **18.6 ms** | 3.20 → **15.7** → **2.5 ms** | 66.1 → **381** → **74.0 KiB** |

La fila del chunked lleva las tres cifras porque las dos últimas columnas del
gadget por chunk (un gadget por fila, §3.5) se escribieron después de esta
tabla.

### 3.5 Una escala por fila, no por operación

La primera versión corría el gadget 32 veces por fila, uno por cada par
operando, y costaba 1346 columnas. No hacía falta, y el motivo es aritmético
y no una asunción sobre los pesos.

Un chunk cubre los MACs `[16r, 16r+16)` y un bloque de `B` elementos cubre
`[Bj, Bj+B)`. El chunk cruza un límite sólo si algún `Bj` cae estrictamente
dentro del intervalo, y como `B` es múltiplo de 16, todos los `Bj` lo son
también. Comprobado para `B ∈ {32, 64, 128, 256}` — el bloque Q4_K real y
los sub-bloques de GGML: cero cruces. Los 16 operandos de un chunk comparten
bloque, luego comparten escala.

Compartir la **columna** de escala es además la forma sound de decirlo. El
prover no puede dar escalas distintas a los 16 slots, porque sólo hay una
columna donde meterlas; no hacen falta constraints de igualdad y no se
añaden. `bindOperands` sigue tomando el patrón fp16 por MAC — es lo que da el
modelo — y rechaza un chunk cuyas escalas no sean constantes, en vez de
promediarlo en silencio y atestiguar la escala equivocada para los slots que
discrepan.

**Medido, mismo bench (`--k 256 --repeat 1`):**

| layout | columnas | composed | prove | verify | proof |
|---|---|---|---|---|---|
| 1 MAC/fila (k=255) | 74 | 55 | 64.7 ms | 0.85 ms | 28.3 KiB |
| 16 MACs/fila (k=240) | **254** (era 1346) | **235** (era 865) | **18.6 ms** (era 356) | **2.5 ms** (era 15.7) | **74.0 KiB** (era 381) |

El chunked recupera la ventaja que tenía antes de la procedencia: prueba un
0.29× lo que la de 1 MAC/fila, frente al 4.8× en contra que costaba el gadget
por slot. Por MAC: 77.5 µs contra 253.8 µs.

Lo que queda sin cerrar: el layout de 1 MAC/fila sigue pagando un gadget por
fila porque **ahí sí** cada fila es un MAC distinto y puede caer en un bloque
distinto, así que no hay nada que compartir. Bajarlo requeriría cambiar ese
layout, no el binding.

---

## 4. Trampas de implementación

Tres que costaron tiempo y conviene no volver a pisar:

**Un puntero colgante silencioso.** `air_builder.freeze` devuelve un `Owned`
cuyos `Term`/`Factor` son los que los `Constraint` apuntan. Copiar los
`Constraint` a un slice nuevo y dejar que el `Owned` muera deja el sistema
apuntando a memoria liberada: los tests crashean con `ABRT` en `prove`, no con
un error legible. `quant_binding.BoundSystem` y `chunk_binding.BoundSystem`
llevan el `Owned` como campo por eso, igual que `range.BuiltSystem` lleva sus
`checks`. Si se añade un tercer binder, la misma regla.

**La multiplicación in-place se pisa a sí misma.**
`multiplyByExemptionFactor` lee `coeffs[i-1]`, que en la iteración anterior ya
es el valor *nuevo*. El síntoma es `ConstraintViolation` en un prover que está
mintiendo con descaro: el polinomio multiplicado no es divisible por `Z_H`.
Guardar el original en una variable antes de sobrescribir.

**Un flag no es un boundary.** Al escribir los tests de la exención, el primer
intento fue comprobar que `b[last]` distinto de `1` se rechaza. Ya no aplica:
la fila es exenta y *cualquier* valor verifica. Lo que no es libre es el claim,
y eso lo atrapa `s[last] = c[last]`. Los tests de
`gemm_test.zig` y `gemm_chunk_test.zig` cubren las dos mitades por separado.

---

## 5. Cómo verificarlo

```sh
zig build fmt
zig build test --summary all        # 213 tests
zig build verify --summary all      # 14/14 pasos, ABI + Python
zig build -Doptimize=ReleaseFast test --summary all
zig build spike                     # auditoría FRI 3/3
zig build bench --summary all -- --k 256 --repeat 1
```

Tests que cubren específicamente esta nota:

| test | qué asserta |
|---|---|
| `stark: a transition exemption waives exactly the last row` | la wrap puede violarse; una fila antes, no |
| `stark: an exemption the FRI residual bound cannot cover is refused` | `k=1` cabe en el bound, `k=2` no |
| `gemm: the boundary constraints are what make the AIR non-vacuous` | la fila exenta es libre; el claim y las filas reales no |
| `chunk: a product parked in an unused closing slot cannot move the claim` | el ataque que los 30 pins paraban |
| `chunk: the exemption, not a wall of pins, is what closes the row` | quitar la exemption reabre el ataque |
| `quant: a fabricated scale no longer proves` | el `KNOWN GAP`, invertido |
| `quant: tampering with a scale's provenance witness is rejected` | el AIR, no sólo el binder |
| `chunk_binding: a scale's provenance is checked per slot, not just once` | las dos familias de operandos, cada una con su gadget |
| `chunk_binding: a chunk whose scales are not constant is refused` | el binder rechaza lo que el AIR no puede atestiguar |
| `chunk_binding: one scale column per row, shared by all 16 slots` | la compartición es estructural, no una igualdad |
