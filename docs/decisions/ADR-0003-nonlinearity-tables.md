# ADR-0003 — Las tablas de no-linealidad son la especificación, y se atan por hash

- **Estado:** aceptado
- **Fecha:** septiembre 2026
- **Ámbito:** soundness de las activaciones (§4.5)
- **Vigente en:** `libs/gadgets/nonlin/root.zig`, `libs/statement/root.zig` (version 2), `libs/gadgets/norm/root.zig`
- **Relacionado:** [ADR-0001](ADR-0001-zkml-vs-blueprint.md) decisión 1 (dual-path)

## Contexto

`§3.1` del blueprint hace el `QuantScheme` normativo y define las no-linealidades
**por tabla**, con el kernel en modo recorded ejecutando *la misma tabla* que
usa el circuito. Hasta ahora eso era una intención: la tabla de SiLU era una
constante de compilación y **nada en la statement la ataba**.

Dos consecuencias, y las dos son fallos de soundness:

1. **La tabla no era una especificación.** Un prover podía usar otra tabla de
   SiLU, construir una traza coherente con ella, y la prueba verificaría una
   función que el modelo no usa. La afirmación "el motor calculó esto" sería
   falsa y verificaría.
2. **Nadie lo notaba porque nadie la llamaba.** En `zkml.zig` los gadgets se
   referencian como `&gadgets.nonlin.SiLULookup.airFragment`, que toma la
   dirección: eso analiza el cuerpo lo suficiente para dar error de compilación,
   pero no ejecuta nada. Un gadget que nadie invoca no falla nunca.

## La trampa descartada: atar el motor

La lectura inmediata de §4.5 era "fijar engine + versión + variante de kernel".
Se verificó contra `llama.cpp@1c3c9674d` y esa vía es un callejón:

```c
// ggml/src/ggml-cpu/vec.cpp:380
void ggml_vec_silu_f32(const int n, float * y, const float * x) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    ggml_v_silu(...)            // x / (1 + ggml_v_expf(-x))
#elif defined(__AVX2__) && defined(__FMA__)
    ...
#elif defined(__SSE2__) ... SVE ... NEON ... RISCV ...
#endif
    for (; i < n; ++i) y[i] = ggml_silu_f32(x[i]);
}
```

`ggml_v_expf` es una aproximación polinómica **distinta en cada una de las seis
variantes**. El resultado difiere en el último bit según la CPU. Un witness
grabado en AVX2 no verifica contra un AIR construido para AVX512.

Atar la variante convierte el repositorio en un producto por máquina: el mismo
modelo daría pruebas distintas según dónde se ejecute, o directamente no
verificaría. No es caro, es inviable.

## Decisión

**La tabla canónica ES la especificación, y su digest es public input.**

1. `gadgets.nonlin` define SiLU, GELU (tanh) y rsqrt sobre la rejilla q8.8 de
   un int8, cada una con su regla de redondeo explícita. No pretende aproximar
   la "SiLU real" de ningún motor: sobre un dominio entero q8.8 no existe tal
   cosa, y una prueba necesita una función total.
2. El AIR la verifica con LogUp, dos lookups de 256 entradas en lugar de una de
   65536.
3. El kernel en modo recorded ejecuta esa misma tabla, no la del engine. El
   fast path nativo queda sin verificar por construcción, sin coste en el camino
   normal. Es exactamente el contrato dual-path de [ADR-0001](ADR-0001-zkml-vs-blueprint.md)
   decisión 1.
4. `nonlin.digest()` —Blake3 con dominio por tabla y longitudes absorbidas— se
   serializa dentro de la statement, que sube a `version = 2`. Un prover con
   otra tabla está afirmando otra statement y no puede producir una prueba para
   la original.
5. `nonlinearity_version` sube junto con cualquier cambio de tabla, y ese cambio
   es rompedor para toda prueba previa.

## Consecuencias

- **El tests de digest alterado es la pieza que importa.** Alterar una sola
  entrada de la tabla por un ulp debe mover el digest; si no lo moviera, la
  statement no estaría atando nada. Eso se verifica explícitamente.
- **La statement sube a v2.** Las statements v1 no se aceptan: en v1 las
  activaciones no estaban atadas, así que una prueba v1 no dice qué SiLU corrió.
- **RMSNorm tiene referencia, no AIR.** `libs/stark/rmsnorm_ref.zig` es la
  transcripción directa de la aritmética, sin sistema de constraints, y es lo
  que un AIR futuro se prueba. Sus stubs anteriores emitían una constraint
  `degree = 1` sin expresión, que es una falsificación y no un gadget; ahora
  `airFragment` falla a compilar con los requisitos en la cabecera.
- **La tabla `rsqrt` tuvo dos errores de diseño que la referencia cazó**, y
  ninguno lo habría visto un test de forma: el dominio cubría `1/sqrt(v)` para
  `v` en `[0, 255]` cuando la media de cuadrados de un int8 llega a 16129, y
  la corrección `2^(-shift/2)` dividía y no multiplicaba. Un AIR sobre esa tabla
  habría probado una normalización 8× débil. El shift por fila es lo que lo
  resuelve, con la condición de que sea par.
- El native path de cualquier engine sigue sin verificarse, y ahora está
  documentado como tal en vez de insinuado.

## Alternativas descartadas

- **Atar engine + variante de kernel**: inviable por las seis variantes de
  `expf`. Ver "Contexto".
- **Tabla derivada de `scheme_ids` sin hash explícito**: atar una función a un
  ordinal de formato es frágil; cualquier cambio futuro la mueve sin que nadie
  lo note.
- **Un AIR por arquitectura**: seis AIR, seis pruebas, y la misma prueba de
  modelo que ya no es portable.
- **Verificar la `expf` polinómica del engine**: exige aritmética float bit-exacta
  por polinomio. `float_air.zig` ya da el orden de magnitud (108 constraints
  compuestas por multiplicación) y solo cubre multiplicación.

## Referencias

- `libs/gadgets/nonlin/root.zig` — tablas, digest, fragmentos
- `libs/statement/root.zig` — `nonlinearity_digest`, `version = 2`
- `libs/gadgets/norm/root.zig` — refusal explícita y requisitos
- `docs/BLUE_PRINT.md` §3.1 — el contrato dual-path que esto hace real
- `llama.cpp@1c3c9674d` — `ggml/src/ggml-cpu/vec.cpp:380`, `vec.h:1200`
