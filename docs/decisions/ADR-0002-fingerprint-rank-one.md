# ADR-0002 — El reto del fingerprint debe ser de rango 1

- **Estado:** aceptado
- **Fecha:** septiembre 2026
- **Ámbito:** verificación de productos GEMM (F2/F4, camino crítico)
- **Sustituye a:** nada; matiza la decisión 6 de [ADR-0001](ADR-0001-zkml-vs-blueprint.md)
- **Vigente en:** `libs/stark/fingerprint.zig`

## Contexto

El camino de sumcheck que [ADR-0001](ADR-0001-zkml-vs-blueprint.md) difirió a
F4+ rests on verifying a product without materialising it. The naive
formulation is: for a challenge matrix r_{ij}, check

    ⟨r, C⟩ = Σ_{i,j} r_{ij}·C_{ij} = Σ_{i,j} r_{ij}·Σ_t A_{it}·B_{jt}

and hope the double sum factors. It does not. With a general r_{ij} there is
nothing to pull out of `Σ_{i,j} r_{ij}·A_{it}·B_{jt}`: any attempted factorisation
leaves one index stranded, and the cost stays O(m·n·k). Worse, the
factorisation that *does* exist — computing (Σ_i r_{it}A_{it})·(Σ_j r_{jt}B_{jt})
and summing over t — is **quadratic** in r, so it does not even evaluate the
bilinear form ⟨r, C⟩. It is a different quantity that happens to be cheap.

So the first implementation was wrong in a way that the type system accepted
and the tests caught only by coincidence. The identity is:

    ⟨u⊗v, C⟩ = Σ_t ⟨u, A_{·t}⟩ · ⟨v, B_{·t}⟩

which requires r = u⊗v, a rank-1 challenge. Only then does the index `t` become
the sole free index and the cost fall to O((m+n)·k) — roughly 1000× below
O(m·n·k) on the 2048×1408 tile that motivates this work.

## Decisión

El reto del fingerprint es de rango 1, `u⊗v`, y la API lo refleja en el tipo:
`Challenge` lleva `u` y `v` por separado y nunca materializa la matriz m×n.

Esto no es una convención que un llamante pueda cumplir por error: pasar una
matriz densa es un error de compilación, no un resultado silenciosamente
incorrecto. La alternativa —aceptar `r` y confiar en que el llamante sabe
factorizar— se rechaza explícitamente.

Consecuenciaderivada y a menudo malentendida: la forma es **lineal en el
objeto `u⊗v`**, no en `u` y `v` por separado. Escalar `u` en solitario no
duplica el valor, porque ⟨u⊗v, C⟩ no factoriza a través de `u`. El test
inicial afirmaba lo contrario y falló; la aserción era incorrecta, no la
implementación.

## Consecuencias

- `fingerprintClaim` es aritmética pura, no una prueba. Sigue siendo necesario
  fijar `u` y `v` en el transcript **después** de comprometer A y B; ese orden
  es la soundness de toda la construcción y lo implementará F3, no este módulo.
- El prover por elemento (`gemm_air`) sigue siendo la referencia y el oráculo
  de soundness. Este camino solo puede ser más rápido, nunca la única
  verificación.
- `productInner` existe como oráculo a propósito lento (O(m·n·k)) para que el
  camino rápido se compruebe contra la definición y no contra sí mismo, tal
  como exige la regla 8 del plan interno.

## Alternativas descartadas

- **Reto denso `r_{ij}`**: descartado, no computa la forma bilinear. Ver
  "Contexto".
- **Un reto escalar** (Σ C_{ij}): más barato aún, pero trivialmente satisfacible
  por un producto incorrecto; no aporta soundness.
- **Materializar `r`**: imposible de forma útil, m·n elementos, que es
  precisamente lo que se intenta evitar.

## Referencias

- `libs/stark/fingerprint.zig` — implementación y tests
- `libs/stark/gemm_air.zig` — camino por elemento, referencia
- [ADR-0001](ADR-0001-zkml-vs-blueprint.md) — decisión 6 (fingerprint diferido)
- `docs/BLUE_PRINT.md` §11 — roadmap F4
