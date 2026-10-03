# ADR-0005 · La composición DEEP: por qué no está escrita todavía

**Estado:** `ABIERTO` · medido en `12222ef` · bloquea la 2.3 y la 3

## Qué se intentó

`ADR-0004` Ribe que la privacidad necesita la composición DEEP, y que el
backend tiene un cociente en su lugar. Se fue a escribirla.

## Lo que se encontró

**La construcción DEEP que había que implementar no es la que sale de tête, y la
que sale de tête no funciona.** Se escribió una versión —trazar `f` enmascarada,
`g(ωⱼ) = ωⱼ·yⱼ/(ωⱼ − r)`, y la reconstrucción
`f(x) = g(x)·(x − r)/Z_H(x)`— y **las pruebas no la sostenían**.

Antes de tocarla se sondeó el algebra en un campo pequeño, porque una identidad
que no se sostiene puede ser un error de la implementación o puede ser una
identidad falsa. Resultó ser lo segundo: se ajustó `Φ(x) = g(x)/f(x)` por
interpolación sobre un coseto, y **no coincidió con ninguna forma simple**
(`x − r`, `x`, `Z_H(x)/(x − r)`, `x·Z_H(x)/(x − r)`). La reconstrucción que se
iba a escribir no era una reconstrucción.

## La fuente

La referencia que se tenía anotada era **equivocada**: `arXiv 1904.00343` es
*Bow shocks, bow waves, and dust waves. III. Diagnostics*, un paper de astrofísica.
La correcta es:

> **DEEP-FRI: Sampling Outside the Box Improves Soundness** — Ben-Sasson,
> Goldberg, Kopparty, Saraf. arXiv **1903.12243**, eprint **2019/336**, ITCS 2020,
> LIPIcs vol. 148.

Y la construcción real es distinta de la que se iba a escribir. El verificador
muestrea `z` **fuera** del dominio, pregunta el valor del interpolante en `z` y en
`−z`, y el prover honesto entrega

```text
    f'(X) := ( f̃(X) − U(X) ) / Z(X)
```

donde `U` es el polinomio de grado ≤ 1 que interpola las dos respuestas y `Z` es el
mónico de grado 2 con raíces `z` y `−z`. Y la versión para STARK, **DEEP-ALI**, no
compromete la traza y el cociente sino **un solo polinomio**, con `Ans(X)`, `Z(X)`
y un *hole filler* `Fill(X)`, que es lo que hace que la consulta fuera del dominio
sea barata.

Nada de eso es el `g(ωⱼ) = ωⱼ·yⱼ/(ωⱼ − r)` que estaba a punto de entrar.

## Por qué no está escrita

Porque **publicar una identidad criptográfica que las pruebas no sostienen es
exactamente el fallo que este repositorio existe para no cometer.** Con la
identidad correcta a la vista, la implementación es trabajoordinary; sin ella, es
publicar una suposición con pruebas de aspecto convincente.

Lo que **sí** queda de esta sesión, y está en el commit anterior: la
falsación de que enmascarar con un múltiplo de `Z_H` no oculta nada
(`libs/stark/masking.zig`, con `tools/masking_mutation.sh` detrás). Eso está
medido, mutado y es cierto independientemente de qué haga DEEP.

## Lo que hay que hacer antes de escribirla

1. Leer `§4` (DEEP-FRI) y `§5` (DEEP-ALI) del paper correcto, y escribir la
   construcción en el ADR **antes** del código, con la identidad y la_oráculo
   encima de la mesa.
2. Decidir si la composición entra como DEEP-ALI — un solo polinomio comprometido,
   que es lo que quiere un prover de tamaño grande — o como la forma ad hoc de
   DEEP-FRI. La segunda es más fácil y peor; la primera es la que hace que esto
   escale.
3. El coste medido sigue pendiente: con la composición, el commitment crece, y
   `libs/stark/root.zig` decide hoy cuántos compromisos emite. Ese número no
   existe.

## Nota sobre la referencia equivocada

La cita `eprint 1904.00343` venía del skill `zkp-theory` de este entorno, no de
este repositorio. Se deja escrita porque la próxima vez que alguien la use va a
perder el mismo tiempo, y porque un número de papel es tan fácil de extender mal
como un `file:line`.
