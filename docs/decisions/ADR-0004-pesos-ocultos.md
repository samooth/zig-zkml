# ADR-0004 · Pesos ocultos, y las dos versiones

**Estado:** `DECISIÓN` del propietario · medido en `ce3a51e` · abre la 2.3 y la 6

## La decisión

**Los pesos son ocultos. Y la biblioteca debe soportar las dos versiones.**

Traducido a los objetivos del README: **(b) es el objetivo**, y (a) se sigue
soportando como el otro modo de la misma biblioteca. No es (c) en el sentido de
«una sola prueba que sea a la vez sound y hiding» — es **una biblioteca con dos
modos**: pesos públicos, o pesos ocultos.

## Por qué «ambas» sale más barato que «solo (b)»

Porque **ZK implica soundness**. Una vez que el backend es zero-knowledge, el modo
con pesos públicos es el mismo código con los pesos declarados públicos. La
dirección que no funciona es al revés: (a) no da (b).

Así que la decisión no añade una segunda ruta que mantener: **fija cuál es el
backend y qué modo es el GENERAL**.

## Lo que cuesta, medido en `ce3a51e`

El backend actual **no puede** producir una prueba zero-knowledge. No es que sea
difícil: la maquinaria no está.

| requisito para (b) | estado medido |
|---|---|
| blinding del cociente y de las aperturas | **ausente** — `grep -rin 'blinding\|zero.knowledge\|deep.fri\|hiding'` sobre `libs/` del pin: nada |
| compromiso que oculte | **Merkle + Blake3** — el pin lo declara en su propio `root.zig` de FRI; aqui, `libs/fri/root.zig` |
| campo | Goldilocks p = 2⁶¹−1, **campo pequeño**, FRI sobre F_p² |
| transcript ZK | el del pin se describe a sí mismo como *«house design, not a specification»* |

El punto que más pesa: **campo pequeño + Merkle-BLAKE3 no oculta.** Esa es
exactamente la razón por la que los STARK de campo pequeño necesitan un hash
amigable para la ocultación. El de aquí es BLAKE3.

## Corrección (2026-10-03) — el papel de Poseidon2 era este, y era demasiado

Este documento afirmaba que Poseidon2 «es lo que un Merkle necesita para ocultar
en campo pequeño». **Medido, y no es así.**

```
grep -rn 'Blake3\|hashBytes' libs/stark/*.zig libs/air/*.zig
  → libs/stark/commit.zig, y solo ahí
```

**La compresión del Merkle vive fuera de la traza.** Nunca aparece como
constraint de un AIR, porque no hace falta: el verificador recorre el árbol
directamente. De ahí se cae la razón por la que se elige Poseidon2 en un ZK
—que es su **gadget**, el coste de verificar la hash dentro del circuito—, y
aquí ese coste no se paga.

Y la ocultación puede venir de otro sitio. Si el prover compromete a
`f + Z·h` con `h` uniforme y secreto —el enmascarado estándar—, las hojas ya son
uniformes, e invertir Blake3 sobre 2⁶¹ candidatos es inviable. **Enmascarar puede
bastar sin cambiar el hash.**

Lo que queda en pie de la versión anterior: **el enmascarado sí hace falta, y
ningún hash lo da.** Eso es lo que no existe hoy.

**No se ha medido todavía** si enmascarar basta sin Poseidon2, ni a qué coste en
headroom de grado. Queda como `DECISIÓN` y no como afirmación, que es la
diferencia entre las dos versiones de esta sección.

## Como se implementa (decision del propietario, 2026-10-03)

**«De momento implementalo tu; ya lo portaremos y adaptaremos.»**

Se conserva `libs/fri/root.zig` y se extiende con ZK. **El borrado queda
cancelado**, y con el la demolición de `tools/fri_diff.{zig,sh}` y el paso
`fri-diff` de `build.zig`, que solo existian para comparar contra el pin.

Lo que se decide es mal el primer paso, asi que se escribe antes de codificar:

| paso | que es | como se mide |
|---|---|---|
| 1 | **headroom de grado del enmascarado**:.width del LDE | ¿con que blowup cabe `f + Z·h`? |
| 2 | commitment enmascarado, hoja a hoja | el commitment no determina la columna |
| 3 | enmascarar el cociente | la apertura no filtra residuos |
| 4 | transcript ZK | simulable con trampa |

El paso 1 va primero porque es el unico cuyo resultado puede cambiar los otros:
si el enmascarado no cabe en el blowup actual, la conclusion no es «añadir
enmascarado» sino «cambiar de geometria», y eso se decide antes de escribir el
resto.

## Lo que esta decisión abre

- **La 2.3 se responde mirndose el objetivo:** pesos ocultos → el AIR tiene que
  conocer los pesos → hace falta (a) o (b) del puente, y el manifiesto firmado
  (c) no basta por sí solo, porque los pesos no están publicados para firmarlos.
- **La 6 pasa a tener sujeto** — declarar qué es lineal y qué no ya no es
 导购_io hygiene, es lo que decide qué se puede cegar y qué no.
- **La 7 se aprieta.** El criterio de parada sigue siendo los criterios de F3, y
  ahora hay un segundo frente: la ocultación no tiene criterio medido porque el
  pin no la tiene.

## Lo que esta decisión NO cierra

**No dice quién implementa el ZK FRI.** Y esa es la pregunta de verdad, porque es
la que choca con una decisión ya tomada:

- el pin es de otro repo, y no tiene ZK;
- `libs/fri/root.zig` — el nuestro — está a punto de borrarse, y es el único
  código de FRI bajo nuestro control.

Las dos salidas son **aportar el ZK FRI al pin** o **conservar el nuestro y
extenderlo**. La segunda deshace el borrado; la primera es trabajo en un repo que
no es nuestro ywhose roadmap no controlamos. Es decisión del propietario, y
`docs/PLAN_PRACTICAL_VALUE.md` §2.3 y §7 son su sitio.

Lo que sí queda dicho: **no se empieza el AIR del fingerprint antes de decidir
esto**, porque las dos respuestas dan AIRs distintos.