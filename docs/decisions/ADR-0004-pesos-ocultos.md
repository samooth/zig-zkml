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
| compromiso que oculte | **Merkle + Blake3** (`libs/fri/src/root.zig:88-98`) |
| campo | Goldilocks p = 2⁶¹−1, **campo pequeño**, FRI sobre F_p² |
| transcript ZK | el del pin se describe a sí mismo como *«house design, not a specification»* |

El punto que más pesa: **campo pequeño + Merkle-BLAKE3 no oculta.** Esa es
exactamente la razón por la que los STARK de campo pequeño necesitan un hash
amigable para la ocultación. El de aquí es BLAKE3.

## La convergencia que conviene ver

**Poseidon2 aparece en dos sitios y sirve a los dos:**

1. es la opción **(b)** del puente de la 2.3 — un hash compatible con el campo;
2. es lo que un Merkle necesita para **ocultar** en campo pequeño.

Elegir Poseidon2 por la ligadura también compra la mitad del costo de la
privacidad. Es el único punto donde dos partes caras del plan comparten una
decisión, y por eso la 2.3 se tiene que responder con esto en mente.

Lo que **no** se compra: el blinding. Eso no lo da ningún hash.

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