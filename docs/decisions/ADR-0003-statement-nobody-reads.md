# ADR-0003 · La statement que nadie lee

**Estado:** `ABIERTO` · medido en `17d6674` · cierra la 2.1, bloquea la 2.2 y la 2.3

## Qué se measuring

`docs/PLAN_PRACTICAL_VALUE.md` §2.2 pide un puente verificable entre `root(W)` y el
compromiso field-native, y avisa de que «es la pieza que más fácilmente hunde un
calendario». Antes de elegir entre las tres opciones de §2.3 hay que medir qué
existe hoy.

## Qué hay

`StatementLayer` (`libs/statement/root.zig`) lleva `weights_root: Hash`, `h_input`,
`h_output` y `weights_leaf` — 32 bytes de Blake3 sobre Merkle. `hash()` y
`serialize()` existen y tienen pruebas.

## Qué pasa

**Nadie fuera de su propio fichero lo construye ni lo hashea.**

| pregunta | respuesta | cómo se comprobó |
|---|---|---|
| ¿quién construye un `StatementLayer`? | 7 sitios, los 7 en sus propios tests | `grep -rn 'StatementLayer{'` |
| ¿quién llama a `StatementLayer.hash`? | nadie | `grep -rn 'StatementLayer'` fuera del fichero |
| ¿qué llega al transcript? | los `DOM_*` y las raíces del fingerprint; ningún hash de statement | `grep -rn 'absorbBytes' libs/` |
| ¿está en el grafo de build? | sí, pero solo como re-export | `zkml.zig:19` |

El único enlace con la superficie pública es:

```zig
// zkml.zig:53
_ = &statement.StatementLayer.serialize;
```

`&f` analiza la **firma** de `serialize`, no su **cuerpo**. Compila, responde a un
grep, y no ejecuta nada.

## Por qué esto importa más de lo que parece

Es la segunda instancia de la misma forma, en el mismo fichero, a cuatro líneas de
distancia. `AGENTS.md` ya nombra la primera:

> `zkml` `zkml.zig` refs · `&gadgets.nonlin.SiLULookup.airFragment` — **the worst
> one: it compiles, answers a grep, and executes nothing**

`zkml.zig:53` es exactamente eso, aplicado al módulo que sostiene la identidad del
modelo. La diferencia es que este está debajo de la casilla que el plan llama la
que más hunde un calendario.

Y hay una asimetría que lo hace peor que inofensivo: `StatementLayer` **sí** tiene
pruebas, y son buenas. Siete tests que comprueban determinismo, ligadura, forgery y
rechazo de la versión v1. Un revisor que llega al final lee «la statement está
probada» y la da por cerrada. Las pruebas son reales; lo que no existe es el
consumidor.

## Qué NO es este documento

No es una propuesta de arreglo, y no cierra la 2.2. La 2.2 no se puede cerrar
hasta que se sepa **qué consume la statement**, y esa pregunta es anterior a las
tres opciones de la 2.3: las tres construyen un puente hacia un consumidor que hoy
no existe.

Lo que sí establece es el orden. Elegir Poseidon2, o un compromiso field-native, o
un manifiesto firmado antes de tener el consumidor es escribir la parte cara de un
sistema que todavía no tiene usuario.

## Lo que hay que decidir

1. **¿Debe la statement entrar en el transcript del prover?** Si sí, es trabajo
   pequeño y es el primer paso de la 2.2. Si no, la 2.2 está planteada sobre una
   premisa que hay que corregir antes.
2. **¿Quién la construye?** Hoy nadie. `libs/prove` y `libs/stark` reciben
   configuración, no statements.
3. **¿Los pesos son públicos u ocultos?** Esto es lo que decide entre las tres
   opciones de la 2.3, y no es una pregunta técnica:
   - **Públicos** → el hash de la statement basta como entrada pública; no hace
     falta puente dentro del AIR.
   - **Ocultos** (objetivo (b), privacidad de pesos) → el AIR necesita conocer los
     pesos, y entonces sí hace falta (a) o (b).

La 3 es la que ordena las otras dos, y la 2.3 dice que la decisión es del
propietario.