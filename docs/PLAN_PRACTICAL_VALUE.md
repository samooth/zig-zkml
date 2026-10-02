# PLAN — Utilidad para casos prácticos

Origen: revisión externa del 2026-10-02 más la pregunta de fondo que surge de ella.
Alcance: `zig-zkml`. Nada de esto cambia F0-F4; lo ordena y le quita trabajo perdido.

Este documento es del propietario. Si algo aquí está equivocado, se corrige y se
dice en el mensaje del commit, no se implementa en silencio.

---

## 0. La pregunta que decide el resto

El objetivo declarado en el README es **(a) Integridad**: la salida Y se produjo
ejecutando realmente el modelo comprometido. **(b) Privacidad de pesos es extensión
futura** y **(c) modo zkVM está fuera de alcance** (`README.md:55-59`).

Eso significa que el ZK todavía no tiene que ganarse su puesto. Bajo el alcance
actual, un manifiesto firmado, una raíz reproducible y una ejecución determinista
establecen (a) sin ZK.

> **¿Puede el verificador volver a ejecutar el modelo?**

| si puede | qué basta | ZK |
|---|---|---|
| sí | manifiesto + raíz reproducible + ejecución determinista | no hace falta |
| no, y el modelo es pequeño | lo mismo; el coste de re-ejecutar es aceptable | no hace falta |
| **no, y el modelo es grande** | **una prueba sucinta** | **sí, y es F3** |

**Esta es una frase y se decide antes de escribir código.** Sale del caso de uso
concreto, no de la hoja de ruta. Está sin responder y todo lo de abajo depende de
ella.

Los criterios de F3 ya la responden solos si se alcanzan: `proof < 1 MB`,
`verify < 100 ms`, `±1 ulp rejected` (`README.md:262`). **Si la ruta del fingerprint
no los alcanza, esa es la señal de que el ZK no se gana el puesto para (a) solo.**

---

## 1. Lo que es pérdida de trabajo, y se para hoy

- [x] **Dejar de medir por MAC.** `338 µs/MAC` y `142 µs/MAC` (`README.md:338-339`)
      miden una granularidad que ya no es el statement. Conservar las cifras como
      histórico; no volver a correr esos bancos.
- [x] **No empezar ninguna optimización de la ruta de 10 a 23 días.** El coste está
      en el statement, no en el código: 5,91 mil millones de restricciones de STARK
      más FFT, RLC, quotient, FRI y aperturas sobre toda la traza.
- [x] **No ajustar el ancho de traza para agrupar MACs.** Agrupar 16 por fila baja el
      tiempo de prueba unas 2,6 veces y sube verificación unas 2,9 y tamaño unas 2,6.
      El propio README ya mide el compromiso; no es un camino.
- [ ] **Anotar en el README que la granularidad por MAC está abandonada**, para que
      las cifras de 338 y 142 no se lean como una meta viva. Una cifra con su papel
      de cifra histórica necesita decirlo.

---

## 2. Lo que no es pérdida y es prerrequisito

Todo lo de esta sección lo exige la respuesta por fingerprint, y casi todo lo exige
**más** que la ruta por MAC. Nada de esto se toca en el punto 1.

- [ ] **Transcript y ligadura Fiat–Shamir.** Con fingerprint, `u` y `v` se derivan
      *después* de fijar todos los compromisos. Es el punto donde el sistema se
      rompe si está mal, y el orden de compromisos ya está implementado y medido
      (`README.md:308-314`).
- [ ] **Puente verificable entre `root(W)` y el compromiso field-native.** No es una
      tarea más: es la que hace que la prueba signifique algo. `root(W)` prueba la
      identidad del artefacto con Merkle y Blake3; el backend trabaja con elementos
      de Goldilocks; convertir un resumen de bytes a un valor de campo no queda
      ligado. **Es la pieza que más fácilmente hunde un calendario.**
- [ ] **Evaluar las tres opciones del puente y quedarse con una, por escrito.**
      (a) compromiso field-native de los pesos dentro del AIR; (b) función hash
      compatible con el campo, Poseidon2, como puente verificable; (c) manifiesto
      firmado que una ambas representaciones, con modelo de confianza explícito.
      **La decisión es del propietario y necesita motivo.**
- [ ] **Ruta `recorded` sólida.** Con statement comprimido la afirmación es sobre una
      aritmética concreta: si la ruta nativa difiere en el último bit, la prueba
      verifica el gadget, no la salida. **La recorded es el suelo de verdad, y el
      eslabón más débil de la cadena es el motor, no la criptografía.**
- [ ] **Semántica del kernel declarada y comprobada**, porque con statement
      comprimido pesa más: esquema de cuantización, formato (Q4_0, Q4_1, Q8_0,
      FP16, BF16, FP8), escala y procedencia de la escala, orden de acumulación,
      dimensiones y relleno, commit del adapter/kernel, modo recorded, CPU o GPU.

---

## 3. F3 en un solo formato, una capa, una geometría

El orden que propone la revisión externa es correcto. Lo que cambia es que **F3
empieza por lo pequeño y no por la geometría de 671B.**

- [ ] **Elegir una geometría fija y pequeña** para el primer F3, y escribirla en el
      README con su número. No 2048x1408x2048 todavía.
- [ ] **Un formato, una capa, una geometría.** Tres variables, una cada vez. La
      agregación por tiles solo se paga cuando hay varios bloques, así que no es lo
      primero.
- [ ] **AIR del fingerprint, que hoy no existe.** El README dice que lo medido es la
      aritmética y el orden de compromisos, y lo no medido es el AIR, el FRI y la
      agregación. **El AIR es el primer hueco real.**
- [ ] **FRI sobre el fingerprint**, aportado a `zig-algebra` en vez de implementado
      aquí, si esa es la decisión.
- [ ] **Medir contra los criterios de F3** y escribir el resultado en la tabla de
      estado con el estado que corresponda: `done`, `measured`, `partial` o
      `decision pending`. **Con el número, no con un adjetivo.**

---

## 4. Binding de los pesos

- [ ] **Decidir y escribir el modelo de confianza del binding, antes del AIR.** Qué
      prueba exactamente `root(W)`, qué prueba el compromiso field-native, y dónde
      está el salto entre los dos.
- [ ] **Si la opción elegida es Poseidon2**, comprobar que el módulo de la variante
      está disponible en el pin y que el round count está medido contra el
      ensamblador, no supuesto.
- [ ] **Dominios separados** por capa, modelo, formato, kernel y tile, escritos como
      regla en el README **antes** de implementar. El orden de compromisos —absorber
      todas las raíces, derivar desafíos, impedir que el prover elija el desafío
      después de ver los tiles— es la parte que se hace mal sistemáticamente.

---

## 5. Tiles y agregación, después

- [ ] **Orden de compromisos por tile, escrito como regla antes de codificar.**
- [ ] **Agregación recursiva de fingerprints**, una prueba por capa.
- [ ] **Verificar que no se están verificando miles de pruebas STARK independientes**,
      que es el motivo de la agregación.

---

## 6. El core y las no linealidades, separado

- [ ] **Declarar qué es lineal y qué no, en el README.** Solo lo lineal es lo que el
      fingerprint comprime.
- [ ] **Normalización, activación, routing y Lookup** quedan fuera del statement
      comprimido y necesitan su propio tratamiento. Cada no linealidad que se añada
      después vuelve a cambiar el statement, y eso es una decisión, no un añadido.

---

## 7. Criterio de parada

- [ ] **Escribir el criterio por el que esto deja de merecer esfuerzo.** No es "no
      hay tiempo": es **los criterios de F3 no se alcanzan**, y en ese caso la
      respuesta es manifiesto más reproducibilidad.
- [ ] **Revisar la tabla de objetivos (a)/(b)/(c)** con la respuesta de la sección 0
      escrita. Si (a) basta con re-ejecución, el objetivo (b) es lo que justifica el
      ZK, y (b) es hoy una extensión futura. **Eso es decisión del propietario, no
      una conclusión del trabajo.**

---

## Anexo — lo que ya está y no hay que rehacer

- Transcript con orden de compromisos: implementado y medido (`README.md:308-314`)
- Aritmética del fingerprint y banco propio: `zig build bench-fingerprint`, unas
  834 veces sobre 2048x1408x2048
- Puertas: trece, con mutación detrás
- Empaquetado: `libs/prove` y `libs/verify` ya usan el FRI del pin
  > **Corrección (2026-10-02), y lo que se encontró al empezar.** El directorio
  > **no existe**: no hay tal directorio en el repositorio. Se nombra sin su
  > ruta porque nombrarla con ruta la haria parecer existente. Y de los dos que
  > menciona el nombre, **ninguno** usa hoy el FRI del pin.
  >
  > Se intentó dos veces y no se puede completar: el pin llama a
  > `transcript.challengeFieldChecked(F)` y nuestro transcript expone
  > `challengeField(F)` sin la variante `Checked`. Añadirla cambia la
  > derivación de desafíos del transcript y cambia los bytes de **toda** prueba.
  > Eso es una migración de protocolo, no un refactor.
  >
  > Lo que sí salió de los intentos es un defecto real y distinto, encontrado
  > porque por fin hay un test que llama a la API pública: `flattenToFp2` pone
  > `.b = 0` en cada punto, y quedarse con la coordenada real de un polinomio
  > sobre F_p² no conserva el grado bajo respecto al toro. **La API pública de
  > `libs/prove` no produce pruebas que su propio verificador acepte**, y hasta
  > ahora no lo sabíamos porque el test existente llama a `fri.prove` directo y
  > se salta `flattenToFp2` por completo.
- F0 y F1 completos
- Puentes de citas en documentos: `zig build doc-paths`

---

## Nota sobre la revisión externa

La revisión recomienda fingerprint, sumcheck, tiles y el binding de pesos. **Los
cuatro ya están en el roadmap** (`README.md:22` reordenó el plan por lo medido;
`:263` es F4 sobre el statement del fingerprint). Lo que la revisión **añade** es la
pregunta de la sección 0, que el repo no se estaba haciendo, y la advertencia de que
(a) solo quizá no necesite ZK. **El resto confirma la dirección; no la cambia.**