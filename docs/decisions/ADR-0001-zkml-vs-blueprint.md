# ADR-0001 — Abandonar el diseño de zkML.md en favor del contrato de aritmética exacta

- **Estado:** aceptado
- **Fecha:** septiembre 2026
- **Ámbito:** diseño de witness, commitment y statement
- **Sustituye a:** [docs/archive/zkML.md](../archive/zkML.md) (documento histórico)
- **Vigente en:** [BLUE_PRINT.md](../BLUE_PRINT.md)

## Contexto

El estudio inicial ([archive/zkML.md](../archive/zkML.md)) propose una arquitectura
para una librería zkML en Zig: inferencia nativa como fuente del witness,
compromiso de pesos por Merkle root, statement implícito y mezcla de backends de
campo. Su encuesta de estado del arte sigue siendo válida y se conserva; su
propuesta de diseño no.

Al implementarlo aparecieron problemas que hacen la propuesta original
insostenible: el witness nativo no reproduce bit a bit la aritmética que el
circuito verifica, y el statement implícito no permite a un verificador
independente comprobar nada.

## Decisión

Se adoptan las ocho decisiones siguientes. La columna § remite a la sección
correspondiente de BLUE_PRINT.md, que es la especificación autoritativa.

| # | El diseño inicial proponía | Decisión adoptada | § |
|---|---|---|---|
| 1 | La inferencia nativa (FP32/BF16) ES el witness | **Contrato de aritmética exacta**: en modo recorded el kernel ejecuta la aritmética del gadget (dual-path); bit-exactness es invariante verificable | §3 |
| 2 | Binius y Goldilocks mezclados dentro del mismo gadget | **Backend único Goldilocks en v1**; Binius diferido a F4+ (evita composición cross-domain) | §4 |
| 3 | Merkle root de pesos sirve a F3 | Los pesos van como **columnas comprometidas en la traza STARK**; el root Merkle (F0) coexiste como attestation y se liga vía hash column Poseidon2 | §5 |
| 4 | Statement implícito | **Public inputs completos** (`H(X)`, `H(Y)`, layer/expert, leaf de pesos, esquema, params) + orden canónico del transcript | §6 |
| 5 | "2048/1408 del experto, DeepSeek-V3" | 2048/1408 es **Qwen3-Next**; DeepSeek-V3 es 7168/2048. Ambas shapes, etiquetadas | §10 |
| 6 | `Constraints{.kind = .sumcheck_fingerprint}` (L2→L1 hand-wave) | **AirGraph** como interfaz formal L3→L1; composición monolítica v1 (AIR por capa) y protocolo fingerprint v2 (GKR/sumcheck) como optimización F4 | §5, §7 |
| 7 | Sin nivel de seguridad ni parámetros | Tabla de parámetros y objetivo ≥80 bits (conjeturado) | §9 |
| 8 | Nits de código: `catch unreachable`, `Shake256 = undefined`, tabla SiLU "int8" siendo `i16`, sin `kt_proof_free` | Todos corregidos en los sketches | §7, §8 |

## Corrección factual

Los lookups tipo LogUp son de **Haböck**; el documento inicial atribuía la
técnica a "Tabrenheim".

## Consecuencias

- El witness pasa a ser una responsabilidad del kernel, no una consecuencia
  afortunada de la inferencia nativa.
- Se acepta coste de dual-path a cambio de que el modelo verificado sea el
  que el circuito comprueba.
- Las decisiones posteriores se registrarán en `decisions/` con su propio ADR.

## Alternativas descartadas

- **Adoptar el diseño original tal cual**: descartado por el fallo de
  bit-exactness descrito arriba.
- **Binius en v1**: descartado por coste de composición cross-domain; reevaluar
  en F4+.

## Referencias

- [BLUE_PRINT.md](../BLUE_PRINT.md) — especificación vigente
- [archive/zkML.md](../archive/zkML.md) — documento superado, conservado por su
  encuesta de estado del arte

---

## Apéndice — Encuesta de estado del arte

Se conserva del documento inicial porque es la única parte de este que sigue
siendo válida: el panorama de sistemas y qué se aprende de cada uno. **Las
conclusiones de diseño del original fueron superadas por las decisiones de esta
ADR**; se marcan aquí para que el contraste sea explícito.

| Sistema | Campo | Enfoque | Lección para el diseño Zig |
|---|---|---|---|
| EZKL | BN254 | ONNX → circuitos Halo2, lookups sobre tablas | El compilador (L3) es la mayor parte del trabajo; los gadgets solos no bastan |
| zkLLM | Goldilocks | Gadgets GEMM/softmax/LayerNorm + comunicación con llama.cpp para el witness | El witness lo genera la inferencia nativa y el circuito solo verifica — **patrón descartado**, ver decisión 1 |
| Binius / IntegerMatrix | GF(2^n) en torres | GEMM int8 bitsliceado sobre torres binarias, sumcheck | El mejor coste de probador para datos int8; cada byte son 8 wires "gratis" — **diferido a F4+**, ver decisión 2 |
| DeepProve | Goldilocks/Starky | Atención lineal verificable + recursion | La atención es el gadget más caro; empezar por MLP/MoE, no por atención |
| zkML práctico (varios) | M31/QM31 | STARK Circle / Circle-STARK | Lookups (LogUp, de **Haböck**) para no-linealidades |

Conclusiones vigentes:

- **Lookups para todo lo no lineal** (GELU, SiLU/SwiGLU, softmax-step): tablas
  precomputadas verificadas con LogUp, nunca constraints polinomiales de alto
  grado.
- **La atención es el último objetivo**, no el primero.

Conclusiones superadas por esta ADR:

- ~~*Dos backends de campo, no uno*~~ → **v1 usa Goldilocks para todo**; Binius se
  reevalúa en F4+ (decisión 2, por coste de composición cross-domain).
- ~~*GEMM por sumcheck/fingerprint en vez del producto*~~ → **v1 compone AIR por
  capa**; el protocolo de fingerprint queda como optimización F4 (decisión 6).
