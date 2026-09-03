# BLUE_PRINT.md — zig-zkml: blueprint técnico de la librería

> Este documento es la especificación de diseño autoritativa de `zig-zkml`.
> Corrige y concreta las decisiones abiertas de `zkML.md` (que queda como
> documento histórico de motivación). Convenciones Zig 0.16 según
> `LESSONS_ZIG.md` de ktransformers-zig (ArrayList unmanaged, allocator
> explícito, shapes runtime, `comptime` solo para esquemas/backend).

## 0. Cambios respecto a zkML.md

| # | zkML.md decía | Este blueprint decide | § |
|---|---|---|---|
| 1 | La inferencia nativa (FP32/BF16) ES el witness | **Contrato de aritmética exacta**: en modo recorded el kernel ejecuta la aritmética del gadget (dual-path); bit-exactness es invariante verificable | §3 |
| 2 | Binius y Goldilocks mezclados dentro del mismo gadget | **Backend único Goldilocks en v1**; Binius diferido a F4+ (evita composición cross-domain) | §4 |
| 3 | Merkle root de pesos sirve a F3 | Los pesos van como **columnas comprometidas en la traza STARK**; el root Merkle (F0) coexiste como attestation y se liga vía hash column Poseidon2 | §5 |
| 4 | Statement implícito | **Public inputs completos** (`H(X)`, `H(Y)`, layer/expert, leaf de pesos, esquema, params) + orden canónico del transcript | §6 |
| 5 | "2048/1408 del experto, DeepSeek-V3" | 2048/1408 es **Qwen3-Next**; DeepSeek-V3 es 7168/2048. Ambas shapes, etiquetadas | §10 |
| 6 | `Constraints{.kind = .sumcheck_fingerprint}` (L2→L1 hand-wave) | **AirGraph** como interfaz formal L3→L1; composición monolítica v1 (AIR por capa) y protocolo fingerprint v2 (GKR/sumcheck) como optimización F4 | §5, §7 |
| 7 | Sin nivel de seguridad ni parámetros | Tabla de parámetros y objetivo ≥80 bits (conjeturado) | §9 |
| 8 | Nits de código: `catch unreachable`, `Shake256 = undefined`, tabla SiLU "int8" siendo `i16`, sin `kt_proof_free` | Todos corregidos en los sketches | §7, §8 |

Cita corregida: los lookups tipo LogUp son de **Haböck** (zkML.md escribía
"Tabrenheim").

## 1. Alcance y threat model

Modos zkML (el coste crece brutalmente de (a) a (c)):

| Modo | Se prueba | Privacidad | Estado en este diseño |
|---|---|---|---|
| **(a) Integridad** | La salida Y fue producida ejecutando de verdad los pesos comprometidos `root(W)` sobre la entrada X | ninguna | **Objetivo v1** (F0–F3) |
| (b) Privacidad de pesos | El prover usa W sin revelarlo | pesos | Extensión futura (commitments con hiding) |
| (c) Ambos | (a)+(b) | ambos | Fuera de alcance (modelos 671B) |

Statement modo (a):

> "Existe una traza de ejecución (activaciones, routing, sumas parciales)
> consistente con los pesos ligados al leaf hash `H(W_layer)` del árbol
> Merkle `root(W)`, la entrada `H(X)` y el esquema de cuantización S, tal que
> la aritmética exacta definida por S produce exactamente `H(Y)`."

Casos de uso: attestation de pesos (F0), sampling determinista (F1),
bloque MoE verificable (F2–F3), multi-bloque/recursion (F4).

## 2. Arquitectura de capas

```
L4  C API                        kt_prove_* / kt_verify_* (patrón kt_kernel.h)
       │
L3  Compilador de modelo         CircuitGraph → AirGraph (columnas, grados,
       │                          lookup tables, schedule de gadgets)
L2  Gadgets zkML   ◄── EL GAP    QuantTensor, gemm, dequant/requant, swiglu,
       │                          layernorm, routing (top-k), lookups
       │
L1  Proof systems  (zig-zk)      STARK+LogUp, AIR, transcript, sumcheck,
       │                          tower/Binius (F4+), recursion Poseidon2
       │
L0  Álgebra        (zig-algebra) Goldilocks, hash (Blake3/Poseidon), merkle,
                                 NTT
```

Build vs vendorear:

| Pieza | Decisión |
|---|---|
| L0 zig-algebra | consumir tal cual (path dep o tarball+hash); fork propio si hace falta |
| L1 zig-zk | consumir `air`, `transcript`, `stark`, `sumcheck`; **spike F2 obligatorio**: existencia/port de STARK FRI sobre Goldilocks (ver §12) |
| L1 Groth16 wrapper | opcional F4, solo si on-chain; el verifier de referencia actual (scalar-mul bit a bit) es 100x demasiado lento |
| L2/L3/L4 | construir aquí — no existe en Zig |

## 3. Contrato de aritmética exacta (invariante central)

### 3.1 El problema

El fast path nativo acumula en FP32/BF16 con rounding en cada paso; la
aritmética de campo (Goldilocks) es exacta. Un witness grabado del fast path
**no satisfará las constraints** (ni siquiera Q4_K: `sum += q * scale_fp16`
redondea en FP32). Ejecutar el modelo "dentro del circuito" es el error
clásico que hace zkML 10x más lento; usar el fast path tal cual hace el
proof **inválido**. La resolución es un contrato dual:

1. **El `QuantScheme` es normativo** (§4.2): define, para cada operación, la
   aritmética *exacta* que el proof verifica (acumuladores enteros en el
   campo, scales en fixed-point, no-linealidades por tabla).
2. **Dual-path en los kernels**: modo normal = fast path actual (sin coste
   añadido, cero hooks); modo recorded = misma descomprensión de pesos y
   shapes, pero ejecutando la aritmética del gadget — acumulación nativa en
   Goldilocks (u64 con reducción lazy), dequant con scales fixed, SiLU/GELU
   por la misma tabla LogUp que usa el circuito.
3. **Cualquier divergencia fast-path vs contrato es un bug del esquema o del
   kernel** y se detecta en test: el cross-check (`exactOutput` vs
   `nativeOutput`) debe caer dentro de la tolerancia documentada del esquema
   (epsilon por redondeos definidos); si no, o el esquema no captura lo que
   hace el kernel, o el kernel no implementa el esquema.

### 3.2 Lo que esto significa en la práctica

- `gemmExpert` gana un gemelo `gemmExpertExact` (mismo layout Q4_K/Q8_0,
  accumulate en u64→Goldilocks, scales dequantizados a fixed q8.16): ~2-4x
  más lento que el fast path, solo se ejecuta al generar witness.
- El statement prueba el **modelo exacto** (el definido por el esquema). El
  fast path es una optimización de ejecución cuya salida debe coincidir
  dentro del epsilon publicado como parte del esquema.
- Regla de honestidad del producto: si epsilon es no-trivial para un bloque,
  se documenta en el statement ("pruebo la aritmética exacta S; la salida
  nativa difiere ≤ epsilon").

### 3.3 Test obligatorio (lección A2: un check que nunca falla no es un check)

- Positivo: witness del path exacto → prove/verify pasa.
- Negativo: witness manipulado (±1 ulp en un elemento, índice de routing
  cambiado, escala alterada) → **RECHAZA**. Suite de mutación sistemática.
- Divergencia: `exactOutput` vs `nativeOutput` por capa dentro de epsilon.

## 4. Campo y representación (backend único Goldilocks en v1)

### 4.1 Por qué un solo campo

zkML.md mezclaba torres GF(2^n) (Binius, para nibbles/int8) y Goldilocks
(scales/acumuladores) dentro del mismo gadget Q4_K. Componer dos dominios
en una prueba es problema abierto. Decisión: **v1 = Goldilocks para todo**;
los int8/q4 caben de sobra en Goldilocks (una celda = un elemento, range
proof por LogUp de bytes). Binius (donde 1 byte = 8 wires y el sumcheck
bitsliceado es más barato) se retoma en F4+ como backend alternativo
**por proof completo**, nunca mezclado intra-gadget.

### 4.2 Esquemas y cotas de overflow derivadas

Goldilocks p = 2^61 − 1. Regla: cada esquema declara cota de magnitud M y
el diseño demuestra que acumulaciones ≤ k·M² < 2^61 con reducción lazy
documentada. **q32.32 queda eliminado** (un producto de dos valores
q32.32 con |x| ≥ 2^29 desborda 61 bits; no hay headroom de acumulación):
los acumuladores GEMM viven en forma nativa de campo (enteros acotados por
k·M²) y se requantizan al final.

| Esquema (`Scheme`) | Uso | Representación | Cota de magnitud | Overflow check |
|---|---|---|---|---|
| `int8_symmetric` | activaciones/pesos int8 | i8 en 1 elemento; range LogUp 2^8 | M = 2^7 | Σ k·M² = 2^11·2^14 = 2^25 ✓ |
| `int4_q4_k` (GGML block 256) | pesos Q4_K/Q6_K | nibble + scale/min fp16→fixed | nibble M = 2^4; scale q4.22 M < 2^5 | dequant por bloque: 256·2^4·2^5 = 2^17 ✓ |
| `int4_q8_0` (bloque 32) | pesos Q8_0/Q4_0 | i4/i8 + scale q4.22 | idem, bloque 32 | 2^5·2^8·2^5 = 2^18 ✓ |
| `fp8_e4m3` / `mxfp8_e4m3` | transporte KT, MXFP8 | bits signo+exp+mantisa + pow2 lookup (UE8M0 en MX es potencia de 2) | M = 2^7 | reencode a int8 por tabla; luego int8 ✓ |
| `fixed_q16_16` | activaciones dequantizadas (BF16 cabe exacto: mantisa 8 bits, \|x\| < 2^15) | 16 int + 16 frac en 1 elemento | M = 2^15 | productos 2^30; Σ 2^11 → 2^41 ✓ (reducción cada 2^5 productos si M²·2^5 < 2^61) |
| `fixed_q4_22` (scales) | scales fp16 dequantizadas | 4 int + 22 frac (26 bits) | \|s\| < 2^4 | uso puntual (mul por bloque) ✓ |
| `fixed_q8_8` (SiLU out) | salida de tabla SiLU | i16 q8.8, \|silu(x)\| ≤ ~128 | M = 2^8 | elemento a elemento ✓ |

Notas:

- fp16 scale exacto en fixed: un fp16 de [2^-12, 2^4) con mantisa 10 bits
  necesita hasta 12+10 = 22 bits fraccionales → `q4.22` (26 bits totales).
  **(Corrección F0: la fila anterior decía `fixed_q8_16` — 16 bits
  fraccionales NO son exactos para scales < 2^-6 y rompen el contrato de
  aritmética exacta §3. El rango aceptable es \|s\| ∈ [2^-12, 2^4);
  fuera de él (subnormal, cero, ≥ 2^4, inf/NaN) el dequant DEBE fallar
  con error — el witness nunca se interpreta mal en silencio. La
  negación de scale negativo es p − \|s\|, NO complemento a dos —
  2^64 ≡ 8 (mod p) haría el valor erróneo en 8 unidades.)** El gadget
  dequant descompone signo/exp/mantisa (range proofs) + lookup pow2
  (17 entradas) + mul entero — todo LogUp, sin constraints de alto grado.
- **Simplificación Q4_K v1 (documentada)**: `dequantQ4K` implementa la
  variante simétrica (nibble − 8)·d de un solo scale por bloque. El
  formato GGML real (super-block con d, m fp16 + 8 sub-scales de 6 bits,
  dequant asimétrico (nibble − 8)·d·s_k + d·m) aterriza con el gadget F2;
  el esquema `int4_gguf_q4_k` del enum NO cambia mientras tanto.
- MXFP8 (que faltaba en el enum de zkML.md): scale UE8M0 por grupo de 32 es
  potencia de 2 → lookup pow2 directo, el esquema más barato de reencuadre.
- **Estabilidad del enum `Scheme`**: los ordinales se serializan en el
  statement (`@intFromEnum`). Nunca reordenar/renumerar; solo añadir al
  final y bump de `statement_version`.

### 4.3 No-linealidades: siempre lookups

GELU, SiLU/SwiGLU, softmax-step: tablas precomputadas verificadas con
LogUp (Haböck), rango probado con lookups encadenados. Rango i16 de salida
de SiLU = 2 lookups de 2^8 (byte alto + byte bajo), no una tabla de 2^16.
Nunca constraints polinomiales de alto grado.

## 5. Composición de proofs y commitments

### 5.1 Composición L2→L1: `AirGraph`

Interfaz formal entre el compilador L3 y zig-zk L1. Un `AirGraph` es:

- **Columnas** con rol (`public` / `advice` / `fixed`), ancho en bits y cota.
- **Constraints**: polinomios de grado ≤ 3 sobre columnas, como AST o lista
  de coeficientes por grado.
- **Lookup tables**: tablas + multiplicidades LogUp.
- **Public inputs**: declarados y ligados al transcript.

Los gadgets L2 emiten `AirFragment` (subgrafo reutilizable con holes de
dims); L3 los instancia y enlaza. Nada de `Constraints{.kind = ...}`
opacos.

### 5.2 Estrategia por fases (evita el problema de composición en v1)

| Fase | Protocolo GEMM | Composición | Coste prover (estimado) |
|---|---|---|---|
| **v1 (F2–F3)** | **AIR monolítico por capa**: running-sum con chunks de 16 MACs por fila (`s' = s + Σ_{j<16} a_j·b_j`, grado 2), requant final por lookup | UN solo STARK por capa; cero composición cross-protocol; reusa zig-zk `stark`+`air`+LogUp tal cual | ~10³x por bloque (cientos de ms–segundos) |
| **v2 (F4)** | **Fingerprint sumcheck** (estilo zkLLM/GKR): el verifier reta r, el prover demuestra ⟨r, C⟩ = ⟨A_row, Bᵀr⟩ vía sumcheck sobre extensiones multilineales; mensajes ~2·log k elementos (~KBs) | Claims encadenados por transcript Fiat-Shamir + eval-proof final reusando el mismo STARK | objetivo ~10–50x por GEMM |

v1 es deliberadamente más caro y deliberadamente simple: valida todo el
stack (esquemas, witness, lookups, statement) sin investigación de
protocolos. v2 sustituye solo el gadget GEMM cuando el stack ya funciona.

Presupuesto v1 (bloque experto Qwen3-Next, decode m=1):
3 GEMMs (2048×1408, 2048×1408, 1408×2048) = 8.6M MACs → 540K filas AIR
(chunk 16) → ×blowup 2^4 ≈ 8.6M elementos ≈ 69 MB de traza. Proof size
estimado: 100–300 KB (queries FRI). Verify: <100 ms.

### 5.3 Commitments de pesos (el Merkle solo no basta)

Verificar una capa exige ligar la prueba a los pesos usados (~2.8–14 MB por
experto). Diseño:

1. **F0 attestation**: Merkle root Blake3 de todos los tensors al cargar
   (`loadWeights`) — barato, ~1x, sin proof system. Vale para "sirvo el
   modelo X".
2. **F3 layer proof**: los pesos son **columnas advice del AIR**, comprometidas
   por el commitment de traza del STARK (Merkle sobre codeword FRI — ya
   existe en zig-zk). El statement incluye `leaf_hash = H(W_layer)` (Blake3
   nativo en load, gratis).
3. **Binding traza↔leaf**: columna de hash incremental **Poseidon2** dentro
   del AIR sobre el stream de pesos tal como lo consume la traza; el hash
   final debe igualar `leaf_hash`. El auditor valida: leaf contra el root
   público del modelo publicado (path Merkle) y leaf contra la prueba.
   Coste: ~1–2M constraints extra (~2–4x slowdown del layer proof) —
   **amortizable por sesión**: hash column una vez por (layer, expert,
   weights); proofs posteriores de la misma sesión referencian el
   commitment de pesos ya hash-eado.
4. Alternativa considerada y descartada: Merkle openings de los pesos por
   proof — no cabe en "proof <1 MB".

## 6. Statement y transcript

### 6.1 Statement completo (public inputs, sin esto no hay soundness)

```
StatementLayer {
  version,                          // versionado del formato
  scheme_ids: []Scheme,             // aritmética exacta normativa (§3)
  magnitude_bounds,                // derivadas de §4.2, parte del statement
  weights_root: [32]u8,            // root Merkle del modelo (F0)
  weights_leaf:  [32]u8,           // H(W_layer) — Blake3
  layer_idx: u32,
  expert_ids: []u16,                // routing seleccionado (o su commitment)
  h_input:  [32]u8,                // Blake3(X)
  h_output: [32]u8,                // Blake3(Y)
  dims: {m, k, n, ...},
  security_params_id,               // §9
}
```

Sin `H(X)`/`H(Y)` un prover malicioso elige X', Y'. Sin `weights_leaf` elige
W'. Sin `scheme_ids` elige aritmética más barata.

### 6.2 Orden canónico del transcript

El `TraceRecorder` es multi-thread (ktransformers-zig ejecuta expertos en
paralelo). Absorber en orden de llegada haría el hash no determinista y el
Fiat-Shamir irreproducible. Diseño:

- Grabación por slots `(layer, op, expert, tp_rank)` — arena por slot, sin
  absorber.
- `finalize()` absorbe en **orden canónico** (layer → op → expert → rank),
  cada evento como `tag || len || bytes`.
- El `Shake256` se inicializa de verdad (fix del sketch de zkML.md) y su
  estado inicial es `domain_separator || statement_serialized`.

## 7. Módulos y API (Zig 0.16)

### 7.1 Layout

```
zkml.zig           # módulo root ÚNICO — file imports de libs/** (§13.1.1:
                   #   los tests solo se recolectan del file-set del root)
libs/
├── field.zig      # [HECHO] Goldilocks p = 2^61−1 (L0 propio)
├── merkle.zig     # [HECHO] Blake3 tree: root cacheado, orphans self-paired
├── transcript.zig # [HECHO] Fiat-Shamir Blake3 (absorb/challenge/squeeze)
├── attestation.zig# [HECHO] WeightsAttestor (F0)
├── tensor/        # [HECHO] QuantTensor + Scheme + q4.22 exact dequant
│   └── root.zig
├── trace/         # [HECHO] TraceRecorder (orden canónico §6.2)
│   └── root.zig
├── statement/     # [HECHO] StatementLayer (§6.1, longitudes u32)
│   └── root.zig
├── gadgets/       # F2+
│   ├── gemm/      # v1: AIR monolítico chunk-16; v2: fingerprint sumcheck
│   ├── quant/     # dequant/requant, fp8/mxfp8 reencode, GGML block scales
│   ├── nonlin/    # lookups SiLU/GELU/softmax-step (LogUp, Haböck)
│   ├── norm/      # rmsnorm/layernorm (sumas por LogUp de rangos)
│   ├── routing/   # group-top2 DeepSeek-V3 / top-k como lógica binaria
│   └── attention/ # F5+: fuera de scope v1 (DeepProve como referencia)
├── compile/       # F3+: CircuitGraph → AirGraph (§5.1)
└── prove/         # F3+: orchestración prove()/verify(), recursion (F4)
```

### 7.2 Tensors

```zig
//! libs/tensor/root.zig
const std = @import("std");

/// Esquema de cuantización — parte del TIPO y del statement (§6.1).
/// La cota M es comptime y el AIR genera el range proof correspondiente.
/// ESTABILIDAD: los ordinales se serializan en el statement — nunca
/// reordenar, solo añadir al final (bump de statement_version entonces).
pub const Scheme = enum(u8) {
    int8_symmetric = 0,
    int4_gguf_q4_k = 1,
    int4_q8_0 = 2,
    fp8_e4m3 = 3,
    mxfp8_e4m3 = 4,
    fixed_q16_16 = 5,   // activaciones dequantizadas
    fixed_q8_8 = 6,     // salida de tablas no lineales

    /// Cota de magnitud del valor representado (§4.2).
    pub fn magnitudeBound(comptime self: Scheme) usize {
        return switch (self) {
            .int8_symmetric, .fp8_e4m3, .mxfp8_e4m3 => 1 << 7,
            .int4_gguf_q4_k, .int4_q8_0 => 1 << 4,
            .fixed_q16_16 => 1 << 15,
            .fixed_q8_8 => 1 << 8,
        };
    }

    /// Bits de rango a probar por LogUp.
    pub fn rangeBits(comptime self: Scheme) u8 {
        return switch (self) {
            .int8_symmetric, .fp8_e4m3, .mxfp8_e4m3 => 8,
            .int4_gguf_q4_k, .int4_q8_0 => 4,
            .fixed_q16_16 => 32, // 4 lookups de byte
            .fixed_q8_8 => 16,   // 2 lookups de byte
        };
    }
};

pub fn QuantTensor(comptime scheme: Scheme) type {
    return struct {
        const Self = @This();
        rows: usize,
        cols: usize,
        data: []const u8,
        scales: []const u16,   // fp16 bits (GGML) según scheme
        zeropoints: ?[]const u8 = null,

        pub fn shape(self: Self) struct { usize, usize } {
            return .{ self.rows, self.cols };
        }
    };
}
```

### 7.3 GEMM v1 (AIR monolítico) — sketch corregido

```zig
//! libs/gadgets/gemm/root.zig

/// v1: C[m,n] = dequant(A)·dequant(B) verificado con running-sum AIR.
/// 16 MACs por fila; el witness viene del kernel exacto (§3), no del fast path.
pub fn GemmGadget(
    comptime A_scheme: tensor.Scheme,
    comptime B_scheme: tensor.Scheme,
) type {
    return struct {
        m: usize,
        k: usize,
        n: usize,
        a: tensor.QuantTensor(A_scheme),
        b: tensor.QuantTensor(B_scheme),
        c: []const Goldilocks, // salida del path exacto (witness)

        pub const macs_per_row: usize = 16;

        /// Fragmento AIR para este GEMM: columnas + constraints + lookups.
        /// El compilador L3 lo instancia dentro del AirGraph de la capa.
        pub fn airFragment(self: @This(), gpa: std.mem.Allocator) !air.Fragment {
            var luts: std.ArrayList(air.LookupTable) = .empty;
            errdefer luts.deinit(gpa);
            switch (A_scheme) {
                .int4_gguf_q4_k, .int4_q8_0 => try luts.append(gpa, tables.blockDequant(A_scheme)),
                .fp8_e4m3, .mxfp8_e4m3 => try luts.append(gpa, tables.fp8Reencode),
                else => {},
            }
            return .{
                .columns = .{
                    .{ .name = "a_chunk", .role = .advice, .scheme = A_scheme },
                    .{ .name = "b_chunk", .role = .advice, .scheme = B_scheme },
                    .{ .name = "s_running", .role = .advice, .bound = self.k * comptimeMagnitude() },
                    .{ .name = "s_final", .role = .advice, .scheme = .fixed_q16_16 },
                },
                .constraints = .{
                    // s' = s + Σ_{j<16} dequant(a_j)·dequant(b_j)   (grado 2)
                    .{ .expr = sum16(), .degree = 2 },
                },
                .lookups = try luts.toOwnedSlice(gpa), // try, nunca catch unreachable
                .rows = (self.m * self.n * self.k) / macs_per_row,
            };
        }

        fn comptimeMagnitude() usize {
            return tensor.Scheme.magnitudeBound(A_scheme) * tensor.Scheme.magnitudeBound(B_scheme);
        }
    };
}
```

### 7.4 Lookups de no-linealidades

```zig
//! libs/gadgets/nonlin/root.zig

/// SwiGLU: gate_out * silu(gate_out).
/// Entrada: gate_out REQUANTIZADO a int8 (paso obligatorio del esquema,
/// el lookup es sobre la rejilla int8, no sobre BF16 — fix de zkML.md).
/// Salida: silu(x)·2^8 en q8.8 i16 (|silu(x)| ≤ ~128 < 2^15 ✓),
/// probada con 2 lookups de byte, no una tabla de 2^16.
pub const SwigluLookup = struct {
    pub const silu_q8_8: [256]i16 = blk: {
        var t: [256]i16 = undefined;
        for (0..256) |i| {
            const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
            const s = x / (1.0 + @exp(-x));
            const scaled = @round(s * 256.0);
            t[i] = @intFromFloat(scaled); // |scaled| < 32513 < 2^15, seguro
        }
        break :blk t;
    };

    pub fn airFragment() air.Fragment {
        return .{
            .lookups = &.{
                .{ .table = lowByteTable(&silu_q8_8), .width = 16 },
                .{ .table = highByteTable(&silu_q8_8), .width = 16 },
            },
        };
    }
};
```

### 7.5 TraceRecorder (orden canónico — sketch corregido 0.16-dev.2535)

Implementación real en `libs/trace/root.zig`. Lecciones del sketch
anterior (todas aplicadas):

- `std.Thread.Mutex` **no existe** en 0.16.0-dev.2535 (el mutex migró a
  `std.Io.Mutex`, que requiere `Io`). La implementación usa
  `std.atomic.Mutex` + `spinLoopHint()` en el bucle de `tryLock` —
  spinlock con backoff; suficiente para critical sections cortas de
  grabación. Migrar a `Io.Mutex` cuando el C API capture `Io`.
- `std.AutoHashMap(...).empty` **no existe** para el mapa managed — es
  `.init(gpa)` (`.empty` es del unmanaged/ArrayList).
- `finalize` NO hace `catch continue` ni traga errores: un transcript
  que descarta slots en OOM rompe la soundness — los errores se
  propagan (`error{OutOfMemory}![32]u8`).
- `finalize` toma el mutex (un `record` concurrente durante el absorb
  sería un data race).
- El transcript es Blake3 (§9: F0/F1) con domain tag `"zkml.trace"` +
  hash del statement; el Shake256 del sketch queda para F4/recursion.
- Invariante de orden: el orden INTRA-slot es semánticamente
  significativo (los appends de un mismo slot deben provenir del mismo
  hilo/experto); el orden INTER-slot se canonicaliza en `finalize`.
  Test multi-thread obligatorio (§13): 4 threads → mismo hash que
  single-thread con los mismos datos.

```zig
//! libs/trace/root.zig — hooks que exporta ktransformers-zig

pub const TraceRecorder = struct {
    gpa: std.mem.Allocator,
    mutex: std.atomic.Mutex = .unlocked,
    slots: std.AutoHashMap(SlotKey, Slot), // (layer, op, expert, tp_rank)

    const SlotKey = struct { layer: u32, op: Op, expert: u16, rank: u8 };

    pub fn init(gpa: std.mem.Allocator) TraceRecorder {
        return .{
            .gpa = gpa,
            .slots = std.AutoHashMap(SlotKey, Slot).init(gpa),
        };
    }

    pub fn record(self: *TraceRecorder, key: SlotKey, data: []const u8) !void {
        while (!self.mutex.tryLock()) std.atomic.spinLoopHint();
        defer self.mutex.unlock();
        const gop = try self.slots.getOrPut(key); // managed: sin gpa
        if (!gop.found_existing) gop.value_ptr.* = .{};
        try gop.value_ptr.data.appendSlice(self.gpa, data);
    }

    /// Absorbe en ORDEN CANÓNICO — determinista independientemente del
    /// scheduling de threads (§6.2). Domain separation con el statement.
    /// Errores de alocación SE PROPAGAN (nunca `catch continue`).
    pub fn finalize(self: *TraceRecorder, stmt_hash: *const [32]u8) ![32]u8 {
        while (!self.mutex.tryLock()) std.atomic.spinLoopHint();
        defer self.mutex.unlock();
        // ... ordenar keys (layer, op, expert, rank); tag || len || bytes
    }
};
```

## 8. C API (superficie aditiva en `kt_kernel.h`)

Patrón de ktransformers-zig: handles opacos, allocator capturado en
`*_new` (regla B1), `pub export fn` en `main.zig`, gates
`tools/verify_abi.py` + `tools/audit_layout.py`, emisión forzada con
`comptime { _ = &mod.fn; }` verificada con `nm`.

```c
// --- zkML attestation & verifiable inference (experimental) ---
// F0: attestation de pesos (Blake3 incremental en loadWeights)
int  kt_weights_merkle_root(KT_MOE* moe, uint8_t root_out[32]);
int  kt_mla_weights_merkle_root(KT_MLA* mla, uint8_t root_out[32]);

// F1: sampling determinista
void kt_transcript_seed(const uint8_t* context, size_t ctx_len,
                        uint8_t seed_out[32]);

// F2+: verifiable inference por capa (handles opacos)
typedef struct KT_PROVER KT_PROVER;
KT_PROVER* kt_prover_new(const KT_MOE* moe, const uint8_t* weights_root);
void kt_prover_free(KT_PROVER* prover);                    // libera TODO el contexto
int  kt_prove_moe_layer(KT_PROVER* prover, int layer_idx,
                        const void* input, void* output,
                        uint8_t** proof_out, size_t* proof_len);
// FIX B1: la memoria del proof sale del allocator capturado del prover —
// sin free dedicado no hay forma sana de liberarla.
void kt_proof_free(KT_PROVER* prover, uint8_t* proof, size_t proof_len);
int  kt_verify_moe_layer(const KT_MOE* moe, const uint8_t* weights_root,
                         int layer_idx, const void* input, const void* output,
                         const uint8_t* proof, size_t proof_len);
```

`kt_verify_moe_layer` standalone: el verifier no necesita runtime L2/L3
ni el modelo completo — solo root, statement y proof.

## 9. Parámetros de seguridad

| Parámetro | Valor v1 | Nota |
|---|---|---|
| Campo | Goldilocks 2^61−1 | |
| STARK | FRI con DEEP, rate 1/16 (blowup 2^4) | port/confirmar sobre Goldilocks (spike F2) |
| Queries FRI | 80 | conjeturado ≥ 80 bits; revisar con la última literatura FRI |
| Lookups | LogUp (Haböck) batched, multiset | zig-zk ya lo implementa |
| Range proofs | LogUp de bytes encadenados | 2^8 por tabla |
| Sumcheck (v2) | retos FS del transcript compartido; error ≈ rounds/2^61 | despreciable |
| Transcript | Fiat-Shamir, domain-separated, versionado | Poseidon2 (recursion-ready) para F4; Blake3 en F0/F1 |
| Objetivo | **≥ 80 bits conjeturado** (audit off-chain) | on-chain → wrap Groth16 (F4, solo si producto lo pide) |
| Determinismo | statement serializado + orden canónico (§6.2) | tests de reproducibilidad multi-thread obligatorios |

Cualquier cambio de parámetro = bump de `security_params_id` en el
statement → proofs antiguos no verifican contra params nuevos.

## 10. Shapes corregidas y costes

| Modelo | gate/up por experto | down por experto | MACs por bloque experto (decode) |
|---|---|---|---|
| **Qwen3-Next** (2048/1408 — zkML.md lo atribuía erróneamente a DeepSeek-V3) | 2048×1408 ×2 | 1408×2048 | 8.6M |
| **DeepSeek-V3** | 7168×2048 ×2 | 2048×7168 | 44M |

| Operación | Nativo | Prover v1 (estimado) | Ratio |
|---|---|---|---|
| Attestation Merkle pesos (F0) | 1 lectura lineal | ~1x (hash en load) | **~1x** |
| Bloque experto Qwen3-Next | ~1 ms | 0.5–3 s | ~10³x |
| Bloque experto DeepSeek-V3 | ~10 ms | 5–20 s | ~10³x |
| Bloque completo MoE (8+ expertos activos) | — | ×n_expertos | recursion F4 o batching v2 |
| Modelo completo (61 capas) | ~1 s/capa | horas | **inviable hoy** — estado del arte zkML: ≤1B params con equipos dedicados |

Estimaciones de órdenes de magnitud con blowup 2^4 y chunk-16; se
recalibran con el bench de F2. Los números duros de go/no-go están en §11.

## 11. Roadmap F0–F4

| Fase | Entregable | Deps | Go/no-go |
|---|---|---|---|
| **F0** | `kt_weights_merkle_root` (moe+mla) sobre `loadWeights`; test root estable ante reorden de lectura; `verify_weights.py` | ~~zig-algebra `merkle` (o ~200 líneas propias)~~ **HECHO**: libs propias (`merkle`, `attestation`, `transcript`, `statement`) — 32/32 tests, pendiente: integración C API con ktransformers-zig | overhead de carga < 5% |
| **F1** | `kt_transcript_seed`; sampling reproducible | zig-algebra `hash` | cero cambio en kernels |
| **F2** | `libs/tensor` + gadget GEMM v1 (AIR chunk-16) + suite positiva/negativa vs `gemmExpertExact` | zig-zk `air`+`stark`+`transcript` | **Spikes previos**: (1) `zig fetch` semver 0.16.0-dev; (2) STARK/FRI Goldilocks existe o se porta; después: overhead < 100x en shape decode Qwen3-Next (criterio provisional, se fija post-spikes) |
| **F3** | `kt_prove_moe_layer`/`kt_verify_moe_layer` para 1 experto (Qwen3-Next shape real); binding Poseidon2 traza↔leaf; bench por bloque | F2 | proof < 1 MB, verify < 100 ms, test negativo ±1 ulp RECHAZA |
| **F4** | fingerprint sumcheck v2 (GKR/zkLLM-style) + multi-bloque con recursion Poseidon2; Groth16 wrap solo si on-chain | zig-zk `sumcheck`, `snark` (si madura) | overhead GEMM < 50x; decisión de producto |

Vendorear por defecto (path dep + tarball hash); fork propio en
Se adopta upstream cuando zig-zk/zig-algebra publique tags.

## 12. Riesgos y mitigaciones

| Riesgo | Severidad | Mitigación |
|---|---|---|
| Package manager rechaza deps (`minimum_zig_version = "0.16.0"` vs toolchain `0.16.0-dev.2535`; semver dev < release) | ~~bloquea F2~~ **resuelto para F0/F1** | vendor de `libs/{field,merkle,transcript,tensor,trace,statement,attestation}` funcionando (32/32 tests); reevaluar el fetch real para zig-zk en F2 |
| **STARK/FRI sobre Goldilocks no confirmado en zig-zk** (su STARK es M31; Binius es torres) | alto | spike F2 #2: port del STARK M31→Goldilocks (zig-algebra ya tiene NTT Goldilocks; el FRI es consumidor) — presupuesto 1–2 semanas si no existe |
| Upstream zig-zk/zig-algebra: autor único, 0 stars, 5 días, sin tags | alto | fork propio desde F2; F0/F1 solo dependen de zig-algebra `merkle`/`hash` (bajo riesgo, o 200 líneas propias — HECHO: libs propias, sin dependencias externas) |
| Divergencia aritmética exacta vs fast path mayor que epsilon | medio | contrato §3 + cross-check continuo por capa en CI; si diverge, el esquema no captura al kernel y se corrige el esquema (o el kernel) antes de avanzar |
| Coste binding Poseidon2 de pesos (2–4x en layer proof) | medio | amortización por sesión (hash column una vez por (layer, expert)); bench en F3 decide |
| Composición cross-protocol (GEMM sumcheck + AIR + lookups) | medio en F4 | v1 monolítica evita el problema hasta que el stack funciona; v2 cambia solo el gadget GEMM |
| Groth16 wrapper zig-zk lento (scalar-mul bit a bit, sin MSM) | bajo (solo F4 on-chain) | fuera de scope si no hay caso on-chain |
| Tests de libs no ejecutados por multi-módulo (§13.1.1) | medio (pasado) | módulo único en `zkml.zig`; gate `--summary all` con conteo explícito en CI |

## 13. Test y verificación

- **Unit**: gadget vs kernel exacto (mismos A,B → mismo C), tabla SiLU vs
  referencia float con epsilon del esquema.
- **Integración**: prove+verify sobre witness del path exacto de
  `gemmExpertExact`/routing reales de ktransformers-zig (shapes Qwen3-Next).
- **Negativos (obligatorios)**: ±1 ulp en un elemento, índice de routing
  cambiado, scale alterada, weights_leaf distinto → RECHAZA. Scale fp16
  malformada (inf/NaN/subnormal/≥ 2^4) → **error en dequant**, nunca
  reinterpretación silenciosa.
- **Determinismo**: mismo witness multi-thread → mismo hash de transcript →
  misma seed FS (2 ejecuciones, mismo proof). Test implementado en
  `libs/trace/root.zig` (4 threads vs single-thread, mismo hash).
- **ABI**: reusar `tools/verify_abi.py` (exports+arity) y
  `tools/audit_layout.py`; `nm` para emisión real de símbolos.
- **Bench**: extensión de `bench/gemm_bench.zig` midiendo overhead prover
  por GEMM y por bloque (ReleaseFast, nunca Debug).

### 13.1 Lecciones de ingeniería F0 (bugs reales encontrados y corregidos)

Estos defectos existían en el código inicial y quedaron documentados como
tests de regresión — actualizar los sketches de este documento cuando el
código demuestre lo contrario:

1. **Colección de tests multi-módulo**: `zig build test` SOLO recolecta
   tests del file-set del módulo root. Con libs como módulos separados
   (`--dep`), los tests de las libs **no se ejecutan** — el suite "pasaba"
   con 1/27 tests reales. Fix: módulo único rooteado en `zkml.zig` con
   file imports (`@import("libs/field.zig")`); el test block del root
   referencia cada lib. Verificar SIEMPRE con
   `zig build --summary all test` (cuenta de tests explícita).
2. **Merkle `root()` mutaba las leaves in-place**: el fold destruye el
   array de leaves → `proof()` tras `root()` devolvía pruebas inválidas y
   un segundo `root()` daba otro valor. Fix: root cacheado en `init`
   (árbol inmutable) — `root()` es O(1), idempotente y no-mutante.
3. **Merkle orphan inconsistency**: `root()` promovía el huérfano sin
   hashearlo, pero `proof()` lo emparejaba consigo mismo → las pruebas de
   hojas huérfanas (árboles de tamaño impar) NUNCA verificaban. Fix:
   self-pairing (H(orphan, orphan)) en ambas partes — seguro aquí por la
   separación de dominios leaf/node (un node hash jamás se reinterpreta
   como leaf).
4. **errdefer fuera de scope**: un `errdefer for (...)` declarado DESPUÉS
   del loop que puede fallar no cubre esos fallos → leak de los names
   duped en OOM. Los errdefer se declaran ANTES del código que falla y
   trackean el progreso (`duped` counter).
5. **Colecciones managed vs unmanaged**: `std.AutoHashMap` (managed) es
   `.init(gpa)`/`getOrPut(key)`; `std.ArrayList` es unmanaged
   `.empty`/`appendSlice(gpa, ...)`. Mezclar los estilos no compila o
   (peor) compila con semántica equivocada en sketches.
6. **`@intCast` a u8 en longitudes**: pánico en safe mode para > 255
   (DeepSeek-V3 tiene 256 expertos/capa). Longitudes serializadas: u32.

---

*Este blueprint corrige y sustituye las decisiones abiertas de zkML.md. El
item accionable barato hoy sigue siendo F0 (attestation de pesos); F2 no
arranca sin los dos spikes (toolchain semver, STARK Goldilocks) resueltos.*
