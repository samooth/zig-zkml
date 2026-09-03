# zkML.md — Diseño de una librería zkML en Zig y su integración con ktransformers-zig

> Evaluación motivada por el estudio de `samooth/zig-algebra` y `samooth/zig-zk`
> (septiembre 2026). Ninguno de los dos se adopta como dependencia hoy; este
> documento describe **qué sí serviría** y cómo debería diseñarse la pieza que
> falta para que la adopción tenga sentido.

## 1. Motivación y gap

Estudio previo (resumen):

- **zig-algebra** (v0.3.x, 189+ tests): L0 cripto — campos finitos (M31,
  Goldilocks, BN254, BLS12-381), torres binarias GF(2^n), bigint, hash
  (Blake3/Keccak/Poseidon), merkle (binary/MMR/sparse), NTT, pairing, linalg
  *sobre campos finitos con dimensiones comptime*. Cero aritmética flotante.
- **zig-zk** (v0.1.0, 275 tests, 5 días de antigüedad): L1 proof systems —
  STARK M31 (DEEP-FRI + LogUp lookups), stack Binius casi completo (tower,
  sumcheck, FRI-over-binarias, PCS, recursion Poseidon2), framework AIR
  genérico, Groth16 verifier BN254, transcript Fiat-Shamir.

**Gap**: entre ambos no existe ni una línea de zkML. Buscar
`inference|neural|tensor|llm|gemm` en los dos repos solo produce coincidencias
en comentarios ("tensor-product basis", "tensor M_4"). Los e2e tests son
Fibonacci y un sumador binario.

La librería que sí serviría es, por tanto, una **capa L2 de gadgets ML** (y un
compilador L3 encima), apoyada en L0/L1 existentes. Este documento la
especifica.

Por qué es interesante *aquí* y no solo en abstracto: ktransformers-zig ya
tiene la mitad difícil del witness — kernels cuantizados (INT8/INT4/Q4_K…
MXFP8), MoE con routing, MLA, todo con shapes reales de DeepSeek-V3. El
principio rector del diseño (§4.3) es reutilizar esa ejecución nativa como
generador de witness en lugar de re-ejecutar el modelo dentro de constraints.

## 2. Threat model

zkML tiene tres modos; conviene separarlos porque el coste crece brutalmente:

| Modo | Se prueba | Privacidad | Realista para kt-zig hoy |
|---|---|---|---|
| **(a) Integridad** | La salida fue producida por ejecutar *de verdad* el modelo comprometido W sobre la entrada X | ninguna | **Sí** — es el caso natural de inferencia local/servida |
| (b) Privacidad de pesos | El prover usa W sin revelarlo | pesos | Futuro; exige commitments homomórficos del modelo completo |
| (c) Ambos (zkVM-style) | (a) + (b) | ambos | No con modelos 671B |

El documento diseña para **(a)** con salida a (b) como extensión. En el modo
(a) el statement es:

> "Existe un witness (activaciones intermedias) consistente con los pesos
> comprometidos `root(W)` y el prompt X, tal que el cálculo determinista
> produce exactamente la salida Y".

Casos de uso concretos en nuestro ecosistema:

1. **Attestation de pesos** (sin ZK, prerrequisito): `root(W)` Merkle/serial
   sobre los tensors cargados por `loadWeights` — "sirvo el modelo X sin
   modificar".
2. **Sampling determinista auditable**: seeds de temperature/top-p derivadas
   por Fiat-Shamir del transcript (reproducibilidad verificable).
3. **Inferencia verificable de bloques/capas**: probar un bloque MoE o una
   capa MLA int8 — la unidad económicamente viable.
4. **Modelo completo**: solo con recursion y modelos pequeños; ver tabla de
   costes (§8).

## 3. Estado del arte → lecciones de diseño

| Sistema | Campo | Enfoque | Lección para el diseño Zig |
|---|---|---|---|
| EZKL | BN254 | ONNX → Halo2 circuits, lookups Tabrenheim | El compilador (L3) es la mayor parte del trabajo; los gadgets alone no bastan |
| zkLLM | Goldilocks | Gadgets GEMM/softmax/LayerNorm + comunicación con Llama.cpp para el witness | **Confirmación del patrón §4.3**: inferencia nativa genera witness, el circuito solo verifica |
| Binius / IntegerMatrix | GF(2^n) torres | int8 GEMM bitsliceado sobre torres binarias, sumcheck | El mejor coste probador para datos int8; cada byte son 8 wires "gratis" |
| DeepProve | Goldilocks/Starky | Atención lineal verificable + recursion | La atención es el gadget más caro; empezar por MLP/MoE, no por atención |
| zkML práctico (varios) | M31/QM31 | STARK Circle/Circle-STARK | Lookups (LogUp) para no-linealidades — ya implementados en zig-zk |

Conclusiones de diseño:

- **Dos backend de campo, no uno**: Binius para int8/fp8 (dominio nativo de
  nuestros kernels cuantizados), Goldilocks para fixed-point/fp16 (rangos
  cómodos, inverso barato, 64-bit native).
- **Lookups para todo lo no lineal** (GELU, SiLU/SwiGLU, softmax-step):
  tablas precomputadas verificadas con LogUp, nunca constraints polinomiales
  de alto grado.
- **GEMM verificado por sumcheck/fingerprint, no por producto**: probar
  `C = A·B` con el fingerprint aleatorio `⟨r, C⟩ = ⟨r, A·B⟩` (zkLLM/
  IntegerMatrix style) evita materializar el producto en la traza — O(n²)
  en vez de O(n³).

## 4. Decisiones de diseño nucleares

### 4.1 Campos por tipo de dato

| Dato del kernel | Representación en circuito | Campo/torre | Motivo |
|---|---|---|---|
| INT8 (activaciones/pesos AMX) | 8 wires binarios | Binius GF(2^128) | 1 byte = 8 wires gratis; sumcheck bitslice |
| FP8 (E4M3/E5M2 transporte) | signo+exp+magnitude separados | Binius + lookups de reencuadre | mantisa denormal cases via tabla |
| Q4_K/Q6_K (GGML blocks) | scale (fp16→fixed) + nibbles | Binius para nibbles, Goldilocks para scales | el bloque de escalado es un gadget propio |
| BF16/FP32 (acumuladores, MLA) | fixed-point q16.16 / q32.32 | Goldilocks | rango hasta 2^32 sin overflow con reducción lazy |
| Índices de routing MoE | wires binarios | Binius | el top-k/routing ES lógica binaria |

**Regla**: `Goldilocks` (M31 no) como campo prime por defecto — módulo
2^61-1 soporta acumulaciones q32.32 sin overflow (el módulo 2^31-1 de M31 se
desborda con cualquier acumulación seria de GEMM; M31 solo es aceptable para
trazas STARK circle con reducción por lane).

### 4.2 Sistema de prueba

- **Base**: STARK + FRI con lookups LogUp — exactamente lo que zig-zk `stark`
  ya implementa (M31) y `binius` (torres). Confiabilidad Rust→Zig ya validada
  por su suite (275 tests, e2e prove/verify + fuzz).
- **Verificación compacta**: envoltura Groth16/PLONK del statement STARK solo
  si hace falta on-chain; en nuestro caso (attestation a un auditor con el
  binario) el proof STARK plano ya sirve. Dejarlo como trait opcional, no
  como requisito.
- **Proof size vs prover time**: STARK está bien en el extremo prover-rápido;
  no perseguir SNARKs succinct hasta F4.

### 4.3 Principio rector: la inferencia nativa ES el witness generator

El error clásico que hace a zkML 10x más lento de lo necesario es ejecutar el
modelo *dentro* del circuito (prover-side interpretación de constraints).
Diseño contrario:

1. La inferencia corre por los kernels nativos (`kt_moe_forward`,
   `gemmExpert`, MLA — código existente, velocidad existente).
2. Los kernels emiten *anchos de traza* (activaciones, índices de routing,
   sumas parciales ya reducidas) a un `TraceRecorder` thread-safe.
3. El prover STARK solo interpola la traza, evalúa constraints y hace FRI.
   El coste prover es O(traza · blowup), independiente del cómputo nativo.

Esto convierte a ktransformers-zig de "posible usuario" en **componente
necesario**: la librería zkML sin el runtime nativo no tiene witness barato.

### 4.4 Cuantización como contrato explícito

Cada gadget declara su esquema de cuantización en el tipo (`QuantScheme`),
no en runtime. El verifier conoce el esquema por el statement. La
verificación es bit-exacta contra el esquema — los mismos reducción/acumulación
que hacen los kernels nativos (FP32→BF16 en down-proj, etc.) deben estar
especificados como gadgets de re-cuantización verificables.

## 5. Arquitectura por capas

```
L4  C API                        kt_prove_* / kt_verify_* (patrón kt_kernel.h)
      │
L3  Compilador de modelo         GGUF/safetensors-lite → CircuitGraph
      │                          (schedule de gadgets, alocación de columnas AIR)
      │
L2  Gadgets zkML   ◄── EL GAP    QuantTensor, gemm, dequant/requant, swiglu,
      │                          layernorm, routing (top-k), attention-linear,
      │                          lookups (GELU/SiLU/softmax-step)
      │
L1  Proof systems  (zig-zk)      STARK M31+LogUp, Binius tower/sumcheck/FRI/PCS,
      │                          AIR, transcript, (Groth16 wrapper)
      │
L0  Álgebra        (zig-algebra) Goldilocks, torres GF(2^n), hash, merkle, NTT
```

Módulos de la librería nueva (nombre de trabajo `zig-zkml`, mismo layout de
workspace que zig-zk):

```
libs/
├── tensor/        # QuantTensor + traits de esquema de cuantización
├── gadgets/
│   ├── gemm/      # fingerprint-sumcheck int8 (Binius) y fixed-point (Goldilocks)
│   ├── quant/     # dequant/requant, fp8 reencuadre, GGML block scales
│   ├── nonlin/    # lookups GELU/SiLU/SwiGLU/softmax-step (LogUp)
│   ├── norm/      # layernorm/rmsnorm con sumcheck de sumas
│   ├── routing/   # top-k / group-top2 DeepSeek-V3 como lógica binaria
│   └── attention/ # F5+: atención lineal verificable (ver DeepProve)
├── trace/         # TraceRecorder: captura desde kernels nativos
├── compile/       # CircuitGraph → AIR (columnas, grados, lookup tables)
└── prove/         # orchestración: prove()/verify() por capa, recursion
```

**Build vs vendorear**:

| Pieza | Decisión |
|---|---|
| L0 álgebra | consumir zig-algebra tal cual (path dep o tarball + hash) |
| L1 STARK/Binius/AIR | consumir zig-zk; hacer fork solo si el bloqueo toolchain (§10) persiste |
| L1 Groth16 wrapper | consumir zig-zk `snark` si madura; el verifier de referencia actual (scalar-mul bit a bit, sin MSM) es 100x demasiado lento para F4 |
| L2/L3/L4 | **construir aquí** — no existe en ningún sitio en Zig |

## 6. Sketch de API (Zig 0.16)

Patrones de LESSONS_ZIG.md respetados: ArrayList unmanaged (`.empty`,
`append(allocator, x)`), allocator pasado explícito, dimensiones runtime en
slices (no comptime — aprendido de zig-algebra `linalg`, que es inutilizable
para shapes de LLM), `comptime` solo para esquemas y backend.

### 6.1 Tensors y esquemas de cuantización

```zig
//! libs/tensor/root.zig
const std = @import("std");

/// Esquema de cuantización — parte del TIPO, conocida por el verifier.
pub const Scheme = enum {
    int8_symmetric,   // rango [-127,127], escala power-of-2 por tensor
    int4_gguf_q4_k,   // bloque de 256 con scales fp16 + mins fp16
    fp8_e4m3,         // transporte KT; reencuadre a int8 via lookup
    fixed_q16_16,     // activaciones BF16 dequantizadas para Goldilocks
    fixed_q32_32,     // acumuladores FP32
};

/// Tensor cuantizado: slices runtime, layout row-major.
/// `scales` es por-bloque (GGML) o por-tensor (int8) según `scheme`.
pub fn QuantTensor(comptime scheme: Scheme) type {
    return struct {
        const Self = @This();
        rows: usize,
        cols: usize,
        data: []const u8,      // empaquetado según scheme (nibbles para int4)
        scales: []const u16,   // fp16 bits (GGML) o exponente (int8)
        zeropoints: ?[]const u8 = null,

        pub fn shape(self: Self) struct { usize, usize } {
            return .{ self.rows, self.cols };
        }

        /// Número de wires binarios por elemento (Binius backend).
        pub fn wireWidth(self: Self) usize {
            _ = self;
            return switch (scheme) {
                .int8_symmetric, .fp8_e4m3 => 8,
                .int4_gguf_q4_k => 4,
                .fixed_q16_16 => 32,
                .fixed_q32_32 => 64,
            };
        }
    };
}

/// Backend de campo elegido a partir del esquema — decisión automática.
pub fn FieldOf(comptime scheme: Scheme) type {
    return switch (scheme) {
        .int8_symmetric, .int4_gguf_q4_k, .fp8_e4m3 => binius.Gf2_128,
        .fixed_q16_16, .fixed_q32_32 => zf.Goldilocks,
    };
}
```

### 6.2 GEMM por fingerprint-sumcheck (gadget central)

```zig
//! libs/gadgets/gemm/root.zig

/// Prueba C[m,n] = dequant(A[m,k]) · dequant(B[k,n]) + bias
/// SIN materializar el producto en la traza: el verifier elige un reto
/// aleatorio r ∈ F^n (Fiat-Shamir del transcript) y el prover demuestra
/// ⟨r, C⟩ = Σ_k (⟨r, A[:,k]⟩ · B[k,:] desquantizado) vía sumcheck.
///
/// Coste prover: O(m·k + k·n) field ops + sumcheck O(log n) rounds.
/// Coste traza: solo A, B, C (O(m·k + k·n + m·n)), no O(m·k·n).
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
        c: []const FieldOf(.fixed_q32_32),  // salida dequantizada (el witness)

        /// Devuelve el conjunto de constraints AIR + tablas lookup
        /// que el compilador L3 debe instanciar para este gadget.
        pub fn constraints(self: @This(), gpa: std.mem.Allocator) !Constraints {
            var lut: std.ArrayList(LookupTable) = .empty;
            defer lut.deinit(gpa);
            switch (A_scheme) {
                .int4_gguf_q4_k => try lut.append(gpa, try tables.q4kDequant(gpa, self.a.scales)),
                .fp8_e4m3 => try lut.append(gpa, tables.fp8Reencode),
                else => {},
            }
            return .{
                .kind = .sumcheck_fingerprint,
                .degree = 2,
                .lookups = lut.toOwnedSlice(gpa) catch unreachable,
                .public_scalars = .{ .m = self.m, .k = self.k, .n = self.n },
            };
        }

        /// Extensión a evaluar por el prover: alimenta el sumcheck con
        /// las columnas/polinomios interpolados de la traza.
        pub fn evalFingerprint(
            self: @This(),
            r: []const FieldOf(.fixed_q32_32), // reto del transcript
            out: []FieldOf(.fixed_q32_32),     // ⟨r,C⟩ y sumas parciales
        ) void {
            // Implementación específica del backend (Binius bitslice o
            // Goldilocks multilinear). El witness viene del kernel nativo.
        }
    };
}
```

### 6.3 Lookups para no-linealidades

```zig
//! libs/gadgets/nonlin/root.zig

/// SwiGLU: `gate_out * silu(gate_out)` con SiLU por tabla LogUp.
/// Tabla de 256 entradas int8 (input) → int8 (output): cabe en UNA
/// columna lookup; el rango se prueba por rangos LogUp encadenados.
pub const SwigluLookup = struct {
    /// Tabla precomputada al build time — constante comptime, 0 coste runtime.
    pub const silu_int8: [256]i16 = blk: {
        var t: [256]i16 = undefined;
        for (0..256) |i| {
            const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))) - 0.0);
            const s = x / (1.0 + @exp(-x));
            t[i] = @intFromFloat(@round(s * 256.0));
        }
        break :blk t;
    };

    pub fn constraints() Constraints {
        return .{
            .kind = .logup,
            .degree = 1,
            .lookups = &.{.{ .table = &silu_int8, .width = 8 + 16 }},
        };
    }
};
```

### 6.4 Grabación de witness desde los kernels nativos

```zig
//! libs/trace/recorder.zig — lo que ktransformers-zig exporta como hooks

/// Grabador thread-safe. Los kernels nativos llaman a `record*` tras
/// cada fase; el recorder solo copia a un arena por bloque.
pub const TraceRecorder = struct {
    arena: std.heap.ArenaAllocator,
    mutex: std.Thread.Mutex = .{},
    /// Hash incremental de todo lo grabado — SHAKE/Blake3, para poder
    /// demostrar que el witness corresponde a ESTA ejecución.
    absorb: std.crypto.hash.sha3.Shake256,

    pub fn init(child: std.mem.Allocator) TraceRecorder {
        return .{ .arena = std.heap.ArenaAllocator.init(child), .absorb = undefined };
    }

    pub fn recordQuantTensor(self: *TraceRecorder, tag: []const u8, t: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        _ = tag;
        // copia al arena + absorbe bytes en `absorb`
    }

    pub fn recordRouting(self: *TraceRecorder, expert_ids: []const u32, weights: []const f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        // routing MoE: ids + pesos softmax del gate
    }
};
```

### 6.5 Uso end-to-end (bloque MLP de un experto)

```zig
const zkml = @import("zig-zkml");

// 1. Inferencia NATIVA (kernels existentes, velocidad nativa) con grabación.
var rec = zkml.trace.TraceRecorder.init(gpa);
moe.loadWeights(...); // ya computa el merkle root de pesos (F0, §9)
try moe.forwardGateUpRecorded(0, input, &gate, &up, &rec);
try moe.forwardDownRecorded(0, inter, &out, &rec);

// 2. Compilar el statement de UNA capa.
var graph = try zkml.compile.CircuitGraph.init(gpa);
defer graph.deinit();
try graph.addGemm(zkml.gadgets.gemm.GemmGadget(.int4_gguf_q4_k, .int4_gguf_q4_k){
    .m = 1, .k = 2048, .n = 1408, ... // gate_proj del experto
});
try graph.addSwiglu(zkml.gadgets.nonlin.SwigluLookup);
try graph.addGemm(...); // down_proj

// 3. Prove (STARK Binius sobre la traza grabada).
const statement = try zkml.prove.layerStatement(gpa, &graph, &rec, weights_root);
var proof = try zkml.prove.proveLayer(gpa, params, statement, rec.trace());
defer proof.deinit(gpa);

// 4. Verify — cualquiera con el binario y el merkle root de pesos.
try std.testing.expect(try zkml.prove.verifyLayer(gpa, params, statement, &proof));
```

## 7. Integración con ktransformers-zig

### 7.1 Superficie C API aditiva (sección "Zig extensions" de `kt_kernel.h`)

Siguiendo el patrón establecido (§"C API export" de AGENTS.md; additive-only):

```c
// --- zkML attestation & verifiable inference (experimental) ---
// F0: attestation de pesos
int  kt_weights_merkle_root(KT_MOE* moe, uint8_t root_out[32]);       // Blake3
int  kt_mla_weights_merkle_root(KT_MLA* mla, uint8_t root_out[32]);

// F1: sampling determinista (Fiat-Shamir seed)
void kt_transcript_seed(const uint8_t* context, size_t ctx_len,
                        uint8_t seed_out[32]);

// F2+: verifiable inference por capa (handles opacos, patrón KT_MOE)
typedef struct KT_PROVER KT_PROVER;
KT_PROVER* kt_prover_new(const KT_MOE* moe, const uint8_t* weights_root);
void kt_prover_free(KT_PROVER* prover);
int  kt_prove_moe_layer(KT_PROVER* prover, int layer_idx,
                         const void* input, void* output,
                         uint8_t** proof_out, size_t* proof_len);
int  kt_verify_moe_layer(const KT_MOE* moe, const uint8_t* weights_root,
                         int layer_idx, const void* input, const void* output,
                         const uint8_t* proof, size_t proof_len);
```

Reglas de la casa que aplican:

- Opaque handles `opaque {}` + cast en wrappers (`kt_moe_free` style).
- **Allocator capturado en `*_new`** (regla B1) — `kt_prover_free` nunca lee
  el default actual.
- Declarar `pub export fn` en `main.zig` para los tests directos; añadir al
  gate de `tools/verify_abi.py` (exports + arity) y `tools/audit_layout.py`.
- Forzar emisión vía `comptime { _ = &mod.fn; }` en `root.zig` (trampa de
  lazy-analysis) y verificar con `nm`.

### 7.2 Flujo del witness

```
Python (pybind11)          .so (Zig)
─────                      ─────────────────────────────────────────
load_weights ───────────►  moe.loadWeights()
                              └─► F0: hash Blake3 incremental de TODOS los
                                   tensors → root expuesto por
                                   kt_weights_merkle_root
forward ────────────────►  kt_moe_forward(...)
                              └─► si hay KT_PROVER registrado: hooks
                                   recordQuantTensor tras gate/up/down y
                                   recordRouting tras el gate softmax
kt_prove_moe_layer ─────►  L3 compile → L2 constraints → L1 STARK FRI
kt_verify_moe_layer ────►  L1 verify (standalone, sin L2/L3, sin runtime)
```

Puntos de calzada concretos en el código actual:

- `TpMoe.forwardGateUp`/`forwardDown` (`src/kernels/moe/moe.zig`): ya
  separan el paso por experto — el hook de recorder es un parámetro
  opcional, cero coste cuando null (branch predecible).
- Routing: ya implementado el group-top2 DeepSeek-V3 (D4) — el witness de
  routing son exactamente esos `expert_ids` + pesos, ya computados.
- MLA queda **fuera** de F0–F3 (atención es el gadget más caro; DeepProve
  confirma que requiere diseño propio — F5).

### 7.3 Tests

- Unittest de gadgets contra los kernels: `gemmExpert` produce C; el gadget
  verifica el mismo C desde los mismos A,B — **el test de integración es que
  prove+verify pase sobre el witness del kernel nativo, y que un witness
  manipulado (±1 ulp en un elemento) RECHACE** (negativos obligatorios,
  lección del guard AMX A2: un check que nunca falla no es un check).
- Reusar `tools/test_runner.zig` simple mode; suites con `addRunArtifact`.
- Bench: extensión de `bench/gemm_bench.zig` midiendo overhead prover por
  GEMM (target inicial: <50x por capa int8 — ver §8).

## 8. Costes realistas

Overhead prover (órdenes de magnitud, STARK/FRI con blowup 2^3-2^4):

| Operación | Cómputo nativo | Coste prover | Ratio |
|---|---|---|---|
| Attestation Merkle de pesos (F0) | 1 lectura lineal | ~1x | **~1x** |
| GEMM int8 1×2048×1408 (decode MLP) por fingerprint | ~5.7 M MACs | sumcheck + interp: ~10-30 ms | ~10-50x |
| Bloque MoE (3 GEMM + SwiGLU + routing) | ~0.5 ms nativo | ~100-500 ms | ~10³x por bloque |
| Atención (MLA decode, 1 capa) | ~2 ms | gadgets lineal-attn: horas-hombre de diseño, coste ~10⁴x | F5 |
| DeepSeek-V3 completo (61 capas × MoE) | ~1 s/capa | ~30-60 s/capa | **inviable hoy sin recursion + modelos pequeños** |

Veredictos por caso:

- **F0 attestation de pesos**: coste ~1x, cero riesgo cripto (solo Blake3 +
  merkle). Hacer YA — no requiere zig-zk, solo zig-algebra `merkle` (o 200
  líneas propias: `MerkleTree(Blake3)` ya existe verificado).
- **F1 transcript determinista**: barato, valor de producto moderado.
- **F2 gadgets int8 + GEMM fingerprint**: el primer paso cripto real; el
  backend Binius de zig-zk es el adecuado y sus 275 tests bajan el riesgo.
- **F3 capa MoE verificable**: demostrable con las shapes reales de
  DeepSeek-V3 (2048/1408 del experto) — el objetivo de "viable hoy".
- **F4/F5 modelo completo / atención**: no prometer nada; el estado del arte
  zkML verifica modelos de ≤1B params con equipos dedicados.

## 9. Roadmap F0–F4 con criterios build-vs-vendorear

| Fase | Entregable | Deps | Criterio go/no-go |
|---|---|---|---|
| **F0** | `kt_weights_merkle_root` sobre `loadWeights` (moe+mla); test root estable ante reorden de lectura; python `verify_weights.py` | zig-algebra `merkle` (o propio) | overhead de carga <5% |
| **F1** | `kt_transcript_seed`; sampling reproducible testado | zig-algebra `hash` | cero cambio en kernels |
| **F2** | lib `tensor` + `gemm` fingerprint int8; test positivo+negativo vs `gemmExpert` | zig-zk `binius` (spike toolchain §10) | overhead <50x en decode shape |
| **F3** | `kt_prove_moe_layer`/`kt_verify_moe_layer` para 1 experto; bench por bloque | zig-zk `stark`/`air` | proof <1 MB, verify <100 ms |
| **F4** | multi-bloque + recursion (Poseidon2 ya en zig-zk); Groth16 wrap solo si on-chain | zig-zk `snark` (si madura) | decisión de producto |

Vendorear (path-dep + tarball hash) por defecto; migrar a versiones taggeadas
cuando zig-zk/zig-algebra publiquen releases (hoy: ninguno tiene).

## 10. Apéndice: bloqueo toolchain y mitigaciones

- Nuestro toolchain: `0.16.0-dev.2535`. zig-algebra/zig-zk declaran
  `minimum_zig_version = "0.16.0"`; semver `0.16.0-dev` < `0.16.0`, así que
  **el package manager puede rechazar la dep directamente**. Primer paso de
  F2: spike `zig fetch` + path-dep para confirmar (2 min).
- Mitigaciones: (1) path dependency con `--dep` override; (2) vendor de
  `libs/field`+`libs/merkle`+`libs/binius` (licencia MIT/Apache-2.0 lo
  permite); (3) actualizar el toolchain del repo a 0.16.0 release (decisión
  mayor, tocar build.zig y LESSONS_ZIG.md).
- Riesgo upstream: ambos repos son de autor único, 0 stars, 5 días de
  antigüedad, sin tags zig-zk. F0/F1 (solo zig-algebra) son bajo riesgo;
  para F2+ conviene un fork propio con nuestros fixes.

---

*Este documento es un diseño de referencia, no un plan comprometido. El único
item accionable barato hoy es F0 (attestation de pesos); el resto requiere
decisión explícita de producto.*
