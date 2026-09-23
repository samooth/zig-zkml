# PLAN — Multi-Engine Compatibility (llama.cpp, vLLM, zig-ai, ktransformers-zig)

> Redesign de zig-zkml para ser compatible con múltiples motores de inferencia,
> no solo ktransformers. Supersedes la sección de glue `kt_*` puntual de
> `BLUE_PRINT.md` §8 con una matriz de adapters por motor.
>
> **Decisiones del usuario (confirmadas):**
> 1. Staging: `0 → 5 → 1 → 2 → 3 → 4` (contract → witness ABI → motores).
> 2. Adapters viven **dentro** de zig-zkml (`adapters/<engine>/`).
> 3. llama.cpp: **wrapper separado** (zero-fork, API pública).
> 4. vLLM: **ctypes** sobre `.so` (más simple).
> 5. Witness recording hooks **incluidos** ahora (no diferidos a F2).

---

## 0. Hallazgos de exploración

| Motor | Repo | Ruta de pesos | Mejor hook para zig-zkml |
|---|---|---|---|
| ktransformers-zig | `ktransformers-zig/` | GGUF v3, loader propio | Glue `kt_*` planificado (referencia) |
| llama.cpp | `llama.cpp/` | GGUF via `llama-model-loader` | **API pública**: `llama_model_init_from_user` + `llama_model_set_tensor_data_t` (cada `ggml_tensor*` con nombre/shape/type/data); `cb_eval` para decode-time; prior art `examples/gguf-hash/` |
| zig-ai | `zig-ai/` | GGUF + safetensors, Zig | **Mismo lenguaje**: import directo del módulo `zig_zkml`; contrato `MetricHooks` existente (`src/engine_api/contract.zig`) |
| vLLM | `vllm/` | safetensors/HF via `BaseModelLoader` | Loader subclass + **ctypes/cffi** sobre `libzkml.so` |
| Otros (colibri, koboldcpp, PowerInfer, BitNet, beellama.cpp) | — | linaje GGUF | Mismos patrones que llama.cpp / Python |

**Insight clave:** zig-zkml ya es ~95% agnóstico al motor. El acoplamiento actual
es solo (a) doc/comments `kt_*`, (b) tipos con forma MoE (`SlotKey{layer,op,expert,tp_rank}`,
`expert_ids`), (c) ABI diseñado para las convenciones de glue de kt. Los símbolos
`kt_*` **no existen en este repo** — viven (vivirán) en el lado del motor.

---

## Wrapper separado para llama.cpp — pros / contra

### Pros
1. **Zero fork maintenance** — sobrevive rebases upstream (llama.cpp se mueve a diario).
2. **Los hooks fuertes ya son públicos** — `llama_model_init_from_user` +
   `set_tensor_data` entrega cada `ggml_tensor*`; `cb_eval` observa ops en decode.
3. **Patrón conocido** — igual que `examples/gguf-hash` (ya trae SHA-256 en-tree).
4. **Opt-in** — sin cambio de comportamiento para usuarios vanilla de llama.cpp.
5. **Testeable standalone** — binario propio; CI corre wrapper + gguf diminuto → root.
6. **Apalancamiento** — koboldcpp, PowerInfer, beellama.cpp, BitNet comparten
   `llama.h` → **un wrapper cubre 5+ motores**.

### Contras (con mitigación)
1. **No intercepta apps existentes** sin que usen el load path del wrapper →
   aceptable: el attestation es propiedad de *cómo cargaste*, opt-in es correcto.
2. **mmap fast-path**: `init_from_user` hace que el wrapper sea dueño de la
   materialización → hashear through-mapping en `set_tensor_data` (una pasada
   de lectura, no una copia).
3. **ABI más ancho que un patch** — trackear firma de `llama_model_init_from_user`
   (rara, versionada en `llama.h`) → pinneamos commit en README del adapter.
4. **Witness en decode** depende de `cb_eval` al crear contexto → el wrapper
   expone `zkml_llama_create_context()` que lo inyecta.

**Veredicto: los pros dominan.** Un patch solo ganaría si necesitáramos bytes de
`load_data_for()` específicamente, pero `set_tensor_data` los cubre.

---

## Stages

### Stage 0 — Core de-kt + engine contract (prerrequisito)

**Goal:** quitar lenguaje específico de ktransformers; añadir header universal de contrato.

| Archivo | Cambio |
|---|---|
| `libs/api.zig` | Doc: "ktransformers-zig" → "host inference engine" |
| `include/zkml_c.h` | Reordenar docs; **las 9 firmas `zkml_*` sin cambios** (ABI v1) |
| `libs/merkle.zig`, `libs/attestation.zig`, `libs/trace/root.zig`, `zkml.zig` | Quitar referencias doc a kt |
| **NUEVO `include/zkml_engine.h`** | Contrato del adapter: callbacks `zkml_weight_stream`, hooks de witness (`zkml_witness_begin_layer/record_op/end_layer`), secuencias de ciclo de vida (load→attest, forward→record, prove→verify), checklist por stage (F0 attestation-only vs F2 full proof) |
| `BLUE_PRINT.md` §8, `TODO.md` | Reemplazar sección glue `kt_*` por **matriz de adapters por motor** |
| `build.zig` | Añadir `adapters` al gate `fmt` |

**Gate:** `zig build test --summary all` (67/67), `zig build abi` intacto, `zig build fmt`.

**Commit:** `refactor: de-ktransformers core, add engine adapter contract (zkml_engine.h)`

---

### Stage 5 — Witness hook ABI (solidificación cross-cutting) — ✅ DONE

Incluido ahora (decisión del usuario), no diferido a F2.

**Implementado**: `ZKML_Witness` handle + `zkml_witness_session_create/begin_layer/record_op/end_layer/finalize/session_destroy`, `ZKML_SlotKey` (8 B, alignment-packed: layer u32, expert u16, op u8, rank u8), `ZKML_OP_*`, `ZKML_ABI_VERSION = 2`. Pruebas 69/69.

| Archivo | Cambio |
|---|---|
| `libs/api.zig` | Surface C aditivo: `zkml_witness_session_create/record_op/finalize` (ABI **v2**). `record_op` toma struct plana equivalente a `SlotKey` (layer, op, expert, tp_rank + ptr/len payload) |
| `include/zkml_c.h` | Declarar funciones v2 (aditivo-only, append) |
| `libs/trace/root.zig` | Entrada C-friendly para `TraceRecorder` (ingest por puntero crudo, además de API Zig) |
| `libs/statement/root.zig` | `expert_ids` opcional (dense pasa NULL/0) — ya genérico |
| `tools/abi_check.zig` + `build.zig` nm gate | Assert nuevos símbolos witness |
| `include/zkml_engine.h` | Documentar ciclo witness: `begin_layer → record_op* → end_layer → finalize` |

**Gate:** test de determinismo (multi-thread → mismo hash) sigue pasando; nuevo round-trip witness.

**Commit:** `feat: witness recording ABI (v2) — engine-agnostic SlotKey ingest`

---

### Stage 1 — llama.cpp wrapper (primer motor) — ✅ DONE

**Layout:** `adapters/llama_cpp/` dentro de zig-zkml.

**Implementado:** `zkml_llama_attest_gguf(path, root_out)` +
`zkml_llama_attest_gguf_manifest(path, root_out, manifest_path)`, shared
`libzkml_llama.so`, test gateado (positivo + determinismo + negativo), y
`zig build llama-adapter` (CMake + test + cross-check Python). Añadido
también `zkml_allocator_process()` al core (allocator Zig estable para
consumidores C puros) y el target **shared** `libzkml.so` en `build.zig`.

| Archivo | Rol |
|---|---|
| `adapters/llama_cpp/wrapper.cpp` | **Attestation**: recorre el GGUF con el lector público `gguf.h` (metadata `no_alloc` + lectura *streamed* de cada tensor) → `zkml_attestor_add/finish/root`; emite manifest JSON para el auditor |
| `adapters/llama_cpp/zkml_llama.h` | API pública del wrapper: `zkml_llama_attest_gguf` (+ variante manifest) |
| `adapters/llama_cpp/CMakeLists.txt` | Build standalone de `libzkml_llama.so` + `zkml_llama_test` (linkea `libggml-base.so` + `libzkml.so`) |
| `adapters/llama_cpp/README.md` | Build, commit pineado de llama.cpp (`1c3c9674d`, v0.0.10269), notas de diseño |
| `adapters/llama_cpp/test_attest.cpp` | Positivo + determinismo + **negativo** (1 byte corrupto → root distinto); emite `root.hex` + `manifest.json` |
| `tools/integration/verify_llama_adapter.py` | Wrapper fino que delega en `tools/verify_weights.py` (auditor independiente) |
| `libs/api.zig` + `include/zkml_c.h` | `zkml_allocator_process()` — allocator de proceso para consumidores C (aditivo) |
| `build.zig` | Target shared `libzkml.so`; paso `llama-adapter` (no bloquea el build default) |

**Desviación del plan (documentada):** la ruta (a) usa el lector
`gguf.h` público en vez de `llama_model_init_from_user` + `set_tensor_data`.
Razón: el hook `set_tensor_data` se invoca *antes* de que existan los
bytes (el callback debe *rellenar* el tensor, no observarlo), así que para
atestar un fichero GGUF haría falta un canal lateral con mmap del fichero
y el resultado sería idéntico. La materialización del modelo (y por tanto
el hook) sólo aporta valor cuando además se va a hacer inferencia — eso es
la ruta (b) de witness con `cb_eval`, que llega con la integración F2.
Cubre los mismos motores (koboldcpp, PowerInfer, beellama.cpp, BitNet
comparten GGUF/ggml) sin fork.

**Build:** `zig build llama-adapter` (invoca CMake; no toca build default).
llama.cpp se toma como checkout hermano: `-Dllama-dir=/ruta/a/llama.cpp`.

**Gate (verde):** test wrapper (positivo+negativo), `nm -D` muestra
`zkml_llama_*` + `zkml_*` heredados, y `tools/integration/verify_llama_adapter.py`
verifica el root contra el auditor independiente.

**Commit:** `feat(adapters): llama.cpp attestation + witness wrapper (zero-fork, public API)`

---

### Stage 2 — zig-ai direct module (más fácil — mismo lenguaje)

**Layout:** `adapters/zig_ai/` — al ser ambos Zig, el "adapter" es mostly wiring
doc + ejemplo + snippet de patch.

| Archivo | Rol |
|---|---|
| `adapters/zig_ai/integration.zig` | Referencia: cómo el `build.zig` de zig-ai añade `b.dependency("zig_zkml", ...)` e importa `zig_zkml` |
| `adapters/zig_ai/gguf_attestation.zig` | Wraps iteración de tensores en `src/loader/gguf.zig` / `gguf_model.zig`: tras cargar cada tensor → `zkml_attestor_add(name, bytes)` → `zkml_attestor_root` |
| `adapters/zig_ai/witness_hooks.zig` | Mapea el contrato existente **`MetricHooks`** (`src/engine_api/contract.zig`) de zig-ai a `TraceRecorder`: `LayerMetrics` → `SlotKey{layer, op, ...}` |
| `adapters/zig_ai/README.md` | Diffs/patch exactos para `build.zig` + loader de zig-ai (repo hermano — documentamos el patch, no lo editamos desde aquí) |
| `adapters/zig_ai/test_adapter.zig` | Unit test: stream sintético de tensores → estabilidad de root bajo reorden |

**Por qué funciona:** sin FFI, sin CMake; zig-ai ya tiene loader GGUF + hooks; su
sistema de módulos es deps de `build.zig` — adición de 3 líneas.

**Gate:** `zig build test` recolecta tests del adapter (añadir a imports de test de `zkml.zig`).

**Commit:** `feat(adapters): zig-ai GGUF attestation + MetricHooks→TraceRecorder witness bridge`

---

### Stage 3 — vLLM ctypes binding (Python, camino más simple)

**Goal:** shared library + módulo Python fino — sin extensión compilada.

| Archivo | Cambio |
|---|---|
| `build.zig` | Añadir target **shared** junto al estático: `b.addLibrary(.{ .linkage = .dynamic, .name = "zkml" })` → `zig-out/lib/libzkml.so`. `.a` sigue default |
| `adapters/vllm/zkml.py` | Loader `ctypes.CDLL`; wrappers Python: `class Attestor` (context manager: `add(name, bytes)` → `finish()` → `.root` bytes), `verify(proof, expected_root)`, `transcript_seed(ctx)`. `argtypes`/`restype` type-safe |
| `adapters/vllm/model_loader.py` | `ZkmlAttestedLoader(BaseModelLoader)` — subclass del registry de vLLM, itera shards safetensors vía `safe_open`, streamea cada tensor al `Attestor`, expone `model.weights_root`. Registrado vía `register_model_loader("zkml_attested")` |
| `adapters/vllm/witness.py` | Witness bridge: wraps `forward_context` / custom-op hooks de vLLM → serializa I/O de ops → alimenta `TraceRecorder` vía llamada C en batch (nuevas `zkml_witness_*` en `libs/api.zig`, ABI **v2 aditivo**) |
| `adapters/vllm/test_zkml.py` | pytest: attestation sobre tensoeros fake → root; byte corrupto → root cambia; round-trip `verify` |
| `adapters/vllm/pyproject.toml` | Paquete `zkml-vllm`, snippet de entry-point para `vllm.general_plugins` |

**ABI:** llamadas witness son **aditivas** (`ZKML_ABI_VERSION 1 → 2`); las 9
funciones existentes intactas — llama/kt adapters siguen en v1.

**Gate:** `zig build test` (tests nuevos `zkml_witness_*`), `zig build abi` (nm assert), `pytest adapters/vllm/`.

**Commit:** `feat(adapters): shared lib + vLLM ctypes loader & witness bridge`

---

### Stage 4 — ktransformers-zig reference adapter (target original, ahora último)

**Layout:** `adapters/ktransformers/` — implementa el glue `kt_*` que
`TODO.md`/`zkML.md` ya especifican, pero **llama al contrato `zkml_engine.h`**
como todos los demás.

| Archivo | Rol |
|---|---|
| `adapters/ktransformers/kt_glue.c` | Los 4 wrappers planificados: `kt_weights_merkle_root`, `kt_mla_weights_merkle_root`, `kt_transcript_seed`, + witness hooks del recorded mode de `kt_kernel.h` |
| `adapters/ktransformers/README.md` | Mapea nombres `kt_*` viejos → llamadas `zkml_*` genéricas (nota de migración para ktransformers-zig) |
| `adapters/ktransformers/test_glue.c` | Mismos tests de attestation positivo/negativo que el adapter de llama |

**Posicionamiento:** este es el **adapter de referencia** — el más simple; los
demás adapters se miden contra su checklist.

**Gate:** `zig build kt-adapter` + su test.

**Commit:** `feat(adapters): ktransformers-zig reference glue (kt_* → zkml_* mapping)`

---

## Orden de stages y dependencias

```
Stage 0 (contract + de-kt)          ← bloquea todo, pequeño
   │
   ├─► Stage 5 (witness ABI v2)     ← necesario por witness de 1(b)/2/3;
   │                                    path attestation-only de 1/2/4 puede
   │                                    empezar tras Stage 0
   ├─► Stage 1 (llama.cpp wrapper)
   ├─► Stage 2 (zig-ai module)
   ├─► Stage 3 (vLLM ctypes + shared .so)
   └─► Stage 4 (kt reference)       ← último, más simple, valida el contrato
```

**Orden de ejecución elegido:** `0 → 5 → 1 → 2 → 3 → 4`
(contract → witness ABI → motores de menor a mayor fricción de integración:
wrapper llama autocontenido, zig-ai mismo lenguaje, vLLM necesita FFI+shared lib,
kt último como referencia).

---

## Reglas transversales (todos los stages)

- **ABI solo aditiva** — las 9 funciones v1 nunca cambian; lo nuevo se agrega.
- **Los adapters nunca editan repos hermanos** — cambios en zig-ai / llama.cpp /
  vLLM / ktransformers-zig se documentan como patches/READMEs dentro de
  `adapters/<engine>/`.
- **Cada adapter shippea su test negativo** (byte corrupto → root cambia /
  verify rechaza).
- **`tools/verify_weights.py` sigue siendo el auditor Python independiente** —
  el root de cada adapter se cross-checa contra él.
- **`zig build test` default debe seguir verde** (hoy 67/67); los adapters añaden
  tests pero no bloquean el core si su toolchain (CMake, Python) no está presente
  — usar availability checks o steps separados como los existentes
  `zig build abi/verify/spike`.

---

## Fuera de scope de este plan

- El backend STARK F2 (composición de constraints, etc.) — sigue en `TODO.md`.
  Este plan solo **conecta motores al core ya genérico**.
- Editar repos upstream (llama.cpp, vllm, zig-ai, ktransformers-zig) — todos los
  cambios caen en zig-zkml.
- Groth16 / recursión (F4) — intacto.

---

## Pendiente de confirmación

- [ ] Commit/tag pineado de llama.cpp para el README del adapter (usar el
      `LLAMA_API` version actual del checkout si no hay preferencia).
