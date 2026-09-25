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
| `BLUE_PRINT.md` §8 y §11 | Reemplazar sección glue `kt_*` por **matriz de adapters por motor** |
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
| `tools/integration/verify_adapter_root.py` | Wrapper fino que delega en `tools/verify_weights.py` (auditor independiente) |
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
`zkml_llama_*` + `zkml_*` heredados, y `tools/integration/verify_adapter_root.py`
verifica el root contra el auditor independiente.

**Commit:** `feat(adapters): llama.cpp attestation + witness wrapper (zero-fork, public API)`

---

### Stage 2 — zig-ai direct module (mismo lenguaje) — ✅ DONE

**Layout:** `adapters/zig_ai/` — al ser ambos Zig 0.16.0, el "adapter" es
wiring: import de módulo, sin FFI ni CMake.

**Implementado:** `TensorSource` (interfaz estructural `count`/`name`/`data`)
+ `attestSource` → root, `ProofSession`, parser de nombres GGUF, y el bridge
de witness sobre `MetricHooks`. 11 tests en el gate default (`zig build
test` = 81/81). `integration.zig` contiene el glue compilable dentro de
zig-ai (`GgufSnapshot` sobre `GgufFile`); los patches exactos para el repo
hermano están en el README del adapter.

| Archivo | Rol |
|---|---|
| `adapters/zig_ai/root.zig` | Raíz del módulo: re-exporta las piezas compiladas |
| `adapters/zig_ai/gguf_attestation.zig` | `TensorSource` + `attestSource` (tabla de tensores → root Merkle), `ProofSession`, clasificación de nombres GGUF (`blk.N.*`, `mlp.*`, `feed_forward.*`) |
| `adapters/zig_ai/witness_hooks.zig` | `Recorder` (estado del witness ABI) + bridge `MetricHooks.on_layer` (sesión `threadlocal`, porque el contrato no lleva userdata) |
| `adapters/zig_ai/abi.zig` | Único punto de contacto con los punteros crudos del C ABI |
| `adapters/zig_ai/integration.zig` | `GgufSnapshot` sobre `GgufFile.tensors` + `tensorData()` (mmap, zero-copy) — se compila **dentro** de zig-ai |
| `adapters/zig_ai/README.md` | Patches exactos: `build.zig.zon` (dep path), módulo en `build.zig`, hook de attestation al cargar, hook de witness |
| `adapters/zig_ai/test_adapter.zig` | Gate: orden-independencia, byte corrupto, duplicados, acuerdo con el core, proof round-trip, parseo de nombres, witness determinista y negativos de máquina de estados |
| `build.zig` | Módulo de test propio para el adapter con import `zig_zkml` (así el engine lo consume igual: dependencia de path + módulo) |

**Hallazgo que condiciona el diseño:** el build de zig-ai usa
`b.createModule` y **nunca** `b.addModule`, así que no existe módulo
exportado a nivel de paquete del que depender. De ahí la interfaz
estructural `TensorSource` en vez de importar `gguf` desde aquí; el binding
concreto (`GgufSnapshot`) vive en `integration.zig` y lo compila el engine.

**Bug real encontrado por estos tests (core):** `merkle.Builder.deinit`
asignaba `.empty` a las `ArrayList` sin liberar su buffer (fuga en el path de
nombres duplicados / attestor destruido a medio cargar). Corregido en
`libs/merkle.zig`.

**Gate:** `zig build test` recolecta los tests del adapter (11 tests, gate
default verde: 81/81).

**Commit:** `feat(adapters): zig-ai GGUF attestation + MetricHooks→TraceRecorder witness bridge`

---

### Stage 3 — vLLM ctypes binding (Python, camino más simple) — ✅ DONE

**Goal:** shared library + módulo Python fino — sin extensión compilada.

**Implementado:** binding `ctypes` completo sobre `libzkml.so` con
`argtypes`/`restype` explícitos, `ZkmlAttestedLoader` (load format
`zkml_attested`, registrado por el entry point `vllm.general_plugins`),
bridge de witness con `WitnessRecorder`/`LayerScope`, y 15 tests
**herméticos** (sin pytest, sin torch, sin vLLM) que entran en el gate
default vía `zig build verify`.

| Archivo | Cambio |
|---|---|
| `build.zig` | Paso `vllm-adapter` (python3 + test) y su inclusión en `verify`; el target shared ya existía (Stage 1) |
| `adapters/vllm/zkml.py` | `ctypes.CDLL` + firmas tipadas; `Attestor` (context manager, `add`/`finish`/`root`/`proof`), `Witness`, `SlotKey` (8 B, offsets 0/4/6/7), `verify_proof`, `transcript_seed`, `attest_items`; descubrimiento de librería (`$ZKML_LIB` → `zig-out/lib/libzkml.so`) |
| `adapters/vllm/model_loader.py` | `ZkmlAttestedLoader`: hace *tee* del iterador `(name, tensor)` que vLLM ya produce (`safetensors_weights_iterator`) hacia el attestor; publica `model.weights_root`. La clase se construye on-demand (`build_loader_cls()`) para que el módulo importe sin vLLM |
| `adapters/vllm/witness.py` | `WitnessRecorder` (begin/end en un hilo, `record_op` concurrente) + `LayerScope` + helpers de custom-op |
| `adapters/vllm/test_zkml.py` | 15 tests con runner propio (compatible con pytest si está): determinismo, orden, **byte corrupto**, duplicados, vacío, proof round-trip, witness + estado + concurrencia, layout de `SlotKey`, y **cross-check contra `tools/verify_weights.py`** |
| `adapters/vllm/pyproject.toml` | Paquete `zkml-vllm` (nombre de import `zkml_vllm`, para no ensombrecer el paquete `vllm` real); entry point `zkml_vllm.model_loader:register` |
| `adapters/vllm/README.md` | Instalación, qué se attestea (nombres del checkpoint, pre-`WeightsMapper`; bytes crudos) y análisis de los mecanismos de witness de vLLM |

**Decisión de diseño (atestestación):** se hashean los **nombres del
checkpoint** y los **bytes en disco** sin convertir, no los parámetros
fusionados del engine (`q_proj` → shard `qkv_proj`): una raíz publicada
describe el artefacto. El tee va *dentro* del generador, así que los tensores
que vLLM filtra (p. ej. poda por expert-parallel) no entran en la raíz sin
querer.

**Hallazgo (witness):** vLLM no tiene hook de I/O por op en esta revisión —
los module hooks se saltan bajo `torch.compile`/CUDA graphs, los passes de
Inductor sólo ven el `fx.Graph` de compilación, y `set_forward_context` sólo
trae metadatos de batch. El adapter aporta el lado de grabación y documenta
que el hook debe ser explícito (`direct_register_custom_op` /
`CustomOp.register_oot`), en vez de fingir un hook que no existe.

**Gate:** `zig build verify` incluye ahora los 15 tests del adapter; `zig
build test` (81/81), `abi`, `fmt` y `llama-adapter` siguen verdes.

**Commit:** `feat(adapters): shared lib + vLLM ctypes loader & witness bridge`

---

### Stage 4 — ktransformers-zig reference adapter — ✅ DONE

**Layout:** `adapters/ktransformers/` — el glue `kt_*` → `zkml_*` que
`zkML.md` ya esbozaba, pero llama al contrato genérico
(`zkml_engine.h`) como todos los demás.

**Implementado:** `libkt_zkml_glue.so` compilado contra el `kt_kernel.h` del
propio engine y enlazado a su `.so` prebuilt, con primitiva genérica
`kt_zkml_attest_tensors`, helpers tipados para MoE (BF16) y LlamaMoe
(bloques GGUF), `kt_zkml_transcript_seed` y el shim `kt_zkml_witness_*`.
Gate `zig build kt-adapter` con 9 checks + cross-check del auditor.

| Archivo | Rol |
|---|---|
| `adapters/ktransformers/kt_glue.h` | API `kt_zkml_*` sobre `kt_kernel.h` + `zkml_c.h`; `kt_zkml_tensor_view` |
| `adapters/ktransformers/kt_glue.c` | Primitiva genérica, helpers MoE/LlamaMoe, seed F1, shim de witness, `kt_zkml_cpu_variant` |
| `adapters/ktransformers/test_glue.c` | Positivo, determinismo, orden, **negativo**, validación de argumentos, configs BF16 y Q8_0, seed, witness, variante del engine |
| `adapters/ktransformers/CMakeLists.txt` | Build standalone (linkea `libkt_kernel_ext_<variant>.so` + `libzkml.so`) |
| `adapters/ktransformers/README.md` | Tabla de mapeo `kt_*` → `zkml_*` + los huecos documentados |
| `build.zig` | Paso `kt-adapter` (CMake + test + `verify_adapter_root.py`) |

**Hallazgos que corrigieron el plan (los `kt_*` de `zkML.md` nunca se implementaron):**

1. **`kt_weights_merkle_root` / `kt_mla_weights_merkle_root` no existen** —
   sólo están propuestos en `zkML.md`. El glue los provee con prefijo
   `kt_zkml_` (no reclama esos nombres: si el engine los implementa un día,
   dos `.so` con el mismo símbolo harían ambiguo el binder).
2. **Los handles `KT_MOE`/`KT_MLA` son opacos y no hay getter de pesos ni
   callback por tensor** — `kt_moe_load_weights` sólo copia desde los
   punteros del config que el caller conserva. Por eso los helpers tipados
   attestan **el config** que se va a pasar a `kt_moe_new`: los bytes exactos
   que el engine copiará, hasheados antes de la copia.
3. **No hay recorded mode**: `kt_kernel.h` no expone ciclo begin/end de
   grabación, ni tipo de registro, ni replay. El único cache interno
   (`ForwardCache`, path SFT) es para backprop, no es alcanzable desde C y
   no es replayable. `kt_zkml_witness_*` queda como la superficie de hook
   que el engine debería invocar cuando exista, probada hoy.
4. **MLA/DSV3 sin shapes en el header** (falta `v_head_dim`): no hay helper
   tipado; se documenta el uso de la primitiva genérica con views del
   caller. Igual con `gate/up/down_scale`: su layout depende de
   `quant_config` y no está fijado en C, así que se excluyen en vez de
   adivinar.
5. **Los tamaños salen del engine**: `kt_type_row_bytes(n, type)` (exportado
   por la `.so` prebuilt) en vez de duplicar la tabla de tamaños de bloque;
   si no puede dimensionar una fila devuelve 0 y el glue falla en vez de
   hashear un número equivocado de bytes. Multiplicaciones con overflow
   check.

**Gate:** `zig build kt-adapter` verde (9/9 checks, root `7cf94ce9…`
verificado por el auditor independiente). `zig build test` (81/81), `abi`,
`verify`, `llama-adapter` y `vllm-adapter` siguen verdes.

**Commit:** `feat(adapters): ktransformers-zig reference glue (kt_* → zkml_* mapping)`

---

## Estado del plan multi-motor

| Stage | Adapter | Mecanismo | Gate |
|---|---|---|---|
| 0 | core de-kt + `zkml_engine.h` | — | `zig build test` |
| 5 | Witness ABI v2 | — | `zig build test` |
| 1 | `adapters/llama_cpp/` | wrapper C++ zero-fork sobre `gguf.h` | `zig build llama-adapter` |
| 2 | `adapters/zig_ai/` | import de módulo Zig | `zig build test` (11 tests) |
| 3 | `adapters/vllm/` | ctypes sobre `libzkml.so` | `zig build verify` (15 tests) |
| 4 | `adapters/ktransformers/` | glue C de referencia | `zig build kt-adapter` |

Los cuatro adapters están implementados y verificados contra el auditor
independiente. Pendiente para F0: **medir el overhead de carga** (<5%) con
pesos reales, y decidir si `kt_glue`'s exclusión de escalas/zero-points y la
falta de shapes MLA/DSV3 se resuelven upstream (documentado, no parcheado
aquí).

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

- El backend STARK F2 (composición de constraints, operand binding,
  chunking 16 MACs, AIR de routing, núcleo LogUp y el AIR float bit-exacto)
  — vive en `BLUE_PRINT.md` §11, con el ROADMAP ya reorderado. Este plan
  solo **conecta motores al core ya genérico**.
- Editar repos upstream (llama.cpp, vllm, zig-ai, ktransformers-zig) — todos los
  cambios caen en zig-zkml.
- Groth16 / recursión (F4) — intacto.

---

## Pendiente de confirmación

- [ ] Commit/tag pineado de llama.cpp para el README del adapter (usar el
      `LLAMA_API` version actual del checkout si no hay preferencia).
