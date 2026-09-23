# CMakeLists.txt — llama.cpp attestation adapter (standalone, zero-fork).
#
# Pattern after examples/gguf-hash/CMakeLists.txt, but built OUTSIDE the
# llama.cpp tree: point -DLLAMA_DIR at a llama.cpp checkout that has been
# configured+built (build/bin holds libggml-base.so / libllama.so), and
# -DZKML_LIB_DIR at zig-out/lib (libzkml.so from `zig build`).
#
#   cmake -S adapters/llama_cpp -B .zig-cache/llama_adapter \
#         -DLLAMA_DIR=/path/to/llama.cpp \
#         -DZKML_LIB_DIR=/path/to/zig-zkml/zig-out/lib
#   cmake --build .zig-cache/llama_adapter
#
# Produces: libzkml_llama.so (shared wrapper) and zkml_llama_test (gated
# positive/negative test that emits root.hex + manifest.json).
cmake_minimum_required(VERSION 3.16)
project(zkml_llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED LLAMA_DIR)
    # Default: sibling checkout of zig-zkml.
    get_filename_component(_repo_parent "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    get_filename_component(LLAMA_DIR "${_repo_parent}/llama.cpp" ABSOLUTE)
endif()
if(NOT EXISTS "${LLAMA_DIR}/ggml/include/gguf.h")
    message(FATAL_ERROR "LLAMA_DIR=${LLAMA_DIR} does not look like a llama.cpp checkout (missing ggml/include/gguf.h)")
endif()

set(LLAMA_BUILD_BIN "${LLAMA_DIR}/build/bin")
if(NOT EXISTS "${LLAMA_BUILD_BIN}/libggml-base.so")
    message(FATAL_ERROR "llama.cpp not built: expected ${LLAMA_BUILD_BIN}/libggml-base.so (run cmake+make in ${LLAMA_DIR}/build first)")
endif()

get_filename_component(_here "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(_zkml_root "${_here}/../.." ABSOLUTE)
if(NOT DEFINED ZKML_LIB_DIR)
    set(ZKML_LIB_DIR "${_zkml_root}/zig-out/lib")
endif()
if(NOT EXISTS "${ZKML_LIB_DIR}/libzkml.so")
    message(FATAL_ERROR "libzkml.so not found in ${ZKML_LIB_DIR} — run `zig build` in ${_zkml_root} first")
endif()

add_library(zkml_llama SHARED wrapper.cpp)
target_include_directories(zkml_llama PUBLIC
    "${_here}"
    "${_zkml_root}/include"
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
)
# Link: ggml/gguf reader (container walk) + zig-zkml core (attestor).
# libzkml.so re-exports the 9 zkml_* v1 + witness v2 entry points.
target_link_directories(zkml_llama PRIVATE "${LLAMA_BUILD_BIN}" "${ZKML_LIB_DIR}")
target_link_libraries(zkml_llama PRIVATE ggml-base zkml)
# RPATH so the test binary and consumers find the .so files without LD_LIBRARY_PATH.
set_target_properties(zkml_llama PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_BIN};${ZKML_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_BUILD_BIN};${ZKML_LIB_DIR}"
)

add_executable(zkml_llama_test test_attest.cpp)
# The test builds its own GGUF, so it needs gguf/ggml symbols directly in
# addition to the wrapper (shared-lib transitive linkage is not enough).
target_link_directories(zkml_llama_test PRIVATE "${LLAMA_BUILD_BIN}" "${ZKML_LIB_DIR}")
target_link_libraries(zkml_llama_test PRIVATE zkml_llama ggml-base)
set_target_properties(zkml_llama_test PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_BIN};${ZKML_LIB_DIR}"
)

enable_testing()
add_test(NAME zkml_llama_attest COMMAND zkml_llama_test ${CMAKE_CURRENT_BINARY_DIR}/artifacts)
