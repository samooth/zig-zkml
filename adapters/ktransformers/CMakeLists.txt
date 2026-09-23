# CMakeLists.txt — ktransformers-zig reference glue (standalone).
#
# Compiles the glue against the engine's OWN public header and links its
# prebuilt shared library, so row-size questions are answered by
# `kt_type_row_bytes` instead of a duplicated block-size table.
#
#   cmake -S adapters/ktransformers -B .zig-cache/kt_adapter \
#         -DKT_DIR=/path/to/ktransformers-zig \
#         -DZKML_LIB_DIR=/path/to/zig-zkml/zig-out/lib \
#         -DKT_VARIANT=avx2
#   cmake --build .zig-cache/kt_adapter
#
# Produces: libkt_zkml_glue.so and kt_glue_test (gated positive/negative).
cmake_minimum_required(VERSION 3.16)
project(kt_zkml_glue C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED KT_DIR)
    get_filename_component(_repo_parent "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    get_filename_component(KT_DIR "${_repo_parent}/ktransformers-zig" ABSOLUTE)
endif()
if(NOT EXISTS "${KT_DIR}/include/kt_kernel.h")
    message(FATAL_ERROR "KT_DIR=${KT_DIR} does not look like a ktransformers-zig checkout (missing include/kt_kernel.h)")
endif()

# Which prebuilt engine variant to link (must match the running CPU).
if(NOT DEFINED KT_VARIANT)
    set(KT_VARIANT "avx2")
endif()
set(KT_LIB "${KT_DIR}/zig-out/lib/libkt_kernel_ext_${KT_VARIANT}.so")
if(NOT EXISTS "${KT_LIB}")
    message(FATAL_ERROR
        "ktransformers-zig not built: expected ${KT_LIB}\n"
        "Build it in ${KT_DIR} (zig build) or pass -DKT_VARIANT=<amx|avx512_*|avx2|neon>")
endif()

get_filename_component(_here "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(_zkml_root "${_here}/../.." ABSOLUTE)
if(NOT DEFINED ZKML_LIB_DIR)
    set(ZKML_LIB_DIR "${_zkml_root}/zig-out/lib")
endif()
if(NOT EXISTS "${ZKML_LIB_DIR}/libzkml.so")
    message(FATAL_ERROR "libzkml.so not found in ${ZKML_LIB_DIR} — run `zig build` in ${_zkml_root} first")
endif()

add_library(kt_zkml_glue SHARED kt_glue.c)
target_include_directories(kt_zkml_glue PUBLIC
    "${_here}"
    "${_zkml_root}/include"
    "${KT_DIR}/include"
)
target_link_directories(kt_zkml_glue PUBLIC "${ZKML_LIB_DIR}")
target_link_libraries(kt_zkml_glue PUBLIC
    "${KT_LIB}"
    "${ZKML_LIB_DIR}/libzkml.so"
)
set_target_properties(kt_zkml_glue PROPERTIES
    BUILD_RPATH "${ZKML_LIB_DIR};${KT_DIR}/zig-out/lib"
    INSTALL_RPATH "${ZKML_LIB_DIR};${KT_DIR}/zig-out/lib"
)

add_executable(kt_glue_test test_glue.c)
target_link_libraries(kt_glue_test PRIVATE kt_zkml_glue)
set_target_properties(kt_glue_test PROPERTIES
    BUILD_RPATH "${ZKML_LIB_DIR};${KT_DIR}/zig-out/lib"
)

enable_testing()
add_test(NAME kt_zkml_glue COMMAND kt_glue_test ${CMAKE_CURRENT_BINARY_DIR}/artifacts)
