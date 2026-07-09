# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

set(target ivnr)

include(configure_cuda) # skip optix here

# ======================================================================== #
# renderer core
# ======================================================================== #

add_library(${target} SHARED
  samplers/neural_sampler.cu
  samplers/neural_sampler.cpp
  samplers/neural_sampler.h
  samplers/sampler.cu
  samplers/sampler.h

  networks/siren_impl.cu
  networks/siren_impl.cuh
  networks/siren_network.h
  networks/tcnn_impl.cu
  networks/tcnn_network.h
  networks/nnapi.h

  volumes/macrocell.cu
  volumes/macrocell.h
  volumes/neural_volume.cu
  volumes/simple_volume.cu
  volumes/volumes.h

  instantvnr_types.cu
  instantvnr_types.h  
  array.h
  ../app.cu
)

set_target_properties(${target}
PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
# # Hide tcnn's s_current_captures from libivnr.so's dynamic symbol table.
# # NVCC emits thread_local statics as plain BSS, conflicting with the TLS
# # definition in the pre-installed tinycudann .so at link time.
# target_link_options(${target} PRIVATE
#   "LINKER:--version-script=${CMAKE_CURRENT_LIST_DIR}/ivnr_exports.map"
# )

target_include_directories(${target} 
PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}
PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}/..
  ${CMAKE_CURRENT_LIST_DIR}/../..
)

target_compile_options(${target}
PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177">   # -> variable declared but not referenced
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=20044"> # -> extern variable '...' treated as static
)

target_link_libraries(${target}
PRIVATE
  TBB::tbb tcnn # binary .so: PRIVATE — never propagate tinycudann_bindings
  $<$<BOOL:${ENABLE_OUT_OF_CORE}>:$<$<PLATFORM_ID:Linux>:aio>>
PUBLIC
  rendercommon tcnn_headers # headers only: safe for consumers that use tcnn types
)
target_compile_definitions(${target} PUBLIC ${IVNR_GLIBCXX_CXX11_ABI})
target_compile_definitions(${target} 
PRIVATE
  $<$<BOOL:${ENABLE_OUT_OF_CORE}>:ENABLE_OUT_OF_CORE=1>
PUBLIC
  $<$<BOOL:${ENABLE_LOGGING}>:ENABLE_LOGGING=1>
  $<IF:$<BOOL:${MACROCELL_SIZE_MIP}>,MACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP},MACROCELL_SIZE_MIP=4>
  $<IF:$<BOOL:${DISABLE_ADAPTIVE_SAMPLING}>,ADAPTIVE_SAMPLING=0,ADAPTIVE_SAMPLING=1>
)

# Prevent pybind11 from adding pybind11::lto to vnr_network.
# nvcc uses its internal GCC 13 for device linking (cmake_device_link.o),
# but the system linker is GCC 15 — mixing LTO 13.1 and 15.1 bytecode
# causes "bytecode stream generated with LTO version 13.1 instead of 15.1".
# pybind11 skips pybind11::lto when CMAKE_INTERPROCEDURAL_OPTIMIZATION is defined.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

option(VNR_BUILD_NETWORK_PYBIND "Build vnr_network Python extension (pybind11)" ON)
if(VNR_BUILD_NETWORK_PYBIND)
  pybind11_add_module(vnr_network
    networks/network_pybind.cu
  )
  target_compile_definitions(vnr_network PRIVATE TORCH_EXTENSION_NAME=vnr_network)
  target_link_libraries(vnr_network PRIVATE ivnr)
  target_include_directories(vnr_network PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_compile_options(vnr_network PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177">
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=20044">
  )
  # # network_pybind.cu includes nnapi.h → gpu_memory.h → cuda_graph.h.
  # # NVCC emits current_captures()'s static as plain .bss (non-TLS), which
  # # conflicts at runtime with the TLS version in tinycudann_bindings.
  # # Hide it here the same way we do for libivnr.so.
  # target_link_options(vnr_network PRIVATE
  #   "LINKER:--version-script=${CMAKE_CURRENT_LIST_DIR}/ivnr_exports.map"
  # )
  set_target_properties(vnr_network PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION FALSE
  )
endif()
