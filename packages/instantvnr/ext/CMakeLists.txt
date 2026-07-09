# ======================================================================== #
# Copyright 2019-2026 Qi Wu                                                #
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

###############################################################################
# Setup Repositories
###############################################################################
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# # message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
# foreach(pfpath ${CMAKE_PREFIX_PATH})
#   list(APPEND CMAKE_MODULE_PATH ${pfpath}/lib/cmake/OpenVDB)
# endforeach()

find_package(TBB CONFIG REQUIRED)

if (DEFINED IVNR_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES ${IVNR_CUDA_ARCHITECTURES})
	set(TCNN_CUDA_ARCHITECTURES  ${IVNR_CUDA_ARCHITECTURES})
	set(TORCH_CUDA_ARCH_LIST )
	foreach (arch IN LISTS IVNR_CUDA_ARCHITECTURES)
		string(REGEX REPLACE "^(.+)(.)$" "\\1.\\2" modified_arch ${arch})
		list(APPEND TORCH_CUDA_ARCH_LIST ${modified_arch})
	endforeach()
endif()

###############################################################################
# Setup Python and PyTorch
###############################################################################

# Ensure Python Development.Module (needed by pybind11) is found before
# find_package(Torch) runs — Torch may find Python without Development.Module,
# causing pybind11's python_add_library to be undefined later.
# Development.Embed is also needed so that pybind11::embed links libpython.
if(DEFINED Python_EXECUTABLE)
  find_package(Python 3.6 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
endif()

# Find where `pytorch` is by executing `python -c "import torch, os; print(os.path.dirname(torch.__file__))"`
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c 
    "import torch, os; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE PYTHON_TORCH_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if("${PYTHON_TORCH_SITE}" STREQUAL "")
  message(FATAL_ERROR "PyTorch not found")  # TODO make it optional
else()
  message(STATUS "PYTHON_TORCH_SITE=${PYTHON_TORCH_SITE}")
endif()

# Use the pytorch installation path as the libtorch installation path (they are the same)
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_TORCH_SITE}")

# Restrict Torch's gencode to our target arch — Torch's cmake otherwise adds
# many older architectures (sm_50, etc.) that fail TCNN's MIN_GPU_ARCH assertion.
if(CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  set(_ivnr_torch_arch_list "")
  foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
    string(REGEX REPLACE "([0-9]+)([0-9])$" "\\1.\\2" _arch_dot "${_arch}")
    list(APPEND _ivnr_torch_arch_list "${_arch_dot}")
  endforeach()
  set(TORCH_CUDA_ARCH_LIST "${_ivnr_torch_arch_list}" CACHE STRING "" FORCE)
  message(STATUS "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
endif()

find_package(Torch QUIET)

# - https://github.com/pytorch/pytorch/issues/108041
# - Causing the following error if not link to `torch_python` explicitly
#   undefined reference to symbol '_ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb'
#                        -------> 'pybind11::detail::type_caster<at::Tensor, void>::load(pybind11::handle, bool)'
if(Torch_FOUND)
  set(ENABLE_TORCH ON)
  # link to libtorch_python.so manually
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${PYTHON_TORCH_SITE}/lib)
  if(TORCH_PYTHON_LIBRARY)
    list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON_LIBRARY})
  else()
    message(FATAL_ERROR "TORCH_PYTHON_LIBRARY not found")
  endif()
else()
  set(ENABLE_TORCH OFF)
  message(FATAL_ERROR "LibTorch not found !!!")
endif()

# # Torch's cmake appends multi-arch gencode flags to both CUDA_NVCC_FLAGS and
# # CMAKE_CUDA_FLAGS when GPU detection falls back to a broad default list
# # (e.g. sm_50 through sm_90a).  Strip them from both so CMAKE_CUDA_ARCHITECTURES
# # alone controls arch selection.  This also prevents TCNN's static_assert
# # (TCNN_MIN_GPU_ARCH * 10 <= __CUDA_ARCH__) from failing when nvcc compiles
# # TCNN headers for the fallback sm_50 arch.
# # Two formats seen in the wild:
# #   PyTorch <=1.x: -gencode arch=compute_XX,code=sm_XX
# #   PyTorch  >=2.x: --generate-code=arch=compute_XX,code=[compute_XX,sm_XX]
# list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX ".*-gencode.*")
# list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX ".*arch=compute.*")
# list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX ".*--generate-code.*")
# string(REGEX REPLACE " *-gencode +arch=compute_[0-9a-z]+,code=sm_[0-9a-z]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(REGEX REPLACE " *--generate-code=[^ ]*" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
# # message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

################
# tiny-cuda-nn #
################
find_package(TCNN REQUIRED)

###############################################################################
# Modules
###############################################################################

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(ENABLE_OUT_OF_CORE OFF CACHE BOOL "Enable Out-of-Core" FORCE)
set(ENABLE_OPENGL ${OVR_BUILD_OPENGL} CACHE BOOL "Enable OpenGL apps" FORCE)

add_subdirectory(core)
add_subdirectory(device)
add_subdirectory(apps)

# # create symbolic link
# message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E create_symlink 
#       ${CMAKE_CURRENT_LIST_DIR}/infer_coordnet.py ${CMAKE_BINARY_DIR}/infer_coordnet.py)
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E create_symlink 
#       ${CMAKE_CURRENT_LIST_DIR}/infer_hyperinr.py ${CMAKE_BINARY_DIR}/infer_hyperinr.py)
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E create_symlink 
#       ${CMAKE_CURRENT_LIST_DIR}/infer_triplaneinr.py ${CMAKE_BINARY_DIR}/infer_triplaneinr.py)
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E create_symlink 
#       ${CMAKE_CURRENT_LIST_DIR}/infer_siren.py ${CMAKE_BINARY_DIR}/infer_siren.py)
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E create_symlink 
#       ${CMAKE_CURRENT_LIST_DIR}/shadows_INR_config.json ${CMAKE_BINARY_DIR}/shadows_INR_config.json)
