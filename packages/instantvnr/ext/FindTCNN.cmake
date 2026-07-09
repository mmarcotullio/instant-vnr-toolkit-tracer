# -------------------------------------------------------------------
# copy from TCNN to setup compile definitions correctly
# -------------------------------------------------------------------

if (NOT DEFINED CUDA_VERSION)
  find_package(CUDAToolkit REQUIRED)
  set(CUDA_VERSION ${CUDAToolkit_VERSION})
  message(STATUS "CUDA version: ${CUDA_VERSION}")
endif()

if (NOT DEFINED CUDA_VERSION)
  message(FATAL_ERROR "CUDA_VERSION not defined")
endif()

# adapted from https://stackoverflow.com/a/69353718
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained target architecture from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
elseif (TCNN_CUDA_ARCHITECTURES)
	message(STATUS "Obtained target architecture from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
else()
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

# Remove unsupported architectures
list(FILTER CMAKE_CUDA_ARCHITECTURES EXCLUDE REGEX "PTX")
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

# If the CUDA version does not support the chosen architecture, target
# the latest supported one instead.
set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 120)

if (CUDA_VERSION VERSION_LESS 11.8)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 86)
endif()

if (CUDA_VERSION VERSION_LESS 11.1)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 80)
endif()

if (CUDA_VERSION VERSION_LESS 11.0)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 75)
endif()

foreach (CUDA_CC IN LISTS CMAKE_CUDA_ARCHITECTURES)
	if (CUDA_CC GREATER ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(WARNING "CUDA version ${CUDA_VERSION} is too low for architecture ${CUDA_CC}. Targeting architecture ${LATEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
	endif()
endforeach(CUDA_CC)

if (NOT CMAKE_CUDA_ARCHITECTURES)
	list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (TCNN_HAS_PARENT)
	set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
	set(TCNN_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures" FORCE)

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	# Only compile the shampoo optimizer if
	# a new enough cuBLAS version is available.
	list(APPEND TCNN_DEFINITIONS -DTCNN_SHAMPOO)
endif()

# -------------------------------------------------------------------
# TCNN
# -------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
# find_package(Python COMPONENTS Development)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c 
    "import torch, os; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE PYTHON_TORCH_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import pkgutil, os; print(os.path.dirname(pkgutil.get_loader('tinycudann').get_filename()))"
  OUTPUT_VARIABLE PYTHON_TCNN_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Tiny-Cuda-NN SITE: ${PYTHON_TCNN_SITE}")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python extension suffix: ${PYTHON_EXT_SUFFIX}")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
  OUTPUT_VARIABLE PYTHON_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_file(TCNN_tcnn_LIBRARY
  NAMES
  "_C${PYTHON_EXT_SUFFIX}"
  "_${MIN_GPU_ARCH}_C${PYTHON_EXT_SUFFIX}"
  PATHS
    ${PYTHON_TCNN_SITE}/../tinycudann_bindings
    ${PYTHON_TCNN_SITE}/../tinycudann_bindings_${MIN_GPU_ARCH}
  NO_DEFAULT_PATH NO_CMAKE_PATH
  NO_CMAKE_ENVIRONMENT_PATH
)

# Check if CXX11 ABI should be enabled. Must match Python.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c 
    "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE PYTHON_GLIBCXX_USE_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# message(STATUS ${PYTHON_SITE})
message(STATUS "PyTorch PATH: ${PYTHON_TORCH_SITE}")
message(STATUS "Tiny-Cuda-NN PATH: ${PYTHON_TCNN_SITE}")
message(STATUS "Tiny-Cuda-NN LIB:  ${TCNN_tcnn_LIBRARY}")
message(STATUS "Tiny-Cuda-NN ARCH: ${MIN_GPU_ARCH}")

# The tcnn _C.so has no embedded SONAME and shares its filename with torch/_C.so
# and other packages.  To avoid both the "full path baked into DT_NEEDED" problem
# (ephemeral uv build cache) and the "wrong _C.so found" ambiguity, we copy the
# library into CMAKE_LIBRARY_OUTPUT_DIRECTORY under a unique name and link
# against that.  It ships alongside libivnr.so and is found via $ORIGIN RPATH.
get_filename_component(TCNN_tcnn_LIBRARY_NAME "${TCNN_tcnn_LIBRARY}" NAME)
set(TCNN_UNIQUE_SONAME "libtinycudann_${MIN_GPU_ARCH}${PYTHON_EXT_SUFFIX}")
file(COPY "${TCNN_tcnn_LIBRARY}" DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
file(RENAME
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${TCNN_tcnn_LIBRARY_NAME}"
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${TCNN_UNIQUE_SONAME}"
)

add_library(tcnn INTERFACE)
target_link_libraries(tcnn INTERFACE
  "-l:${TCNN_UNIQUE_SONAME}"
  cuda cudadevrt cudart_static
  pybind11::embed
)
target_link_directories(tcnn INTERFACE
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  ${CUDAToolkit_LIBRARY_DIR}
  ${PYTHON_TORCH_SITE}/lib
)
target_include_directories(tcnn INTERFACE
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(tcnn INTERFACE ${TCNN_DEFINITIONS})

if(PYTHON_GLIBCXX_USE_CXX11_ABI STREQUAL "True")
  target_compile_definitions(tcnn INTERFACE IVNR_GLIBCXX_CXX11_ABI=1)
else()
  target_compile_definitions(tcnn INTERFACE IVNR_GLIBCXX_CXX11_ABI=0)
endif()

# -------------------------------------------------------------------
# TCNN finalize
# -------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TCNN
  REQUIRED_VARS TCNN_tcnn_LIBRARY MIN_GPU_ARCH
)

# -------------------------------------------------------------------
# TCNN -- Create a target for the local tcnn repository
# -------------------------------------------------------------------

if(NOT DEFINED TCNN_SOURCE_DIR)

  # Resolve repository URL: env > cmake > default
  if(DEFINED ENV{TCNN_REPOSITORY})
    set(_tcnn_repo "$ENV{TCNN_REPOSITORY}")
  elseif(TCNN_REPOSITORY)
    set(_tcnn_repo "${TCNN_REPOSITORY}")
  else()
    set(_tcnn_repo "https://github.com/wilsonCernWq/tiny-cuda-nn.git")
  endif()

  # Resolve commit ref: env > cmake > default branch
  if(DEFINED ENV{TCNN_COMMIT_HASH})
    set(_tcnn_ref GIT_TAG "$ENV{TCNN_COMMIT_HASH}")
    message(STATUS "TCNN: ref=${_tcnn_ref}")
  elseif(TCNN_COMMIT_HASH)
    set(_tcnn_ref GIT_TAG "${TCNN_COMMIT_HASH}")
  else()
    set(_tcnn_ref "")
  endif()

  message(STATUS "TCNN: repo=${_tcnn_repo} ${_tcnn_ref}")
  FetchContent_Declare(tcnn_content 
    GIT_REPOSITORY "${_tcnn_repo}" 
    ${_tcnn_ref}
  )
  FetchContent_GetProperties(tcnn_content)
  if (NOT tcnn_content_POPULATED)
    FetchContent_Populate(tcnn_content)
  endif()
  set(TCNN_SOURCE_DIR ${tcnn_content_SOURCE_DIR})
  message(STATUS "TCNN: source=${TCNN_SOURCE_DIR}")

endif()

if((NOT TARGET tiny-cuda-nn) AND (NOT TARGET tcnn))
  message(FATAL_ERROR "Please install tiny-cuda-nn!")
endif()
# Header-only target: include paths, compile definitions, and rpath-link hints.
# Use this (PUBLIC) when a library's public API exposes tcnn types so that
# downstream consumers can compile without pulling in libtinycudann at link time.
# The rpath-link hints propagate through the PUBLIC chain so the linker can
# resolve transitive deps (torch, python, c10) of libtinycudann when linking
# standalone executables.  rpath-link is NOT embedded in the binary.
add_library(tcnn_headers INTERFACE)
target_compile_definitions(tcnn_headers INTERFACE TCNN_NAMESPACE=tcnn ${TCNN_DEFINITIONS})
target_include_directories(tcnn_headers INTERFACE
  "${TCNN_SOURCE_DIR}/include"
  "${TCNN_SOURCE_DIR}/dependencies"
  "${TCNN_SOURCE_DIR}/dependencies/cutlass/include"
  "${TCNN_SOURCE_DIR}/dependencies/cutlass/tools/util/include"
  "${TCNN_SOURCE_DIR}/dependencies/fmt/include"
)
target_link_options(tcnn_headers INTERFACE
  "LINKER:-rpath-link,${PYTHON_TORCH_SITE}/lib"
  "LINKER:-rpath-link,${PYTHON_LIBDIR}"
)

# Full target inherits headers and adds the binary link.
target_link_libraries(tcnn INTERFACE tcnn_headers)
