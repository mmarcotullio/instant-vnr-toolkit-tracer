if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_minimum_required(VERSION 3.24)  # 3.24+ required for OVERRIDE_FIND_PACKAGE in dep_openvkl
project(example LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ovr/cmake")
include(configure_build_type)
include(configure_cxx)
include(configure_cuda)
include(configure_optix)
include(configure_python)

###############################################################################
# Python setup
###############################################################################
include(configure_python)
if(PYTHON_GLIBCXX_USE_CXX11_ABI STREQUAL "True")
  set(IVNR_GLIBCXX_CXX11_ABI _GLIBCXX_USE_CXX11_ABI=1)
elseif(PYTHON_GLIBCXX_USE_CXX11_ABI STREQUAL "False")
  set(IVNR_GLIBCXX_CXX11_ABI _GLIBCXX_USE_CXX11_ABI=0)
else()
  message(FATAL_ERROR "PYTHON_GLIBCXX_USE_CXX11_ABI is not True or False")
endif()
message(STATUS "GLIBCXX_CXX11_ABI: ${PYTHON_GLIBCXX_USE_CXX11_ABI}")

###############################################################################
# Staging directory — all shared libs and executables land here so the install
# step can copy them as a single package directory (same pattern as pysampler).
###############################################################################
set(_ivnr_staging "${CMAKE_BINARY_DIR}/instantvnr")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_ivnr_staging}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_ivnr_staging}")
set(CMAKE_INSTALL_RPATH              "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH   ON)
set(CMAKE_POSITION_INDEPENDENT_CODE  ON)

###############################################################################
# OptiX and OSPRay not used; their device subdirs may not be checked out.
###############################################################################
set(OVR_BUILD_OPENGL ON CACHE BOOL "" FORCE)
set(OVR_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(OVR_BUILD_DEVICE_OPTIX7 OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_DEVICE_OSPRAY OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory(ovr)

# # When building with gcc-13 on a system where GCC 15 is the default, system
# # libraries (e.g. libtbb) may require CXXABI_1.3.15 from GCC 15's libstdc++.
# # Add GCC 15's lib dir so the linker finds the right libstdc++ version.
# if(CMAKE_CXX_cmpILER_ID STREQUAL "GNU")
#   execute_process(
#     COMMAND ${CMAKE_CXX_cmpILER} -print-file-name=libstdc++.so
#     OUTPUT_VARIABLE _GCC_LIBSTDCXX_PATH
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#   )
#   get_filename_cmponent(_GCC_LIBDIR "${_GCC_LIBSTDCXX_PATH}" DIRECTORY)
#   # Check if system has a newer libstdc++ (CXXABI_1.3.15 required by system TBB)
#   if(EXISTS "/usr/lib/gcc/x86_64-linux-gnu/15/libstdc++.so")
#     set(CMAKE_EXE_LINKER_FLAGS
#       "-L/usr/lib/gcc/x86_64-linux-gnu/15 ${CMAKE_EXE_LINKER_FLAGS}")
#     set(CMAKE_SHARED_LINKER_FLAGS
#       "-L/usr/lib/gcc/x86_64-linux-gnu/15 ${CMAKE_SHARED_LINKER_FLAGS}")
#   endif()
# endif()

# ###############################################################################
# # CUDA compiler setup
# ###############################################################################
# include(CheckLanguage)

# ## CUDA ## 
# check_language(CUDA)
# if(NOT CMAKE_CUDA_cmpILER)
# 	message(STATUS "CUDA not found")
# 	set(ENABLE_CUDA OFF)
# else()
# 	message(STATUS "CUDA found: ${CMAKE_CUDA_cmpILER}")
# 	set(ENABLE_CUDA ON)
# 	# CUDA compiler setup
# 	enable_language(CUDA)
# 	set(CMAKE_CUDA_STANDARD 17)
# 	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 	set(CMAKE_CUDA_EXTENSIONS OFF)
# 	set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
# 	if (MSVC)
# 		list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
# 	else()
# 		list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
# 		list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
# 		list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
# 		list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
# 	endif()
# 	list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
# 	list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
# 	list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__")
# 	list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_CONVERSIONS__")
# 	list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF2_OPERATORS__")
# endif()

add_subdirectory(ext)
# pysampler lives in packages/sampler/ and is a separate package; build it independently

###############################################################################
# Install — put everything into the instantvnr Python package.
# The staging directory (_ivnr_staging) already collects every shared library
# and executable via CMAKE_LIBRARY/RUNTIME_OUTPUT_DIRECTORY, so a single
# install(DIRECTORY) picks up all transitive deps (glad, imgui, etc.) without
# having to enumerate each FetchContent target.
###############################################################################
set(_pkg "instantvnr")
set(_cmp "instantvnr")

# Python source files
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/python/apps.py"
  DESTINATION ${_pkg}
  COMPONENT ${_cmp}
)

# Everything CMake built (shared libs, pybind11 modules, executables)
install(DIRECTORY "${_ivnr_staging}/"
  DESTINATION ${_pkg}
  COMPONENT ${_cmp}
  USE_SOURCE_PERMISSIONS
  PATTERN "__pycache__" EXCLUDE
)
