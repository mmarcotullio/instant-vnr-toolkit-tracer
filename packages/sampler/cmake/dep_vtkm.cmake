cmake_minimum_required(VERSION 3.24)
include(FetchContent)

option(ENABLE_VTKM "Enable VTKm structured-mesh I/O backend" OFF)
if(NOT ENABLE_VTKM)
  return()
endif()

# ── VTKm ──────────────────────────────────────────────────────────────────────
set(VTKm_NO_DEPRECATED_VIRTUAL        ON  CACHE BOOL "" FORCE)
set(VTKm_USE_64BIT_IDS                OFF CACHE BOOL "" FORCE)
set(VTKm_USE_DOUBLE_PRECISION         ON  CACHE BOOL "" FORCE)
set(VTKm_USE_DEFAULT_TYPES_FOR_ASCENT ON  CACHE BOOL "" FORCE)
# Skip everything we don't need — only vtkm_cont + vtkm_io are linked.
# vtkm_cont is in the Core group (always built).
# vtkm_io is in the IO group and must be explicitly requested when BUILD_ALL_LIBRARIES=OFF.
set(VTKm_BUILD_ALL_LIBRARIES          OFF  CACHE BOOL   "" FORCE)
set(VTKm_GROUP_ENABLE_IO              YES  CACHE STRING "" FORCE)
set(VTKm_ENABLE_RENDERING             OFF CACHE BOOL "" FORCE)
set(VTKm_ENABLE_TESTING               OFF CACHE BOOL "" FORCE)
set(VTKm_ENABLE_BENCHMARKS            OFF CACHE BOOL "" FORCE)
set(VTKm_ENABLE_LOGGING               OFF CACHE BOOL "" FORCE)
set(VTKm_ENABLE_CPACK                 OFF CACHE BOOL "" FORCE)
set(VTKm_ENABLE_DEVELOPER_FLAGS       OFF CACHE BOOL "" FORCE)

if(CMAKE_CUDA_COMPILER)
  set(VTKm_ENABLE_CUDA ON  CACHE BOOL "" FORCE)
else()
  set(VTKm_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(vtkm
  URL "https://github.com/Kitware/VTK-m/archive/refs/tags/v2.3.0.zip"
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(vtkm)
