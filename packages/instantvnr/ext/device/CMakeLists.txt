message(STATUS "Enable Neural Data Cache Device")

# We compile this file separately
add_library(ivnr_pytorch OBJECT renderer_pytorch.cpp)

add_library(device_hyperinr SHARED
  method_pathtracing.cu
  method_raymarching.cu
  method_shadowmap.cu
  renderer.cu

  array.cpp
  device.cu
  space_partition.cu
  volume.cu

  ../serializer.cpp
  ../serializer.h
  ../api.cpp
  ../api.h
)

set_target_properties(device_hyperinr 
PROPERTIES 
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries (device_hyperinr 
PRIVATE 
  ivnr 
  ivnr_pytorch
  rendercommon 
  TBB::tbb
)
# tfnmodule is linked PRIVATE into rendercommon, so its PUBLIC include
# directories (which contain colormap.h) do not propagate to consumers.
# Pull them in explicitly so ext/serializer.cpp can find <colormap.h>.
# Symbol resolution still goes through librendercommon.so; no re-link needed.
target_include_directories(device_hyperinr PRIVATE
  $<TARGET_PROPERTY:tfnmodule,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_options(device_hyperinr 
PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)

# --- Link GcCore libraries ---
target_link_libraries(device_hyperinr 
PRIVATE
  $<$<BOOL:${ENABLE_OPENGL}>:imgui>
  Threads::Threads
)
target_compile_definitions(device_hyperinr 
PRIVATE 
  ${IVNR_GLIBCXX_CXX11_ABI}
  $<$<BOOL:${ENABLE_OPENGL}>:ENABLE_OPENGL=1>
)

# ------------------------------------------------------------------
# LibTorch & PyTorch
# ------------------------------------------------------------------

if(NOT DEFINED TORCH_LIBRARIES)
  message(FATAL_ERROR "TORCH_LIBRARIES not defined")
endif()
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

# ------------------------------------------------------------------
# --- add pytorch support ---
# ------------------------------------------------------------------
target_link_libraries(ivnr_pytorch 
PRIVATE 
  ivnr 
  pybind11::embed
  ${TORCH_LIBRARIES}
)
set_target_properties(ivnr_pytorch PROPERTIES 
  POSITION_INDEPENDENT_CODE ON
)
