#include "renderer.h"

// PyTorch-based inference backend for the neural volume renderer.
// These functions are called from renderer.cu when using the PyTorch
// inference path (non-TCNN). Implemented via pybind11::embed to call
// Python/PyTorch from C++.

namespace ovr::hyperinr {

void pytorch_set_param(float /* time */) {}

void pytorch_set_param(float /* theta */, float /* phi */) {}

void pytorch_inference_kernel(cudaStream_t /* stream */,
                               uint32_t /* count */,
                               vec3f* __restrict__ /* d_coords */,
                               float* __restrict__ /* d_values */) {}

} // namespace ovr::hyperinr
