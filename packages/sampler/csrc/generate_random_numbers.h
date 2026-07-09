#pragma once

#include <cstddef>

#define ASSERT_THROW(cond) { if (!(cond)) throw std::runtime_error(#cond); }

namespace pysampler {

enum struct Device {
    CPU  = 4000,
    CUDA = 4100,
    MPS  = 4200,
    XPU  = 4300,
    HIP  = 4400
};

void generate_random_uniforms_serial(size_t batch, float* buffer, Device device);
void generate_random_uniforms_torch (size_t batch, float* buffer, Device device);
void generate_random_uniforms_cuda  (size_t batch, float* buffer, Device device);
void generate_random_uniforms(float* buffer, size_t batch, Device device);

}
