#define PYSAMPLER_SAMPLER_CPP_IMPL

#include "sampler_cuda.h"
#include "generate_random_numbers.h"

#include <cuda/cuda_buffer.h>

#include <cstdlib>
#include <cstring>

namespace pysampler {

CUDASampler::~CUDASampler() {}

CUDASampler::CUDASampler(VolumeFileStructured desc) 
  : SamplerBase(), m_type(desc.type), m_n_channels(desc.n_channels), m_ranges(desc.ranges)
{
  auto handler = load_regular_grid(desc.filename, desc.dims, desc.n_channels, desc.type, desc.offset, desc.is_big_endian);

  VolumeDataStructured desc2;
  desc2.dims = desc.dims;
  desc2.spacing = desc.spacing;
  desc2.type = desc.type;
  desc2.n_channels = desc.n_channels;
  desc2.data = (uint8_t*)handler.get();
  process_structured(desc2);
}

CUDASampler::CUDASampler(VolumeDataStructured desc)
  : SamplerBase(), m_type(desc.type), m_n_channels(desc.n_channels), m_ranges(desc.ranges)
{
  process_structured(desc);
}

#ifdef ENABLE_WITCHER
CUDASampler::CUDASampler(VolumeFileExaBrick desc)
  : SamplerBase(), m_type(pysampler::FLOAT), m_n_channels(1)
{
  process_exa(desc);
}

CUDASampler::CUDASampler(VolumeFileExaStitch desc)
  : SamplerBase(), m_type(pysampler::FLOAT), m_n_channels(1)
{
  process_exa(desc);
}
#endif

void
CUDASampler::sample(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) {
  generate_random_uniforms((float*)d_coords, count * 3, pysampler::Device::CUDA);
  decode1d(d_coords, d_values, lower, upper, count);
}

void 
CUDASampler::decode1d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) {
  cudaStream_t stream = nullptr;
  switch (m_sampling_mode) {
    case STRUCTURED: sample_structured(this, d_coords, d_values, lower, upper, count, stream); break;
#ifdef ENABLE_WITCHER
    case EXA:        sample_exa       (this, d_coords, d_values, lower, upper, count, stream); break;
#endif
    default: throw std::runtime_error("not implemented");
  }
}

void 
CUDASampler::decode3d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, const vec3i& dims) {
  throw std::runtime_error("not implemented");
}

}
