#pragma once

#ifndef PYSAMPLER_SAMPLER_CPP_IMPL
#error "This header is not supposed to be included directly"
#endif

#include "config.h"
#include "sampler.h"

namespace pysampler {

struct CUDAImplStructured;
struct CUDAImplExa;

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
struct CUDASampler : SamplerBase {
protected:
  const dtype m_type{};
  const int m_n_channels{};
  box3f m_bounds;
  std::vector<range1f> m_ranges;

  std::vector<std::shared_ptr<CUDAImplStructured>> m_fields_structured;
  std::shared_ptr<CUDAImplExa>                     m_fields_exa;

  enum {
    INVALID = 0,
    STRUCTURED,
    EXA,
  } m_sampling_mode = INVALID;

  void process_structured(const VolumeDataStructured& desc);
  friend void sample_structured(const CUDASampler* self, void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count, void* stream);

#ifdef ENABLE_WITCHER
  void process_exa(const VolumeFileExaBrick& desc);
  void process_exa(const VolumeFileExaStitch& desc);
  friend void sample_exa(const CUDASampler* self, void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count, void* stream);
#endif

public:
  CUDASampler(VolumeFileStructured desc);
  CUDASampler(VolumeDataStructured desc);
#ifdef ENABLE_WITCHER
  CUDASampler(VolumeFileExaBrick desc);
  CUDASampler(VolumeFileExaStitch desc);
#endif

  ~CUDASampler() override;
  pysampler::Device device() const override { return pysampler::Device::CUDA; }
  pysampler::box3f bounds() const override { return m_bounds; }
  int n_channels() const override { return m_n_channels; }
  void sample(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) override;
  void decode1d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) override;
  void decode3d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, const vec3i& dims) override;
};

}
