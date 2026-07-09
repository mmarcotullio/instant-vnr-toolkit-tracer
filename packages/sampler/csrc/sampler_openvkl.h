#pragma once

#ifndef PYSAMPLER_SAMPLER_CPP_IMPL
#error "This header is not supposed to be included directly"
#endif

#include "config.h"
#include "sampler.h"

namespace pysampler {

struct VKLContext;

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
struct OpenVKLSampler : SamplerBase {
private:
  struct Impl;
  std::shared_ptr<Impl> pimpl;

public:
  OpenVKLSampler(VolumeFileStructured desc);
  OpenVKLSampler(VolumeDataStructured desc);
  OpenVKLSampler(VolumeFileOpenVDB desc);
  OpenVKLSampler(VolumeFileVTKm desc);
  ~OpenVKLSampler() override;
  pysampler::box3f bounds() const override;
  pysampler::Device device() const override;
  int n_channels() const override;
  void sample(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) override;
  void decode1d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) override;
  void decode3d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, const vec3i& dims) override;
};

} // namespace pysampler
