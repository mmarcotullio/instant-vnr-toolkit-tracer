#pragma once

#ifndef PYSAMPLER_SAMPLER_CPP_IMPL
#error "This header is not supposed to be included directly"
#endif

#include "sampler.h"

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace vtkm::cont { class DataSet; }

namespace pysampler {

struct VTKmDataHelper {
public:
  struct {
    std::vector<vec3f> vertex_position;
    std::vector<float> vertex_data;
    std::vector<uint64_t> index;
    std::vector<uint64_t> cell_index;
    std::vector<uint8_t> cell_type;
  } unstructured;

  struct {
    std::vector<float> grid;
    vec3i dims;
    vec3f origin;
    vec3f spacing;
  } structured;

  std::vector<range1f> ranges;

  VTKmDataHelper(VolumeFileVTKm desc);

private:
  struct Impl;
  std::shared_ptr<Impl> pimpl;

  void process_vtkm_init(std::string field);
  void process_vtkm_add(const vtkm::cont::DataSet& dset);
  void process_vtkm_finalize();
};

} // namespace vnr
