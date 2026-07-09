#define PYSAMPLER_SAMPLER_CPP_IMPL
#include "sampler_openvkl.h"
#ifdef ENABLE_VTKM
#include "sampler_vtkm.h"
#endif
#include "generate_random_numbers.h"
#include "config.h"


#include <openvkl/openvkl.h>
#include <openvkl/device/openvkl.h>

#include <cmath>
#include <cstring>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>

#ifdef OPENVKL_UTILITY_VDB_OPENVDB_ENABLED
#include <rkcommon/math/vec.h>
#include <rkcommon/math/box.h>
#include <rkcommon/math/constants.h>
// Include OpenVDB's generated version header first to avoid accidental
// shadowing by OpenVKL's own version.h when include search order differs.
#include <openvdb/version.h>
namespace openvkl::utility::vdb {
  using box3i = rkcommon::math::box3i;
  using rkcommon::math::one;
  using rkcommon::math::empty;
}
#include <openvkl/utility/vdb/OpenVdbGrid.h>
using openvkl::utility::vdb::OpenVdbFloatGrid;
using openvkl::utility::vdb::OpenVdbVec3sGrid;
#endif

namespace pysampler {

struct VKLImplOpenVDB;
struct VKLImplVTKm;

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

struct OpenVKLSampler::Impl {
  const dtype type{};
  const int n_channels{};
  std::vector<range1f> ranges;
  box3f bounds;

  VKLVolume  volume;
  VKLSampler sampler;

  std::shared_ptr<VKLContext> ctx;

  Impl(dtype type, int n_channels, const std::vector<range1f>& ranges);
  ~Impl();
  void finalize();

  /// Structured Data
  struct {
    std::shared_ptr<char[]> data;
    vec3i dims;
    vec3f spacing;
  } structured;

  /// Unstructured Data
  struct {
    std::shared_ptr<vec3f[]> vertex_position;
    std::shared_ptr<float[]> vertex_data;
    std::shared_ptr<uint64_t[]> index;
    std::shared_ptr<uint64_t[]> cell_index;
    std::shared_ptr<uint8_t[]> cell_type;
    uint64_t num_vertices;
    uint64_t num_indices;
    uint64_t num_cells;
  } unstructured;

  /// OpenVDB Data
  std::shared_ptr<VKLImplOpenVDB> data_openvdb;

  /// Structured Data Loader
  void process_structured();
  void process_unstructured();
};

struct VKLContext {
public:
  VKLDevice device;
  bool is_gpu = false;
public:
  ~VKLContext() { vklReleaseDevice(device); }
  VKLContext() {
    vklInit();
#ifdef ENABLE_XPU // TODO UNTESTED !!
    device = vklNewDevice("cpu"); // TODO: support also GPU devices
    is_gpu = false;
#else
    device = vklNewDevice("cpu"); // TODO: support also GPU devices
    is_gpu = false;
#endif
    vklCommitDevice(device);
  }
};


/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

template<typename T>
static std::pair<T, T> compute_scalar_minmax(const void* _array, size_t count, size_t stride) {
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  if (stride == 0) stride = sizeof(T);

  const T* array = (const T*)_array;
  auto value = [array, stride](size_t index) -> T {
    const auto begin = (const uint8_t*)array;
    const auto curr = (T*)(begin + index * stride);
    return static_cast<T>(*curr);
  };

  T init;

  init = std::numeric_limits<T>::lowest();
  T actual_max = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::max(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::max(x, y); });

  init = std::numeric_limits<T>::max();
  T actual_min = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::min(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::min(x, y); });

  return std::make_pair(actual_min, actual_max);
}

template<typename T>
static range1f compute_scalar_fminmax(const void* _array, size_t count) {
  const auto r = compute_scalar_minmax<T>(_array, count, 0);
  return range1f{ (float)r.first, (float)r.second };
}

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

static VKLDataType vkl_dtype(dtype type) {
  switch (type) {
  case UINT8:  return VKL_UCHAR; 
  case INT8:   return VKL_CHAR;  
  case UINT16: return VKL_USHORT;
  case INT16:  return VKL_SHORT; 
  case UINT32: return VKL_UINT;  
  case INT32:  return VKL_INT;   
  case FLOAT:  return VKL_FLOAT; 
  case DOUBLE: return VKL_DOUBLE;
  default: throw std::runtime_error("unsupported OpenVKL data type");
  }
}

template<int W=16>
static void vkl_sample_cpu(VKLSampler* sampler, vec3f* o_coords, float* o_values, size_t count, const box3f& domain, const box3f& bbox, const range1f& range, int attr, float time) {
  tbb::parallel_for(size_t(0), div_round_up<size_t>(count, W), [&](size_t g) 
  {
    // recompute coordinates in object space
    int valid[W] = {};
    vec3f pos[W];
    for (int i = 0; (i < W) && (g*W+i < count); ++i) {
      valid[i] = -1;
      pos[i] = o_coords[g*W+i] = o_coords[g*W+i] * domain.span() + domain.lower;
      pos[i] = std::clamp(bbox.lower + pos[i] * bbox.span(), bbox.lower, bbox.upper);
    }

    // compute samples
    float samples[W];
    if constexpr (W == 1) {
      const float times[1] = { time };
      const vkl_vec3f coord1 = { pos[0].x, pos[0].y, pos[0].z };
      *samples = vklComputeSample(sampler, &coord1, attr, *times);
    }
    else if constexpr (W == 4) {
      const float times[4] = { time, time, time, time };
      const vkl_vvec3f4 coord4 = {
        { pos[0].x, pos[1].x, pos[2].x, pos[3].x },
        { pos[0].y, pos[1].y, pos[2].y, pos[3].y },
        { pos[0].z, pos[1].z, pos[2].z, pos[3].z } 
      };
      vklComputeSample4(valid, sampler, &coord4, samples, attr, times);
    }
    else if constexpr (W == 8) {
      const float times[8] = { time, time, time, time, time, time, time, time };
      const vkl_vvec3f8 coord8 = {
        { pos[0].x, pos[1].x, pos[2].x, pos[3].x, pos[4].x, pos[5].x, pos[6].x, pos[7].x },
        { pos[0].y, pos[1].y, pos[2].y, pos[3].y, pos[4].y, pos[5].y, pos[6].y, pos[7].y },
        { pos[0].z, pos[1].z, pos[2].z, pos[3].z, pos[4].z, pos[5].z, pos[6].z, pos[7].z } 
      };
      vklComputeSample8(valid, sampler, &coord8, samples, attr, times);
    }
    else if constexpr (W == 16) {
      const float times[16] = { time, time, time, time, time, time, time, time, time, time, time, time, time, time, time, time };
      const vkl_vvec3f16 coord16 = {
        { pos[0].x, pos[1].x, pos[2].x, pos[3].x, pos[4].x, pos[5].x, pos[6].x, pos[7].x, pos[8].x, pos[9].x, pos[10].x, pos[11].x, pos[12].x, pos[13].x, pos[14].x, pos[15].x },
        { pos[0].y, pos[1].y, pos[2].y, pos[3].y, pos[4].y, pos[5].y, pos[6].y, pos[7].y, pos[8].y, pos[9].y, pos[10].y, pos[11].y, pos[12].y, pos[13].y, pos[14].y, pos[15].y },
        { pos[0].z, pos[1].z, pos[2].z, pos[3].z, pos[4].z, pos[5].z, pos[6].z, pos[7].z, pos[8].z, pos[9].z, pos[10].z, pos[11].z, pos[12].z, pos[13].z, pos[14].z, pos[15].z } 
      };
      vklComputeSample16(valid, sampler, &coord16, samples, attr, times);
    }
    else {
      throw std::runtime_error("unsupported vector width: " + std::to_string(W));
    }

    // normalize output values
    for (int i = 0; i < W && valid[i]; ++i) {
      o_values[g*W+i] = (samples[i] - range.lower) / range.span();
    }
  });
}

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

OpenVKLSampler::Impl::Impl(dtype type, int n_channels, const std::vector<range1f>& ranges) 
  : type(type), n_channels(n_channels), ranges(ranges)
{
  ctx = std::make_shared<VKLContext>();
}

void
OpenVKLSampler::Impl::finalize()
{
  auto bbox = vklGetBoundingBox(volume);
  bounds = (box3f&)bbox;
  sampler = vklNewSampler(volume);
  vklCommit(sampler);
}

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

OpenVKLSampler::Impl::~Impl() {
  if (sampler) {
    vklRelease(sampler);
  }
  if (volume) {
    vklRelease(volume);
  }
}

pysampler::box3f 
OpenVKLSampler::bounds() const { 
  return pimpl->bounds;
}

pysampler::Device 
OpenVKLSampler::device() const { 
  return pysampler::Device::CPU;
}

int 
OpenVKLSampler::n_channels() const { 
  return pimpl->n_channels;
}

OpenVKLSampler::~OpenVKLSampler() {
  pimpl.reset();
}

void
OpenVKLSampler::sample(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) {
  generate_random_uniforms((float*)d_coords, count * 3, pysampler::Device::CPU);
  decode1d(d_coords, d_values, lower, upper, count);
}

void 
OpenVKLSampler::decode1d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) {
  vec3f* coords = (vec3f*)d_coords;
  for (int attr = 0; attr < pimpl->n_channels; ++attr) {
    float* values = (float*)d_values + attr * count;
    vkl_sample_cpu(&pimpl->sampler, coords, values, count, box3f(lower, upper), pimpl->bounds, pimpl->ranges[attr], attr, 0.f);
  }
}

void 
OpenVKLSampler::decode3d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, const vec3i& dims) {
  throw std::runtime_error("not implemented");
}

/////////////////////////////////////////////////////////////////////
// Regular Grid
/////////////////////////////////////////////////////////////////////

void 
OpenVKLSampler::Impl::process_structured()
{
  const auto& handler = structured.data;
  const auto& dims = structured.dims;
  const auto count = (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;

  // compute value ranges
  ranges.resize(n_channels);
  for (int i = 0; i < n_channels; ++i) {
    auto& range = ranges[i];
    auto* data = handler.get() + i * count * sizeof_type(type);
    switch (type) {
    case UINT8:  range = compute_scalar_fminmax<uint8_t >(data, count); break;
    case INT8:   range = compute_scalar_fminmax<int8_t  >(data, count); break;
    case UINT16: range = compute_scalar_fminmax<uint16_t>(data, count); break;
    case INT16:  range = compute_scalar_fminmax<int16_t >(data, count); break;
    case UINT32: range = compute_scalar_fminmax<uint32_t>(data, count); break;
    case INT32:  range = compute_scalar_fminmax<int32_t >(data, count); break;
    case FLOAT:  range = compute_scalar_fminmax<float   >(data, count); break;
    case DOUBLE: range = compute_scalar_fminmax<double  >(data, count); break;
    default: throw std::runtime_error("unknown data type");
    }
  }

  // Create vplume  
  std::vector<VKLData> attributes;
  for (int i = 0; i < n_channels; ++i) {
    auto* dptr = handler.get() + i * count * sizeof_type(type);
    VKLData attribute = vklNewData(ctx->device, count, vkl_dtype(type), dptr, VKL_DATA_SHARED_BUFFER);
    attributes.push_back(attribute);
  }

  VKLData attributes_data = vklNewData(ctx->device, attributes.size(), VKL_DATA, attributes.data());
  for (auto &attribute : attributes) {
    vklRelease(attribute);
  }

  volume = vklNewVolume(ctx->device, "structuredRegular");
  vklSetVec3i(volume, "dimensions", dims.x, dims.y, dims.z);
  vklSetVec3f(volume, "gridOrigin", 0.f, 0.f, 0.f);
  vklSetVec3f(volume, "gridSpacing", structured.spacing.x, structured.spacing.y, structured.spacing.z);
  vklSetFloat(volume, "background", 0.f);
  vklSetBool(volume, "cellCentered", false);
  vklSetData(volume, "data", attributes_data);
  vklRelease(attributes_data);
  vklCommit(volume);

  // Create sampler
  finalize();
}

OpenVKLSampler::OpenVKLSampler(VolumeFileStructured desc) 
  : SamplerBase(), pimpl(std::make_shared<Impl>(desc.type, desc.n_channels, desc.ranges))
{
  pimpl->structured.data = load_regular_grid(desc.filename, desc.dims, desc.n_channels, desc.type, desc.offset, desc.is_big_endian);
  pimpl->structured.dims = desc.dims;
  pimpl->structured.spacing = desc.spacing;
  pimpl->process_structured();
}

OpenVKLSampler::OpenVKLSampler(VolumeDataStructured desc)
  : SamplerBase(), pimpl(std::make_shared<Impl>(desc.type, desc.n_channels, desc.ranges))
{
  pimpl->structured.data = std::shared_ptr<char[]>((char*)desc.data, [](char*){});
  pimpl->structured.dims = desc.dims;
  pimpl->structured.spacing = desc.spacing;
  pimpl->process_structured();
}

/////////////////////////////////////////////////////////////////////
// OpenVDB 
/////////////////////////////////////////////////////////////////////

template<typename T>
std::shared_ptr<T[]> copy_vector(std::vector<T>& src) {
  std::shared_ptr<T[]> dst = std::shared_ptr<T[]>(new T[src.size()], [] (void*) {});
  std::memcpy(dst.get(), src.data(), src.size() * sizeof(T));
  src.clear();
  return dst;
}

#ifdef ENABLE_VTKM
OpenVKLSampler::OpenVKLSampler(VolumeFileVTKm desc)
  : SamplerBase(), pimpl(std::make_shared<Impl>(pysampler::FLOAT, 1, std::vector<range1f>{}))
{
  VTKmDataHelper helper(desc);

  pimpl->unstructured.num_vertices = helper.unstructured.vertex_position.size();
  pimpl->unstructured.num_indices = helper.unstructured.index.size();
  pimpl->unstructured.num_cells = helper.unstructured.cell_type.size();

  pimpl->unstructured.vertex_position = copy_vector(helper.unstructured.vertex_position);
  pimpl->unstructured.vertex_data = copy_vector(helper.unstructured.vertex_data);
  pimpl->unstructured.index = copy_vector(helper.unstructured.index);
  pimpl->unstructured.cell_index = copy_vector(helper.unstructured.cell_index);
  pimpl->unstructured.cell_type = copy_vector(helper.unstructured.cell_type);

  pimpl->ranges = helper.ranges;
  pimpl->process_unstructured();
}
#else
OpenVKLSampler::OpenVKLSampler(VolumeFileVTKm /*desc*/)
  : SamplerBase(), pimpl(std::make_shared<Impl>(pysampler::FLOAT, 1, std::vector<range1f>{}))
{
  throw std::runtime_error("VTKm support is disabled (build with VTKm installed to enable)");
}
#endif

void 
OpenVKLSampler::Impl::process_unstructured()
{
  VKLData vertex_position = vklNewData(ctx->device, 
    unstructured.num_vertices, VKL_VEC3F, 
    unstructured.vertex_position.get(), VKL_DATA_SHARED_BUFFER);
  VKLData vertex_data = vklNewData(ctx->device, 
    unstructured.num_vertices, VKL_FLOAT, 
    unstructured.vertex_data.get(), VKL_DATA_SHARED_BUFFER);

  VKLData index = vklNewData(ctx->device, 
    unstructured.num_indices, VKL_ULONG, 
    unstructured.index.get(), VKL_DATA_SHARED_BUFFER);

  VKLData cell_index = vklNewData(ctx->device, 
    unstructured.num_cells, VKL_ULONG, 
    unstructured.cell_index.get(), VKL_DATA_SHARED_BUFFER);
  VKLData cell_type = vklNewData(ctx->device, 
    unstructured.num_cells, VKL_UCHAR, 
    unstructured.cell_type.get(), VKL_DATA_SHARED_BUFFER);

  volume = vklNewVolume(ctx->device, "unstructured");
  vklSetData(volume, "vertex.position", vertex_position);
  vklSetData(volume, "vertex.data", vertex_data);
  vklSetData(volume, "index", index);
  vklSetData(volume, "cell.index", cell_index);
  vklSetData(volume, "cell.type", cell_type);
  vklSetBool(volume, "indexPrefixed", false);
  vklSetFloat(volume, "background", 0.f);
  vklCommit(volume);

  vklRelease(vertex_position);
  vklRelease(vertex_data);
  vklRelease(index);
  vklRelease(cell_index);
  vklRelease(cell_type);

  finalize();
}

#ifdef OPENVKL_UTILITY_VDB_OPENVDB_ENABLED
struct VKLImplOpenVDB {
  std::shared_ptr<OpenVdbFloatGrid> float_grid;
  std::shared_ptr<OpenVdbVec3sGrid> vec3_grid;
};

static VKLVolume
load_vdb_with_openvkl(VKLDevice device,
                      const std::string& filename,
                      const std::string& field,
                      std::shared_ptr<VKLImplOpenVDB>& holder)
{
  holder = std::make_shared<VKLImplOpenVDB>();

  try {
    holder->float_grid = std::make_shared<OpenVdbFloatGrid>(
      device, filename, field, /*deferLeaves=*/false, /*repackNodes=*/false
    );
    return holder->float_grid->createVolume(/*commit=*/false);
  }
  catch (const std::exception& float_error) {
    try {
      holder->vec3_grid = std::make_shared<OpenVdbVec3sGrid>(
        device, filename, field, /*deferLeaves=*/false, /*repackNodes=*/false
      );
      return holder->vec3_grid->createVolume(/*commit=*/false);
    }
    catch (const std::exception& vec_error) {
      throw std::runtime_error(
        "failed to load OpenVDB grid '" + field + "' from '" + filename +
        "' via OpenVKL.\nfloat loader error: " + float_error.what() +
        "\nvec3 loader error: " + vec_error.what()
      );
    }
  }
}
#endif

OpenVKLSampler::OpenVKLSampler(VolumeFileOpenVDB desc)
  : SamplerBase(), pimpl(std::make_shared<Impl>(pysampler::FLOAT, 1, std::vector<range1f>{}))
{
#ifndef OPENVKL_UTILITY_VDB_OPENVDB_ENABLED
  throw std::runtime_error("OpenVDB support is disabled");
#else

  const std::string path = desc.filename;
  const std::string field = desc.field;
  pimpl->volume = load_vdb_with_openvkl(pimpl->ctx->device, path, field, pimpl->data_openvdb);

  // Keep filtering behavior aligned with instantvnr's OpenVDB path.
  vklSetInt(pimpl->volume, "filter", VKL_FILTER_LINEAR);
  vklSetFloat(pimpl->volume, "background", 0.f);
  vklCommit(pimpl->volume);

  pimpl->ranges.resize(pimpl->n_channels);
  {
    // Query after commit; pre-commit value ranges may be undefined/degenerate
    // for some OpenVKL volume backends.
    const auto r = vklGetValueRange(pimpl->volume, /*attributeIndex=*/0);
    if (std::isfinite(r.lower) && std::isfinite(r.upper) && r.upper > r.lower) {
      pimpl->ranges[0] = range1f{ (float)r.lower, (float)r.upper };
    }
    else {
      // OpenVKL test loaders normalize OpenVDB scalar fields to [0,1].
      // Use that as a safe fallback to avoid NaN from divide-by-zero.
      pimpl->ranges[0] = range1f{ 0.f, 1.f };
    }
  }

  // Create sampler
  pimpl->finalize();

#endif
}

} // namespace vnr
