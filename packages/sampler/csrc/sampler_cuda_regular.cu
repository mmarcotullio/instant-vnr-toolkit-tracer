#define PYSAMPLER_SAMPLER_CPP_IMPL

#include "sampler_cuda.h"

#include <cuda/cuda_buffer.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>

namespace {
template<typename T>
struct maximum_op {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ constexpr T operator()(const T& lhs, const T& rhs) const { return lhs < rhs ? rhs : lhs; }
}; // end maximum
template<typename T>
struct minimum_op {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ constexpr T operator()(const T& lhs, const T& rhs) const { return lhs < rhs ? lhs : rhs; }
}; // end minimum
}

namespace pysampler {

struct CUDAImplStructured {
  CUDAArray arr;
  CUDATexture tex;
};

template<typename T>
static range1f compute_scalar_fminmax(CUDABuffer& data, const uint8_t* _data, size_t count) {
  data.alloc_and_upload(_data, count * sizeof(T));
  auto beg = thrust::device_ptr<const T>((const T*)data.d_pointer());
  auto end = beg + count;
  T vmax = thrust::reduce(beg, end, std::numeric_limits<T>::lowest(), maximum_op<T>());
  T vmin = thrust::reduce(beg, end, std::numeric_limits<T>::max(),    minimum_op<T>());
  return range1f{ (float)vmin, (float)vmax };
}

static void to_float(CUDABuffer& dgpu, size_t count) {
  CUDABuffer dgpu_float;
  dgpu_float.alloc(count * sizeof(float));
  double* d_data = (double*)dgpu.d_pointer();
  float* f_data = (float*)dgpu_float.d_pointer();
  util::parallel_for_gpu(0, nullptr, count, [d_data, f_data] __device__ (size_t i) {
    f_data[i] = (float)d_data[i];
  });
  std::swap(dgpu, dgpu_float);
  dgpu_float.free();
}

void 
CUDASampler::process_structured(const VolumeDataStructured& desc)
{
  const int3 dims = make_int3(desc.dims.x, desc.dims.y, desc.dims.z);
  const size_t count = (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;
  m_ranges.resize(m_n_channels);
  m_fields_structured.resize(m_n_channels);

  for (int i = 0; i < m_n_channels; ++i) {
    const uint8_t* data = desc.data + i * count * sizeof_type(m_type);

    CUDABuffer dgpu;
    
    // std::cout << "m_type " << m_type << std::endl;

    range1f& r = m_ranges[i];
    switch (m_type) {
    case UINT8:  r = compute_scalar_fminmax<uint8_t >(dgpu, data, count); break;
    case INT8:   r = compute_scalar_fminmax<int8_t  >(dgpu, data, count); break;
    case UINT16: r = compute_scalar_fminmax<uint16_t>(dgpu, data, count); break;
    case INT16:  r = compute_scalar_fminmax<int16_t >(dgpu, data, count); break;
    case UINT32: r = compute_scalar_fminmax<uint32_t>(dgpu, data, count); break;
    case INT32:  r = compute_scalar_fminmax<int32_t >(dgpu, data, count); break;
    case FLOAT:  r = compute_scalar_fminmax<float   >(dgpu, data, count); break;
    case DOUBLE: r = compute_scalar_fminmax<double  >(dgpu, data, count); break;
    default: throw std::runtime_error("unknown data type");
    }

    if (m_type == UINT8)  { r = range1f{ 0.f, 1.f }; }
    if (m_type == INT8)   { r = range1f{ 0.f, 1.f }; }
    if (m_type == UINT16) { r = range1f{ 0.f, 1.f }; }
    if (m_type == INT16)  { r = range1f{ 0.f, 1.f }; }
  
    // std::cout << "range: " << r.lower << " " << r.upper << std::endl;

    if (m_type == DOUBLE) { to_float(dgpu, count); }

    m_fields_structured[i] = std::make_shared<CUDAImplStructured>();
    auto& arr = m_fields_structured[i]->arr;
    auto& tex = m_fields_structured[i]->tex;
    switch (m_type) {
    case UINT8:  arr.alloc<uint8_t >(dims); break;
    case INT8:   arr.alloc<int8_t  >(dims); break;
    case UINT16: arr.alloc<uint16_t>(dims); break;
    case INT16:  arr.alloc<int16_t >(dims); break;
    case UINT32: arr.alloc<uint32_t>(dims); break;
    case INT32:  arr.alloc<int32_t >(dims); break;
    case FLOAT:  arr.alloc<float   >(dims); break;
    case DOUBLE: arr.alloc<float   >(dims); break; // double not supported, converted to float
    default: throw std::runtime_error("unknown data type");
    }
    arr.fill((void*)dgpu.d_pointer(), true);
    tex.bind(arr);
  }

  m_bounds = box3f(vec3f(0.f), vec3f(dims) * desc.spacing);
  m_sampling_mode = STRUCTURED;
}

void 
sample_structured(const CUDASampler* self, void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count, void* stream) 
{
  const auto i_scale = upper-lower;
  const auto i_lower = lower;

  for (int attr = 0; attr < self->m_n_channels; ++attr) {
    const auto& range = self->m_ranges[attr];

    const auto& tex = self->m_fields_structured[attr]->tex.handler;    
    vec3f* coords = (vec3f*)d_coords;
    float* values = (float*)d_values + attr * count;

    const float o_scale = 1.f / range.span();
    const float o_lower = -range.lower * o_scale;
    // std::cout << "o_scale: " << o_scale << " o_lower: " << o_lower << std::endl;

    util::parallel_for_gpu(0, (cudaStream_t)stream, count, 
    [coords, values, tex, i_lower, i_scale, o_lower, o_scale] __device__ (size_t i) {
      const auto pp = coords[i];
      const auto p = vec3f(
        fmaf(pp.x, i_scale.x, i_lower.x),
        fmaf(pp.y, i_scale.y, i_lower.y),
        fmaf(pp.z, i_scale.z, i_lower.z)
      );
      const auto v = tex3D<float>(tex, p.x, p.y, p.z);
      coords[i] = p;
      values[i] = fmaf(v, o_scale, o_lower);
    });
  }
}

}

#ifdef ENABLE_SHADOW_DECODE
#include "ovr/serializer/serializer.h"
#endif

namespace pysampler {

namespace {

#define float_large 1e20f
#define float_small std::numeric_limits<float>::min() // to avoid division by zero
#define float_epsilon std::numeric_limits<float>::epsilon()
#define nearly_one 0.9999f

template <typename T>
__forceinline__ __device__ T 
lerp(float r, const T& a, const T& b) 
{
  return (1-r) * a + r * b;
}

__forceinline__ __device__ bool 
intersectBox(float& _t0, float& _t1, const vec3f ray_org, const vec3f ray_dir, const vec3f lower, const vec3f upper)
{
  float t0 = _t0;
  float t1 = _t1;
  const vec3i is_small = vec3i(fabs(ray_dir.x) <= float_small, fabs(ray_dir.y) <= float_small, fabs(ray_dir.z) <= float_small);
  const vec3f rcp_dir = vec3f(__frcp_rn(ray_dir.x), __frcp_rn(ray_dir.y), __frcp_rn(ray_dir.z));
  const vec3f t_lo = vec3f(is_small.x ? float_large : (lower.x - ray_org.x) * rcp_dir.x, //
                           is_small.y ? float_large : (lower.y - ray_org.y) * rcp_dir.y, //
                           is_small.z ? float_large : (lower.z - ray_org.z) * rcp_dir.z  //
  );
  const vec3f t_hi = vec3f(is_small.x ? -float_large : (upper.x - ray_org.x) * rcp_dir.x, //
                           is_small.y ? -float_large : (upper.y - ray_org.y) * rcp_dir.y, //
                           is_small.z ? -float_large : (upper.z - ray_org.z) * rcp_dir.z  //
  );
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
  _t0 = t0;
  _t1 = t1;
  return t1 > t0;
}

}


}
