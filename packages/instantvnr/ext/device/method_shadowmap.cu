//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "methods.h"
#include "raytracing.h"
#include "dda.h"

#include <cuda/cuda_buffer.h>

#ifdef ADAPTIVE_SAMPLING
#undef ADAPTIVE_SAMPLING
#endif
#define ADAPTIVE_SAMPLING false

namespace ovr::hyperinr {

using random::RandomTEA;

constexpr auto N_ITERS = 16;

using ShadingMode = MethodShadowMap::ShadingMode;
constexpr auto SHADING = MethodShadowMap::SHADING;

struct Ray;

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<bool ADAPTIVE>
struct ShadowMapIter;

template<>
struct ShadowMapIter<true> : private dda::DDAIter {
public:
  using DDAIter::cell;
  using DDAIter::t_next;
  using DDAIter::next_cell_begin;
public:
  __device__ ShadowMapIter() {}
  __device__ ShadowMapIter(const VNRDeviceVolume& self, const Ray& ray);
  __device__ bool
  resumable(const VNRDeviceVolume& self, const Ray& ray);

  template<typename F>
  __device__ void 
  exec(const VNRDeviceVolume& self, const Ray& ray, 
       const float step, const uint32_t pidx, const F& body);
};

template<>
struct ShadowMapIter<false>{
public:
  float next_cell_begin{};
private:
  uint32_t __pad[2];
public:
  __device__ ShadowMapIter() {}
  __device__ ShadowMapIter(const VNRDeviceVolume& self, const Ray& ray);
  __device__ bool
  resumable(const VNRDeviceVolume& self, const Ray& ray);

  template<typename F>
  __device__ void 
  exec(const VNRDeviceVolume& self, const Ray& ray, 
       const float step, const uint32_t pidx, const F& body);
};

using Iter = ShadowMapIter<ADAPTIVE_SAMPLING>;



// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

struct PackedPayload {
  // 0 //
  vec3f color_or_org;
  float alpha;
  // 16 //
  uint32_t pidx = 0;
  float jitter = 0.f;
  Iter iter; // 3* float
  uint32_t __pad[3];
};

struct Ray {
  vec3f org{};
  vec3f dir{};
  float tnear = 0.f;
  float tfar = float_large;

  uint32_t pidx = 0;
  float jitter = 0.f;
  Iter iter;
};

struct RayPayload : Ray {
  float alpha = 0.f;
  vec3f color = 0.f;
};

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

using Packet = PacketTemplate<PackedPayload>;
static_assert((sizeof(PackedPayload)) % sizeof(uint4) == 0, "Incorrect size of PackedRay");

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

inline __device__
ShadowMapIter<true>::ShadowMapIter(const VNRDeviceVolume& self, const Ray& ray)
{
  const auto org = ray.org;
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  DDAIter::init(m_org, m_dir, tMin, tMax, dims);
}

template<typename F>
inline __device__ void
ShadowMapIter<true>::exec(const VNRDeviceVolume& self, const Ray& ray, const float step, const uint32_t pidx, const F& body)
{
  const auto org = ray.org;
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;

  const auto lambda = [&](const vec3i& cell, float t0, float t1) {
    // calculate max opacity
    float r = opacityUpperBound(self, cell);
    if (fabsf(r) <= float_epsilon) return true; // the cell is empty
    // estimate a step size
    const auto ss = adaptiveSamplingRate(step, r);
    // iterate within the interval
    vec2f t = vec2f(t0, min(t1, t0 + ss));
    while (t.y > t.x) {
      DDAIter::next_cell_begin = t.y - tMin;
      if (!body(t)) return false;
      t.x = t.y;
      t.y = min(t.x + ss, t1);
    }
    return true;
  };

  while (DDAIter::next(m_org, m_dir, tMin, tMax, dims, false, lambda)) {}
}

inline __device__ bool 
ShadowMapIter<true>::resumable(const VNRDeviceVolume& self, const Ray& ray)
{
  const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  const auto& dims = self.macrocell_dims;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  return DDAIter::resumable(m_dir, tMin, tMax, dims);
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------


inline __device__
ShadowMapIter<false>::ShadowMapIter(const VNRDeviceVolume& self, const Ray& ray)
{
}

template<typename F>
inline __device__ void
ShadowMapIter<false>::exec(const VNRDeviceVolume& self, const Ray& ray, const float step, const uint32_t pidx, const F& body)
{
  // MAYBE_UNUSED const auto org = ray.org;
  // MAYBE_UNUSED const auto dir = ray.dir;
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  vec2f t;
  t.x = max(tMin + next_cell_begin, tMin);
  t.y = min(t.x + step, tMax);
  while (t.y > t.x) {
    next_cell_begin = t.y - tMin;
    if (!body(t)) return;
    t.x = t.y;
    t.y = min(t.x + step, tMax);
  }
  next_cell_begin = float_large;
  return;
}

inline __device__ bool 
ShadowMapIter<false>::resumable(const VNRDeviceVolume& self, const Ray& ray)
{
  const auto tMin = ray.tnear;
  const auto tMax = ray.tfar;

  return tMin + next_cell_begin < tMax;
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

struct ShadowMapData : LaunchParams
{
  ShadowMapData(const LaunchParams& p) : LaunchParams(p) {}

  ShadingMode mode;

  VNRDeviceVolume volume;

  vec3f* __restrict__ inference_coords{ nullptr };
  float* __restrict__ inference_values{ nullptr };
  float* __restrict__ scalar_inference_values{ nullptr };
  uint32_t shadow_values_offset{ 0 };

  // belows are only useful for sampling streaming
  uint8_t* flags{ nullptr };

  // per ray payload (ordered by ray index) //
  Packet::DataType* packets[Packet::N] = {};

  // ordered by pixel index //
  vec4f* __restrict__ final_org_and_jitter{ nullptr };
  vec4f* __restrict__ final_highest_rgba{ nullptr };
  vec4f* __restrict__ shading_rgba{ nullptr };
};

struct ShadowMapIterData {
  Packet packet;
};

struct ShadowMapIterView {
  typedef ShadowMapIterView IterView;
  typedef ShadowMapIterData IterData;

  Packet::DataType* packets[Packet::N];

  __host__ __device__ ShadowMapIterView() {}

  __host__ ShadowMapIterView(const ShadowMapData& params, uint32_t offset) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      this->packets[i] = params.packets[i] + offset;
    }
  }

  __host__ __device__ void move(ptrdiff_t n) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      this->packets[i] += n;
    }
  }

  __host__ __device__ ptrdiff_t diff(const IterView& other) const {
    return this->packets[0] - other.packets[0];
  }

  /// Conversion operator
  __host__ __device__ operator IterData() const {
    IterData value;
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      value.packet.data[i] = *this->packets[i];
    }
    return value;
  }

  /// Assignment operator
  __host__ __device__ IterView& operator=(const IterData& value) {
    #pragma unroll
    for (int i = 0; i < Packet::N; ++i) {
      *this->packets[i] = value.packet.data[i];
    }
    return *this;
  }
};

using IterView = ShadowMapIterView;
using IterData = ShadowMapIterData;

// NOTE: what is the best SoA layout here?

__forceinline__ __device__ bool intersectVolume(Ray& ray, const VNRDeviceVolume& self) {
  return intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self);
}

inline __device__ void 
compute_ray(const ShadowMapData& params, RayPayload& ray)
{
  const auto& fbIndex = ray.pidx;

  // compute pixel ID
  const uint32_t ix = fbIndex % params.frame.size.x;
  const uint32_t iy = fbIndex / params.frame.size.x;

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f& otw = params.volume.transform;
  const affine3f wto = otw.inverse();

  // generate ray direction
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
}

inline __device__ RayPayload
load(const ShadowMapData& params, const uint32_t ridx) 
{
  Packet packet;

  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    packet.data[i] = params.packets[i][ridx];
  }

  RayPayload ray;
  ray.pidx = packet.items.pidx;
  ray.jitter = packet.items.jitter;
  ray.alpha = packet.items.alpha;
  ray.color = packet.items.color_or_org;
  ray.iter = packet.items.iter;

  compute_ray(params, ray);

  const bool hashit = intersectVolume(ray, params.volume);
  assert(hashit);

  return ray;
}

inline __device__ void
save(const ShadowMapData& params, const RayPayload& ray, const uint32_t ridx)
{
  Packet packet;
  packet.items.pidx = ray.pidx;
  packet.items.jitter = ray.jitter;
  packet.items.alpha = ray.alpha;
  packet.items.color_or_org = ray.color;
  packet.items.iter = ray.iter;

  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    params.packets[i][ridx] = packet.data[i];
  }

  params.flags[ridx] = 0xFF;
}

/* iterative version */ void
do_shadowmap_iterative(cudaStream_t stream, const ShadowMapData& params, const IterativeSampler& sampler, uint32_t numPixels);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T>
inline T* 
define_buffer(char* begin, size_t& offset, size_t buffer_size)
{
  auto* ret = (T*)(begin + offset); 
  offset += buffer_size * sizeof(T);
  return ret;
}

void
MethodShadowMap::render(cudaStream_t stream, const LaunchParams& _params, const VNRDeviceVolume& volume, const IterativeSampler& sampler, ShadingMode mode, bool dual_nn)
{
  if (!dual_nn && volume.volume.data == 0) {
    warn_once("[Error] scalar field is required for shadowmap renderer, but it is not available. thus black screen is expected.");
  }

  ShadowMapData params = _params;
  params.volume = volume;
  params.mode = mode;

  const uint32_t numPixels = (uint32_t)params.frame.size.long_product();
  const uint32_t numPixelsPadded = util::next_multiple(numPixels, 256U);
  const uint32_t nSamplesPerCoord = N_ITERS;
  const uint32_t nValues = numPixelsPadded * nSamplesPerCoord;

  size_t nBytes = nValues * sizeof(vec4f); // inference coords(vec3f) + values(float)
  if (dual_nn) {
    nBytes += nValues * sizeof(float);     // extra scalar inference values
  }
  nBytes += numPixelsPadded * sizeof(uint8_t);
  nBytes += numPixelsPadded * sizeof(Packet);

  sample_streaming_buffer.resize(nBytes, stream);
  sample_streaming_buffer.memset(0, stream);
  char* begin = (char*)sample_streaming_buffer.d_pointer();
  size_t offset = 0;

  params.inference_coords = define_buffer<vec3f>(begin, offset, nValues);
  if (dual_nn) {
    // dual_nn layout: [scalar_values(nValues) | shadow_values(nValues)]
    // The combined sampler writes scalar to the first half and shadow to the second.
    params.inference_values = define_buffer<float>(begin, offset, 2 * nValues);
    params.scalar_inference_values = params.inference_values;
    params.shadow_values_offset = nValues;
  } else {
    params.inference_values = define_buffer<float>(begin, offset, nValues);
  }

  #pragma unroll
  for (int i = 0; i < Packet::N; ++i) {
    params.packets[i] = define_buffer<Packet::DataType>(begin, offset, numPixelsPadded);
  }

  params.flags = define_buffer<uint8_t>(begin, offset, numPixelsPadded);
  assert(offset == nBytes);

  do_shadowmap_iterative(stream, params, sampler, numPixels);
}

// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

__global__ void
iterative_intersect_kernel(uint32_t numRays, const ShadowMapData params, int N_ITERS)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = params.volume;
  MAYBE_UNUSED const auto& volume = self.volume;

  RayPayload ray = load(params, i);

  vec3f* __restrict__ coords = (vec3f*)params.inference_coords;

  int k = 0;
  ray.iter.exec(self, ray, self.step, ray.pidx, [&] (const vec2f& t) {
    assert(k < N_ITERS);
    assert(t.x < t.y);
    const vec3f c = ray.org + lerp(ray.jitter, t.x, t.y) * ray.dir;
    coords[numRays * k + i] = c;
    return (++k) < N_ITERS;
  });
}

__global__ void
iterative_compose_kernel(uint32_t numRays, const ShadowMapData params, int N_ITERS)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = params.volume;
  const auto& otw = params.volume.transform;
  const auto* __restrict__ shadows = params.inference_values + params.shadow_values_offset;
  
  RayPayload ray = load(params, i);

  int k = 0;
  ray.iter.exec(self, ray, self.step, ray.pidx, [&] (const vec2f& t) {
    assert(k < N_ITERS);
    assert(t.x < t.y);

    // classification
    const auto c = ray.org + lerp(ray.jitter, t.x, t.y) * ray.dir;
    const auto sampleValue = params.scalar_inference_values
        ? params.scalar_inference_values[numRays * k + i]
        : sampleVolume(self.volume, c);
    vec3f sampleColor;
    float sampleAlpha;
    sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
    opacityCorrection(self, t.y - t.x, sampleAlpha);

    // access gradient (only when sampling from GT volume; unused in shadow map shading)
    if (!params.scalar_inference_values) {
      const vec3f No = -sampleGradient(self.volume, c, sampleValue, self.grad_step);
      const vec3f Nw = xfmNormal(otw, No);
      (void)Nw;
    }

    // const vec3f ambient_light = vec3f(0.1, 0.1, 0.1);
    const vec3f ambient_light_strength = vec3f(1.4, 1.4, 1.4);
    // shading
    if (params.mode == SHADING) {
      float coef = clamp(shadows[numRays * k + i], 0.f, 1.f);
      sampleColor = lerp(0.9, sampleColor * ambient_light_strength, coef * sampleColor * ambient_light_strength);
      // sampleColor = lerp(0.9, sampleColor, coef * sampleColor) + ambient_light;
    }

    // blending
    const float tr = 1.f - ray.alpha;
    ray.alpha += tr * sampleAlpha;
    ray.color += tr * sampleColor * sampleAlpha;

    // conditions to continue iterating
    return ((++k) < N_ITERS) && (ray.alpha < nearly_one);
  });

  const bool resumable = ray.iter.resumable(self, ray);
  if (ray.alpha < nearly_one && resumable) {
    save(params, ray, i);
  }
  else {
    const uint32_t& pidx = ray.pidx;
    writePixelColor(params, vec4f(ray.color, ray.alpha), pidx);
  }
}

__global__ void
iterative_raygen_kernel_camera(uint32_t numRays, const ShadowMapData params) 
{
  // compute ray ID
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  // generate data
  const auto& self = params.volume;

  // random number generator
  RandomTEA rng = RandomTEA(params.frame_index, i);
  vec2f jitters = rng.get_floats();

  // payload & ray
  RayPayload ray;  
  ray.pidx = i;
  ray.jitter = jitters.x;
  compute_ray(params, ray);

  // intersect with volume bbox & write outputs
  if (intersectVolume(ray, self)) {
    ray.iter = Iter(self, ray);
    save(params, ray, i);
  }
  else {
    writePixelColor(params, vec4f(ray.color, ray.alpha), ray.pidx);
  }
}

inline bool 
iterative_ray_compaction(cudaStream_t stream, uint32_t& numRays, const ShadowMapData& params)
{
  return inplace_compaction<IterData, IterView>(stream, numRays, params.flags, IterView(params, 0)) > 0;
}

void iterative_shadowmap_loop(cudaStream_t stream, const ShadowMapData& params, const IterativeSampler& sampler, uint32_t numRays)
{
  CUDA_CHECK(cudaStreamSynchronize(stream));
  util::linear_kernel(iterative_raygen_kernel_camera, 0, stream, numRays, params);
  TRACE_CUDA;
  while (iterative_ray_compaction(stream, numRays, params)) {
    TRACE_CUDA;    
    util::linear_kernel(iterative_intersect_kernel, 0, stream, numRays, params, N_ITERS);
    TRACE_CUDA;
    sampler(stream, N_ITERS * numRays, params.inference_coords, params.inference_values);
    TRACE_CUDA;
    util::linear_kernel(iterative_compose_kernel, 0, stream, numRays, params, N_ITERS);
    TRACE_CUDA;
  }
}

void
do_shadowmap_iterative(cudaStream_t stream, const ShadowMapData& params, const IterativeSampler& sampler, uint32_t numRays)
{
  iterative_shadowmap_loop(stream, params, sampler, numRays);
}

void default_shadowmap_sampler(
  const VNRDeviceVolume& self, 
  const LaunchParams& params, 
  vec3f* __restrict__ d_coords, 
  float* __restrict__ d_values, 
  uint32_t count, 
  cudaStream_t stream) 
{
  if (self.volume.data == 0) {
    warn_once("[Error] scalar field texture is zero, indicating that the scalar field "
              "is likely not loaded, thus black screen is expected.");
  }

  const auto marching_step = 1.f * self.step;
  const auto wto = self.transform.inverse();
  const auto dir = xfmVector(wto, normalize(params.l_distant.direction));
  util::parallel_for_gpu(stream, count, [=] __device__ (uint32_t i) {
    const auto org = d_coords[i];
    auto tnear = 0.f;
    auto tfar = float_large;
    float alpha = 0;
    if (intersectVolume(tnear, tfar, org, dir, self)) {
      vec2f t = vec2f(tnear, min(tnear + marching_step, tfar));
      do {
        const auto c = org + lerp(/*jitter*/0.5f, t.x, t.y) * dir;
        const auto sampleValue = sampleVolume(self.volume, c);
        vec3f sampleColor;
        float sampleAlpha;
        sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
        opacityCorrection(self, t.y - t.x, sampleAlpha);
        alpha += (1.f - alpha) * sampleAlpha;
        t.x = t.y;
        t.y = min(t.x + marching_step, tfar);
      } while (t.y > t.x && alpha < nearly_one);
    }
    d_values[i] = 1.f - alpha;
  });
}

}
