//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#include "renderer.h"
#include <core/volumes/volumes.h>
#include <cuda/cuda_buffer.h>

#include <iostream>

#ifdef ENABLE_LOGGING
#define log() std::cout
#else
static std::ostream null_output_stream(0);
#define log() null_output_stream
#endif

namespace ovr::hyperinr {

// --------------------------------------------------------------------------------------------------------
//
// --------------------------------------------------------------------------------------------------------

const uint32_t MAX_INFERENCE_SIZE = 1 << 24;

void network_inference_kernel(vnr::NeuralVolume* network, vec3f* __restrict__ d_coords, float* __restrict__ d_values, uint32_t count, cudaStream_t stream) {
  for (uint32_t i = 0; i < count; i += MAX_INFERENCE_SIZE) {
    // 'sample_coord' and 'sample_value' are allocated with padding to the next multiple of 256
    const uint32_t batch = util::next_multiple(std::min(count - i, MAX_INFERENCE_SIZE), 256U);
    network->inference(batch, (float*)(d_coords+i), d_values+i, stream);
  }
}

void groundtruth_inference_kernel(const VNRDeviceVolume& volume, vec3f* __restrict__ d_coords, float* __restrict__ d_values, uint32_t count, cudaStream_t stream) {
  util::parallel_for_gpu(stream, count, [=] __device__ (uint32_t i) {
    const auto p = d_coords[i];
    d_values[i] = tex3D<float>(volume.volume.data, p.x, p.y, p.z);
  });
}

void default_shadowmap_sampler(const VNRDeviceVolume& self, const LaunchParams& params, vec3f* d_coords, float* d_values, uint32_t count, cudaStream_t stream);

// --------------------------------------------------------------------------------------------------------
//
// --------------------------------------------------------------------------------------------------------

void RenderObject::init(
  affine3f transform, 
  ValueType type, vec3i dims,
  vec3i macrocell_dims, 
  vec3f macrocell_spacings, 
  vec2f* macrocell_d_value_range, 
  float* macrocell_d_max_opacity
) {
  self.transform = transform;
  self.volume.dims = dims;
  self.volume.type = type;
  self.macrocell_value_range = macrocell_d_value_range;
  self.macrocell_max_opacity = macrocell_d_max_opacity;
  self.macrocell_dims = macrocell_dims;
  self.macrocell_spacings = macrocell_spacings;
  self.macrocell_spacings_rcp = 1.f / macrocell_spacings;

  // properly initialize light angle
  const float radius = 1000.0;
  const float init_phi   = 0.0f * 2.f * M_PI;
  const float init_theta = 0.0f * M_PI;
  params.l_distant.direction = vec3f( 
    radius * sin(init_phi) * cos(init_theta),
    radius * sin(init_phi) * sin(init_theta),
    radius * cos(init_phi)
  );

  // shadowmap_net = vnrCreateNetwork(
  //   vnrCreateJsonBinary("pretrained-model.json") // load pre-trained model
  //   // "empty-model.json" // load empty model
  // );
}

void RenderObject::update(int rendering_mode, 
  const DeviceTransferFunction& tfn,
  float sampling_rate,
  float density_scale,
  vec3f clip_lower, 
  vec3f clip_upper,
  const vnr::Camera& camera,
  const vec2i& framesize
) {
  if (this->rendering_mode != rendering_mode) {
    this->rendering_mode = rendering_mode;
    rm.clear(stream);
    pt.clear(stream);
    shadowmap.clear(stream);
  }

  self.step = 1.f / sampling_rate;
  self.step_rcp = sampling_rate;
  self.grad_step = vec3f(1.f / vec3f(self.volume.dims));
  self.density_scale = density_scale;
  self.tfn = tfn;
  self.tfn.range_rcp_norm = 1.f / self.tfn.value_range.span();
  self.bbox.lower = clip_lower;
  self.bbox.upper = clip_upper;

  // resize our cuda frame buffer
  params.frame.size = framesize;
  framebuffer_accumulation.resize(framesize.long_product() * sizeof(vec4f), stream);
  params.accumulation = (vec4f*)framebuffer_accumulation.d_pointer();

  /* camera ... */
  /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
  const float fovy = camera.fovy;
  const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * (float)M_PI / 180.f);
  const float aspect = params.frame.size.x / float(params.frame.size.y);
  params.camera.position = camera.from;
  params.camera.direction = normalize(camera.at - camera.from);
  params.camera.horizontal = t * aspect * normalize(cross(params.camera.direction, camera.up));
  params.camera.vertical = cross(params.camera.horizontal, params.camera.direction) / aspect;

  // flag to reset frame data
  params.frame_index = 0;
}

void RenderObject::render(vec4f* fb, IterativeSampler sampler, bool iterative) {
  if (params.frame.size.x == 0 || params.frame.size.y == 0) return;
  params.frame_index++;
  params.frame.rgba = fb;

  switch (rendering_mode) {
  case VNR_RAYMARCHING:          rm.render(stream, params, self, sampler, MethodRayMarching::NO_SHADING, iterative); break;
  case VNR_RAYMARCHING_GRADIENT: rm.render(stream, params, self, sampler, MethodRayMarching::GRADIENT_SHADING, iterative); break;
  case VNR_RAYMARCHING_SSH:      rm.render(stream, params, self, sampler, MethodRayMarching::SINGLE_SHADE_HEURISTIC, iterative); break;
  case VNR_RAYMARCHING_SHADOW:   shadowmap.render(stream, params, self, sampler, MethodShadowMap::SHADING); break;
  case VNR_PATHTRACING:          pt.render(stream, params, self, sampler, iterative); break;
  default: throw std::runtime_error("unknown rendering mode " + std::to_string(rendering_mode));
  }
}

void RenderObject::render(vec4f* fb, vnr::NeuralVolume* network, cudaTextureObject_t grid) {
  self.volume.data = grid;
  if (!network && self.volume.data == 0) {
    throw std::runtime_error("no volume data to render");
  }

  // Auto-inferred NN shadow: if shadow_field_net is provided, use shadow mapping automatically
  if (shadow_field_net) {
    if (scalar_field_net) {
      // Dual-NN: both scalar and shadow fields are explicit neural networks
      if (params.frame.size.x == 0 || params.frame.size.y == 0) return;
      params.frame_index++;
      params.frame.rgba = fb;
      const uint32_t numPixels = (uint32_t)params.frame.size.long_product();
      const uint32_t numPixelsPadded = util::next_multiple(numPixels, 256U);
      const uint32_t nValues = numPixelsPadded * MethodShadowMap::SHADOW_N_ITERS;
      auto combined_sampler = [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
        const auto padded = util::next_multiple(count, 256U);
        vnrInfer(scalar_field_net, (vnrDevicePtr)d_coords, (vnrDevicePtr)d_values, padded, stream);
        vnrInfer(shadow_field_net, (vnrDevicePtr)d_coords, (vnrDevicePtr)(d_values + nValues), padded, stream);
      };
      shadowmap.render(stream, params, self, combined_sampler, MethodShadowMap::SHADING, /*dual_nn=*/true);
    }
    else if (network) {
      // Dual-NN using main neural volume network as scalar field
      if (params.frame.size.x == 0 || params.frame.size.y == 0) return;
      params.frame_index++;
      params.frame.rgba = fb;
      const uint32_t numPixels = (uint32_t)params.frame.size.long_product();
      const uint32_t numPixelsPadded = util::next_multiple(numPixels, 256U);
      const uint32_t nValues = numPixelsPadded * MethodShadowMap::SHADOW_N_ITERS;
      auto combined_sampler = [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
        const auto padded = util::next_multiple(count, 256U);
        network_inference_kernel(network, d_coords, d_values, padded, stream);
        vnrInfer(shadow_field_net, (vnrDevicePtr)d_coords, (vnrDevicePtr)(d_values + nValues), padded, stream);
      };
      shadowmap.render(stream, params, self, combined_sampler, MethodShadowMap::SHADING, /*dual_nn=*/true);
    }
    else {
      // Single-NN: shadow field from network, scalar field from GT texture
      render(fb, [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
        vnrInfer(shadow_field_net, (vnrDevicePtr)d_coords, (vnrDevicePtr)d_values, util::next_multiple(count, 256U), stream);
      }, true);
    }
    return;
  }

  // GT shadow mode (mode 3): brute-force transmittance from volume texture
  if (rendering_mode == VNR_RAYMARCHING_SHADOW) {
    if (self.volume.data) {
      render(fb, [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
        default_shadowmap_sampler(self, params, d_coords, d_values, count, stream);
      }, true);
    }
    else {
      warn_once("[Warning] VNR_RAYMARCHING_SHADOW requires a GT volume texture; "
                "not available for neural volumes. Skipping frame.");
    }
    return;
  }

  // PyTorch override: if env var is set, use the stub sampler for all modes
  static const char* VNR_PYTORCH_SCRIPT = std::getenv("VNR_PYTORCH_SCRIPT");
  if (VNR_PYTORCH_SCRIPT) {
    return render(fb, pytorch_inference_kernel, /*iterative=*/true);
  }

  // Standard path: select sampler based on available data
  if (self.volume.data) {
    render(fb, [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
      groundtruth_inference_kernel(self, d_coords, d_values, count, stream);
    }, /*iterative=*/false);
  }
  else if (network) {
    render(fb, [=] (cudaStream_t stream, uint32_t count, vec3f* __restrict__ d_coords, float* __restrict__ d_values) {
      network_inference_kernel(network, d_coords, d_values, count, stream);
    }, /*iterative=*/true);
  }
}

} // namespace vnr
