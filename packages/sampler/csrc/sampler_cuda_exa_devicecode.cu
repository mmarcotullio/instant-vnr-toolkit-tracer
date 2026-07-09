// ======================================================================== //
// Copyright 2023-2024 Qi Wu                                                //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
// ======================================================================== //
// Copyright 2022-2023 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <stdint.h>

#include <cuda_runtime.h>

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/box.h>
#include <owl/common/math/random.h>

#include "sampler/ExaBrickSampler.h"
#include "sampler/ExaStitchSampler.h"
// #include "sampler/AMRCellSampler.h"
// #include "sampler/BigMeshSampler.h"
// #include "sampler/QuickClustersSampler.h"

#include "common.h"

namespace pysampler {
  
struct LaunchParams {
  union {
    exa::ExaBrickSampler::LP  ebs;
    exa::ExaStitchSampler::LP ess;
    // exa::AMRCellSampler::LP   acs;
    // exa::BigMeshSampler::LP   bms;
    // exa::QuickClustersSampler::LP qcs;
  } sampler;
  vec3f *coords;
  float *values;
  box3f worldSpaceBounds;
  interval<float> valueRange;
};

}

#ifndef SAMPLER_CUDA_STITCHER_DEVICECODE_IMPL

#include <float.h>

using owl::vec2f;
using owl::vec2i;
using owl::vec3f;
using owl::vec3i;
using owl::vec4f;
using owl::vec4i;

#define INVALID_VALUE 0.f

namespace pysampler {

  extern "C" __constant__ LaunchParams optixLaunchParams;

  // ------------------------------------------------------------------
  // RAYGEN
  // ------------------------------------------------------------------

  template <typename Sampler>
  inline void __device__ raygen_sample(const Sampler &sampler) {
    auto& lp = optixLaunchParams;  
    const auto idx = owl::getLaunchIndex().x;
    const auto pos = lp.coords[idx] * lp.worldSpaceBounds.span() + lp.worldSpaceBounds.lower;
    const auto s = exa::sample(sampler, {}, pos);
    lp.values[idx] = (s.primID < 0 || isnan(s.value)) 
      ? INVALID_VALUE : ((s.value - lp.valueRange.lower) / lp.valueRange.span());
  }

  // OPTIX_RAYGEN_PROGRAM(renderFrame_AMRCellSampler)()
  // {
  //   auto& lp = optixLaunchParams;
  //   renderFrame_SelectIntegrator<Default>(lp.majorantGrid,lp.sampler.acs);
  // }

  // OPTIX_RAYGEN_PROGRAM(renderFrame_BigMeshSampler)()
  // {
  //   auto& lp = optixLaunchParams;
  //   renderFrame_SelectIntegrator<Default>(lp.majorantGrid,lp.sampler.bms);
  // }

  // OPTIX_RAYGEN_PROGRAM(renderFrame_QuickClustersSampler)()
  // {
  //   auto& lp = optixLaunchParams;
  //   renderFrame_SelectIntegrator<Default>(lp.majorantGrid,lp.sampler.qcs);
  // }

  OPTIX_RAYGEN_PROGRAM(renderFrame_ExaBrickSampler)() { 
    raygen_sample(optixLaunchParams.sampler.ebs); 
  }

  OPTIX_RAYGEN_PROGRAM(renderFrame_ExaStitchSampler)() { 
    raygen_sample(optixLaunchParams.sampler.ess); 
  }
}

#endif // SAMPLER_CUDA_STITCHER_DEVICECODE_IMPL
