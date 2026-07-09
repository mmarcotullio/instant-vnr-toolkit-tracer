#include "generate_random_numbers.h"

#include "config.h"
#include <cuda/cuda_utils.h>

#include <curand.h>

#include <iostream>

#define CURAND_CALL(x) do { \
  if ((x)!=CURAND_STATUS_SUCCESS) { printf("CURAND Error at %s:%d (%d)\n",__FILE__,__LINE__,x); } \
} while(0)

namespace pysampler {
    
curandGenerator_t curand_create()
{
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  return gen;
}

static curandGenerator_t generator = curand_create();

void generate_random_uniforms_cuda(size_t batch, float* buffer, Device device) {
  if (device == Device::CPU) {
    float* dptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dptr, batch * sizeof(float)));
    CURAND_CALL(curandGenerateUniform(generator, dptr, batch));
    CUDA_CHECK(cudaMemcpy(buffer, dptr, batch * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dptr));
  }
  else if (device == Device::CUDA) {
    CURAND_CALL(curandGenerateUniform(generator, buffer, batch));
  }
  else {
    throw std::runtime_error("Unsupported device: " + std::to_string((int)device));
  }
}

}
