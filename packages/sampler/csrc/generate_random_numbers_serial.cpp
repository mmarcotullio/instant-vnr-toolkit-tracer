#include "generate_random_numbers.h"

#include "config.h"

#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace pysampler {

static std::mt19937 random_generator() {
  // seed: https://simplecxx.github.io/2018/11/03/seed-mt19937.html
	std::random_device rd;
	std::seed_seq sd{rd(), rd(), rd(), rd()}; // 4 values are enough. 8 max.
	return std::mt19937(sd); // advanced seeding   
}

static std::mt19937 rng = random_generator();

static uint32_t random_uint32(const uint32_t& min, const uint32_t& max) {
  ASSERT_THROW(min < max && "calling 'random_uint32' with invalid range.");
  std::uniform_int_distribution<uint32_t> distribution(min, max);
  return distribution(rng);
}

static uint32_t random_uint32(const uint32_t& count) {
  ASSERT_THROW(count != 0 && "calling 'random_uint32' with zero range.");
  return random_uint32(0, count - 1);
}

static uint64_t random_uint64(const uint64_t& min, const uint64_t& max) {
  ASSERT_THROW(min < max && "calling 'uint64_random' with invalid range.");
  std::uniform_int_distribution<uint64_t> distribution(min, max);
  return distribution(rng);
}

static uint64_t random_uint64(const uint64_t& count) {
  ASSERT_THROW(count != 0 && "calling 'uint64_random' with zero range.");
  return random_uint64(0, count - 1);
}

static float random_uniform(const float& min, const float& max) {
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(rng);
}

void generate_random_uniforms_serial(size_t batch, float* h_buffer, Device device) {
  if (device != Device::CPU) {
    throw std::runtime_error("generate_random_uniforms: device (" + std::to_string((int)device) + ") not supported.");
  }
  for (size_t i = 0; i < batch; ++i) h_buffer[i] = random_uniform(0.f, 1.f);
}

// ------------------------------------------------
//
// ------------------------------------------------
void generate_random_uniforms(float* h_buffer, size_t batch, Device device) {
#if defined(ENABLE_CUDA)
  return generate_random_uniforms_cuda(batch, h_buffer, device);
#elif defined(ENABLE_TORCH)
  return generate_random_uniforms_torch(batch, h_buffer, device);
#else
  return generate_random_uniforms_serial(batch, h_buffer, device);
#endif
}

}
