#include "generate_random_numbers.h"

#include "config.h"

#include <torch/torch.h>

namespace pysampler {

torch::DeviceType same_device(Device device) {
  if (device == Device::CPU) {
    return torch::kCPU;
  }
#ifdef ENABLE_CUDA
  else if (device == Device::CUDA && torch::cuda::is_available()) {
    return torch::kCUDA;
  }
#endif
#ifdef ENABLE_XPU // TODO UNTESTED !!
  else if (device == Device::XPU /*&& torch::xpu::is_available()*/) {
    return torch::kXPU;
  }
#endif
#ifdef ENABLE_MPS
  else if (device == Device::MPS && torch::mps::is_available()) {
    return torch::kMPS;
  }
#endif
  else {
    throw std::runtime_error("get_device: device (" + std::to_string((int)device) + ") not supported.");
  }
}

torch::DeviceType best_device() {
  if (torch::cuda::is_available()) {
    return torch::kCUDA;
  }
#ifdef ENABLE_XPU // TODO UNTESTED !!
  else if (/*torch::xpu::is_available()*/ true) {
    return torch::kXPU;
  }
#endif
#ifdef ENABLE_MPS
  else if (torch::mps::is_available()) {
    return torch::kMPS;
  }
#endif
  else {
    return torch::kCPU;
  }
}

// create a torch tensor to generate random numbers
void generate_random_uniforms_torch(size_t batch, float* h_buffer, Device device) {
  // construct a tensor in place
  auto d = same_device(device);
  auto output = torch::from_blob(h_buffer, { (int64_t)batch }, 
    torch::TensorOptions().dtype(torch::kFloat32).device(d)
  );
#if 0
  // create a tensor on the best device
  auto tensor = torch::empty({ (int64_t)batch }, 
    torch::TensorOptions().dtype(torch::kFloat32).device(best_device())
  );
  tensor.uniform_(0.f, 1.f);
  output.copy_(tensor);
#else
  output.uniform_(0.f, 1.f);
#endif
}

}
