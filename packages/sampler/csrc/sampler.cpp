#include "config.h"
#include "sampler.h"
#define PYSAMPLER_SAMPLER_CPP_IMPL

#ifdef ENABLE_CUDA
#include "sampler_cuda.h"
#endif

#ifdef ENABLE_OPENVKL
#include "sampler_openvkl.h"
#endif

#include "sampler_file_regular.h"

#include <string>
#include <fstream>

#ifdef ENABLE_CUDA
#define IFF_CUDA_DEVICE { if (std::string(device) == "cuda") { return std::make_shared<CUDASampler>(desc); } }
#else
#define IFF_CUDA_DEVICE ((void)0)
#endif

#ifdef ENABLE_OPENVKL
#define IFF_OPENVKL_DEVICE { if (std::string(device) == "openvkl") { return std::make_shared<OpenVKLSampler>(desc); } }
#else
#define IFF_OPENVKL_DEVICE ((void)0)
#endif

#define IFF_VIRTUAL_MEMORY_DEVICE { if (std::string(device) == "virtual_memory") { return pysampler::create_virtual_memory_sampler(desc); } }
#define IFF_OUT_OF_CORE_DEVICE    { if (std::string(device) == "out_of_core")    { return pysampler::create_out_of_core_sampler(desc); } }

#define THROW_UNKNOWN_DEVICE(msg) throw std::runtime_error("Error: unknown device \"" + std::string(device) + "\" for " + msg + ".")

pysampler::sampler_t pysampler::create_sampler(const char* device, VolumeFileStructured desc) {
  IFF_CUDA_DEVICE;
  IFF_OPENVKL_DEVICE;
  IFF_VIRTUAL_MEMORY_DEVICE;
  IFF_OUT_OF_CORE_DEVICE;
  THROW_UNKNOWN_DEVICE("regular grid file");
}

pysampler::sampler_t pysampler::create_sampler(const char* device, VolumeDataStructured desc) {
  IFF_CUDA_DEVICE;
  IFF_OPENVKL_DEVICE;
  THROW_UNKNOWN_DEVICE("regular grid data");
}

pysampler::sampler_t pysampler::create_sampler(const char* device, VolumeFileOpenVDB desc) {
  IFF_OPENVKL_DEVICE;
  THROW_UNKNOWN_DEVICE("OpenVDB file");
}

pysampler::sampler_t pysampler::create_sampler(const char* device, VolumeFileExaBrick desc) {
#ifdef ENABLE_WITCHER
  IFF_CUDA_DEVICE;
#endif
  THROW_UNKNOWN_DEVICE("ExaBrick file (build with ENABLE_WITCHER=ON)");
}

pysampler::sampler_t pysampler::create_sampler(const char* device, VolumeFileExaStitch desc) {
#ifdef ENABLE_WITCHER
  IFF_CUDA_DEVICE;
#endif
  THROW_UNKNOWN_DEVICE("ExaStitch file (build with ENABLE_WITCHER=ON)");
}

pysampler::sampler_t pysampler::create_sampler(const char* device, VolumeFileVTKm desc) {
  IFF_OPENVKL_DEVICE;
  THROW_UNKNOWN_DEVICE("VTKm files");
}

namespace pysampler {

namespace {
template<size_t Size>
inline void swap_bytes(void* data) { char* p = reinterpret_cast<char*>(data); char* q = p + Size - 1; while (p < q) std::swap(*(p++), *(q--)); }
template<> inline void swap_bytes<1>(void*) {}
template<> inline void swap_bytes<2>(void* data) { char* p = reinterpret_cast<char*>(data); std::swap(p[0], p[1]); }
template<> inline void swap_bytes<4>(void* data) { char* p = reinterpret_cast<char*>(data); std::swap(p[0], p[3]); std::swap(p[1], p[2]); }
template<> inline void swap_bytes<8>(void* data) { char* p = reinterpret_cast<char*>(data); std::swap(p[0], p[7]); std::swap(p[1], p[6]); std::swap(p[2], p[5]); std::swap(p[3], p[4]); }
template<typename T>
inline void swap_bytes(T* data) { swap_bytes<sizeof(T)>(reinterpret_cast<void*>(data)); }
inline void reverse_byte_order(char* data, size_t elem_count, size_t elem_size) {
  switch (elem_size) {
  case 1: break;
  case 2: for (size_t i = 0; i < elem_count; ++i) { swap_bytes<2>(&data[i * elem_size]); } break;
  case 4: for (size_t i = 0; i < elem_count; ++i) { swap_bytes<4>(&data[i * elem_size]); } break;
  case 8: for (size_t i = 0; i < elem_count; ++i) { swap_bytes<8>(&data[i * elem_size]); } break;
  default: throw std::runtime_error("unsupported element size"); }
}
}

std::shared_ptr<char[]>
load_regular_grid(const char* _filename, vec3i dims, int n_channels, dtype type, size_t offset, bool is_big_endian) {
  const std::string filename = _filename;

  // data geometry
  assert(dims.x > 0 && dims.y > 0 && dims.z > 0);
  size_t elem_count = (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;
  size_t elem_size = sizeof_type(type);
  size_t data_size = elem_count * elem_size * n_channels;

  // load the data
  std::shared_ptr<char[]> data_buffer;

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (ifs.fail()) { // cannot open the file
    throw std::runtime_error("Cannot open the file: " + filename);
  }
  ifs.seekg(0, std::ios::end);

  size_t file_size = ifs.tellg();
  if (file_size < offset + data_size) { // file size does not match data size
    throw std::runtime_error("File size does not match data size");
  }
  ifs.seekg(offset, std::ios::beg);

  try {
    data_buffer.reset(new char[data_size]);
  }
  catch (std::bad_alloc&) { // memory allocation failed
    throw std::runtime_error("Cannot allocate memory for the data");
  }

  // read data
  ifs.read(data_buffer.get(), data_size);
  if (ifs.fail()) { // reading data failed
    throw std::runtime_error("Cannot read the file");
  }

  // reverse byte-order if necessary
  const bool reverse = (is_big_endian && elem_size > 1);
  if (reverse) {
    reverse_byte_order(data_buffer.get(), elem_count, elem_size);
  }
  ifs.close();

  return data_buffer;
}

}