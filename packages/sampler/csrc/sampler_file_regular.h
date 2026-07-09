#pragma once

#ifndef PYSAMPLER_SAMPLER_CPP_IMPL
#error "This header is not supposed to be included directly"
#endif

#include "config.h"
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pysampler {

// ------------------------------------------------------------------
// Regular grid index helper.  Volumes are stored in row-major order:
//   byte_offset(x, y, z) = (x + y*Dx + z*Dx*Dy) * sizeof_dtype
// ------------------------------------------------------------------

inline size_t flatten_grid_index(vec3i idx, vec3i dims) {
  const size_t stride_y = (size_t)dims.x;
  const size_t stride_z = (size_t)dims.y * (size_t)dims.x;
  return (size_t)idx.x + (size_t)idx.y * stride_y + (size_t)idx.z * stride_z;
}

// ------------------------------------------------------------------
// Trilinear interpolation matching CUDA tex3D normalized addressing.
//
//   p_norm in [0, 1]^3 (normalized volume coords)
//   tex3D index formula: f_axis = p_axis * dims_axis - 0.5, clamped to [0, dims-1]
//
// `accessor(vec3i)` returns a float voxel value (raw or normalized).
// This helper is normalization-agnostic: callers may normalize before or after
// interpolation depending on the desired semantics.
// ------------------------------------------------------------------

template<typename Accessor>
inline float trilinear_normalized(const vec3f& p_norm, vec3i dims, const Accessor& accessor) {
  const float fx = std::min(std::max(p_norm.x * (float)dims.x - 0.5f, 0.0f), (float)dims.x - 1.0f);
  const float fy = std::min(std::max(p_norm.y * (float)dims.y - 0.5f, 0.0f), (float)dims.y - 1.0f);
  const float fz = std::min(std::max(p_norm.z * (float)dims.z - 0.5f, 0.0f), (float)dims.z - 1.0f);

  const int ix0 = (int)std::floor(fx);
  const int iy0 = (int)std::floor(fy);
  const int iz0 = (int)std::floor(fz);
  const int ix1 = std::min(ix0 + 1, dims.x - 1);
  const int iy1 = std::min(iy0 + 1, dims.y - 1);
  const int iz1 = std::min(iz0 + 1, dims.z - 1);

  const float tx = fx - (float)ix0;
  const float ty = fy - (float)iy0;
  const float tz = fz - (float)iz0;

  const float c000 = accessor(vec3i(ix0, iy0, iz0));
  const float c100 = accessor(vec3i(ix1, iy0, iz0));
  const float c010 = accessor(vec3i(ix0, iy1, iz0));
  const float c110 = accessor(vec3i(ix1, iy1, iz0));
  const float c001 = accessor(vec3i(ix0, iy0, iz1));
  const float c101 = accessor(vec3i(ix1, iy0, iz1));
  const float c011 = accessor(vec3i(ix0, iy1, iz1));
  const float c111 = accessor(vec3i(ix1, iy1, iz1));

  const float c00 = c000 * (1.f - tx) + c100 * tx;
  const float c10 = c010 * (1.f - tx) + c110 * tx;
  const float c01 = c001 * (1.f - tx) + c101 * tx;
  const float c11 = c011 * (1.f - tx) + c111 * tx;
  const float c0  = c00  * (1.f - ty) + c10  * ty;
  const float c1  = c01  * (1.f - ty) + c11  * ty;
  return c0 * (1.f - tz) + c1 * tz;
}

// ------------------------------------------------------------------
// Read-only view of a regular grid file.
//
// Exposes two complementary access modes against the same open file:
//   - `data()`        : pointer into a memory-mapped region; per-voxel
//                       reads are essentially free once the OS page cache
//                       is warm, but a page-fault I/O failure raises
//                       SIGBUS (fatal).  Used by the `virtual_memory`
//                       sampler.
//   - `pread_into(...)`: explicit blocking pread() / ReadFile() that
//                       returns I/O errors as thrown exceptions, never
//                       SIGBUS.  Used by the `out_of_core` sampler to
//                       populate its heap-resident block cache.
// ------------------------------------------------------------------

class RegularGridFile {
public:
  explicit RegularGridFile(const std::string& filename);
  ~RegularGridFile();

  RegularGridFile(const RegularGridFile&)            = delete;
  RegularGridFile& operator=(const RegularGridFile&) = delete;
  RegularGridFile(RegularGridFile&&) noexcept;
  RegularGridFile& operator=(RegularGridFile&&) noexcept;

  size_t      file_size() const { return m_size; }
  const char* data()      const { return m_data; }

  // Direct-I/O read into a caller-provided buffer.  Throws on I/O error.
  void pread_into(size_t byte_offset, void* dst, size_t nbytes) const;

private:
#if defined(_WIN32)
  void* m_handle  = nullptr; // HANDLE to the file
  void* m_mapping = nullptr; // HANDLE to the file mapping
#else
  int m_fd = -1;
#endif
  const char* m_data = nullptr;  // pointer into the mmap'd region
  size_t      m_size = 0;
};

// ------------------------------------------------------------------
// Read a single scalar voxel and return it as float (raw, unnormalized).
// `data` must point to the start of the volume payload (already shifted by
// the user-provided file offset).
// ------------------------------------------------------------------

inline float read_scalar_as_float(const char* data, size_t voxel_index, dtype type) {
  const size_t es = (size_t)sizeof_type(type);
  const char* p = data + voxel_index * es;
  switch (type) {
  case UINT8:  { uint8_t  v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  case INT8:   { int8_t   v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  case UINT16: { uint16_t v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  case INT16:  { int16_t  v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  case UINT32: { uint32_t v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  case INT32:  { int32_t  v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  case FLOAT:  { float    v; std::memcpy(&v, p, sizeof(v)); return v; }
  case DOUBLE: { double   v; std::memcpy(&v, p, sizeof(v)); return float(v); }
  default: throw std::runtime_error("file-backed sampler: unsupported data type");
  }
}

// ------------------------------------------------------------------
// Logging helpers used by the file-backed samplers' construction-time
// info messages.
// ------------------------------------------------------------------

inline const char* dtype_name(dtype type) {
  switch (type) {
  case UINT8:  return "uint8";
  case INT8:   return "int8";
  case UINT16: return "uint16";
  case INT16:  return "int16";
  case UINT32: return "uint32";
  case INT32:  return "int32";
  case FLOAT:  return "float32";
  case DOUBLE: return "float64";
  default: return "unknown";
  }
}

inline std::string pretty_bytes(size_t s) {
  char buf[64];
  if (s >= (1ULL << 40))      std::snprintf(buf, sizeof(buf), "%.2f TiB", (double)s / (1ULL << 40));
  else if (s >= (1ULL << 30)) std::snprintf(buf, sizeof(buf), "%.2f GiB", (double)s / (1ULL << 30));
  else if (s >= (1ULL << 20)) std::snprintf(buf, sizeof(buf), "%.2f MiB", (double)s / (1ULL << 20));
  else if (s >= (1ULL << 10)) std::snprintf(buf, sizeof(buf), "%.2f KiB", (double)s / (1ULL << 10));
  else                        std::snprintf(buf, sizeof(buf), "%zu B",    s);
  return std::string(buf);
}

// ------------------------------------------------------------------
// Validate that a structured-volume descriptor is supported by the
// file-backed (CPU) regular-grid samplers.
// ------------------------------------------------------------------

inline void validate_file_backed_desc(const VolumeFileStructured& desc, const char* mode) {
  if (desc.n_channels != 1) {
    throw std::runtime_error(
      std::string(mode) + " sampler currently supports n_channels=1 only (got " +
      std::to_string(desc.n_channels) + ")"
    );
  }
  if (desc.is_big_endian) {
    throw std::runtime_error(
      std::string(mode) + " sampler does not support big-endian input"
    );
  }
  if (desc.dims.x <= 0 || desc.dims.y <= 0 || desc.dims.z <= 0) {
    throw std::runtime_error(
      std::string(mode) + " sampler requires positive dims, got [" +
      std::to_string(desc.dims.x) + ", " +
      std::to_string(desc.dims.y) + ", " +
      std::to_string(desc.dims.z) + "]"
    );
  }
  if (!desc.filename || desc.filename[0] == '\0') {
    throw std::runtime_error(
      std::string(mode) + " sampler requires a filename"
    );
  }
}

inline range1f resolve_required_range(const std::vector<range1f>& ranges, const char* mode) {
  if (ranges.empty()) {
    throw std::runtime_error(
      std::string(mode) +
      " sampler requires an explicit value range (pass `range=[vmin, vmax]`)"
    );
  }
  const range1f& r = ranges.front();
  if (r.is_empty() || !(r.upper > r.lower)) {
    throw std::runtime_error(
      std::string(mode) +
      " sampler requires a non-empty value range with vmax > vmin"
    );
  }
  return r;
}

// ------------------------------------------------------------------
// Factory functions registered by sampler.cpp dispatch.
// ------------------------------------------------------------------

sampler_t create_virtual_memory_sampler(VolumeFileStructured desc);
sampler_t create_out_of_core_sampler(VolumeFileStructured desc);

} // namespace pysampler
