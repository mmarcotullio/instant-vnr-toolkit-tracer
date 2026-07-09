#define PYSAMPLER_SAMPLER_CPP_IMPL

#include "sampler_file_regular.h"

#include <algorithm>
#include <iostream>
#include <random>

namespace pysampler {

namespace {

// ------------------------------------------------------------------
// VirtualMemorySampler
//
// CPU-backed regular-grid sampler that relies on OS virtual memory to
// page voxel data in on demand: the volume is mmap'd once at
// construction and per-voxel reads are direct memcpy from the mapped
// region.  This matches InstantVNR's `VirtualMemorySampler` semantics
// and is the right choice when the volume may be larger than RAM but
// the underlying storage is reliable (local SSD / NVMe).
//
// Note on robustness: mmap page-fault failures raise SIGBUS, which is
// fatal.  On unreliable / slow storage (network mounts, slow HDDs that
// time out, etc.) prefer the `out_of_core` sampler instead — its
// explicit heap-resident block cache is populated via `pread()` and
// reports I/O errors as exceptions.
//
// Trilinear interpolation is performed after per-voxel normalization,
// matching InstantVNR's behavior exactly.
//
// Buffers passed to sample() / decode1d() are HOST pointers (matching
// the OpenVKL backend's calling convention).
// ------------------------------------------------------------------

class VirtualMemorySampler : public SamplerBase {
public:
  explicit VirtualMemorySampler(VolumeFileStructured desc);

  pysampler::Device device()    const override { return pysampler::Device::CPU; }
  pysampler::box3f  bounds()    const override { return m_bounds; }
  int               n_channels() const override { return 1; }

  void sample(void* h_coords, void* h_values,
              const vec3f& lower, const vec3f& upper, size_t count) override;
  void decode1d(void* h_coords, void* h_values,
                const vec3f& lower, const vec3f& upper, size_t count) override;
  void decode3d(void* h_coords, void* h_values,
                const vec3f& lower, const vec3f& upper, const vec3i& dims) override;

private:
  // Sample a single normalized point in [0,1]^3 in volume-normalized space.
  // The accessor reads one voxel and returns a normalized value clamped to [0,1].
  float sample_point(const vec3f& p_norm) const;

  vec3i   m_dims{};
  dtype   m_type{};
  range1f m_range{};
  size_t  m_offset      = 0;
  int     m_elem_size   = 0;
  float   m_value_scale = 0.f;
  box3f   m_bounds{};

  RegularGridFile m_file;
  std::mt19937    m_rng;
};

VirtualMemorySampler::VirtualMemorySampler(VolumeFileStructured desc)
  : m_dims(desc.dims)
  , m_type(desc.type)
  , m_range(resolve_required_range(desc.ranges, "virtual_memory"))
  , m_offset(desc.offset)
  , m_elem_size(sizeof_type(desc.type))
  , m_value_scale(1.f / (m_range.upper - m_range.lower))
  , m_bounds(vec3f(0.f), vec3f((float)m_dims.x, (float)m_dims.y, (float)m_dims.z) * desc.spacing)
  , m_file(desc.filename)
  , m_rng(1337u)
{
  // Sanity check: file must contain enough bytes for the requested volume.
  const size_t expected = (size_t)m_dims.x * (size_t)m_dims.y * (size_t)m_dims.z * (size_t)m_elem_size;
  if (m_file.file_size() < m_offset + expected) {
    throw std::runtime_error(
      "virtual_memory sampler: file too small for given dims/dtype/offset"
    );
  }

  std::cout
    << "[pysampler/virtual_memory] sampler ready\n"
    << "  filename     : " << desc.filename << "\n"
    << "  dims         : " << m_dims.x << " x " << m_dims.y << " x " << m_dims.z << "\n"
    << "  dtype        : " << dtype_name(m_type)
    << "  (elem_size = " << m_elem_size << " B)\n"
    << "  range        : [" << m_range.lower << ", " << m_range.upper << "]"
    << "  (1/span = " << m_value_scale << ")\n"
    << "  spacing      : " << desc.spacing.x << " x " << desc.spacing.y << " x " << desc.spacing.z << "\n"
    << "  file offset  : " << m_offset << " B\n"
    << "  file size    : " << pretty_bytes(m_file.file_size()) << "\n"
    << "  volume bytes : " << pretty_bytes(expected) << "\n"
    << "  access mode  : mmap + lazy paging (MADV_WILLNEED hint applied)\n"
    << "  warning      : a transient I/O failure on the underlying storage\n"
    << "                 will raise SIGBUS during sampling.  For unreliable\n"
    << "                 storage (slow / network), prefer the out_of_core\n"
    << "                 sampler — its block cache uses pread() and surfaces\n"
    << "                 I/O errors as exceptions instead of SIGBUS.\n"
    << std::flush;
}

float VirtualMemorySampler::sample_point(const vec3f& p_norm) const {
  // Read a single voxel; normalize before contributing to the trilinear sum.
  // This matches the InstantVNR `VirtualMemorySampler` behavior:
  // `clamp((v - vmin) / (vmax - vmin), 0, 1)` per voxel, then trilinear blend.
  const char* base       = m_file.data() + m_offset;
  const range1f range    = m_range;
  const float   vscale   = m_value_scale;
  const dtype   type     = m_type;
  const vec3i   dims     = m_dims;

  const auto accessor = [base, range, vscale, type, dims](const vec3i& ip) -> float {
    const size_t vidx = flatten_grid_index(ip, dims);
    const float  v    = read_scalar_as_float(base, vidx, type);
    const float  n    = (v - range.lower) * vscale;
    return std::min(std::max(n, 0.f), 1.f);
  };

  return trilinear_normalized(p_norm, dims, accessor);
}

void VirtualMemorySampler::decode1d(void* h_coords, void* h_values,
                                    const vec3f& lower, const vec3f& upper, size_t count)
{
  const vec3f* coords = (const vec3f*)h_coords;
  float*       values = (float*)h_values;

  const vec3f span = upper - lower;
  for (size_t i = 0; i < count; ++i) {
    const vec3f c = coords[i];
    // Map caller coords [0,1]^3 onto [lower, upper] in normalized volume space.
    const vec3f p = vec3f(c.x * span.x + lower.x,
                          c.y * span.y + lower.y,
                          c.z * span.z + lower.z);
    values[i] = sample_point(p);
  }
}

void VirtualMemorySampler::sample(void* h_coords, void* h_values,
                                  const vec3f& lower, const vec3f& upper, size_t count)
{
  vec3f* coords = (vec3f*)h_coords;
  float* values = (float*)h_values;

  // Generate uniform random positions in [lower, upper] in volume-normalized
  // [0,1]^3 space, then sample directly at those positions.
  const vec3f span = upper - lower;
  std::uniform_real_distribution<float> uniform(0.f, 1.f);
  for (size_t i = 0; i < count; ++i) {
    const vec3f u(uniform(m_rng), uniform(m_rng), uniform(m_rng));
    const vec3f p(u.x * span.x + lower.x,
                  u.y * span.y + lower.y,
                  u.z * span.z + lower.z);
    coords[i] = p;
    values[i] = sample_point(p);
  }
}

void VirtualMemorySampler::decode3d(void* /*h_coords*/, void* /*h_values*/,
                                    const vec3f& /*lower*/, const vec3f& /*upper*/,
                                    const vec3i& /*dims*/)
{
  throw std::runtime_error("virtual_memory sampler: decode3d is not implemented yet");
}

} // namespace

sampler_t create_virtual_memory_sampler(VolumeFileStructured desc) {
  validate_file_backed_desc(desc, "virtual_memory");
  return std::make_shared<VirtualMemorySampler>(std::move(desc));
}

} // namespace pysampler
