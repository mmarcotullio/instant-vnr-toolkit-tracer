#define PYSAMPLER_SAMPLER_CPP_IMPL

#include "sampler_file_regular.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

namespace pysampler {

namespace {

// ------------------------------------------------------------------
// OutOfCoreSampler
//
// CPU-backed regular-grid sampler that maintains a fixed-size cache of
// blocks loaded from a raw volume file via `pread()`.  Random samples
// are drawn from the currently cached blocks, mirroring the behavior of
// InstantVNR's `OutOfCoreSampler`.
//
// Why pread() rather than mmap-and-memcpy?
//   The OOC sampler's whole reason for existing is to cope with volumes
//   that may not be reliably page-faulted into memory (network shares,
//   slow HDDs, multi-TB datasets).  A page-fault failure during
//   sampling raises SIGBUS, which is fatal.  pread() instead returns
//   I/O errors via errno, which we convert to thrown exceptions, so a
//   bad block surfaces as a recoverable error rather than a crash.
//
// Compared to InstantVNR, this implementation:
//   - uses a synchronous block cache (no Linux AIO).  Block loads are
//     issued one at a time via pread().
//   - keeps the block-index-space layout (x = full width, y/z subdivided)
//     so behavior translates 1:1 if AIO is added later.
//   - supports `VNR_NUM_CONCURRENT_BLOCKS` / `VNR_NUM_BLOCKS` env vars
//     for parity with the upstream sampler.
//
// `decode1d` reads voxels directly via the memory map (same as the
// `virtual_memory` sampler), so tests can verify it agrees with that
// backend on caller-supplied coordinates.  If you are training on
// unreliable storage and want SIGBUS-free decoding too, sample only via
// `sample()` so all reads go through the pread-loaded block cache.
// ------------------------------------------------------------------

struct Block {
  vec3i origin{};            // origin voxel within the volume (no ghost)
  vec3i extent{};            // block extent (no ghost)
  vec3i ghost_origin{};      // origin including 1-voxel ghost padding
  vec3i ghost_extent{};      // ghost extent
  std::vector<char> data;    // raw voxel bytes for the (ghost-padded) block
};

class OutOfCoreSampler : public SamplerBase {
public:
  explicit OutOfCoreSampler(VolumeFileStructured desc);

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
  // Sample a single voxel from the file via mmap (no cache), normalized.
  // Used by decode1d() only; sample() reads exclusively from the cache.
  float file_sample_point(const vec3f& p_norm) const;
  // Sample a single voxel using a specific cached (heap-resident) block.
  float block_sample_point(const Block& blk, const vec3f& p_norm) const;
  // Load one block from the file into `dst.data` via pread().
  void  load_block(Block& dst, const vec3i& block_index);
  // Refresh a contiguous slice of the block cache with random block indices.
  void  refresh_blocks(size_t begin, size_t count);

  // Pick env-int with default fallback.
  static int env_int(const char* name, int default_value);

  // ---- volume metadata ---------------------------------------------
  vec3i   m_dims{};
  dtype   m_type{};
  range1f m_range{};
  size_t  m_offset      = 0;
  int     m_elem_size   = 0;
  float   m_value_scale = 0.f;
  box3f   m_bounds{};

  // ---- block geometry ----------------------------------------------
  vec3i  m_block_extent{};         // block dims (no ghost)
  vec3i  m_block_index_space{};    // # of blocks along each axis
  size_t m_num_block_indices = 0;  // total number of distinct block positions

  // ---- block cache -------------------------------------------------
  size_t              m_num_blocks            = 0;
  size_t              m_num_concurrent_blocks = 0;
  std::vector<Block>  m_cache;
  size_t              m_refresh_cursor = 0;

  RegularGridFile m_file;
  std::mt19937    m_rng;
};

int OutOfCoreSampler::env_int(const char* name, int default_value) {
  if (const char* env_p = std::getenv(name)) {
    try {
      return std::stoi(env_p);
    }
    catch (...) {
      // fall through and use default
    }
  }
  return default_value;
}

void OutOfCoreSampler::load_block(Block& dst, const vec3i& block_index) {
  // Voxel-space origin/extent of the block (without ghost padding).
  const vec3i o(block_index.x * m_block_extent.x,
                block_index.y * m_block_extent.y,
                block_index.z * m_block_extent.z);
  const vec3i e(std::min(m_block_extent.x, m_dims.x - o.x),
                std::min(m_block_extent.y, m_dims.y - o.y),
                std::min(m_block_extent.z, m_dims.z - o.z));

  // Ghost padding: extend by 1 voxel on each side, clamped to volume bounds.
  const vec3i go(std::max(0, o.x - 1),
                 std::max(0, o.y - 1),
                 std::max(0, o.z - 1));
  const vec3i ge(std::min(m_dims.x, o.x + e.x + 1) - go.x,
                 std::min(m_dims.y, o.y + e.y + 1) - go.y,
                 std::min(m_dims.z, o.z + e.z + 1) - go.z);

  dst.origin       = o;
  dst.extent       = e;
  dst.ghost_origin = go;
  dst.ghost_extent = ge;

  const size_t bytes_per_row   = (size_t)ge.x * (size_t)m_elem_size;
  const size_t bytes_per_slice = (size_t)ge.y * bytes_per_row;
  dst.data.resize((size_t)ge.z * bytes_per_slice);

  // The block is a full-x-width slab, so all `ge.y` rows of a given z-slice
  // are contiguous in the file (file layout is volume[z][y][x]).  We can
  // therefore fetch each z-slice in a single pread() call instead of one
  // pread per row -- ~ge.y x fewer syscalls per block load.
  //
  // pread() returns I/O errors via errno (translated to exceptions inside
  // RegularGridFile), so a transient disk hiccup becomes a recoverable
  // runtime_error rather than a SIGBUS.
  for (int z = 0; z < ge.z; ++z) {
    const vec3i  src_idx(go.x, go.y, go.z + z);
    const size_t src_off = m_offset + flatten_grid_index(src_idx, m_dims) * (size_t)m_elem_size;
    const size_t dst_off = (size_t)z * bytes_per_slice;
    m_file.pread_into(src_off, dst.data.data() + dst_off, bytes_per_slice);
  }
}

void OutOfCoreSampler::refresh_blocks(size_t begin, size_t count) {
  std::uniform_int_distribution<size_t> pick(0, m_num_block_indices - 1);
  for (size_t i = 0; i < count; ++i) {
    const size_t k = (begin + i) % m_num_blocks;
    const size_t flat = pick(m_rng);
    const vec3i  idx  = vec3i(0,
                              (int)((flat / (size_t)m_block_index_space.x) % (size_t)m_block_index_space.y),
                              (int)(flat / ((size_t)m_block_index_space.x * (size_t)m_block_index_space.y)));
    load_block(m_cache[k], idx);
  }
}

OutOfCoreSampler::OutOfCoreSampler(VolumeFileStructured desc)
  : m_dims(desc.dims)
  , m_type(desc.type)
  , m_range(resolve_required_range(desc.ranges, "out_of_core"))
  , m_offset(desc.offset)
  , m_elem_size(sizeof_type(desc.type))
  , m_value_scale(1.f / (m_range.upper - m_range.lower))
  , m_bounds(vec3f(0.f), vec3f((float)m_dims.x, (float)m_dims.y, (float)m_dims.z) * desc.spacing)
  , m_file(desc.filename)
  , m_rng(1337u)
{
  const size_t expected = (size_t)m_dims.x * (size_t)m_dims.y * (size_t)m_dims.z * (size_t)m_elem_size;
  if (m_file.file_size() < m_offset + expected) {
    throw std::runtime_error(
      "out_of_core sampler: file too small for given dims/dtype/offset"
    );
  }

  // Block layout: full-x slabs along y/z (matching InstantVNR's row-major
  // streaming layout).  Use a small default y-extent so the cache stays
  // cheap during tests.
  constexpr int default_block_y = 8;
  m_block_extent = vec3i(m_dims.x,
                         std::min(default_block_y, m_dims.y),
                         1);

  m_block_index_space = vec3i(1,
                              (m_dims.y + m_block_extent.y - 1) / m_block_extent.y,
                              (m_dims.z + m_block_extent.z - 1) / m_block_extent.z);
  m_num_block_indices = (size_t)m_block_index_space.x *
                        (size_t)m_block_index_space.y *
                        (size_t)m_block_index_space.z;

  // Cache sizing: env-tunable for parity with InstantVNR; defaults are
  // small so unit tests stay deterministic and cheap.
  const int env_concurrent = env_int("VNR_NUM_CONCURRENT_BLOCKS", 16);
  const int env_total      = env_int("VNR_NUM_BLOCKS",            64);
  m_num_concurrent_blocks = (size_t)std::max(1, env_concurrent);
  m_num_blocks            = (size_t)std::max((int)m_num_concurrent_blocks, env_total);
  m_num_blocks            = std::max<size_t>(m_num_blocks, 1);

  m_cache.resize(m_num_blocks);
  // Pre-fill the cache so the first sample() call has data to draw from.
  refresh_blocks(0, m_num_blocks);
  m_refresh_cursor = 0;

  // Cache memory footprint: each block stores its (ghost-padded) extent.
  // Worst-case ghost extent is m_block_extent + 2 voxels along y / z, capped
  // at the volume bounds; estimate using that worst case for the log line.
  const vec3i ghost_max(m_dims.x,
                        std::min(m_block_extent.y + 2, m_dims.y),
                        std::min(m_block_extent.z + 2, m_dims.z));
  const size_t bytes_per_block = (size_t)ghost_max.x * (size_t)ghost_max.y *
                                 (size_t)ghost_max.z * (size_t)m_elem_size;
  const size_t cache_bytes = bytes_per_block * m_num_blocks;
  const size_t volume_bytes = (size_t)m_dims.x * (size_t)m_dims.y *
                              (size_t)m_dims.z * (size_t)m_elem_size;

  std::cout
    << "[pysampler/out_of_core] sampler ready\n"
    << "  filename            : " << desc.filename << "\n"
    << "  dims                : " << m_dims.x << " x " << m_dims.y << " x " << m_dims.z << "\n"
    << "  dtype               : " << dtype_name(m_type)
    << "  (elem_size = " << m_elem_size << " B)\n"
    << "  range               : [" << m_range.lower << ", " << m_range.upper << "]"
    << "  (1/span = " << m_value_scale << ")\n"
    << "  spacing             : " << desc.spacing.x << " x " << desc.spacing.y << " x " << desc.spacing.z << "\n"
    << "  file offset         : " << m_offset << " B\n"
    << "  file size           : " << pretty_bytes(m_file.file_size()) << "\n"
    << "  volume bytes        : " << pretty_bytes(volume_bytes) << "\n"
    << "  block extent        : " << m_block_extent.x << " x " << m_block_extent.y << " x " << m_block_extent.z
    << "  (with 1-voxel ghost)\n"
    << "  block index space   : " << m_block_index_space.x << " x " << m_block_index_space.y << " x " << m_block_index_space.z
    << "  (=" << m_num_block_indices << " distinct positions)\n"
    << "  cache               : " << m_num_blocks << " blocks, "
    << m_num_concurrent_blocks << " refreshed per sample()\n"
    << "  cache bytes (max)   : " << pretty_bytes(cache_bytes) << "\n"
    << "  block I/O           : pread() (SIGBUS-safe; reports errors via exceptions)\n"
    << "  env tunables        : VNR_NUM_BLOCKS, VNR_NUM_CONCURRENT_BLOCKS\n"
    << std::flush;
}

float OutOfCoreSampler::file_sample_point(const vec3f& p_norm) const {
  const char*   base   = m_file.data() + m_offset;
  const range1f range  = m_range;
  const float   vscale = m_value_scale;
  const dtype   type   = m_type;
  const vec3i   dims   = m_dims;

  const auto accessor = [base, range, vscale, type, dims](const vec3i& ip) -> float {
    const size_t vidx = flatten_grid_index(ip, dims);
    const float  v    = read_scalar_as_float(base, vidx, type);
    const float  n    = (v - range.lower) * vscale;
    return std::min(std::max(n, 0.f), 1.f);
  };

  return trilinear_normalized(p_norm, dims, accessor);
}

float OutOfCoreSampler::block_sample_point(const Block& blk, const vec3f& p_norm) const {
  // Read all corners from the block's heap-resident bytes.  Sampling is
  // restricted to a random voxel within the (no-ghost) block extent, so the
  // trilinear footprint always lies inside the ghost-padded extent in exact
  // arithmetic.  We still clamp the corner index to the ghost range to
  // tolerate float-precision edge cases at the block boundary -- this way
  // sample() never touches the mmap'd region and is therefore SIGBUS-safe
  // even on flaky storage.
  const char*   block_data = blk.data.data();
  const range1f range      = m_range;
  const float   vscale     = m_value_scale;
  const dtype   type       = m_type;
  const vec3i   bg_origin  = blk.ghost_origin;
  const vec3i   bg_extent  = blk.ghost_extent;

  const auto accessor = [block_data, range, vscale, type,
                         bg_origin, bg_extent](const vec3i& ip) -> float {
    const vec3i clamped(
      std::min(std::max(ip.x, bg_origin.x), bg_origin.x + bg_extent.x - 1),
      std::min(std::max(ip.y, bg_origin.y), bg_origin.y + bg_extent.y - 1),
      std::min(std::max(ip.z, bg_origin.z), bg_origin.z + bg_extent.z - 1)
    );
    const vec3i  rel(clamped.x - bg_origin.x,
                     clamped.y - bg_origin.y,
                     clamped.z - bg_origin.z);
    const size_t vidx = (size_t)rel.x +
                        (size_t)rel.y * (size_t)bg_extent.x +
                        (size_t)rel.z * (size_t)bg_extent.x * (size_t)bg_extent.y;
    const float v = read_scalar_as_float(block_data, vidx, type);
    const float n = (v - range.lower) * vscale;
    return std::min(std::max(n, 0.f), 1.f);
  };

  return trilinear_normalized(p_norm, m_dims, accessor);
}

void OutOfCoreSampler::decode1d(void* h_coords, void* h_values,
                                const vec3f& lower, const vec3f& upper, size_t count)
{
  const vec3f* coords = (const vec3f*)h_coords;
  float*       values = (float*)h_values;

  const vec3f span = upper - lower;
  for (size_t i = 0; i < count; ++i) {
    const vec3f c = coords[i];
    const vec3f p = vec3f(c.x * span.x + lower.x,
                          c.y * span.y + lower.y,
                          c.z * span.z + lower.z);
    values[i] = file_sample_point(p);
  }
}

void OutOfCoreSampler::sample(void* h_coords, void* h_values,
                              const vec3f& lower, const vec3f& upper, size_t count)
{
  if (m_num_blocks == 0 || m_cache.empty()) {
    throw std::runtime_error("out_of_core sampler: cache is empty");
  }

  vec3f* coords = (vec3f*)h_coords;
  float* values = (float*)h_values;

  const vec3f rfdims(1.f / (float)m_dims.x,
                     1.f / (float)m_dims.y,
                     1.f / (float)m_dims.z);
  const vec3f span = upper - lower;

  std::uniform_int_distribution<size_t> pick_block(0, m_num_blocks - 1);
  std::uniform_real_distribution<float> uniform01(0.f, 1.f);

  for (size_t i = 0; i < count; ++i) {
    const Block& blk = m_cache[pick_block(m_rng)];

    // Pick a random voxel within the (no-ghost) block extent.
    const vec3f r(uniform01(m_rng), uniform01(m_rng), uniform01(m_rng));
    const vec3f voxel_pos(
      (float)blk.origin.x + r.x * (float)blk.extent.x,
      (float)blk.origin.y + r.y * (float)blk.extent.y,
      (float)blk.origin.z + r.z * (float)blk.extent.z
    );

    // Map voxel position back to volume-normalized [0, 1]^3 space, then
    // restrict to the caller's [lower, upper] subdomain.
    const vec3f p_volume_norm(voxel_pos.x * rfdims.x,
                              voxel_pos.y * rfdims.y,
                              voxel_pos.z * rfdims.z);
    const vec3f p_user = vec3f(
      span.x > 0.f ? (p_volume_norm.x - lower.x) / span.x : 0.f,
      span.y > 0.f ? (p_volume_norm.y - lower.y) / span.y : 0.f,
      span.z > 0.f ? (p_volume_norm.z - lower.z) / span.z : 0.f
    );

    coords[i] = vec3f(std::min(std::max(p_user.x, 0.f), 1.f),
                      std::min(std::max(p_user.y, 0.f), 1.f),
                      std::min(std::max(p_user.z, 0.f), 1.f));
    values[i] = block_sample_point(blk, p_volume_norm);
  }

  // Rotate part of the cache so subsequent samples see fresh blocks,
  // mirroring InstantVNR's RandomBuffer behavior.
  refresh_blocks(m_refresh_cursor, m_num_concurrent_blocks);
  m_refresh_cursor = (m_refresh_cursor + m_num_concurrent_blocks) % m_num_blocks;
}

void OutOfCoreSampler::decode3d(void* /*h_coords*/, void* /*h_values*/,
                                const vec3f& /*lower*/, const vec3f& /*upper*/,
                                const vec3i& /*dims*/)
{
  throw std::runtime_error("out_of_core sampler: decode3d is not implemented yet");
}

} // namespace

sampler_t create_out_of_core_sampler(VolumeFileStructured desc) {
  validate_file_backed_desc(desc, "out_of_core");
  return std::make_shared<OutOfCoreSampler>(std::move(desc));
}

} // namespace pysampler
