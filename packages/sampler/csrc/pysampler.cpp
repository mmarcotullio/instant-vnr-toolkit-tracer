#include "sampler.h"
#include "config.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pysampler {

// Parse `range=[vmin, vmax]` (single attribute) and/or
// `ranges=[[vmin, vmax], ...]` (multi-attribute) kwargs into a list of
// `range1f`. Either form is optional; backends decide whether a range
// is required (e.g. file-backed samplers) or computed from the data.
static std::vector<range1f> parse_ranges_kwarg(const py::kwargs& kwargs) {
  std::vector<range1f> ranges;
  if (kwargs.contains("ranges")) {
    auto _ranges = py::cast<std::vector<std::array<double, 2>>>(kwargs["ranges"]);
    ranges.reserve(_ranges.size());
    for (const auto& r : _ranges) {
      ranges.push_back(range1f{ (float)r[0], (float)r[1] });
    }
  }
  if (kwargs.contains("range")) {
    auto _range = py::cast<std::array<double, 2>>(kwargs["range"]);
    if (ranges.empty()) {
      ranges.push_back(range1f{ (float)_range[0], (float)_range[1] });
    }
    else {
      // If both `range` and `ranges` are passed, treat `range` as the first
      // attribute (overrides). Anything beyond stays from `ranges`.
      ranges.front() = range1f{ (float)_range[0], (float)_range[1] };
    }
  }
  return ranges;
}

sampler_t create_structured(std::string device, const py::kwargs& kwargs)
{
  auto _dims = py::cast<std::array<int, 3>>(kwargs["dims"]);
  auto dims = vec3i{ _dims[0], _dims[1], _dims[2] };

  auto _type = py::cast<std::string>(kwargs["dtype"]);
  dtype type;
  if      (_type == "uint8")   type = dtype::UINT8;
  else if (_type == "int8")    type = dtype::INT8;
  else if (_type == "uint16")  type = dtype::UINT16;
  else if (_type == "int16")   type = dtype::INT16;
  else if (_type == "uint32")  type = dtype::UINT32;
  else if (_type == "int32")   type = dtype::INT32;
  else if (_type == "float")   type = dtype::FLOAT;
  else if (_type == "float32") type = dtype::FLOAT;
  else if (_type == "double")  type = dtype::DOUBLE;
  else if (_type == "float64") type = dtype::DOUBLE;
  else throw std::runtime_error("unknown data type: " + _type);

  vec3f spacing = vec3f{ 1, 1, 1 };
  if (kwargs.contains("spacing")) {
    auto _spacing = py::cast<std::array<double, 3>>(kwargs["spacing"]);
    spacing = vec3f{ (float)_spacing[0], (float)_spacing[1], (float)_spacing[2] };
  }

  const int n_channels = kwargs.contains("n_channels") ? py::cast<int>(kwargs["n_channels"]) : 1;
  const std::vector<range1f> ranges = parse_ranges_kwarg(kwargs);

  if (kwargs.contains("filename")) {
    const auto filename   = py::cast<std::string>(kwargs["filename"]);
    const auto offset     = kwargs.contains("offset")         ? py::cast<size_t>(kwargs["offset"])       : 0ULL;
    const bool big_endian = kwargs.contains("is_big_endian")  ? py::cast<bool>(kwargs["is_big_endian"])  : false;
    VolumeFileStructured desc;
    desc.type          = type;
    desc.dims          = dims;
    desc.spacing       = spacing;
    desc.n_channels    = n_channels;
    desc.ranges        = ranges;
    desc.filename      = filename.c_str();
    desc.offset        = offset;
    desc.is_big_endian = big_endian;
    return create_sampler(device.c_str(), desc);
  }
  else {
    VolumeDataStructured desc;
    desc.type       = type;
    desc.dims       = dims;
    desc.spacing    = spacing;
    desc.n_channels = n_channels;
    desc.ranges     = ranges;
    return create_sampler(device.c_str(), desc);
  }
}

sampler_t py_create_sampler(std::string type, std::string device, const py::kwargs& kwargs)
{
  if (type == "structuredRegular")
    return create_structured(device, kwargs);

  if (type == "openvdb") {
    const auto filename = py::cast<std::string>(kwargs["filename"]);
    const auto field    = py::cast<std::string>(kwargs["field"]);
    VolumeFileOpenVDB desc;
    desc.filename = filename.c_str();
    desc.field    = field.c_str();
    return create_sampler(device.c_str(), desc);
  }

  if (type == "vtkm") {
    const auto files = py::cast<std::vector<std::string>>(kwargs["files"]);
    const auto field = py::cast<std::string>(kwargs["field"]);
    VolumeFileVTKm desc;
    for (const auto& f : files) desc.files.push_back(f.c_str());
    desc.field = field.c_str();
    return create_sampler(device.c_str(), desc);
  }

  if (type == "exastitch") {
    const auto umesh  = py::cast<std::string>(kwargs["umesh"]);
    const auto grids  = py::cast<std::string>(kwargs["grids"]);
    const auto scalar = py::cast<std::string>(kwargs["scalar"]);
    VolumeFileExaStitch desc;
    desc.umesh  = umesh.c_str();
    desc.grids  = grids.c_str();
    desc.scalar = scalar.c_str();
    return create_sampler(device.c_str(), desc);
  }

  if (type == "exabrick") {
    const auto bricks = py::cast<std::string>(kwargs["bricks"]);
    const auto scalar = py::cast<std::string>(kwargs["scalar"]);
    VolumeFileExaBrick desc;
    desc.bricks = bricks.c_str();
    desc.scalar = scalar.c_str();
    return create_sampler(device.c_str(), desc);
  }

  throw std::runtime_error("unknown sampler type: " + type);
}

// ---------------------------------------------------------------------------
// All sampling functions take pre-allocated buffer pointers. The caller
// (Python) owns the buffers and controls their lifetime.
//
// Whether the pointers are CUDA device pointers or host pointers depends on
// the sampler's backend:
//   - device "cuda"           -> CUDA device pointers
//   - device "openvkl"        -> host pointers (CPU)
//   - device "virtual_memory" -> host pointers (CPU)
//   - device "out_of_core"    -> host pointers (CPU)
//
// coords_ptr : float32 buffer, shape (count, 3) — sample positions in [0,1]^3
// values_ptr : float32 buffer, shape (n_channels, count) — scalar values
//              (column-major; transpose on the Python side to get (count, n_channels))
// ---------------------------------------------------------------------------

void py_sample(sampler_t sampler, uintptr_t coords_ptr, uintptr_t values_ptr, int64_t count) {
  sampler->sample(
    reinterpret_cast<void*>(coords_ptr),
    reinterpret_cast<void*>(values_ptr),
    vec3f{0, 0, 0}, vec3f{1, 1, 1},
    static_cast<uint32_t>(count)
  );
}

void py_decode(sampler_t sampler, uintptr_t coords_ptr, uintptr_t values_ptr, int64_t count) {
  sampler->decode1d(
    reinterpret_cast<void*>(coords_ptr),
    reinterpret_cast<void*>(values_ptr),
    vec3f{0, 0, 0}, vec3f{1, 1, 1},
    static_cast<uint32_t>(count)
  );
}

} // namespace pysampler

PYBIND11_MODULE(EXTENSION_NAME, m) {
  py::class_<pysampler::SamplerBase, pysampler::sampler_t>(m, "Sampler")
    .def("n_channels", &pysampler::SamplerBase::n_channels,
         "Number of scalar channels in the volume (1 for scalar fields).");

  m.def("create_sampler", &pysampler::py_create_sampler,
        "Create a volume sampler. "
        "type: 'structuredRegular', 'openvdb', 'vtkm', 'exastitch', 'exabrick'. "
        "device for structuredRegular: 'cuda', 'openvkl', 'virtual_memory', 'out_of_core'. "
        "Optional kwargs include `range=[vmin, vmax]` (or `ranges=[...]`) which is "
        "required for the file-backed CPU samplers ('virtual_memory', 'out_of_core').");

  m.def("sample", &pysampler::py_sample,
        py::arg("sampler"), py::arg("coords_ptr"), py::arg("values_ptr"), py::arg("count"),
        "Fill pre-allocated buffers (CUDA device pointers for the CUDA backend, "
        "host pointers for CPU backends). coords: (count,3) float32; values: (n_channels,count) float32.");

  m.def("decode", &pysampler::py_decode,
        py::arg("sampler"), py::arg("coords_ptr"), py::arg("values_ptr"), py::arg("count"),
        "Sample at caller-supplied coordinates. Pointer kind matches the sampler's backend "
        "(CUDA device pointers for 'cuda', host pointers for CPU samplers).");

  m.doc() = "Volume sampler with CUDA, OpenVKL, virtual-memory, and out-of-core backends.";
}
