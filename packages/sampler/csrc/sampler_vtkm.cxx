#define PYSAMPLER_SAMPLER_CPP_IMPL
#include "sampler_vtkm.h"
#include "generate_random_numbers.h"
#include "config.h"

#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/cont/Initialize.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayCopyDevice.h>
#include <vtkm/cont/EnvironmentTracker.h>
#include <vtkm/cont/MergePartitionedDataSet.h>
#include <vtkm/cont/PartitionedDataSet.h>
#include <vtkm/cont/FieldRangeCompute.h>

// cell types definition for unstructured volumes, values are set to match VTK
typedef enum
# if __cplusplus >= 201103L
: uint8_t
#endif
{
  VKL_TETRAHEDRON = 10,
  VKL_HEXAHEDRON = 12,
  VKL_WEDGE = 13,
  VKL_PYRAMID = 14
} VKLUnstructuredCellType;

namespace pysampler {

struct ToVKLCellType
{
  VTKM_EXEC_CONT vtkm::UInt8 operator()(vtkm::UInt8 shape) const { 
    if(shape == vtkm::CELL_SHAPE_TETRA)
    {
      return VKL_TETRAHEDRON;
    }
    else if(shape == vtkm::CELL_SHAPE_HEXAHEDRON)
    {
      return VKL_HEXAHEDRON;
    }
    else if(shape == vtkm::CELL_SHAPE_WEDGE)
    {
      return VKL_WEDGE;
    }
    else if(shape == vtkm::CELL_SHAPE_PYRAMID)
    {
      return VKL_PYRAMID;
    }
    return uint8_t(-1);
  }
};

struct VTKmDataHelper::Impl {
  vtkm::cont::PartitionedDataSet dataset;
  std::string field_name;
};

struct VTKmArrays_Unstructured {
  vtkm::cont::ArrayHandle<vtkm::Vec3f_32> VertexPosition;
  vtkm::cont::ArrayHandle<vtkm::Float32> VertexData;
  vtkm::cont::ArrayHandle<vtkm::UInt64> Index;
  vtkm::cont::ArrayHandle<vtkm::UInt64> CellIndex;
  vtkm::cont::ArrayHandle<vtkm::UInt8> CellType;
};

template<typename T1, typename T2> 
void VectorAppendCopy(std::vector<T1>& vec, const vtkm::cont::ArrayHandle<T2>& array)
{
  static_assert(sizeof(T1) == sizeof(T2), "size mismatch");

  size_t n_old = vec.size();
  size_t n_app = array.GetNumberOfValues();
  vec.resize(n_old + n_app);

  vtkm::cont::Token token;
  auto* ptr = (T2*)array.GetBuffers()[0].ReadPointerHost(token);
  std::memcpy(vec.data() + n_old, ptr, n_app * sizeof(T1));
}

template<typename CellType>
void 
vtkm_copy_unstructured_cells(VTKmArrays_Unstructured& arrays, const CellType& cells) 
{
  // 1. Cell Type
  auto shapes = cells.GetShapesArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());
  vtkm::cont::ArrayCopyDevice(
    vtkm::cont::make_ArrayHandleTransform(shapes, ToVKLCellType{}), 
    arrays.CellType
  );
  // 2. Cell Connectivity
  auto conn = cells.GetConnectivityArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());
  vtkm::cont::ArrayCopyDevice(conn, arrays.Index);
  // 3. Cell Index
  auto offsets = cells.GetOffsetsArray(
    vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint()
  );
  vtkm::cont::ArrayCopyDevice(offsets, arrays.CellIndex);
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

static int _init = []() { vtkm::cont::Initialize(); return 0; } ();

VTKmDataHelper::VTKmDataHelper(VolumeFileVTKm desc)
  : pimpl(std::make_shared<Impl>())
{
  process_vtkm_init(desc.field);
  for (const auto& file : desc.files) {
    vtkm::io::VTKDataSetReader reader(file);
    process_vtkm_add(reader.ReadDataSet());
  }
  process_vtkm_finalize();
}

void 
VTKmDataHelper::process_vtkm_init(std::string field)
{
  pimpl->field_name = field;

  structured.grid.clear();

  unstructured.vertex_position.clear();
  unstructured.vertex_data.clear();
  unstructured.index.clear();
  unstructured.cell_index.clear();
  unstructured.cell_type.clear();
}

void 
VTKmDataHelper::process_vtkm_add(const vtkm::cont::DataSet& dset)
{
  const auto& cells = dset.GetCellSet();
  const bool isStructured = cells.CanConvert<vtkm::cont::CellSetStructured<3>>();
  if (isStructured)
  {
    throw std::runtime_error("not implemented");
  }
  else 
  {
    pimpl->dataset.AppendPartition(dset);
  }
}

void 
VTKmDataHelper::process_vtkm_finalize()
{
  vtkm::cont::DataSet dset = vtkm::cont::MergePartitionedDataSet(pimpl->dataset);

  /////////////////////////////////////////////////////////////////////
  /// Parse the Dataset
  /////////////////////////////////////////////////////////////////////

  // dset.PrintSummary(std::cout);

  const auto& coords = dset.GetCoordinateSystem();
  const auto& cells = dset.GetCellSet();
  const auto& field = dset.GetField(pimpl->field_name);

  const auto& fieldArray = field.GetData();
  const auto  fieldRangeArray = vtkm::cont::FieldRangeCompute(dset, pimpl->field_name);

  const bool isPointBased = field.GetAssociation() == vtkm::cont::Field::Association::Points;
  const bool isStructured = cells.CanConvert<vtkm::cont::CellSetStructured<3>>();
  const bool isScalar = fieldArray.GetNumberOfComponentsFlat() == 1;

  if (!isScalar) 
  {
    throw std::runtime_error("not implemented for unstructured multi-component fields");
  }

  if (isStructured)
  {
    throw std::runtime_error("not implemented");
  }
  else 
  {
    VTKmArrays_Unstructured arrays;

    if (cells.IsType<vtkm::cont::CellSetSingleType<>>()) {
      vtkm::cont::CellSetSingleType<> sgl = cells.AsCellSet<vtkm::cont::CellSetSingleType<>>();
      vtkm_copy_unstructured_cells(arrays, sgl);
    }
    else if (cells.IsType<vtkm::cont::CellSetExplicit<>>()) {
      vtkm::cont::CellSetExplicit<> exp = cells.AsCellSet<vtkm::cont::CellSetExplicit<>>();
      vtkm_copy_unstructured_cells(arrays, exp);
    }

    vtkm::cont::ArrayCopyShallowIfPossible(coords.GetData(), arrays.VertexPosition);
    vtkm::cont::ArrayCopyShallowIfPossible(fieldArray, arrays.VertexData);

    VectorAppendCopy(unstructured.vertex_position, arrays.VertexPosition);
    VectorAppendCopy(unstructured.vertex_data, arrays.VertexData);
    VectorAppendCopy(unstructured.index, arrays.Index);
    VectorAppendCopy(unstructured.cell_index, arrays.CellIndex);
    VectorAppendCopy(unstructured.cell_type, arrays.CellType);
  }

  unstructured.vertex_position.shrink_to_fit();
  unstructured.vertex_data.shrink_to_fit();
  unstructured.index.shrink_to_fit();
  unstructured.cell_index.shrink_to_fit();
  unstructured.cell_type.shrink_to_fit();

  // std::cout << "**vertex_position: " << unstructured.vertex_position.size() << std::endl;
  // std::cout << "**vertex_data: " << unstructured.vertex_data.size() << std::endl;
  // std::cout << "**index: " << unstructured.index.size() << std::endl;
  // std::cout << "**cell_index: " << unstructured.cell_index.size() << std::endl;
  // std::cout << "**cell_type: " << unstructured.cell_type.size() << std::endl;

  auto fieldRangePortal = fieldRangeArray.ReadPortal();
  if (fieldRangePortal.GetNumberOfValues() != 1) 
  {
    throw std::runtime_error("not implemented for unstructured multi-component fields");
  }

  ranges.resize(fieldRangePortal.GetNumberOfValues());
  for (int i = 0; i < fieldRangePortal.GetNumberOfValues(); ++i) {
    vtkm::Range range = fieldRangePortal.Get(i);
    ranges[i] = range1f{ (float)range.Min, (float)range.Max };
  }
}

} // namespace vnr
