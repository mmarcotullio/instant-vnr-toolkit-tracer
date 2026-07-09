#define PYSAMPLER_SAMPLER_CPP_IMPL

#include "sampler_cuda.h"

#include <cstdlib>
#include <cstring>

#ifdef __both__
#undef __both__
#endif

// I want to avoid creating a very small header file separately ...
#define SAMPLER_CUDA_STITCHER_DEVICECODE_IMPL
#include "sampler_cuda_exa_devicecode.cu"

#include "model/ExaBrickModel.h"
#include "model/ExaStitchModel.h"
// #include "model/AMRCellModel.h"
// #include "model/BigMeshModel.h"
// #include "model/QuickClustersModel.h"

#include "sampler/ExaBrickSampler.h"
#include "sampler/ExaStitchSampler.h"
// #include "sampler/AMRCellSampler.h"
// #include "sampler/BigMeshSampler.h"
// #include "sampler/QuickClustersSampler.h"

namespace pysampler { // TODO support multi-attributes

extern "C" char sampler_cuda_exa_devicecode[];

static const OWLVarDecl rayGenVars[] = {
  { nullptr /* sentinel to mark end of list */ }
};

static const std::vector<OWLVarDecl> commonLPVars = {
  { "coords",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, coords) },
  { "values",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, values) },
  { "worldSpaceBounds.lower",  OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, worldSpaceBounds.lower)},
  { "worldSpaceBounds.upper",  OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, worldSpaceBounds.upper)},
  { "valueRange.lower",  OWL_FLOAT, OWL_OFFSETOF(LaunchParams, valueRange.lower)},
  { "valueRange.upper",  OWL_FLOAT, OWL_OFFSETOF(LaunchParams, valueRange.upper)},
};

struct CUDAImplExa {
  // Model can e.g. be ExaStitcher, ExaBricks, etc.
  exa::Model::SP model { 0 };
  // Sampler is generated automatically from the given model
  exa::Sampler::SP sampler { 0 };

  OWLContext owl;
  OWLModule  owl_module;
  OWLParams  lp;
  OWLRayGen  rayGen;

  owl::common::box3f  modelBounds;
  owl::common::interval<float>  valueRange;

  void initHost(std::string umeshFileName, std::string gridsFileName, std::string exaBrickFileName, std::string scalarFileName);
  void initDevice();
};

void CUDAImplExa::initDevice() {
  using namespace exa;

  // ==================================================================
  // Upload to GPU
  // ==================================================================
  if (model->as<ExaStitchModel>()) {
    sampler = std::make_shared<ExaStitchSampler>();
  }
  else if (model->as<ExaBrickModel>()) {
    sampler = std::make_shared<ExaBrickSampler>();
  }
  // else if (model->as<AMRCellModel>()) {
  //   sampler = std::make_shared<AMRCellSampler>();
  // } else if (model->as<BigMeshModel>()) {
  //   sampler = std::make_shared<BigMeshSampler>();
  // } else if (model->as<QuickClustersModel>()) {
  //   sampler = std::make_shared<QuickClustersSampler>();
  // }

  if (!sampler) {
    throw std::runtime_error("Could not load module");
  }
  owl = owlContextCreate(nullptr, 1);
  owlContextSetRayTypeCount(owl, 2); // !! matches the definitions in samplers
  owl_module = owlModuleCreate(owl, sampler_cuda_exa_devicecode);

  const std::string renderFrame = "renderFrame_" + sampler->className();
  rayGen = owlRayGenCreate(owl, owl_module, renderFrame.c_str(), 0, (OWLVarDecl *)rayGenVars, -1);
  // PRINT(renderFrame);

  // -------------------------------------------------------
  // set up launch params
  // -------------------------------------------------------
  std::vector<OWLVarDecl> samplerLPVars = sampler->getLPVariables();
  std::vector<OWLVarDecl> lpVars;
  for (auto var : commonLPVars) {
    if (var.name) lpVars.push_back(var);
  }
  for (auto var : samplerLPVars) {
    if (var.name) {
      var.offset += OWL_OFFSETOF(LaunchParams, sampler);
      lpVars.push_back(var);
    }
  }
  lpVars.push_back({nullptr}); // sentinel

  lp = owlParamsCreate(owl, sizeof(LaunchParams), lpVars.data(), -1);
  if (sampler->build(owl,model)) {
    // Currently only for ExaBricks, build an optional BVH over the majorant grid
    // (the OPtiX code for that lives in *this* module!)
    if (auto ebs = sampler->as<ExaBrickSampler>()) {
      if (ebs->traversalMode == MC_BVH_TRAVERSAL && !ebs->buildOptixBVH(owl, owl_module)) {
        throw std::runtime_error("Building BVH from grid failed");
      }
    }
    sampler->setLPs(lp);
  } else {
    throw std::runtime_error("GPU upload failed");
  }

  // ==================================================================
  // mesh geom
  // ==================================================================
  // owlParamsSetRaw(lp,"voxelSpaceTransform", &model->voxelSpaceTransform);
  owlParamsSet1f(lp,"valueRange.lower", valueRange.lower);
  owlParamsSet1f(lp,"valueRange.upper", valueRange.upper);
  owlParamsSet3f(lp,"worldSpaceBounds.lower",
                 modelBounds.lower.x,
                 modelBounds.lower.y,
                 modelBounds.lower.z);
  owlParamsSet3f(lp,"worldSpaceBounds.upper",
                 modelBounds.upper.x,
                 modelBounds.upper.y,
                 modelBounds.upper.z);
  owlBuildPipeline(owl);
  owlBuildSBT(owl);
}

void CUDAImplExa::initHost(std::string umeshFileName, std::string gridsFileName, std::string exaBrickFileName, std::string scalarFileName) 
{
  using namespace exa;

  std::string bigMeshFileName, quickClustersFileName, amrCellFileName, kdtreeFileName;

  if (!umeshFileName.empty() || !gridsFileName.empty()) { // only need one of them
    model = ExaStitchModel::load(umeshFileName, gridsFileName, scalarFileName);
    printf(">>>> ExaStitchModel <<<<\n");
  }
  else if (!exaBrickFileName.empty() && !scalarFileName.empty()) {
    printf(">>>> ExaBrickModel <<<<\n");
    model = ExaBrickModel::load(exaBrickFileName, scalarFileName, kdtreeFileName);
  }
  // else if (!amrCellFileName.empty() && !scalarFileName.empty()) {
  //   printf(">>>> AMRCellModel <<<<\n");
  //   model = AMRCellModel::load(amrCellFileName,scalarFileName);
  // }
  // else if (!bigMeshFileName.empty()) {
  //   printf(">>>> BigMeshModel <<<<\n");
  //   model = BigMeshModel::load(bigMeshFileName);
  // }
  // else if (!quickClustersFileName.empty()) {
  //   printf(">>>> QuickClustersModel <<<<\n");
  //   model = QuickClustersModel::load(quickClustersFileName);
  // }

  if (!model) {
    throw std::runtime_error("Could not load module");
  }

  owl::common::box3f remap_from { owl::common::vec3f(0.f), owl::common::vec3f(1.f) };
  owl::common::box3f remap_to   { owl::common::vec3f(0.f), owl::common::vec3f(1.f) };
  model->setVoxelSpaceTransform(remap_from,remap_to);

#ifdef EXA_STITCH_MIRROR_EXAJET
  model->initMirrorExajet(); // before extending model bounds!
#endif

  modelBounds.extend(model->getBounds());
  valueRange.extend(model->valueRange);
  model->setNumGridCells(owl::common::vec3i{64, 64, 64}); // SKIP computation of macrocells

  std::cout << "Model bounds: " << modelBounds << std::endl;
  std::cout << "Value range: " << valueRange << std::endl;

  // ==================================================================
  // Print memory stats
  // ==================================================================
  bool printMemoryStats = true;

  if (printMemoryStats) {
    size_t elemVertexBytes = 0;
    size_t elemIndexBytes = 0;
    size_t gridletBytes = 0;
    size_t emptyScalarsBytes = 0;
    size_t nonEmptyScalarsBytes = 0;
    size_t exaBrickBytes = 0;
    size_t exaScalarBytes = 0;
    size_t abrBytes = 0;
    size_t abrLeafListBytes = 0;
    size_t amrCellBytes = 0;
    size_t amrScalarBytes = 0;
    size_t meshIndexBytes = 0;
    size_t meshVertexBytes = 0;
    if (auto mod = model->as<ExaStitchModel>()) {
      mod->memStats(elemVertexBytes,elemIndexBytes,gridletBytes,
                    emptyScalarsBytes,nonEmptyScalarsBytes);
    }
    else if (auto mod = model->as<ExaBrickModel>()) {
      mod->memStats(exaBrickBytes,exaScalarBytes,abrBytes,abrLeafListBytes);
    }
    // else if (auto mod = model->as<AMRCellModel>()) {
    //   mod->memStats(amrCellBytes,amrScalarBytes);
    // }
    size_t totalBytes = elemVertexBytes+elemIndexBytes+emptyScalarsBytes+nonEmptyScalarsBytes
                      + gridletBytes+amrCellBytes+amrScalarBytes
                      + exaBrickBytes+exaScalarBytes+abrBytes
                      + meshIndexBytes+meshVertexBytes;
    std::cout << " ====== Memory Stats (bytes) ======= \n";
    std::cout << "elem.vertex.........: " << owl::prettyBytes(elemVertexBytes) << '\n';
    std::cout << "elem.index..........: " << owl::prettyBytes(elemIndexBytes) << '\n';
    std::cout << "Non-empty scalars...: " << owl::prettyBytes(nonEmptyScalarsBytes) << '\n';
    std::cout << "Empty scalars.......: " << owl::prettyBytes(emptyScalarsBytes) << '\n';
    std::cout << "Gridlets............: " << owl::prettyBytes(gridletBytes) << '\n';
    std::cout << "AMR cells...........: " << owl::prettyBytes(amrCellBytes) << '\n';
    std::cout << "AMR cells...........: " << owl::prettyBytes(amrCellBytes) << '\n';
    std::cout << "EXA bricks..........: " << owl::prettyBytes(exaBrickBytes) << '\n';
    std::cout << "EXA scalars.........: " << owl::prettyBytes(exaScalarBytes) << '\n';
    std::cout << "EXA ABRs............: " << owl::prettyBytes(abrBytes) << '\n';
    std::cout << "EXA ABR leaf list...: " << owl::prettyBytes(abrLeafListBytes) << '\n';
    std::cout << "mesh.vertex.........: " << owl::prettyBytes(meshVertexBytes) << '\n';
    std::cout << "mesh.index..........: " << owl::prettyBytes(meshIndexBytes) << '\n';
    std::cout << "TOTAL...............: " << owl::prettyBytes(totalBytes) << '\n';
  }

  // ==================================================================
  // Upload to GPU
  // ==================================================================
  initDevice();
  // std::cout << "Model bounds: " << modelBounds << std::endl;
}

void 
CUDASampler::process_exa(const VolumeFileExaBrick& desc)
{
  m_sampling_mode = EXA;
  m_fields_exa = std::make_shared<CUDAImplExa>();

  auto& ctx = *m_fields_exa;
  ctx.initHost("", "", desc.bricks, desc.scalar);

  m_bounds = box3f(
    vec3f(ctx.modelBounds.lower.x, ctx.modelBounds.lower.y, ctx.modelBounds.lower.z), 
    vec3f(ctx.modelBounds.upper.x, ctx.modelBounds.upper.y, ctx.modelBounds.upper.z)
  );
  m_ranges = { 
    range1f(ctx.valueRange.lower, ctx.valueRange.upper)
  };
}

void 
CUDASampler::process_exa(const VolumeFileExaStitch& desc)
{
  m_sampling_mode = EXA;
  m_fields_exa = std::make_shared<CUDAImplExa>();

  auto& ctx = *m_fields_exa;
  ctx.initHost(desc.umesh, desc.grids, "", desc.scalar);

  m_bounds = box3f(
    vec3f(ctx.modelBounds.lower.x, ctx.modelBounds.lower.y, ctx.modelBounds.lower.z), 
    vec3f(ctx.modelBounds.upper.x, ctx.modelBounds.upper.y, ctx.modelBounds.upper.z)
  );
  m_ranges = { 
    range1f(ctx.valueRange.lower, ctx.valueRange.upper)
  };
}

void 
sample_exa(const CUDASampler* self, void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count, void* stream)
{
  auto& ctx = *(self->m_fields_exa);

  // std::cout << "sample_exastitch" << std::endl;
  owlParamsSetPointer(ctx.lp, "coords", d_coords);
  owlParamsSetPointer(ctx.lp, "values", d_values);
  owlLaunch2D(ctx.rayGen, count, 1, ctx.lp);
  cudaDeviceSynchronize();

  // // copy back to host
  // std::vector<vec3f> h_coords(count);
  // std::vector<float> h_values(count);
  // cudaMemcpy(h_coords.data(), d_coords, count * sizeof(vec3f), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_values.data(), d_values, count * sizeof(float), cudaMemcpyDeviceToHost);
  // for (size_t i = 0; i < count; ++i) {
  //   // if (isnan(h_values[i])) {
  //   //   std::cout << "coord: " << h_coords[i] << " value: " << h_values[i] << std::endl;
  //   // }
  //   std::cout << "coord: " << h_coords[i] << " value: " << h_values[i] << std::endl;
  // }
}

}
