#include "device.h"
#include <core/instantvnr_types.h>

#ifdef ENABLE_OPENGL
#include <imgui.h>
#endif

namespace ovr::hyperinr {

// ------------------------------------------------------------------
// Implementation of the DeviceHyperINR
// ------------------------------------------------------------------

void
DeviceHyperINR::init(int argc, const char** argv)
{
  if (initialized) throw std::runtime_error("[nncache] device already initialized!");
  initialized = true;

  // --------------------------------------------
  // setup scene
  // --------------------------------------------
  const auto& scene = current_scene;
  auto& sv = parse_single_volume_scene(scene, scene::Volume::STRUCTURED_REGULAR_VOLUME).structured_regular;
  auto& st = scene.instances[0].models[0].volume_model.transfer_function;
  params.tfn.assign([&](TransferFunctionData& d) {
    d.tfn_value_range = st.value_range;
  });

  // --------------------------------------------------------------------------
  // create cache manager
  // --------------------------------------------------------------------------
  vnr::UserConfig* user_data = (vnr::UserConfig*)scene.user_data;

  // --------------------------------------------
  // framebuffer creation
  // --------------------------------------------
  framebuffer.create();
  framebuffer_stream = framebuffer.back_stream();

  // --------------------------------------------
  // create volume texture & transformation
  // --------------------------------------------
  const vec3f scale = sv.grid_spacing * vec3f(sv.data->dims);
  const vec3f translate = sv.grid_origin;
  volume.load_from_array3d_scalar(sv.data); // NOTE: no actually loading the data
  volume.set_space_partition_size(user_data->macrocell_size);
  volume.set_transform(affine3f::translate(translate) * affine3f::scale(scale));
  volume.set_sampling_rate(scene.volume_sampling_rate);
  volume.set_transfer_function(CreateArray1DFloat4CUDA(st.color), CreateArray1DScalarCUDA(st.opacity), st.value_range);
  volume.commit(); // commit volume to make sure we have a valid device handler

  // --------------------------------------------
  // Load precomputed macrocell from file
  // --------------------------------------------
  // NOTE: in NN Cache project, we assume CUDA-style data normalization rule:
  // -- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
  //    about cudaTextureDesc::readMode:
  //       Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted,
  //       regardless of whether or not this cudaTextureDesc::readMode is set cudaReadModeNormalizedFloat is specified.
  range1f mc_range = range1f(st.value_range.x, st.value_range.y);
  switch (volume.device.volume.type) {
  case VALUE_TYPE_UINT8:
    mc_range.lower = (float)std::numeric_limits<uint8_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<uint8_t>::max();
    break;
  case VALUE_TYPE_INT8:
    mc_range.lower = (float)std::numeric_limits<int8_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<int8_t>::max();
    break;
  case VALUE_TYPE_UINT16:
    mc_range.lower = (float)std::numeric_limits<uint16_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<uint16_t>::max();
    break;
  case VALUE_TYPE_INT16:
    mc_range.lower = (float)std::numeric_limits<int16_t>::lowest();
    mc_range.upper = (float)std::numeric_limits<int16_t>::max();
    break;
  default: break;
  }

  std::cout << "[mc] range " << mc_range << std::endl;
  set_space_partition(volume.device.sp, user_data->macrocell, mc_range, framebuffer_stream);

  // --------------------------------------------
  // create wavefront renderer
  // --------------------------------------------
  const auto& data = volume.device.volume;
  const auto& iter = volume.device.sp;
  api.stream = framebuffer_stream;
  api.init(volume.device.transform, 
    data.type, data.dims, 
    iter.dims, iter.spac,
    (vnr::vec2f*)iter.value_ranges,
    iter.majorants
  );

  // --------------------------------------------------------------------------
  // call commit
  // --------------------------------------------------------------------------  
  commit();
}

void
DeviceHyperINR::swap()
{
  framebuffer.safe_swap();
  framebuffer_stream = framebuffer.back_stream();
  api.stream = framebuffer_stream;
}

void 
DeviceHyperINR::commit_material() 
{
  if (check(ctls.ambient))   { lp.mat_scivis.ambient   = ctls.ambient.ref(); }
  if (check(ctls.diffuse))   { lp.mat_scivis.diffuse   = ctls.diffuse.ref(); }
  if (check(ctls.specular))  { lp.mat_scivis.specular  = ctls.specular.ref(); }
  if (check(ctls.shininess)) { lp.mat_scivis.shininess = ctls.shininess.ref(); }
}

void 
DeviceHyperINR::commit_lighting() 
{
  if (check(ctls.phi) || check(ctls.theta)) {
    const float phi_rad   = ctls.phi.get()   * (M_PI/180);
    const float theta_rad = ctls.theta.get() * (M_PI/180);
    const float radius = 2415.8;
    lp.l_distant.direction = vec3f( 
      radius * sin(phi_rad) * cos(theta_rad) * (180/M_PI),
      radius * sin(phi_rad) * sin(theta_rad) * (180/M_PI),
      radius * cos(phi_rad) * (180/M_PI)
    );
  }
  if (check(ctls.intensity)) {
    lp.l_distant.color = vec3f(ctls.intensity.get());
  }
}

void
DeviceHyperINR::commit()
{
  if (check(params.fbsize)) {
    lp.frame.size = params.fbsize.ref();
    CUDA_SYNC_CHECK(); /* sync rendering */
    framebuffer.resize(params.fbsize.ref());
  }

  // camera parameters
  if (check(params.camera)) { 
    camera = params.camera.ref(); 
  }

  // volume parameters
  if (check(params.tfn)) {
    const auto& tfn = params.tfn.ref();
    volume.set_transfer_function(tfn.tfn_colors, tfn.tfn_alphas, tfn.tfn_value_range);
    volume_changed = true;
  }
  if (check(params.volume_sampling_rate)) {
    volume.set_sampling_rate(params.volume_sampling_rate.get());
    volume_changed = true;
  }
  if (check(params.volume_density_scale)) {
    volume.set_density_scale(params.volume_density_scale.get());
    volume_changed = true;
  }
  if (volume_changed) {
    volume.commit();
    volume_changed = false;
  }

  // light & material parameters
  commit_material();
  commit_lighting();

  // other parameters
  if (check(params.path_tracing)) {
    lp.enable_path_tracing = params.path_tracing.ref();
    rendering_mode = lp.enable_path_tracing ? VNR_PATHTRACING : VNR_RAYMARCHING;
  }
  if (check(params.sparse_sampling)) {
    lp.enable_sparse_sampling = params.sparse_sampling.ref();
  }
  if (check(params.frame_accumulation)) {
    lp.enable_frame_accumulation = params.frame_accumulation.ref();
  }
  // finalize
  if (dirty) {
    api.update(rendering_mode, volume.device.tfn, 
      volume.sampling_rate, volume.density_scale,
      vec3f(0), vec3f(1), to_vnr(camera), lp.frame.size
    );
  }
}

void
DeviceHyperINR::render()
{
  if (lp.frame.size.x <= 0 || lp.frame.size.y <= 0) return;
  lp.frame.rgba = (vec4f*)framebuffer.back_dpointer(/*layout=*/0);

  api.render((vec4f*)framebuffer.back_dpointer(0), nullptr, volume.device.volume.data);

  /* post rendering */
  variance = 0.f; // TODO properly calculate variance
  dirty = false;
}

void
DeviceHyperINR::mapframe(FrameBufferData* fb)
{
  // CUDA_CHECK(cudaStreamSynchronize(framebuffer_stream));
  const size_t num_bytes = framebuffer.size().long_product();
  fb->rgba->set_data(framebuffer.front_dpointer(0), num_bytes * sizeof(vec4f), CrossDeviceBuffer::DEVICE_CUDA);
  fb->size = framebuffer.size();
}

#ifdef ENABLE_OPENGL

void 
DeviceHyperINR::ui(ImGuiContext* context) 
{
  ImGui::SetCurrentContext(context);
  static struct {
    float ambient{ .6f };
    float diffuse{ .9f };
    float specular{ .4f };
    float shininess{ 40.f };
    float phi{ 99.53f };
    float theta{ 112.2f };
    float intensity{ 1.f };

    bool edit_lm{ false };
  } locals;

  if (ImGui::Begin("NN-Cache Panel", NULL)) {

    ImGui::Checkbox("Edit Light & Material", &locals.edit_lm); 
    if (locals.edit_lm) {
      bool updated_mat = false;
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Ambient",   &locals.ambient,   0.f,   1.f, "%.3f");
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Diffuse",   &locals.diffuse,   0.f,   1.f, "%.3f");
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Specular",  &locals.specular,  0.f,   1.f, "%.3f");
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Shininess", &locals.shininess, 0.f, 100.f, "%.3f");
      bool updated_light = false;
      updated_light = updated_light || ImGui::SliderFloat("Light: Phi",       &locals.phi,       0.f, 360.f, "%.2f");
      updated_light = updated_light || ImGui::SliderFloat("Light: Theta",     &locals.theta,     0.f, 360.f, "%.2f");
      updated_light = updated_light || ImGui::SliderFloat("Light: Intensity", &locals.intensity, 0.f,   4.f, "%.3f");
      if (updated_mat) {
        ctls.ambient   = locals.ambient;
        ctls.diffuse   = locals.diffuse;
        ctls.specular  = locals.specular;
        ctls.shininess = locals.shininess;
      }
      if (updated_light) {
        ctls.phi       = locals.phi;
        ctls.theta     = locals.theta;
        ctls.intensity = locals.intensity;
      }
    }
  }
  ImGui::End();
}

#endif

}

static bool file_exists_test(std::string name) { std::ifstream f(name.c_str()); return f.good(); }
static bool file_exists_test(std::string name, const std::string& dir, std::string& out) {
  if (file_exists_test(name)) { out = name; return true; }
  else if (!dir.empty()) {
    if      (file_exists_test(dir + "/"  + name)) { out = name; return true; }
    else if (file_exists_test(dir + "\\" + name)) { out = name; return true; }
  }
  return false;
}
static std::string valid_filename(const vnr::json& in, std::string dir, const std::string& key) {
  std::string file;
  if (!in.contains(key)) { throw std::runtime_error("JSON key '" + key + "' doesnot exist"); }
  const auto& js = in[key];
  if (js.is_array()) {
    for (auto& s : js) { if (file_exists_test(s.get<std::string>(), dir, file)) { return file; } }
    throw std::runtime_error("Cannot find file for '" + key + "'.");
  }
  else {
    if (file_exists_test(js.get<std::string>(), dir, file)) { return file; }
    throw std::runtime_error("File '" + js.get<std::string>() + "' does not exist.");
  }
}

OVR_REGISTER_OBJECT(ovr::MainRenderer, renderer, ovr::hyperinr::DeviceHyperINR, hyperinr)

OVR_REGISTER_SCENE_LOADER(hyperinr, filename)
{
  std::ifstream file(filename);
  vnr::json root = vnr::json::parse(file, nullptr, true, true);

  vnr::TransferFunction tfn;
  vnr::Camera camera;
  vnr::vec3i dims; 
  vnr::ValueType type;
  vnr::create_json_scene_stringify(root, tfn, camera, dims, type);

  std::string volfile = valid_filename(root["dataSource"][0], "", "fileName");

  // ------------------------------------------------------------------
  // create scene
  // ------------------------------------------------------------------
  using namespace ovr;
  using namespace ovr::scene;

  Scene scene;

  Instance instance;
  instance.transform = affine3f::translate(vec3f(0));

  Volume volume{};
  volume.type = ovr::scene::Volume::STRUCTURED_REGULAR_VOLUME;
#if 0
  volume.structured_regular.data = std::make_shared<Array<3>>();
  volume.structured_regular.data->dims = dims;
  volume.structured_regular.data->type = hyperinr::to_ovr(type);
#else
  volume.structured_regular.data = CreateArray3DScalarFromFile(volfile.c_str(), dims, hyperinr::to_ovr(type), 0, false);
#endif

  ovr::scene::TransferFunction transfer_function{};

  std::vector<vec4f> color(tfn.color.size());
  for (int i = 0; i < tfn.color.size(); ++i) {
    color[i] = vec4f(tfn.color[i], 1.f);
  }

  std::vector<float> alpha(tfn.alpha.size());
  for (int i = 0; i < tfn.alpha.size(); ++i) {
    alpha[i] = tfn.alpha[i].y;
  }

  transfer_function.color   = ovr::CreateArray1DFloat4(color);
  transfer_function.opacity = ovr::CreateArray1DScalar(alpha);
  transfer_function.value_range.x = tfn.range.lo;
  transfer_function.value_range.y = tfn.range.hi;

  Texture texture;
  texture.type = Texture::VOLUME_TEXTURE;
  texture.volume.volume = volume;
  scene.textures.push_back(texture);
  
  Model model;
  model.type = Model::VOLUMETRIC_MODEL;
  model.volume_model.volume_texture = scene.textures.size() - 1;
  model.volume_model.transfer_function = transfer_function;

  instance.models.push_back(model);
  scene.instances.push_back(instance);
  scene.camera = hyperinr::to_ovr(camera);

  vnr::UserConfig* user_data = new vnr::UserConfig();

  // ------------------------------------------------------------------
  // parse macrocell configuration
  // ------------------------------------------------------------------
  if (!root.contains("macrocell")) { throw std::runtime_error("JSON missing macrocell configuration"); }
  const auto& mc = root["macrocell"];

  if (mc.contains("fileName")) {
    user_data->macrocell = valid_filename(mc, "", "fileName");
  }

  vnr::vec3i mc_size;
  if (mc.contains("mcSize")) {
    user_data->macrocell_size = mc_size = vnr::vec3i(
      mc["mcSize"]["x"].get<int>(), 
      mc["mcSize"]["y"].get<int>(), 
      mc["mcSize"]["z"].get<int>()
    );
  }
  else {
    throw std::runtime_error("JSON macrocell missing dimensions");
  }

  // Compute macrocell on the fly
  if (user_data->macrocell.empty()) {
    std::string volume_path = valid_filename(root["dataSource"][0], "", "fileName");
    std::string output_path = "";
    tdns::ProgressiveMacroCell macrocell(mc_size, dims, type);
    macrocell.process(volume_path, output_path);
    user_data->macrocell = output_path;
  }

  // ------------------------------------------------------------------
  // write output
  // ------------------------------------------------------------------
  scene.user_data = user_data;

  return scene;
}                            
