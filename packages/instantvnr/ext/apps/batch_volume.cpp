//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include <api.h>

#include "cmdline.h"

#include <cuda/cuda_buffer.h>
#ifdef NDEBUG
#define TRACE_CUDA ((void)0)
#else
#define TRACE_CUDA CUDA_SYNC_CHECK()
#endif

#include <vidi_logger.h>

#include <stbi/stb_image_write.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define SCENE_SCALE 1024

using namespace vnr::math;

struct Camera {
  vec3f from;
  vec3f at;
  vec3f up;
  float fovy = 60;
};

struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;
  args::Group group1;
  args::Group group2;

  args::ValueFlag<std::string> m_simple_volume;
  args::ValueFlag<std::string> m_neural_volume;
  bool has_simple_volume() { return m_simple_volume; }
  bool has_neural_volume() { return m_neural_volume; }
  std::string volume() { return (m_simple_volume) ? args::get(m_simple_volume) : args::get(m_neural_volume); }

  args::ValueFlag<std::string> scalar_field;
  args::ValueFlag<std::string> shadow_field;
  args::ValueFlag<std::string> photon_field;
  args::ValueFlag<std::string> nerf;

  args::ValueFlag<std::string> m_tfn;
  bool has_tfn() { return m_tfn; }
  std::string tfn() { return args::get(m_tfn); }

  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_from;
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_up;
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_at;
  vec3f camera_from() { return (m_camera_from) ? args::get(m_camera_from) : vec3f(0.f, 0.f, -1000.f); }
  vec3f camera_at()   { return (m_camera_at)   ? args::get(m_camera_at)   : vec3f(0.f, 0.f, 0.f);     }
  vec3f camera_up()   { return (m_camera_up)   ? args::get(m_camera_up)   : vec3f(0.f, 1.f, 0.f);     }
  bool  has_camera()  { return m_camera_from || m_camera_at || m_camera_up; }

  args::ValueFlag<float> m_sampling_rate;
  float sampling_rate() { return (m_sampling_rate) ? args::get(m_sampling_rate) : 1.f; }

  args::ValueFlag<float> m_density_scale;
  float density_scale() { return (m_density_scale) ? args::get(m_density_scale) : 1.f; }

  args::ValueFlag<int> m_rendering_mode;
  int rendering_mode() { return (m_rendering_mode) ? args::get(m_rendering_mode) : 0; }

  args::ValueFlag<std::string> m_output;
  std::string output() { return (m_output) ? args::get(m_output) : "output.png"; }

  args::ValueFlag<int> m_width;
  args::ValueFlag<int> m_height;
  int width()  { return (m_width)  ? args::get(m_width)  : 1024; }
  int height() { return (m_height) ? args::get(m_height) : 1024; }

  args::ValueFlag<int> m_spp;
  int spp() { return (m_spp) ? args::get(m_spp) : 1; }

  args::ValueFlag<int> m_warmup;
  int warmup() { return (m_warmup) ? args::get(m_warmup) : 5; }

  args::ValueFlag<float> m_param_time;
  args::ValueFlag<float> m_param_theta;
  args::ValueFlag<float> m_param_phi;
  float param_time()  { return (m_param_time)  ? args::get(m_param_time)  : 0.5f; }
  float param_theta() { return (m_param_theta) ? args::get(m_param_theta) : 0.0f; }
  float param_phi()   { return (m_param_phi)   ? args::get(m_param_phi)   : 0.0f; }

  args::Flag denoise;

public:
  CmdArgs(const char* title, int argc, char** argv)
    : parser(title)
    , help(parser, "help", "display the help menu", {'h', "help"})
    , group1(parser, "This group is all exclusive:", args::Group::Validators::Xor)
    , group2(parser, "This group is all inclusive:", args::Group::Validators::AllOrNone)
    //
    , m_simple_volume(group1, "filename", "the simple volume to render", {"simple-volume"})
    , m_neural_volume(group1, "filename", "the neural volume to render", {"neural-volume"})
    //
    , m_camera_from(group2, "vec3f", "camera position (from)", {"camera-from"})
    , m_camera_at(group2, "vec3f", "camera lookat target (at)", {"camera-at"})
    , m_camera_up(group2, "vec3f", "camera up vector", {"camera-up"})
    //
    , scalar_field(parser, "filename", "add a scalar field INR for rendering", {"scalarfield"})
    , shadow_field(parser, "filename", "add a shadow field for rendering", {"shadowfield"})
    , photon_field(parser, "filename", "add a photon field for rendering", {"photonfield"})
    , nerf(parser, "filename", "add a Neural Radiance Field for rendering", {"nerf"})
    //
    , m_tfn(parser, "filename", "the transfer function preset", {"tfn"})
    , m_sampling_rate(parser, "float", "ray marching sampling rate", {"sampling-rate"})
    , m_density_scale(parser, "float", "path tracing density scale", {"density-scale"})
    , m_rendering_mode(parser, "int", "rendering mode (0-4)", {"rendering-mode"})
    //
    , m_output(parser, "filename", "output image path (default: output.png)", {"output", 'o'})
    , m_width(parser, "int", "framebuffer width (default: 1024)", {"width"})
    , m_height(parser, "int", "framebuffer height (default: 1024)", {"height"})
    , m_spp(parser, "int", "samples per pixel / accumulation frames (default: 1)", {"spp"})
    , m_warmup(parser, "int", "warmup frames before timed rendering (default: 5)", {"warmup"})
    //
    , m_param_time(parser, "float", "HINR time parameter", {"param-time"})
    , m_param_theta(parser, "float", "HINR theta parameter", {"param-theta"})
    , m_param_phi(parser, "float", "HINR phi parameter", {"param-phi"})
    //
    , denoise(parser, "flag", "enable denoiser", {"denoise"})
  {
    exec(parser, argc, argv);
  }
};

static void
save_image(const std::string& fname, vec2i size, const vec4f* pixels)
{
  std::vector<unsigned char> image((uint64_t)size.x * size.y * 4);
  for (uint64_t i = 0; i < (uint64_t)size.x * size.y; ++i) {
    const auto in = pixels[i];
    image[4 * i + 0] = (unsigned char)(255.99f * clamp(in.x, 0.f, 1.f));
    image[4 * i + 1] = (unsigned char)(255.99f * clamp(in.y, 0.f, 1.f));
    image[4 * i + 2] = (unsigned char)(255.99f * clamp(in.z, 0.f, 1.f));
    image[4 * i + 3] = (unsigned char)(255.99f * clamp(in.w, 0.f, 1.f));
  }

  const std::string ext = fname.substr(fname.find_last_of('.') + 1);
  int success = 0;
  if (ext == "jpg" || ext == "jpeg") {
    success = stbi_write_jpg(fname.c_str(), size.x, size.y, 4, image.data(), 100);
  }
  else if (ext == "png") {
    success = stbi_write_png(fname.c_str(), size.x, size.y, 4, image.data(), size.x * 4);
  }
  else {
    std::cerr << "[batch] unsupported file extension: " << ext << " (use .png or .jpg)" << std::endl;
    return;
  }

  if (!success) {
    std::cerr << "[batch] failed to write: " << fname << std::endl;
  }
  else {
    std::cout << "[batch] saved: " << fname << " (" << size.x << "x" << size.y << ")" << std::endl;
  }
}

extern "C" int
main(int ac, char** av)
{
  CmdArgs args("Batch Volume Renderer (headless)", ac, av);

  vnrCompilationStatus("[vnr]");

  // -------------------------------------------------------
  // create volume
  // -------------------------------------------------------
  vnrVolume volume;
  if (args.has_simple_volume()) {
    volume = vnrCreateSimpleVolume(args.volume(), "GPU", false);
  }
  else {
    vnrJson params;
    vnrLoadJsonBinary(params, args.volume());
    volume = vnrCreateNeuralVolume(params);
  }

  // -------------------------------------------------------
  // create renderer
  // -------------------------------------------------------
  vnrRenderer renderer = vnrCreateRenderer(volume);

  if (args.scalar_field) {
    vnrRendererSetScalarField(renderer, vnrCreateJsonBinary(args::get(args.scalar_field)));
  }
  if (args.shadow_field) {
    vnrRendererSetShadowField(renderer, vnrCreateJsonBinary(args::get(args.shadow_field)));
  }
  if (args.photon_field) {
    vnrRendererSetPhotonField(renderer, vnrCreateJsonBinary(args::get(args.photon_field)));
  }
  if (args.nerf) {
    vnrRendererSetNeRF(renderer, vnrCreateJsonBinary(args::get(args.nerf)));
  }

  // -------------------------------------------------------
  // create & configure camera
  // -------------------------------------------------------
  Camera camera = { args.camera_from(), args.camera_at(), args.camera_up() };

  vnrCamera cam = vnrCreateCamera();
  vnrCameraSet(cam, camera.from, camera.at, camera.up);

  if (args.has_simple_volume()) {
    vnrCameraSet(cam, args.volume());
  }
  if (args.has_tfn()) {
    vnrCameraSet(cam, args.tfn());
  }

  if (args.has_camera()) {
    vnrCameraSet(cam, camera.from, camera.at, camera.up);
  }

  vnrRendererSetCamera(renderer, cam);

  std::cout << "[batch] camera from: " << vnrCameraGetPosition(cam) << std::endl;
  std::cout << "[batch] camera at:   " << vnrCameraGetFocus(cam) << std::endl;
  std::cout << "[batch] camera up:   " << vnrCameraGetUpVec(cam) << std::endl;

  // -------------------------------------------------------
  // create & configure transfer function
  // -------------------------------------------------------
  vnrTransferFunction tfn;
  if (args.has_tfn()) {
    tfn = vnrCreateTransferFunction(args.tfn());
  }
  else {
    tfn = vnrCreateTransferFunction();
  }
  vnrTransferFunctionSetValueRange(tfn, range1f(0, 1));
  vnrRendererSetTransferFunction(renderer, tfn);

  // -------------------------------------------------------
  // configure renderer params
  // -------------------------------------------------------
  const vec2i fb_size(args.width(), args.height());
  vnrRendererSetFramebufferSize(renderer, fb_size);
  vnrRendererSetVolumeSamplingRate(renderer, args.sampling_rate());
  vnrRendererSetVolumeDensityScale(renderer, args.density_scale());
  vnrRendererSetMode(renderer, args.rendering_mode());
  vnrRendererSetDenoiser(renderer, args.denoise);

  if (args.m_param_time) {
    vnrRendererSetParams(renderer, args.param_time());
  }
  if (args.m_param_theta || args.m_param_phi) {
    vnrRendererSetParams(renderer, args.param_theta(), args.param_phi());
  }

  CUDA_SYNC_CHECK();

  // -------------------------------------------------------
  // warmup
  // -------------------------------------------------------
  const int warmup_frames = args.warmup();
  if (warmup_frames > 0) {
    std::cout << "[batch] warming up (" << warmup_frames << " frames) ..." << std::endl;
    vnrRendererResetAccumulation(renderer);
    for (int i = 0; i < warmup_frames; ++i) {
      vnrRender(renderer);
    }
    CUDA_SYNC_CHECK();
    std::cout << "[batch] warmup done." << std::endl;
  }

  // -------------------------------------------------------
  // render
  // -------------------------------------------------------
  const int spp = args.spp();
  std::cout << "[batch] rendering " << fb_size.x << "x" << fb_size.y
            << " with " << spp << " spp ..." << std::endl;

  vnrRendererResetAccumulation(renderer);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < spp; ++i) {
    vnrRender(renderer);
  }

  CUDA_SYNC_CHECK();

  double elapsed = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now() - t0).count();
  double fps = spp / elapsed;
  std::cout << "[batch] rendering time: " << elapsed << " s ("
            << spp << " frames, " << (elapsed / spp) << " s/frame, "
            << std::fixed << std::setprecision(2) << fps << " fps)" << std::endl;

  // -------------------------------------------------------
  // save image
  // -------------------------------------------------------
  vec4f* pixels = vnrRendererMapFrame(renderer);
  save_image(args.output(), fb_size, pixels);

  vnrMemoryQueryPrint("[vnr] memory");

  return 0;
}
