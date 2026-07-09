#pragma once
#ifndef OVR_IVNR_DEVICE_H
#define OVR_IVNR_DEVICE_H

#include <ovr/renderer.h>

#include "array.h"
#include "framebuffer.h"
#include "renderer.h"
#include "serializer.h"

#include <memory>

namespace ovr::hyperinr {

using FrameBuffer  = DoubleBufferObject<vec4f>;

class DeviceHyperINR : public MainRenderer {
public:
  /*! constructor - performs all setup, including initializing ospray, creates scene graph, etc. */
  void init(int argc, const char** argv) override;

  /*! render one frame */
  void swap() override;
  void commit() override;
  void render() override;
  void mapframe(FrameBufferData* fb) override;

  /*! control device specific UIs */
#ifdef ENABLE_OPENGL
  void ui(ImGuiContext* context) override;
#endif

private:
  void commit_material();
  void commit_lighting();

  template<typename T>
  bool check(TransactionalValue<T>& ctl) {
    if (ctl.update()) { dirty = true; return true; }
    return false;
  }

private:
  bool initialized = false;
  bool dirty = true;

  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  bool framebuffer_size_updated{ false };

  Camera camera;

  StructuredRegularVolume volume;
  bool volume_changed{ true };

  int rendering_mode{ VNR_RAYMARCHING };
  RenderObject api;
  LaunchParams &lp = api.params;

  struct {
    TransactionalValue<float> ambient;
    TransactionalValue<float> diffuse;
    TransactionalValue<float> specular;
    TransactionalValue<float> shininess;
    TransactionalValue<float> phi;
    TransactionalValue<float> theta;
    TransactionalValue<float> intensity;
  } ctls;
};

} // namespace ovr::hyperinr

#endif // OVR_IVNR_DEVICE_H
