#pragma once

#include "nnapi.h"

namespace vnr {

struct SirenNetwork : AbstractNetwork {
private:
  struct Impl;
  std::unique_ptr<Impl> m;

public:
  ~SirenNetwork() override;
  SirenNetwork(int n_input_dims, int n_output_dims);
  SirenNetwork(const SirenNetwork& other) = delete;
  SirenNetwork(SirenNetwork&& other) noexcept = default;
  SirenNetwork& operator=(const SirenNetwork& other) = delete;
  SirenNetwork& operator=(SirenNetwork&& other) noexcept = default;

  int n_input_dims() const override;
  int n_output_dims() const override;
  int n_neurons() const override;
  int n_features_per_level() const override;
  bool valid() const override;
  size_t get_model_size() const override;
  size_t get_mlp_size() const override;
  size_t get_enc_size() const override;
  size_t training_step() const override;
  double training_loss() const override;
  json serialize_params() const override;
  void deserialize_params(const json& parameters) override;
  json serialize_model() const override;
  void deserialize_model(const json& config) override;

  void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) override;
  void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const override;
};

}
