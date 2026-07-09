#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <json/json.hpp>

#include <memory>

#include "../instantvnr_types.h"

namespace vnr {

/// JSON type used by the network serialization APIs.
using json = nlohmann::json;

using TCNN_NAMESPACE::GPUMatrix;
using TCNN_NAMESPACE::GPUMatrixDynamic;
using TCNN_NAMESPACE::GPUMemory;

/// Column-major matrix used for batched tensors where each column is a sample.
using GPUColumnMatrix =
    TCNN_NAMESPACE::GPUMatrix<float, TCNN_NAMESPACE::MatrixLayout::ColumnMajor>;

/// Row-major matrix alias kept for APIs that need row-major storage.
using GPURowMatrix =
    TCNN_NAMESPACE::GPUMatrix<float, TCNN_NAMESPACE::MatrixLayout::RowMajor>;

using TCNN_NAMESPACE::gpu_memory_to_json_binary;
using TCNN_NAMESPACE::json_binary_to_gpu_memory;

/// Common interface for GPU-backed neural network implementations used by VNR.
///
/// Concrete implementations own the model state, training buffers, and any
/// backend-specific resources. The interface exposes:
/// - static model metadata such as input/output dimensionality
/// - training progress and loss reporting
/// - serialization helpers for weights and configuration
/// - train / inference entry points operating on GPU-resident matrices
struct AbstractNetwork {
  /// Polymorphic base destructor.
  virtual ~AbstractNetwork() {}

  /// Returns the dimensionality expected for each input sample.
  virtual int n_input_dims() const = 0;

  /// Returns the dimensionality produced for each output sample.
  virtual int n_output_dims() const = 0;

  /// Returns the network width, when the backend exposes one.
  virtual int n_neurons() const = 0;

  /// Returns the number of features produced per encoding level.
  virtual int n_features_per_level() const = 0;

  /// Indicates whether the network has been initialized successfully.
  virtual bool valid() const = 0;

  /// Returns the total serialized model size in bytes.
  virtual size_t get_model_size() const = 0;

  /// Returns the size of the MLP portion of the model in bytes.
  virtual size_t get_mlp_size() const = 0;

  /// Returns the size of the encoding portion of the model in bytes.
  virtual size_t get_enc_size() const = 0;

  /// Returns the number of training updates completed so far.
  virtual size_t training_step() const = 0;

  /// Returns the most recently observed training loss value.
  virtual double training_loss() const = 0;

  /// Serializes learnable parameters only.
  virtual json serialize_params() const = 0;

  /// Restores learnable parameters from a JSON payload.
  virtual void deserialize_params(const json& parameters) = 0;

  /// Serializes the full model configuration and state.
  virtual json serialize_model() const = 0;

  /// Restores the full model configuration and state.
  virtual void deserialize_model(const json& config) = 0;

  /// Runs one training step on GPU-resident batched data.
  ///
  /// `input` and `target` are expected to be column-major batched matrices,
  /// and `stream` selects the CUDA stream used by the implementation.
  virtual void train(
      const GPUColumnMatrix& input,
      const GPUColumnMatrix& target,
      cudaStream_t stream
  ) = 0;

  /// Runs inference on GPU-resident input data and writes results to `output`.
  virtual void infer(
      const GPUMatrixDynamic<float>& input,
      GPUMatrixDynamic<float>& output,
      cudaStream_t stream
  ) const = 0;
};

}  // namespace vnr
