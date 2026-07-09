//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "siren_network.h"
#include "siren_impl.cuh"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/trainer.h>

#include <limits>

#ifdef ENABLE_LOGGING
#define logging() std::cout
#else
static std::ostream null_output_stream(0);
#define logging() null_output_stream
#endif

/* ------------------------------------------------------------------ */
/* namespace instant neural volume details                            */
/* ------------------------------------------------------------------ */
using precision_t = TCNN_NAMESPACE ::network_precision_t;

namespace vnr
{
  namespace siren_impl
  {

    using Loss = TCNN_NAMESPACE ::Loss<precision_t>;
    using Optimizer = TCNN_NAMESPACE ::Optimizer<precision_t>;
    using Trainer = TCNN_NAMESPACE ::Trainer<float, precision_t, precision_t>;
    using Encoding = TCNN_NAMESPACE ::Encoding<precision_t>;
    using ForwardContext = Trainer ::ForwardContext;
    using TCNN_NAMESPACE ::create_encoding;
    using TCNN_NAMESPACE ::create_loss;
    using TCNN_NAMESPACE ::create_optimizer;

    NetworkMode network_mode_from_json(const json& network_opts) {
      const std::string mode = network_opts.value("mode", std::string{"PlainMLP"});
      if (mode == "PlainMLP") {
        return NetworkMode::PlainMLP;
      }
      if (mode == "NeurCompResidual") {
        return NetworkMode::NeurCompResidual;
      }
      throw std::runtime_error{fmt::format("SirenNetwork: unsupported mode '{}'.", mode)};
    }

    const json identity_encoding_opts = {{"otype", "Identity"}};
    const json dummy_network_opts = {{"otype", "FullyFusedMLP"}, {"n_hidden_layers", 2}, {"activation", "Sine"}, {"output_activation", "None"}};

    using NetworkImpl = TCNN_NAMESPACE ::NetworkWithInputEncoding<precision_t>;

    class SirenNetworkImpl : public NetworkImpl
    {
    public:
      SirenNetworkImpl(uint32_t n_input_dims, uint32_t n_output_dims, const json &network_opts)
          : NetworkImpl(n_input_dims, n_output_dims, identity_encoding_opts, dummy_network_opts)
      {
        const int n_neurons = network_opts.value("n_neurons", 32);
        const float omega_0 = network_opts.value("omega_0", 30.0f);
        const bool use_bias = network_opts.value("bias", false);
        const uint32_t n_hidden_layers = network_opts.value("n_hidden_layers", 4u);
        const NetworkMode mode = network_mode_from_json(network_opts);
        // Match ``NetworkWithInputEncoding``: inner network input width is identity encoding output after alignment.
        const uint32_t network_input_width = m_encoding->padded_output_width();
        switch (n_neurons)
        {
        case 16:
          NetworkImpl::m_network.reset(new FuyllyFusedSIREN<precision_t, 16>(
              network_input_width, n_output_dims, n_hidden_layers, Activation::SirenSine, Activation::None, omega_0, use_bias, mode));
          break;
        case 32:
          NetworkImpl::m_network.reset(new FuyllyFusedSIREN<precision_t, 32>(
              network_input_width, n_output_dims, n_hidden_layers, Activation::SirenSine, Activation::None, omega_0, use_bias, mode));
          break;
        case 64:
          NetworkImpl::m_network.reset(new FuyllyFusedSIREN<precision_t, 64>(
              network_input_width, n_output_dims, n_hidden_layers, Activation::SirenSine, Activation::None, omega_0, use_bias, mode));
          break;
        case 128:
          NetworkImpl::m_network.reset(new FuyllyFusedSIREN<precision_t, 128>(
              network_input_width, n_output_dims, n_hidden_layers, Activation::SirenSine, Activation::None, omega_0, use_bias, mode));
          break;
        default:
          throw std::runtime_error{
              fmt::format("SirenNetwork: n_neurons={} is unsupported for the fused WMMA path. Supported widths: 16, 32, 64, 128.", n_neurons)};
        }
      }
    };

  } // namespace siren_impl
} // namespace vnr

/* ------------------------------------------------------------------ */
/* namespace instant neural volume                                    */
/* ------------------------------------------------------------------ */
namespace vnr
{
  using namespace siren_impl;

  struct SirenNetwork::Impl
  {
    int input_dims = 0;
    int output_dims = 0;
    int n_neurons = -1;
    int n_features_per_level = -1;

    mutable std::shared_ptr<Loss> loss;
    mutable std::shared_ptr<Optimizer> optimizer;
    mutable std::shared_ptr<SirenNetworkImpl> network;
    mutable std::shared_ptr<Trainer> trainer;
    std::unique_ptr<ForwardContext> ctx;
    json model_opts;
    float omega_0 = 30;
  };

  SirenNetwork::~SirenNetwork() {}

  SirenNetwork::SirenNetwork(int n_input_dims, int n_output_dims)
      : m(new Impl())
  {
    m->input_dims = n_input_dims;
    m->output_dims = n_output_dims;
  }

  int SirenNetwork::n_input_dims() const
  {
    return m->input_dims;
  }

  int SirenNetwork::n_output_dims() const
  {
    return m->output_dims;
  }

  int SirenNetwork::n_neurons() const
  {
    return m->n_neurons;
  }

  int SirenNetwork::n_features_per_level() const
  {
    return m->n_features_per_level;
  }

  bool SirenNetwork::valid() const
  {
    return m->trainer.get() != nullptr;
  }

  size_t SirenNetwork::get_model_size() const
  {
    if (!m->network)
    {
      return 0;
    }
    return sizeof(precision_t) * m->network->n_params();
  }

  size_t SirenNetwork::get_mlp_size() const
  {
    return get_model_size();
  }

  size_t SirenNetwork::get_enc_size() const
  {
    return 0;
  }

  size_t SirenNetwork::training_step() const
  {
    return 0;
  }

  double SirenNetwork::training_loss() const
  {
    if (!m->trainer || !m->ctx)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return m->trainer->loss(0, *m->ctx);
  }

  json SirenNetwork::serialize_params() const
  {
    if (!m->trainer)
    {
      return json::object();
    }
    return m->trainer->serialize();
  }

  void SirenNetwork::deserialize_params(const json &parameters)
  {
    if (!m->trainer)
    {
      throw std::runtime_error{"SirenNetwork::deserialize_params: network not initialized"};
    }
    m->trainer->deserialize(parameters);
  }

  json SirenNetwork::serialize_model() const
  {
    return m->model_opts;
  }

  void SirenNetwork::deserialize_model(const json &config)
  {
    TRACE_CUDA;

    json loss_opts = config.value("loss", json::object());
    json optimizer_opts = config.value("optimizer", json::object());
    json network_opts = config.value("network", json::object());
    if (!network_opts.contains("otype"))
    {
      network_opts["otype"] = "FuyllyFusedSIREN";
    }
    network_opts["mode"] = network_opts.value("mode", "PlainMLP");
    network_opts["bias"] = network_opts.value("bias", false);

    m->model_opts["loss"] = loss_opts;
    m->model_opts["optimizer"] = optimizer_opts;
    m->model_opts["network"] = network_opts;

    m->n_neurons = -1;
    m->n_features_per_level = -1;

    m->loss.reset();
    m->optimizer.reset();
    m->network.reset();
    m->trainer.reset();

    TRACE_CUDA;

    try
    {
      m->loss = std::shared_ptr<Loss>{create_loss<precision_t>(loss_opts)};
      m->optimizer = std::shared_ptr<Optimizer>{create_optimizer<precision_t>(optimizer_opts)};
      m->network = std::make_shared<SirenNetworkImpl>(m->input_dims, m->output_dims, network_opts);
      m->trainer = std::make_shared<Trainer>(m->network, m->optimizer, m->loss, (uint32_t)time(NULL));
      m->omega_0 = network_opts.value("omega_0", 30.0f);
      m->n_neurons = network_opts.value("n_neurons", 32);
      m->n_features_per_level = 0;
    }
    catch (std::runtime_error &e)
    {
      std::cerr << e.what() << std::endl;
    }

    if (m->network)
    {
      logging() << "[network] total # of parameters = " << m->network->n_params() << std::endl;
    }

    TRACE_CUDA;
  }

  void SirenNetwork::train(const GPUColumnMatrix &input, const GPUColumnMatrix &target, cudaStream_t stream)
  {
    (void)input;
    (void)target;
    (void)stream;
    TRACE_CUDA;
    throw std::runtime_error("SirenNetwork: training is not supported (inference-only)");
    TRACE_CUDA;
  }

  void SirenNetwork::infer(const GPUMatrixDynamic<float> &input, GPUMatrixDynamic<float> &output, cudaStream_t stream) const
  {
    TRACE_CUDA;
    try
    {
      m->network->inference(stream, input, output);
    }
    catch (std::runtime_error &e)
    {
      std::cerr << e.what() << std::endl;
      m->loss.reset();
      m->optimizer.reset();
      m->network.reset();
      m->trainer.reset();
      return;
    }
    TRACE_CUDA;
  }

} // namespace vnr
