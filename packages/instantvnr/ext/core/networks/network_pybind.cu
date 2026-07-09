//. ======================================================================== //
//. Python bindings for vnr::AbstractNetwork / vnr::TcnnNetwork               //
//. Must be compiled as CUDA (.cu): nnapi.h / tcnn_network.h pull in TCNN and //
//. CUDA types that are not supported in a plain C++ TU.                       //
//. Python module name: ``TORCH_EXTENSION_NAME`` (set by CMake ``-D``, same pattern as //
//. PyTorch/tiny-cuda-nn torch bindings).                                      //
//. ======================================================================== //

#ifndef TORCH_EXTENSION_NAME
#error "TORCH_EXTENSION_NAME must be defined by the build (e.g. -DTORCH_EXTENSION_NAME=vnr_network)"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nnapi.h"
#include "siren_network.h"
#include "tcnn_network.h"

#include <cuda_runtime.h>

namespace py = pybind11;
using namespace vnr;
using json = nlohmann::json;

namespace
{

  json py_to_json(const py::object &o)
  {
    if (o.is_none())
      return json{};
    py::module_ jm = py::module_::import("json");
    std::string s = py::cast<std::string>(jm.attr("dumps")(o));
    return json::parse(s);
  }

  py::object json_to_py(const json &j)
  {
    py::module_ jm = py::module_::import("json");
    return jm.attr("loads")(j.dump());
  }

  void train_device(AbstractNetwork &self, uintptr_t in_ptr, uintptr_t tgt_ptr, uint32_t batch,
                    uintptr_t stream_ptr)
  {
    const int in_d = self.n_input_dims();
    const int out_d = self.n_output_dims();
    GPUColumnMatrix input((float *)in_ptr, in_d, batch);
    GPUColumnMatrix target((float *)tgt_ptr, out_d, batch);
    self.train(input, target, (cudaStream_t)stream_ptr);
  }

  void infer_device(AbstractNetwork &self, uintptr_t in_ptr, uintptr_t out_ptr, uint32_t batch,
                    uintptr_t stream_ptr)
  {
    const int in_d = self.n_input_dims();
    const int out_d = self.n_output_dims();
    GPUColumnMatrix input((float *)in_ptr, in_d, batch);
    GPUColumnMatrix output((float *)out_ptr, out_d, batch);
    self.infer(input, output, (cudaStream_t)stream_ptr);
  }

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "AbstractNetwork, TcnnNetwork (tiny-cuda-nn), and SirenNetwork bindings for the VNR render core.";

  py::class_<AbstractNetwork, std::shared_ptr<AbstractNetwork>>(m, "AbstractNetwork")
      .def("n_input_dims", &AbstractNetwork::n_input_dims)
      .def("n_output_dims", &AbstractNetwork::n_output_dims)
      .def("n_neurons", &AbstractNetwork::n_neurons)
      .def("n_features_per_level", &AbstractNetwork::n_features_per_level)
      .def("valid", &AbstractNetwork::valid)
      .def("get_model_size", &AbstractNetwork::get_model_size)
      .def("get_mlp_size", &AbstractNetwork::get_mlp_size)
      .def("get_enc_size", &AbstractNetwork::get_enc_size)
      .def("training_step", &AbstractNetwork::training_step)
      .def("training_loss", &AbstractNetwork::training_loss)
      .def(
          "serialize_params",
          [](const AbstractNetwork &self)
          { return json_to_py(self.serialize_params()); },
          "Return trainer parameter blob as a JSON-serializable Python object (same schema as tiny-cuda-nn).")
      .def(
          "deserialize_params", [](AbstractNetwork &self, const py::object &p)
          { self.deserialize_params(py_to_json(p)); },
          py::arg("parameters"))
      .def(
          "serialize_model", [](const AbstractNetwork &self)
          { return json_to_py(self.serialize_model()); },
          "Return the last loaded model JSON (loss/encoding/network/optimizer sections).")
      .def(
          "deserialize_model", [](AbstractNetwork &self, const py::object &cfg)
          { self.deserialize_model(py_to_json(cfg)); },
          py::arg("config"))
      .def(
          "train", train_device, py::arg("input_ptr"), py::arg("target_ptr"), py::arg("batch_size"),
          py::arg("stream_ptr") = 0,
          "Train one step. ``input_ptr`` / ``target_ptr`` are device pointers (e.g. ``tensor.data_ptr()``) "
          "to column-major batches of shape (n_input_dims, batch) and (n_output_dims, batch).")
      .def(
          "infer", infer_device, py::arg("input_ptr"), py::arg("output_ptr"), py::arg("batch_size"),
          py::arg("stream_ptr") = 0,
          "Inference. Output buffer must be pre-allocated on device.");

  py::class_<TcnnNetwork, AbstractNetwork, std::shared_ptr<TcnnNetwork>>(m, "TcnnNetwork")
      .def(py::init<int, int>(), py::arg("n_input_dims"), py::arg("n_output_dims"));

  py::class_<SirenNetwork, AbstractNetwork, std::shared_ptr<SirenNetwork>>(m, "SirenNetwork")
      .def(py::init<int, int>(), py::arg("n_input_dims"), py::arg("n_output_dims"));
}
