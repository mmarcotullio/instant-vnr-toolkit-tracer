/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   fully_fused_mlp.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Fully fused CUDA implementation of a multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */
#pragma once

#include "nnapi.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

#include <algorithm>
#include <mma.h>

/* namespace instant neural volume */
namespace vnr {
namespace siren_impl {

/** Global per-device constant for SirenSine; upload via sync_siren_omega_0_to_device (last writer wins if multiple networks). */
__device__ __constant__ float siren_omega_0;

inline __host__ void sync_siren_omega_0_to_device(float omega) {
	CUDA_CHECK_THROW(cudaMemcpyToSymbol(siren_omega_0, &omega, sizeof(float), 0, cudaMemcpyHostToDevice));
}

enum class Activation {
	SirenSine   = 100,
	ReLU        = (int)TCNN_NAMESPACE :: Activation::ReLU,
	Exponential = (int)TCNN_NAMESPACE :: Activation::Exponential,
	Sine        = (int)TCNN_NAMESPACE :: Activation::Sine,
	Sigmoid     = (int)TCNN_NAMESPACE :: Activation::Sigmoid,
	Squareplus  = (int)TCNN_NAMESPACE :: Activation::Squareplus,
	Softplus    = (int)TCNN_NAMESPACE :: Activation::Softplus,
	None        = (int)TCNN_NAMESPACE :: Activation::None,
};

enum class NetworkMode {
	PlainMLP = 0,
	NeurCompResidual = 1,
};

// type aliases //
using pcg32          = TCNN_NAMESPACE :: pcg32;
using Context        = TCNN_NAMESPACE :: Context;
using EGradientMode  = TCNN_NAMESPACE :: EGradientMode;
using GPUMemoryArena = TCNN_NAMESPACE :: GPUMemoryArena;
using GPUMatrixBase  = TCNN_NAMESPACE :: GPUMatrixBase;
using MatrixLayout   = TCNN_NAMESPACE :: MatrixLayout;
constexpr MatrixLayout RM  = MatrixLayout::RowMajor;
constexpr MatrixLayout SoA = MatrixLayout::SoA;
constexpr MatrixLayout CM  = MatrixLayout::ColumnMajor;
constexpr MatrixLayout AoS = MatrixLayout::AoS;

// template aliases //
using TCNN_NAMESPACE :: Network;
using TCNN_NAMESPACE :: GPUMatrix;

// function aliases //
using TCNN_NAMESPACE :: div_round_up;
using TCNN_NAMESPACE :: n_blocks_linear;
using TCNN_NAMESPACE :: n_threads_linear;
inline std::string to_string(Activation activation) 
{
	switch (activation) {
		case Activation::SirenSine: return "SirenSine";
		default: return TCNN_NAMESPACE :: to_string((TCNN_NAMESPACE :: Activation)activation);
	}
}

inline std::string to_string(NetworkMode mode) {
	switch (mode) {
		case NetworkMode::PlainMLP: return "PlainMLP";
		case NetworkMode::NeurCompResidual: return "NeurCompResidual";
		default: throw std::runtime_error{"Unsupported network mode."};
	}
}



// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
template <typename T, int WIDTH>
class FuyllyFusedSIREN : public Network<T> {
public:
	FuyllyFusedSIREN(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers, Activation activation, Activation output_activation, float omega_0, bool use_bias, NetworkMode mode);

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override;

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override;

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		TCNN_NAMESPACE :: EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override 
	{
		throw std::runtime_error("FuyllyFusedSIREN: backward_impl is not implemented");
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override;
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;

	GPUMatrix<T, RM>& input_weight_matrix(bool inference) {
		auto& weight_matrices = m_weight_matrices;
		return weight_matrices.front();
	}

	GPUMatrix<T, RM>& weight_matrix_at(bool inference, uint32_t idx) {
		auto& weight_matrices = m_weight_matrices;
		return weight_matrices.at(1 + idx);
	}

	GPUMatrix<T, RM>& output_weight_matrix(bool inference) {
		auto& weight_matrices = m_weight_matrices;
		return weight_matrices.back();
	}

	GPUMatrix<T, RM>& input_bias(bool inference) {
		auto& biases = m_biases;
		return biases.front();
	}

	GPUMatrix<T, RM>& bias_at(bool inference, uint32_t idx) {
		auto& biases = m_biases;
		return biases.at(1 + idx);
	}

	GPUMatrix<T, RM>& output_bias(bool inference) {
		auto& biases = m_biases;
		return biases.back();
	}

	GPUMatrix<T, RM>& weight_matrix(size_t idx) {
		return m_weight_matrices.at(idx);
	}

	GPUMatrix<T, RM>& bias_matrix(size_t idx) {
		return m_biases.at(idx);
	}

	GPUMatrix<T, RM>& input_gradient_matrix() {
		return dummy_matrix;
	}

	GPUMatrix<T, RM>& gradient_matrix_at(uint32_t idx) {
		return dummy_matrix;
	}

	GPUMatrix<T, RM>& output_gradient_matrix() {
		return dummy_matrix;
	}

	size_t n_params() const override {
		return m_total_n_params;
	}

	uint32_t input_width() const override {
		return m_input_width;
	}

	uint32_t padded_output_width() const override {
		return m_padded_output_width;
	}

	uint32_t output_width() const override {
		return m_output_width;
	}

	static uint32_t REQUIRED_ALIGNMENT() {
		return 16; // Uses 16x16x16 tensor ops
	}

	uint32_t required_input_alignment() const override {
		return REQUIRED_ALIGNMENT();
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		std::vector<std::pair<uint32_t, uint32_t>> result;
		for (auto& matrix : m_weight_matrices) {
			result.emplace_back(matrix.m(), matrix.n());
		}
		return result;
	}

	uint32_t width(uint32_t layer) const override {
		return WIDTH;
	}

	uint32_t num_forward_activations() const override {
		return m_mode == NetworkMode::PlainMLP ? m_n_hidden_layers : (m_n_hidden_matmuls + 1);
	}

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		return {forward.hidden.at(layer).data(), CM};
	}

	json hyperparams() const override {
		return {
			{"otype", "FuyllyFusedSIREN"},
			{"mode", to_string(m_mode)},
			{"activation", to_string(m_activation)},
			{"output_activation", to_string(m_output_activation)},
			{"n_neurons", m_network_width},
			{"n_hidden_layers", m_n_hidden_layers},
			{"bias", m_use_bias},
		};
	}

public:
	struct ForwardContext : public Context {
		std::vector<GPUMatrix<T>> hidden;
		GPUMemoryArena::Allocation alloc;
	};

	std::unique_ptr<ForwardContext> allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size);

	uint32_t m_n_hidden_layers;
	uint32_t m_n_hidden_matmuls;
	uint32_t m_input_width;
	uint32_t m_network_width;
	uint32_t m_output_width;
	uint32_t m_padded_output_width;
	float m_omega_0;
	Activation m_activation;
	Activation m_output_activation;
	bool m_use_bias;
	NetworkMode m_mode;

	// Storage of params
	std::vector<GPUMatrix<T, RM>> m_weight_matrices;
	std::vector<GPUMatrix<T, RM>> m_biases;
	size_t m_total_n_params;

	GPUMatrix<T, RM> dummy_matrix;
};

void check_shmem_error(cudaError_t error) {
	if (error != cudaSuccess) {
		throw std::runtime_error{"FuyllyFusedSIREN: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead."};
	}
}

/** SirenSine uses siren_omega_0; else TCNN_NAMESPACE::warp_activation. */
template <typename T, typename fragment_t>
__device__ inline void warp_activation(Activation activation, const fragment_t& frag, fragment_t& result) {
	if (activation == Activation::SirenSine) {
		const float w = siren_omega_0;
		TCNN_PRAGMA_UNROLL
		for (int t = 0; t < result.num_elements; ++t) {
			result.x[t] = (T)(sinf(w * (float)frag.x[t]));
		}
		return;
	}
	TCNN_NAMESPACE::warp_activation<T>((TCNN_NAMESPACE::Activation)((int)activation), frag, result);
}

template <int WIDTH>
__device__ void prepare_bias_broadcast_tile_in_shmem(
	__half* __restrict__ bias_broadcast_tile_shmem,
	const __half* __restrict__ bias_this_layer,
	const uint32_t bias_col_offset
) {
	if (!bias_this_layer) {
		return;
	}

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	const uint32_t li = threadIdx.x;
	const uint32_t lane_offset = (8 * li) % 16;
	const uint32_t row = (8 * li) / 16;
	__half* tile_ptr = bias_broadcast_tile_shmem + bias_col_offset + row * (WIDTH + SKEW) + lane_offset;

	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < 8; ++i) {
		tile_ptr[i] = bias_this_layer[bias_col_offset + lane_offset + i];
	}
}

template <int WIDTH, typename OUT_T>
__device__ void load_bias_fragment(
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T>& result_frag,
	const __half* __restrict__ bias_broadcast_tile_shmem,
	const bool has_bias,
	const uint32_t bias_col_offset
) {
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	if (!has_bias) {
		nvcuda::wmma::fill_fragment(result_frag, 0.0f);
		return;
	}

	nvcuda::wmma::load_matrix_sync(result_frag, bias_broadcast_tile_shmem + bias_col_offset, WIDTH + SKEW, nvcuda::wmma::mem_row_major);
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_copy_static(__half* __restrict__ dst_shmem, const __half* __restrict__ src_shmem) {
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	const uint32_t li = threadIdx.x;
	const uint32_t wi = threadIdx.y;
	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		*(int4*)&dst_shmem[lane_offset + (row + 16 * l) * (WIDTH + SKEW)] =
			*(const int4*)&src_shmem[lane_offset + (row + 16 * l) * (WIDTH + SKEW)];
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_scale_static(__half* __restrict__ shmem, float scale) {
	if (scale == 1.0f) {
		return;
	}

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	const __half scale_h = __float2half(scale);

	const uint32_t li = threadIdx.x;
	const uint32_t wi = threadIdx.y;
	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		__half* ptr = shmem + lane_offset + (row + 16 * l) * (WIDTH + SKEW);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 8; ++i) {
			ptr[i] = __hmul(ptr[i], scale_h);
		}
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_residual_add_static(__half* __restrict__ shmem, const __half* __restrict__ residual_shmem, float output_scale) {
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	const __half output_scale_h = __float2half(output_scale);

	const uint32_t li = threadIdx.x;
	const uint32_t wi = threadIdx.y;
	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		__half* ptr = shmem + lane_offset + (row + 16 * l) * (WIDTH + SKEW);
		const __half* residual_ptr = residual_shmem + lane_offset + (row + 16 * l) * (WIDTH + SKEW);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 8; ++i) {
			ptr[i] = __hmul(__hadd(ptr[i], residual_ptr[i]), output_scale_h);
		}
	}
}

template <typename T>
__global__ void kernel_scale_shift(const uint32_t n_elements, const float scale, const float bias, const T* __restrict__ input, T* __restrict__ output) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	output[i] = (T)((float)input[i] * scale + bias);
}

template <int WIDTH, int N_ITERS, typename OUT_T>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, const __half* __restrict__ bias_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;
	__half* __restrict__ bias_broadcast_tile_shmem = act_shmem + N_ITERS * 16 * (WIDTH + SKEW);
	prepare_bias_broadcast_tile_in_shmem<WIDTH>(bias_broadcast_tile_shmem, bias_this_layer, weights_col);

	__syncthreads();

	// Load N_BLOCKS chunks of weights from global memory into registers.
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_BLOCKS; ++i) {
		wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
	}

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		load_bias_fragment<WIDTH>(result_frag[l], bias_broadcast_tile_shmem, bias_this_layer != nullptr, weights_col);

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * l) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
		}

		warp_activation<__half>(activation, result_frag[l], result_frag[l]);
	}

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + l * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		TCNN_PRAGMA_UNROLL
		for (int l = 0; l < N_ITERS; ++l) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * l) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * l) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
	// act_shmem will be filled by the thread block's chunk of input_threadblock

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	TCNN_PRAGMA_UNROLL
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * i) * WIDTH];
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, const __half* __restrict__ bias_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width, const uint32_t batch_size) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// input_threadblock points to the thread block's chunk of the input batch in global memory
	// weights_this_layer points to the weight matrix of the current layer
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// in_width is the dynamic width of the input layer

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__half* __restrict__ weights_shmem = act_shmem + 16 * (in_width + INPUT_SKEW);
	__half* __restrict__ bias_broadcast_tile_shmem = weights_shmem + WIDTH * (in_width + INPUT_SKEW);
	prepare_bias_broadcast_tile_in_shmem<WIDTH>(bias_broadcast_tile_shmem, bias_this_layer, weights_col);

	// Load input weight matrix (fits completely into shared memory)
	// Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS warps
	const uint32_t n_elems_per_load = N_BLOCKS * 32 * 8;
	const uint32_t thread_elem_idx = (li + wi * 32) * 8;

	const uint32_t n_elems_b = WIDTH * in_width;

	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
		const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
		*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
	}

	const uint32_t n_tensor_ops = in_width / 16;

	if (std::is_same<INPUT_LAYOUT, wmma::col_major>::value) {
		__syncthreads();
	}

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
			// Load chunk of inputs into shmem.
			// This is faster than loading it from gmem directly, even though it is only used once.
			// (Possibly due to latency hiding through staging.)
			const uint32_t n_elems_a = 16 * in_width;

			TCNN_PRAGMA_UNROLL
			for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
				const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
				*(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
			}

			__syncthreads();
		}

		load_bias_fragment<WIDTH>(result_frag[l], bias_broadcast_tile_shmem, bias_this_layer != nullptr, weights_col);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < n_tensor_ops; ++i) {
			// Load chunk of inputs and weights from shared memory and multiply them
			if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
				wmma::load_matrix_sync(act_frag, act_shmem + 16 * i, in_width + INPUT_SKEW);
			} else {
				wmma::load_matrix_sync(act_frag, input_threadblock + 16 * i * batch_size + 16 * l, batch_size);
			}
			wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
		}

		if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
			__syncthreads();
		}

		warp_activation<__half>(activation, result_frag[l], result_frag[l]);

	}

	if (std::is_same<INPUT_LAYOUT, wmma::col_major>::value) {
		__syncthreads();
	}

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + (16 * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		TCNN_PRAGMA_UNROLL
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, const __half* __restrict__ bias_this_layer, OUT_T* __restrict__ out, const uint32_t output_stride, const nvcuda::wmma::layout_t output_layout) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// weights_this_layer points to the weight matrix of the current layer
	// out points to the location where the result produced by the thread block should be written to.
	//   Can be nullptr if nothing should be written.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16 * (WIDTH + SKEW);
	__half* __restrict__ bias_broadcast_tile_shmem = weights_shmem + 16 * (WIDTH + SKEW);

	const uint32_t weights_row = (8 * li) % WIDTH;
	const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;

	// Load weight matrix into shared memory for the last multiplication.
	// Loading into shared memory as opposed to directly into registers is faster
	// because unlike in the previous layers, each warp uses the same entries of the weight matrix.
	*(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];
	if (wi == 0) {
		prepare_bias_broadcast_tile_in_shmem<WIDTH>(bias_broadcast_tile_shmem, bias_this_layer, 0);
	}

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_BLOCKS; ++i)
		wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);

	// Perform last layer by parallelizing over iters
	for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
		load_bias_fragment<WIDTH>(result_frag, bias_broadcast_tile_shmem, bias_this_layer != nullptr, 0);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * idx) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
		}

		warp_activation<__half>(activation, result_frag, result_frag);

		if (output_layout == wmma::mem_row_major) {
			wmma::store_matrix_sync(out + idx * 16 * output_stride, result_frag, output_stride, output_layout);
		} else {
			wmma::store_matrix_sync(out + idx * 16, result_frag, output_stride, output_layout);
		}
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
	// output_threadblock will be filled by the thread block's act_shmem

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_threadblock[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void kernel_neurcomp_residual(const Activation output_activation, const __half* __restrict__ params, const bool use_bias, const __half* __restrict__ input, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t output_stride, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_residual_blocks, const nvcuda::wmma::layout_t input_layout, const nvcuda::wmma::layout_t output_layout) {
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;
	const uint32_t input_aux_shmem_elements = (WIDTH + 16) * (in_width + INPUT_SKEW);
	const uint32_t last_layer_aux_shmem_elements = 32u * (WIDTH + SKEW);
	const uint32_t aux_shmem_elements = input_aux_shmem_elements > last_layer_aux_shmem_elements ? input_aux_shmem_elements : last_layer_aux_shmem_elements;
	const uint32_t activation_tile_elements = N_ITERS * 16 * (WIDTH + SKEW);

	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;
	__half* residual_shmem = shmem + activation_tile_elements + aux_shmem_elements;

	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS;
	const uint32_t layer_stride = WIDTH * batch_size;
	const uint32_t padded_out_width = ((out_width + 15) / 16) * 16;

	const __half* params_this_layer = params;
	const __half* first_layer_weights = params_this_layer;
	params_this_layer += WIDTH * in_width;
	const __half* first_layer_bias = nullptr;
	if (use_bias) {
		first_layer_bias = params_this_layer;
		params_this_layer += WIDTH;
	}

	if (input_layout == nvcuda::wmma::mem_col_major || in_width != WIDTH) {
		if (input_layout == nvcuda::wmma::mem_row_major) {
			threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, first_layer_weights, first_layer_bias, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
		} else {
			threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, first_layer_weights, first_layer_bias, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
		}
	} else {
		threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, first_layer_weights, first_layer_bias, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
	}

	for (uint32_t block_idx = 0; block_idx < n_residual_blocks; ++block_idx) {
		__syncthreads();
		const float weight_1 = block_idx > 0 ? 0.5f : 1.0f;
		const float weight_2 = block_idx + 1 == n_residual_blocks ? 0.5f : 1.0f;

		threadblock_copy_static<WIDTH, N_ITERS>(residual_shmem, act_shmem);
		__syncthreads();
		threadblock_scale_static<WIDTH, N_ITERS>(act_shmem, weight_1);
		__syncthreads();

		const __half* linear_1_weights = params_this_layer;
		params_this_layer += WIDTH * WIDTH;
		const __half* linear_1_bias = nullptr;
		if (use_bias) {
			linear_1_bias = params_this_layer;
			params_this_layer += WIDTH;
		}
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, linear_1_weights, linear_1_bias, nullptr);

		const __half* linear_2_weights = params_this_layer;
		params_this_layer += WIDTH * WIDTH;
		const __half* linear_2_bias = nullptr;
		if (use_bias) {
			linear_2_bias = params_this_layer;
			params_this_layer += WIDTH;
		}
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, linear_2_weights, linear_2_bias, nullptr);

		__syncthreads();
		threadblock_residual_add_static<WIDTH, N_ITERS>(act_shmem, residual_shmem, weight_2);

		if (!INFERENCE) {
			__syncthreads();
			threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + layer_stride * (block_idx + 1) + elem_idx * WIDTH);
		}
	}

	__syncthreads();

	if (out_width > 16) {
		if (INFERENCE) {
			threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
		}
	} else if (out) {
		const __half* output_weights = params_this_layer;
		params_this_layer += padded_out_width * WIDTH;
		const __half* output_bias = nullptr;
		if (use_bias) {
			output_bias = params_this_layer;
		}
		if (output_layout == nvcuda::wmma::mem_row_major) {
			threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, output_weights, output_bias, out + elem_idx * output_stride, output_stride, output_layout);
		} else {
			threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, output_weights, output_bias, out + elem_idx, output_stride, output_layout);
		}
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ params, const bool use_bias, const __half* __restrict__ input, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t output_stride, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_hidden_matmuls, const nvcuda::wmma::layout_t input_layout, const nvcuda::wmma::layout_t output_layout) {
	// `input` points to the input matrix. Can be any width.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
	// `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

	// Commented out due to isolated strange side-effects on Windows
	// if (INFERENCE) {
	// 	assert(out_intermediate == nullptr);
	// } else {
	// 	assert(out_intermediate);
	// }

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// In some cases, it also contains the weight matrix for the first and last layer.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	// Each block computes exactly one 16-element chunk of the batch.
	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS;
	const uint32_t padded_out_width = ((out_width + 15) / 16) * 16;
	const __half* params_this_layer = params;
	const __half* first_layer_weights = params_this_layer;
	params_this_layer += WIDTH * in_width;
	const __half* first_layer_bias = nullptr;
	if (use_bias) {
		first_layer_bias = params_this_layer;
		params_this_layer += WIDTH;
	}

	// First layer
	if (input_layout == nvcuda::wmma::mem_col_major || in_width != WIDTH) {
		if (input_layout == nvcuda::wmma::mem_row_major) {
			threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, first_layer_weights, first_layer_bias, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
		} else {
			threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, first_layer_weights, first_layer_bias, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
		}
	} else {
		// If the input has the same width & layout as the hidden layers, we can simply use the network's regular layer routine (with static size)
		// instead of using the slower dynamic input layer routine.
		threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, first_layer_weights, first_layer_bias, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
	}

	const uint32_t layer_stride = WIDTH * batch_size;

	// Hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		const __half* hidden_weights = params_this_layer;
		params_this_layer += WIDTH * WIDTH;
		const __half* hidden_bias = nullptr;
		if (use_bias) {
			hidden_bias = params_this_layer;
			params_this_layer += WIDTH;
		}
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, hidden_weights, hidden_bias, !INFERENCE ? (out_intermediate + layer_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
	}

	if (out_width > 16) {
		// In the forward pass, intermediate activations are already written out.
		if (INFERENCE) {
			threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
		}
	} else if (out) {
		// Last layer
		const __half* output_weights = params_this_layer;
		params_this_layer += padded_out_width * WIDTH;
		const __half* output_bias = nullptr;
		if (use_bias) {
			output_bias = params_this_layer;
		}
		if (output_layout == nvcuda::wmma::mem_row_major) {
			threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, output_weights, output_bias, out + elem_idx * output_stride, output_stride, output_layout);
		} else {
			threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, output_weights, output_bias, out + elem_idx, output_stride, output_layout);
		}
	}
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, RM>& biases,
	bool use_bias,
	const GPUMatrixDynamic<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	(void)output_activation;
	(void)weights;
	(void)biases;
	(void)use_bias;
	(void)input;
	(void)output_intermediate;
	(void)output;
	(void)n_hidden_layers;
	throw std::runtime_error{"The fully fused forward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, RM>& biases,
	bool use_bias,
	const GPUMatrixDynamic<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	(void)biases;
	const uint32_t batch_size = input.cols();
	const uint32_t in_width = input.rows();

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
	constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
	constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

	static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");

	CHECK_THROW(in_width % 16 == 0);
	CHECK_THROW(weights.rows() == WIDTH);
	CHECK_THROW(weights.cols() % 16 == 0);
	CHECK_THROW(output_intermediate.cols() == batch_size);
	CHECK_THROW(!output || output->cols() == batch_size);
	CHECK_THROW(input.layout() == RM || input.stride() == input.m());

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;

	if (batch_size % (16 * N_ITERS) != 0) {
		throw std::runtime_error{fmt::format("Batch size must be a multiple of {}.", 16 * N_ITERS)};
	}

	const dim3 threads = { 32u, N_BLOCK_ROWS, 1 }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	size_t shmem_size = sizeof(__half) * (32 + 16 * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of last-layer weights + 16*WIDTH rows of block-shared bias tile + 16*WIDTH*N_ITERS rows of intermediate activations
	if (in_width != WIDTH || input.layout() == RM) {
		// If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
		shmem_size = std::max(shmem_size, sizeof(__half) * ((WIDTH + 16) * (in_width + INPUT_SKEW) + 16 * (WIDTH + SKEW)));
	}

	const dim3 blocks = { n_blocks, 1u, 1u };

	check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
	kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE><<<blocks, threads, shmem_size, stream>>>(
		output_activation,
		weights.data(),
		use_bias,
		input.data(),
		output_intermediate.data(),
		output ? output->data() : nullptr,
		output ? output->stride() : 0,
		batch_size,
		in_width,
		output ? output->rows() : 0,
		n_hidden_layers,
		// The kernels operate with transposed layouts compared with the MLP code
		input.layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
		output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major
	);
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<!std::is_same<__half, T>::value> neurcomp_residual_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, RM>& biases,
	bool use_bias,
	const GPUMatrixDynamic<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_residual_blocks
) {
	(void)stream;
	(void)output_activation;
	(void)weights;
	(void)biases;
	(void)use_bias;
	(void)input;
	(void)output_intermediate;
	(void)output;
	(void)n_residual_blocks;
	throw std::runtime_error{"The fully fused forward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<std::is_same<__half, T>::value> neurcomp_residual_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, RM>& biases,
	bool use_bias,
	const GPUMatrixDynamic<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_residual_blocks
) {
	(void)biases;
	const uint32_t batch_size = input.cols();
	const uint32_t in_width = input.rows();

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;
	constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

	static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");

	CHECK_THROW(in_width % 16 == 0);
	CHECK_THROW(weights.rows() == WIDTH);
	CHECK_THROW(weights.cols() % 16 == 0);
	CHECK_THROW(output_intermediate.cols() == batch_size);
	CHECK_THROW(!output || output->cols() == batch_size);
	CHECK_THROW(input.layout() == RM || input.stride() == input.m());

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t activation_tile_elements = N_ITERS * 16 * (WIDTH + SKEW);
	const uint32_t aux_shmem_elements = std::max((WIDTH + 16) * (in_width + INPUT_SKEW), 32u * (WIDTH + SKEW));
	if (batch_size % (16 * N_ITERS) != 0) {
		throw std::runtime_error{fmt::format("Batch size must be a multiple of {}.", 16 * N_ITERS)};
	}

	const dim3 threads = {32u, N_BLOCK_ROWS, 1};
	const dim3 blocks = {div_round_up(batch_size, 16u * N_ITERS), 1u, 1u};
	size_t shmem_size = sizeof(__half) * (activation_tile_elements * 2 + aux_shmem_elements);

	check_shmem_error(cudaFuncSetAttribute(kernel_neurcomp_residual<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
	kernel_neurcomp_residual<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE><<<blocks, threads, shmem_size, stream>>>(
		output_activation,
		weights.data(),
		use_bias,
		input.data(),
		output_intermediate.data(),
		output ? output->data() : nullptr,
		output ? output->stride() : 0,
		batch_size,
		in_width,
		output ? output->rows() : 0,
		n_residual_blocks,
		input.layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
		output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major
	);
}

template <typename T, int WIDTH>
FuyllyFusedSIREN<T, WIDTH>::FuyllyFusedSIREN(
	uint32_t input_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation,
	float omega_0,
	bool use_bias,
	NetworkMode mode
)
	: m_input_width{input_width}
	, m_network_width{WIDTH}
	, m_output_width{output_width}
	, m_n_hidden_layers{n_hidden_layers}
	, m_activation{activation}
	, m_output_activation{output_activation}
	, m_omega_0{omega_0}
	, m_use_bias{use_bias}
	, m_mode{mode}
{
	using TCNN_NAMESPACE :: next_multiple;

	if (m_n_hidden_layers <= 0) {
		throw std::runtime_error("FuyllyFusedSIREN requires at least 1 hidden layer (3 layers in total).");
	}

	m_n_hidden_matmuls = m_mode == NetworkMode::PlainMLP ? (n_hidden_layers - 1) : n_hidden_layers;

	m_padded_output_width = next_multiple(m_output_width, REQUIRED_ALIGNMENT());

	// Create matrices related to weights
	m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
	m_biases.emplace_back(nullptr, m_network_width, 1);

	if (m_mode == NetworkMode::PlainMLP) {
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_biases.emplace_back(nullptr, m_network_width, 1);
		}
	} else {
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_biases.emplace_back(nullptr, m_network_width, 1);
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_biases.emplace_back(nullptr, m_network_width, 1);
		}
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_biases.emplace_back(nullptr, m_padded_output_width, 1);

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_total_n_params += m_weight_matrices[i].n_elements();
		if (m_use_bias) {
			m_total_n_params += m_biases[i].n_elements();
		}
	}

	sync_siren_omega_0_to_device(omega_0);
}

template <typename T, int WIDTH>
void FuyllyFusedSIREN<T, WIDTH>::inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();

	GPUMatrix<T> inference_tmp = m_output_width > 16 ? GPUMatrix<T>{m_network_width, batch_size, stream} : GPUMatrix<T>{nullptr, m_network_width, batch_size};
	GPUMatrixDynamic<T> normalized_input;
	const GPUMatrixDynamic<T>* effective_input = &input;
	if (m_mode == NetworkMode::NeurCompResidual) {
		normalized_input = GPUMatrixDynamic<T>{input.rows(), batch_size, stream, input.layout()};
		kernel_scale_shift<<<n_blocks_linear(input.n_elements()), n_threads_linear, 0, stream>>>(input.n_elements(), 2.0f, -1.0f, input.data(), normalized_input.data());
		effective_input = &normalized_input;
	}

	switch (m_activation) {
		case Activation::None:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::None, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::None, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Exponential:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Exponential, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Exponential, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Sigmoid:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Sigmoid, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Sigmoid, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::ReLU:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::ReLU, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::ReLU, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Squareplus:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Squareplus, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Squareplus, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Softplus:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Softplus, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Softplus, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Sine:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Sine, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Sine, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		case Activation::SirenSine:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::SirenSine, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::SirenSine, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, inference_tmp, &output, m_n_hidden_matmuls);
			}
			break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	if (m_mode == NetworkMode::NeurCompResidual) {
		kernel_scale_shift<<<n_blocks_linear(output.n_elements()), n_threads_linear, 0, stream>>>(output.n_elements(), 0.5f, 0.5f, output.data(), output.data());
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (m_output_width > 16) {
		throw std::runtime_error("FuyllyFusedSIREN: inference_mixed_precision_impl is not implemented for output widths greater than 16.");
	}
}

template <typename T, int WIDTH>
std::unique_ptr<Context> FuyllyFusedSIREN<T, WIDTH>::forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);
	GPUMatrixDynamic<T> normalized_input;
	const GPUMatrixDynamic<T>* effective_input = &input;
	if (m_mode == NetworkMode::NeurCompResidual) {
		normalized_input = GPUMatrixDynamic<T>{input.rows(), batch_size, stream, input.layout()};
		kernel_scale_shift<<<n_blocks_linear(input.n_elements()), n_threads_linear, 0, stream>>>(input.n_elements(), 2.0f, -1.0f, input.data(), normalized_input.data());
		effective_input = &normalized_input;
	}

	switch (m_activation) {
		case Activation::None:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::None, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::None, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Exponential:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Exponential, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Exponential, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Sigmoid:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Sigmoid, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Sigmoid, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::ReLU:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::ReLU, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::ReLU, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Squareplus:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Squareplus, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Squareplus, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Softplus:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Softplus, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Softplus, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::Sine:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::Sine, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::Sine, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		case Activation::SirenSine:
			if (m_mode == NetworkMode::PlainMLP) {
				mlp_fused_forward<WIDTH, T, Activation::SirenSine, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			} else {
				neurcomp_residual_forward<WIDTH, T, Activation::SirenSine, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input_bias(use_inference_params), m_use_bias, *effective_input, forward->hidden.at(0), output, m_n_hidden_matmuls);
			}
			break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	if (m_mode == NetworkMode::NeurCompResidual && output) {
		kernel_scale_shift<<<n_blocks_linear(output->n_elements()), n_threads_linear, 0, stream>>>(output->n_elements(), 0.5f, 0.5f, output->data(), output->data());
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (output && m_output_width > 16) {
		throw std::runtime_error("FuyllyFusedSIREN: forward_impl is not implemented for output widths greater than 16.");
	}

	return forward;
}

template <typename T, int WIDTH>
std::unique_ptr<typename FuyllyFusedSIREN<T, WIDTH>::ForwardContext> FuyllyFusedSIREN<T, WIDTH>::allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size) {
	auto forward = std::make_unique<ForwardContext>();

	// Use GPUMatrixBase::allocate_shared_memory to ensure the matrices occupy contiguous memory.
	// (Needed in the fully-fused kernels.)
	forward->hidden.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden[i].set_size_unsafe(m_network_width, batch_size);
	}

	forward->alloc = GPUMatrixBase::allocate_shared_memory(stream, forward->hidden);

	return forward;
}

template <typename T, int WIDTH>
void FuyllyFusedSIREN<T, WIDTH>::set_params_impl(T* params, T* inference_params, T* gradients) {
	(void)inference_params;
	(void)gradients;
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data_unsafe(params + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
		if (m_use_bias) {
			m_biases[i].set_data_unsafe(params + current_pos);
			current_pos += m_biases[i].n_elements();
		} else {
			m_biases[i].set_data_unsafe(nullptr);
		}
	}
}

template <typename T, int WIDTH>
void FuyllyFusedSIREN<T, WIDTH>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
	// Construct weight matrices
	std::vector<GPUMatrix<float, RM>> weight_matrices_full_precision;
	std::vector<GPUMatrix<float, RM>> biases_full_precision;
	weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_input_width);
	params_full_precision += weight_matrices_full_precision.back().n_elements();
	if (m_use_bias) {
		biases_full_precision.emplace_back(params_full_precision, m_network_width, 1);
		params_full_precision += biases_full_precision.back().n_elements();
	}

	if (m_mode == NetworkMode::PlainMLP) {
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_network_width);
			params_full_precision += weight_matrices_full_precision.back().n_elements();
			if (m_use_bias) {
				biases_full_precision.emplace_back(params_full_precision, m_network_width, 1);
				params_full_precision += biases_full_precision.back().n_elements();
			}
		}
	} else {
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_network_width);
			params_full_precision += weight_matrices_full_precision.back().n_elements();
			if (m_use_bias) {
				biases_full_precision.emplace_back(params_full_precision, m_network_width, 1);
				params_full_precision += biases_full_precision.back().n_elements();
			}

			weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_network_width);
			params_full_precision += weight_matrices_full_precision.back().n_elements();
			if (m_use_bias) {
				biases_full_precision.emplace_back(params_full_precision, m_network_width, 1);
				params_full_precision += biases_full_precision.back().n_elements();
			}
		}
	}

	weight_matrices_full_precision.emplace_back(params_full_precision, m_padded_output_width, m_network_width);
	params_full_precision += weight_matrices_full_precision.back().n_elements();
	if (m_use_bias) {
		biases_full_precision.emplace_back(params_full_precision, m_padded_output_width, 1);
	}

	// Initialize matrices
	for (size_t i = 0; i < weight_matrices_full_precision.size(); ++i) {
		if (m_activation == Activation::Sine) {
			if (i == 0) {
				weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else {
			weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}

	for (auto& bias_matrix : biases_full_precision) {
		bias_matrix.initialize_constant(0.0f);
	}
}

}
}
