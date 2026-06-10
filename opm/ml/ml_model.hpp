/*
  Copyright (c) 2016 Robert W. Rose
  Copyright (c) 2018 Paul Maevskikh
  Copyright (c) 2024 NORCE

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

  Note: This file is based on kerasify/keras_model.hh
*/

#ifndef ML_MODEL_H_
#define ML_MODEL_H_

#include <fmt/format.h>

#include <opm/common/ErrorMacros.hpp>
#include <opm/material/densead/Math.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>
#include <fstream>

#ifdef OPM_ML_USE_BLAS
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace Opm
{

namespace ML
{

    // NN layer
    // ---------------------
    /** \class Tensor class
     * Implements mathematical tensor (Max 4d)
     */
    template <class T>
    class Tensor
    {
    public:
        Tensor()
        {
        }

        explicit Tensor(int i)
        {
            resizeI<std::vector<int>>({i});
        }

        Tensor(int i, int j)
        {
            resizeI<std::vector<int>>({i, j});
        }

        Tensor(int i, int j, int k)
        {
            resizeI<std::vector<int>>({i, j, k});
        }

        Tensor(int i, int j, int k, int l)
        {
            resizeI<std::vector<int>>({i, j, k, l});
        }

        template <typename Sizes>
        void resizeI(const Sizes& sizes)
        {
            if (sizes.size() == 1)
                dims_ = {(int)sizes[0]};
            if (sizes.size() == 2)
                dims_ = {(int)sizes[0], (int)sizes[1]};
            if (sizes.size() == 3)
                dims_ = {(int)sizes[0], (int)sizes[1], (int)sizes[2]};
            if (sizes.size() == 4)
                dims_ = {(int)sizes[0], (int)sizes[1], (int)sizes[2], (int)sizes[3]};

            data_.resize(std::accumulate(begin(dims_), end(dims_), 1.0, std::multiplies<>()));
        }

        void flatten()
        {
            OPM_ERROR_IF(dims_.size() == 0, "Invalid tensor");

            int elements = dims_[0];
            for (unsigned int i = 1; i < dims_.size(); i++) {
                elements *= dims_[i];
            }
            dims_ = {elements};
        }

        T& operator()(int i)
        {
            OPM_ERROR_IF(dims_.size() != 1, "Invalid indexing for tensor");

            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));

            return data_[i];
        }

        const T& operator()(int i) const
        {
            OPM_ERROR_IF(dims_.size() != 1, "Invalid indexing for tensor");

            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));

            return data_[i];
        }

        T& operator()(int i, int j)
        {
            OPM_ERROR_IF(dims_.size() != 2, "Invalid indexing for tensor");
            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));
            OPM_ERROR_IF(!(j < dims_[1] && j >= 0),
                         fmt::format(fmt::runtime(" Invalid j:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     j,
                                     dims_[1]));

            return data_[dims_[1] * i + j];
        }

        const T& operator()(int i, int j) const
        {
            OPM_ERROR_IF(dims_.size() != 2, "Invalid indexing for tensor");
            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));
            OPM_ERROR_IF(!(j < dims_[1] && j >= 0),
                         fmt::format(fmt::runtime(" Invalid j:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     j,
                                     dims_[1]));
            return data_[dims_[1] * i + j];
        }

        T& operator()(int i, int j, int k)
        {
            OPM_ERROR_IF(dims_.size() != 3, "Invalid indexing for tensor");
            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));
            OPM_ERROR_IF(!(j < dims_[1] && j >= 0),
                         fmt::format(fmt::runtime(" Invalid j:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     j,
                                     dims_[1]));
            OPM_ERROR_IF(!(k < dims_[2] && k >= 0),
                         fmt::format(fmt::runtime(" Invalid k:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     k,
                                     dims_[2]));

            return data_[dims_[2] * (dims_[1] * i + j) + k];
        }

        const T& operator()(int i, int j, int k) const
        {
            OPM_ERROR_IF(dims_.size() != 3, "Invalid indexing for tensor");
            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));
            OPM_ERROR_IF(!(j < dims_[1] && j >= 0),
                         fmt::format(fmt::runtime(" Invalid j:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     j,
                                     dims_[1]));
            OPM_ERROR_IF(!(k < dims_[2] && k >= 0),
                         fmt::format(fmt::runtime(" Invalid k:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     k,
                                     dims_[2]));

            return data_[dims_[2] * (dims_[1] * i + j) + k];
        }

        T& operator()(int i, int j, int k, int l)
        {
            OPM_ERROR_IF(dims_.size() != 4, "Invalid indexing for tensor");
            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));
            OPM_ERROR_IF(!(j < dims_[1] && j >= 0),
                         fmt::format(fmt::runtime(" Invalid j:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     j,
                                     dims_[1]));
            OPM_ERROR_IF(!(k < dims_[2] && k >= 0),
                         fmt::format(fmt::runtime(" Invalid k:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     k,
                                     dims_[2]));
            OPM_ERROR_IF(!(l < dims_[3] && l >= 0),
                         fmt::format(fmt::runtime(" Invalid l:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     l,
                                     dims_[3]));

            return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
        }

        const T& operator()(int i, int j, int k, int l) const
        {
            OPM_ERROR_IF(dims_.size() != 4, "Invalid indexing for tensor");
            OPM_ERROR_IF(!(i < dims_[0] && i >= 0),
                         fmt::format(fmt::runtime(" Invalid i:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     i,
                                     dims_[0]));
            OPM_ERROR_IF(!(j < dims_[1] && j >= 0),
                         fmt::format(fmt::runtime(" Invalid j:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     j,
                                     dims_[1]));
            OPM_ERROR_IF(!(k < dims_[2] && k >= 0),
                         fmt::format(fmt::runtime(" Invalid k:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     k,
                                     dims_[2]));
            OPM_ERROR_IF(!(l < dims_[3] && l >= 0),
                         fmt::format(fmt::runtime(" Invalid l:  "
                                     "{}"
                                     "  max: "
                                     "{}"),
                                     l,
                                     dims_[3]));

            return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
        }

        void fill(const T& value)
        {
            std::ranges::fill(data_, value);
        }

        // Tensor addition
        Tensor operator+(const Tensor& other)
        {
            OPM_ERROR_IF(dims_.size() != other.dims_.size(),
                         "Cannot add tensors with different dimensions");
            Tensor result;
            result.dims_ = dims_;
            result.data_.resize(data_.size());

            std::ranges::transform(data_, other.data_, result.data_.begin(),
                                   [](const T& x, const T& y) { return x + y; });

            return result;
        }

        // Tensor multiplication
        Tensor multiply(const Tensor& other)
        {
            OPM_ERROR_IF(dims_.size() != other.dims_.size(),
                         "Cannot multiply elements with different dimensions");

            Tensor result;
            result.dims_ = dims_;
            result.data_.resize(data_.size());

            std::ranges::transform(data_, other.data_, result.data_.begin(),
                                   [](const T& x, const T& y) { return x * y; });

            return result;
        }

        // Tensor dot for 2d tensor
        Tensor dot(const Tensor& other)
        {
            OPM_ERROR_IF(dims_.size() != 2, "Invalid tensor dimensions");
            OPM_ERROR_IF(other.dims_.size() != 2, "Invalid tensor dimensions");

            OPM_ERROR_IF(dims_[1] != other.dims_[0],
                         "Cannot multiply with different inner dimensions");

            Tensor tmp(dims_[0], other.dims_[1]);

            for (int i = 0; i < dims_[0]; i++) {
                for (int j = 0; j < other.dims_[1]; j++) {
                    for (int k = 0; k < dims_[1]; k++) {
                        tmp(i, j) += (*this)(i, k) * other(k, j);
                    }
                }
            }

            return tmp;
        }

        void swap(Tensor& other)
        {
            dims_.swap(other.dims_);
            data_.swap(other.data_);
        }

        std::vector<int> dims_;
        std::vector<T> data_;
    };

    // NN layer
    // ---------------------
    /** \class Neural Network  Layer base class.
     * Objects of type Evaluation can be AD object Opm::DenseAd::Evaluation,
    double or float types.
     */
    template <class Evaluation>
    class NNLayer
    {
    public:
        NNLayer()
        {
        }

        virtual ~NNLayer()
        {
        }

        // Loads the ML trained file, returns true if the file exists
        virtual bool loadLayer(std::ifstream& file) = 0;
        // Apply the NN layers
        virtual bool apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out) = 0;
    };

    //! Activation types
    enum class ActivationType {
        kLinear = 1,
        kRelu = 2,
        kSoftPlus = 3,
        kSigmoid = 4,
        kTanh = 5,
        kHardSigmoid = 6
    };

    /** \class Activation  Layer class
     * Applies an activation function
     */
    template <class Evaluation>
    class NNLayerActivation : public NNLayer<Evaluation>
    {
    public:

        NNLayerActivation(ActivationType activation_type = ActivationType::kLinear)
            : activation_type_(activation_type)
        {
        }

        bool loadLayer(std::ifstream& file) override;

        bool apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out) override;

    private:
        template <class E> friend class NNModel;
        ActivationType activation_type_;
    };

    /** \class Scaling Layer class
     * A preprocessing layer which rescales input values to a new range.
     */
    template <class Evaluation>
    class NNLayerScaling : public NNLayer<Evaluation>
    {
    public:
        NNLayerScaling(float data_min = 1.0f,
                       float data_max = 1.0f,
                       float feat_inf = 1.0f,
                       float feat_sup = 1.0f);

        bool loadLayer(std::ifstream& file) override;

        bool apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out) override;

    private:
        template <class E> friend class NNModel;
        float data_min_;
        float data_max_;
        float feat_inf_;
        float feat_sup_;
    };

    /** \class Unscaling Layer class
     * A postprocessing layer to undo the scaling according to feature_range.
     */
    template <class Evaluation>
    class NNLayerUnScaling : public NNLayer<Evaluation>
    {
    public:
        NNLayerUnScaling(float data_min = 1.0f,
                         float data_max = 1.0f,
                         float feat_inf = 1.0f,
                         float feat_sup = 1.0f);

        bool loadLayer(std::ifstream& file) override;

        bool apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out) override;

    private:
        template <class E> friend class NNModel;
        float data_min_;
        float data_max_;
        float feat_inf_;
        float feat_sup_;
    };

    /** \class Dense Layer class
     * Densely-connected NN layer.
     */
    template <class Evaluation>
    class NNLayerDense : public NNLayer<Evaluation>
    {
    public:
        NNLayerDense(Tensor<float> weights = {}, Tensor<float> biases = {}, ActivationType activation_type = ActivationType::kLinear);

        bool loadLayer(std::ifstream& file) override;

        bool apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out) override;

    private:
        template <class E> friend class NNModel;
        Tensor<float> weights_;
        Tensor<float> biases_;

        NNLayerActivation<Evaluation> activation_;
    };

    /** \class Neural Network Model class
     * A model grouping layers into an object
     */
    template <class Evaluation>
    class NNModel
    {
    public:
        enum class LayerType { kScaling = 1, kUnScaling = 2, kDense = 3, kActivation = 4 };

        virtual ~NNModel() = default;

        // loads models (.model files) generated by Kerasify
        virtual bool loadModel(const std::string& filename);

        virtual bool apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out);

        // Vectorised inference: evaluate the model for N samples in one pass.
        // in_flat  : flat row-major array of length n_samples * n_inputs
        // out_flat : flat array of length n_samples * n_outputs (caller-allocated)
        // n_samples: number of samples in the batch
        // Only scalar (value()) paths are used — derivatives are not propagated,
        // matching the existing pointwise path in FlowProblem::updateRelperms.
        virtual bool applyBatch(const std::vector<float>& in_flat,
                                std::vector<float>&       out_flat,
                                int                       n_samples) const;

    private:
        std::vector<std::unique_ptr<NNLayer<Evaluation>>> layers_;

        // --- Precomputed flat schedule for fast batched inference ------------
        // Built once from layers_ (no per-call dynamic_cast / Tensor indexing).
        // Holds raw pointers into the layer weight/bias storage, which stays
        // valid for the lifetime of layers_ (never mutated after loadModel).
        struct BatchLayer {
            LayerType      type;
            int            in_dim  = 0;
            int            out_dim = 0;
            const float*   weights = nullptr;   // dense: [in_dim*out_dim] row-major
            const float*   biases  = nullptr;   // dense: [out_dim]
            ActivationType act     = ActivationType::kLinear;
            float dmin = 0, dmax = 0, finf = 0, fsup = 0; // (un)scaling params
        };
        mutable std::vector<BatchLayer> batchSchedule_;
        mutable bool                    batchScheduleBuilt_ = false;
        mutable std::vector<float>      batchCur_;   // reusable scratch (ping)
        mutable std::vector<float>      batchNxt_;   // reusable scratch (pong)

        void buildBatchSchedule() const;
    };

    /** \class Neural Network Timer class
     */
    class NNTimer
    {
    public:
        //! Start the timer
        void start();
        //! Stop the timer and return elapsed time in milliseconds
        float stop();

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };




    template <typename T>
    bool readFile(std::ifstream& file, T& data)
    {
        file.read(reinterpret_cast<char*>(&data), sizeof(T));
        return !file.fail();
    }

    template <typename T>
    bool readFile(std::ifstream& file, T* data, size_t n)
    {
        file.read(reinterpret_cast<char*>(data), sizeof(T) * n);
        return !file.fail();
    }

    template <class Evaluation>
    bool NNLayerActivation<Evaluation>::loadLayer(std::ifstream& file)
    {
        unsigned int activation = 0;
        OPM_ERROR_IF(!readFile<unsigned int>(file, activation), "Failed to read activation type");

        switch (static_cast<ActivationType>(activation)) {
        case ActivationType::kLinear:
            activation_type_ = ActivationType::kLinear;
            break;
        case ActivationType::kRelu:
            activation_type_ = ActivationType::kRelu;
            break;
        case ActivationType::kSoftPlus:
            activation_type_ = ActivationType::kSoftPlus;
            break;
        case ActivationType::kHardSigmoid:
            activation_type_ = ActivationType::kHardSigmoid;
            break;
        case ActivationType::kSigmoid:
            activation_type_ = ActivationType::kSigmoid;
            break;
        case ActivationType::kTanh:
            activation_type_ = ActivationType::kTanh;
            break;
        default:
            OPM_ERROR_IF(true,
                         fmt::format(fmt::runtime("\n Unsupported activation type "
                                     "{}"),
                                     activation));
        }

        return true;
    }

    template <class Evaluation>
    bool NNLayerActivation<Evaluation>::apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out)
    {
        out = in;

        switch (activation_type_) {
        case ActivationType::kLinear:
            break;
        case ActivationType::kRelu:
            for (size_t i = 0; i < out.data_.size(); i++) {
                if (out.data_[i] < 0.0) {
                    out.data_[i] = 0.0;
                }
            }
            break;
        case ActivationType::kSoftPlus:
            for (size_t i = 0; i < out.data_.size(); i++) {
                out.data_[i] = log(1.0 + exp(out.data_[i]));
            }
            break;
        case ActivationType::kHardSigmoid:
            for (size_t i = 0; i < out.data_.size(); i++) {
                constexpr double sigmoid_scale = 0.2;
                const Evaluation& x = (out.data_[i] * sigmoid_scale) + 0.5;

                if (x <= 0) {
                    out.data_[i] = 0.0;
                } else if (x >= 1) {
                    out.data_[i] = 1.0;
                } else {
                    out.data_[i] = x;
                }
            }
            break;
        case ActivationType::kSigmoid:
            for (size_t i = 0; i < out.data_.size(); i++) {
                const Evaluation& x = out.data_[i];

                if (x >= 0) {
                    out.data_[i] = 1.0 / (1.0 + exp(-x));
                } else {
                    const Evaluation& z = exp(x);
                    out.data_[i] = z / (1.0 + z);
                }
            }
            break;
        case ActivationType::kTanh:
            for (size_t i = 0; i < out.data_.size(); i++) {
                out.data_[i] = sinh(out.data_[i]) / cosh(out.data_[i]);
            }
            break;
        default:
            break;
        }

        return true;
    }

    template <class Evaluation>
    NNLayerScaling<Evaluation>::NNLayerScaling(float data_min, float data_max, float feat_inf, float feat_sup)
        : data_min_(data_min)
        , data_max_(data_max)
        , feat_inf_(feat_inf)
        , feat_sup_(feat_sup)
    {
    }

    template <class Evaluation>
    bool NNLayerScaling<Evaluation>::loadLayer(std::ifstream& file)
    {
        OPM_ERROR_IF(!readFile<float>(file, data_min_), "Failed to read data min");
        OPM_ERROR_IF(!readFile<float>(file, data_max_), "Failed to read data max");
        OPM_ERROR_IF(!readFile<float>(file, feat_inf_), "Failed to read feat inf");
        OPM_ERROR_IF(!readFile<float>(file, feat_sup_), "Failed to read feat sup");
        return true;
    }

    template <class Evaluation>
    bool NNLayerScaling<Evaluation>::apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out)
    {
        out.data_.resize(in.data_.size());
        out.dims_ = in.dims_;
        for (size_t i = 0; i < out.data_.size(); i++) {
            auto tempscale = (in.data_[i] - data_min_) / (data_max_ - data_min_);
            out.data_[i] = tempscale * (feat_sup_ - feat_inf_) + feat_inf_;
        }

        return true;
    }

    template <class Evaluation>
    NNLayerUnScaling<Evaluation>::NNLayerUnScaling(
        float data_min, float data_max, float feat_inf, float feat_sup)
        : data_min_(data_min)
        , data_max_(data_max)
        , feat_inf_(feat_inf)
        , feat_sup_(feat_sup)
    {
    }

    template <class Evaluation>
    bool NNLayerUnScaling<Evaluation>::loadLayer(std::ifstream& file)
    {
        OPM_ERROR_IF(!readFile<float>(file, data_min_), "Failed to read data min");
        OPM_ERROR_IF(!readFile<float>(file, data_max_), "Failed to read data max");
        OPM_ERROR_IF(!readFile<float>(file, feat_inf_), "Failed to read feat inf");
        OPM_ERROR_IF(!readFile<float>(file, feat_sup_), "Failed to read feat sup");

        return true;
    }

    template <class Evaluation>
    bool NNLayerUnScaling<Evaluation>::apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out)
    {
        out.data_.resize(in.data_.size());
        out.dims_ = in.dims_;
        for (size_t i = 0; i < out.data_.size(); i++) {
            auto tempscale = (in.data_[i] - feat_inf_) / (feat_sup_ - feat_inf_);

            out.data_[i] = tempscale * (data_max_ - data_min_) + data_min_;
        }

        return true;
    }

    template <class Evaluation>
    NNLayerDense<Evaluation>::NNLayerDense(Tensor<float> weights, Tensor<float> biases, ActivationType activation_type)
        : weights_(weights)
        , biases_(biases)
        , activation_(activation_type)
    {
    }

    template <class Evaluation>
    bool NNLayerDense<Evaluation>::loadLayer(std::ifstream& file)
    {
        unsigned int weights_rows = 0;
        OPM_ERROR_IF(!readFile<unsigned int>(file, weights_rows), "Expected weight rows");
        OPM_ERROR_IF(!(weights_rows > 0), "Invalid weights # rows");

        unsigned int weights_cols = 0;
        OPM_ERROR_IF(!readFile<unsigned int>(file, weights_cols), "Expected weight cols");
        OPM_ERROR_IF(!(weights_cols > 0), "Invalid weights shape");

        unsigned int biases_shape = 0;
        OPM_ERROR_IF(!readFile<unsigned int>(file, biases_shape), "Expected biases shape");
        OPM_ERROR_IF(!(biases_shape > 0), "Invalid biases shape");

        weights_.resizeI<std::vector<unsigned int>>({weights_rows, weights_cols});
        OPM_ERROR_IF(!readFile<float>(file, weights_.data_.data(), weights_rows * weights_cols), "Expected weights");

        biases_.resizeI<std::vector<unsigned int>>({biases_shape});
        OPM_ERROR_IF(!readFile<float>(file, biases_.data_.data(), biases_shape), "Expected biases");

        OPM_ERROR_IF(!activation_.loadLayer(file), "Failed to load activation");

        return true;
    }

    /**
     * @brief Applies the forward pass of a dense (fully connected) neural-network layer.
     *
     * This method performs a matrix–vector multiplication between the layer's weight
     * matrix and the input tensor, adds the bias vector, and then applies the
     * configured activation function.
     *
     * ### Shape conventions
     * - `in` is treated as a 1D row vector of length `weights_.dims_[0]`.
     * - `weights_` has shape `(input_dim, output_dim)`:
     *      - rows = input features
     *      - columns = output neurons
     * - `biases_` is a vector of length `output_dim`.
     * - `out` is a 1D vector of length `output_dim`.
     *
     * This implements:
     * @f[
     *     \text{tmp}_j = \sum_i \text{in}_i \cdot W_{i,j} + b_j,
     *     \qquad \text{out} = \text{activation}(\text{tmp})
     * @f]
     *
     * ### Note on row-major vs column-major
     * The current implementation assumes row-major access to `W` and is efficient
     * when using larger batch sizes. For inference with very small batches
     * (especially `(1 × input_dim)`), a column-major layout or transposed multiply
     * could improve cache locality because each output neuron would read contiguous
     * memory. Whether to switch depends on expected inference batch sizes and the
     * storage layout of `Tensor<Evaluation>`. This will depend on future applications
     * of ML. 
     * Current applications and best related convention:
     * - Hybrid Newton: 
     *      - input (1, N_cells x N_in_feat)  --> output(1, N_cells x N_out_feat)
     *      - Best convention: column-major
     */
    template <class Evaluation>
    bool NNLayerDense<Evaluation>::apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out)
    {
        Tensor<Evaluation> tmp(weights_.dims_[1]);
        Tensor<Evaluation> temp_in(in);
        for (int i = 0; i < weights_.dims_[0]; i++) {
            for (int j = 0; j < weights_.dims_[1]; j++) {
                tmp(j) += (temp_in)(i)*weights_(i, j);
            }
        }

        for (int i = 0; i < biases_.dims_[0]; i++) {
            tmp(i) += biases_(i);
        }

        OPM_ERROR_IF(!activation_.apply(tmp, out), "Failed to apply activation");

        return true;
    }

    template <class Evaluation>
    bool NNModel<Evaluation>::loadModel(const std::string& filename)
    {
        std::ifstream file(filename.c_str(), std::ios::binary);
        OPM_ERROR_IF(!file.is_open(),
                     fmt::format(fmt::runtime("\n Unable to open file "
                                 "{}"),
                                 filename.c_str()));

        unsigned int num_layers = 0;
        OPM_ERROR_IF(!readFile<unsigned int>(file, num_layers), "Expected number of layers");

        for (unsigned int i = 0; i < num_layers; i++) {
            unsigned int layer_type = 0;
            OPM_ERROR_IF(!readFile<unsigned int>(file, layer_type), "Expected layer type");

            std::unique_ptr<NNLayer<Evaluation>> layer = nullptr;

            switch (static_cast<LayerType>(layer_type)) {
            case LayerType::kScaling:
                layer = std::make_unique<NNLayerScaling<Evaluation>>();
                break;
            case LayerType::kUnScaling:
                layer = std::make_unique<NNLayerUnScaling<Evaluation>>();
                break;
            case LayerType::kDense:
                layer = std::make_unique<NNLayerDense<Evaluation>>();
                break;
            case LayerType::kActivation:
                layer = std::make_unique<NNLayerActivation<Evaluation>>();
                break;
            default:
                break;
            }

            OPM_ERROR_IF(!layer,
                         fmt::format(fmt::runtime("\n Unknown layer type "
                                     "{}"),
                                     layer_type));
            OPM_ERROR_IF(!layer->loadLayer(file),
                         fmt::format(fmt::runtime("\n Failed to load layer "
                                     "{}"),
                                     i));

            layers_.emplace_back(std::move(layer));
        }

        return true;
    }

    template <class Evaluation>
    bool NNModel<Evaluation>::apply(const Tensor<Evaluation>& in, Tensor<Evaluation>& out)
    {
        Tensor<Evaluation> temp_in(in);

        for (unsigned int i = 0; i < layers_.size(); i++) {

            if (i > 0) {
                temp_in.swap(out);
            }

            OPM_ERROR_IF(!(layers_[i]->apply(temp_in, out)),
                         fmt::format(fmt::runtime("\n Failed to apply layer "
                                     "{}"),
                                     i));
        }
        return true;
    }

    // In-place activation over a flat buffer (batched).
    template <class T>
    inline void applyActivationFlat(std::vector<T>& v, ActivationType act)
    {
        switch (act) {
        case ActivationType::kRelu:
            for (auto& x : v) x = x > T{0} ? x : T{0};
            break;
        case ActivationType::kSigmoid:
            for (auto& x : v)
                x = x >= T{0} ? T{1} / (T{1} + std::exp(-x))
                              : std::exp(x) / (T{1} + std::exp(x));
            break;
        case ActivationType::kTanh:
            for (auto& x : v) x = std::tanh(x);
            break;
        case ActivationType::kSoftPlus:
            for (auto& x : v) x = std::log(T{1} + std::exp(x));
            break;
        case ActivationType::kHardSigmoid:
            for (auto& x : v) {
                T y = x * T{0.2} + T{0.5};
                x = y <= T{0} ? T{0} : (y >= T{1} ? T{1} : y);
            }
            break;
        default:
            break;  // kLinear — no-op
        }
    }

    // Build the flat inference schedule once: a single dynamic_cast per layer,
    // caching raw weight/bias pointers and parameters.  Subsequent applyBatch
    // calls iterate this schedule with no RTTI and direct float* access.
    template <class Evaluation>
    void NNModel<Evaluation>::buildBatchSchedule() const
    {
        batchSchedule_.clear();
        batchSchedule_.reserve(layers_.size());
        for (const auto& layer_ptr : layers_) {
            BatchLayer bl;
            if (auto* d = dynamic_cast<const NNLayerDense<Evaluation>*>(layer_ptr.get())) {
                bl.type    = LayerType::kDense;
                bl.in_dim  = d->weights_.dims_[0];
                bl.out_dim = d->weights_.dims_[1];
                bl.weights = d->weights_.data_.data();
                bl.biases  = d->biases_.data_.data();
                bl.act     = d->activation_.activation_type_;
            } else if (auto* s = dynamic_cast<const NNLayerScaling<Evaluation>*>(layer_ptr.get())) {
                bl.type = LayerType::kScaling;
                bl.dmin = s->data_min_; bl.dmax = s->data_max_;
                bl.finf = s->feat_inf_; bl.fsup = s->feat_sup_;
            } else if (auto* u = dynamic_cast<const NNLayerUnScaling<Evaluation>*>(layer_ptr.get())) {
                bl.type = LayerType::kUnScaling;
                bl.dmin = u->data_min_; bl.dmax = u->data_max_;
                bl.finf = u->feat_inf_; bl.fsup = u->feat_sup_;
            } else if (auto* a = dynamic_cast<const NNLayerActivation<Evaluation>*>(layer_ptr.get())) {
                bl.type = LayerType::kActivation;
                bl.act  = a->activation_type_;
            } else {
                continue;
            }
            batchSchedule_.push_back(bl);
        }
        batchScheduleBuilt_ = true;
    }

    // -------------------------------------------------------------------------
    // applyBatch: vectorised forward pass over N samples.
    //
    // Layout (row-major):  in_flat[s*n_inputs + f], out_flat[s*n_outputs + k].
    // Only float (scalar) arithmetic — no AD derivatives — matching the
    // pointwise path which stores krn.value().
    //
    // Optimisations vs. the reference implementation:
    //   * Precomputed flat schedule: no per-call dynamic_cast or Tensor::operator().
    //   * Reused ping-pong scratch buffers (batchCur_/batchNxt_): no per-layer
    //     reallocation across calls.
    //   * Dense layer is a matrix product OUT[N,out] = IN[N,in] * W[in,out] + b.
    //     With -DOPM_ML_USE_BLAS and a large enough batch it dispatches to
    //     cblas_sgemm; otherwise a cache-friendly scalar loop is used.
    // -------------------------------------------------------------------------
    template <class Evaluation>
    bool NNModel<Evaluation>::applyBatch(const std::vector<float>& in_flat,
                                         std::vector<float>&       out_flat,
                                         int                       n_samples) const
    {
        if (layers_.empty() || n_samples <= 0)
            return false;
        if (!batchScheduleBuilt_)
            buildBatchSchedule();

        batchCur_.assign(in_flat.begin(), in_flat.end());

        for (const auto& L : batchSchedule_) {
            switch (L.type) {
            case LayerType::kDense: {
                const int in_dim  = L.in_dim;
                const int out_dim = L.out_dim;
                batchNxt_.assign(static_cast<std::size_t>(n_samples) * out_dim, 0.0f);

#ifdef OPM_ML_USE_BLAS
                if (static_cast<long>(n_samples) * in_dim * out_dim >= 4096) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                n_samples, out_dim, in_dim,
                                1.0f, batchCur_.data(), in_dim,
                                L.weights, out_dim,
                                0.0f, batchNxt_.data(), out_dim);
                    for (int s = 0; s < n_samples; ++s) {
                        float* o = &batchNxt_[static_cast<std::size_t>(s) * out_dim];
                        for (int j = 0; j < out_dim; ++j) o[j] += L.biases[j];
                    }
                } else
#endif
                {
                    // Cache-friendly scalar GEMM: stream weight rows contiguously.
                    for (int s = 0; s < n_samples; ++s) {
                        const float* x = &batchCur_[static_cast<std::size_t>(s) * in_dim];
                        float*       o = &batchNxt_[static_cast<std::size_t>(s) * out_dim];
                        for (int i = 0; i < in_dim; ++i) {
                            const float  xi = x[i];
                            const float* wr = &L.weights[static_cast<std::size_t>(i) * out_dim];
                            for (int j = 0; j < out_dim; ++j) o[j] += xi * wr[j];
                        }
                        for (int j = 0; j < out_dim; ++j) o[j] += L.biases[j];
                    }
                }
                applyActivationFlat(batchNxt_, L.act);
                batchCur_.swap(batchNxt_);
                break;
            }
            case LayerType::kScaling:
                for (auto& v : batchCur_) {
                    float t = (v - L.dmin) / (L.dmax - L.dmin);
                    v = t * (L.fsup - L.finf) + L.finf;
                }
                break;
            case LayerType::kUnScaling:
                for (auto& v : batchCur_) {
                    float t = (v - L.finf) / (L.fsup - L.finf);
                    v = t * (L.dmax - L.dmin) + L.dmin;
                }
                break;
            case LayerType::kActivation:
                applyActivationFlat(batchCur_, L.act);
                break;
            }
        }

        out_flat = batchCur_;
        return true;
    }

} // namespace ML

} // namespace Opm

#endif // ML_MODEL_H_
