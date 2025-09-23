#include "nn.hpp"

NeuralNetwork::NeuralNetwork(vec<uint> layer_sizes) noexcept
{
    this->layer_sizes = layer_sizes;
    uint num_layers = layer_sizes.size();

    gen = std::mt19937(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Resize weights and biases for each layer (starting from layer 1)
    weights.resize(num_layers);
    biases.resize(num_layers);

    weights[0] = {{}};
    biases[0]  = {};
    for (uint i = 1; i < num_layers; ++i)
    {
        weights[i].resize(layer_sizes[i]);
        biases[i].resize(layer_sizes[i]);

        for (uint j = 0; j < layer_sizes[i]; ++j)
        {
            weights[i][j].resize(layer_sizes[i - 1]); // Each neuron connects to all neurons in previous layer
            for (uint k = 0; k < layer_sizes[i - 1]; ++k)
                weights[i][j][k] = (num)dist(gen);
            biases[i][j] = (num)dist(gen);
        }
    }

    this->delta.resize(this->layer_sizes.size());

    // Only initialize once
    if (__nn_math_context.local_work_size == 0)
        cl_math_init();
}

num_arr NeuralNetwork::forward(const num_arr* input) noexcept
{
    this->layer_outputs.clear();
    this->layer_outputs.resize(this->layer_sizes.size());
    this->layer_outputs[0] = *input;

    num_arr current = *input;

    for (uint layer = 1; layer < layer_sizes.size(); ++layer)
    {
        const uint in_size  = layer_sizes[layer - 1];
        const uint out_size = layer_sizes[layer];

        // Flatten weights
        std::vector<float> weights_flat(out_size * in_size);
        for (uint o = 0; o < out_size; ++o)
            for (uint i = 0; i < in_size; ++i)
                weights_flat[o * in_size + i] = weights[layer][o][i];

        // Setup OpenCL buffers
        cl::Buffer buf_input(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(float) * in_size, current.data());
        cl::Buffer buf_weights(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * weights_flat.size(), weights_flat.data());
        cl::Buffer buf_biases(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * out_size, biases[layer].data());
        cl::Buffer buf_output(__nn_math_context.context, CL_MEM_WRITE_ONLY,
                              sizeof(float) * out_size);

        // Set up kernel
        cl::Kernel kernel = __nn_math_context.getKernel("forward_layer");
        kernel.setArg(0, buf_input);
        kernel.setArg(1, buf_weights);
        kernel.setArg(2, buf_biases);
        kernel.setArg(3, buf_output);
        kernel.setArg(4, (int)in_size);
        kernel.setArg(5, (int)out_size);

        // Run kernel

        __nn_math_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
            cl::NDRange(out_size), cl::NullRange);
        __nn_math_context.queue.finish();

        // Read back result
        current.resize(out_size);
        __nn_math_context.queue.enqueueReadBuffer(buf_output, CL_TRUE, 0,
            sizeof(float) * out_size, current.data());

        // ✅ Save current layer output
        this->layer_outputs[layer] = current;
    }

    return current;
}

void NeuralNetwork::backprop(const dataset_t* dset) noexcept
{
#ifdef DEBUG
    if (__builtin_expect(this->layer_sizes.empty(), 0))
    {
        std::cerr << "[FATAL] layer_sizes is empty in backprop(). Neural network not initialized properly." << std::endl;
        std::exit(1);
    }
#endif

    // Clamp batch sizes
    if (this->batch_size > dset->size)
        this->batch_size = dset->size;

    const uint num_batches = (dset->size + this->batch_size - 1) / this->batch_size;

    vec<num_arr2D> X(num_batches), y(num_batches);

    // Split data into batches
    vec<uint> batch_sizes;
    for (uint i = 0; i <= dset->size; i += this->batch_size)
    {
        const uint batch_idx = i / this->batch_size;
        const uint end = std::min(i + this->batch_size, dset->size);

        batch_sizes.push_back(end - i);

        X[batch_idx].insert(X[batch_idx].end(), dset->X.begin() + i, dset->X.begin() + end);
        y[batch_idx].insert(y[batch_idx].end(), dset->y.begin() + i, dset->y.begin() + end);
    }

    const uint L           = this->layer_sizes.size() - 1;
    const uint output_size = this->layer_sizes[L];

    num_arr output;
    num_arr losses(output_size);
    for (uint batch = 0; batch < num_batches; ++batch)
    {
        for (uint i = 0; i < X[batch].size(); ++i)
        {
            output = this->forward(&X[batch][i]);

            // Get loss
            num_arr _losses = error(output, y[batch][i], output_size);
            for (uint j = 0; j < output_size; ++j) 
                losses[j] += _losses[j];
        }

        for (num& n : losses)
            n /= X[batch].size();
        
        // Output layer deltas
        delta[L].assign(num_batches, zero); // Safe initialization for accumulation
        for (uint i = 0; i < X[batch].size(); ++i)
        {
            const num_arr& d_act = activation_derv(output, output_size);

            // Output layer delta
            for (uint k = 0; k < output_size; ++k)
                delta[L][k] = losses[k] * d_act[k] + delta[L][k];
        }

        // Hidden layer deltas
        for (uint l = L; l > 1; --l)
        {
            uint neurons = this->layer_sizes[l - 1];
            delta[l].resize(this->layer_sizes[l]);
            delta[l - 1].resize(neurons);
            const num_arr& dervs = activation_derv(layer_outputs[l - 1], layer_outputs[l - 1].size());
            for (uint j = 0; j < neurons; ++j)
            {
                num sum = zero;
                for (uint k = 0; k < layer_sizes[l]; ++k)
                    sum += weights[l][k][j] * delta[l][k];
                delta[l - 1][j] = sum * dervs[j];
            }
        }

        // Gradient descent step
        for (uint l = 1; l < this->layer_sizes.size(); ++l)
        {
            for (uint j = 0; j < this->layer_sizes[l]; ++j)
            {
                for (uint k = 0; k < weights[l][j].size(); ++k)
                    weights[l][j][k] -= (num)(this->lr * delta[l][j] * layer_outputs[l - 1][k]);
                biases[l][j] -= (num)(this->lr * delta[l][j]);
            }
        }
    }
}
