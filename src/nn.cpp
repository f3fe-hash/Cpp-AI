#include "nn.hpp"

NeuralNetwork::NeuralNetwork(vec<std::size_t> layer_sizes) noexcept
{
    this->layer_sizes = layer_sizes;
    std::size_t num_layers = layer_sizes.size();

    gen = std::mt19937(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Resize weights and biases for each layer (starting from layer 1)
    weights.resize(num_layers);
    biases.resize(num_layers);

    weights[0] = {{}};
    biases[0]  = {};
    for (std::size_t i = 1; i < num_layers; ++i)
    {
        weights[i].resize(layer_sizes[i]);
        biases[i].resize(layer_sizes[i]);

        for (std::size_t j = 0; j < layer_sizes[i]; ++j)
        {
            weights[i][j].resize(layer_sizes[i - 1]); // Each neuron connects to all neurons in previous layer
            for (std::size_t k = 0; k < layer_sizes[i - 1]; ++k)
                weights[i][j][k] = (num)dist(gen);
            biases[i][j] = (num)dist(gen);
        }
    }

    this->delta.resize(this->layer_sizes.size());
}

num_arr NeuralNetwork::forward(const num_arr* input) noexcept
{
    this->deltax = *input;

    this->layer_outputs.clear();
    this->layer_outputs.push_back(this->deltax); // Save input layer activations

    for (std::size_t layer = 1; layer < layer_sizes.size(); ++layer)
    {
        std::size_t num_neurons = layer_sizes[layer];
        this->deltay.resize(num_neurons);

        for (std::size_t neuron = 0; neuron < num_neurons; ++neuron)
            this->deltay[neuron] = mult_add(deltax, weights[layer][neuron], biases[layer][neuron], deltax.size());

        // Apply activation function
        this->deltay = activation(this->deltay, this->deltay.size());
        this->deltax.swap(this->deltay);

        // Save current layer output
        this->layer_outputs.push_back(this->deltax);
    }

    return this->deltax;
}

// Warning: size IS NOT THE AMOUNT OF EPOCHS TO RUN. Instead it is the number of dataset values to do at a time.
// If greater than the total size of the dataset, will default to the total size
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
    if ((std::size_t)this->batch_size > dset->size)
        this->batch_size = dset->size;


    const std::size_t num_batches = (dset->size + this->batch_size - 1) / this->batch_size;

    vec<vec<num_arr>> X(num_batches), y(num_batches);

    // Split data into batches
    for (std::size_t i = 0; i < dset->size; i += this->batch_size)
    {
        std::size_t batch_idx = i / this->batch_size;
        std::size_t end = std::min(i + this->batch_size, dset->size);

        X[batch_idx].insert(X[batch_idx].end(), dset->X.begin() + i, dset->X.begin() + end);
        y[batch_idx].insert(y[batch_idx].end(), dset->y.begin() + i, dset->y.begin() + end);
    }

    for (std::size_t batch = 0; batch < num_batches; ++batch)
    {
        std::size_t L = this->layer_sizes.size() - 1;

        for (std::size_t i = 0; i < X[batch].size(); ++i)
        {
            const num_arr output = this->forward(&X[batch][i]);

            if (i == 0)
                delta[L].assign(output.size(), (num)0.0); // Safe initialization for accumulation

#ifdef DEBUG
            if (__builtin_expect(L >= delta.size(), 0))
            {
                std::cerr << "[ERROR] L is out of bounds for delta!" << std::endl;
                std::exit(1);
            }
#endif

            const num_arr& d_act = activation_derv(output, output.size());
            const num_arr& losses = error(output, y[batch][i], output.size());

#ifdef DEBUG
            if (__builtin_expect(losses.size() != d_act.size(), false))
            {
                std::cerr << "[ERROR] Mismatch: losses.size() = " << losses.size()
                        << ", d_act.size() = " << d_act.size() << std::endl;
                std::exit(1);
            }

            if (__builtin_expect(delta[L].size() != losses.size(), 0))
            {
                std::cerr << "[ERROR] Mismatch: delta[" << L << "].size() = " << delta[L].size()
                        << ", expected = " << losses.size() << std::endl;
                std::exit(1);
            }
#endif

            // Output layer delta
            for (std::size_t k = 0; k < output.size(); ++k)
                delta[L][k] = losses[k] * d_act[k] + delta[L][k];
        }

        // Hidden layer deltas
        for (std::size_t l = L; l > 1; --l)
        {
            std::size_t neurons = this->layer_sizes[l - 1];
            delta[l].resize(this->layer_sizes[l]);
            delta[l - 1].resize(neurons);
            const num_arr& dervs = activation_derv(layer_outputs[l - 1], layer_outputs[l - 1].size());
            for (std::size_t j = 0; j < neurons; ++j)
            {
                num sum = zero;
                for (std::size_t k = 0; k < layer_sizes[l]; ++k)
                    sum += weights[l][k][j] * delta[l][k];
                delta[l - 1][j] = sum * dervs[j];
            }
        }

        // Gradient descent step
        for (std::size_t l = 1; l < this->layer_sizes.size(); ++l)
        {
            for (std::size_t j = 0; j < this->layer_sizes[l]; ++j)
            {
                for (std::size_t k = 0; k < weights[l][j].size(); ++k)
                    weights[l][j][k] -= (num)(this->lr * delta[l][j] * layer_outputs[l - 1][k]);
                biases[l][j] -= (num)(this->lr * delta[l][j]);
            }
        }
    }
}
