#include "nn.hpp"

NeuralNetwork::NeuralNetwork(vec<std::size_t> layer_sizes) noexcept
{
    this->layer_sizes = layer_sizes;
    std::size_t num_layers = layer_sizes.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    // Resize weights and biases for each layer (starting from layer 1)
    weights.resize(num_layers);
    biases.resize(num_layers);

    for (std::size_t i = 1; i < num_layers; ++i)
    {
        weights[i].resize(layer_sizes[i]);
        biases[i].resize(layer_sizes[i - 1]);

        for (std::size_t j = 0; j < layer_sizes[i]; ++j)
        {
            weights[i][j].resize(layer_sizes[i - 1]); // Each neuron connects to all neurons in previous layer
            for (std::size_t k = 0; k < layer_sizes[i - 1]; ++k)
                weights[i][j][k] = (num)dist(gen);
            biases[i][j] = (num)0.00;
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{}

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
        {
            this->deltay[neuron] = mult_add(deltax, weights[layer][neuron], biases[layer][neuron], deltax.size());
        }

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
void NeuralNetwork::backprop(const dataset_t* dset, std::size_t size) noexcept
{
    if (size > dset->size)
        size = dset->size;

    vec<num_arr> delta(this->layer_sizes.size()); // delta[layer][neuron]

    for (std::size_t i = 0; i < size; i++)
    {
        const num_arr& input = dset->X[i];
        const num_arr& target = dset->y[i];

        const num_arr output = this->forward(&input);

        // Output layer delta
        std::size_t L = this->layer_outputs.size() - 1;
        delta[L].resize(output.size());
        const num_arr& d_act = activation_derv(output, output.size());
        for (std::size_t j = 0; j < output.size(); ++j)
        {
            num out = output[j];
            delta[L][j] = (out - target[j]) * d_act[j];
        }

        // Hidden layer deltas
        for (std::size_t l = L; l > 1; --l)
        {
            std::size_t neurons = this->layer_sizes[l - 1];
            delta[l].resize(this->layer_sizes[l]);
            delta[l - 1].resize(neurons);
            const num_arr& dervs = activation_derv(layer_outputs[l], layer_outputs[l].size());
            for (std::size_t j = 0; j < neurons; ++j)
            {
                num sum = (num)0.00;
                for (std::size_t k = 0; k < layer_sizes[l]; ++k)
                {
                    sum += weights[l][k][j] * delta[l][k];
                }

                delta[l - 1][j] = sum * dervs[j];
            }
        }

        // Gradient descent step
        for (std::size_t l = 1; l < this->layer_sizes.size(); ++l)
        {
            for (std::size_t j = 0; j < this->layer_sizes[l]; ++j)
            {
                for (std::size_t k = 0; k < weights[l][j].size(); ++k)
                {
                    weights[l][j][k] -= (num)(this->lr * delta[l][j] * layer_outputs[l - 1][k]);
                }
                biases[l][j] -= (num)(this->lr * delta[l][j]);
            }
        }
    }
}
