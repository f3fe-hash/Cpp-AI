#pragma once

#include <vector>
#include <random>

#include "utils/defs.hpp"
#include "utils/math.hpp"

struct dataset_t
{
    vec<num_arr> X;
    vec<num_arr> y;

    std::size_t size;
};

class NeuralNetwork
{
public:
    num_arr deltax, deltay;
    vec<num_arr> layer_outputs;
    vec<num_arr> biases;
    vec<vec<num_arr>> weights;

    vec<std::size_t> layer_sizes;

    double lr = 0.01;

    explicit NeuralNetwork(vec<std::size_t> layer_sizes) noexcept;
    ~NeuralNetwork() noexcept;

    num_arr forward(const num_arr* input) noexcept;

    void backprop(const dataset_t* dset, std::size_t size) noexcept;
};