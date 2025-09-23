#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "utils/defs.hpp"
#include "math/math.hpp"

class NeuralNetwork
{
    num_arr deltax, deltay;
    num_arr2D delta;
    num_arr2D layer_outputs;
    num_arr2D biases;
    vec<num_arr2D> weights;

    vec<uint> layer_sizes;

    std::random_device rd{};
    std::mt19937       gen;

    double lr      = 0.03;
    ushort batch_size = 16;

public:
    explicit NeuralNetwork(vec<uint> layer_sizes) noexcept;
    ~NeuralNetwork() = default;

    num_arr forward(const num_arr* input) noexcept;

    void backprop(const dataset_t* dset) noexcept;

    // Set the batch size
    inline void setBatchSize(uint batch_size) noexcept
    {
        this->batch_size = batch_size;
    }

    // Set the lr
    inline void setLearningRate(float lr) noexcept
    {
        this->lr = lr;
    }
};