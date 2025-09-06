#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "utils/defs.hpp"
#include "math/math.hpp"

#define DEBUG

struct dataset_t
{
    vec<num_arr> X;
    vec<num_arr> y;

    std::size_t size;
};

class NeuralNetwork
{
    num_arr deltax, deltay;
    vec<num_arr> delta;
    vec<num_arr> layer_outputs;
    vec<num_arr> biases;
    vec<vec<num_arr>> weights;

    vec<std::size_t> layer_sizes;

    std::random_device rd{};
    std::mt19937       gen;

    double lr      = 0.05;
    int batch_size = 1;

public:
    explicit NeuralNetwork(vec<std::size_t> layer_sizes) noexcept;
    ~NeuralNetwork() = default;

    num_arr forward(const num_arr* input) noexcept;

    void backprop(const dataset_t* dset) noexcept;
};