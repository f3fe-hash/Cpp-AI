#pragma once

#include <vector>
#include <iostream>

#ifdef DEBUG
#include <cassert>
#endif

#define _debug(x) std::cout << "[DEBUG] " << x << std::endl;

template<typename T>
using vec = std::vector<T>;

template<typename T>
using vec2D = std::vector<std::vector<T>>;

using num       = float;
using num_arr   = vec<num>;
using num_arr2D = vec2D<num>;

using uchar  = unsigned char;
using ushort = unsigned short;
using uint   = unsigned int;

constexpr const num zero = (num)0.00;

struct dataset_t
{
    vec<num_arr> X;
    vec<num_arr> y;

    uint size;
};