#pragma once

#include <vector>
#include <cmath>

#include "utils/defs.hpp"

__attribute__((const))
inline num mult_add(const num_arr& a, const num_arr& b, const num c, std::size_t n) noexcept
{
    
    num sum = zero;
    for (std::size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum + c;
}

__attribute__((const))
inline num_arr activation(const num_arr a, std::size_t n) noexcept
{
    num_arr arr(n);
    for (std::size_t i = 0; i < n; i++)
        arr[i] = (num)(1 / (1 + std::exp((double)-a[i])));
    return arr;
}

__attribute__((const))
inline num_arr activation_derv(const num_arr a, std::size_t n) noexcept
{
    num_arr arr(n);
    for (std::size_t i = 0; i < n; i++)
        arr[i] = a[i] * (1 - a[i]);
    return arr;
}

__attribute__((const))
inline num_arr error(const num_arr x, const num_arr y, std::size_t n) noexcept
{
    num_arr arr(n);
    for (std::size_t i = 0; i < n; i++)
    {
        arr[i] = x[i] - y[i];
    }
    return arr;
}