#pragma once

#include <cmath>

#include "utils/defs.hpp"

inline num mult_add(const num_arr a, const num_arr b, const num c, std::size_t n) noexcept
{
    num n_ = (num)0.00;
    for (std::size_t i = 0; i < n; i++)
        n_ += a[i] * b[i];
    return n_ + c;
}

inline num_arr activation(const num_arr a, std::size_t n) noexcept
{
    num_arr arr = num_arr(n);
    for (std::size_t i = 0; i < n; i++)
        arr[i] = (num)(1 / (1 + std::exp((double)-a[i])));
    return arr;
}

inline num_arr activation_derv(const num_arr a, std::size_t n) noexcept
{
    num_arr arr = num_arr(n);
    num_arr num = activation(a, n);
    for (std::size_t i = 0; i < n; i++)
        arr[i] = num[i] * (1 - num[i]);
    return arr;
}

inline num error(const num_arr x, const num_arr y, std::size_t n)
{
    num out  = (num)0.00;
    num diff = (num)0.00;

    for (std::size_t i = 0; i < n; i++)
    {
        diff = x[i] - y[i];
        out += diff * diff;
    }

    return out;
}
