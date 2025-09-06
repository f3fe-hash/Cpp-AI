#pragma once

#include <vector>

using num     = double;
using num_arr = std::vector<num>;

using score = double;

template<typename T>
using vec = std::vector<T>;

// Uncomment if the num type is signed
#define signed_num

#ifdef signed_num
#define num_min -(1 << (sizeof(num) * 8 - 1))
#define num_max 1 << (sizeof(num) * 8 - 1)
#else
#define num_max 1 << (sizeof(num) * 8)
#define num_min 0
#endif