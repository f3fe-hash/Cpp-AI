#pragma once

#include "utils/defs.hpp"

#if defined(USE_CUDA)
    #include "math_cuda.cu"
#elif defined(USE_OPENCL)
    #error "OpenCL is not yet supported."
    //#include "math_opencl.hpp"
#else
    #include "math_cpu.hpp"
#endif
