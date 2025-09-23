#include "cl.hpp"
#include "utils/defs.hpp"

#include <cmath>
#include <fstream>

#define PI 3.141592653f
#define DEG_TO_RAD (PI / 180.0f)

inline CLContext __nn_math_context;

inline void cl_math_init()
{
    std::ifstream cl_file("math_cl.cl");
    if (!cl_file.is_open())
        std::cerr << "Error opening \"math_cl.cl\"" << std::endl;
    
    std::string fileContent(
        (std::istreambuf_iterator<char>(cl_file)),
        std::istreambuf_iterator<char>()
    );

    cl_file.close();

    __nn_math_context = CLContext(10);
    __nn_math_context.compile(fileContent);
}

inline num mult_add(const num_arr& a, const num_arr& b, const num c, uint n) noexcept
{
#ifdef DEBUG
    assert(a.size() == b.size());
    assert(a.size() == n);
#endif

    float result = 0.0f;
    cl::Buffer bufferA(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, (void*)a.data());
    cl::Buffer bufferB(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, (void*)b.data());
    cl::Buffer bufferOut(__nn_math_context.context, CL_MEM_WRITE_ONLY, sizeof(float));

    cl::Kernel kernel = __nn_math_context.getKernel("mult_add");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, c);
    kernel.setArg(3, bufferOut);
    kernel.setArg(4, static_cast<int>(n));

    // Single work item kernel, so no local work size needed
    __nn_math_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
    __nn_math_context.queue.finish();

    __nn_math_context.queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, sizeof(float), &result);

    return result;
}

inline num_arr activation(const num_arr a, uint n) noexcept
{
    num_arr result(n);

    cl::Buffer bufferIn(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, (void*)a.data());
    cl::Buffer bufferOut(__nn_math_context.context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

    cl::Kernel kernel = __nn_math_context.getKernel("activation");
    kernel.setArg(0, bufferIn);
    kernel.setArg(1, bufferOut);
    kernel.setArg(2, static_cast<int>(n));

    // Calculate global work size rounded up to multiple of local_work_size
    size_t local_size = __nn_math_context.local_work_size;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    __nn_math_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    __nn_math_context.queue.finish();

    __nn_math_context.queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, sizeof(float) * n, result.data());

    return result;
}

inline num_arr activation_derv(const num_arr a, uint n) noexcept
{
    num_arr result(n);

    cl::Buffer bufferIn(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, (void*)a.data());
    cl::Buffer bufferOut(__nn_math_context.context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

    cl::Kernel kernel = __nn_math_context.getKernel("activation_derv");
    kernel.setArg(0, bufferIn);
    kernel.setArg(1, bufferOut);
    kernel.setArg(2, static_cast<int>(n));

    size_t local_size = __nn_math_context.local_work_size;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    __nn_math_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    __nn_math_context.queue.finish();

    __nn_math_context.queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, sizeof(float) * n, result.data());

    return result;
}

inline num_arr error(const num_arr x, const num_arr y, uint n) noexcept
{
    num_arr result(n);

    cl::Buffer bufferX(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, (void*)x.data());
    cl::Buffer bufferY(__nn_math_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, (void*)y.data());
    cl::Buffer bufferOut(__nn_math_context.context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

    cl::Kernel kernel = __nn_math_context.getKernel("error");
    kernel.setArg(0, bufferX);
    kernel.setArg(1, bufferY);
    kernel.setArg(2, bufferOut);
    kernel.setArg(3, static_cast<int>(n));

    size_t local_size = __nn_math_context.local_work_size;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    __nn_math_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    __nn_math_context.queue.finish();

    __nn_math_context.queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, sizeof(float) * n, result.data());

    return result;
}

