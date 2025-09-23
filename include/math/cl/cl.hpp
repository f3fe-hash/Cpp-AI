#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "utils/defs.hpp"

#define N 10'000'000

class CLContext
{
public:
    explicit CLContext()
    {}

    explicit inline CLContext(int local_work_size)
    {
        this->local_work_size = local_work_size;

        // 1. Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            std::cerr << "No OpenCL platforms found.\n";
            std::exit(1);
        }

        // 2. Select default platform and device
        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
        {
            std::cerr << "No OpenCL GPU devices found.\n";
            std::exit(1);
        }

        this->device = devices[0];
        std::cout << "Using device: " << this->device.getInfo<CL_DEVICE_NAME>() << "\n";

        std::string version = this->device.getInfo<CL_DEVICE_VERSION>();
        std::cout << "Device OpenCL version: " << version << std::endl;

        this->context = cl::Context(this->device);
        this->queue = cl::CommandQueue(this->context, this->device);
    }

    ~CLContext()
    {}

    void compile(const std::string& kernelSource)
    {
        this->prog = cl::Program(kernelSource.c_str());
        try
        {
            this->prog.build({ this->device });
        }
        catch (cl::Error& err)
        {
            std::cerr << "Build error: " << err.what() << "\n";
            std::cerr << this->prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device) << "\n";
            throw;
        }

        std::string kernelNames = this->prog.getInfo<CL_PROGRAM_KERNEL_NAMES>();
        std::stringstream ss(kernelNames);
        std::string name;
        while (std::getline(ss, name, ';'))
            this->kernels[name] = cl::Kernel(prog, name.c_str());
    }

    inline const cl::Kernel& getKernel(const std::string& name)
    {
        return this->kernels[name];
    }

    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Kernel kernel;
    cl::Program prog;

    std::unordered_map<std::string, cl::Kernel> kernels;

    int local_work_size;
};
