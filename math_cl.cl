__kernel void mult_add(
    __global const float* A,
    __global const float* B,
    float c,
    __global float* result,
    int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += A[i] * B[i];
    *result = sum + c;
}

__kernel void activation(
    __global const float* input,
    __global float* output,
    int n)
{
    int id = get_global_id(0);
    if (id < n)
    {
        output[id] = 1.0f / (1.0f + exp(-input[id]));
    }
}

__kernel void activation_derv(
    __global const float* input,
    __global float* output,
    int n)
{
    int id = get_global_id(0);
    if (id < n)
    {
        output[id] = input[id] * (1.0f - input[id]);
    }
}

__kernel void error(
    __global const float* x,
    __global const float* y,
    __global float* out,
    int n)
{
    int id = get_global_id(0);
    if (id < n)
    {
        out[id] = x[id] - y[id];
    }
}

__kernel void forward_layer(
    __global const float* input,   // [in_size]
    __global const float* weights, // [out_size * in_size]
    __global const float* biases,  // [out_size]
    __global float* output,        // [out_size]
    int in_size,
    int out_size)
{
    int neuron_id = get_global_id(0);

    if (neuron_id < out_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_size; ++i)
        {
            float input_val = input[i];
            float weight_val = weights[neuron_id * in_size + i]; // row-major
            sum += input_val * weight_val;
        }

        sum += biases[neuron_id];

        // Activation (sigmoid)
        output[neuron_id] = 1.0f / (1.0f + exp(-sum));
    }
}