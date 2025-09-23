#include <iostream>
#include <iomanip>

#include "nn.hpp"

#define PRINT_EPOCH_EVERY 500

// Train on binary full adder dataset
int main()
{
    // A, B, Carry
    const num_arr2D X =
    {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
    };

    // Q, Carry
    const num_arr2D y =
    {
   	    {0, 0},
        {1, 0},
        {1, 0},
        {0, 1},
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 1},
    };

    dataset_t set;
    set.X = X;
    set.y = y;
    set.size = X.size();

    NeuralNetwork nn = NeuralNetwork({3, 50, 2});
    nn.setBatchSize(1);
    nn.setLearningRate(0.05);

    for (uint i = 0; i < 20'000; ++i)
    {
        nn.backprop(&set);
        if (!(i % PRINT_EPOCH_EVERY))
	        std::cout << "Epoch " << i << std::endl;
    }

    // Print the output
    for (uint i = 0; i < set.size; i++)
    {
        const num_arr y = nn.forward(&X[i]);

        std::cout << "{";
        for (uint j = 0; j < y.size(); ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << (double)y[j];
            if (j != y.size() - 1)
                std::cout << ", ";
        }
        std::cout << "};" << std::endl;
    }

    return 0;
}
