#include <iostream>
#include <iomanip>

#include "nn.hpp"

int main()
{
    const vec<num_arr> X =
    {
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1}
    };

    const vec<num_arr> y =
    {
        {0},
        {1},
        {1},
        {0}
    };

    dataset_t set;
    set.X = X;
    set.y = y;
    set.size = X.size();

    NeuralNetwork nn = NeuralNetwork({2, 3, 1});

    for (std::size_t i = 0; i < 100'000; ++i)
        nn.backprop(&set, 4);

    // Print the output
    for (std::size_t i = 0; i < set.size; i++)
    {
        const num_arr y = nn.forward(&X[i]);

        std::cout << "{";
        for (std::size_t j = 0; j < y.size(); ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << (double)y[j];
            if (j != y.size() - 1)
                std::cout << ", ";
        }
        std::cout << "};" << std::endl;
    }

    return 0;
}