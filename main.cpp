#include <iostream>

#include "src/GeneticAlg.hpp"

int main()
{
    genetic_alg::GeneticAlg ga(100, 50, [](const genetic_alg::Individual &ind)
                               { return ind.sum(); });

    for (int i = 0; i < 20; ++i)
    {
        ga.run_epoch();
        std::cout << "Epoch " << i << ": " << ga.get_fittest().first.transpose() << std::endl;
    }

    return 0;
}