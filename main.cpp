#include <iostream>

#include "src/GeneticAlg.hpp"

int main()
{
    genetic_alg::Parameters params;
    params.pop_size = 100;
    params.ind_size = 50;
    params.fitness_function = [](const genetic_alg::Individual &ind)
    { return ind.sum(); };
    params.selection = genetic_alg::TOURNAMENT;
    params.crossover = genetic_alg::ONE_POINT;
    params.uniform_crossover_parent_ratio = 0.3;

    genetic_alg::GeneticAlg ga(params);

    for (int i = 0; i < 40; ++i)
    {
        ga.run_epoch();
        std::pair<genetic_alg::Individual, double> fittest = ga.get_fittest();
        std::cout << "Epoch " << i << ": " << fittest.first.transpose() << ", score: " << fittest.second << std::endl;
        if (ga.early_stop())
        {
            std::cout << "Early stop at epoch " << i << std::endl;
            break;
        }
    }

    return 0;
}