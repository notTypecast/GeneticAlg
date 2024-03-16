#include <iostream>
#include <chrono>

#include "src/GeneticAlg.hpp"

int main()
{
    genetic_alg::Parameters params;
    params.pop_size = 20000;
    params.ind_size = 1000;
    params.fitness_function = [](const genetic_alg::Individual &ind)
    { return ind.sum(); };
    params.selection = genetic_alg::SelectionType::TOURNAMENT;
    params.crossover = genetic_alg::CrossoverType::ONE_POINT;

    genetic_alg::GeneticAlg ga(params);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    ga.run_epochs(3000);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::pair<genetic_alg::Individual, double> fittest = ga.get_fittest();
    std::cout << "Fittest individual: " << fittest.first.transpose() << ", score: " << fittest.second << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return 0;
}