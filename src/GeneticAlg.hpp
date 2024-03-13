#ifndef GENETIC_ALG_GENETICALG_HPP
#define GENETIC_ALG_GENETICALG_HPP

#include <Eigen/Core>
#include <vector>
#include <functional>
#include <random>
#include <cassert>

namespace genetic_alg
{
    using Individual = Eigen::VectorXi;
    using Population = Eigen::MatrixXi;

    enum SelectionType
    {
        TOURNAMENT,
        ROULETTE_FITNESS,
        ROULETTE_ORDER
    };

    class GeneticAlg
    {
    public:
        GeneticAlg(int pop_size,
                   int ind_size,
                   std::function<double(const Individual &)> fitness_function,
                   double crossover_rate = 0.8,
                   double mutation_rate = 0.01,
                   SelectionType selection = TOURNAMENT,
                   int tournament_size = 3);

        bool run_epoch();

        const std::pair<Individual, double> get_fittest() const;

    protected:
        Population _population;
        std::function<double(const Individual &)> _fitness_function;
        double _crossover_rate;
        double _mutation_rate;
        SelectionType _selection;
        int _tournament_size;

        double _max_fitness;
        Individual _fittest_individual;

        const std::pair<int, int> _find_fittest() const;

        Population _tournament_selection() const;
        Population _roulette_fitness_selection() const; // TODO: Implement
        Population _roulette_order_selection() const;   // TODO: Implement

        void _one_point_crossover(Population &selected) const;
        void _multi_point_crossover(Population &selected) const; // TODO: Implement
        void _uniform_crossover(Population &selected) const;     // TODO: Implement

        void _mutate(int skip_index = -1);
    };

    Individual random_individual(int size);
}

#endif
