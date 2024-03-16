#ifndef GENETIC_ALG_GENETICALG_HPP
#define GENETIC_ALG_GENETICALG_HPP

#include <Eigen/Core>
#include <vector>
#include <unordered_map>
#include <set>
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
        ROULETTE_WHEEL,
        RANK
    };

    enum CrossoverType
    {
        ONE_POINT,
        MULTI_POINT,
        UNIFORM
    };

    struct Parameters
    {
        int pop_size;
        int ind_size;
        std::function<double(const Individual &)> fitness_function;
        double crossover_rate = 0.8;
        double mutation_rate = 0.01;
        SelectionType selection = TOURNAMENT;
        CrossoverType crossover = ONE_POINT;
        int tournament_size = 3;
        int multi_point_crossover_points = 2;
        double uniform_crossover_parent_ratio = 0.5;
        bool elitism = true;
        int epoch_improvement_threshold = 10;
        double minimum_improvement_rate = 0.01;
    };

    class GeneticAlg
    {
    public:
        GeneticAlg(const Parameters &params);

        void run_epoch();

        void run_epochs(size_t epochs);

        const std::pair<Individual, double> get_fittest() const { return {_fittest_individual, _max_fitness}; }

        const bool early_stop() const { return _stop; }

    protected:
        Population _population;
        Parameters _params;

        double _max_fitness = -std::numeric_limits<double>::max();
        Individual _fittest_individual;

        int _no_improvement_count = 0;
        bool _stop = false;

        const std::pair<int, double> _find_fittest();

        Population _tournament_selection();
        Population _roulette_wheel_selection();
        Population _rank_selection();

        void _one_point_crossover(Population &selected) const;
        void _multi_point_crossover(Population &selected) const;
        void _uniform_crossover(Population &selected) const;

        void _mutate(int skip_index);
    };

    Individual random_individual(int size);
}

#endif
