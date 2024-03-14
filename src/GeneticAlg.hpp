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

    // Hash function for Eigen matrix and vector.
    // The code is from `hash_combine` function of the Boost library. See
    // http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
    template <typename T>
    struct matrix_hash : std::unary_function<T, size_t>
    {
        std::size_t operator()(T const &matrix) const
        {
            // Note that it is oblivious to the storage order of Eigen matrix (column- or
            // row-major). It will give you the same hash value for two different matrices if they
            // are the transpose of each other in different storage order.
            size_t seed = 0;
            for (size_t i = 0; i < matrix.size(); ++i)
            {
                auto elem = *(matrix.data() + i);
                seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
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
    };

    class GeneticAlg
    {
    public:
        GeneticAlg(const Parameters &params);

        bool run_epoch();

        const std::pair<Individual, double> get_fittest() const;

    protected:
        Population _population;
        Parameters _params;

        double _max_fitness = -std::numeric_limits<double>::max();
        Individual _fittest_individual;

        std::unordered_map<Individual, double, matrix_hash<Individual>> _fitness_cache;

        const double _evaluate_fitness(const Individual &ind);

        const std::pair<int, double> _find_fittest();

        Population _tournament_selection();
        Population _roulette_wheel_selection();
        Population _rank_selection();

        void _one_point_crossover(Population &selected) const;
        void _multi_point_crossover(Population &selected) const;
        void _uniform_crossover(Population &selected) const;

        void _mutate(int skip_index = -1);
    };

    Individual random_individual(int size);
}

#endif
