#include "GeneticAlg.hpp"
#include <iostream>

namespace genetic_alg
{
    GeneticAlg::GeneticAlg(int pop_size,
                           int ind_size,
                           std::function<double(const Individual &)> fitness_function,
                           double crossover_rate,
                           double mutation_rate,
                           SelectionType selection,
                           int tournament_size)
        : _fitness_function(fitness_function), _crossover_rate(crossover_rate), _mutation_rate(mutation_rate), _selection(selection), _tournament_size(tournament_size)
    {
        assert(crossover_rate >= 0.0 && crossover_rate <= 1.0 && "Crossover rate must be between 0 and 1");
        assert(mutation_rate >= 0.0 && mutation_rate <= 1.0 && "Mutation rate must be between 0 and 1");
        assert(tournament_size > 0 && "Tournament size must be greater than 0");

        _population = Population(ind_size, pop_size);

        for (int i = 0; i < pop_size; ++i)
        {
            _population.block(0, i, ind_size, 1) = random_individual(ind_size);
        }
    }

    const std::pair<int, int> GeneticAlg::_find_fittest() const
    {
        int max_fitness = _fitness_function(_population.block(0, 0, _population.rows(), 1));
        int max_index = 0;

        for (int i = 1; i < _population.cols(); ++i)
        {
            const Individual &individual = _population.block(0, i, _population.rows(), 1);
            int fitness = _fitness_function(individual);

            if (fitness > max_fitness)
            {
                max_fitness = fitness;
                max_index = i;
            }
        }

        return std::make_pair(max_index, max_fitness);
    }

    Population GeneticAlg::_tournament_selection() const
    {
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_int_distribution<int> dis(0, _population.cols() - 1);

        Population selected(_population.rows(), _population.cols());

        for (int i = 0; i < _population.cols(); ++i)
        {
            std::vector<int> tournament;

            for (int j = 0; j < _tournament_size; ++j)
            {
                tournament.push_back(dis(gen));
            }

            Individual max_individual = _population.block(0, tournament[0], _population.rows(), 1);
            int max_fitness = _fitness_function(max_individual);

            for (int j = 1; j < _tournament_size; ++j)
            {
                const Individual &individual = _population.block(0, tournament[j], _population.rows(), 1);
                int fitness = _fitness_function(individual);

                if (fitness > max_fitness)
                {
                    max_individual = individual;
                    max_fitness = fitness;
                }
            }

            selected.block(0, i, _population.rows(), 1) = max_individual;
        }

        return selected;
    }

    void GeneticAlg::_one_point_crossover(Population &selected) const
    {
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<double> rdis(0.0, 1.0);
        std::uniform_int_distribution<int> dis(1, selected.rows() - 1);

        for (int i = 0; i < selected.cols(); i += 2)
        {
            if (rdis(gen) < _crossover_rate)
            {
                int crossover_point = dis(gen);
                Eigen::VectorXi temp = selected.block(0, i, crossover_point, 1);
                selected.block(0, i, crossover_point, 1) = selected.block(0, i + 1, crossover_point, 1);
                selected.block(0, i + 1, crossover_point, 1) = temp;
            }
        }
    }

    void GeneticAlg::_mutate(int skip_index)
    {
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<double> rdis(0.0, 1.0);

        for (int i = 0; i < _population.rows(); ++i)
        {
            if (i == skip_index)
            {
                continue;
            }
            for (int j = 0; j < _population.cols(); ++j)
            {
                if (rdis(gen) < _mutation_rate)
                {
                    _population(i, j) = !_population(i, j);
                }
            }
        }
    }

    bool GeneticAlg::run_epoch()
    {
        Population selected = _tournament_selection();

        _one_point_crossover(selected);

        _population = selected;

        std::pair<int, int> fittest = _find_fittest();

        _mutate(fittest.first);

        fittest = _find_fittest();

        if (fittest.second > _max_fitness)
        {
            _max_fitness = fittest.second;
            _fittest_individual = _population.block(0, fittest.first, _population.rows(), 1);
        }

        return true;
    }

    const std::pair<Individual, double> GeneticAlg::get_fittest() const
    {
        return std::make_pair(_fittest_individual, _max_fitness);
    }

    Individual random_individual(int size)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dis(0, 1);

        Eigen::VectorXi ind(size);

        for (int i = 0; i < size; ++i)
        {
            ind[i] = dis(gen);
        }

        return ind;
    }
}