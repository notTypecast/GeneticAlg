#include "GeneticAlg.hpp"
#include <iostream>

namespace genetic_alg
{
    GeneticAlg::GeneticAlg(const Parameters &params)
        : _params(params)
    {
        assert(params.pop_size > 0 && "Population size must be greater than 0");
        assert(params.ind_size > 0 && "Individual size must be greater than 0");
        assert(params.crossover_rate >= 0.0 && params.crossover_rate <= 1.0 && "Crossover rate must be between 0 and 1");
        assert(params.mutation_rate >= 0.0 && params.mutation_rate <= 1.0 && "Mutation rate must be between 0 and 1");
        assert(params.tournament_size > 0 && "Tournament size must be greater than 0");
        assert(params.multi_point_crossover_points > 0 && params.multi_point_crossover_points < params.ind_size && "Multi-point crossover points must be greater than 0 and less than individual size");
        assert(params.uniform_crossover_parent_ratio >= 0.0 && params.uniform_crossover_parent_ratio <= 1.0 && "Uniform crossover parent ratio must be between 0 and 1");
        assert(params.epoch_improvement_threshold > 0 && "Epoch improvement threshold must be greater than 0");
        assert(params.minimum_improvement_rate >= 0.0 && params.minimum_improvement_rate <= 1.0 && "Minimum improvement rate must be between 0 and 1");

        _population = Population(params.ind_size, params.pop_size);

        for (int i = 0; i < params.pop_size; ++i)
        {
            _population.block(0, i, params.ind_size, 1) = random_individual(params.ind_size);
        }
    }

    const double GeneticAlg::_evaluate_fitness(const Individual &ind)
    {
        if (_fitness_cache.find(ind) != _fitness_cache.end())
        {
            return _fitness_cache[ind];
        }

        _fitness_cache[ind] = _params.fitness_function(ind);

        return _fitness_cache[ind];
    }

    const std::pair<int, double> GeneticAlg::_find_fittest()
    {
        double max_fitness = _evaluate_fitness(_population.block(0, 0, _population.rows(), 1));
        int max_index = 0;

        for (int i = 1; i < _population.cols(); ++i)
        {
            const Individual &individual = _population.block(0, i, _population.rows(), 1);
            const double fitness = _evaluate_fitness(individual);

            if (fitness > max_fitness)
            {
                max_fitness = fitness;
                max_index = i;
            }
        }

        return std::make_pair(max_index, max_fitness);
    }

    Population GeneticAlg::_tournament_selection()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, _population.cols() - 1);

        Population selected(_population.rows(), _population.cols());

        for (int i = 0; i < _population.cols(); ++i)
        {
            std::vector<int> tournament;

            for (int j = 0; j < _params.tournament_size; ++j)
            {
                tournament.push_back(dis(gen));
            }

            Individual max_individual = _population.block(0, tournament[0], _population.rows(), 1);
            int max_fitness = _evaluate_fitness(max_individual);

            for (int j = 1; j < _params.tournament_size; ++j)
            {
                const Individual &individual = _population.block(0, tournament[j], _population.rows(), 1);
                int fitness = _evaluate_fitness(individual);

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

    Population GeneticAlg::_roulette_wheel_selection()
    {
        double offset = 0.0;
        double total_fitness = 0.0;
        for (int i = 0; i < _population.cols(); ++i)
        {
            double fitness = _evaluate_fitness(_population.block(0, i, _population.rows(), 1));
            total_fitness += fitness;

            if (fitness < offset)
            {
                offset = fitness;
            }
        }

        total_fitness -= offset * _population.cols();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rdis(0.0, total_fitness);

        std::vector<double> spins(_population.cols());

        for (int i = 0; i < _population.cols(); ++i)
        {
            spins[i] = rdis(gen);
        }

        std::sort(spins.begin(), spins.end());

        Population selected(_population.rows(), _population.cols());

        int ind_index = 0;
        double current_fitness = 0.0;

        for (int i = 0; i < _population.cols() && ind_index < _population.cols(); ++i)
        {
            current_fitness += _evaluate_fitness(_population.block(0, i, _population.rows(), 1)) - offset;

            while (ind_index < _population.cols() && spins[ind_index] < current_fitness)
            {
                selected.block(0, ind_index++, _population.rows(), 1) = _population.block(0, i, _population.rows(), 1);
            }
        }

        return selected;
    }

    Population GeneticAlg::_rank_selection()
    {
        int rank_sum = _population.cols() * (_population.cols() + 1) / 2;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rdis(0, rank_sum);

        std::vector<double> ranks(_population.cols());
        std::vector<double> spins(_population.cols());
        for (int i = 0; i < _population.cols(); ++i)
        {
            ranks[i] = i;
            spins[i] = rdis(gen);
        }

        std::sort(ranks.begin(), ranks.end(), [&](int i, int j)
                  { return _evaluate_fitness(_population.block(0, i, _population.rows(), 1)) < _evaluate_fitness(_population.block(0, j, _population.rows(), 1)); });
        std::sort(spins.begin(), spins.end());

        Population selected(_population.rows(), _population.cols());

        int ind_index = 0;
        double current_rank = 0.0;

        for (int i = 0; i < _population.cols() && ind_index < _population.cols(); ++i)
        {
            current_rank += i + 1;

            while (ind_index < _population.cols() && spins[ind_index] < current_rank)
            {
                selected.block(0, ind_index++, _population.rows(), 1) = _population.block(0, ranks[i], _population.rows(), 1);
            }
        }

        return selected;
    }

    void GeneticAlg::_one_point_crossover(Population &selected) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rdis(0.0, 1.0);
        std::uniform_int_distribution<int> dis(1, selected.rows() - 1);

        for (int i = 0; i < selected.cols() - 1; i += 2)
        {
            if (rdis(gen) < _params.crossover_rate)
            {
                int crossover_point = dis(gen);
                Eigen::VectorXi temp = selected.block(0, i, crossover_point, 1);
                selected.block(0, i, crossover_point, 1) = selected.block(0, i + 1, crossover_point, 1);
                selected.block(0, i + 1, crossover_point, 1) = temp;
            }
        }
    }

    void GeneticAlg::_multi_point_crossover(Population &selected) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rdis(0.0, 1.0);
        std::uniform_int_distribution<int> dis(1, selected.rows() - 1);

        for (int i = 0; i < selected.cols() - 1; i += 2)
        {
            if (rdis(gen) < _params.crossover_rate)
            {
                std::set<int> crossover_points_set;
                for (int j = 0; j < _params.multi_point_crossover_points; ++j)
                {
                    int crossover_point;
                    do
                    {
                        crossover_point = dis(gen);
                    } while (crossover_points_set.find(crossover_point) != crossover_points_set.end());
                    crossover_points_set.insert(crossover_point);
                }
                crossover_points_set.insert(selected.rows());

                bool swap = false;
                int prev_crossover_point = 0;
                for (const int &crossover_point : crossover_points_set)
                {
                    if (swap)
                    {
                        Eigen::VectorXi temp = selected.block(prev_crossover_point, i, crossover_point - prev_crossover_point, 1);
                        selected.block(prev_crossover_point, i, crossover_point - prev_crossover_point, 1) = selected.block(prev_crossover_point, i + 1, crossover_point - prev_crossover_point, 1);
                        selected.block(prev_crossover_point, i + 1, crossover_point - prev_crossover_point, 1) = temp;
                    }
                    swap = !swap;
                    prev_crossover_point = crossover_point;
                }
            }
        }
    }

    void GeneticAlg::_uniform_crossover(Population &selected) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rdis(0.0, 1.0);

        for (int i = 0; i < selected.cols() - 1; i += 2)
        {
            if (rdis(gen) < _params.crossover_rate)
            {
                for (int j = 0; j < selected.rows(); ++j)
                {
                    if (rdis(gen) < _params.uniform_crossover_parent_ratio)
                    {
                        int temp = selected(j, i);
                        selected(j, i) = selected(j, i + 1);
                        selected(j, i + 1) = temp;
                    }
                }
            }
        }
    }

    void GeneticAlg::_mutate(int skip_index)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rdis(0.0, 1.0);

        if (!_params.elitism)
        {
            skip_index = -1;
        }

        for (int i = 0; i < _population.rows(); ++i)
        {
            if (i == skip_index)
            {
                continue;
            }
            for (int j = 0; j < _population.cols(); ++j)
            {
                if (rdis(gen) < _params.mutation_rate)
                {
                    _population(i, j) = !_population(i, j);
                }
            }
        }
    }

    void GeneticAlg::run_epoch()
    {
        Population selected;

        switch (_params.selection)
        {
        case TOURNAMENT:
            selected = _tournament_selection();
            break;
        case ROULETTE_WHEEL:
            selected = _roulette_wheel_selection();
            break;
        case RANK:
            selected = _rank_selection();
            break;
        }

        switch (_params.crossover)
        {
        case ONE_POINT:
            _one_point_crossover(selected);
            break;
        case MULTI_POINT:
            _multi_point_crossover(selected);
            break;
        case UNIFORM:
            _uniform_crossover(selected);
            break;
        }

        _population = selected;

        std::pair<int, double> fittest = _find_fittest();

        _mutate(fittest.first);

        fittest = _find_fittest();

        double prev_max_fitness = _max_fitness;
        if (fittest.second > _max_fitness)
        {
            _max_fitness = fittest.second;
            _fittest_individual = _population.block(0, fittest.first, _population.rows(), 1);
        }

        if (_max_fitness < prev_max_fitness * (1.0 + _params.minimum_improvement_rate))
        {
            ++_no_improvement_count;
        }
        else
        {
            _no_improvement_count = 0;
        }

        if (_no_improvement_count == _params.epoch_improvement_threshold)
        {
            _stop = true;
        }
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