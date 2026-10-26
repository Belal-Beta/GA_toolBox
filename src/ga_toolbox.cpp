#include "headers/ga_toolbox.hpp"
using namespace std;
namespace ga_toolbox
{

    template <typename T>
    vector<Chromosome<T>> RankSelection<T>::select(
        const vector<Chromosome<T>> &population,
        int numberOfParents)
    {
        vector<Chromosome<T>> parents;
        if (population.empty() || numberOfParents <= 0)
            return parents;
        parents.reserve(numberOfParents);

        // 1. Create a vector of indices to sort without copying full Chromosome objects
        vector<size_t> indices(population.size());
        iota(indices.begin(), indices.end(), 0);

        // 2. Sort indices by fitness ascending (Index 0 becomes the worst, Index N-1 the best)
        sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
            { return population[a].getFitness() < population[b].getFitness(); });

        // 3. Assign weights based on rank (worst gets weight 1, best gets weight N)
        vector<double> weights(population.size());
        for (size_t i = 0; i < population.size(); ++i)
        {
            // weights[original_index] = rank
            weights[indices[i]] = static_cast<double>(i + 1);
        }

        // 4. Use discrete_distribution to handle the weighted selection automatically
        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<size_t> dist(weights.begin(), weights.end());

        for (int i = 0; i < numberOfParents; ++i)
        {
            parents.push_back(population[dist(gen)]);
        }

        return parents;
    }

    template <typename T>
    vector<Chromosome<T>> TournamentSelection<T>::select(
        const vector<Chromosome<T>> &population,
        int numberOfParents)
    {
        vector<Chromosome<T>> parents;
        if (population.empty() || numberOfParents <= 0)
            return parents;
        parents.reserve(numberOfParents);

        // Safeguard: default to a binary tournament (size 2) if size wasn't set
        size_t tSize = (this->tournamentSize > 0) ? this->tournamentSize : 2;

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<size_t> dist(0, population.size() - 1);

        for (int i = 0; i < numberOfParents; ++i)
        {
            // Pick the first candidate
            size_t best_idx = dist(gen);
            int best_fitness = population[best_idx].getFitness();

            // Compete with (tSize - 1) other random candidates
            for (size_t j = 1; j < tSize; ++j)
            {
                size_t candidate_idx = dist(gen);
                int candidate_fitness = population[candidate_idx].getFitness();

                if (candidate_fitness > best_fitness)
                {
                    best_idx = candidate_idx;
                    best_fitness = candidate_fitness;
                }
            }
            parents.push_back(population[best_idx]);
        }

        return parents;
    }

    template <typename T>
    pair<Chromosome<T>, Chromosome<T>> MultiPointCrossover<T>::cross(
        const Chromosome<T> &parent1,
        const Chromosome<T> &parent2)
    {
        Chromosome<T> offspring1, offspring2;
        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution shouldCross(crossoverRate);
        if (shouldCross(gen))
        {
            const size_t n = chromosome.genes.size();
            uniform_int_distribution<size_t> nPoints(1, n - 2);
            const size_t numberOfPoints = nPoints(gen);
            for (int i = 0; i < nPoints; i++)
            {
                const size_t point = nPoints(gen);
                crossoverPoints.push_back(point);
            }
            sort(crossoverPoints.begin(), crossoverPoints.end());
            int j = 0;
            for (size_t i = 0; i < n - 1; ++i)
            {
                offspring1.genes[i] = parent1.genes[i];
                offspring2.genes[i] = parent2.genes[i];
                if (i == crossoverPoints[j])
                {
                    swap(parent1, parent2);
                    j++;
                }
            }
        }
        return {offspring1, offspring2};
    }

    template <typename T>
    void fillOffspring(const Chromosome<T> &parent_giver, Chromosome<T> &offspring, int start, int end)
    {
        int size = parent_giver.size();

        unordered_set<int> existing_genes;
        for (int i = start; i <= end; ++i)
        {
            existing_genes.insert(offspring[i]);
        }

        int current_parent_idx = (end + 1) % size;
        int current_offspring_idx = (end + 1) % size;

        for (int i = 0; i < size; ++i)
        {
            int candidate_gene = parent_giver[current_parent_idx];

            if (existing_genes.find(candidate_gene) == existing_genes.end())
            {
                offspring[current_offspring_idx] = candidate_gene;
                current_offspring_idx = (current_offspring_idx + 1) % size;
            }
            current_parent_idx = (current_parent_idx + 1) % size;
        }
    }

    template <typename T>
    pair<Chromosome<T>, Chromosome<T>> OrderCrossover<T>::cross(
        const Chromosome<T> &parent1,
        const Chromosome<T> &parent2)
    {
        Chromosome<T> offspring1, offspring2;
        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution shouldCross(crossoverRate);
        if (shouldCross(gen))
        {
            const size_t n = chromosome.genes.size();
            uniform_int_distribution<size_t> nPoints(1, n - 2);
            int start = nPoints(gen);
            int end = nPoints(gen);
            if (start > end)
                swap(start, end);
            for (int i = start; i <= end; i++)
            {
                offspring1.genes[i] = parent1.genes[i];
                offspring2.genes[i] = parent2.genes[i];
            }
            fillOffspring(parent2, offspring1, start, end);
            fillOffspring(parent1, offspring2, start, end);
        }
        return {offspring1, offspring2};
    }

    void BitFlipMutation::mutate(Chromosome<bool> &chromosome)
    {
        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution shouldMutate(mutationRate);

        for (size_t i = 0; i < chromosome.genes.size() - 1; ++i)
        {
            if (shouldMutate(gen))
                chromosome.genes[i] = !chromosome.genes[i];
        }
    }

    void SwapMutation::mutate(Chromosome<int> &chromosome)
    {
        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution shouldMutate(mutationRate);
        const size_t n = chromosome.genes.size();
        uniform_int_distribution<size_t> randomIndex(0, n - 1);

        for (size_t i = 0; i < n - 1; ++i)
        {
            if (shouldMutate(gen))
            {
                const size_t j = randomIndex(gen);
                swap(chromosome.genes[i], chromosome.genes[j]);
            }
        }
    }

    void UniformMutation::mutate(Chromosome<float> &chromosome)
    {
        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution shouldMutate(mutationRate);
        const size_t n = chromosome.genes.size();
        for (size_t i = 0; i < n - 1; i++)
        {
            if (shouldMutate(gen))
            {
                float dlower = chromosome.lowerBound - chromosome.genes[i];
                float dupper = chromosome.upperBound - chromosome.genes[i];
                uniform_real_distribution<double> distrib(0.0, 1.0);
                double direction = distrib(gen);
                double delta = (direction <= 0.5) ? dlower : dupper;
                uniform_real_distribution<double> step(0.0, delta);
                double r2 = step(gen);
                chromosome.genes[i] += r2;
            }
        }
    }

    template <typename T>
    void SteadyStateReplacement<T>::replace(
        vector<Chromosome<T>> &population,
        vector<Chromosome<T>> &offSprings)
    {
        if (population.empty())
        {
            offSprings.clear();
            return;
        }

        vector<Chromosome<T>> nextPopulation = population;
        const size_t replacementCount = min(population.size(), offSprings.size());

        sort(offSprings.begin(), offSprings.end(),
            [](const Chromosome<T> &left, const Chromosome<T> &right)
            {
                return left.getFitness() > right.getFitness();
            });

        for (size_t i = 0; i < replacementCount; ++i)
        {
            auto worst = min_element(
                nextPopulation.begin(),
                nextPopulation.end(),
                [](const Chromosome<T> &left, const Chromosome<T> &right)
                {
                    return left.getFitness() < right.getFitness();
                });
            *worst = offSprings[i];
        }

        offSprings = move(nextPopulation);
    }

    template <typename T>
    void GenerationalReplacement<T>::replace(
        vector<Chromosome<T>> &population,
        vector<Chromosome<T>> &offSprings)
    {
        population = offSprings;
    }

    template <typename T>
    Chromosome<T> GA_toolbox<T>::run(
        vector<Chromosome<T>> population,
        FitnessFunction<T> fitness,
        FeasibilityFunction<T> feasibilityCheck)
    {
        // Ensure the required smart pointers are initialized before execution
        if (!selectionMethod || !crossoverMethod || !mutationMethod || !replacementMethod)
        {
            throw runtime_error("GA components (Selection, Crossover, Mutation, Replacement) must be initialized.");
        }

        Chromosome<T> bestOverall;
        bool foundValid = false;

        for (size_t generation = 0; generation < maxGenerations; ++generation)
        {

            // 1. Evaluation: Check Feasibility and Assign Fitness
            for (auto &chromo : population)
            {
                bool valid = feasibilityCheck(chromo);
                chromo.setValid(valid);

                if (valid)
                {
                    chromo.setFitness(fitness(chromo));
                }
                else
                {
                    // Heavily penalize infeasible chromosomes
                    chromo.setFitness(-1);
                }

                // Track the best global chromosome across all generations
                if (chromo.isValid())
                {
                    if (!foundValid || chromo.getFitness() > bestOverall.getFitness())
                    {
                        bestOverall = chromo;
                        foundValid = true;
                    }
                }
            }

            if (population.empty())
                break;

            // 2. Selection
            vector<Chromosome<T>> parents = selectionMethod->select(population, populationSize);

            // 3. Crossover
            vector<Chromosome<T>> offspring;
            offspring.reserve(populationSize);

            for (size_t i = 0; i < parents.size(); i += 2)
            {
                if (i + 1 < parents.size())
                {
                    auto [child1, child2] = crossoverMethod->cross(parents[i], parents[i + 1]);
                    offspring.push_back(child1);
                    offspring.push_back(child2);
                }
                else
                {
                    // Carry over the last parent safely if the parent count is odd
                    offspring.push_back(parents[i]);
                }
            }

            // 4. Mutation
            for (auto &child : offspring)
            {
                mutationMethod->mutate(child);
            }

            // 5. Replacement
            replacementMethod->replace(population, offspring);
        }

        // Final evaluation sweep for the final generation
        for (auto &chromo : population)
        {
            bool valid = feasibilityCheck(chromo);
            chromo.setValid(valid);
            if (valid)
            {
                chromo.setFitness(fitness(chromo));
                if (!foundValid || chromo.getFitness() > bestOverall.getFitness())
                {
                    bestOverall = chromo;
                    foundValid = true;
                }
            }
        }

        return bestOverall;
    }
} // namespace ga_toolbox