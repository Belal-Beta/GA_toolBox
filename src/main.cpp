#include <iostream>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <random>
#include <unordered_set>
#include <numeric>

using namespace std;

namespace ga_toolbox
{
    template <typename T>
    class Selection;
    template <typename T>
    class Mutation;
    template <typename T>
    class Crossover;
    template <typename T>
    class Replacement;

    template <typename T>
    class Chromosome
    {
    public:
        vector<T> genes;
        float lowerBound = 0.0f;
        float upperBound = 0.0f;

        Chromosome()
        {
            fitnessValue = -1;
            isvalid = false;
        }

        const vector<T> &getGenes() const { return genes; }
        bool isValid() const { return isvalid; }
        int getFitness() const { return fitnessValue; }
        void setFitness(int f) { fitnessValue = f; }
        void setValid(bool v) { isvalid = v; }

    protected:
        int fitnessValue;
        bool isvalid;
    };

    template <typename T>
    using FitnessFunction = function<int(const Chromosome<T> &)>;

    template <typename T>
    using FeasibilityFunction = function<bool(const Chromosome<T> &)>;

    template <typename T>
    class Selection
    {
    public:
        virtual ~Selection() = default;
        virtual vector<Chromosome<T>> select(
            const vector<Chromosome<T>> &population,
            int numberOfParents) = 0;
    };

    template <typename T>
    class RankSelection : public Selection<T>
    {
    public:
        vector<Chromosome<T>> select(
            const vector<Chromosome<T>> &population,
            int numberOfParents) override;
    };

    template <typename T>
    class TournamentSelection : public Selection<T>
    {
    public:
        size_t tournamentSize{};
        vector<Chromosome<T>> select(
            const vector<Chromosome<T>> &population,
            int numberOfParents) override;
    };

    template <typename T>
    class Crossover
    {
    protected:
        double crossoverRate;
    public:
        Crossover(double pc) : crossoverRate(pc) {}
        virtual ~Crossover() = default;
        virtual pair<Chromosome<T>, Chromosome<T>> cross(
            const Chromosome<T> &parent1,
            const Chromosome<T> &parent2) = 0;
    };

    template <typename T>
    class OrderCrossover : public Crossover<T>
    {
    public:
        OrderCrossover(double pc) : Crossover<T>(pc) {}
        pair<Chromosome<T>, Chromosome<T>> cross(
            const Chromosome<T> &parent1,
            const Chromosome<T> &parent2) override;
    };

    template <typename T>
    class MultiPointCrossover : public Crossover<T>
    {
    public:
        vector<int> crossoverPoints;
        MultiPointCrossover(double pc) : Crossover<T>(pc) {}
        pair<Chromosome<T>, Chromosome<T>> cross(
            const Chromosome<T> &parent1,
            const Chromosome<T> &parent2) override;
    };

    template <typename T>
    class Mutation
    {
    protected:
        double mutationRate;
    public:
        Mutation(double pr) : mutationRate(pr) {}
        virtual ~Mutation() = default;
        virtual void mutate(Chromosome<T> &chromosome) = 0;
    };

    class UniformMutation : public Mutation<float>
    {
    public:
        UniformMutation(double pr) : Mutation<float>(pr) {}
        void mutate(Chromosome<float> &chromosome) override;
    };

    class BitFlipMutation : public Mutation<bool>
    {
    public:
        BitFlipMutation(double pr) : Mutation<bool>(pr) {}
        void mutate(Chromosome<bool> &chromosome) override;
    };

    class SwapMutation : public Mutation<int>
    {
    public:
        SwapMutation(double pr) : Mutation<int>(pr) {}
        void mutate(Chromosome<int> &chromosome) override;
    };

    template <typename T>
    class Replacement
    {
    public:
        virtual ~Replacement() = default;
        virtual void replace(
            vector<Chromosome<T>> &population,
            vector<Chromosome<T>> &offSprings) = 0;
    };

    template <typename T>
    class SteadyStateReplacement : public Replacement<T>
    {
    public:
        void replace(
            vector<Chromosome<T>> &population,
            vector<Chromosome<T>> &offSprings) override;
    };

    template <typename T>
    class GenerationalReplacement : public Replacement<T>
    {
    public:
        void replace(
            vector<Chromosome<T>> &population,
            vector<Chromosome<T>> &offSprings) override;
    };

    template <typename T>
    class GA_toolbox
    {
    public:
        size_t populationSize{};
        double crossoverRate{};
        double mutationRate{};
        size_t maxGenerations{};

        void setSelection(unique_ptr<Selection<T>> s) { selectionMethod = move(s); }
        void setMutation(unique_ptr<Mutation<T>> m) { mutationMethod = move(m); }
        void setCrossover(unique_ptr<Crossover<T>> c) { crossoverMethod = move(c); }
        void setReplacement(unique_ptr<Replacement<T>> r) { replacementMethod = move(r); }

        Chromosome<T> run(
            vector<Chromosome<T>> population,
            FitnessFunction<T> fitness,
            FeasibilityFunction<T> feasibilityCheck);

    private:
        unique_ptr<Selection<T>> selectionMethod;
        unique_ptr<Mutation<T>> mutationMethod;
        unique_ptr<Crossover<T>> crossoverMethod;
        unique_ptr<Replacement<T>> replacementMethod;
    };


    template <typename T>
    vector<Chromosome<T>> RankSelection<T>::select(
        const vector<Chromosome<T>> &population,
        int numberOfParents)
    {
        vector<Chromosome<T>> parents;
        if (population.empty() || numberOfParents <= 0) return parents;
        parents.reserve(numberOfParents);

        vector<size_t> indices(population.size());
        iota(indices.begin(), indices.end(), 0);

        sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { 
            return population[a].getFitness() < population[b].getFitness(); 
        });

        vector<double> weights(population.size());
        for (size_t i = 0; i < population.size(); ++i) {
            weights[indices[i]] = static_cast<double>(i + 1);
        }

        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<size_t> dist(weights.begin(), weights.end());

        for (int i = 0; i < numberOfParents; ++i) {
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
        if (population.empty() || numberOfParents <= 0) return parents;
        parents.reserve(numberOfParents);

        size_t tSize = (this->tournamentSize > 0) ? this->tournamentSize : 2;

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<size_t> dist(0, population.size() - 1);

        for (int i = 0; i < numberOfParents; ++i) {
            size_t best_idx = dist(gen);
            int best_fitness = population[best_idx].getFitness();

            for (size_t j = 1; j < tSize; ++j) {
                size_t candidate_idx = dist(gen);
                int candidate_fitness = population[candidate_idx].getFitness();

                if (candidate_fitness > best_fitness) {
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
        bernoulli_distribution shouldCross(this->crossoverRate);

        const size_t n = parent1.genes.size();
        offspring1.genes = parent1.genes;
        offspring2.genes = parent2.genes;

        if (shouldCross(gen) && n > 2) {
            uniform_int_distribution<size_t> nPointsDist(1, n - 2);
            size_t numberOfPoints = nPointsDist(gen);
            
            crossoverPoints.clear();
            for (size_t i = 0; i < numberOfPoints; i++) {
                crossoverPoints.push_back(nPointsDist(gen));
            }
            sort(crossoverPoints.begin(), crossoverPoints.end());
            
            bool swap_genes = false;
            size_t point_idx = 0;
            
            for (size_t i = 0; i < n; ++i) {
                if (point_idx < crossoverPoints.size() && i == crossoverPoints[point_idx]) {
                    swap_genes = !swap_genes;
                    point_idx++;
                }
                if (swap_genes) {
                    swap(offspring1.genes[i], offspring2.genes[i]);
                }
            }
        }
        return {offspring1, offspring2};
    }

    template <typename T>
    void fillOffspring(const Chromosome<T> &parent_giver, Chromosome<T> &offspring, int start, int end)
    {
        int size = parent_giver.genes.size();
        unordered_set<T> existing_genes;
        
        for (int i = start; i <= end; ++i) {
            existing_genes.insert(offspring.genes[i]);
        }

        int current_parent_idx = (end + 1) % size;
        int current_offspring_idx = (end + 1) % size;

        for (int i = 0; i < size; ++i) {
            T candidate_gene = parent_giver.genes[current_parent_idx];

            if (existing_genes.find(candidate_gene) == existing_genes.end()) {
                offspring.genes[current_offspring_idx] = candidate_gene;
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
        bernoulli_distribution shouldCross(this->crossoverRate);
        
        const size_t n = parent1.genes.size();
        offspring1.genes.resize(n);
        offspring2.genes.resize(n);

        if (shouldCross(gen) && n > 2) {
            uniform_int_distribution<size_t> nPointsDist(1, n - 2);
            int start = nPointsDist(gen);
            int end = nPointsDist(gen);
            if (start > end) swap(start, end);

            for (int i = start; i <= end; i++) {
                offspring1.genes[i] = parent1.genes[i];
                offspring2.genes[i] = parent2.genes[i];
            }
            fillOffspring(parent2, offspring1, start, end);
            fillOffspring(parent1, offspring2, start, end);
        } else {
            offspring1.genes = parent1.genes;
            offspring2.genes = parent2.genes;
        }
        return {offspring1, offspring2};
    }

    void BitFlipMutation::mutate(Chromosome<bool> &chromosome)
    {
        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution shouldMutate(mutationRate);

        for (size_t i = 0; i < chromosome.genes.size(); ++i) {
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

        for (size_t i = 0; i < n; ++i) {
            if (shouldMutate(gen)) {
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
        for (size_t i = 0; i < n; i++) {
            if (shouldMutate(gen)) {
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
        if (population.empty() || offSprings.empty()) return;

        sort(offSprings.begin(), offSprings.end(),
            [](const Chromosome<T> &left, const Chromosome<T> &right) {
                return left.getFitness() > right.getFitness();
            });

        size_t replacementCount = min(population.size(), offSprings.size());

        for (size_t i = 0; i < replacementCount; ++i) {
            auto worst = min_element(
                population.begin(),
                population.end(),
                [](const Chromosome<T> &left, const Chromosome<T> &right) {
                    return left.getFitness() < right.getFitness();
                });
            *worst = offSprings[i];
        }
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
        if (!selectionMethod || !crossoverMethod || !mutationMethod || !replacementMethod) {
            throw runtime_error("GA components (Selection, Crossover, Mutation, Replacement) must be initialized.");
        }

        Chromosome<T> bestOverall;
        bool foundValid = false;

        for (size_t generation = 0; generation < maxGenerations; ++generation) {
            for (auto &chromo : population) {
                bool valid = feasibilityCheck(chromo);
                chromo.setValid(valid);

                if (valid) {
                    chromo.setFitness(fitness(chromo));
                } else {
                    chromo.setFitness(-1);
                }

                if (chromo.isValid()) {
                    if (!foundValid || chromo.getFitness() > bestOverall.getFitness()) {
                        bestOverall = chromo;
                        foundValid = true;
                    }
                }
            }

            if (population.empty()) break;

            vector<Chromosome<T>> parents = selectionMethod->select(population, populationSize);
            vector<Chromosome<T>> offspring;
            offspring.reserve(populationSize);

            for (size_t i = 0; i < parents.size(); i += 2) {
                if (i + 1 < parents.size()) {
                    auto [child1, child2] = crossoverMethod->cross(parents[i], parents[i + 1]);
                    offspring.push_back(child1);
                    offspring.push_back(child2);
                } else {
                    offspring.push_back(parents[i]);
                }
            }

            for (auto &child : offspring) {
                mutationMethod->mutate(child);
            }

            replacementMethod->replace(population, offspring);
        }

        for (auto &chromo : population) {
            bool valid = feasibilityCheck(chromo);
            chromo.setValid(valid);
            if (valid) {
                chromo.setFitness(fitness(chromo));
                if (!foundValid || chromo.getFitness() > bestOverall.getFitness()) {
                    bestOverall = chromo;
                    foundValid = true;
                }
            }
        }
        return bestOverall;
    }
} 


using namespace ga_toolbox;


const int NUM_VEHICLES = 4;
const int VEHICLE_CAPACITY = 15;
const int DEPOT = 0;
const int NUM_CUSTOMERS = 16;

struct Node { int x, y; };
const vector<Node> LOCATIONS = {
    {456, 320}, {228, 0}, {912, 0}, {0, 80}, {114, 80}, {570, 160}, 
    {798, 160}, {342, 240}, {684, 240}, {570, 400}, {912, 400}, 
    {114, 480}, {228, 480}, {342, 560}, {684, 560}, {0, 640}, {798, 640}
};

const vector<int> DEMANDS = {
    0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8
};

int getDistance(int from, int to) {
    long long dx = LOCATIONS[from].x - LOCATIONS[to].x;
    long long dy = LOCATIONS[from].y - LOCATIONS[to].y;
    return static_cast<int>(std::round(std::sqrt(dx*dx + dy*dy)));
}

vector<vector<int>> decodeChromosome(const Chromosome<int>& chromo) {
    vector<vector<int>> routes;
    vector<int> current_route;
    int current_load = 0;

    for (int customer : chromo.genes) {
        if (current_load + DEMANDS[customer] > VEHICLE_CAPACITY) {
            routes.push_back(current_route);
            current_route.clear();
            current_load = 0;
        }
        current_route.push_back(customer);
        current_load += DEMANDS[customer];
    }
    if (!current_route.empty()) {
        routes.push_back(current_route);
    }
    return routes;
}

bool cvrpFeasible(const Chromosome<int>& chromo) {
    auto routes = decodeChromosome(chromo);
    return routes.size() <= NUM_VEHICLES;
}

int cvrpFitness(const Chromosome<int>& chromo) {
    auto routes = decodeChromosome(chromo);
    int total_distance = 0;
    
    for (const auto& route : routes) {
        int current_node = DEPOT;
        for (int customer : route) {
            total_distance += getDistance(current_node, customer);
            current_node = customer;
        }
        total_distance += getDistance(current_node, DEPOT);
    }
    
    return 100000 - total_distance; 
}

void printORToolsFormat(const Chromosome<int>& best) {
    auto routes = decodeChromosome(best);
    int total_distance = 100000 - best.getFitness();
    int total_load = 0;
    
    cout << "Objective: " << total_distance << "m\n\n";
    
    for (size_t i = 0; i < routes.size(); ++i) {
        cout << "Route for vehicle " << i << ":\n";
        cout << " " << DEPOT;
        
        int route_dist = 0;
        int route_load = 0;
        int current_node = DEPOT;
        
        for (int customer : routes[i]) {
            cout << " -> " << customer;
            route_dist += getDistance(current_node, customer);
            route_load += DEMANDS[customer];
            current_node = customer;
        }
        cout << " -> " << DEPOT << "\n";
        route_dist += getDistance(current_node, DEPOT);
        
        cout << "Distance of the route: " << route_dist << "m\n";
        cout << "Load of the route: " << route_load << "\n\n";
        total_load += route_load;
    }
    cout << "Total distance of all routes: " << total_distance << "m\n";
    cout << "Total load of all routes: " << total_load << "\n";
}

int main() {
    const int POP_SIZE = 150;
    
    vector<Chromosome<int>> initPopulation(POP_SIZE);
    random_device rd;
    mt19937 gen(rd());
    
    vector<int> base_permutation(NUM_CUSTOMERS);
    iota(base_permutation.begin(), base_permutation.end(), 1); 

    for (int i = 0; i < POP_SIZE; ++i) {
        shuffle(base_permutation.begin(), base_permutation.end(), gen);
        initPopulation[i].genes = base_permutation;
        initPopulation[i].setValid(true);
    }

    GA_toolbox<int> ga;
    ga.populationSize = POP_SIZE;
    ga.crossoverRate = 0.85;
    ga.mutationRate = 0.15;
    ga.maxGenerations = 500;

    auto selection = make_unique<TournamentSelection<int>>();
    selection->tournamentSize = 5;
    ga.setSelection(move(selection));
    ga.setCrossover(make_unique<OrderCrossover<int>>(ga.crossoverRate));
    ga.setMutation(make_unique<SwapMutation>(ga.mutationRate));
    ga.setReplacement(make_unique<GenerationalReplacement<int>>());

    cout << "Solving CVRP with GA Toolbox...\n\n";
    Chromosome<int> best = ga.run(initPopulation, cvrpFitness, cvrpFeasible);
    printORToolsFormat(best);

    return 0;
}