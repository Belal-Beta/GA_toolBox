#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace ga_toolbox {

template <typename T>
class Chromosome {
public:

    const std::vector<T>& getGenes() 
    {
        return genes; 
    };
    bool isValid()
    {
        return isvalid; 
    };
    Chromosome<T>() 
    {
        fitnessValue = -1; 
        isValid = false; 
    }
    
    protected:
    std::vector<T> genes;
    int fitnessValue;
    bool isvalid; 
};

template <typename T>
using FitnessFunction = std::function<int(const Chromosome<T>&)>;

template <typename T>
using FeasibilityFunction = std::function<bool(const Chromosome<T>&)>;

template <typename T>
class Selection {
public:
    virtual ~Selection() = default;

    virtual std::vector<Chromosome<T>> select(
        const std::vector<Chromosome<T>>& population,
        int numberOfParents) = 0;
};

template <typename T>
class Rank : public Selection<T> {
};

template <typename T>
class Tournament : public Selection<T> {
public:
    std::size_t tournamentSize{};
};

template <typename T>
class Crossover {
public:
    virtual ~Crossover() = default;

    virtual std::pair<Chromosome<T>, Chromosome<T>> cross(
        const Chromosome<T>& parent1,
        const Chromosome<T>& parent2) = 0;
};

template <typename T>
class OrderCrossover : public Crossover<T> {
};

// The diagram calls this type "UnnamedClass"; it represents a crossover
// implementation that stores crossover points.
template <typename T>
class MultiPointCrossover : public Crossover<T> {
public:
    std::vector<int> crossoverPoints;
};

template <typename T>
class Mutation {
public:
    virtual ~Mutation() = default;

    virtual void mutate(Chromosome<T>& chromosome, double mutationRate) = 0;
};

template <typename T>
class NonUniformFloatMutation : public Mutation<T> {
public:
    double mutationRate{};
};

template <typename T>
class BitFlipMutation : public Mutation<T> {
public:
    double mutationRate{};
};

template <typename T>
class SwapMutation : public Mutation<T> {
public:
    double mutationRate{};
};

template <typename T>
class Replacement {
public:
    virtual ~Replacement() = default;

    virtual void replace(
        const std::vector<Chromosome<T>>& population,
        std::vector<Chromosome<T>>& offSprings) = 0;
};

template <typename T>
class SteadyStateReplacement : public Replacement<T> {
};

template <typename T>
class GenerationalReplacement : public Replacement<T> {
};

template <typename T>
class GA_toolbox {
public:
    std::size_t populationSize{};
    double crossoverRate{};
    double mutationRate{};
    std::size_t maxGenerations{};

private:
    std::unique_ptr<Selection<T>> selectionMethod;
    std::unique_ptr<Mutation<T>> mutationMethod;
    std::unique_ptr<Crossover<T>> crossoverMethod;
    std::unique_ptr<Replacement<T>> replacementMethod;

public:
    Chromosome<T> run(
        std::vector<Chromosome<T>> population,
        FitnessFunction<T> fitness,
        FeasibilityFunction<T> feasibilityCheck);
};

} // namespace ga_toolbox
