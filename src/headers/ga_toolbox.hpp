#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <random> 
using namespace std; 

namespace ga_toolbox {

template <typename T>
class Chromosome {
public:

    const vector<T>& getGenes() 
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
    int getFitness() 
    {
        return fitnessValue;
    }
    vector<T> genes;
    float lowerBound;
    float upperBound;
    protected:
    int fitnessValue;
    bool isvalid; 
};

template <typename T>
using FitnessFunction = function<int(const Chromosome<T>&)>;

template <typename T>
using FeasibilityFunction = function<bool(const Chromosome<T>&)>;

template <typename T>
class Selection {
public:
    virtual ~Selection() = default;

    virtual vector<Chromosome<T>> select(
        const vector<Chromosome<T>>& population,
        int numberOfParents) = 0;
};

template <typename T>
class RankSelection : public Selection<T> {
public:
    vector<Chromosome<T>> select(
        const vector<Chromosome<T>>& population,
        int numberOfParents) override;
    };

template <typename T>
class TournamentSelection : public Selection<T> {
public:
    vector<Chromosome<T>> select(
        const vector<Chromosome<T>>& population,
        int numberOfParents) override;
    size_t tournamentSize{};
};

template <typename T>
class Crossover {
protected: 
    double crossoverRate; 
public:
    virtual ~Crossover() = default;

    virtual pair<Chromosome<T>, Chromosome<T>> cross(
        const Chromosome<T>& parent1,
        const Chromosome<T>& parent2) = 0;
    Crossover(double pc) : crossoverRate(pc){}
    };

template <typename T>
class OrderCrossover : public Crossover<T> {
public:     
    pair<Chromosome<T>, Chromosome<T>> cross(
        const Chromosome<T>& parent1,
        const Chromosome<T>& parent2);
    OrderCrossover(double pc) : Crossover<T>(pc){}
};

template <typename T>
class MultiPointCrossover : public Crossover<T> {
public:
    vector<int> crossoverPoints;
    pair<Chromosome<T>, Chromosome<T>> cross(
        const Chromosome<T>& parent1,
        const Chromosome<T>& parent2);
    MultiPointCrossover(double pc) : Crossover<T>(pc){}
};

template <typename T>
class Mutation {
protected: 
    double mutationRate; 
public:
    virtual ~Mutation() = default;

    virtual void mutate(Chromosome<T>& chromosome) = 0;
    Mutation(double pr) : mutationRate(pr){}
};

class UniformMutation : public Mutation<float> {
public:
    void mutate(Chromosome<float>& chromosome);
    UniformMutation(double pr) : Mutation<float>(pr){}
};

class BitFlipMutation : public Mutation<bool> {
public:
    void mutate(Chromosome<bool>& chromosome);
    BitFlipMutation(double pr) : Mutation<bool>(pr){}
};

class SwapMutation : public Mutation<int> {
public:
    void mutate(Chromosome<int>& chromosome);
    SwapMutation(double pr) : Mutation<int>(pr){}
};

template <typename T>
class Replacement {
public:
    virtual ~Replacement() = default;

    virtual void replace(
        const vector<Chromosome<T>>& population,
        vector<Chromosome<T>>& offSprings) = 0;
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
    size_t populationSize{};
    double crossoverRate{};
    double mutationRate{};
    size_t maxGenerations{};

private:
    unique_ptr<Selection<T>> selectionMethod;
    unique_ptr<Mutation<T>> mutationMethod;
    unique_ptr<Crossover<T>> crossoverMethod;
    unique_ptr<Replacement<T>> replacementMethod;

public:
    Chromosome<T> run(
        vector<Chromosome<T>> population,
        FitnessFunction<T> fitness,
        FeasibilityFunction<T> feasibilityCheck);
};

} // namespace ga_toolbox
