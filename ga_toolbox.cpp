#include "ga_toolbox.hpp"

namespace ga_toolbox {


template <typename T>
std::vector<Chromosome<T>> Selection<T>::select(
    const std::vector<Chromosome<T>>& population,
    int numberOfParents)
{
}

template <typename T>
std::pair<Chromosome<T>, Chromosome<T>> Crossover<T>::cross(
    const Chromosome<T>& parent1,
    const Chromosome<T>& parent2)
{
}

template <typename T>
void Mutation<T>::mutate(Chromosome<T>& chromosome, double mutationRate)
{
}

template <typename T>
void Replacement<T>::replace(
    const std::vector<Chromosome<T>>& population,
    std::vector<Chromosome<T>>& offSprings)
{
}

template <typename T>
Chromosome<T> GA_toolbox<T>::run(
    std::vector<Chromosome<T>> population,
    FitnessFunction<T> fitness,
    FeasibilityFunction<T> feasibilityCheck)
{
}

} // namespace ga_toolbox
