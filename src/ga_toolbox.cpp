#include "ga_toolbox.hpp"
using namespace std; 
namespace ga_toolbox {


template <typename T>
vector<Chromosome<T>> RankSelection<T>::select(
    const vector<Chromosome<T>>& population,
    int numberOfParents)
{
    vector<Chromosome<T>> parents; 

}

template <typename T>
pair<Chromosome<T>, Chromosome<T>> MultiPointCrossover<T>::cross(
    const Chromosome<T>& parent1,
    const Chromosome<T>& parent2)
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
        for(int i = 0; i < nPoints; i++)
        {
            const size_t point = nPoints(gen); 
            crossoverPoints.push_back(point); 
        }
        sort(crossoverPoints.begin(), crossoverPoints.end()); 
        int j = 0; 
        for(size_t i = 0; i < n-1; ++i)
        {
            offspring1.genes[i] = parent1.genes[i]; 
            offspring2.genes[i] = parent2.genes[i]; 
            if(i == crossoverPoints[j])
            {
                swap(parent1, parent2); 
                j++;
            }
        }
    }
    return {offspring1, offspring2}; 
}

template <typename T> 
void fillOffspring(const Chromosome<T>& parent_giver, Chromosome<T>& offspring, int start, int end) {
    int size = parent_giver.size();
    
    unordered_set<int> existing_genes;
    for (int i = start; i <= end; ++i) {
        existing_genes.insert(offspring[i]);
    }

    int current_parent_idx = (end + 1) % size;
    int current_offspring_idx = (end + 1) % size;

    for (int i = 0; i < size; ++i) {
        int candidate_gene = parent_giver[current_parent_idx];
        
        if (existing_genes.find(candidate_gene) == existing_genes.end()) {
            offspring[current_offspring_idx] = candidate_gene;
            current_offspring_idx = (current_offspring_idx + 1) % size;
        }
        current_parent_idx = (current_parent_idx + 1) % size;
    }
}


template <typename T> 
pair<Chromosome<T>, Chromosome<T>> OrderCrossover<T>::cross(
    const Chromosome<T>& parent1,
    const Chromosome<T>& parent2)
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
        if(start > end) swap(start, end); 
        for(int i = start; i <= end; i++)
        {
            offspring1.genes[i] = parent1.genes[i];
            offspring2.genes[i] = parent2.genes[i];
        }
        fillOffspring(parent2, offspring1, start, end); 
        fillOffspring(parent1, offspring2, start, end); 
    }
    return {offspring1, offspring2}; 
}


void BitFlipMutation::mutate(Chromosome<bool>& chromosome)
{
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution shouldMutate(mutationRate);

    for(size_t i = 0; i < chromosome.genes.size()-1; ++i)
    {
        if(shouldMutate(gen))
            chromosome.genes[i] = !chromosome.genes[i];
    }
}


void SwapMutation::mutate(Chromosome<int>& chromosome)
{
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution shouldMutate(mutationRate);
    const size_t n = chromosome.genes.size();
    uniform_int_distribution<size_t> randomIndex(0, n - 1);

    for(size_t i = 0; i < n-1; ++i)
    {
        if (shouldMutate(gen))
        {
            const size_t j = randomIndex(gen);
            swap(chromosome.genes[i], chromosome.genes[j]);
        }
    }
}


void UniformMutation::mutate(Chromosome<float>& chromosome)
{
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution shouldMutate(mutationRate);
    const size_t n = chromosome.genes.size();
    for(size_t i = 0; i < n-1; i++)
    {
        if(shouldMutate(gen))
        {
            float dlower = chromosome.lowerBound - chromosome.genes[i]; 
            float dupper = chromosome.upperBound - chromosome.genes[i]; 
            uniform_real_distribution<double> distrib(0.0, 1.0);
            double direction = distrib(gen);
            double delta = (direction <= 0.5)? dlower : dupper;
            uniform_real_distribution<double> step(0.0, delta);
            double r2 = step(gen); 
            chromosome.genes[i] += r2; 
        }
    }
}

template <typename T>
void Replacement<T>::replace(
    const vector<Chromosome<T>>& population,
    vector<Chromosome<T>>& offSprings)
{
}

template <typename T>
Chromosome<T> GA_toolbox<T>::run(
    vector<Chromosome<T>> population,
    FitnessFunction<T> fitness,
    FeasibilityFunction<T> feasibilityCheck)
{
}

} // namespace ga_toolbox
