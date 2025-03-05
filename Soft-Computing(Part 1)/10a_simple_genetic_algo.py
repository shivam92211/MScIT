import random

# Number of individuals in each generation
POPULATION_SIZE = 100

# Valid genes
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated
TARGET = "Mithilesh Chauhan"

class Individual(object):
    '''
    Class representing an individual in the population
    '''

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        '''
        Create random genes for mutation
        '''
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(cls):
        '''
        Create chromosome or string of genes
        '''
        global TARGET
        gnome_len = len(TARGET)
        return [cls.mutated_genes() for _ in range(gnome_len)]

    def mate(self, par2):
        '''
        Perform mating and produce new offspring
        '''
        # chromosome for offspring
        child_chromosome = []

        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            # random probability
            prob = random.random()

            # if prob is less than 0.45, insert gene
            # from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)

            # if prob is between 0.45 and 0.90, insert
            # gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)

            # otherwise insert random gene (mutate),
            # for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())

        # create a new Individual (offspring) using
        # the generated chromosome for offspring
        return Individual(child_chromosome)

    def cal_fitness(self):
        '''
        Calculate fitness score, which is the number of
        characters in the string that differ from the target
        string.
        '''
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt:
                fitness += 1
        return fitness

# Driver code
def main():
    global POPULATION_SIZE

    # current generation
    generation = 1
    found = False
    population = []

    # create the initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found:
        # sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness)

        # if the individual has the lowest fitness score (i.e., 0),
        # then we know that we have reached the target
        # and break the loop
        if population[0].fitness <= 0:
            found = True
            break

        # Otherwise, generate new offsprings for the new generation
        new_generation = []

        # Perform Elitism, meaning 10% of the fittest population
        # goes to the next generation
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        # From 50% of the fittest population,
        # individuals will mate to produce offspring
        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print("Generation: {}\tString: {}\tFitness: {}".format(
            generation, "".join(population[0].chromosome), population[0].fitness))

        generation += 1

    print("Generation: {}\tString: {}\tFitness: {}".format(
        generation, "".join(population[0].chromosome), population[0].fitness))

if __name__ == '__main__':
    main()
