import random

POPULATION_SIZE = 100
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
TARGET = "Mithilesh Chauhan"

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    @classmethod
    def create_random_gene(cls):
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        return [cls.create_random_gene() for _ in range(len(TARGET))]

    def mate(self, partner):
        child_chromosome = []

        for gene1, gene2 in zip(self.chromosome, partner.chromosome):
            prob = random.random()

            if prob < 0.45:
                child_chromosome.append(gene1)
            elif prob < 0.90:
                child_chromosome.append(gene2)
            else:
                child_chromosome.append(self.create_random_gene())

        return Individual(child_chromosome)

    def calculate_fitness(self):
        return sum(1 for gs, gt in zip(self.chromosome, TARGET) if gs != gt)

def main():
    global POPULATION_SIZE
    generation = 1
    found = False
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]

    while not found:
        population.sort(key=lambda x: x.fitness)

        if population[0].fitness == 0:
            found = True
            break

        new_generation = population[:int(0.10 * POPULATION_SIZE)]

        for _ in range(int(0.90 * POPULATION_SIZE)):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        generation += 1

    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

if __name__ == '__main__':
    main()
