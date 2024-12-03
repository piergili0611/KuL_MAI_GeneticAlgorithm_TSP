from pygad import *

class GA_pyGad: 
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]  
        self.best_fitness = None
        self.mean_fitness = None    
    



    def fitness_function(self,ga_instance,solution, solution_idx):
        total_distance = 0
        for i in range(len(solution)):
            start_city = solution[i - 1]  # Previous city
            end_city = solution[i]        # Current city
            total_distance += self.distance_matrix[start_city][end_city]
        return 1.0 / total_distance  # Minimize distance

    def create_ga_instance(self):
        # Genetic Algorithm parameters
        ga_instance = pygad.GA(
            num_generations=1000*5,
            num_parents_mating=15,
            fitness_func=self.fitness_function,
            sol_per_pop=15,
            num_genes=self.num_cities,
            gene_type=int,
            gene_space=list(range(self.num_cities)),
            parent_selection_type="tournament",
            K_tournament=3,

            crossover_type="two_points",
            mutation_type="random",
            mutation_percent_genes=10
        )
        return ga_instance

    def run(self):
        ga_instance = self.create_ga_instance()
        ga_instance.run()
        self.best_solution, self.best_fitness, _ = ga_instance.best_solution()
        self.route_distance = 1 / self.best_fitness
        print("Best distance: {0}".format(self.route_distance))
        self.plot_fitness(ga_instance)

    def plot_fitness(self,ga_instance):
        ga_instance.plot_fitness(title="PyGAD: Fitness Progression", linewidth=2) 

   

