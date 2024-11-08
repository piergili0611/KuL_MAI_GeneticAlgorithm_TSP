import Reporter
import numpy as np
from Evol_algorithm_K import GA_K

# Modify the class name to match your student number.
class r0818807:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.num_simulations_to_run = 2

	def k_means_algorithm(self, filename):
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		

		model = GA_K(seed=42)
		model.set_distance_matrix(distanceMatrix)
		#model.k_means_distanced(distance_matrix=distanceMatrix)
		#model.iterative_refinement(distance_matrix=distanceMatrix)
		model.k_medoids_clustering()

		

	# The evolutionary algorithm's main loop
	def optimize(self, filename,mutation_prob=0.008):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		

		model = GA_K(mutation_prob=mutation_prob,seed=42)
		model.set_distance_matrix(distanceMatrix)
		model.set_initialization()
		model.plot_distance_matrix()
		

		# Your code here.
		yourConvergenceTestsHere = False
		num_iterations = 500
		iterations = 0
		while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
			'''
			meanObjective = 0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])
			'''
			iterations += 1
			#print(f"\n Iteration number {iterations}")
			parents = model.selection_k_tournament(num_individuals=model.population_size, k=model.k_tournament_k)	
			offspring = model.crossover_singlepoint_population(parents)
			offspring_mutated = model.mutation_singlepoint_population(offspring)

			model.eliminate_population(population=model.population, offsprings=offspring_mutated)
			#model.eliminate_population_elitism(population=model.population, offsprings=offspring_mutated)
			meanObjective, bestObjective , bestSolution  = model.calculate_information_iteration()
			yourConvergenceTestsHere = False

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			'''
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			'''
		model.plot_fitness_dynamic()
		# Your code here.
		return 0
	
	def run_multiple_times(self,filename,mutation_test=True):
		
		#make me a mutation prob that goes from 0.001 to 0.08 in steps of 0.001
		mutations_prob = np.round(np.linspace(0.0,0.08,80))
		if mutation_test:
			for mutation_prob in mutations_prob:
				for i in range(self.num_simulations_to_run):
					self.optimize(filename,mutation_prob)

		
	


