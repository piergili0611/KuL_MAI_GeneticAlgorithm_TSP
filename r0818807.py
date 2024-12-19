import Reporter
import numpy as np
from algorithm import algorithm

# Modify the class name to match your student number.
class r0818807:

	def __init__(self):
		#self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.num_simulations_to_run = 2
		self.distanceMatrix = None
		self.algorithm = None
		self.filename = None

	def load_distance_matrix(self, filename):
		self.filename = filename
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")

		#replace if there are any infinities in the distance matrix
		distanceMatrix[distanceMatrix == np.inf] = 100000
		file.close()
		self.distanceMatrix = distanceMatrix

	def load_algorithm(self,number_of_cities=None):
		self.algorithm = algorithm(num_cities=number_of_cities)

	def creatAndGet_distnace_matrix(self,filename):
		self.load_distance_matrix(filename=filename)
		return self.distanceMatrix
		

	def run_k_means_algorithm(self, filename):
		self.load_distance_matrix(filename)

		# 1) Create algorithm object and set distance matrix
		self.load_algorithm()
		self.algorithm.set_distance_matrix(distance_matrix=self.distanceMatrix)

		# 2) Add K_clusters model
		self.algorithm.add_K_clusters_model()

		# 3) Run the K_clusters model
		self.algorithm.run_k_cluster_model()

	def test_k_means_algorithm(self, filename):
		self.load_distance_matrix(filename)

		# 1) Create algorithm object and set distance matrix
		self.load_algorithm()
		self.algorithm.set_distance_matrix(distance_matrix=self.distanceMatrix)

		# 2) ACreate cities and test the cluster
		self.algorithm.test_k_cluster_model()

	def run(self,filename,generateDataSets = True,clusters=True,local_search=True):
		'''
		- Run the algorithm
		'''
		if generateDataSets:
			number_of_cities = int(filename.split(".")[0].split("tour")[1])

			self.load_algorithm(number_of_cities=number_of_cities)
			self.algorithm.run_algorithm_main(clusters=clusters,generateDataSets=generateDataSets,local_search=local_search)

		else:
			
			self.load_distance_matrix(filename)

			self.load_algorithm()
			self.algorithm.set_distance_matrix(distance_matrix=self.distanceMatrix)
			
			self.algorithm.run_algorithm_main(clusters=clusters,generateDataSets=generateDataSets,local_search=local_search)
		

	
	def optimize(self,filename):
		'''
		- Run the algorithm
		'''
		self.run(filename=filename,generateDataSets=False,clusters=False,local_search=True)
		
		return 0

	def run_test_multipleTimes(self,filename,outer_iterations=1,test_mutation_rates=False):
		

		self.load_distance_matrix(filename)
		self.load_algorithm()
		self.algorithm.set_distance_matrix(distance_matrix=self.distanceMatrix)

		self.algorithm.run_test_algorithm(number_oftimes=outer_iterations,test_mutation_rates=test_mutation_rates)
		
	def post_processing(self,filename,flag_750):
		
		self.load_distance_matrix(filename)
		self.load_algorithm()
		self.algorithm.set_distance_matrix(distance_matrix=self.distanceMatrix)
		self.algorithm.post_process_csv(flag_750=flag_750)


	# The evolutionary algorithm's main loop
	def optimize_old(self, filename,mutation_prob=0.008):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		

		

		# Your code here.
		yourConvergenceTestsHere = False
		num_iterations = 500
		iterations = 0
		while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
			
			meanObjective = 0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])
			

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			
		#model.plot_fitness_dynamic()
		# Your code here.
		return 0
	
	def run_multiple_times(self,filename,mutation_test=True):
		
		#make me a mutation prob that goes from 0.001 to 0.08 in steps of 0.001
		mutations_prob = np.round(np.linspace(0.0,0.08,80))
		if mutation_test:
			for mutation_prob in mutations_prob:
				for i in range(self.num_simulations_to_run):
					self.optimize(filename,mutation_prob)

		
	


