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

	def post_processing_histogram(self,filename):
		
		self.load_distance_matrix(filename)
		self.load_algorithm()
		self.algorithm.set_distance_matrix(distance_matrix=self.distanceMatrix)
		self.algorithm.post_process_histogram_csv()


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

		
	




#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
																			# Class algorithm
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================

import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time 
import math
#from Evol_algorithm_K import r0818807
#from Evol_algorithm_L2 import GA_K_L2
#from k_clusters import k_clusters
#from cities import cities
#from clusterdashboard import ClusterDashboard
from concurrent.futures import ProcessPoolExecutor, as_completed
import os,sys
import pandas as pd 

#CSV:
import csv


class algorithm:

    def __init__(self,num_cities=None):
        self.num_city = num_cities
        self.GA_level1_model = None
        self.GA_level2_model = None
        self.K_clusters_model = None
        #self.cluster_dashboard = ClusterDashboard()
        self.distanceMatrix = None
        self.clusters_list = []
        self.distance_matrix_cluster_list = []

        self.deltatime_cluster_list = []
        self.clusters_solution_list = []

        self.cities_cluster_list = []
        

        #To check k_clusterl model
        self.cities_model = None

        #GA level 1
        self.sigma_value = None



    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Tests: ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def test_algorithm(self,generateDataSets=True,clusters=True,local_search=True,number_iterations = 10):
        '''
        - Test the algorithm
        '''
        for i in range(number_iterations):
            print(f"Test iteration: {i}")
            self.run_algorithm_main(generateDataSets=generateDataSets,clusters=clusters,local_search=local_search)
            self.reset_algorithm()

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 1) Distance Matrix ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf_distance_matrix(distance_matrix=distance_matrix,replace_value=100000)
        self.num_city = len(self.distance_matrix)
        #print(f"Distance matrix is {self.distance_matrix}")
        
    def check_inf_distance_matrix(self,distance_matrix, replace=False ,replace_value=1000000):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''

        if replace:
            distance_matrix[distance_matrix == np.inf] = replace_value
        
        return distance_matrix
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 2) K_Cluster Model ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def add_K_clusters_model(self):
        '''
        - Add the K_clusters model
        '''
        model = k_clusters(distance_matrix=self.distance_matrix)
        self.K_clusters_model = model

    
    def run_k_cluster_model(self,model=None,num_clusters=None,min_cluster_size=None):
        '''
        - Run the k_cluster also
        '''

        if model:
            num_cities = int(model.num_cities)
            if num_clusters is None:
                #num_clusters = math.ceil(num_cities/50)
                #available_cores = os.cpu_count()
                available_cores = 2
                num_clusters = math.ceil(available_cores)
            if min_cluster_size is None:
                min_cluster_size = int(num_cities/60)
            #model.run_model( k=num_clusters, min_cluster_size=min_cluster_size) 
            model.run_model_KMedoids(num_clusters=num_clusters)
            cluster_list = model.clusters_list
        else:
            print(f"Please add the K_cluster model")
            cluster_list = None

        return cluster_list
    
  
    
    def add_run_k_cluster_model(self,num_clusters=None):
        '''
        - Add and run the K_cluster model
        '''
        self.add_K_clusters_model()
        self.cluster_list = self.run_k_cluster_model(model=self.K_clusters_model,num_clusters=num_clusters)
        for cluster in self.cluster_list:
            print(f"Cluster: {cluster}")
        self.assigned_cities_list = [cluster['assigned_cities'] for cluster in self.cluster_list]
        print(f"Assigned cities list: {self.assigned_cities_list}")

    def K_cluster_model_generate_distance_matrix_cluster(self):
        '''
        - Generate the distance matrix for the clusters
        '''
        #self.distance_matrix_cluster_list,self.cities_cluster_list = self.K_clusters_model.generate_distance_matrix_cluster(self.cluster_list)
        self.distance_matrix_cluster_list = [self.distance_matrix]
        cities_list = []
        for element in range(len(self.distance_matrix)):
            #print(f"Element: {element}")
            cities_list.append(element)
            
            a = 1
        self.cities_cluster_list.append(cities_list)
        print(f"Distance matrix cluster list: {self.distance_matrix_cluster_list}& Cities cluster list: {self.cities_cluster_list}")

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 3) GA_Level1 ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def add_GA_level1_model(self,distance_matrix,cities,local_search=True,initial_solution=None,max_iterations=1500, mutation_rate=0.1):
        '''
        - Add the GA model
        '''
        if self.sigma_value is not None:
            model = r0818807_GA_Level1(cities=cities,mutation_rate=mutation_rate,seed=42,local_search=local_search,max_iterations=max_iterations,sigma_value= self.sigma_value)
        else:
             model = r0818807_GA_Level1(cities=cities,mutation_rate=mutation_rate,seed=42,local_search=local_search,max_iterations=max_iterations)
        model.set_distance_matrix(distance_matrix)
        if initial_solution is not None:
            model.add_initialSolution(initial_solution=initial_solution)
        self.GA_level1_model = model

    def run_GA_level1_model(self,cities,plot=False):
        '''
        - Run the GA model
        '''
        self.GA_level1_model.run_model(plot=plot)

    def add_run_GA_level1_model(self,distance_matrix,cities,local_search=True,initial_solution=None,max_iterations=50,plot = True,mutation_rate=0.1):
        '''
        - Add and run the GA model
        '''
        
        self.add_GA_level1_model(distance_matrix=distance_matrix,cities=cities,local_search=local_search,initial_solution=initial_solution,max_iterations=max_iterations, mutation_rate=mutation_rate)
        self.run_GA_level1_model(cities,plot=plot)

    def GA_level1_model_retrieveBestSolution(self):
        '''
        - Retrieve the best solution
        '''
        return self.GA_level1_model.best_solution_cities
    
    def GA_level1_model_retrieveBestSolutionAndFitness(self):
        '''
        - Retrieve the best solution
        '''
        return self.GA_level1_model.best_solution_cities,self.GA_level1_model.best_objective,self.GA_level1_model.mean_objective

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 4) GA_Level2 ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def add_GA_level2_model(self,distance_matrix,cluster_solutions_matrix,mutation_prob=0.8,local_search=True):
        '''
        - Add the GA model
        '''
        model = GA_K_L2(clusters_solutions_matrix=cluster_solutions_matrix,cities_model=self.cities_model,mutation_prob=mutation_prob,seed=42,local_search=local_search)
        model.set_distance_matrix(distance_matrix)
        self.GA_level2_model = model

    def run_GA_level2_model(self):
        '''
        - Run the GA model
        '''
        self.GA_level2_model.run_model()
    
    def add_run_GA_level2_model(self,distance_matrix,cluster_solutions_matrix,local_search=True):
        '''
        - Add and run the GA model
        '''
        self.add_GA_level2_model(distance_matrix=distance_matrix,cluster_solutions_matrix=cluster_solutions_matrix,local_search=local_search)
        self.run_GA_level2_model()
    
    def GA_level2_model_retrieveBestSolution(self,fitness=True):
        '''
        - Retrieve the best solution
        '''
        
        return self.GA_level2_model.best_solution_cities, self.GA_level2_model.best_objective, self.GA_level2_model.clusters_solution_matrix

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) City Model ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def add_cities_model(self):
        '''
        - Add the cities model
        '''
        num_cities = int(self.num_city)
        model = cities(num_cities=num_cities)
        self.cities_model = model

    def run_city_model(self,):
        '''
        - Run the city model
        '''
        self.cities_model.generate_distance_matrix()
        self.cities_model.plot_cities()


    def add_run_cities_model(self):
        '''
        - Add and run the cities model
        '''
        self.add_cities_model()
        self.run_city_model()

    def cities_model_generate_dataSets(self):
        '''
        - Create City Model
        - Generate the data sets
        '''
        self.add_run_cities_model()
        self.distance_matrix = self.cities_model.distanceMatrix

    def cities_model_plot_clusters(self):
        '''
        - Plot the clusters
        '''
        self.cities_model.clusters_list = self.cluster_list
        self.cities_model.plot_clusters()

    def cities_model_generateDistanceMatrix(self):
        '''
        - Generate the distance matrix
        '''
        self.distance_matrix_cluster_list, self.cities_cluster_list = self.cities_model.generate_distance_matrix_cluster()

    def cities_model_add_cities_sequence(self,city_sequence):
        '''
        - Add the city sequence
        '''
        self.cities_model.add_cities_sequence(city_sequence)
    
    def cities_model_plot_clusters_sequence(self):
        '''
        - Plot the clusters sequence
        '''
        self.cities_model.plot_clusters_sequence(city_sequence=True)
    
    def cities_model_addAndPlotClustersSequence(self,city_sequence):
        '''
        - Add and plot the clusters sequence
        '''
        self.cities_model_add_cities_sequence(city_sequence)
        self.cities_model_plot_clusters_sequence()
        


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 6) Run Algorithm ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    

    def run_algorithm_main(self,generateDataSets = True,clusters=True,local_search=True):
        '''
        - Run the algorithm
        '''
        if generateDataSets:
            self.run_algorithm_yesGenerateDataSets(clusters=clusters,local_search=local_search)
        else:
            self.run_algorithm_noGenerateDataSets(clusters=clusters,local_search=local_search)


    def run_algorithm_yesGenData(self,clusters=True,local_search=True):
        '''
        - Run the algorithm
        '''
        index = 0
        final_fitness = 0
        final_solution = None
        print(f"Number of clusters: {len(self.cities_cluster_list)}")
        
        print(self.cities_cluster_list)

        for cities in self.cities_cluster_list:
            # 1) Add the GA model and run it for each cluster of cities and measure the time taken
            time_start_GA_level1 = time.time()

            distance_matrix = self.distance_matrix_cluster_list[index]
            self.add_run_GA_level1_model(cities=np.array(cities),distance_matrix=distance_matrix,local_search=local_search)
            best_solution = self.GA_level1_model_retrieveBestSolution()
            self.add_cluster_solution(best_solution)

            time_end_GA_level1 = time.time()
            delta_time = time_end_GA_level1-time_start_GA_level1    
            print(f"Time taken for GA level 1: {delta_time}")
            self.deltatime_cluster_list.append(delta_time)

            self.cities_model_addAndPlotClustersSequence(best_solution)

            index += 1

        self.plot_ExecutionTime_Clusters()

        # 2) Create and run Higher level GA model
        if clusters:
            
            time_start_GA_level2 = time.time()
            self.add_run_GA_level2_model(distance_matrix=self.distance_matrix,cluster_solutions_matrix=self.clusters_solution_list,local_search=local_search)
            final_solution, final_fitness = self.GA_level2_model_retrieveBestSolution(fitness=True)
            time_end_GA_level2 = time.time()
            delta_time = time_end_GA_level2-time_start_GA_level2
            print(f"Time taken for GA level 2: {delta_time}")
        else:
            final_solution = best_solution  
        print(f"Final solution: {final_solution} ")
        self.check_city_solution(final_solution)

        # 2) Plot teh resulting sequence
        self.cities_model_addAndPlotClustersSequence(final_solution)


    def run_algorithm_noGenData(self,clusters=True,local_search=True):
        '''
        - Run the algorithm
        '''
        index = 0
        final_fitness = 0
        final_solution = None

        for cities in self.cities_cluster_list:
            # 1) Add the GA model and run it for each cluster of cities and measure the time taken
            time_start_GA_level1 = time.time()

            distance_matrix = self.distance_matrix_cluster_list[index]
            self.add_run_GA_level1_model(cities=np.array(cities),distance_matrix=distance_matrix,local_search=local_search,max_iterations=1500,plot = True)
            best_solution = self.GA_level1_model_retrieveBestSolution()
            self.add_cluster_solution(best_solution)

            time_end_GA_level1 = time.time()
            delta_time = time_end_GA_level1-time_start_GA_level1    
            print(f"Time taken for GA level 1: {delta_time}")
            self.deltatime_cluster_list.append(delta_time)

            #self.cities_model_addAndPlotClustersSequence(best_solution)

            index += 1

        self.plot_ExecutionTime_Clusters()

        # 2) Create and run Higher level GA model
        if clusters:
            time_start_GA_level2 = time.time()
            self.add_run_GA_level2_model(distance_matrix=self.distance_matrix,cluster_solutions_matrix=self.clusters_solution_list,local_search=local_search)
            final_solution, final_fitness = self.GA_level2_model_retrieveBestSolution(fitness=True)
            time_end_GA_level2 = time.time()
            delta_time = time_end_GA_level2-time_start_GA_level2
            print(f"Time taken for GA level 2: {delta_time}")
            print(f"Final solution: {final_solution} & Final fitness: {final_fitness}")
            self.check_city_solution(final_solution)
        else:
            final_solution = best_solution  
            print(f"Final solution: {final_solution} ")
            self.check_city_solution(final_solution)

        # 2) Plot teh resulting sequence
        #self.cities_model_addAndPlotClustersSequence(final_solution)





        

    def run_algorithm_yesGenerateDataSets(self,clusters=True,local_search=True):
        '''
        - Run the algorithm with generating data sets
        '''
        print(f"-- Running the algorithm: Yes generateDataSets --")

        self.cities_model_generate_dataSets()
        if clusters:
            self.add_run_k_cluster_model()
        else:
            self.add_run_k_cluster_model(num_clusters=1)

        self.cities_model_plot_clusters()
        self.cities_model_generateDistanceMatrix()

        print(f"-- Running the algorithm: generatedDatsSets --")
        self.run_algorithm_yesGenData(clusters=clusters,local_search=local_search)


    def run_algorithm_noGenerateDataSets(self,clusters=True,local_search=True):
        '''
        - Run the algorithm without generating data sets, --> KU Leuven DataSets
        '''
        print(f"-- Running the algorithm: No generatedDatsSets --")


        if clusters:
            #self.add_run_k_cluster_model()
            a = 1
        else:
            #self.add_run_k_cluster_model(num_clusters=1)
            a = 0
        
        self.K_cluster_model_generate_distance_matrix_cluster()

        print(f"-- Running the algorithm: generatedDatsSets --")
        #self.run_algorithm_noGenData_parallel(clusters=clusters,local_search=local_search)
        self.run_algorithm_noGenData(clusters=clusters,local_search=local_search)





    def run_algorithm_noGenData_parallel(self, clusters=True, local_search=True):
        '''
        - Run the algorithm with parallelized GA_Level1
        '''
        final_fitness = 0
        final_solution = None
        futures = []
        print(f"self.cities_cluster_list: {self.cities_cluster_list}")

        # Create an executor for parallel processing
        #Detect available CPU cores
        #available_cores = os.cpu_count()
        available_cores = 2
        max_workers = available_cores
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, cities in enumerate(self.cities_cluster_list):
                distance_matrix = self.distance_matrix_cluster_list[index]
                # Submit each GA_Level1 task to the executor
                futures.append(
                    executor.submit(
                        self.run_GA_Level1_task, 
                        np.array(cities), 
                        distance_matrix, 
                        local_search
                    )
                )

            # Collect results as they complete
            for future in as_completed(futures):
                best_solution, delta_time = future.result()
                self.add_cluster_solution(best_solution)
                self.deltatime_cluster_list.append(delta_time)
                print(f"Time taken for GA level 1: {delta_time}")

        self.plot_ExecutionTime_Clusters()

        # 2) Create and run Higher level GA model
        if clusters:
            time_start_GA_level2 = time.time()
            self.add_run_GA_level2_model(
                distance_matrix=self.distance_matrix,
                cluster_solutions_matrix=self.clusters_solution_list,
                local_search=local_search
            )
            final_solution, final_fitness, cities = self.GA_level2_model_retrieveBestSolution(fitness=True)
            time_end_GA_level2 = time.time()
            delta_time = time_end_GA_level2 - time_start_GA_level2
            print(f"Time taken for GA level 2: {delta_time}")
            print(f"Final solution: {final_solution} & Final fitness: {final_fitness}")
            self.check_city_solution(final_solution)
            self.add_run_GA_level1_model(cities=final_solution, distance_matrix=self.distance_matrix, 
                                        local_search=local_search,initial_solution = final_solution,max_iterations = 500,plot = True)
        else:
            final_solution = best_solution  
            print(f"Final solution: {final_solution}")
            self.check_city_solution(final_solution)

    def run_GA_Level1_task(self, cities, distance_matrix, local_search):
        '''
        Function to run GA_Level1 task. Returns the best solution and elapsed time.
        '''
        time_start = time.time()
        self.add_run_GA_level1_model(cities=cities, distance_matrix=distance_matrix, local_search=local_search,plot=True,max_iterations=500)
        best_solution = self.GA_level1_model_retrieveBestSolution()
        time_end = time.time()
        delta_time = time_end - time_start
        return best_solution, delta_time
    

    
    def check_city_solution(self,city_solution):
        '''
        - Check the city solution
        '''
        num_cities = len(city_solution)
        unique_cities = len(set(city_solution))
        print(f"Number of cities: {num_cities} & Number of unique cities: {unique_cities}")
        

    def test_k_cluster_model(self,clusters=True):
        '''
        - Test the k_cluster model
        '''

        self.add_run_cities_model()
        distance_matrix = self.cities_model.distanceMatrix
        
        
        K_cluster_model = k_clusters(distance_matrix=distance_matrix)
        if clusters is False:
            num_clusters = 1
            self.cluster_list = self.run_k_cluster_model(K_cluster_model,num_clusters=num_clusters)
        else:
            self.cluster_list = self.run_k_cluster_model(K_cluster_model)
        

        # 2) Add the clusters to the cities model and plot them
        self.cities_model.show_clusters(self.cluster_list)
        self.distance_matrix_cluster_list,self.cities_cluster_list = self.cities_model.generate_distance_matrix_cluster()
        print(f"Distance matrix cluster list: {self.distance_matrix_cluster_list}& Cities cluster list: {self.cities_cluster_list}")
        return distance_matrix
    

    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 6) Test Algorithm: Multiple runs ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_test_algorithm(self,number_oftimes=10,test_mutation_rates=False):
        '''
        - Run the test algorithm
        '''
        #Variables
        best_fitness_list = []
        best_route_list = []
        mean_fitness_list = []
        mutation_rates_list = [0.1,0.5,0.8]
        sigma_list = [0.1,0.5,0.8]
        results_dict = {}


        if test_mutation_rates:
            
            for mutation_rate in mutation_rates_list:
                print(f"\n Mutation rate: {mutation_rate}")
                
                for sigma in sigma_list:    
                    print(f"\n Sigma: {sigma}")
                    self.sigma_value = sigma
                    best_fitness_list = []
                    best_route_list = []
                    mean_fitness_list = []
                    results_dict = {}

                    self.save_info_csv(results_dict,append=False,filename=f"{mutation_rate}{sigma}.csv")

                    for i in range(number_oftimes):
                        print(f"\n Test iteration: {i}")
                        best_route, best_fitness, mean_fitness = self.run_algorithm_main_test(generateDataSets=False,clusters=False,local_search=True,mutation_rate = mutation_rate)
                        best_route_list.append(best_route)
                        best_fitness_list.append(best_fitness)
                        mean_fitness_list.append(mean_fitness)
                        results_dict[i] = [best_fitness,mean_fitness]
                    self.plot_test_lineplot(best_fitness_list=best_fitness_list,mean_fitness_list= mean_fitness_list)
                    self.plot_test_histogram(best_fitness_list=best_fitness_list,mean_fitness_list=mean_fitness_list)
                    #results_dict[f"Mutation rate: {mutation_rate} & Sigma: {sigma}"] = [best_route_list,best_fitness_list,mean_fitness_list]
                    #print(f"Results dict: {results_dict}")
                    self.save_info_csv(results_dict,append=True,filename=f"{mutation_rate}{sigma}.csv")
        else:
            self.save_info_csv(results_dict,append=False)
            for i in range(number_oftimes):
                print(f"\n Test iteration: {i}")
                best_route, best_fitness, mean_fitness = self.run_algorithm_main_test(generateDataSets=False,clusters=False,local_search=True)
                best_route_list.append(best_route)
                best_fitness_list.append(best_fitness)
                mean_fitness_list.append(mean_fitness)
                results_dict[i] = [best_fitness,mean_fitness]
            self.save_info_csv(results_dict,append=True)
            #print(f"Results dict: {results_dict}")

            self.plot_test_lineplot(best_fitness_list=best_fitness_list,mean_fitness_list= mean_fitness_list)
            self.plot_test_histogram(best_fitness_list=best_fitness_list,mean_fitness_list=mean_fitness_list)




    
    def run_algorithm_main_test(self,generateDataSets = True,clusters=True,local_search=True, mutation_rate = 0.1):
        '''
        - Run the algorithm
        '''
        if generateDataSets:
            best_route,best_fitness,mean_fitness = self.run_algorithm_yesGenerateDataSets(clusters=clusters,local_search=local_search)
        else:
            best_route, best_fitness, mean_fitness = self.run_algorithm_noGenerateDataSets_test(clusters=clusters,local_search=local_search,mutation_rate=mutation_rate)

        return best_route,best_fitness, mean_fitness
    
    def run_algorithm_noGenerateDataSets_test(self,clusters=True,local_search=True,mutation_rate=0.1):
        '''
        - Run the algorithm without generating data sets, --> KU Leuven DataSets
        '''
        print(f"-- Running the algorithm: No generatedDatsSets --")


        if clusters:
            self.add_run_k_cluster_model()
        else:
            self.add_run_k_cluster_model(num_clusters=1)

        
        self.K_cluster_model_generate_distance_matrix_cluster()

        print(f"-- Running the algorithm: generatedDatsSets --")
        #self.run_algorithm_noGenData_parallel(clusters=clusters,local_search=local_search)
        best_route,best_fitness, mean_fitness = self.run_algorithm_noGenData_test(clusters=clusters,local_search=local_search, mutation_rate = mutation_rate)

        return best_route,best_fitness, mean_fitness
    
    def run_algorithm_noGenData_test(self,clusters=True,local_search=True, mutation_rate = 0.1):
        '''
        - Run the algorithm
        '''
        index = 0
        final_fitness = 0
        final_solution = None

        for cities in self.cities_cluster_list:
            # 1) Add the GA model and run it for each cluster of cities and measure the time taken
            time_start_GA_level1 = time.time()

            distance_matrix = self.distance_matrix_cluster_list[index]
            self.add_run_GA_level1_model(cities=np.array(cities),distance_matrix=distance_matrix,local_search=local_search,max_iterations=1500,plot = False, mutation_rate=mutation_rate)
            best_solution,best_fitness,mean_fitness = self.GA_level1_model_retrieveBestSolutionAndFitness()

        return best_solution,best_fitness, mean_fitness

       

    
    def save_info_csv(self, results_dict,append, filename = "results.csv"):
        """
        Save the results to a CSV file.

        Parameters:
            results_dict (dict): A dictionary where keys are iterations, and values are tuples (best_fitness, mean_fitness).
            student_number (str): The student's identification number.
        """
        #filename = "results.csv"
        
        if append:
            # Write results
            with open(filename, mode="a", newline="") as outFile:
                writer = csv.writer(outFile)
                for key, value in results_dict.items():
                    print(f"Key: {key} & Value: {value}")
                    best_fitness, mean_fitness = value
                    writer.writerow([key, best_fitness, mean_fitness])

        else:
            # Open the file in write mode and add headers
            with open(filename, mode="w", newline="") as outFile:
                writer = csv.writer(outFile)
                
                # Write student number and headers
                writer.writerow([f"# Filename: {filename}"])
                writer.writerow(["Iteration", "Best Fitness", "Mean Fitness"])
            
    def post_process_csv(self,flag_750=False):
        if flag_750:
            directory = "Results\Results_750_Hyperparameter"
        else:
            directory = "Results\Results_1000_Hyperparameter"
        legend_labels = ["Mutation Rate: 0.1 & Sigma: 0.1", "Mutation Rate: 0.1 & Sigma: 0.5", "Mutation Rate: 0.1 & Sigma: 0.8",
                         "Mutation Rate: 0.5 & Sigma: 0.1", "Mutation Rate: 0.5 & Sigma: 0.5", "Mutation Rate: 0.5 & Sigma: 0.8",
                         "Mutation Rate: 0.8 & Sigma: 0.1", "Mutation Rate: 0.8 & Sigma: 0.5", "Mutation Rate: 0.8 & Sigma: 0.8"]
        combined_pd = self.combine_csv_files(directory)
        self.plot_boxplot(combined_pd)
        self.plot_custom_boxplot(combined_pd, x_col='Filename', y_col='Best Fitness', ylabel='Distance', legend_labels=legend_labels)


    def combine_csv_files(self,directory):
        """
        Combine all CSV files in the specified directory into a single DataFrame.
        Each file's data will include a column identifying the file it came from.

        Args:
            directory (str): Path to the directory containing the CSV files.

        Returns:
            pd.DataFrame: A combined DataFrame with an added 'Filename' column.
        """
        combined_df = pd.DataFrame()

        for filename in os.listdir(directory):
            if filename.endswith(".csv"):  # Process only CSV files
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path, skiprows=1)  # Skip the comment line
                df['Filename'] = filename  # Add a column for the filename
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        return combined_df   
    
    def plot_boxplot(self,df):
        """
        Plot a box-and-whisker plot for the Best Fitness from each file.

        Args:
            df (pd.DataFrame): A DataFrame with data from multiple files.
        """
        # Group data by filename and extract 'Best Fitness'
        grouped = df.groupby('Filename')['Best Fitness']

        # Create a list of values for the boxplot
        data_to_plot = [group for _, group in grouped]

        # Create the boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(data_to_plot, labels=grouped.groups.keys(), patch_artist=True)
        plt.title("Box-and-Whisker Plot of Best Fitness")
        plt.ylabel("Best Fitness")
        plt.xlabel("Filename")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_custom_boxplot(self, df, x_col='Filename', y_col='Best Fitness', ylabel='Distance', legend_labels=None):
        """
        Plot a customized box-and-whisker plot with legend labels replacing x-axis category names.

        Args:
            df (pd.DataFrame): A DataFrame with data from multiple files.
            x_col (str): Column name for the x-axis categories.
            y_col (str): Column name for the y-axis values.
            ylabel (str): Label for the y-axis.
            legend_labels (list): Custom labels for the x-axis categories.
        """
        import matplotlib.patches as mpatches

        plt.figure(figsize=(18, 12))

        # Create a boxplot using seaborn
        sns.boxplot(
            x=x_col, y=y_col, data=df, palette="pastel", width=0.6, showmeans=True, meanline=True,
            meanprops={"linestyle": "--", "color": "red", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            boxprops={"edgecolor": "black", "linewidth": 1.5},
            medianprops={"color": "blue", "linewidth": 2},
            capprops={"linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "green", "alpha": 0.7}
        )

        # Customizing the plot
        plt.title("Hyperparameter search of 750 cities instance problem using Box-and-Whisker Plot", fontsize=16, weight='bold')
        plt.ylabel(ylabel, fontsize=18)
        plt.xlabel("Parameters", fontsize=18)

        # Replace x-axis ticks with legend labels if provided
        if legend_labels:
            plt.xticks(ticks=range(len(legend_labels)), labels=legend_labels, rotation=45, fontsize=12)
        else:
            plt.xticks(rotation=45, fontsize=14)

        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Add legend for plot elements
        handles = [
            mpatches.Patch(color="black", label="Box: IQR (Q1 to Q3)"),
            mpatches.Patch(color="blue", label="Median (blue line)"),
            mpatches.Patch(color="red", label="Mean (--red dashed line)"),
            mpatches.Patch(color="green", label="Outliers (green dots)"),
            mpatches.Patch(color="none", edgecolor="black", label="Whiskers: Range of data within 1.5*IQR", linewidth=1.5)
        ]

        # Add optional legend for group-specific labels
        '''
        if legend_labels:
            group_handles = [
                mpatches.Patch(color=color, label=label)
                for color, label in zip(sns.color_palette("pastel"), legend_labels)
            ]
            handles.extend(group_handles)
        '''

        # Position the legend outside the plot
        # Position the legend outside the plot
        plt.legend(
            handles=handles,  # Define your handles for the legend
            loc='upper left',  # Anchor the legend to the top-left of the plot's bounding box
            bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot
            fontsize=18,
            frameon=True
        )

        # Adjust the layout to prevent clipping of the legend
        plt.subplots_adjust(right=0.8)  # This adjusts the space for the plot (reduce the right margin)
        plt.show()



    def post_process_histogram_csv(self):
       
        directory = "Results\Results_50_Histogram"
        
        combined_pd = self.combine_csv_files(directory)

        # Calculate the mean and standard deviation for each group
        mean_values = combined_pd['Best Fitness'].mean()
        std_values = combined_pd['Best Fitness'].std()
        print(f"Mean values: {mean_values}")
        print(f"Std values: {std_values}")
        #print(f"Combined pd: {combined_pd}")
        self.plot_histogram(df=combined_pd)
        #self.plot_boxplot(combined_pd)
        #self.plot_custom_boxplot(combined_pd, x_col='Filename', y_col='Best Fitness', ylabel='Distance', legend_labels=legend_labels)
        #     
    def plot_histogram(self, df):
        # Set up Seaborn style
        #sns.set_theme(style="whitegrid")
        sns.set_theme(style="white")  # Removes the grid lines


        # Create a histogram using Seaborn
        plt.figure(figsize=(20, 12))
        sns.histplot(
            df["Best Fitness"],
            bins=40,  # Decrease number of bins to make bars wider
            binrange=[25000, 26000],  # Ensure range matches the data
            #kde=True,
            color="purple",
        )

        sns.histplot(
            df["Mean Fitness"],
            bins=100,  # Decrease number of bins to make bars wider
            binrange=[25000, 45000],  # Ensure range matches the data
            #kde=True,
            color="orange",
        )

        # Set labels and title
        plt.xlabel("Best Fitness (distance)", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        # Customize x and y tick labels
        plt.xticks(fontsize=18)  # Increase font size for x-tick labels
        plt.yticks(fontsize=18)  # Increase font size for y-tick labels
        plt.title("Distribution of Best and Mean fitness values for 50 Cities instance problem.", fontsize=30)
        plt.legend(["Best Fitness", "Mean Fitness"], fontsize=20)
        # Show the plot
        plt.show()

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 7) Extras: Execution Time, Plots ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def plot_test_lineplot(self,best_fitness_list,mean_fitness_list):
        '''
        - Plot the test line plot
        '''
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        
        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(best_fitness_list))),
            y=best_fitness_list,
            mode='lines+markers',
            name='Best Objective',
            line=dict(color='blue'),
            marker=dict(symbol='x')
        ))

        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(mean_fitness_list))),
            y=mean_fitness_list,
            mode='lines+markers',
            name='Mean Objective',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))



        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Best and Mean fitness for multiple runs of the GA algorithm with number of cities: {self.num_city}',
            xaxis_title='Outer Iterations',
            yaxis_title='Fitness (Distance)',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='linear',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the first plot
        fig_obj.show()


    def plot_test_histogram(self,best_fitness_list, mean_fitness_list):
        '''
        - Plot the test histogram
        '''
        
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        
        # Add the mean objective trace
        fig_obj.add_trace(go.Histogram(
            x=best_fitness_list,
            name='Best Objective',
            marker=dict(color='blue'),
            opacity=0.75
        ))

        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Best fitness for multiple runs of the GA algorithm with number of cities: {self.num_city}',
            xaxis_title='Outer Iterations',
            yaxis_title='Best fitness',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='linear',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the first plot
        fig_obj.show()
        
        # ================================================================================================================
        # ================================================================================================================
        # ================================================================================================================
        # ================================================================================================================

        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        
        # Add the mean objective trace
        fig_obj.add_trace(go.Histogram(
            x=mean_fitness_list,
            name='Mean Objective',
            marker=dict(color='orange'),
            opacity=0.75
        ))

        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Mean fitness for multiple runs of the GA algorithm with number of cities: {self.num_city}',
            xaxis_title='Outer Iterations',
            yaxis_title='Mean Fitness',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='linear',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the first plot
        fig_obj.show()

    def plot_ExecutionTime_Clusters(self):
            '''
            - Plot
            '''
            plt.plot(self.deltatime_cluster_list)
            plt.xlabel('Clusters')
            plt.ylabel('Execution Time')
            plt.title('Execution Time for each cluster')
            plt.show()

        
    def add_cluster_solution(self,cluster_solution):
        '''
        - Add the cluster solution to teh solutions list
        '''
        self.clusters_solution_list.append(cluster_solution)





    


#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
																			# Class clustering
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn_extra.cluster import KMedoids


class k_clusters:
    def __init__(self,distance_matrix):
        self.num_simulations_to_run = 2
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.clusters_list = []

    def run_model_KMedoids(self,num_clusters,min_cluster_size=10):
        '''
        - Run the KMedoids clustering
        '''
        print("\n---------- KMedoids Clustering: ------")
        # Replace np.inf with a large number
        self.distance_matrix[self.distance_matrix == np.inf] = 1e5
        print(f"Distance matrix: {self.distance_matrix}")

        # K-Medoids clustering using a precomputed distance matrix
        self.kmedoids = KMedoids(n_clusters=num_clusters, metric="precomputed",max_iter=10000, random_state=0)
        self.labels = self.kmedoids.fit_predict(self.distance_matrix)

        print("Cluster assignments:", self.labels)
        print("Cluster medoids:", self.kmedoids.medoid_indices_)
        #print("Inertia:", self.kmedoids.inertia_)
        self.post_process_clusters()
    
    def post_process_clusters(self):
        '''
        - Post process the clusters
        '''
        print("\n---------- Post Processing Clusters: ------")

        # Ensure medoid indices and labels are available
        medoids = self.kmedoids.medoid_indices_
        labels = self.labels
        distance_matrix = self.distance_matrix

        # Initialize the dictionary to store cluster information
        self.clusters_list = []
         

        for cluster in range(len(medoids)):
            # Get the medoid for the current cluster
            medoid = medoids[cluster]

            # Find the cities assigned to the current cluster
            cluster_cities = np.where(labels == cluster)[0]

            # Calculate the intra-cluster distance (sum of distances from each point to the medoid)
            intra_cluster_distance = np.sum([distance_matrix[medoid][city] for city in cluster_cities])

            # Store cluster information in the dictionary
            cluster_info_dict = {
                "medoid": medoid,
                "num_cities": len(cluster_cities),
                "assigned_cities": cluster_cities.tolist(),
                "intra_cluster_distance": intra_cluster_distance
            }

            self.clusters_list.append(cluster_info_dict)
        #self.plot_post_processed_clusters()

    def plot_post_processed_clusters(self):
        """
        Plots the intra-cluster distances and the number of cities for each post-processed cluster.
        The function uses the data stored in `self.clusters_list` after post-processing the clusters.
        """
        print("\nPlotting Post-Processed Clusters:")
        print(f"Number of Clusters: {len(self.clusters_list)}")

        # Initialize lists to store the intra-cluster distances and number of cities for plotting
        intra_cluster_distances = []
        cluster_cities = []

        # Loop through the post-processed clusters to collect the needed data
        for cluster_info in self.clusters_list:
            # Extract intra-cluster distance and the number of cities for each cluster
            intra_cluster_distances.append(cluster_info['intra_cluster_distance'])
            cluster_cities.append(cluster_info['num_cities'])

        # Plotting the data
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns

        # Plot the intra-cluster distances
        axes[0].bar(range(len(self.clusters_list)), intra_cluster_distances, color='skyblue')
        axes[0].set_xlabel('Cluster Index')
        axes[0].set_ylabel('Intra-Cluster Distance')
        axes[0].set_title('Intra-Cluster Distances for Each Cluster')
        axes[0].set_xticks(range(len(self.clusters_list)))

        # Plot the number of cities in each cluster
        axes[1].bar(range(len(self.clusters_list)), cluster_cities, color='lightgreen')
        axes[1].set_xlabel('Cluster Index')
        axes[1].set_ylabel('Number of Cities')
        axes[1].set_title('Number of Cities in Each Cluster')
        axes[1].set_xticks(range(len(self.clusters_list)))

        # Show both plots
        plt.tight_layout()  # Adjusts layout for better spacing
        plt.show()
            
        




    def print_info_run_model(self,k,min_cluster_size):
        '''
        - Print the information
        '''
        print("\n---------- K-Medoids Clustering: ------")
        print(f"   * Model Info:")
        print(f"       - Number of cities: {self.num_cities}")
        print(f"       - K: {k}")
        print(f"       - Min Cluster Size: {min_cluster_size}")
        print(f"       - Distance Matrix: {self.distance_matrix}")
        print(f"   * Running model:")

        #print(f"Distance matrix: {self.distance_matrix}")

    def run_model(self,k,min_cluster_size):
        '''
        - Run the model
        '''
        self.print_info_run_model(k,min_cluster_size)
        self.k_medoids_clustering(k=k, min_cluster_size=min_cluster_size)



    def find_closest_finite_city(self,distance_matrix, city_index):
        """
        Find the closest city to the given city_index that has a finite distance.

        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix with possible np.inf values.
        - city_index (int): The index of the city for which we want to find the closest connected city.

        Returns:
        - closest_city (int): The index of the closest city with a finite distance, or None if all distances are infinite.
        """
        finite_distances = distance_matrix[city_index]
        finite_city_indices = np.where(np.isfinite(finite_distances))[0]
        
        if len(finite_city_indices) == 0:
            return None  # No reachable cities
        
        # Find the index of the closest city with a finite distance
        closest_city = finite_city_indices[np.argmin(finite_distances[finite_city_indices])]
        return closest_city
    
    def handle_infinite_distances(self,distance_matrix, labels, medoids):
        """
        Reassigns cities with infinite distances to medoids based on the cluster assignment
        of the nearest connected cities.
        
        Parameters:
        - distance_matrix: numpy.ndarray, the distance matrix with possible np.inf values.
        - labels: numpy.ndarray, initial cluster assignments.
        - medoids: list, the current medoids.
        
        Returns:
        - labels: numpy.ndarray, updated cluster assignments.
        """
        n_cities = distance_matrix.shape[0]
        for city in range(n_cities):
            # Skip cities that already have a valid cluster assignment
            if labels[city] != -1:
                continue
            
            # Use the helper function to find the closest city with a finite distance
            closest_city = self.find_closest_finite_city(distance_matrix, city)
            
            if closest_city is not None:
                # Assign the current city to the cluster of the closest connected city
                labels[city] = labels[closest_city]
                #print(f"City {city} assigned to cluster {labels[closest_city]} based on closest city {closest_city}.")
            else:
                print(f"Warning: City {city} has no reachable cities and remains unassigned.")
        
        return labels

    def find_closest_finite_city(self,distance_matrix, city):
            """
            Finds the closest city with a finite distance.
            
            Parameters:
            - distance_matrix: numpy.ndarray, the distance matrix with possible np.inf values.
            - city: int, the index of the city for which we are finding the closest city.
            
            Returns:
            - int: the index of the closest city with a finite distance, or None if none found.
            """
            n_cities = distance_matrix.shape[0]
            min_distance = np.inf
            closest_city = None
            
            for other_city in range(n_cities):
                if np.isfinite(distance_matrix[city, other_city]) and distance_matrix[city, other_city] < min_distance:
                    min_distance = distance_matrix[city, other_city]
                    closest_city = other_city
            
            return closest_city
    


    def k_medoids_clustering(self, k=4, max_iterations=20, tolerance=1e-1, distance_tolerance=100, min_cluster_size=10):
        """
        Perform k-Medoids clustering from a distance matrix.
        
        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - k (int): The number of clusters.
        - max_iterations (int): The maximum number of iterations to run the algorithm.
        - tolerance (float): The tolerance level to stop the algorithm if medoids stop changing.
        
        Returns:
        - medoids (list): The final medoids (cluster centers).
        - labels (numpy.ndarray): The cluster assignment for each city.
        """

        # Initialize medoids randomly
        
        distance_matrix = self.distance_matrix  
        n_cities = distance_matrix.shape[0]
        medoids = np.random.choice(n_cities, k, replace=False)
        prev_medoids = np.copy(medoids)
        prev_intra_cluster_distance = np.inf
        intra_cluster_distances_dict = {}  # To store intra-cluster distances
        total_distance_list = []

        for iteration in range(max_iterations):
            #print(f"\nIteration {iteration + 1}:")
            #print(f"Current Medoids: {medoids}")
            
            labels = np.full(n_cities, -1)  # Initialize all cities as unassigned
            for city in range(n_cities):
                min_distance = np.inf
                for medoid in medoids:
                    if distance_matrix[city, medoid] != np.inf:  # Avoiding inf distances
                        if distance_matrix[city, medoid] < min_distance:
                            min_distance = distance_matrix[city, medoid]
                            labels[city] = medoid

            # Handle cities with infinite distances
            labels = self.handle_infinite_distances(distance_matrix, labels, medoids)
            
            new_medoids = np.copy(medoids)
            total_intra_cluster_distance = 0  # Total distance to compute the convergence criteria

            # Update medoids by minimizing the total intra-cluster distance
            for cluster in range(k):
                cluster_cities = np.where(labels == medoids[cluster])[0]  # Get cities assigned to the current cluster
                if len(cluster_cities) > 0:  # Only proceed if the cluster is not empty
                    min_total_distance = np.inf  # Initialize to a very large number to minimize late.
                    #print(f"\nCluster {cluster}")

                    for meloid in cluster_cities:
                        best_medoid = meloid
                        #print(f"\n       Medoid:{best_medoid} cluster_cities: {cluster_cities} number of cities: {len(cluster_cities)}")
                       
                        total_distance = 0

                        for city in cluster_cities:
                        
                            
                            # Distance from the medoid to the current city
                            distance = distance_matrix[best_medoid, city]
                            #print(f"Cluster: {best_medoid}  & City {city} Total Distance: {total_distance} with distance: {distance}")
                            
                            # If the distance is finite, add it to the total distance
                            if np.isfinite(distance):
                                total_distance += distance
                            else:
                                # If the distance is infinite, find the closest city in the cluster with a finite distance
                                #print(f"City {city} has infinite distance to the medoid.")
                                closest_city = None
                                closest_distance = np.inf
                                dist_closest_to_meloid = 0
                                
                                for other_city in cluster_cities:
                                    if np.isfinite(distance_matrix[city, other_city]) and distance_matrix[city, other_city] < closest_distance and np.isfinite(distance_matrix[best_medoid, other_city]):
                                        closest_city = other_city
                                        closest_distance = distance_matrix[city, other_city]
                                        dist_closest_to_meloid = distance_matrix[best_medoid, closest_city]    
                                
                                # Add the distance to the closest city to the total distance if found
                                if closest_city is not None:
                                    #print(f"City {city} has infinite distance to the medoid. Closest city: {closest_city} Distance: {closest_distance}")
                                    total_distance += closest_distance + dist_closest_to_meloid
                                else:
                                    print(f"Warning: City {city} has no reachable cities in the cluster.")

                                        
                        # Update the best medoid if the total distance is smaller
                        if total_distance < min_total_distance:
                            #print(f"Total Distance: {total_distance} < {min_total_distance}")
                            min_total_distance = total_distance
                            best_medoid_final = best_medoid
                            #print(f"Best Medoid Final: {best_medoid_final} Total Distance: {min_total_distance}")

                    # Update the medoid for the cluster with the best one found
                    new_medoids[cluster] = best_medoid_final
                    intra_cluster_distances_dict[cluster] = min_total_distance
                    total_intra_cluster_distance += min_total_distance

            # Check for convergence
            #print(f"Total Intra-Cluster Distance: {total_intra_cluster_distance}")
            total_distance_list.append(total_intra_cluster_distance)
            #get me the previous total distance form the list

            diff_dist = np.inf
            prev_intra_cluster_distance = total_distance_list[-2] if len(total_distance_list) > 1 else np.inf
            if prev_intra_cluster_distance != np.inf:   
                diff_dist = np.abs(total_intra_cluster_distance - prev_intra_cluster_distance)
                #print(f"Change in Total Intra-Cluster Distance: {diff_dist}")
                #print(f"total_intr_cluster_distance: {total_intra_cluster_distance} prev_intra_cluster_distance: {prev_intra_cluster_distance}")    
            
            #diff = np.abs(np.sum(new_medoids - prev_medoids))
            if np.all(new_medoids == prev_medoids) or (diff_dist <= distance_tolerance):
                print(f"Converged after {iteration + 1} iterations.")
                #print me teh reason of the convergence
                #print(f"Reason of convergence: {diff_dist} <= {distance_tolerance} or {np.all(new_medoids == prev_medoids)}") 
                #print(f"Final Medoids: {new_medoids}")  
                
                break
            
            prev_medoids = np.copy(new_medoids)
            medoids = np.copy(new_medoids)
        
        # Print final clustering results
        print("\nFinal Clustering Results:")
        intra_cluster_distances = np.array(list(intra_cluster_distances_dict.values()))
        

        for cluster in range(k):
            cluster_cities = np.where(labels == medoids[cluster])[0]
            intra_cluster_distance = intra_cluster_distances_dict.get(cluster, 0)
            print(f"\nCluster {cluster}:")
            print(f"  Medoid: {medoids[cluster]}")
            print(f"  Number of Cities: {len(cluster_cities)}")
            print(f"  Assigned Cities: {cluster_cities}")
            print(f"  Intra-Cluster Distance: {intra_cluster_distance}")
            cluster_info = {"cluster": cluster, "medoid": medoids[cluster], "num_cities": len(cluster_cities), "assigned_cities": cluster_cities, "intra_cluster_distance": intra_cluster_distance}
            self.clusters_list.append(cluster_info)
            
        self.verify_unique_cluster_assignment(labels, k)    
        self.plot_intra_cluster_distances(distance_matrix, intra_cluster_distances, labels, medoids, k)
        medoids, labels, k = self.handle_empty_or_small_clusters(distance_matrix, medoids, labels, k, min_cluster_size=min_cluster_size)
        intra_cluster_distances = self.calculate_intra_cluster_distance(distance_matrix, labels, medoids)    
        self.plot_intra_cluster_distances(distance_matrix, intra_cluster_distances, labels, medoids, k) 

        return medoids, labels, intra_cluster_distances_dict
    
    def handle_empty_or_small_clusters(self,distance_matrix, clusters, labels, k, min_cluster_size=1):
        """
        Checks for empty or small clusters and reassigns cities in these clusters to the nearest cluster.

        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - clusters (list): The list of current medoids.
        - labels (numpy.ndarray): Array of cluster assignments for each city.
        - k (int): The total number of clusters.
        - min_cluster_size (int): The minimum size a cluster can have before being considered "small" and removed.

        Returns:
        - clusters (list): Updated medoids (clusters).
        - labels (numpy.ndarray): Updated labels after reassignment.
        """
        print("\nHandling Empty or Small Clusters:")
        print(f"Number of Clusters: {k}")
        #print(f"Clusters: {clusters}")
        print(f"Number of Cities: {len(labels)}")
        #print(f"Cluster Assignments: {labels}")
        print(f"Minimum Cluster Size: {min_cluster_size}")
        

        # We will need to modify clusters, so we will create a new list for valid clusters
        new_clusters = []
        new_labels = np.copy(labels)
        k_new = k
        cluster_idx = 0
        
        # We need to loop over the clusters using the original `clusters` list length
        while cluster_idx < k_new:
            # Find the cities assigned to the current cluster
            print(f"\nCluster {cluster_idx}:")
            
            cluster_cities = np.where(new_labels == clusters[cluster_idx])[0]
            
            # If the cluster is empty or has fewer cities than the minimum size
            if len(cluster_cities) < min_cluster_size:
                medoid = clusters[cluster_idx]
                print(f"Cluster {cluster_idx} with medoid: {medoid} is too small with number of cities {len(cluster_cities)} or empty, reassigning cities.")
                
                # For each city in this cluster, find the closest valid medoid
                for city in cluster_cities:
                    min_distance = np.inf
                    best_medoid = -1
                    
                    # Find the closest medoid that is not the current one
                    for other_cluster_idx in range(k):
                        if other_cluster_idx != cluster_idx:
                            distance_to_medoid = distance_matrix[city, clusters[other_cluster_idx]]
                            # Ensure that we don't consider infinite distances
                            if distance_to_medoid != np.inf and distance_to_medoid < min_distance:
                                min_distance = distance_to_medoid
                                best_medoid = clusters[other_cluster_idx]
                    
                    # Reassign the city to the closest cluster
                    new_labels[city] = best_medoid
                    print(f"City {city} reassigned to medoid {best_medoid} with distance {min_distance}.")
                # Do not add the current cluster to the new_clusters list (remove it)
                k_new -= 1

            else:
                # If the cluster is valid (not empty or too small), keep it
                new_clusters.append(clusters[cluster_idx])
                

            cluster_idx += 1



        '''
        for cluster_idx in range(k):
            # Find the cities assigned to the current cluster
            cluster_cities = np.where(labels == clusters[cluster_idx])[0]
            
            # If the cluster is empty or has fewer cities than the minimum size
            if len(cluster_cities) < min_cluster_size:
                medoid = clusters[cluster_idx]
                print(f"Cluster {cluster_idx} with medoid: {medoid} is too small with number of cities {len(cluster_cities)} or empty, reassigning cities.")
                
                # For each city in this cluster, find the closest valid medoid
                for city in cluster_cities:
                    min_distance = np.inf
                    best_medoid = -1
                    
                    # Find the closest medoid that is not the current one
                    for other_cluster_idx in range(k):
                        if other_cluster_idx != cluster_idx:
                            distance_to_medoid = distance_matrix[city, clusters[other_cluster_idx]]
                            # Ensure that we don't consider infinite distances
                            if distance_to_medoid != np.inf and distance_to_medoid < min_distance:
                                min_distance = distance_to_medoid
                                best_medoid = clusters[other_cluster_idx]
                    
                    # Reassign the city to the closest cluster
                    new_labels[city] = best_medoid
                    print(f"City {city} reassigned to medoid {best_medoid} with distance {min_distance}.")
                # Do not add the current cluster to the new_clusters list (remove it)

            else:
                # If the cluster is valid (not empty or too small), keep it
                new_clusters.append(clusters[cluster_idx])


        '''


           

        # We may have fewer clusters after removal, update k accordingly
        k = len(new_clusters)
        print(f"\n New Number of Clusters: {k}")
        #print(f"New Cluster Assignments: {new_labels}")
        #print(f"New Clusters: {new_clusters}")
        print(f"New Number of Cities: {len(new_labels)}")
        self.clusters_list.clear()
        for cluster in range(k):
            cluster_cities = np.where(new_labels == new_clusters[cluster])[0]
            intra_cluster_distance = 0
            print(f"\nCluster {cluster}:")
            print(f"  Medoid: {new_clusters[cluster]}")
            print(f"  Number of Cities: {len(cluster_cities)}")
            print(f"  Assigned Cities: {cluster_cities}")
            print(f"  Intra-Cluster Distance: {intra_cluster_distance}")
            cluster_info = {"cluster": cluster, "medoid": clusters[cluster], "num_cities": len(cluster_cities), "assigned_cities": cluster_cities, "intra_cluster_distance": intra_cluster_distance}
            self.clusters_list.append(cluster_info)

        # Return the updated clusters and labels
        return new_clusters, new_labels, k
    
    def calculate_intra_cluster_distance(self, distance_matrix, labels, medoids):
        """
        Calculates the intra-cluster distance for each cluster, considering cities assigned to that cluster.
        
        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - labels (numpy.ndarray): Array containing the cluster assignments for each city.
        - medoids (list): List of current medoids (cluster centers).
        
        Returns:
        - intra_cluster_distances (dict): Dictionary with cluster index as the key and the total intra-cluster distance as the value.
        """
        print("\nCalculating Intra-Cluster Distances:")

        k = len(medoids)  # Number of clusters
        intra_cluster_distances = {}  # To store the intra-cluster distances
        
        # Iterate over each cluster
        for cluster in range(k):
            cluster_cities = np.where(labels == medoids[cluster])[0]  # Get cities assigned to the current cluster
            total_intra_cluster_distance = 0  # To accumulate the total intra-cluster distance
            #print(f"\nCluster {cluster}: cluster_cities: {cluster_cities} number of cities: {len(cluster_cities)}")
            
            # Only proceed if the cluster is not empty
            if len(cluster_cities) > 0:
                for city in cluster_cities:
                    # Distance from the medoid to the current city
                    distance = distance_matrix[medoids[cluster], city]
                    #print(f"Cluster: {medoids[cluster]}  & City {city} Distance: {distance}")
                    
                    # If the distance is finite, add it to the total distance
                    if np.isfinite(distance):
                        total_intra_cluster_distance += distance
                    else:
                        # If the distance is infinite, find the closest city in the cluster with a finite distance
                        closest_city = None
                        closest_distance = np.inf
                        dist_closest_to_meloid = 0
                        
                        for other_city in cluster_cities:
                            if np.isfinite(distance_matrix[city, other_city]) and distance_matrix[city, other_city] < closest_distance and np.isfinite(distance_matrix[medoids[cluster], other_city]):
                                closest_city = other_city
                                closest_distance = distance_matrix[city, other_city]
                                dist_closest_to_meloid = distance_matrix[medoids[cluster], closest_city]    
                        
                        # Add the distance to the closest city to the total distance if found
                        if closest_city is not None:
                            total_intra_cluster_distance += closest_distance + dist_closest_to_meloid

            # Store the total intra-cluster distance for this cluster
            intra_cluster_distances[cluster] = total_intra_cluster_distance

        # Return the dictionary of intra-cluster distances
        intra_cluster_distances = np.array(list(intra_cluster_distances.values()))
        #print(f"Intra-Cluster Distances: {intra_cluster_distances}")

        return intra_cluster_distances

    

    def verify_unique_cluster_assignment(self, labels, k):
        """
        Verify that each city has been assigned to exactly one cluster.
        
        Parameters:
        - labels (numpy.ndarray): The cluster assignment for each city.
        - k (int): The number of clusters.
        
        Returns:
        - bool: True if all cities are assigned to exactly one cluster, False otherwise.
        """
        print("\nVerifying Cluster Assignments:")
        print(f"Number of Clusters: {k}")
        print(f"Number of Cities: {len(labels)}")
        print(f"Cluster Assignments: {labels}")
        # Check if the label for each city is within the valid range (0 to k-1)
        valid_assignment = np.all((labels >= 0) & (labels < len(labels)))
        
        if not valid_assignment:
            print("Error: Some cities are assigned to an invalid cluster.")
            return False
        
        # Check that each city is assigned to exactly one cluster
        unique_labels = np.unique(labels)
        
        if len(unique_labels) != k:
            print(f"Error: The number of unique clusters is {len(unique_labels)}, but expected {k}.")
            return False
        
        # If all checks pass
        print("All cities are assigned to exactly one cluster.")
        return True



    def plot_intra_cluster_distances(self,distance_matrix,inter_cluster_distance ,labels, medoids, k):
        """
        Plots the intra-cluster distances for each cluster.
        
        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - labels (numpy.ndarray): The array of cluster assignments for each city.
        - medoids (numpy.ndarray): The medoids of the clusters.
        - k (int): The number of clusters.
        """
        print("\nPlotting Intra-Cluster Distances:")
        print(f"Number of Clusters: {k}")
        print(f"Number of Cities: {len(labels)}")
        print(f"Cluster Assignments: {labels}")
        print(f"Medoids: {medoids}")
        print(f"Inter-Cluster Distances: {inter_cluster_distance}")
        cluster_cities = []
        for cluster in range(k):
            cluster_cities.append(len(np.where(labels == medoids[cluster])[0]))
       
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns

        # Plot the intra-cluster distances
        axes[0].bar(range(k), inter_cluster_distance, color='skyblue')
        axes[0].set_xlabel('Cluster Index')
        axes[0].set_ylabel('Intra-Cluster Distance')
        axes[0].set_title('Intra-Cluster Distances for Each Cluster')
        axes[0].set_xticks(range(k))

        # Plot the number of cities in each cluster
        axes[1].bar(range(k), cluster_cities, color='skyblue')
        axes[1].set_xlabel('Cluster Index')
        axes[1].set_ylabel('Number of Cities')
        axes[1].set_title('Number of Cities in Each Cluster')
        axes[1].set_xticks(range(k))

        # Show both plots
        plt.tight_layout()  # Adjusts layout for better spacing
        plt.show()


    def generate_distance_matrix_cluster(self,cluster_list):
        '''
        - Generate the distance matrix for the clusters
        '''
        print("\n---------- Generating Distance Matrix for Clusters: ------")
        #print(f"   * Model Info:")
        #print(f"       - Number of clusters: {len(self.clusters_list)}")
        #print(f"       - Clusters: {self.clusters_list}")
        #print(f"       - Cities [0]: {self.distanceMatrix.shape[0]}")
        #print(f"       - Cities [1]: {self.distanceMatrix.shape[1]}")
        #print(f"       - Cities: {self.cities}")
        #print(f"       - Distance Matrix: {self.distanceMatrix}")
        #print(f"       - Number of cities: {self.num_cities}")
        #print(f"       - Number of clusters: {len(self.clusters_list)}")
        #print(f"       - Clusters: {self.clusters_list}")

        assigned_cities_list = [cluster['assigned_cities'] for cluster in cluster_list]
        cities_cluster_list = []
        distance_matrix_cluster = []
        for cities in assigned_cities_list:
            cities_cluster_list.append(cities)
            num_cities = len(cities)
            distanceMatrix = np.zeros((num_cities,num_cities))
            #print(f"Number of cities: {num_cities}")
            #print(f"Cities: {cities}")
            row = 0
            for i in cities:
                
                column = 0
                for j in cities:
                    dist = self.distance_matrix[i,j]
                    #print(f"Distance between city {i} and city {j}: {dist}")
                    distanceMatrix[row,column] = dist
                    column += 1
                row += 1
            #print(f"Distance matrix: {distanceMatrix}")
            distance_matrix_cluster.append(distanceMatrix)
        
     
        return distance_matrix_cluster, cities_cluster_list













#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
																			# Class Evol Algorithm_K
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================

import numpy as np
import Reporter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time
from numba import jit,njit,prange
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from multiprocessing import active_children, Pool
import logging
import os
# Setup logging to capture all levels of messages
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')


class r0818807_GA_Level1:

    def __init__(self,cities,seed=None ,mutation_rate = 0.1,mutation_rate_pop = 0.1,elitism_percentage = 20,local_search=True,max_iterations = 200, sigma_value = 0.1):
        #Reporter:
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        
        #model_key_parameters
        self.cities = cities 
        self.k_tournament_k = 3
        self.population_size = 0.0
        #self.mutation_rate = mutation_prob
        self.mutation_rate = 0.8   # Prev 0.8 or 0.5
        self.mutation_rate_pop = 0.8
        self.elistism = 1

        self.max_iterations = max_iterations     

        #Population Stcuck
        self.prev_bestFit = None
        self.stuck_flag = False  
        
        # Fitness Sharing
        self.sigma = 0.8  # Prev 0.9
        self.alpha = 0.1   # Prev 0.1


        self.mean_objective = 0.0
        self.best_objective = 0.0
        self.mean_fitness_list = []
        self.best_fitness_list = [] 
        self.best_distances_scores_list = []
        self.best_average_bdp_scores_list = []

        self.mean_distances_scores_list = []
        self.mean_average_bdp_scores_list = []
        self.best_objective_list = []
        self.mean_objective_list = []
        self.second_best_objective_list = []

        #Unique Solutions
        self.num_unique_solutions_list = []
        self.num_repeated_solutions_list = []

        #Diversity: Hamming Distance
        self.hamming_distance_list = []
        self.hamming_distance_crossover_list = []
        self.hamming_distance_crossoverOld_list = []
        self.hamming_distance_mutation1_list = []
        self.hamming_distance_mutation2_list = []
        self.hamming_distance_elimination_list = []
        self.hamming_distance_local_search_list = []

        # Diversity: Edges
        self.edges_dict = {}
       
        
        

        self.weight_distance = 1
        self.weight_bdp = 0
        
        #Local Search
        self.local_search = local_search
        self.max_iter_ls = 100
        self.n_best = 2
        self.stuck_flag = False
        


        #Time
        self.time_iteration_lists = []
        self.time_initialization_list = []
        self.time_selection_list = []
        self.time_crossover_list = []
        self.time_mutation_list = []
        self.time_elimination_list = []
        self.time_mutation_population_list = []
        self.time_local_search_list = []
        self.time_iteration_list = []


        # Initial Solution
        self.initial_solution = None

        #Stopping Criteria
        self.counter_stuck = 0
        self.stopping_criteria = False

        #Cache for local search of population:
        self.visited_population =  []
        self.visited_population_fitness = []
        self.visisted_population_numIterations = []
        self.cache_ls = {}
       

        #Random seed
        if seed is not None:
            #np.random.seed(seed) 
            a = 1   
        
        

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Settings:------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    def retieve_order_cities(self,best_solution):

        '''
        - Retrieve the order of the cities
        '''
        #print(f"\n Best solution is : {best_solution}")
        #print(f"\n Cities are : {self.cities}")
       
        
        self.best_solution_cities = self.cities[best_solution]
        #self.best_solution_cities = best_solution

        #print(f"\n Best solution cities are : {self.best_solution_cities}")

        

    def print_model_info(self):
        print("\n------------- GA_Level1: -------------")
        print(f"   * Model Info:")
        print(f"       - Population Size: {self.population_size}")
        print(f"       - Number of cities: {self.gen_size}")
        #print(f"       - Cities: {self.cities}")
        #print(f"       - Distance Matrix: {self.distance_matrix}")
        print(f"   * Model Parameters:")
        print(f"       - K: {self.k_tournament_k}")
        print(f"       - Mutation rate: {self.mutation_rate}")
        print(f"       - Sigma: {self.sigma}")
        print(f"       - Elitism percentage: {self.elistism} %")
        print(f"   * Running model:")
        print(f"       - Local search: {self.local_search}")
        #print(f"       - Initial Fitness: {self.fitness}")
        

        

    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf(distance_matrix=distance_matrix,replace_value=1e8)
        self.gen_size = len(distance_matrix)
        #self.population_size = 2*self.gen_size
        if self.gen_size < 200:
            self.population_size = 15
        else:
            self.population_size = 15

        #self.population_size = 50
        self.k_tournament_k = 3
        #self.k_tournament_k = int((3/100)*self.population_size)
        #print(f"Distance matrix is {self.distance_matrix}")
        #print(f"Gen size is {self.gen_size}")




    def check_inf(self,distance_matrix,replace_value = 1e8):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''
        distance_matrix[distance_matrix == np.inf] = replace_value
        return distance_matrix
    
    def set_inf(self,distance_matrix,replace_value):
        '''
        - Set the inf values in the distance matrix
        '''
        distance_matrix[distance_matrix >= replace_value] = np.inf
        return distance_matrix
    

    
    def calculate_information_iteration(self):
        '''
        - Calculate the mean and best objective function value of the population
        '''
        self.mean_objective = np.mean(self.distance_scores)
        self.best_objective = np.min(self.distance_scores)

        self.mean_objective_list.append(self.mean_objective)
        self.best_objective_list.append(self.best_objective)  

        
        self.mean_fitness_list.append(np.mean(self.fitness))
        self.best_fitness_list.append(np.min(self.fitness))

        self.best_average_bdp_scores_list.append(np.max(self.average_bpd_scores))
        self.mean_average_bdp_scores_list.append(np.mean(self.average_bpd_scores))

        sorted_fitness = np.sort(self.fitness)
        self.second_best_objective_list.append(sorted_fitness[1])
        best_index = np.argmin(self.distance_scores)
        best_solution = self.population[best_index]

        #self.best_distances_scores_list.append(self.distance_scores[best_index])
        #print(f"Mean Objective --> {self.mean_objective} \n Best Objective --> {self.best_objective} \n Best Solution --> {best_solution}")
        self.retieve_order_cities(best_solution)    
        #print(f"Mean Objective --> {self.mean_objective} \n Best Objective --> {self.best_objective} \n Best Solution --> {best_solution}")

        #Diversity: Hamming Distance
        
        self.hamming_distance_list.append(self.hamming_distance)

        #Unique and repeated solutions
        self.caclulate_numberRepeatedSolution(population=self.population)
        self.num_unique_solutions_list.append(self.num_unique_solutions)
        self.num_repeated_solutions_list.append(self.num_repeated_solutions)

        return self.mean_objective,self.best_objective,best_solution
    
    def print_best_solution(self):
        '''
        - Print the best solution
        '''
        
        print(f"\n Best solution is : {self.best_objective} \n Best solution cities are : {self.best_solution_cities}")
    
    def update_time(self,time_initalization, time_selection, time_crossover, time_mutation, time_elimination,time_mutation_population,time_local_search,time_iteration):
        '''
        - Update the time
        '''
        self.time_initialization_list.append(time_initalization)
        self.time_selection_list.append(time_selection)
        self.time_crossover_list.append(time_crossover)
        self.time_mutation_list.append(time_mutation)
        self.time_elimination_list.append(time_elimination)
        self.time_mutation_population_list.append(time_mutation_population)
        self.time_local_search_list.append(time_local_search)
        self.time_iteration_list.append(time_iteration)

        new_lists = [self.time_initialization_list,self.time_selection_list,self.time_crossover_list,self.time_mutation_list,self.time_elimination_list,self.time_mutation_population_list,self.time_local_search_list,self.time_iteration_list]
        self.time_iteration_lists.append(new_lists)



    def plot_time_dynamic(self):
        '''
        - Plot the time dynamic
        '''


    def print_cache_ls(self):
        '''
        - Print the cache of the local search
        '''
        print(f"\n Cache of the local search is : {self.cache_ls}")

    def add_entry_cache_ls_individual(self,entry_fitness,entry_num_iterations):
        '''
        - Add an entry to the cache of the local search
        '''
        if entry_fitness in self.cache_ls:
            self.cache_ls['fitness'] = entry_num_iterations
        else:
            if len(self.cache_ls) <=1:
                self.cache_ls[entry_fitness] = entry_num_iterations
            if entry_fitness < min(self.cache_ls.keys()):
                self.cache_ls['fitness'] = entry_num_iterations
    
    def add_entry_cache_ls_population(self, entry_fitness_pop, entry_num_iterations_pop,max_iterations_cte):
        """
        Add an entry to the cache for local search while maintaining specific constraints:
        - Update iterations if fitness already exists.
        - Limit cache size to 1. If full, replace the entry with the smallest fitness if the new fitness is smaller.
        """
        #print(f"\n ---------------------------------------------------------------------")
        #print(f"\n -------------------- Cache before adding entries --------------------")
        #print(f"\n ---------------------------------------------------------------------")
        #print(f"Entry population fitness: {entry_fitness_pop}")
        #print(f"Entry population iterations: {entry_num_iterations_pop}")
        for idx, entry_fitness in enumerate(entry_fitness_pop):
            #print(f"\n Entry fitness: {entry_fitness} and the number of iterations is {entry_num_iterations_pop[idx]}")
            iterations = entry_num_iterations_pop[idx]


            # Case 1: Update iterations if fitness already in cache
            if entry_fitness in self.cache_ls:
                self.cache_ls[entry_fitness] = iterations
                #print(f"\n Case 1: Updating number of iterations. Entry fitness is in the cache: {entry_fitness} and the number of iterations is {iterations}")
                continue

            # Case 2: Add new entry if cache size is less than 2
            if len(self.cache_ls) < 2:
                self.cache_ls[entry_fitness] = iterations
                #print(f"\n Case 2: Populating cahe (len < 2). Entry fitness is not in the cache: {entry_fitness} and the number of iterations is {iterations}")
                continue
            else:

                # Case 3: Cache is full, decide which entry to replace
                min_fitness = min(self.cache_ls.keys())
                max_fitness = max(self.cache_ls.keys())

                if entry_fitness < min_fitness:
                    #print(f"\n Case 3a: Replacing min fitness. Entry fitness: {entry_fitness}, iterations: {iterations}")
                    del self.cache_ls[min_fitness]  # Replace the min entry
                    self.cache_ls[entry_fitness] = max_iterations_cte
                elif min_fitness < entry_fitness < max_fitness:
                    #print(f"\n Case 3b: Replacing second smallest fitness. Entry fitness: {entry_fitness}, iterations: {iterations}")
                    del self.cache_ls[max_fitness]  # Replace the larger fitness
                    self.cache_ls[entry_fitness] = max_iterations_cte

        #self.print_cache_ls()

    def add_entry_cache_ls_pop_lastBest(self,route,last_best):
        '''
        - Add an entry to the cache of the local search
        '''
        self.cache_ls[tuple(route)] = last_best

    
    def check_stuck_pop(self,new_bestFit):
        if self.prev_bestFit is None:
            self.prev_bestFit = new_bestFit
            self.stuck_flag = False
        else:
            if new_bestFit == self.prev_bestFit:
                #self.max_iter_ls = 500
                #self.n_best = 4
                self.stuck_flag = True
                #print(f"\n Population is stuck")
            else:
                self.stuck_flag = False
                self.prev_bestFit = new_bestFit
                #self.n_best = 2
                #self.max_iter_ls = self.initial_iter
    
    def check_stoppingCriteria(self,time_left,limit_stuck = 2):
        '''
        - Check the stopping criteria
        '''
        if time_left < 0:
            self.stopping_criteria = True
        else:
            self.stopping_criteria = False

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Run ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_model(self,plot = False):
        time_start = time.time()
        #self.set_initialization()
        self.initial_iter = 200
        self.max_iter_ls = 50

        if self.gen_size <= 750:
            self.mutation_rate = 0.8
            self.mutation_rate_pop = 0.8
            self.sigma = 0.8
            
        elif self.gen_size == 1000:
            self.mutation_rate = 0.1
            self.mutation_rate_pop = 0.1
            self.sigma = 0.5

        self.set_initialization_onlyValid_numpy_incremental(fitness_threshold=1e5)
        time_end = time.time()
        initialization_time = time_end - time_start 
        
       
        
    
       
        #num_iterations = 200
        num_iterations = self.max_iterations
        iterations = 0
        while( (self.stopping_criteria is False)):
            
           
            time_start_iteration = time.time()
            iterations += 1
            self.iteration = iterations
            #print(f"\n Iteration number {iterations}")
            
            time_start = time.time()
            parents = self.selection_k_tournament(num_individuals=self.population_size)
            time_end = time.time()
            selection_time = time_end - time_start
            
            time_start = time.time()  
            #offspring = self.crossover_singlepoint_population(parents)
            #offspring1 = self.crossover_order_population(parents)
            offspring1 = self.crossover_order_population_hybrid(parents)    
            #self.check_equality_twoPopulations(offspring,offspring1)
            time_end = time.time()
            time_crossover = time_end - time_start
            self.calculate_add_hamming_distance(population=offspring1,crossover=True)
            self.calculate_add_hamming_distance(population=offspring1,crossover_old=True)
            

            time_start = time.time()
            offspring_mutated = self.mutation_singlepoint_population(offspring1,offsprings_flag=True)
            offspring_mutated = self.mutation_inversion_population(offspring_mutated)
            time_end = time.time()
            time_mutation = time_end - time_start
            self.calculate_add_hamming_distance(population=offspring_mutated,mutation1=True)
            


            #Mutate also population
            time_start = time.time()
            self.population = self.mutation_singlepoint_population(self.population,offsprings_flag=False)
            #self.population = self.mutation_inversion_population(self.population)
            time_end = time.time()
            time_mutation_population = time_end - time_start
            self.calculate_add_hamming_distance(population=self.population,mutation2=True)
            
            
            #self.population = self.local_search_population_2opt_multip(self.population,n_best = 2,max_iterations=500)
            
            if self.local_search:
                
                #offspring_mutated = np.vstack((offspring_mutated,self.population))
                time_start = time.time()
                #offspring_mutated, self.population = self.local_search_population_jit(population=self.population,mutation_population=offspring_mutated,n_best = 3,max_iterations=200)
                
                if self.stuck_flag or iterations < 3:
                    if self.counter_stuck > 50 and iterations % 50 == 0:
                        self.population,offspring_mutated= self.local_search_population_2opt_multip(population=self.population,mutation_population=offspring_mutated,n_best = 2,
                                                                                                max_iterations=self.max_iter_ls,k_neighbors=30,heavy_ls=True,random=False)
                        #self.population, offspring_mutated = self.local_search_population_2opt_multip_parallel(population=self.population,mutation_population=offspring_mutated,n_best = 1,
                                                                                                               #max_iterations=self.max_iter_ls)
                        a = 1
                    else:
                        if iterations < 3:
                            #self.population,offspring_mutated= self.local_search_population_2opt_multip(population=self.population,mutation_population=offspring_mutated,n_best = self.n_best,
                                                                                                #max_iterations=self.max_iter_ls,k_neighbors=10,heavy_ls=False)
                            a= 1
                    if iterations > 3:
                        self.counter_stuck +=1
                else:
                    self.counter_stuck = 0
                
                #self.population,offspring_mutated= self.local_search_population_2opt_multip(population=self.population,mutation_population=offspring_mutated,n_best = self.n_best,
                                                                                            #max_iterations=self.max_iter_ls)
                self.population,offspring_mutated= self.local_search_population_greedy(population=self.population,mutation_population=offspring_mutated,n_best = 15,max_iterations=5)
                self.population,offspring_mutated = self.local_search_population_jit(population=self.population,mutation_population=offspring_mutated,n_best = 5,max_iterations=200)
        
                #self.population,offspring_mutated= self.local_search_population_2opt_multip(population=self.population,mutation_population=offspring_mutated,n_best = self.n_best,
                                                                                            #max_iterations=self.max_iter_ls,random=self.stuck_flag)
                #self.population,offspring_mutated= self.local_search_population_2opt_multip_parallel(population=self.population,mutation_population=offspring_mutated,n_best = 6,
                                                                                            #max_iterations=self.max_iter_ls)
                #offspring_mutated, self.population = self.local_search_population_2opt_multip_parallel(population=offspring_mutated,mutation_population=self.population,n_best = 3,max_iterations=100)
                #offspring_mutated= self.local_search_population_2opt_multip(offspring_mutated,n_best = 2,max_iterations=1)
                #self.poulation,offspring_mutated = self.local_search_population_2opt(population=self.population,mutation_population=offspring_mutated,n_best = 2,max_iterations=300)
                
                #offspring_mutated= self.local_search_population_2opt(offspring_mutated,n_best = 2,max_iterations=10)
                #offspring_mutated= self.local_search_population_3opt(offspring_mutated,max_iterations=50)
                #offspring_mutated= self.local_search_population_jit(offspring_mutated,max_iterations=50)
                #offspring_mutated= self.local_search_population_2opt_cum(offspring_mutated,max_iterations=50)
                time_end = time.time()
                time_local_search = time_end - time_start
                self.calculate_add_hamming_distance(population=offspring_mutated,local_search=True)
            else:
                time_local_search = 0
                self.calculate_add_hamming_distance(population=offspring_mutated,local_search=True)
               
            
            time_start = time.time()
            #self.eliminate_population(population=self.population, offsprings=offspring_mutated)
            #self.elimnation_population_lamdaMu(population=self.population, offsprings=offspring_mutated)
            #self.eliminate_population_elitism(population=self.population, offsprings=offspring_mutated)
            #self.eliminate_population_kTournamenElitism(population=self.population, offsprings=offspring_mutated,elitism_percentage=self.elistism)
            #self.check_insert_individual(num_iterations=100,threshold_percentage = 20)
            #self.eliminate_population_fs(population=self.population, offsprings=offspring_mutated, sigma=self.sigma, alpha=self.alpha)
            self.eliminate_population_fs_tournament(population=self.population, offsprings=offspring_mutated, sigma=self.sigma, alpha=self.alpha, k=3)
            #self.eliminate_population_fs_edges(population=self.population, offsprings=offspring_mutated, sigma=self.sigma, alpha=self.alpha)
            
            time_end = time.time()
            time_elimination = time_end - time_start
            
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
            #print(f"\n Mean Objective --> {meanObjective} \n Best Objective --> {bestObjective}")
            self.check_stuck_pop(new_bestFit=self.best_objective)
            #self.check_stoppingCriteria(limit_stuck=15)


             
            time_end_iteration = time.time()
            diff_time_iteration = time_end_iteration - time_start_iteration
            self.update_time(time_initalization=initialization_time,time_selection=selection_time,time_crossover=time_crossover,time_mutation=time_mutation,time_elimination=time_elimination,time_mutation_population=time_mutation_population,time_local_search=time_local_search,time_iteration=diff_time_iteration)


            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
			
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
           
            if timeLeft < 0:
                break

            self.check_stoppingCriteria(time_left=timeLeft,limit_stuck=15)
            
       
            
           

        if plot is True:
            self.plot_fitness_dynamic()
            self.plot_timing_info()
        #self.print_best_solution()
        #self.plot_fitness_dynamic()
        #self.plot_timing_info()
        
        
        return 0


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 1) Initalization ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def add_initialSolution(self,initial_solution):
        '''
        - Add an initial solution to the population
        '''
        self.initial_solution = initial_solution
        print(f"\n Initial solution is : {self.initial_solution}")  

    def set_initialization(self,num_individuals_to_add=None):
        '''
        - Initialize the population
        '''
        if num_individuals_to_add is None:
            self.population = np.array([np.random.permutation(self.gen_size) for _ in range(self.population_size)])
        else:
            self.population = np.array([np.random.permutation(self.gen_size) for _ in range(num_individuals_to_add)])
        #self.fitness = self.calculate_distance_population(self.population)
        self.fitness, self.distance_scores, self.average_bdp_scores = self.fitness_function_calculation(population=self.population, weight_distance=self.weight_distance, weight_bdp=self.weight_bdp, distance_matrix=self.distance_matrix)
        print(f"\n Initial Population: {self.population}")
        print(f"\n Initial Fitness: {self.fitness}")
        self.print_model_info()
    
    
    def set_initialization_onlyValid_numpy(self,fitness_threshold=1e8):
        """
        Initialize the population with valid individuals (fitness < fitness_threshold), using fast NumPy operations.

        Parameters:
        - fitness_threshold: float
            Maximum allowed fitness for valid individuals. Higher fitness individuals are discarded.
        """
        

        # Pre-allocate population array and fitness list
        population_size = self.population_size
        gen_size = self.gen_size
        distance_matrix = self.distance_matrix

        population = np.empty((population_size, gen_size), dtype=np.int32)
        fitness = np.empty(population_size, dtype=np.float64)

        current_index = 0

        while current_index < population_size:
        
            # Generate a batch of random individuals
            batch_size = population_size - current_index
            candidates = np.array([np.random.permutation(gen_size) for _ in range(batch_size)])

            # Calculate fitness for the batch
            candidate_fitness = self.calculate_distance_population(candidates)

            # Filter valid candidates with fitness < fitness_threshold
            valid_mask = candidate_fitness < fitness_threshold
            valid_candidates = candidates[valid_mask]
            valid_fitness = candidate_fitness[valid_mask]

            # Check how many valid candidates were found and add them to the population
            num_valid_candidates = len(valid_candidates)
            if num_valid_candidates > 0:
                # Add valid candidates to population and fitness
                population[current_index:current_index + num_valid_candidates] = valid_candidates
                fitness[current_index:current_index + num_valid_candidates] = valid_fitness
                current_index += num_valid_candidates

            # Print status
            #print(f"Accepted {num_valid_candidates} candidates (fitness < {fitness_threshold}). "f"Total: {current_index}/{population_size}")
        
        #print(f"\nFinal Initial Population: {population}")
        #print(f"\nFinal Initial Fitness: {fitness}")

        self.distance_matrix=self.check_inf(self.distance_matrix,replace_value=1e8)
        self.population = population
        self.fitness = fitness
        self.print_model_info()

    def set_initialization_onlyValid_numpy_incremental(self, fitness_threshold=1e8):
        """
        Initialize the population with valid individuals (fitness < fitness_threshold), using fast NumPy operations.
        This version generates individuals incrementally, adding one city at a time while ensuring the fitness remains below the threshold.
        
        Parameters:
        - fitness_threshold: float
            Maximum allowed fitness for valid individuals. Higher fitness individuals are discarded.
        """
        random_prob = 0.01
        percentage_greedy = 0.2
        index_greedy = int(percentage_greedy*self.population_size)
        percentage_random = 0.8
        index_random = int(percentage_random*self.population_size)
        print(f"Index greedy is : {index_greedy} and index random is : {index_random} of the population size {self.population_size}")
        population_size = self.population_size
        gen_size = self.gen_size
        distance_matrix = self.distance_matrix

        # Pre-allocate population array and fitness list
        population = -1*np.ones((population_size, gen_size), dtype=np.int32)
        fitness = -1*np.ones(population_size, dtype=np.float64)

        if self.initial_solution is not None:
            # Add the initial solution to the population
            population[0] = self.initial_solution
            fitness[0] = self.calculate_total_distance_individual(self.initial_solution,distance_matrix=distance_matrix)
            #print(f"\nInitial solution added to the population: {population} with fitness {fitness}")
            current_index = 1
        else:
            current_index = 0

        while current_index < population_size:
            #print(f"\nPopulation progress: {current_index}")
            if current_index <= index_greedy:
                
                random_prob = 0.01
                #print(f" Greedy: Random prob is : {random_prob}")
            else:
                random_prob = 0.5
                #print(f" Random: Random prob is : {random_prob}")
            
            # Initialize a new individual incrementally
           
            route = np.full(gen_size, -1, dtype=np.int32)  # -1 indicates unassigned cities
            visited = np.zeros(gen_size, dtype=bool)  # City visitation status
            current_distance = 0.0

            # Start with a random city
            start_city = np.random.randint(gen_size)
            route[0] = start_city
            visited[start_city] = True

            # Vectorized approach to add cities incrementally
            for i in range(1, gen_size):
                prev_city = route[i - 1]
                
                # Get the distances from the previous city to all unvisited cities
                remaining_cities = np.where(~visited)[0]  # Indices of unvisited cities
                remaining_distances = distance_matrix[prev_city, remaining_cities]

                # Mask out cities that exceed the fitness threshold
                valid_cities_mask = remaining_distances < fitness_threshold

                if np.any(valid_cities_mask):
                    # Get the valid cities and their corresponding distances
                    valid_cities = remaining_cities[valid_cities_mask]
                    valid_distances = remaining_distances[valid_cities_mask]

                    # Select the city with the minimum distance
                    random_index = np.random.choice(len(valid_cities))
                    #best_city_idx = np.argmax(valid_distances)     #Smallest distance
                    #best_city_idx = np.argmin(valid_distances)     #Smallest distance
                    #print(f"Best distnace is : {valid_distances[best_city_idx]} vs random distance {valid_distances[random_index]}")
                    #best_city_idx = random_index

                    # if i is even, select the city with the maximum distance
                    if np.random.rand() < random_prob:
                        best_city_idx = random_index
                    
                    else:
                        best_city_idx = np.argmin(valid_distances)     #Smallest distance
                        

                    best_city = valid_cities[best_city_idx]

                    # Assign the best city to the current position in the route
                    route[i] = best_city
                    current_distance += valid_distances[best_city_idx]
                    visited[best_city] = True
                else:
                    # If no valid city is found, discard the individual
                    break
            else:
                # If a valid individual is created (i.e., the loop didn't break)
                #print(f"CUrrent index is : {current_index}")
                population[current_index] = route
                fitness[current_index] = current_distance
                current_index += 1

            # Optional status print (can be removed for better performance)
            if current_index % 100 == 0:
                #print(f"Population progress: {current_index}/{population_size}")
                a=12

        

        # Final population setup
        self.population = population
        self.fitness = fitness

        # Update internal distance matrix (replace any inf values with the specified value)
        self.distance_matrix = self.check_inf(self.distance_matrix, replace_value=1e8)
        
        # Recalculate the fitness for the entire population
        self.fitness = self.calculate_distance_population(self.population)

        #print me the best solution
        #best_idx = np.argmin(self.fitness)
        #best_solution = self.population[best_idx]
        
        #print(f"\nBest solution: {best_solution} with fitness {self.fitness[best_idx]}")

        #fitness_check = calculate_total_distance_individual(best_solution,distance_matrix=self.distance_matrix)
        #print(f"\n Fitness check: {fitness_check}")

        
        # Print final status
        #print(f"\nInitial Population shape: {self.population.shape}")
        #print(f"\nInitial population: {self.population}")
        self.check_population(self.population)
        self.print_model_info()

    def check_population(self, population):
        """
        Check the validity of the population. This includes:
        - Ensuring each individual has exactly `self.gen_size` cities.
        - Ensuring each individual has no duplicate cities.
        - Ensuring that every individual contains all the cities exactly once, i.e., the solution is a permutation of the cities.

        Parameters:
        - population: numpy array of shape (population_size, gen_size), representing the population.

        Raises:
        - ValueError: if any of the checks fail.
        """

        # Check 1: Ensure each individual has exactly `self.gen_size` cities
        if population.shape[1] != self.gen_size:
            raise ValueError(f"Each solution must have exactly {self.gen_size} cities, but found {population.shape[1]}.")

        # Check 2: Ensure no duplicate cities within each individual
        for i in range(population.shape[0]):
            if len(np.unique(population[i])) != self.gen_size:
                #print the individual with the duplicate cities and teh cities duplicates and the number of duplicates
                print(f"Individual {i} contains duplicate cities. Each individual should have unique cities.")
                print(f"Individual {i} is : {population[i]}")
                #print(f"Unique cities: {np.unique(population[i])}")
                print(f"Number of duplicates: {self.gen_size - len(np.unique(population[i]))}")
                raise ValueError(f"Individual {i} contains duplicate cities. Each individual should have unique cities.")

        # Check 3: Ensure every individual contains all cities exactly once
        all_cities_set = set(self.cities)  # This will be the set of all cities.
        for i in range(population.shape[0]):
            individual_set = set(population[i])  # Convert individual to set to check uniqueness
            if individual_set != all_cities_set:
                #raise ValueError(f"Individual {i} does not contain all the cities. Found cities: {individual_set}. Expected cities: {all_cities_set}.")
                print("Individual {i} does not contain all the cities. Found cities: {individual_set}. Expected cities: {all_cities_set}.")

        print("Population is valid.")

     
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 2) Selection ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    def selection_k_tournament(self, num_individuals, k=3):
        #print(f"\n K tournament selection: {num_individuals} individuals with k={k} and population size {self.population_size}")
        tournament_indices = np.random.choice(self.population_size, size=(num_individuals, k), replace=True)
       

        # Step 2: Get the fitness scores of the selected individuals
        tournament_fitness = self.fitness[tournament_indices]
        #print(f"\n Fitness of the tournament is : {tournament_fitness}")    
        

        # Step 3: Get the index of the individual with the best fitness in each tournament
        best_indices_in_tournament = np.argmin(tournament_fitness, axis=1)

        # Step 4: Use these indices to select the best individuals from each tournament
        best_selected_indices = tournament_indices[np.arange(num_individuals), best_indices_in_tournament]

        # Step 5: Return the selected individualsi
        return self.population[best_selected_indices]
    
    def selection_k_tournament_population(self,num_individuals,population,fitness,k):
        '''
        - Select the best individuals from the population based on the k_tournament
        '''
    
        # Step 1: Randomly choose k individuals for each tournament
        tournament_indices = np.random.choice(population.shape[0], size=(num_individuals, k), replace=False)
        #print(f"\n tournament indices: {tournament_indices}")

        # Step 2: Get the fitness scores of the selected individuals
        tournament_fitness = fitness[tournament_indices]
        #print(f"\n Fitness of the tournament is : {tournament_fitness}")    
        

        # Step 3: Get the index of the individual with the best fitness in each tournament
        best_indices_in_tournament = np.argmin(tournament_fitness, axis=1)

        # Step 4: Use these indices to select the best individuals from each tournament
        best_selected_indices = tournament_indices[np.arange(num_individuals), best_indices_in_tournament]

        #step 5 retrieve also teh fitness of the best individuals
        best_selected_fitness = fitness[best_selected_indices]

        # Step 5: Return the selected individuals and its fitness
        return population[best_selected_indices],best_selected_fitness
    
        
        


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 4) Crossover & Mutation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def crossover_order_population_hybrid(self,population):
        num_parents, num_cities = population.shape
        
        # Generate crossover points
        crossover_indices = np.random.randint(0, num_cities, size=(num_parents, 2))
        crossover_indices.sort(axis=1)  # Ensure crossover_index1 <= crossover_index2

        # Parent pairing (circular shift)
        parent1 = population
        parent2 = np.roll(population, shift=-1, axis=0)

        # Initialize children with -1 (unfilled)
        children_population = -1 * np.ones_like(population)

        # Step 1: Copy the crossover segments from parent1 to children
        for i in range(num_parents):
            start, end = crossover_indices[i]
            children_population[i, start:end] = parent1[i, start:end]

        # Step 2: Randomize the filling of the remaining cities
        for i in range(num_parents):
            start, end = crossover_indices[i]
            copied_cities = parent1[i, start:end]

            # Mask to identify cities in parent2 that are not in the copied segment
            mask = ~np.isin(parent2[i], copied_cities)

            # Extract the remaining cities in parent2
            remaining_cities = parent2[i, mask]

            # Shuffle the remaining cities to encourage diversity
            #np.random.shuffle(remaining_cities)                         #------- Importannt: Maybe too much diversity!

            # Find unfilled positions in the child
            unfilled_indices = np.where(children_population[i] == -1)[0]

            # Place shuffled remaining cities into the unfilled indices
            children_population[i, unfilled_indices] = remaining_cities

        return children_population
    
    def crossover_order_population(self, population):
        """
        Perform Order Crossover (OX) for a population of TSP tours using NumPy.

        Parameters:
        - population: numpy array of shape (num_parents, num_cities),
                    where each row is a TSP tour.

        Returns:
        - children_population: numpy array of shape (num_parents, num_cities),
                                containing the offspring population.
        """
        num_parents, num_cities = population.shape

        # Step 1: Generate two random crossover points for all parents
        crossover_indices = np.random.randint(0, num_cities, size=(num_parents, 2))
        crossover_indices.sort(axis=1)  # Ensure crossover_index1 <= crossover_index2

        # Step 2: Pair parents (parent1 pairs with parent2, which is the next parent in population)
        parent1 = population
        parent2 = np.roll(population, shift=-1, axis=0)  # Circular shift for parent pairing

        # Step 3: Initialize children with -1 (unfilled)
        children_population = -1 * np.ones_like(population)

        # Step 4: Copy crossover segments from parent1 to children
        for i in range(num_parents):
            start, end = crossover_indices[i]
            children_population[i, start:end] = parent1[i, start:end]

        # Step 5: Fill the remaining slots in each child from parent2
        for i in range(num_parents):
            # Extract the already filled segment
            start, end = crossover_indices[i]
            copied_cities = parent1[i, start:end]

            # Create a mask for cities in parent2 not in the copied segment
            mask = ~np.isin(parent2[i], copied_cities)

            # Extract the remaining cities in parent2
            remaining_cities = parent2[i, mask]

            # Find indices in the child that are unfilled (-1)
            unfilled_indices = np.where(children_population[i] == -1)[0]

            # Place the remaining cities into the unfilled indices
            children_population[i, unfilled_indices] = remaining_cities

        return children_population
        
    
    def crossover_singlepoint_population(self, population):
        # Number of parents in the population
        num_parents = population.shape[0]
        #print(f"\n POPULATION: Parent Population is : {population[1]}" )
        
        # Initialize the children population
        children_population = np.zeros_like(population)
        
        # Generate random crossover points
        crossover_index = np.random.randint(1, self.gen_size)
        
        # Perform crossover for each pair of parents
        for i in range(num_parents):
            # Get the indices of the parents
            parent1 = population[i]
            parent2 = population[(i + 1) % num_parents]
            
            # Perform crossover
            child1, child2 = self.crossover_singlepoint(parent1, parent2)
            
            # Add the children to the population
            children_population[i] = child1
            children_population[(i + 1) % num_parents] = child2

        #print(f"\n CROSSOVER: Children Population is : {children_population[1]}" )
        
        return children_population
    

    def crossover_singlepoint(self, parent1, parent2):
        """
        Perform Order Crossover (OX) for valid TSP offspring.

        Parameters:
        - parent1: numpy array, representing a parent tour (e.g., [0, 1, 2, 3, ...])
        - parent2: numpy array, representing a parent tour (e.g., [3, 2, 1, 0, ...])

        Returns:
        - child1, child2: Two offspring with no duplicate cities and valid tours.
        """
        num_cities = len(parent1)
        
        # Step 1: Select two crossover points randomly
        crossover_index1 = np.random.randint(0, num_cities)
        crossover_index2 = np.random.randint(crossover_index1, num_cities)
        
        # Step 2: Initialize offspring as arrays of -1 (indicating unfilled positions)
        child1 = -1 * np.ones(num_cities, dtype=int)
        child2 = -1 * np.ones(num_cities, dtype=int)
        
        # Step 3: Copy the selected segment from parent1 to child1, and from parent2 to child2
        child1[crossover_index1:crossover_index2] = parent1[crossover_index1:crossover_index2]
        child2[crossover_index1:crossover_index2] = parent2[crossover_index1:crossover_index2]
        
        # Step 4a: Fill the remaining positions in child1 with cities from parent2, while avoiding duplicates
        def fill_child(child, parent_segment, other_parent):
            current_idx = (crossover_index2) % num_cities  # Start filling from the end of the copied segment
            for city in other_parent:
                if city not in child:  # Avoid duplicates
                    child[current_idx] = city
                    current_idx = (current_idx + 1) % num_cities  # Wrap around circularly

        # Step 4b: PMX method
        def pmx_fill_child(child, parent_segment, other_parent, start, end):
            # Fill the remaining cities from the other parent using the PMX method
            for i in range(start, end):
                if parent_segment[i] not in child[start:end]:
                    city_to_fill = parent_segment[i]
                    place_index = i
                    while child[place_index] != -1:
                        city_to_fill = other_parent[place_index]
                        place_index = np.where(parent_segment == city_to_fill)[0][0]
                    child[place_index] = city_to_fill

            for i in range(num_cities):
                if child1[i] == -1:
                    child1[i] = parent2[i]
                if child2[i] == -1:
                    child2[i] = parent1[i]
                    
        fill_child(child1, parent1[crossover_index1:crossover_index2], parent2)
        fill_child(child2, parent2[crossover_index1:crossover_index2], parent1)

        # pmx_fill_child(child1, parent1, parent2, crossover_index1, crossover_index2)
        # pmx_fill_child(child2, parent2, parent1, crossover_index1, crossover_index2)

        
        return child1, child2




    def mutation_singlepoint_population(self, population,offsprings_flag=False):
        if offsprings_flag:
            mutation_rate = self.mutation_rate
        else:
            mutation_rate = self.mutation_rate_pop

        # Number of individuals in the population
        num_individuals = population.shape[0]
        
        # Initialize the mutated population
        mutated_population = np.copy(population)
        mutated_distance = self.calculate_distance_population(mutated_population)
        best_index = np.argmin(mutated_distance)
        
        # Perform mutation for each individual
        
        for i in range(num_individuals):
            if np.random.rand() < mutation_rate and i != best_index:
                mutated_population[i] = self.mutation_singlepoint(population[i], mutation_rate)
             
        
              

        #print(f"\n MUTATION: Children Population is : {mutated_population[1]}" )
        
        return mutated_population

    def mutation_singlepoint(self, individual, mutation_rate):
        mutated_individual = np.copy(individual)
        num_genes = len(mutated_individual)

        # Ensure valid mutation range
        max_mutations = max(1, num_genes)  # Ensure at least 1 possible mutation
        #num_mutations1 = np.random.randint(1, (num_genes-1)/2)
        num_mutations1 = np.random.randint(1, 5)

        # Select first set of mutation indices
        mutation_indices1 = np.random.choice(num_genes, size=num_mutations1, replace=False)
        
        # Calculate available indices
        available_indices = np.setdiff1d(np.arange(num_genes), mutation_indices1)
        # Ensure num_mutations is not larger than available_indices
        num_mutations2 = num_mutations1
        mutation_indices2 = np.random.choice(available_indices, size=num_mutations2, replace=False)

        #print(f"\n Mutation indices 1: {mutation_indices1} \n Mutation indices 2: {mutation_indices2}")

        # Perform the mutation (e.g., swap)
        mutated_individual[mutation_indices1], mutated_individual[mutation_indices2] = mutated_individual[mutation_indices2], mutated_individual[mutation_indices1],
    

        return mutated_individual

    



        # Function to calculate the total distance of a tour (route)
    

    def mutation_inversion_population(self, population):
        mutation_rate = self.mutation_rate

        # Number of individuals in the population
        num_individuals = population.shape[0]
        
        # Initialize the mutated population
        mutated_population = np.copy(population)
        mutated_distance = self.calculate_distance_population(mutated_population)
        best_index = np.argmin(mutated_distance)
        
        # Perform mutation for each individual
        for i in range(num_individuals):
            if np.random.rand() < mutation_rate and i != best_index:
                mutated_population[i] = self.mutation_inversion(mutated_population[i])
        
        return mutated_population

    def mutation_inversion(self, individual):
        mutated_individual = np.copy(individual)
        num_genes = len(mutated_individual)

        # Randomly choose the start and end indices for inversion
        start, end = np.sort(np.random.choice(num_genes, size=2, replace=False))

        # Reverse the subsequence between the start and end indices
        mutated_individual[start:end+1] = mutated_individual[start:end+1][::-1]

        return mutated_individual
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) Local Search ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Optimized Local Search Function

    # 0) Numba: Jit compilation
    def local_search_population_jit(self, population,mutation_population,n_best=2, max_iterations=10, k_neighbors=10, min_improvement_threshold=100):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals.
        '''
        distance_matrix = self.distance_matrix

        # Step 1: Fitness Calculation
        distances_population = self.calculate_distance_population(population)
        distances_mutation = self.calculate_distance_population(mutation_population)

        # Step 2: Find the top n_best individuals in each population
        possible_num_indices_mut = int(len(distances_mutation))
        #possible_num_indices_mut = 0
        best_indices_mutation = np.argsort(distances_mutation)[:possible_num_indices_mut]
        number_mutations = len(best_indices_mutation)
        #best_indices_population = np.argsort(distances_population)[:n_best]
        possible_num_indices_pop = int(len(distances_population))
        best_indices_population = np.argsort(distances_population)[:possible_num_indices_pop]
        #print(f"\n Best indices population have distance: {distances_population[best_indices_population]}")

        # Step 3: Combine individuals for local search (top n_best from both populations)
        population_toLocalSearch = np.concatenate((
            mutation_population[best_indices_mutation],
            population[best_indices_population]
        ))

        # Step 4: Perform local search on the combined population
        new_population = perform_local_search_population_jit(population=population_toLocalSearch, 
                                                            distance_matrix=distance_matrix, 
                                                            max_iterations=max_iterations, 
                                                            k_neighbors=k_neighbors, 
                                                            min_improvement_threshold=min_improvement_threshold)

        # Step 5: Update the populations with the results from local search
        # First, split the results from the local search back into the mutation and population parts
        new_mutation_population = new_population[:number_mutations]  # Top n_best from mutation population
        new_population_individuals = new_population[number_mutations:]  # Remaining individuals for the main population

        # Update mutation_population and population with the new local search results
        mutation_population[best_indices_mutation] = new_mutation_population
        population[best_indices_population] = new_population_individuals

        # Output timing and overhead information (you can include time measurement code if needed)
        # print(f"Total Parallel Execution Time: {total_parallel_time:.4f} seconds")

        return population, mutation_population
            
   
    def calculate_total_distance_individual(self,route, distance_matrix):
        '''
        Calculate the total distance of a given route
        '''
        current_indices = route[:-1]
        next_indices = route[1:]
        distances = distance_matrix[current_indices, next_indices]
        total_distance = np.sum(distances) + distance_matrix[route[-1], route[0]]  # Return to start
        return total_distance

    def two_opt_no_loops_opt(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
         #print(f"\n ------------LOCAL SEARCH------------")
        best_route = np.copy(route)
        #best_distance = self.calculate_total_distance_individual(best_route, distance_matrix)
        n = len(route)
        
        # Generate all pairs of indices i, j (i < j)
        i_indices, j_indices = np.triu_indices(n, k=2)
        
        # Calculate the old and new distances for all (i, j) pairs
        i_next = (i_indices + 1) % n
        j_next = (j_indices + 1) % n
        old_distances = (
            distance_matrix[best_route[i_indices], best_route[i_next]] +
            distance_matrix[best_route[j_indices], best_route[j_next]]
        )
        
        new_distances = (
            distance_matrix[best_route[i_indices], best_route[j_indices]] +
            distance_matrix[best_route[i_next], best_route[j_next]]
        )
        
        # Calculate delta_distances once
        delta_distances = new_distances - old_distances
        
        improvement = True
        iteration = 0
        while improvement and iteration < max_iterations:
            improvement = False
            #print(f"\n Iteration number {iteration}")
            
            # Identify the top k pairs with the largest improvement (most negative delta_distances)
            #time_start = time.time()
            #top_k_indices = heapq.nsmallest(k_neighbors, range(len(delta_distances)), key=lambda x: delta_distances[x])
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]
            #time_end = time.time()
            #time_sorting = time_end - time_start
            #print(f"\n Time for sorting: {time_sorting}")
            
            if np.any(delta_distances[top_k_indices] < 0):
                #print(f"\n There is an improvement")
                #print(f"\n Iteration number {iteration}")
                
        
               
                improvement = True
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
                i, j = i_indices[best_swap_index], j_indices[best_swap_index]
                
                # Perform the 2-opt swap: reverse the segment between i and j
                best_route[i + 1 : j + 1] = best_route[i + 1 : j + 1][::-1]
                
                # Update the distances for the swapped pairs
                old_distances = (
                    distance_matrix[best_route[i_indices], best_route[i_next]] +
                    distance_matrix[best_route[j_indices], best_route[j_next]]
                )
                
                new_distances = (
                    distance_matrix[best_route[i_indices], best_route[j_indices]] +
                    distance_matrix[best_route[i_next], best_route[j_next]]
                )
                
                # Recalculate delta_distances after the swap
                delta_distances = new_distances - old_distances
             
            
            iteration += 1
        print(f"\n    LS Iterations: {iteration}")
        return best_route

   
    
    # 1) Normal 2 opt: NEE
    def compute_nearest_neighbors_vectorized(distance_matrix, k_neighbors=10):
        """
        Compute the k nearest neighbors for each city using vectorized operations.
        """
        n = distance_matrix.shape[0]
        nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k_neighbors + 1]
        return nearest_neighbors

   
        
    # 1) Normal 2 opt
    def local_search_population_2opt(self, population,mutation_population,n_best=2, max_iterations=10):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals using threading.
        '''
        
            
        distance_matrix = self.distance_matrix

        # Step 1: Fitness Calculation
        distances_population = self.calculate_distance_population(population)
        print(f"\n Best n distance from population: {np.sort(distances_population)[:n_best]}")
        distances_mutation = self.calculate_distance_population(mutation_population)
        print(f"\n Best n distance from mutation population: {np.sort(distances_mutation)[:n_best]}")
        

        best_indices_mutation = np.argsort(distances_mutation)[:n_best]
        print(f"\n Best indices mutation: {best_indices_mutation}")
        best_indices_population = np.argsort(distances_population)[:n_best]
        print(f"\n Best indices population: {best_indices_population}")

        population_toLocalSearch = np.concatenate((
            mutation_population[best_indices_mutation],
            population[best_indices_population]
        ))

        print(f"\n Population to local search: {population_toLocalSearch}")
        print(f"\n Population to local search shape: {population_toLocalSearch.shape}")

        # Measure Overhead Time
        start_setup = time.perf_counter()
        # Store optimized solutions in lists
        optimized_solutions = []

        for solution in population_toLocalSearch:
            sol_dist = self.calculate_total_distance_individual(solution, distance_matrix)
            print(f"\n Initial distance: {sol_dist}")
            sol = self.two_opt_no_loops_opt_out_noloops(solution ,distance_matrix, max_iterations=max_iterations)
            optimized_solutions.append(sol)
    
        end_task = time.perf_counter()  # End timing task execution
        end_setup = time.perf_counter()  # End timing full setup and execution

        # Overhead times
    
        total_parallel_time = end_setup - start_setup
        aggregation_time = end_setup - end_task


        print(f"Aggregation Time: {aggregation_time:.4f} seconds")
        print(f"Total Parallel Execution Time: {total_parallel_time:.4f} seconds")

         # First update the mutation_population with the best mutated individuals
        mutation_population[best_indices_mutation] = np.array(optimized_solutions[:len(best_indices_mutation)])

        # Now update the population with the best solutions from the population
        population[best_indices_population] = np.array(optimized_solutions[len(best_indices_mutation):])

        #calculate me again teh fitness of the population
        distances_population = self.calculate_distance_population(population)
        print(f"\n Best n distance from population: {np.sort(distances_population)[:n_best]}")
        distances_mutation = self.calculate_distance_population(mutation_population)
        print(f"\n Best n distance from mutation population: {np.sort(distances_mutation)[:n_best]}")
       
        return population, mutation_population
    
    def two_opt_no_loops_opt_out(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
        #print(f"\n ------------ 2-opt LOCAL SEARCH------------")
        print(f"\n ------------LOCAL SEARCH------------")   
        print(f" Max number of iterations: {max_iterations}")
        time_start = time.time()
        best_route = np.copy(route)
        inital_fitness = self.calculate_total_distance_individual(best_route, distance_matrix)
        routes = [] 
        n = len(route)
        
        # Generate all pairs of indices i, j (i < j)
        i_indices, j_indices = np.triu_indices(n, k=2)
        
        # Calculate the old and new distances for all (i, j) pairs
        i_next = (i_indices + 1) % n
        j_next = (j_indices + 1) % n
        old_distances = (
            distance_matrix[best_route[i_indices], best_route[i_next]] +
            distance_matrix[best_route[j_indices], best_route[j_next]]
        )
        
        new_distances = (
            distance_matrix[best_route[i_indices], best_route[j_indices]] +
            distance_matrix[best_route[i_next], best_route[j_next]]
        )

        print(f"Len of new distances: {len(new_distances)}")
        
        # Calculate delta_distances once
        delta_distances = new_distances - old_distances
        
        improvement = True
        iteration = 0
        while improvement and iteration < max_iterations:
            improvement = False
            #print(f"\n LS: Iteration number {iteration}")
            
            
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]
       
            best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
            i, j = i_indices[best_swap_index], j_indices[best_swap_index]
            #print(f"\n Route at iteration {iteration}: {best_route}")
            #print(f"\n Delta distances at iteration {iteration}: {delta_distances}")
            #print(f"\n Best swap index: {best_swap_index}")
            #print(f"\n i: {i} - j: {j}")
            
            # Perform the 2-opt swap: reverse the segment between i and j
            #print(f"Route segment before swap: {best_route[i + 1:j + 1]}")
            best_route[i + 1: j + 1] = best_route[i + 1: j + 1][::-1]
            #print(f"Route segment after swap: {best_route[i + 1:j + 1]}")
            
            #routes.append(best_route)
            routes.append(np.copy(best_route))

            improvement = True
            
            # Update the distances for the swapped pairs
            old_distances = (
                distance_matrix[best_route[i_indices], best_route[i_next]] +
                distance_matrix[best_route[j_indices], best_route[j_next]]
            )
            
            new_distances = (
                distance_matrix[best_route[i_indices], best_route[j_indices]] +
                distance_matrix[best_route[i_next], best_route[j_next]]
            )
            
            # Recalculate delta_distances after the swap
            delta_distances = new_distances - old_distances
            #print(f"\n Delta distances at iteration {iteration}: {delta_distances}")
            
            
            iteration += 1
        if len(routes) > 0:
            #print(f"\n Routes: {routes}")

            routes_np = np.array(routes)
            #check how many of the routes are unique
         
            #print(f"\n Unique Routes: {unique_routes}")

            final_fitness = self.calculate_distance_population(routes_np)
            #print(f"\n Final Fitness: {final_fitness}")
            best_sol = routes_np[np.argmin(final_fitness)]
            best_fit = final_fitness[np.argmin(final_fitness)]
            print(f"\n LS 2opt: Initial Fitness: {inital_fitness} - Final Fitness: {best_fit} at index: {np.argmin(final_fitness)}") 

            if inital_fitness > best_fit:
                best_route = best_sol
            else:
                best_route = route
                best_fit = inital_fitness
        else:
            print(f"\n LS: No routes found")
            best_route = route
            best_fit = inital_fitness
            #best_route = self.three_opt_no_loops_opt_out(best_route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10)

        #print(f"\n    LS Iterations: {iteration}")   
        time_end = time.time()
        time_local_search = time_end - time_start
        #print(f"\n Time for the local search: {time_local_search}",flush=True)
        return best_route
    
    def two_opt_no_loops_opt_out_epsilonAndGreedy(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=100):
        #print(f"\n ------------ 2-opt LOCAL SEARCH------------")
        # Parameters:
        epsilon = 0.2

        print(f"\n ------------LOCAL SEARCH------------")   
        print(f" Max number of iterations: {max_iterations}")
        time_start = time.time()
        best_route = np.copy(route)
        inital_fitness = self.calculate_total_distance_individual(best_route, distance_matrix)
        routes = [] 
        n = len(route)
        
        # Generate all pairs of indices i, j (i < j)
        i_indices, j_indices = np.triu_indices(n, k=2)
        
        # Calculate the old and new distances for all (i, j) pairs
        i_next = (i_indices + 1) % n
        j_next = (j_indices + 1) % n
        old_distances = (
            distance_matrix[best_route[i_indices], best_route[i_next]] +
            distance_matrix[best_route[j_indices], best_route[j_next]]
        )
        
        new_distances = (
            distance_matrix[best_route[i_indices], best_route[j_indices]] +
            distance_matrix[best_route[i_next], best_route[j_next]]
        )

        print(f"Len of new distances: {len(new_distances)}")
        
        # Calculate delta_distances once
        delta_distances = new_distances - old_distances
        
        improvement = True
        iteration = 0
        while  iteration < max_iterations:
            #improvement = False
            #print(f"\n LS: Iteration number {iteration}")
            
            
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]

            if np.random.rand() < epsilon:
                # Randomly select a pair of indices
                
                best_swap_index = np.random.choice(range(len(delta_distances)))
                #print(f"\n Randomly selecting a pair of indices being {best_swap_index}")
            else:
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]

            i, j = i_indices[best_swap_index], j_indices[best_swap_index]
         
            best_route[i + 1: j + 1] = best_route[i + 1: j + 1][::-1]
   
            routes.append(np.copy(best_route))

            # Update the distances for the swapped pairs
            old_distances = (
                distance_matrix[best_route[i_indices], best_route[i_next]] +
                distance_matrix[best_route[j_indices], best_route[j_next]]
            )
            
            new_distances = (
                distance_matrix[best_route[i_indices], best_route[j_indices]] +
                distance_matrix[best_route[i_next], best_route[j_next]]
            )
            
            # Recalculate delta_distances after the swap
            delta_distances = new_distances - old_distances
            #print(f"\n Delta distances at iteration {iteration}: {delta_distances}")
            
            iteration += 1
        if len(routes) > 0:

            routes_np = np.array(routes)
        

            final_fitness = self.calculate_distance_population(routes_np)
            #print(f"\n Final Fitness: {final_fitness}")
            best_sol = routes_np[np.argmin(final_fitness)]
            best_fit = final_fitness[np.argmin(final_fitness)]
            print(f"\n LS 2opt: Initial Fitness: {inital_fitness} - Final Fitness: {best_fit} at index: {np.argmin(final_fitness)}") 

            if inital_fitness > best_fit:
                best_route = best_sol
            else:
                best_route = route
                best_fit = inital_fitness
        else:
            print(f"\n LS: No routes found")
            best_route = route
            best_fit = inital_fitness
            #best_route = self.three_opt_no_loops_opt_out(best_route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10)

        #print(f"\n    LS Iterations: {iteration}")   
        time_end = time.time()
        time_local_search = time_end - time_start
        #print(f"\n Time for the local search: {time_local_search}",flush=True)
        return best_route
    
    def two_opt_no_loops_opt_out_epsilonAndGreedy_saved(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=100, last_route=None):
        # Parameters:
        epsilon = 0.0
        improvement = True
        iteration = 0
        prev_indices = None
        


        #print(f"\n ------------LOCAL SEARCH------------")   
       
        #print(f" Max number of iterations: {max_iterations} with epsilon: {epsilon} and last route: {last_route}")


        # 1. Intialize the best route
        time_start = time.time()
        routes = [] 
        n = len(route)
        if last_route is None:
            best_route = np.copy(route)
        else:
            best_route = last_route
            #print(f"  - LS: Using last route from before:")
            epsilon = 0.0

        #print(f" Max number of iterations: {max_iterations} with epsilon: {epsilon}")
    
        inital_fitness = self.calculate_total_distance_individual(route, distance_matrix)

        # 2. Genetae pair indices and calculate distances and deltas
        i_indices, j_indices = np.triu_indices(n, k=2)
        
        i_next = (i_indices + 1) % n
        j_next = (j_indices + 1) % n
        old_distances = (
            distance_matrix[best_route[i_indices], best_route[i_next]] +
            distance_matrix[best_route[j_indices], best_route[j_next]]
        )
    
        new_distances = (
            distance_matrix[best_route[i_indices], best_route[j_indices]] +
            distance_matrix[best_route[i_next], best_route[j_next]]
        )

        delta_distances = new_distances - old_distances
        #print(f"Delta distances: {delta_distances}")
        

        # 3. Start the local search
        while  iteration < max_iterations:
            #print(f"\n LS: Iteration number {iteration}")
            
            
            top_k_indices = np.argsort(delta_distances)

            if np.random.rand() < epsilon:
                best_swap_index = np.random.choice(range(len(delta_distances)))
            else:
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]

        
            i, j = i_indices[best_swap_index], j_indices[best_swap_index]
            #print(f"\n i: {i} - j: {j}")
         
            best_route[i + 1: j + 1] = best_route[i + 1: j + 1][::-1]

            if len(routes) > 0:
                start_time = time.time()
                count = np.sum(np.all(np.array(routes)==best_route,axis=1))
                end_time = time.time()
                time_count = end_time - start_time
                #print(f"Time for counting: {time_count}")

                if count >=2:
                    #print(f"  - LS: Stuck in a loop, count: {count}")

                    break
   
            routes.append(np.copy(best_route))
            #print(f"Route at iteration {iteration}: {best_route}")

            # Update the distances for the swapped pairs
            old_distances = (
                distance_matrix[best_route[i_indices], best_route[i_next]] +
                distance_matrix[best_route[j_indices], best_route[j_next]]
            )
            
            new_distances = (
                distance_matrix[best_route[i_indices], best_route[j_indices]] +
                distance_matrix[best_route[i_next], best_route[j_next]]
            )
            
            # Recalculate delta_distances after the swap
            delta_distances = new_distances - old_distances
            #print(f"\n Delta distances at iteration {iteration}: {delta_distances}")
            
            iteration += 1


        if len(routes) > 0:

            routes_np = np.array(routes)
        

            final_fitness = self.calculate_distance_population(routes_np)
            #print(f"\n Final Fitness: {final_fitness}")
            best_sol = routes_np[np.argmin(final_fitness)]
            best_fit = final_fitness[np.argmin(final_fitness)]
            #print(f"\n LS 2opt: Initial Fitness: {inital_fitness} - Final Fitness: {best_fit} at index: {np.argmin(final_fitness)} - Iterations: {iteration}") 

            if inital_fitness > best_fit:
                best_route = best_sol
                last_route = None
                #print(f"            LS: Improved: Last route: {last_route}")
                
            else:
                best_route = route
                best_fit = inital_fitness
                last_route = routes_np[-1]
                #print(f"            LS: Not Improved: ")
        else:
            #print(f"\n LS: No routes found")
            best_route = route
            best_fit = inital_fitness
  
        time_end = time.time()
        time_local_search = time_end - time_start
   
        return best_route,last_route
    

    # 2) MultiProcessors: threads

    def local_search_population_2opt_multip_parallel(self, population, mutation_population, n_best=5, max_iterations=10):
        distance_matrix = self.distance_matrix

        # Step 1: Fitness Calculation
        distances_population = self.calculate_distance_population(population)
        distances_mutation = self.calculate_distance_population(mutation_population)

        best_indices_mutation = np.argsort(distances_mutation)[:n_best]
        best_indices_population = np.argsort(distances_population)[:n_best]

        # Combine individuals for local search
        population_toLocalSearch = np.concatenate((
            mutation_population[best_indices_mutation],
            population[best_indices_population]
        ))

        # Measure overhead time
        start_setup = time.perf_counter()

        futures = []  # Store futures for parallel tasks

        # Detect available CPU cores
        available_cores = os.cpu_count()
        available_cores = 2
        max_workers = min(available_cores, n_best*2)  # Set a maximum of 12 or available cores, whichever is smaller
        print(f"Using {max_workers} worker processes based on available cores ({available_cores} cores detected).")

        # Create a ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to the executor
            for individual in population_toLocalSearch:
                print(f"Submitting task /{len(population_toLocalSearch)}")
                futures.append(
                    executor.submit(
                        self.local_search_for_individual_parallel, 
                        individual, 
                        distance_matrix, 
                        max_iterations
                    )
                )
            # Collect results as tasks complete
            print("Waiting for tasks to complete...")
            # Collect results as tasks complete
            results = []
            for future in as_completed(futures):
                result = future.result()  # Get the result of the task
                results.append(result)
                print(f"Task completed, {len(results)}/{len(population_toLocalSearch)} results collected.")

        end_setup = time.perf_counter()

        # Step 4: Update population and mutation_population with the results
        mutation_population[best_indices_mutation] = np.array(results[:len(best_indices_mutation)])
        population[best_indices_population] = np.array(results[len(best_indices_mutation):])

        # Output timing and overhead information
        total_parallel_time = end_setup - start_setup
        print(f"Total Parallel Execution Time: {total_parallel_time:.4f} seconds")

        return population, mutation_population

    def local_search_population_2opt_multip(self, population,mutation_population,n_best=5, max_iterations=10,random=False, k_neighbors=10, heavy_ls = False):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals using threading.
        '''
        #Parameters:
        

        if self.counter_stuck >1:
            n_best_mut = n_best+2
            #print(f"\n Stuck flag is on, increasing the number of best mutations to {n_best_mut}")
            mut_iterations = 50
            n_best = n_best +2
            max_iterations = 300
        
            
        mut_iterations = 0
        n_best_mut = 0


        # Variables
        max_iterations_pop_list = max_iterations*np.ones(n_best)
        max_iterations_mut_list = max_iterations*np.ones(n_best)
        update_all_iter = False
        all_mutated = False
        last_route_list = []

        distance_matrix = self.distance_matrix

        # Step 1: Fitness Calculation
        distances_population = self.calculate_distance_population(population)
        distances_mutation = self.calculate_distance_population(mutation_population)

       

        best_indices_mutation = np.argsort(distances_mutation)[:n_best_mut]
        num_mutations = len(best_indices_mutation)

        if random:
            #best_indices_population = np.random.choice(len(distances_population), n_best, replace=False)
            best_indices_mutation2 = np.random.choice(len(distances_population), n_best, replace=False)
            population_toLocalSearch = np.concatenate((
                mutation_population[best_indices_mutation],
                mutation_population[best_indices_mutation2]
            ))
        else:
            best_indices_population = np.argsort(distances_population)[:n_best]
            best_population_selected = population[best_indices_population]
            #print(f"\n Best n distance from population: {np.sort(distances_population)[:n_best]}")
            
    
            
            population_toLocalSearch = np.concatenate((
                mutation_population[best_indices_mutation],
                population[best_indices_population]
            ))

            num_pop = len(best_indices_population)  

        #fitness_toLocalSearch = self.calculate_distance_population(population_toLocalSearch)
        #print(f"\n Fitness to local search: {fitness_toLocalSearch}")

        num_total = num_pop + num_mutations
        max_iterations_toLocalSearch = max_iterations*np.ones(num_total)
        last_route_list = [-1]*num_total
       
        for idx,element in enumerate(population_toLocalSearch.tolist()):
            #print(f"\n Element: {element} and idx {idx}")
            if idx < num_mutations:
                max_iterations_toLocalSearch[idx] = mut_iterations
            if tuple(element) in self.cache_ls:
                last_route_list[idx]= self.cache_ls[tuple(element)]
                #print(f"\n Found in cache: {last_route_list[idx]}")
            else:
                last_route_list[idx] = None

        #print(f"Last route list: {last_route_list}")

        
        with ThreadPoolExecutor() as executor:
            # Submit each individual to the pool for parallel processing
            futures = [
                executor.submit(self.local_search_for_individual, ind, distance_matrix, max_iterations_toLocalSearch[idx],k_neighbors,heavy_ls = heavy_ls, last_route = last_route_list[idx])
                for idx,ind in enumerate(population_toLocalSearch)
            ]
            
            # Collect results from each future
            results = [future.result() for future in futures]
            
        
        end_task = time.perf_counter()  # End timing task execution
        end_setup = time.perf_counter()  # End timing full setup and execution

        # Step 4: Update population and mutation_population with the results
        if random:
            mutation_population[best_indices_mutation] = np.array(results[:len(best_indices_mutation)])
            #population[best_indices_population] = np.array(results[len(best_indices_mutation):])
            mutation_population[best_indices_mutation2] = np.array(results[len(best_indices_mutation):])    #Omnly for random

        else:
            #mutation_population[best_indices_mutation] = np.array(results[:len(best_indices_mutation)])
            #population[best_indices_population] = np.array(results[len(best_indices_mutation):])
            # new way
            for idx,element in enumerate(results):
                #print(f"\n Element: {element} and idx: {idx}")
                if idx < num_mutations:
                    mutation_population[best_indices_mutation[idx]] = element[0]
                else:
                    population[best_indices_population[idx-num_mutations]] = element[0]
                    self.add_entry_cache_ls_pop_lastBest(route=element[0],last_best=element[1])

       
        return population, mutation_population
    
  
    def local_search_for_individual(self,population, distance_matrix, max_iterations=10,k_neighbors=10,heavy_ls = False, last_route = None):
        #fitness_original = self.calculate_total_distance_individual(population[i], distance_matrix) 
        #population[i] = self.two_opt_no_loops_opt(population[i], distance_matrix, max_iterations)
        #population = self.two_opt_no_loops_opt_out(population, distance_matrix, max_iterations)
        #population = self.two_opt_no_loops_opt_out(population, distance_matrix, max_iterations)
        if heavy_ls:
            #population = self.two_opt_no_loops_opt_out(population, distance_matrix, max_iterations)
            #population = self.two_opt_no_loops_opt_out_epsilonAndGreedy(route=population, distance_matrix=distance_matrix, max_iterations=max_iterations)   # Previous best
        
            population,last_route = self.two_opt_no_loops_opt_out_epsilonAndGreedy_saved(route=population, distance_matrix=distance_matrix, max_iterations=max_iterations,last_route=last_route)
         
        else:
            a = 1
            #population = self.two_opt_no_loops_opt_out_epsilonAndGreedy(route=population, distance_matrix=distance_matrix, max_iterations=max_iterations)

        #population[i] = self.two_opt_better1_n(population[i], distance_matrix, max_iterations)
        #fitness_after = self.calculate_total_distance_individual(population[i], distance_matrix)
        #print(f"\n Fitness before: {fitness_original} - Fitness after: {fitness_after}")

        
        return population,last_route
    
  
    def local_search_for_individual_parallel(self,individual ,distance_matrix, max_iterations=10):
        #logging.info(f"Process ID: {os.getpid()}")  # Log the process ID
        pid = os.getpid()  # Get the process ID
        logging.debug(f"Task is running in process {pid}")
        #fitness_original = self.calculate_total_distance_individual(population[i], distance_matrix) 
        #population[i] = self.two_opt_no_loops_opt(population[i], distance_matrix, max_iterations)
        #individual = self.two_opt_no_loops_opt_out(individual, distance_matrix, max_iterations)
        #individual = self.two_opt_no_loops_opt_out_noloops(individual, distance_matrix, max_iterations)
        individual = self.two_opt_no_loops_opt_out(individual, distance_matrix, max_iterations)
        #population[i] = self.two_opt_better1_n(population[i], distance_matrix, max_iterations)
        #fitness_after = self.calculate_total_distance_individual(population[i], distance_matrix)
        #print(f"\n Fitness before: {fitness_original} - Fitness after: {fitness_after}")

        
        return individual


    

    # 3) 3-opt   
    def three_opt_no_loops_opt_out(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
        print(f"\n ------------ 3-opt LOCAL SEARCH------------")   
        print(f" Max number of iterations: {max_iterations}")
        time_start = time.time()
        best_route = np.copy(route)
        initial_fitness = self.calculate_total_distance_individual(best_route, distance_matrix)
        routes = [] 
        n = len(route)
        
        # Generate all triples of indices (i < j < k)
        i_indices, j_indices, k_indices = np.array(np.meshgrid(range(n), range(n), range(n))).reshape(3, -1)
        valid_triples = (i_indices < j_indices) & (j_indices < k_indices)
        i_indices, j_indices, k_indices = i_indices[valid_triples], j_indices[valid_triples], k_indices[valid_triples]

        # Calculate old and new distances for all (i, j, k) triples
        i_next = (i_indices + 1) % n
        j_next = (j_indices + 1) % n
        k_next = (k_indices + 1) % n

        old_distances = (
            distance_matrix[best_route[i_indices], best_route[i_next]] +
            distance_matrix[best_route[j_indices], best_route[j_next]] +
            distance_matrix[best_route[k_indices], best_route[k_next]]
        )
        
        new_distances = (
            distance_matrix[best_route[i_indices], best_route[k_indices]] +
            distance_matrix[best_route[j_indices], best_route[i_next]] +
            distance_matrix[best_route[k_indices], best_route[j_next]]
        )

        # Calculate delta_distances
        delta_distances = new_distances - old_distances
        
        improvement = True
        iteration = 0
        while improvement and iteration < max_iterations:
            improvement = False
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]  # Select top k neighbors

            if np.any(delta_distances[top_k_indices] < 0):
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
                i, j, k = i_indices[best_swap_index], j_indices[best_swap_index], k_indices[best_swap_index]

                # Perform the 3-opt swap (reverse and rejoin segments)
                # Split the route into three parts: [i...j], [j...k], [k...end]
                best_route[i + 1:j + 1] = best_route[i + 1:j + 1][::-1]
                best_route[j + 1:k + 1] = best_route[j + 1:k + 1][::-1]
                best_route[k + 1:] = best_route[k + 1:][::-1]

                routes.append(np.copy(best_route))
                improvement = True

                # Update the distances for the swapped triples
                old_distances = (
                    distance_matrix[best_route[i_indices], best_route[i_next]] +
                    distance_matrix[best_route[j_indices], best_route[j_next]] +
                    distance_matrix[best_route[k_indices], best_route[k_next]]
                )

                new_distances = (
                    distance_matrix[best_route[i_indices], best_route[k_indices]] +
                    distance_matrix[best_route[j_indices], best_route[i_next]] +
                    distance_matrix[best_route[k_indices], best_route[j_next]]
                )

                # Recalculate delta_distances after the swap
                delta_distances = new_distances - old_distances

            iteration += 1
        
        if len(routes) > 0:
            routes_np = np.array(routes)
            final_fitness = self.calculate_distance_population(routes_np)
            best_sol = routes_np[np.argmin(final_fitness)]
            best_fit = final_fitness[np.argmin(final_fitness)]
            print(f"\n LS 3opt: Initial Fitness: {initial_fitness} - Final Fitness: {best_fit} at index: {np.argmin(final_fitness)}")

            if initial_fitness > best_fit:
                best_route = best_sol
            else:
                best_route = route
                best_fit = initial_fitness
        else:
            print(f"\n LS: No routes found")
            best_route = route
            best_fit = initial_fitness

        time_end = time.time()
        time_local_search = time_end - time_start
        return best_route

    def local_search_population_3opt(self, population, max_iterations=10):
        '''
        Optimized local search for the population: applies 3-opt to the top individuals
        '''
        distance_matrix = self.distance_matrix
        
        # Step 1: Evaluate fitness for all individuals in the population
        distances = self.calculate_distance_population(population)
        
        # Step 2: Select the top `n_best` individuals
        n_best = 2  # Or set to the desired number of top individuals
        best_indices = np.argsort(distances)[:n_best]
        
        # Step 3: Apply 3-opt to the selected top individuals
        for i in best_indices:
            population[i] = self.three_opt_no_loops_opt(population[i], distance_matrix, max_iterations, k_neighbors=10)
        
        return population
    
   

   
    # 5) Greedy Algorithm:
    def local_search_population_greedy(self, population,mutation_population,n_best=5, max_iterations=10,random=False, k_neighbors=10, heavy_ls = False):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals using threading.
        '''
        #Parameters:
        results= []

        n_best = n_best +5
       
        
            
        mut_iterations = 0
        n_best_mut = n_best


        # Variables
        max_iterations_pop_list = max_iterations*np.ones(n_best)
        max_iterations_mut_list = max_iterations*np.ones(n_best)
        update_all_iter = False
        all_mutated = False
        last_route_list = []

        distance_matrix = self.distance_matrix

        # Step 1: Fitness Calculation
        distances_population = self.calculate_distance_population(population)
        distances_mutation = self.calculate_distance_population(mutation_population)

       

        best_indices_mutation = np.argsort(distances_mutation)[:n_best_mut]
        num_mutations = len(best_indices_mutation)

        if random:
            #best_indices_population = np.random.choice(len(distances_population), n_best, replace=False)
            best_indices_mutation2 = np.random.choice(len(distances_population), n_best, replace=False)
            population_toLocalSearch = np.concatenate((
                mutation_population[best_indices_mutation],
                mutation_population[best_indices_mutation2]
            ))
        else:
            best_indices_population = np.argsort(distances_population)[:n_best]
            best_population_selected = population[best_indices_population]
            #print(f"\n Best n distance from population: {np.sort(distances_population)[:n_best]}")
            
            population_toLocalSearch = np.concatenate((
                mutation_population[best_indices_mutation],
                population[best_indices_population]
            ))

            num_pop = len(best_indices_population)  

        fitness_toLocalSearch = self.calculate_distance_population(population_toLocalSearch)
        #print(f"\n Fitness to local search: {fitness_toLocalSearch}")

        num_total = num_pop + num_mutations
        max_iterations_toLocalSearch = max_iterations*np.ones(num_total)
        
       

        '''
        with ThreadPoolExecutor() as executor:
            # Submit each individual to the pool for parallel processing
            futures = [
                executor.submit(self.local_search_for_individual, ind, distance_matrix, max_iterations_toLocalSearch[idx],k_neighbors,heavy_ls = heavy_ls, last_route = last_route_list[idx])
                for idx,ind in enumerate(population_toLocalSearch)
            ]
            
            # Collect results from each future
            results = [future.result() for future in futures]
            
        '''
        
        for id,element in enumerate(population_toLocalSearch):
            #new_sol = self.local_search_greedy_rearrange(route=element, distance_matrix=distance_matrix, max_iterations=int(max_iterations_toLocalSearch[id]))
            new_sol = local_search_greedy_rearrange_jit(route=element, distance_matrix=distance_matrix, max_iterations=int(max_iterations_toLocalSearch[id]))
              
            results.append(new_sol)
            
        end_task = time.perf_counter()  # End timing task execution
        end_setup = time.perf_counter()  # End timing full setup and execution

        # Step 4: Update population and mutation_population with the results
        if random:
            mutation_population[best_indices_mutation] = np.array(results[:len(best_indices_mutation)])
            #population[best_indices_population] = np.array(results[len(best_indices_mutation):])
            mutation_population[best_indices_mutation2] = np.array(results[len(best_indices_mutation):])    #Omnly for random

        else:
            #mutation_population[best_indices_mutation] = np.array(results[:len(best_indices_mutation)])
            #population[best_indices_population] = np.array(results[len(best_indices_mutation):])
            #new_fitness = self.calculate_distance_population(np.array(results))
            #print(f"\n New Fitness: {new_fitness} vs old fitness: {fitness_toLocalSearch}")

            for idx,element in enumerate(results):
                #print(f"\n Element: {element} and idx: {idx}")
                if idx < num_mutations:
                    mutation_population[best_indices_mutation[idx]] = element
                else:
                    population[best_indices_population[idx-num_mutations]] = element
                    #self.add_entry_cache_ls_pop_lastBest(route=element[0],last_best=element[1])
            
            

       
        return population, mutation_population
    
    def greedy_rearrange_subsection(self,route, distance_matrix, start_index, end_index):
        """
        Rearrange a subsection of the route using a greedy strategy.

        Parameters:
        - route: Current route (1D array of city indices).
        - distance_matrix: Distance matrix.
        - start_index: Start index of the subsection to rearrange.
        - end_index: End index of the subsection to rearrange (inclusive).

        Returns:
        - new_route: Route with the subsection rearranged.
        """
        # Extract the subsection
        subsection = route[start_index:end_index+1]
        #print(f"        Greedy LS: Subsection: {subsection}")
        
        # Greedy rearrangement
        rearranged = []
        visited = set()
        
        # Start with the first city in the subsection
        current_city = subsection[0]
        rearranged.append(current_city)
        visited.add(current_city)
        
        while len(rearranged) < len(subsection):
            # Find the nearest unvisited city
            remaining_cities = [city for city in subsection if city not in visited]
            distances = [distance_matrix[current_city, city] for city in remaining_cities]
            next_city = remaining_cities[np.argmin(distances)]
            
            rearranged.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        # Replace the subsection in the route
        
        new_route = route.copy()
        new_route[start_index:end_index+1] = rearranged

        # Debugging: Explicit comparison
        #print(f"        Greedy LS: Subsection: {subsection}")
        #print(f"        Greedy LS: Rearranged: {rearranged}")
        #print(f"        Greedy LS: New route: {new_route}")
        #print(f"        Greedy LS: Old route: {route}")

        return new_route

    def local_search_greedy_rearrange(self,route, distance_matrix, max_iterations=50):
        """
        Apply local search by rearranging random subsections of the route using a greedy strategy.
        
        Parameters:
        - route: Initial route (1D array of city indices).
        - distance_matrix: Distance matrix.
        - max_iterations: Maximum number of iterations for the local search.
        
        Returns:
        - best_route: Optimized route.
        - best_fitness: Fitness of the optimized route.
        """
        best_route = route.copy()
        best_fitness = calculate_total_distance_individual(best_route, distance_matrix)
        print(f"\n Greedy LS: IMax number of iterations: {max_iterations}")
        for _ in range(max_iterations):
            #print(f"\n Iteration: {_}")
            # Select a random subsection
            start_index = np.random.randint(0, len(route) - 1)
            end_index = np.random.randint(start_index + 1, len(route))
            
            # Rearrange the subsection
            new_route = self.greedy_rearrange_subsection(best_route, distance_matrix, start_index, end_index)
            new_fitness = calculate_total_distance_individual(new_route, distance_matrix)
            #print(f"\        + Fitness before: {best_fitness} - Fitness after: {new_fitness}")
            
            # Update the best solution if improvement is found
            if new_fitness < best_fitness:
                print(f"\        + Fitness before: {best_fitness} - Fitness after: {new_fitness}")
                best_route = new_route
                best_fitness = new_fitness
        
        return best_route

    # 4) Common Functions
    

    def calculate_cumulatives_np(self,distance_matrix, sequence):
        """
        Calculate the forward and backward cumulative arrays for a given distance matrix and sequence using numpy built-in functions.

        Args:
        - distance_matrix (np.array): The distance matrix (NxN), where distance_matrix[i][j] is the distance from city i to city j.
        - sequence (list or np.array): The sequence of cities in the current tour.

        Returns:
        - forward_cumulative (np.array): Cumulative distance from the first city to each city.
        - backward_cumulative (np.array): Cumulative distance from the last city back to each city.
        """
        N = len(sequence)  # number of cities

        # Forward cumulative using numpy's cumsum and array slicing
        forward_distances = distance_matrix[sequence[:-1], sequence[1:]]  # distances between consecutive cities
        forward_cumulative = np.concatenate(([0], np.cumsum(forward_distances)))

        # Backward cumulative using numpy's cumsum and array slicing (reverse the sequence for backward)
        backward_distances = distance_matrix[sequence[1:], sequence[:-1]]  # distances between consecutive cities in reverse
        backward_cumulative = np.concatenate(([0], np.cumsum(backward_distances[::-1])))[::-1]  # reverse after cumsum

        return forward_cumulative, backward_cumulative


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) Elimnation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
    def eliminate_population_fs_tournament(self, population, offsprings, sigma, alpha, k):
        """
        Elimination with fitness sharing for TSP using k-tournament selection.
        Ensures the best individual is always selected and avoids duplicates.
        """
        # 1) Combine population & calculate their fitness
        combined_population = np.vstack((population, offsprings))

        combined_fitness, _, _ = self.fitness_function_calculation(
            population=combined_population,
            weight_distance=self.weight_distance,
            weight_bdp=self.weight_bdp,
            distance_matrix=self.distance_matrix
        )

        # 2) Initialize survivors
        survivors_idxs = []
        best_index = np.argmin(combined_fitness)  # Get the index of the best individual
        survivors_idxs.append(best_index)  # Always include the best individual first

        # Create a mask to track valid individuals
        valid_mask = np.ones(len(combined_population), dtype=bool)
        valid_mask[best_index] = False  # Mark the best individual as already selected

        # 3) Perform k-tournament selection for remaining survivors
        while len(survivors_idxs) < self.population_size:
            # Compute fitness sharing for the remaining individuals
            new_fitness = self.fitness_sharing_individual_np(
                population=combined_population,
                survivors=survivors_idxs,
                population_fitness=combined_fitness,
                sigma=sigma,
                alpha=alpha
            )

            # Randomly select k candidates from the valid candidates
            valid_indices = np.where(valid_mask)[0]  # Get indices of valid candidates
            if len(valid_indices) == 0:
                raise ValueError("No valid candidates remain for selection.")

            tournament_candidates = np.random.choice(valid_indices, size=min(k, len(valid_indices)), replace=False)

            # Select the individual with the best fitness among the tournament candidates
            best_in_tournament = tournament_candidates[np.argmin(new_fitness[tournament_candidates])]

            # Add the best candidate to survivors
            survivors_idxs.append(best_in_tournament)

            # Update the mask to exclude the selected individual
            valid_mask[best_in_tournament] = False

        # 4) Select the best individuals from the combined population
        self.population = combined_population[survivors_idxs]
        self.fitness = combined_fitness[survivors_idxs]
        self.distance_scores = self.calculate_distance_population(self.population)
        self.average_bpd_scores = self.average_bpd(self.population)
        self.hamming_distance, _ = self.calculate_hamming_distance_population(self.population)

    def eliminate_population_fs(self, population, offsprings, sigma, alpha):
        """
        Elimination with fitness sharing for TSP.
        """
        # 1) Combine population & calculate their fitness
        combined_population = np.vstack((population, offsprings))

        combined_fitness, _, _ = self.fitness_function_calculation(
            population=combined_population,
            weight_distance=self.weight_distance,
            weight_bdp=self.weight_bdp,
            distance_matrix=self.distance_matrix
        )

        # 2) Initialize survivors and get the best individual
        survivors_idxs = []
        best_index = np.argmin(combined_fitness)  # Index of the best individual (minimum fitness)
        survivors_idxs.append(best_index)  # Add the best individual to the survivors

        # Create a mask to track valid individuals
        valid_mask = np.ones(len(combined_population), dtype=bool)
        valid_mask[best_index] = False  # Mark the best individual as already selected

        # Start filling survivors
        while len(survivors_idxs) < self.population_size:
            # Compute fitness sharing for the remaining individuals
            new_fitness = self.fitness_sharing_individual_np(
                population=combined_population,
                survivors=survivors_idxs,
                population_fitness=combined_fitness,
                sigma=sigma,
                alpha=alpha
            )

            # Set the fitness of already-selected individuals to infinity to exclude them
            valid_fitness = np.where(valid_mask, new_fitness, np.inf)
            
            # Get the index of the next best individual
            best_index = np.argmin(valid_fitness)
            survivors_idxs.append(best_index)  # Add to survivors
            valid_mask[best_index] = False  # Update the mask to exclude this individual

        # 4) Select the best individuals from the combined population
        self.population = combined_population[survivors_idxs]
        self.fitness = combined_fitness[survivors_idxs]
        self.distance_scores = self.calculate_distance_population(self.population)
        self.average_bpd_scores = self.average_bpd(self.population)
        self.hamming_distance, _ = self.calculate_hamming_distance_population(self.population)
        

    def eliminate_population(self, population, offsprings):
        """
        Selects the best individuals from the combined population and offspring
        based on fitness.

        Parameters:
        - population: numpy array of shape (population_size, individual_size)
        - offspring: numpy array of shape (offspring_size, individual_size)

        Returns:
        - new_population: numpy array of shape (self.population_size, individual_size)
        """
        # Combine the original population with the offspring
        #print(f"\n Orginial --> {population}")
        #print(f"\n Offspring--> {offspring}")
        combined_population = np.vstack((population, offsprings))

        # Calculate fitness for the combined population
        fitness_scores, distances_scores, averageBdp_scores = self.fitness_function_calculation(population=combined_population, weight_distance=self.weight_distance, weight_bdp=self.weight_bdp, distance_matrix=self.distance_matrix)
    
        # Find unique rows in the combined population
        # Return unique population and the indices to map back to the original array
        unique_population, unique_indices = np.unique(combined_population, axis=0, return_index=True)

        # Get the fitness scores of unique individuals
        unique_fitness_scores = fitness_scores[unique_indices]
        unique_distances_scores = distances_scores[unique_indices]
        unique_averageBdp_scores = averageBdp_scores[unique_indices]

        # Sort unique individuals by fitness (ascending if lower is better)
        sorted_indices = np.argsort(unique_fitness_scores)[:self.population_size]

        # Select the best unique individuals
        self.population = unique_population[sorted_indices]
        self.fitness = unique_fitness_scores[sorted_indices]
        self.distance_scores = unique_distances_scores[sorted_indices]
        self.average_bpd_scores = unique_averageBdp_scores[sorted_indices]
        self.hamming_distance,_ = self.calculate_hamming_distance_population(self.population)
        ''''
        #best_indices = np.argsort(fitness_scores)[::-1][:self.population_size]
        best_indices = np.argsort(fitness_scores)[:self.population_size]

        # Select the best individuals
        self.population = combined_population[best_indices]
        self.fitness = fitness_scores[best_indices]
        self.distance_scores = distances_scores[best_indices]
        self.average_bpd_scores = averageBdp_scores[best_indices]
        self.hamming_distance = self.calculate_hamming_distance_population(self.population)
        '''
        
    def elimnation_population_lamdaMu(self,population,offsprings):
        '''
        - Elim
        '''
   
        combined_population = offsprings

        # Calculate fitness for the combined population
        fitness_scores, distances_scores, averageBdp_scores = self.fitness_function_calculation(population=combined_population, weight_distance=self.weight_distance, weight_bdp=self.weight_bdp, distance_matrix=self.distance_matrix)
       
        #print(f"\n Fitness V1--> {fitness_scores}")
       

        # Get the indices of the best individuals based on fitness
        
        best_indices = np.argsort(fitness_scores)[::-1][:self.population_size]
   


        # Select the best individuals
        self.population = combined_population[best_indices]
        self.fitness = fitness_scores[best_indices]
        self.distance_scores = distances_scores[best_indices]
        self.average_bpd_scores = averageBdp_scores[best_indices]

    def eliminate_population_kTournamenElitism(self,population,offsprings,elitism_percentage):
        '''
        - Eliminate
        '''

        # 1) Combine the original population with the offspring
        combined_population = np.vstack((population, offsprings))
        combined_fitness, combined_distance, combined_avg_bpd = self.fitness_function_calculation(population=combined_population, weight_distance=self.weight_distance, weight_bdp=self.weight_bdp, distance_matrix=self.distance_matrix)

        
        # 2) Number of indivuals to keep
        num_individual_keep = int((elitism_percentage/100)*self.population_size)
        if num_individual_keep <= 0:
            #raise ValueError(f"Elitism percentage ({elitism_percentage}%) is less than 0.")
            print(f" Num indivuals to keep is lower than zero, assigning 2 indivuals")
            num_individual_keep = 2
        
        unique_population, unique_indices = np.unique(combined_population, axis=0, return_index=True)

        # Get the fitness scores of unique individuals
        unique_fitness = combined_fitness[unique_indices]

        # Sort unique individuals by fitness (ascending if lower is better)
        best_indices = np.argsort(unique_fitness)[:num_individual_keep]
        best_individuals = unique_population[best_indices]
        best_fitness = unique_fitness[best_indices]
        #print(f"Best individuals: {best_individuals}")

        # 3) Select the best individuals from the remaining population based on the k_tournament
        num_individuals_rest = self.population_size - num_individual_keep
        if num_individuals_rest <= 0:
            raise ValueError(f"Elitism percentage ({elitism_percentage}%) is too high for the current population size ({self.population_size}).")
        
        remaining_indices = np.argsort(combined_fitness)[num_individual_keep:]
        remaining_population = combined_population[remaining_indices]
        remaining_population_fitness = combined_fitness[remaining_indices]
        
        kPopulation, kfitness = self.selection_k_tournament_population(num_individuals=num_individuals_rest, population=remaining_population, 
                                               fitness=remaining_population_fitness, k=2)
       
        #Make a print onm how many of the k.population are unique
        unique_k_population = np.unique(kPopulation, axis=0)
        print(f"Unique individuals in the k_population: {unique_k_population.shape[0]}")

        # 4) Combine the best individuals with the remaining population
        self.population = np.vstack((best_individuals, kPopulation))
        self.fitness = np.hstack((best_fitness, kfitness))
        self.distance_scores = self.calculate_distance_population(self.population)
        self.average_bpd_scores = self.average_bpd(self.population)
        self.hamming_distance,_ = self.calculate_hamming_distance_population(self.population)

        #print(f"Self popualtion shape: {self.population.shape}")

        if self.population.shape[0] > self.population_size:
            raise ValueError(f"New population size ({self.population.shape[0]}) is greater than the maximum allowed size ({self.population_size}).")

        
        

    def eliminate_population_elitism(self,population,offsprings):
        '''
        - Eliminate the population based on the elitism percentage. FOr the rest of the population use the k_tournament function
        '''
        #print(f"-----ELIMINATEWITH ELÑITISM FUNCTION-----")

        # 1) Combine the original population with the offspring
        combined_population = np.vstack((population, offsprings))

        # 2) Calculate fitness for the combined population
        fitness_scores = self.calculate_fitness(offsprings)
        combined_fitness = np.hstack((self.fitness, fitness_scores))
        #print(f"\n Combined population -> {combined_population} \n Combined fitness -> {combined_fitness}")
       

        # 3) Get the elite population based on the elitism percentage
        elitism_size = int((self.elistism/100)*self.population_size)
        best_indices = np.argsort(combined_fitness)[:elitism_size]
        best_indv = combined_population[best_indices]
        #print(f"\n Elite population -> {combined_population[best_indices]} \n Elite fitness -> {combined_fitness[best_indices]}")

        # 4) Get the remaining population size
        remaining_size = self.population_size - elitism_size
        remaining_population = combined_population[np.argsort(combined_fitness)[elitism_size:]]
        remaining_population_fitness = self.calculate_fitness(remaining_population)
        #print(f"\n Remaining population -> {remaining_population} \n Remaining fitness -> {remaining_population_fitness}")	

        

        # 5) Select the best individuals from the remaining population based on the k_tournament
        remaining_population_ordered, remaining_population_fitness_ordered = self.selection_k_tournament_population(num_individuals=remaining_size, population=remaining_population, fitness=remaining_population_fitness, k=self.k_tournament_k)


        # Print the elite population and the remaining population ordered
        #print(f"\n Elite population -> {best_indv} \n Remaining population ordered -> {remaining_population_ordered}")


        self.population = np.vstack((best_indv,remaining_population_ordered))
        self.fitness = self.calculate_fitness(self.population)
        
        
    def check_insert_individual(self,num_iterations=100,threshold_percentage = 10):
        '''
        - Insert the individual in the population
        '''
        # Convert lists to numpy arrays for faster indexing and operations
    
        
        if self.iteration > num_iterations:

            repeated_solutions_percentage = (self.num_repeated_solutions/self.population_size) *100

            if repeated_solutions_percentage >= threshold_percentage:

                print(f"Condition met for {num_iterations} iterations. with threshold {threshold_percentage}") 
                #print(f"Best fitness: {best_fitness_last} vs Mean fitness: {mean_fitness_last} witha diff of {mean_fitness_last - best_fitness_last}")
                self.population, self.fitness, self.distance_scores, self.average_bdp_scores = self.insert_individual(num_best_keep=2, population=self.population, fitness_scores=self.fitness)
                #self.mutation_rate = self.mutation_rate + 20
            
        
    
    def insert_individual(self,num_best_keep, population,fitness_scores):
        '''
        - Insert the individual in the population
        '''
        #print(f"Inserting {num_insert} individuals into the population.")
        #print(f"Population before insertion: {population}")
        
        # Get the best individuals from the population
       
        #best_indices = np.argsort(fitness_scores)[::-1][:num_best_keep]
        best_indices = np.argsort(fitness_scores)[:num_best_keep]
        best_individuals = population[best_indices]
        
        # Generate new individuals to insert
        num_insert = self.population_size - num_best_keep
        new_individuals = np.array([np.random.permutation(self.gen_size) for _ in range(num_insert)])
        
        # Insert the new individuals into the population
        new_population = np.vstack((best_individuals, new_individuals))

        #chekk the size of the population
        if new_population.shape[0] > self.population_size:
            raise ValueError(f"New population size ({new_population.shape[0]}) is greater than the maximum allowed size ({self.population_size}).")
        
        new_fitness_scores, new_distance_scores, new_bdp_scores = self.fitness_function_calculation(population=new_population, weight_distance=self.weight_distance, weight_bdp=self.weight_bdp, distance_matrix=self.distance_matrix)
            
        

        return new_population, new_fitness_scores, new_distance_scores, new_bdp_scores



    def eliminate_population_fs_edges(self, population, offsprings, sigma, alpha):
        '''
        - Elimination with fitness sharing for TSP
        '''
        # 1) Combine population & calculate their fitness
        combined_population = np.vstack((population, offsprings))

        self.calculate_edges_population(population=combined_population)
        

        combined_fitness, _, _ = self.fitness_function_calculation(population=combined_population, 
                                                                weight_distance=self.weight_distance, 
                                                                weight_bdp=self.weight_bdp, 
                                                                distance_matrix=self.distance_matrix)

        # 2) Initialize survivors and get the best individual
        survivors_idxs = -1 * np.ones(self.population_size, dtype=int)  # Initialize survivors
        best_index = np.argmin(combined_fitness)  # Index of the best individual (minimum fitness)
        survivors_idxs[0] = best_index  # The first survivor is the best individual

        counter = 1  # Start filling from index 1 since index 0 is already filled
        while np.any(survivors_idxs == -1):
            #print(f"Counter: {counter}")
            # Compute fitness sharing for the remaining individuals
            new_fitness = self.fitness_sharing_individual_np_edges(population=combined_population, survivors=survivors_idxs, 
                                                        population_fitness=combined_fitness, sigma=sigma, alpha=alpha)

            # Get the index of the next best individual
            best_index = np.argmin(new_fitness)
            survivors_idxs[counter] = best_index  # Add this individual to the survivors
            #print(f"Best index: {best_index} is added to the survivors")
            counter += 1

        # 4) Select the best individuals from the combined population
        self.population = combined_population[survivors_idxs]
        self.fitness = combined_fitness[survivors_idxs]
        self.distance_scores = self.calculate_distance_population(self.population)
        self.average_bpd_scores = self.average_bpd(self.population)
        self.hamming_distance,_ = self.calculate_hamming_distance_population(self.population)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 6) Fitness Calculation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def calculate_add_hamming_distance(self,population,crossover=False,crossover_old= False,mutation1=False,mutation2=False,local_search=False,elimination=False): 
        '''
        - Calculate the fitness of the population
        '''
        hamming_distance,_ = self.calculate_hamming_distance_population(population)
        if crossover:
            self.hamming_distance_crossover_list.append(hamming_distance)
        elif mutation1:
            self.hamming_distance_mutation1_list.append(hamming_distance)
        elif mutation2:
            self.hamming_distance_mutation2_list.append(hamming_distance)
        elif local_search:
            self.hamming_distance_local_search_list.append(hamming_distance)
        elif elimination:
            self.hamming_distance_elimination_list.append(hamming_distance)
        elif crossover_old:
            self.hamming_distance_crossoverOld_list.append(hamming_distance)   
        

    def check_equality_twoPopulations(self,population1,population2):
        '''
        - Check the equality between two populations
        '''
        equal = np.array_equal(population1,population2)
        #print(f"\n Equality between two populations: {equal}, with population 1: {population1} and population 2: {population2}")

        hamm_distance1,_ = self.calculate_hamming_distance_population(population1)
        hamm_distance2,_ = self.calculate_hamming_distance_population(population2)
        print(f"\n Hamming distance between two populations:  Population 1 (Old): {hamm_distance1} vs Population 2 (New):{hamm_distance2}")
        return 

    def calculate_distance_population(self,population):
        '''
        - Calculate the fitness of the population
        '''
        '''
        num_individuals = population.shape[0]
        fitness = np.zeros(num_individuals)
        for i in range(num_individuals):
            fitness[i] = np.sum(self.distance_matrix[population[i],np.roll(population[i],-1)])
        #print(f"Fitness shape --> {np.shape(fitness)} \n Fitness is : {fitness[1]}" )
        '''

        num_individuals = population.shape[0]

        # Get the indices for the current and next cities
        current_indices = population[:, :-1]
        next_indices = population[:, 1:]
        
        # Calculate the distances between consecutive cities
        distances = self.distance_matrix[current_indices, next_indices]
        
        # Sum the distances for each individual
        fitness = np.sum(distances, axis=1)
        
        # Add the distance from the last city back to the first city
        last_to_first_distances =self.distance_matrix[population[:, -1], population[:, 0]]
        fitness += last_to_first_distances

        
        #print(f"Fitness shape --> {np.shape(fitness)} \n Fitness is : {fitness}") 
        
        return fitness
    


    def calculate_bpd_matrix(self,population):
        """
        Efficiently calculates the Broken Pair Distance (BPD) for a population of TSP solutions
        without explicit loops, using NumPy's broadcasting features.
        
        Args:
        - population (numpy.ndarray): A 2D array where each row is a TSP solution (a permutation of cities).
        
        Returns:
        - bpd_matrix (numpy.ndarray): A matrix of BPD values between each pair of solutions in the population.
        """
        
        # Number of solutions (rows) and cities (columns) in the population
        num_solutions, num_cities = population.shape
        
        # Create edge pairs for each solution using roll (shift)
        edges = np.column_stack((population, np.roll(population, -1, axis=1)))  # (city, next_city) pairs
        
        # Broadcast edges of all pairs of solutions to compare them
        # Using broadcasting to compare all pairs of solutions in a vectorized way
        edges_i = edges[:, None, :]  # Shape: (num_solutions, 1, num_cities)
        edges_j = edges[None, :, :]  # Shape: (1, num_solutions, num_cities)
        
        # Compare the edges between all pairs: where edges are different, we get a `True` (1), else `False` (0)
        differences = (edges_i != edges_j).astype(int)
        
        # Sum the differences to get the BPD for all pairs
        bpd_matrix = np.sum(differences, axis=2)  # Sum over the edge dimension (cities)
        
        return bpd_matrix

    def average_bpd(self,population):
        """
        Calculate the average Broken Pair Distance (BPD) for each solution in the population.
        
        Args:
        - population (numpy.ndarray): 2D array where each row is a solution (TSP route).
        
        Returns:
        - avg_bpd (numpy.ndarray): Array of average BPD for each solution in the population.
        """
        bpd_matrix = self.calculate_bpd_matrix(population)
        avg_bpd = np.mean(bpd_matrix, axis=1)  # Calculate average BPD for each solution
        return avg_bpd

    def fitness_sharing_individual_np(self, population, survivors, population_fitness, sigma, alpha):
        '''
        - Vectorized fitness sharing for TSP using `calculate_hamming_distance_individual`.
        - Computes the fitness for each individual based on its pairwise Hamming distance to the survivors.
        '''
        # Number of individuals in the population
        num_individuals = len(population)
        
        # Initialize the fitness sharing multipliers as 1
        fitness_sharing = np.ones(num_individuals)
        
        # For each survivor, apply fitness sharing to all individuals
        for survivor_idx in survivors:
            if survivor_idx == -1:
                break
            
            # Get the survivor
            survivor = population[survivor_idx]
            
            # Compute pairwise Hamming distances for each individual to the current survivor
            survivor_distances = np.array([self.calculate_hamming_distance_individual(ind, survivor) for ind in population])
            #print(f"Survivor distances: {survivor_distances}")
            
            
            
            
            # Apply the fitness sharing: if distance <= sigma, apply the sharing term (1 + alpha), else 1
            sharing_term = np.where(survivor_distances <= sigma, (1-((survivor_distances)/sigma)**alpha), 1)
            # Handle identical individuals (distance = 0) explicitly by applying the penalty
            #sharing_term[survivor_distances == 0] = 1 - (1 / sigma) ** alpha  # Apply custom penalty for identical individuals
            sharing_term[survivor_distances == 0] = 0.00000000000000000001  # Apply custom penalty for identical individuals
            #print(f"Sharing term: {sharing_term}")
            
            # Multiply the fitness sharing terms with the current fitness sharing values
            fitness_sharing *= 1/sharing_term
        
        # Compute the new fitness values by applying the sharing effect
        #print(f"Fitness sharing: {fitness_sharing}")
        #print(f"Population fitness: {population_fitness}")
        fitness_new = population_fitness * fitness_sharing
        #print(f"Fitness new: {fitness_new}")
        
        return fitness_new

    def fitness_sharing_individual(self,population,survivors,population_fitness,sigma,alpha):
        fitness_population = population_fitness
        fitness_new = np.zeros_like(fitness_population)
        #print(f"Fitness_new: {fitness_new}")
        for idx,individual in enumerate(population):
            oneplusbeta = 0
            for ids,survivor in enumerate(survivors):
                    if survivor == -1:
                        break
                    hamming_distance = self.calculate_hamming_distance_individual(individual,population[survivor])
                    #print(f"Hamming distance: {hamming_distance}")
                    if hamming_distance <= sigma:
                        oneplusbeta += 100000
                    old_fitness = fitness_population[idx]
                    fitness_new[idx] = old_fitness * oneplusbeta
                    print(f"Fitness new: {fitness_new[idx]} vs {fitness_population[idx]}")   
                
                    #print(f"Fitness is the same : {fitness_new[idx]} vs {fitness_population[idx]}")
        return fitness_new
    
    def fitness_function_calculation(self,population, weight_distance, weight_bdp, distance_matrix):
        '''
        - Calculate the fitness of the population based on the distance and the broken pair distance.
        
        Args:
        - population (numpy.ndarray): 2D array where each row is a TSP solution (a permutation of cities).
        - weight_distance (float): Weight factor for distance contribution to fitness.
        - weight_bdp (float): Weight factor for BPD contribution to fitness.
        - distance_matrix (numpy.ndarray): Precomputed distance matrix for the cities.
        
        Returns:
        - fitness (numpy.ndarray): Array of fitness values for each solution in the population.
        '''
        # Step 1: Calculate the distance fitness
        distance_fitness = self.calculate_distance_population(population)
        
        # Step 2: Calculate the BPD fitness (average BPD for each solution)
        #avg_bpd = self.average_bpd(population)
        avg_bpd = np.zeros(population.shape[0])
        
        # Step 3: Calculate the fitness using both distance and BPD
        #fitness =  ( 1 / (weight_distance * distance_fitness) + weight_bdp * avg_bpd)
        fitness = distance_fitness

        
        
        return fitness, distance_fitness, avg_bpd

    def calculate_hamming_distance_individual(self,individual1,individual2):
        '''
        - Calculate the Hamming distance between two individuals
        '''
        #print(f"Individual 1: {individual1}")
        #print(f"Individual 2: {individual2}")
        # Number of cities (positions) in each solution
        num_cities = len(individual1)
        
        # Compute pairwise Hamming distances
        # diff_matrix[i, j] -> True if city `i` in solution `individual1` differs from city `i` in solution `individual2`, else False
        diff_matrix = individual1 != individual2
        
        # Sum the differences across city positions
        hamming_distance = np.sum(diff_matrix)/num_cities
        #print(f"Hamming distance: {hamming_distance}")
        
        
        
        return hamming_distance

    def calculate_hamming_distance_population(self,population):
        # Number of cities (positions) in each solution
        num_cities = population.shape[1]
        
        # Compute pairwise Hamming distances
        # diff_matrix[i, j, k] -> True if city `k` in solution `i` differs from city `k` in solution `j`, else False
        diff_matrix = population[:, None, :] != population[None, :, :]
        
        # Sum the differences across city positions
        hamming_distances = np.sum(diff_matrix, axis=2)
        
        # Normalize the Hamming distance: divide each distance by the number of cities
        normalized_hamming_distances = hamming_distances / num_cities

        #print(f"Normalized Hamming distances: {normalized_hamming_distances}")
        #print(f"Normalized Hamming distances shape: {normalized_hamming_distances.shape}")

        # Calculate the mean of the normalized Hamming distances for each solution
        row_means = np.mean(normalized_hamming_distances, axis=1)
        #print(f"Row means: {row_means}")
        #print(f"Row means shape: {row_means.shape}")
        
        # Finally, compute the average of these row means
        avg_diversity = np.mean(row_means)
        #print(f"Average diversity: {avg_diversity}")
        return avg_diversity, hamming_distances


    
    # --------- Fitness Sharing with Commone edges:
    def fitness_sharing_individual_np_edges(self, population, survivors, population_fitness, sigma, alpha):
        '''
        - Vectorized fitness sharing for TSP using `calculate_hamming_distance_individual`.
        - Computes the fitness for each individual based on its pairwise Hamming distance to the survivors.
        '''
        # Number of individuals in the population
        num_individuals = len(population)
        
        # Initialize the fitness sharing multipliers as 1
        fitness_sharing = np.ones(num_individuals)
        
        # For each survivor, apply fitness sharing to all individuals
        for survivor_idx in survivors:
            if survivor_idx == -1:
                break
            
            # Get the survivor
            survivor = population[survivor_idx]
            
            # Compute pairwise Hamming distances for each individual to the current survivor
            #survivor_distances = np.array([self.calculate_hamming_distance_individual(ind, survivor) for ind in population])
            survivor_distances = np.array([self.calculate_common_edges(ind1=ind, ind2=survivor) for ind in population])
            print(f"Survivor distances: {survivor_distances}")
            
            
            
            
            # Apply the fitness sharing: if distance <= sigma, apply the sharing term (1 + alpha), else 1
            sharing_term = np.where(survivor_distances <= sigma, (1-((survivor_distances)/sigma)**alpha), 1)
            # Handle identical individuals (distance = 0) explicitly by applying the penalty
            #sharing_term[survivor_distances == 0] = 1 - (1 / sigma) ** alpha  # Apply custom penalty for identical individuals
            sharing_term[survivor_distances == 0] = 0.00000000000000000001  # Apply custom penalty for identical individuals
            print(f"Sharing term: {sharing_term}")
            
            # Multiply the fitness sharing terms with the current fitness sharing values
            fitness_sharing *= 1/sharing_term
        
        # Compute the new fitness values by applying the sharing effect
        #print(f"Fitness sharing: {fitness_sharing}")
        print(f"Population fitness: {population_fitness}")
        fitness_new = population_fitness * fitness_sharing
        print(f"Fitness new: {fitness_new}")
        
        return fitness_new


    def calculate_edges_population(self,population):
        
        for individual in population:
            # Check if the edges for the individual have already been calculated and stored
            
            individual_key = tuple(individual)  # Converting the numpy array to a tuple

            if individual_key not in self.edges_dict:
                # Calculate edges using the build_edges method
                edges = build_edges(order=individual, length=len(individual))  # Call build_edges on the order
                self.edges_dict[individual_key] = edges  # Store the calculated edges in the dictionary
        #print(f"Edges dict: {self.edges_dict}")

    def calculate_common_edges(self,ind1,ind2):
        '''
        - Calculate the common edges between two individuals
        '''

        #print(f"Individual 1: {ind1}")
        #print(f"Individual 2: {ind2}")
        # Convert the individuals (arrays) to tuples so they can be used as dictionary keys
        ind1_key = tuple(ind1)
        ind2_key = tuple(ind2)

        # Get the edges for the two individuals
        edges1 = self.edges_dict[ind1_key]
        edges2 = self.edges_dict[ind2_key]

        # Calculate the common edges using set intersection
        common_edges = set(edges1).intersection(edges2)

        # Normalize by the number of edges in each individual
        normalization_factor = min(len(edges1), len(edges2))

        # Calculate the normalized score (ratio of common edges to the total number of edges for the least-sized individual)
        normalized_common_edges = len(common_edges) / normalization_factor if normalization_factor > 0 else 0

        #print(f"Common edges: {len(common_edges)} with common edges: {common_edges}")   
        #print(f"Normalized Common edges: {normalized_common_edges}")
        #print(f"Return: {1-normalized_common_edges}")

        return 1-normalized_common_edges
            


            




    def caclulate_numberRepeatedSolution(self,population):
        '''
        - Calculate the number of repeated solutions in the population
        '''
        self.num_unique_solutions = len(np.unique(population,axis=0))
        self.num_repeated_solutions = len(population) - self.num_unique_solutions
        


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 7) Plotting------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        

    def plot_fitness_dynamic(self):
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.best_objective_list))),
            y=self.best_objective_list,
            mode='lines+markers',
            name='Best Objective',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))
        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.second_best_objective_list))),
            y=self.second_best_objective_list,
            mode='lines+markers',
            name='Second Best Objective',
            line=dict(color='Brown'),
            marker=dict(symbol='circle')
        ))

        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.mean_objective_list))),
            y=self.mean_objective_list,
            mode='lines+markers',
            name='Mean Objective',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        # Get the last iteration's best and mean objective values
        last_best_objective = self.best_objective_list[-1]
        last_mean_objective = self.mean_objective_list[-1]

        # Add text annotation for the last iteration's objective values
        fig_obj.add_annotation(
            x=len(self.best_objective_list) - 1,
            y=last_best_objective,
            text=f'Best: {last_best_objective}<br>Mean: {last_mean_objective}',
            showarrow=True,
            arrowhead=1,
            ax=-10,
            ay=-40,
            bgcolor='white',
            bordercolor='black'
        )

        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Objective Distance over Iterations with mutation rate {self.mutation_rate} % and sigma: {self.sigma} and alpha: {self.alpha}',
            xaxis_title='Iterations',
            yaxis_title='Objective Distance',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='log',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the first plot
        fig_obj.show()

        # Create the second plot for Best and Mean Fitness values
        fig_fitness = go.Figure()

        # Add the best fitness trace
        fig_fitness.add_trace(go.Scatter(
            x=list(range(len(self.best_fitness_list))),
            y=self.best_fitness_list,
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean fitness trace
        fig_fitness.add_trace(go.Scatter(
            x=list(range(len(self.mean_fitness_list))),
            y=self.mean_fitness_list,
            mode='lines+markers',
            name='Mean Fitness',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        # Get the last iteration's best and mean fitness values
        last_best_fitness = self.best_fitness_list[-1]
        last_mean_fitness = self.mean_fitness_list[-1]

        # Add text annotation for the last iteration's fitness values
        fig_fitness.add_annotation(
            x=len(self.best_fitness_list) - 1,
            y=last_best_fitness,
            text=f'Best: {last_best_fitness}<br>Mean: {last_mean_fitness}',
            showarrow=True,
            arrowhead=1,
            ax=-10,
            ay=-40,
            bgcolor='white',
            bordercolor='black'
        )

        # Set the title and axis labels for the fitness plot
        fig_fitness.update_layout(
            title=f'Fitness over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Fitness',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='log',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the second plot
        fig_fitness.show()

        



        #Plotting unique solutions
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.num_unique_solutions_list))),
            y=self.num_unique_solutions_list,
            mode='lines+markers',
            name='Num Unique Solutions',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.num_repeated_solutions_list ))),
            y=self.num_repeated_solutions_list,
            mode='lines+markers',
            name=' Num Repeated Solutions',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

       
        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Number of unique solutions and repeated solution over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Number Unique and Repeated solutions',
            legend=dict(x=0, y=1),
            hovermode='x'
            
        )

        # Show the first plot
        fig_obj.show()


        #hamming distance:
         #Plotting unique solutions
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.hamming_distance_list))),
            y=self.hamming_distance_list,
            mode='lines+markers',
            name='Hamming distance',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        #Make me more traces for each self.hammind_distance_list
        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_crossover_list))),
            y=self.hamming_distance_crossover_list,
            mode='lines+markers',
            name='Hamming distance crossover',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_mutation1_list))),
            y=self.hamming_distance_mutation1_list,
            mode='lines+markers',
            name='Hamming distance mutation1',
            line=dict(color='green'),
            marker=dict(symbol='x')
        ))

        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_mutation2_list))),
            y=self.hamming_distance_mutation2_list,
            mode='lines+markers',
            name='Hamming distance mutation2',
            line=dict(color='red'),
            marker=dict(symbol='x')
        ))

        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_local_search_list))),
            y=self.hamming_distance_local_search_list,
            mode='lines+markers',
            name='Hamming distance local search',
            line=dict(color='purple'),
            marker=dict(symbol='x')
        ))

    

        fig_obj.add_trace(go.Scatter
        (   
            x=list(range(len(self.hamming_distance_crossoverOld_list))),
            y=self.hamming_distance_crossoverOld_list,
            mode='lines+markers',
            name='Hamming distance crossoverOld',
            line=dict(color='pink'),
            marker=dict(symbol='x')
        ))  

       
       

       
        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Hamming distance over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Hamming distance',
            legend=dict(x=0, y=1),
            hovermode='x'
            
        )

        # Show the first plot
        fig_obj.show()


    
    def plot_timing_info(self):
        """
        Plot timing information for each stage of the genetic algorithm over iterations.
        """
        # Create a figure for the timing information
        fig_time = go.Figure()

        # Add a trace for each timing component
        timing_labels = [
            "Initialization", "Selection", "Crossover", "Mutation",
            "Elimination", "Mutation Population", "Local Search", "Total Iteration"
        ]
        timing_lists = [
            self.time_initialization_list, self.time_selection_list, self.time_crossover_list,
            self.time_mutation_list, self.time_elimination_list, self.time_mutation_population_list,
            self.time_local_search_list, self.time_iteration_list
        ]

        colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "black"]  # Assign colors to each process

        # Add each timing list to the plot
        for label, time_list, color in zip(timing_labels, timing_lists, colors):
            fig_time.add_trace(go.Scatter(
                x=list(range(len(time_list))),
                y=time_list,
                mode='lines+markers',
                name=label,
                line=dict(color=color),
                marker=dict(symbol='circle')
            ))

        # Add annotations for the last iteration's timing values
        for i, (label, time_list) in enumerate(zip(timing_labels, timing_lists)):
            if time_list:  # Ensure the list is not empty
                fig_time.add_annotation(
                    x=len(time_list) - 1,
                    y=time_list[-1],
                    text=f'{label}: {time_list[-1]:.2f}s',
                    showarrow=True,
                    arrowhead=2,
                    ax=-20 * (i + 1),  # Offset annotations for readability
                    ay=-40,
                    bgcolor='white',
                    bordercolor='black'
                )

        # Set the title and axis labels for the timing plot
        fig_time.update_layout(
            title='Timing Information Over Iterations',
            xaxis_title='Iterations',
            yaxis_title='Time (seconds)',
            legend=dict(x=1, y=1),
            hovermode='x',
        )

        # Show the timing plot
        fig_time.show()
        
def calculate_total_distance_individual(route, distance_matrix):
    '''
    Calculate the total distance of a given route (individual)
    '''
    # Get the indices for the current and next cities
    current_indices = route[:-1]
    next_indices = route[1:]

    # Calculate the distances between consecutive cities
    distances = distance_matrix[current_indices, next_indices]

    # Sum the distances for the route
    total_distance = np.sum(distances)

    # Add the distance from the last city back to the first city
    total_distance += distance_matrix[route[-1], route[0]]

    return total_distance



# Optimized Local Search Function
@njit()
def perform_local_search_population_jit(population,distance_matrix,max_iterations=10, k_neighbors=10, min_improvement_threshold=100):
    '''
    Optimized local search for the population: applies 2-opt to the top individuals
    '''
    
    # Step 3: Apply 2-opt to the selected top individuals
    num_localsearch = len(population)
    #print(f"\n Number of individuals to apply local search: {num_localsearch}")
    for i in range(num_localsearch):
        #print(f"\n -------------------- Individual: {i}----------------")
        #population[i] = two_opt_no_loops_opt_out_jit(population[i], distance_matrix, max_iterations, k_neighbors, min_improvement_threshold)
        population[i]  = local_search_operator_2_opt_jit_solution(distanceMatrix=distance_matrix,order=population[i])    
        #population[i] = local_search_operator_3_opt_jit_solution(distanceMatrix=distance_matrix,order=population[i])
        #population[i]  = local_search_operator_2_opt_inline_jit(distanceMatrix=distance_matrix,order=population[i])  
        #population[i] = local_search_operator_larger_changes_v2(distanceMatrix=distance_matrix,order=population[i])
        #population[i] = simulated_annealing_tsp(distance_matrix=distance_matrix,initial_solution=population[i],max_iterations=1000)
        #population[i] = local_search_2opt_simulated_annealing(distanceMatrix=distance_matrix,order=population[i],max_iters=2)
    return population


@jit(nopython=True)
def calculate_total_distance_individual_jit(route, distance_matrix):
    '''
    Calculate the total distance of a given route
    '''
    n = len(route)
    total_distance = 0
    for i in range(n - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Return to start
    return total_distance

@njit(nopython=True)
def calculate_distance_population_jit(population, distance_matrix):
    '''
    - Calculate the fitness of the population (sum of distances in the route)
    '''
    num_individuals = population.shape[0]
    fitness = np.zeros(num_individuals)

    # Loop over each individual in the population
    for i in range(num_individuals):
        route = population[i]  # Get the individual route
        route_length = len(route)
        
        # Calculate the total distance for the current route
        total_distance = 0
        for j in range(route_length - 1):
            total_distance += distance_matrix[route[j], route[j + 1]]
        
        # Add the distance from the last city back to the first city
        total_distance += distance_matrix[route[-1], route[0]]
        
        fitness[i] = total_distance  # Store the fitness (total distance) for the individual

    return fitness

@jit(nopython=True)
def two_opt_no_loops_opt_out_jit(route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
    best_route = np.copy(route)
    initial_fitness = calculate_total_distance_individual_jit(best_route, distance_matrix)
    
    # Pre-allocate numpy array for routes (assuming max 100 routes)
    max_routes = 200  # Set this to the maximum number of routes you expect
    routes = np.empty((max_routes, len(route)), dtype=np.int32)  # Assuming integer routes
    route_index = 0  # Keep track of the index
    
    n = len(route)
    i_indices, j_indices = np.triu_indices(n, k=2)
    i_next = (i_indices + 1) % n
    j_next = (j_indices + 1) % n
    
    old_distances = np.zeros(i_indices.shape[0])
    new_distances = np.zeros(i_indices.shape[0])
    
    for idx in range(len(i_indices)):
        i = i_indices[idx]
        j = j_indices[idx]
        i_next_idx = i_next[idx]
        j_next_idx = j_next[idx]

        old_distances[idx] = (
            distance_matrix[best_route[i], best_route[i_next_idx]] + 
            distance_matrix[best_route[j], best_route[j_next_idx]]
        )
        new_distances[idx] = (
            distance_matrix[best_route[i], best_route[j]] + 
            distance_matrix[best_route[i_next_idx], best_route[j_next_idx]]
        )
    
    delta_distances = new_distances - old_distances
    
    improvement = True
    iteration = 0
    while improvement and iteration < max_iterations:
        improvement = False
        top_k_indices = np.argsort(delta_distances)[:k_neighbors]
        
        
        best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
        i, j = i_indices[best_swap_index], j_indices[best_swap_index]
        best_route[i + 1 : j + 1] = best_route[i + 1 : j + 1][::-1]
        
        # Store in pre-allocated numpy array
        routes[route_index] = np.copy(best_route)
        route_index += 1
        
        
        
        for idx in range(len(i_indices)):
            i = i_indices[idx]
            j = j_indices[idx]
            i_next_idx = i_next[idx]
            j_next_idx = j_next[idx]
            
            old_distances[idx] = (
                distance_matrix[best_route[i], best_route[i_next_idx]] + 
                distance_matrix[best_route[j], best_route[j_next_idx]]
            )
            new_distances[idx] = (
                distance_matrix[best_route[i], best_route[j]] + 
                distance_matrix[best_route[i_next_idx], best_route[j_next_idx]]
            )
        
        delta_distances = new_distances - old_distances
        
        iteration += 1
    
    # Now process the routes stored in the numpy array
    routes = routes[:route_index]  # Truncate unused rows
    if len(routes) > 0:
        #final_fitness = np.array([calculate_distance_population_jit(population=routes,distance_matrix=distance_matrix) for r in routes])
        final_fitness = calculate_distance_population_jit(population=routes, distance_matrix=distance_matrix)
        best_sol = routes[np.argmin(final_fitness)]
        best_fit = final_fitness[np.argmin(final_fitness)]
        
        if initial_fitness > best_fit:
            best_route = best_sol
        else:
            best_route = route
    else:
        best_route = route
    
    return best_route


        
# Fitness function (assuming it calculates the total fitness of the solution)
@jit(nopython=True)
def fitness(distanceMatrix: np.ndarray, order: np.ndarray) -> float:
    total_distance = 0.0
    for i in range(len(order) - 1):
        total_distance += distanceMatrix[order[i], order[i + 1]]
    total_distance += distanceMatrix[order[-1], order[0]]  # Return to start
    return total_distance


@jit(nopython=True)
def fitness_cumulative(distanceMatrix: np.ndarray, order: np.ndarray) -> float:
    total_distance = 0.0
    cum_distance = np.zeros(len(order)+1)
    diff_distance = np.zeros(len(order))
    for i in range(len(order) - 1):
        increment = distanceMatrix[order[i], order[i + 1]]
        total_distance += increment
        cum_distance[i + 1] = cum_distance[i] + increment
        diff_distance[i] = increment

    last_increment = distanceMatrix[order[-1], order[0]]
    total_distance += last_increment  # Return to start
    cum_distance[i+2] = cum_distance[i+1] + last_increment
    diff_distance[-1] = last_increment
    #print(f"Total distance: {total_distance} vs calculated: {calculate_total_distance_individual(order, distanceMatrix)}")
    #print(f"Cumulative distance: {cum_distance}")
    #print(f"Difference distance: {diff_distance}")
    return cum_distance,diff_distance

# Function to calculate cumulative distances (these are precomputed for efficiency)
@jit(nopython=True)
def build_cumulatives(distanceMatrix: np.ndarray, order: np.ndarray, length: int) -> tuple:
    # Cumulative distance from 0 to each node (excluding the last one)
    cum_from_0_to_first = np.zeros((length))
    cum_from_second_to_end = np.zeros((length))
    cum_from_second_to_end[length - 1] = partial_fitness_one_value(distanceMatrix, 
                                                                   frm=order[-1], 
                                                                   to=order[0])
    for i in range(1, length - 1):
        cum_from_0_to_first[i] = cum_from_0_to_first[i - 1] \
            + partial_fitness_one_value(distanceMatrix, frm=order[i-1], to=order[i])
        cum_from_second_to_end[length - 1 - i] = cum_from_second_to_end[length - i] \
            + partial_fitness_one_value(distanceMatrix, frm=order[length -1 - i], to=order[length - i])
    return cum_from_0_to_first, cum_from_second_to_end

# Partial fitness calculation for one value (this computes the fitness between two nodes)
@jit(nopython=True)
def partial_fitness_one_value(distanceMatrix: np.ndarray, frm: int, to: int) -> float:
    return distanceMatrix[frm, to]

# In-place 2-opt local search operator
@jit(nopython=True)
def local_search_operator_2_opt_jit_solution(distanceMatrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)
    best_combination = (0, 0)
    flag_improved = False

    # Build cumulative arrays
    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)

    # Try swapping edges
    for first in range(1, length - 3):
        
        fit_first_part = cum_from_0_to_first[first-1]
        if fit_first_part > best_fitness:
            break
        fit_middle_part = 0.0
        
        for second in range(first + 2, length):
            # Update middle part progressively
            fit_middle_part += partial_fitness_one_value(distanceMatrix, 
                                                        frm=order[second-1], 
                                                        to=order[second-2])
            
            fit_last_part = cum_from_second_to_end[second]

            # Calculate fitness for the new possible swap
            bridge_first = partial_fitness_one_value(distanceMatrix, 
                                                     frm=order[first-1], 
                                                     to=order[second-1])
            bridge_second = partial_fitness_one_value(distanceMatrix, 
                                                      frm=order[first], 
                                                      to=order[second])
            temp = fit_first_part + fit_middle_part
            
            if temp > best_fitness:
                continue
            new_fitness =  temp + fit_last_part + bridge_first + bridge_second
            
            

            #print(f"New fitness: {new_fitness} vs New fitness calc: {new_fitness_calc}. diff: {new_fitness - new_fitness_calc}")
            #print(f"First: {first}, Second: {second}, First Bridge: {bridge_first}, Second Bridge: {bridge_second}")
            #print(f"Forward cumulative at {first}: {fit_first_part}, Middle part: {fit_middle_part}, Last part: {fit_last_part}")
            
            if new_fitness < best_fitness:
                flag_improved = True
                #print(f"New fitness: {new_fitness} vs Best fitness: {best_fitness}")
                best_combination = (first, second)
                best_fitness = new_fitness
                #best_order = new_order.copy()
    #print(f"LS: Best fitness: {best_fitness} vs Original fitness: {fitness(distanceMatrix, order)} with combination: {best_combination}")
    best_first, best_second = best_combination
    if flag_improved is False:  # No improvement found
        #print(f"LS: No improvement --> Performing 3-opt")
        #order = local_search_operator_larger_changes_v2(distanceMatrix=distanceMatrix, order=order)
        #order = local_search_2opt_simulated_annealing(distanceMatrix=distanceMatrix,order=order,max_iters=20)
    
        return order  # Return the original order if no better solution is found
    
    # Perform the 2-opt swap in-place
    #print(f"LS: Improvement found. Best first: {best_combination} --> Performing 2-opt swap")
    #print(f"LS: Order before swap: {order}")
    order[best_first:best_second] = order[best_first:best_second][::-1]
    #print(f"LS: Order after swap: {order}")
    #print(f"LS: Checking fitness: {best_fitness} vs calculated: {fitness(distanceMatrix, order)}")
    
    return order

@jit(nopython=True)
def local_search_operator_2_opt_inline_jit(distanceMatrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Local search operator using 2-opt, where changes are applied immediately when
    a better solution is found. The cumulative arrays are rebuilt after each improvement.
    """
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)

    # Build cumulative arrays initially
    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)

    for first in range(1, length - 3):
        fit_first_part = cum_from_0_to_first[first - 1]

        if fit_first_part > best_fitness:
            break  # Early termination if partial fitness exceeds current best

        fit_middle_part = 0.0

        for second in range(first + 2, length):  # Ensure valid range for 2-opt
            # Update the middle part progressively
            fit_middle_part += partial_fitness_one_value(
                distanceMatrix, frm=order[second - 1], to=order[second - 2]
            )

            fit_last_part = cum_from_second_to_end[second]

            # Calculate potential new fitness
            bridge_first = partial_fitness_one_value(
                distanceMatrix, frm=order[first - 1], to=order[second - 1]
            )
            bridge_second = partial_fitness_one_value(
                distanceMatrix, frm=order[first], to=order[second]
            )
            temp = fit_first_part + fit_middle_part

            if temp > best_fitness:
                continue

            new_fitness = temp + fit_last_part + bridge_first + bridge_second

            if new_fitness < best_fitness:
                # Update the route with a 2-opt swap
                order[first:second] = order[first:second][::-1]

                # Rebuild cumulative arrays with the new order
                cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)

                # Update the best fitness
                best_fitness = new_fitness

                # Continue to evaluate new possibilities after this change
                break  # Break to move to the next `first` index after applying this improvement

    return order

@jit(nopython=True)
def local_search_operator_larger_changes_v2(distanceMatrix: np.ndarray, order: np.ndarray, k_neighbors=10):
    best_route = np.copy(order)
    n = len(order)
    best_fitness = fitness(distanceMatrix, best_route)
    #print(f"Initial fitness: {best_fitness}")
    
    # Build cumulative arrays for fast fitness calculation
    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, best_route, n)
    cum_distance, diff_distance = fitness_cumulative(distanceMatrix=distanceMatrix, order=best_route)

    #now select the indices with largest diff_distances 
    # Get indices of the k largest elements
    largest_k_indices = np.argpartition(-diff_distance, k_neighbors)[:k_neighbors]

    # (Optional) Sort these indices by their actual values in descending order
    largest_k_indices_sorted = largest_k_indices[np.argsort(-diff_distance[largest_k_indices])]

   

    # Results
    #print(f"Top {k_neighbors} indices with largest diff_distance: {largest_k_indices_sorted}")
    #print(f"Corresponding diff_distance values: {diff_distance[largest_k_indices_sorted]}")
    
    # Try swapping edges to try to improve the edges that are giving me the largest diff_distance
    # Attempt to improve problematic edges
    global_flag_improved = False
    for idx in largest_k_indices_sorted:
        #print(f"\n --- Processing index: {idx}")
        flag_improved = False
        first = idx
        fit_first_part = cum_from_0_to_first[first - 1]
        if fit_first_part > best_fitness:
            break  # No point in continuing

        fit_middle_part = 0.0
        # Iterate over possible improvements for the current edge
        for second in range(first + 2, n):  # Ensure valid edge range
            fit_middle_part += partial_fitness_one_value(distanceMatrix, 
                                                         frm=best_route[second - 1],
                                                         to=best_route[second-2])
            
            fit_last_part = cum_from_second_to_end[second]
            bridge_first = partial_fitness_one_value(
                distanceMatrix, frm=best_route[first - 1], to=best_route[second - 1]
            )
            bridge_second = partial_fitness_one_value(
                distanceMatrix, frm=best_route[first], to=best_route[second]
            )

            # Calculate potential new fitness
            temp = fit_first_part + fit_middle_part
            if temp > best_fitness:
                continue
                
            new_fitness = temp + fit_last_part + bridge_first + bridge_second
            #print(f"New fitness: {new_fitness} vs Best fitness: {best_fitness}")

            
            #print(f"New fitness: {new_fitness} vs New fitness calc: {new_fitness_calc}. diff: {new_fitness - new_fitness_calc}")
           
            if new_fitness < best_fitness:
                flag_improved = True
                global_flag_improved = True
                #print(f"Improved fitness to {new_fitness} from {best_fitness} by modifying edges near index {first}")
                best_fitness = new_fitness
                # Apply the improvement: reverse segment [first:second]
        if flag_improved is False:
            #print("No improvement found")
            a = 1
            best_route = np.copy(order)
       
        else:
            #print(f"Improved fitness to {best_fitness}")
            best_route[first:second] = best_route[first:second][::-1]
            cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, best_route, n)

    if global_flag_improved is False:
        best_route = np.copy(order)
    
    return best_route

       

@jit(nopython=True)
def local_search_2opt_simulated_annealing(distanceMatrix: np.ndarray, order: np.ndarray, initial_temp=1000, cooling_rate=0.995, max_iters=1000) -> np.ndarray:
    """
    Simulated annealing with 2-opt for TSP, enabling escape from local maxima.

    Parameters:
        distanceMatrix (np.ndarray): Distance matrix for the TSP.
        order (np.ndarray): Initial TSP route.
        initial_temp (float): Initial temperature for simulated annealing.
        cooling_rate (float): Cooling rate for temperature reduction.
        max_iters (int): Maximum number of iterations.

    Returns:
        np.ndarray: Improved route after simulated annealing.
    """
    #print(f"\n Using Simmulated Annealing with 2-opt")
    best_order = order.copy()
    best_fitness = fitness(distanceMatrix, best_order)
    current_order = order.copy()    
    current_fitness = best_fitness
    temperature = initial_temp
    length = len(order)

    for _ in range(max_iters):
        improved = False

        # Build cumulative arrays initially
        cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, current_order, length)

        for first in range(1, length - 3):
            fit_first_part = cum_from_0_to_first[first - 1]

            if fit_first_part > best_fitness:
                break  # Early termination if partial fitness exceeds current best

            fit_middle_part = 0.0

            for second in range(first + 2, length):  # Ensure valid range for 2-opt
                # Update the middle part progressively
                fit_middle_part += partial_fitness_one_value(
                    distanceMatrix, frm=current_order[second - 1], to=current_order[second - 2]
                )

                fit_last_part = cum_from_second_to_end[second]

                # Calculate potential new fitness
                bridge_first = partial_fitness_one_value(
                    distanceMatrix, frm=current_order[first - 1], to=current_order[second - 1]
                )
                bridge_second = partial_fitness_one_value(
                    distanceMatrix, frm=current_order[first], to=current_order[second]
                )
                temp = fit_first_part + fit_middle_part

                if temp > current_fitness:
                    continue

                new_fitness = temp + fit_last_part + bridge_first + bridge_second

                # Simulated annealing acceptance criteria
                if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / temperature):
                    #print(f"New fitness: {new_fitness} vs Current fitness: {current_fitness} accepted")
                    # Apply the 2-opt swap
                    current_order[first:second] = current_order[first:second][::-1]

                    # Rebuild cumulative arrays with the new order
                    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, current_order, length)

                    # Update the current fitness
                    current_fitness = new_fitness

                    # Check if this is the best solution so far
                    if new_fitness < best_fitness:
                        #print(f"\n UPDATING: New fitness: {new_fitness} vs Best fitness: {best_fitness}")
                        print(f"Improved via Simulated Annealing")
                        best_order = current_order.copy()
                        best_fitness = new_fitness

                    improved = True
                    break  # Break to move to the next `first` index after applying this improvement

        # Cool down
        temperature *= cooling_rate

        if not improved:
            break  # Stop if no improvement was made in the iteration

        if temperature < 1e-5:
            break  # Stop if the temperature is very low

    return best_order


@jit(nopython=True)
def simulated_annealing_tsp(distance_matrix, initial_solution=None, initial_temp=1000, cooling_rate=0.995, max_iterations=500):
    """
    Simulated Annealing for the Traveling Salesman Problem using existing functions.
    
    Parameters:
        distance_matrix (np.ndarray): Matrix of distances between cities.
        initial_solution (np.ndarray): Initial route (optional).
        initial_temp (float): Initial temperature for annealing.
        cooling_rate (float): Cooling rate for temperature reduction.
        max_iterations (int): Maximum number of iterations.
    
    Returns:
        best_solution (np.ndarray): Best route found.
        best_cost (float): Cost of the best route.
    """
    num_cities = len(distance_matrix)
    
    # Initialize the solution
    if initial_solution is None:
        initial_solution = np.arange(num_cities, dtype=np.int64)
        np.random.shuffle(initial_solution)
    else:
        initial_solution = initial_solution.astype(np.int64)

    current_solution = initial_solution[:]
    current_cost = fitness(distanceMatrix=distance_matrix, order=current_solution)
    best_solution = current_solution[:]  # Use slicing instead of np.copy
    best_cost = current_cost

    temperature = initial_temp

    for iteration in range(max_iterations):
        # Generate a neighboring solution
        neighbor_solution = mutation_singlepoint(current_solution)
        neighbor_cost = fitness(distanceMatrix=distance_matrix, order=neighbor_solution)

        # Calculate acceptance probability
        delta = neighbor_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            # Accept the new solution
            current_solution[:] = neighbor_solution[:]
            current_cost = neighbor_cost

            # Update the best solution
            if neighbor_cost < best_cost:
                best_solution[:] = neighbor_solution[:]
                best_cost = neighbor_cost

        # Cool down
        temperature *= cooling_rate

        if temperature < 1e-5:
            break

    return best_solution

    
  
#@jit(nopython=True)
def local_search_operator_3_opt_jit_solution(distanceMatrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Local search operator, which makes use of 3-opt. Swap three edges within a cycle."""
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)
    best_combination = (0, 0, 0)  # For 3-opt, we need three indices
    flag_improved = False

    # Build cumulative arrays
    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)

    # Try swapping edges
    for first in range(1, length - 4):  # Stop 4 places before the last to avoid overflow
        fit_first_part = cum_from_0_to_first[first - 1]
        if fit_first_part > best_fitness:
            break
        fit_middle_part = 0.0
        
        for second in range(first + 2, length - 1):  # Need second to be at least two positions ahead of first
            # Update middle part progressively
            fit_middle_part += partial_fitness_one_value(distanceMatrix, 
                                                        frm=order[second - 1], 
                                                        to=order[second - 2])
            
            for third in range(second + 2, length):  # third needs to be at least two positions after second
                fit_last_part = cum_from_second_to_end[third]

                # Calculate fitness for the new possible swap (3-opt)
                bridge_first = partial_fitness_one_value(distanceMatrix, 
                                                     frm=order[first - 1], 
                                                     to=order[second - 1])
                bridge_second = partial_fitness_one_value(distanceMatrix, 
                                                      frm=order[second], 
                                                      to=order[third - 1])
                bridge_third = partial_fitness_one_value(distanceMatrix, 
                                                      frm=order[third], 
                                                      to=order[first])
                
                temp = fit_first_part + fit_middle_part
                
                if temp > best_fitness:
                    continue
                new_fitness = temp + fit_last_part + bridge_first + bridge_second + bridge_third

                new_route = order.copy()
                new_route[first:second] = order[first:second][::-1]
                new_route[second:third] = order[second:third][::-1]
                new_route[third:] = order[third:][::-1]
                new_fitness_calc = fitness(distanceMatrix, new_route)
                print(f"New fitness: {new_fitness} vs New fitness calc: {new_fitness_calc}. diff: {new_fitness - new_fitness_calc}")

                if new_fitness_calc < best_fitness:
                    flag_improved = True
                    best_combination = (first, second, third)
                    best_fitness = new_fitness_calc

    # Perform the 3-opt swap in-place if an improvement was found
    best_first, best_second, best_third = best_combination
    #print(f"Best fitness: {best_fitness} vs Original fitness: {fitness(distanceMatrix, order)} with combination: {best_combination}")
    if not flag_improved:  # No improvement found
        return order  # Return the original order if no better solution is found

    # 3-opt swap: reverse the segments between best_first, best_second, and best_third
    # We will swap the 3 segments by reordering them
    order[best_first:best_second] = order[best_first:best_second][::-1]
    order[best_second:best_third] = order[best_second:best_third][::-1]
    order[best_third:] = order[best_third:][::-1]

    return order

@jit(nopython=True)
def local_search_operator_3_opt_jit_solution_old(distanceMatrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Local search operator, which makes use of 3-opt. Swap three edges within a cycle."""
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)

    # Try swapping three edges
    best_order = order.copy()  # Start with the initial order

    for first in range(1, length - 3):
        for second in range(first + 2, length - 1):
            for third in range(second + 2, length):

                # Remove the 3 edges (from first to second, second to third, third to first)
                # and create 7 new possible reconfigurations by reversing the segments
                segment_1 = order[:first]
                segment_2 = order[first:second]
                segment_3 = order[second:third]
                segment_4 = order[third:]

                # 7 possible reconfigurations
                new_orders = [
                    np.concatenate((segment_1, segment_2[::-1], segment_3[::-1], segment_4)),
                    np.concatenate((segment_1, segment_2[::-1], segment_4, segment_3[::-1])),
                    np.concatenate((segment_1, segment_3[::-1], segment_2[::-1], segment_4)),
                    np.concatenate((segment_1, segment_3[::-1], segment_4, segment_2[::-1])),
                    np.concatenate((segment_1, segment_4, segment_2[::-1], segment_3[::-1])),
                    np.concatenate((segment_1, segment_4, segment_3[::-1], segment_2[::-1])),
                    np.concatenate((segment_2[::-1], segment_1, segment_3[::-1], segment_4))
                ]

                # Evaluate fitness for each new reconfiguration and select the best one
                for new_order in new_orders:
                    new_fitness = fitness(distanceMatrix, new_order)
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_order = new_order

    return best_order

@jit(nopython=True)
def two_opt_no_loops_opt_out_wiki_jit(route, distance_matrix, max_iterations=10, min_improvement_threshold=100):
        """
        Perform a complete 2-opt search, considering every possible swap.
        Store the new route for each swap and apply the one with the best improvement.
        """
        #print(f"\n ------------LOCAL SEARCH------------")
        #print(f"Max number of iterations: {max_iterations}")
        
        best_route = np.copy(route)
        #initial_fitness = calculate_total_distance_individual_jit(best_route, distance_matrix)
        
        n = len(route)
        iteration = 0
    

        
       
        # Store routes and fitnesses for all possible swaps
        all_swaps = []

        for i in range(n - 1):
            for j in range(i + 2, n):  # Ensure no overlapping
                # Create a new route by performing the 2-opt swap
                new_route = np.copy(best_route)
                new_route[i + 1:j + 1] = new_route[i + 1:j + 1][::-1]
                
                # Calculate the fitness (distance) of the new route
                #new_fitness = calculate_total_distance_individual_jit(new_route, distance_matrix)
                
                # Store the swap and its fitness
                all_swaps.append((new_route))
                
            
        return all_swaps

    
@jit(nopython=True)
def hill_climbing_2_opt_with_deltas_jit(distanceMatrix: np.ndarray, order: np.ndarray, max_iterations: int = 100, k_neighbors: int = 10) -> np.ndarray:
    """Hill climbing method that iteratively improves the solution using 2-opt with delta distance calculations (optimized for numba)."""
    # Initial best solution and fitness
    best_order = order.copy()
    best_fitness = fitness(distanceMatrix, best_order)

    n = len(order)
    
    # Generate all pairs of indices (i, j) where i < j
    i_indices, j_indices = np.triu_indices(n, k=2)
    
    # Precompute the "next" indices in the route
    i_next = (i_indices + 1) % n
    j_next = (j_indices + 1) % n
    
    # Old distances: distance between consecutive cities in the route
    old_distances = np.zeros(i_indices.shape[0])
    new_distances = np.zeros(i_indices.shape[0])
    
    for idx in range(i_indices.shape[0]):
        old_distances[idx] = distanceMatrix[best_order[i_indices[idx]], best_order[i_next[idx]]] + distanceMatrix[best_order[j_indices[idx]], best_order[j_next[idx]]]
        new_distances[idx] = distanceMatrix[best_order[i_indices[idx]], best_order[j_indices[idx]]] + distanceMatrix[best_order[i_next[idx]], best_order[j_next[idx]]]

    # Calculate delta distances (new distance - old distance)
    delta_distances = new_distances - old_distances

    improvement = True
    iteration = 0

    while improvement and iteration < max_iterations:
        improvement = False
        iteration += 1

        # Sort by delta distances and get the top k swaps with the smallest delta
        # This part uses a simple loop to avoid np.argsort (not supported in numba's nopython mode)
        top_k_indices = np.zeros(k_neighbors, dtype=np.int32)
        min_deltas = np.full(k_neighbors, np.inf)
        
        # Find the top k swaps with the smallest delta
        for idx in range(i_indices.shape[0]):
            delta = delta_distances[idx]
            for k in range(k_neighbors):
                if delta < min_deltas[k]:
                    min_deltas[k] = delta
                    top_k_indices[k] = idx
                    break
        
        # Try the best (smallest delta) swap from the top k
        best_swap_index = top_k_indices[0]
        i, j = i_indices[best_swap_index], j_indices[best_swap_index]

        # Perform the 2-opt swap: reverse the segment between i and j
        best_order[i + 1: j + 1] = best_order[i + 1: j + 1][::-1]

        # Recompute the distances for the swapped pairs
        for idx in range(i_indices.shape[0]):
            old_distances[idx] = distanceMatrix[best_order[i_indices[idx]], best_order[i_next[idx]]] + distanceMatrix[best_order[j_indices[idx]], best_order[j_next[idx]]]
            new_distances[idx] = distanceMatrix[best_order[i_indices[idx]], best_order[j_indices[idx]]] + distanceMatrix[best_order[i_next[idx]], best_order[j_next[idx]]]
        
        # Recalculate delta distances after the swap
        delta_distances = new_distances - old_distances
        
        # Update fitness
        new_fitness = fitness(distanceMatrix, best_order)
        
        # If the new solution improves the fitness, we continue to improve
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            improvement = True
        
        # Optionally, you could limit the number of iterations where no improvement occurs
        if not improvement:
            print(f"Iteration {iteration}: No improvement found.")
    
    return best_order
    

@jit(nopython=True)
def mutation_singlepoint(route, mutation_rate=1):
    """
    Perform a random 2-swap mutation (exchange two cities in the route).
    
    Parameters:
        route (np.ndarray): Order of cities in the route.
        mutation_rate (float): Mutation probability. For SA, this is always 1.

    Returns:
        mutated_route (np.ndarray): Route after the mutation.
    """
    mutated_route = route[:]  # Copy the array using slicing
    num_genes = len(mutated_route)

    # Ensure valid mutation range
    num_mutations1 = np.random.randint(1, max(1, (num_genes - 1) // 2))

    # Select the first set of mutation indices
    mutation_indices1 = np.random.choice(num_genes, size=num_mutations1, replace=False)

    # Calculate available indices manually
    available_indices = []
    for i in range(num_genes):
        if i not in mutation_indices1:
            available_indices.append(i)
    available_indices = np.array(available_indices, dtype=np.int64)

    # Ensure num_mutations1 is not larger than available_indices
    num_mutations2 = num_mutations1
    mutation_indices2 = np.random.choice(available_indices, size=num_mutations2, replace=False)

    # Perform the mutation (swap)
    for i in range(num_mutations1):
        idx1 = mutation_indices1[i]
        idx2 = mutation_indices2[i]
        mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]

    return mutated_route



@jit(nopython=True)
def greedy_rearrange_subsection_jit(route, distance_matrix, start_index, end_index):
    """
    Rearrange a subsection of the route using a greedy strategy.

    Parameters:
    - route: Current route (1D array of city indices).
    - distance_matrix: Distance matrix.
    - start_index: Start index of the subsection to rearrange.
    - end_index: End index of the subsection to rearrange (inclusive).

    Returns:
    - new_route: Route with the subsection rearranged.
    """
    # Extract the subsection
    subsection = route[start_index:end_index+1]
    
    # Initialize greedy rearrangement
    rearranged = np.empty(len(subsection), dtype=np.int64)
    visited = set()
    
    # Start with the first city in the subsection
    current_city = subsection[0]
    rearranged[0] = current_city
    visited.add(current_city)
    
    for i in range(1, len(subsection)):
        # Find the nearest unvisited city
        remaining_cities = np.array([city for city in subsection if city not in visited])
        distances = np.array([distance_matrix[current_city, city] for city in remaining_cities])
        
        # Get the next city
        next_city = remaining_cities[np.argmin(distances)]
        rearranged[i] = next_city
        visited.add(next_city)
        current_city = next_city
    
    # Replace the subsection in the route
    new_route = route.copy()
    new_route[start_index:end_index+1] = rearranged

    return new_route

@jit(nopython=True)
def local_search_greedy_rearrange_jit(route, distance_matrix, max_iterations=50):
    """
    Apply local search by rearranging random subsections of the route using a greedy strategy.
    
    Parameters:
    - route: Initial route (1D array of city indices).
    - distance_matrix: Distance matrix.
    - max_iterations: Maximum number of iterations for the local search.
    
    Returns:
    - best_route: Optimized route.
    - best_fitness: Fitness of the optimized route.
    """
    best_route = route.copy()
    best_fitness = fitness(distanceMatrix=distance_matrix,order=best_route)
    #print(f" LS_greedy: max iterations: {max_iterations}")
    for _ in range(max_iterations):
        #print(f"\n Iteration: {_}")
        # Select a random subsection
        start_index = np.random.randint(0, len(route) - 1)
        end_index = np.random.randint(start_index + 1, len(route))
        
        # Rearrange the subsection
        new_route = greedy_rearrange_subsection_jit(best_route, distance_matrix, start_index, end_index)
        new_fitness = fitness(distanceMatrix=distance_matrix,order=new_route)
        #print(f"\        + Fitness before: {best_fitness} - Fitness after: {new_fitness}")
        
        # Update the best solution if improvement is found
        if new_fitness < best_fitness:
            #print(f"\        + Fitness before: {best_fitness} - Fitness after: {new_fitness}")
            best_route = new_route
            best_fitness = new_fitness
    
    return best_route

        


# Calculate edges
#@jit(nopython=True)
def build_edges(order,length):
    edges = [None] * length
    prev = order[0]
    for i in range(length):
        next = order[(i + 1) % length]
        edges[i] = (prev, next)
        prev = next
    edges = set(edges)
    return edges
    



    









#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
																			# Class Evol_Algorithm_level_2
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time 


class GA_K_L2:

    def __init__(self,clusters_solutions_matrix,cities_model,seed=None ,mutation_prob = 0.00,elitism_percentage = 20,local_search = False):
        #model_key_parameters
        #self.cities = cities 
        self.k_tournament_k = 3
        self.population_size = 0.0
        self.mutation_rate = mutation_prob
        self.elistism = 25                        #Elitism rate as a percentage


        self.mean_objective = 0.0
        self.best_objective = 0.0
        self.mean_fitness_list = []
        self.best_fitness_list = [] 

        self.clusters_solution_matrix= clusters_solutions_matrix

        self.num_clusters = len(clusters_solutions_matrix)
        self.possible_entryAndExit_points_list = []
       

      
        self.cities_model = cities_model

        # Fitness Sharing
        self.sigma = 0.9
        self.alpha = 0.1

        #Unique Solutions
        self.num_unique_solutions_list = []
        self.num_repeated_solutions_list = []

        #Diversity: Hamming Distance
        self.hamming_distance = 0
        self.hamming_distance_list = []
        self.hamming_distance_crossover_list = []
        self.hamming_distance_crossoverOld_list = []
        self.hamming_distance_mutation1_list = []
        self.hamming_distance_mutation2_list = []
        self.hamming_distance_elimination_list = []
        self.hamming_distance_local_search_list = []
        
        #Local Search
        self.local_search = local_search

        #Time
        self.time_iteration_lists = []
        self.time_initialization_list = []
        self.time_selection_list = []
        self.time_crossover_list = []
        self.time_mutation_list = []
        self.time_elimination_list = []
        self.time_mutation_population_list = []
        self.time_local_search_list = []
        self.time_iteration_list = []
    

        #Random seed
        if seed is not None:
            np.random.seed(seed)    
        
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Settings:------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def caclulate_numberRepeatedSolution(self,population):
        '''
        - Calculate the number of repeated solutions in the population
        '''
        self.num_unique_solutions = len(np.unique(population,axis=0))
        self.num_repeated_solutions = len(population) - self.num_unique_solutions

    def retieve_order_cities(self,best_solution):

        '''
        - Retrieve the order of the cities
        '''
        #print(f"\n Best solution is : {best_solution}")
        #print(f"\n Cities are : {self.cities}")
        
        self.best_solution_cities = self.cities[best_solution]

        print(f"\n Best solution cities are : {self.best_solution_cities}")

    def print_model_info(self):
        print("\n------------- GA_Level2: -------------")
        print(f"   * Model Info:")
        print(f"       - Population Size: {self.population_size}")
        print(f"       - Number of cities: {self.gen_size}")
        print(f"       - Number of clusters: {self.num_clusters}")
        print(f"       - Cluster Soluitions: {self.clusters_solution_matrix}")
        print(f"       - Distance Matrix: {self.distance_matrix}")
        print(f"   * Model Parameters:")
        print(f"       - K: {self.k_tournament_k}")
        print(f"       - Mutation rate: {self.mutation_rate}")
        print(f"       - Elitism percentage: {self.elistism} %")
        print(f"   * Running model:")
        
    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf(distance_matrix=distance_matrix,replace_value=100000)
        self.gen_size = len(distance_matrix)
        #self.population_size = 2*self.gen_size
        if self.gen_size > 200:
            self.population_size = 15
        else:
            self.population_size = 15
        
        
        self.k_tournament_k = 3
        print(f"Distance matrix is {self.distance_matrix}")
        #print(f"Gen size is {self.gen_size}")
        print(f"Population size is {self.population_size}")


    def check_inf(self,distance_matrix,replace_value=10000000):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''
        distance_matrix[distance_matrix == np.inf] = replace_value
        return distance_matrix

    def calculate_information_iteration(self):
        '''
        - Calculate the mean and best objective function value of the population
        '''
        self.mean_objective = np.mean(self.fitness)
        self.best_objective = np.min(self.fitness)
        self.mean_fitness_list.append(self.mean_objective)
        self.best_fitness_list.append(self.best_objective)  
        best_index = np.argmin(self.fitness)
        best_solution = self.population_cluster[best_index]
        self.best_solution_cities = self.population_cities[best_index]

        #Diversity: Hamming Distance
        if self.hamming_distance:
            self.hamming_distance_list.append(self.hamming_distance)

        #Unique and repeated solutions
        self.caclulate_numberRepeatedSolution(population=self.population_cities)
        self.num_unique_solutions_list.append(self.num_unique_solutions)
        self.num_repeated_solutions_list.append(self.num_repeated_solutions)
        #self.retieve_order_cities(best_solution_merged)    
        #print(f"Mean Objective --> {self.mean_objective} \n Best Objective --> {self.best_objective} \n Best Solution Clusters --> {best_solution} \n Best Solution Cities --> {self.best_solution_cities}")
        return self.mean_objective,self.best_objective,best_solution
    
    def print_best_solution(self):
        '''
        - Print the best solution
        '''
        
        print(f"\n Best solution is : {self.best_objective} \n Best solution cities are : {self.best_solution_cities}")
    
    def check_stopping_criteria(self):
        '''
        - Check the stopping criteria
        '''
        
        if round(self.best_objective) == round(self.mean_objective):
            return True
        else:
            return False

    def update_time(self,time_initalization, time_selection, time_crossover, time_mutation, time_elimination,time_mutation_population,time_local_search,time_iteration):
        '''
        - Update the time
        '''
        self.time_initialization_list.append(time_initalization)
        self.time_selection_list.append(time_selection)
        self.time_crossover_list.append(time_crossover)
        self.time_mutation_list.append(time_mutation)
        self.time_elimination_list.append(time_elimination)
        self.time_mutation_population_list.append(time_mutation_population)
        self.time_local_search_list.append(time_local_search)
        self.time_iteration_list.append(time_iteration)

        new_lists = [self.time_initialization_list,self.time_selection_list,self.time_crossover_list,self.time_mutation_list,self.time_elimination_list,self.time_mutation_population_list,self.time_local_search_list,self.time_iteration_list]
        self.time_iteration_lists.append(new_lists)

    def combine_all_lists(self,all_list1,all_list2):
        '''
        - Combine all lists
        '''
        cluster_all_list1 = all_list1[0]
        startEnd_all_list1 = all_list1[1]
        cities_all_list1 = all_list1[2]

        cluster_all_list2 = all_list2[0]
        startEnd_all_list2 = all_list2[1]
        cities_all_list2 = all_list2[2]

        combined_cluster_all_list = np.vstack((cluster_all_list1, cluster_all_list2))
        combined_startEnd_all_list = np.vstack((startEnd_all_list1, startEnd_all_list2))
        combined_cities_all_list = np.vstack((cities_all_list1, cities_all_list2))

        return [combined_cluster_all_list,combined_startEnd_all_list,combined_cities_all_list]
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Run ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_model(self):
        
        time_start = time.time()
        self.set_initialization()
        self.calculate_information_iteration()
        time_end = time.time()
        intialization_time = time_end - time_start 
        yourConvergenceTestsHere = False
        num_iterations = 0
        iterations = 0
        while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
            '''
            meanObjective = 0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            '''
            # 1) Selection
            time_start_iteration = time.time()
            iterations += 1
            print(f"\n Iteration number {iterations}")
            time_start = time.time()
            parents_all_list = self.selection_k_tournament(num_individuals=self.population_size)    
            time_end = time.time()
            time_selection = time_end - time_start

            # 2) Crossover
            time_start = time.time()
            offspring_all_list = self.crossover_singlepoint_population(parents_all_list)
            time_end = time.time()
            time_crossover = time_end - time_start
            self.calculate_add_hamming_distance(population=offspring_all_list[0],crossover_old=True)
            self.calculate_add_hamming_distance(population=offspring_all_list[0],crossover=True)

            # 3) Mutation Offspring
            time_start = time.time()
            mutated_all_list = self.mutation_singlepoint_population(offspring_all_list)
            time_end = time.time()
            time_mutation = time_end - time_start
            self.calculate_add_hamming_distance(population=mutated_all_list[0],mutation1=True)
            
            # 4) Mutation Population
            time_start = time.time()
            #self.population_all_list = self.mutation_singlepoint_population(self.population_all_list)    
            time_end = time.time()
            time_mutation_population = time_end - time_start
            self.calculate_add_hamming_distance(population=self.population_all_list[0],mutation2=True)
            #self.population_all_list= self.local_search_population(population_all_list=self.population_all_list,n_best=2,max_iterations=50)

            # 5) Local Search
            if self.local_search:
                time_start = time.time()
                
                #mutated_all_list = self.combine_all_lists(all_list1=self.population_all_list,all_list2=mutated_all_list)
                mutated_all_list = self.local_search_population(population_all_list=mutated_all_list,n_best=2,max_iterations=50)
                time_end = time.time()
                time_local_search = time_end - time_start
            else:
                time_local_search = 0
            self.calculate_add_hamming_distance(population=mutated_all_list[0],local_search=True)

            # 6) Elimination
            time_start = time.time()
            #self.eliminate_population(population_all_list=self.population_all_list, mutated_all_list=mutated_all_list)
            #self.eliminate_population_elitism(population=self.population, offsprings=offspring_mutated)
            self.eliminate_population_fs_tournament(population_all_list=self.population_all_list, mutated_all_list=mutated_all_list, 
                                                    sigma=self.sigma, alpha=self.alpha, k=self.k_tournament_k)
            time_end = time.time()
            time_elimination = time_end - time_start
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
            yourConvergenceTestsHere = False
            time_end_iteration = time.time()
            diff_time_iteration = time_end_iteration - time_start_iteration
            self.update_time(time_initalization=intialization_time,time_selection=time_selection,time_crossover=time_crossover,time_mutation=time_mutation,time_elimination=time_elimination,time_mutation_population=time_mutation_population,time_local_search=time_local_search,time_iteration=diff_time_iteration)
        self.print_best_solution()
        #self.plot_fitness_dynamic()
        #self.plot_timing_info()
        
        return 0
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 1) Initalization ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_initialization(self):
        '''
        - Initialize the population
        '''
    
       
        # 1) Initialize the population with random permutations of the number of clusters
        self.population_cluster = np.array([np.random.permutation(self.num_clusters) for _ in range(self.population_size)])
        
        # 2) Generate possible entries and exit points for each cluster
        self.generate_possible_entries_and_exit_points()
        
        # 3) Merge clusters based on the cluster sequence and entry/exit points
        self.population_cities,self.population_startEnd = self.merge_clusters(self.population_cluster)
        #print(f"\nInitial Population: {self.population_cities}")

        # 4) Merge clusters using greedy approach
        self.population_cities = self.greedyMerge_populationClusters(self.population_cluster)
        #print(f"\nMerged Population: {self.population_merged}")
    
        
        # 5) Calculate fitness for the merged population
        self.fitness = self.calculate_fitness(self.population_cities)
        print(f"\nInitial Fitness: {self.fitness}")
    
        self.population_all_list = [self.population_cluster,self.population_startEnd,self.population_cities]
        #print(f"\n Initial Population All List: {self.population_all_list}")
        
        # 6) Print model info
        self.print_model_info()

    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 2) Selection ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def selection_k_tournament(self, num_individuals, k=3):
        #print(f"\n ----------------- Selection K Tournament -----------------")
     
        population_cluster = self.population_all_list[0]
        population_startEnd = self.population_all_list[1]
        population_merged = self.population_all_list[2]

        # Step 1: Randomly choose k individuals for each tournament
        tournament_indices = np.random.choice(self.population_size, size=(num_individuals, k), replace=True)
        #print(f"\n tournament indices: {tournament_indices}")

        # Step 2: Get the fitness scores of the selected individuals
        tournament_fitness = self.fitness[tournament_indices]
        #print(f"\n Fitness of the tournament is : {tournament_fitness}")    
        

        # Step 3: Get the index of the individual with the best fitness in each tournament
        best_indices_in_tournament = np.argmin(tournament_fitness, axis=1)

        # Step 4: Use these indices to select the best individuals from each tournament
        best_selected_indices = tournament_indices[np.arange(num_individuals), best_indices_in_tournament]

        # Step 5: Return the selected individualsi
        # print this self.population[best_selected_indices],self.population_merged[best_selected_indices]
        #print(f"\n K_Tournament indices: {self.population[best_selected_indices]} and {self.population_merged[best_selected_indices]}")  
        selected_all_list = []
        selected_all_list = [population_cluster[best_selected_indices],population_startEnd[best_selected_indices],population_merged[best_selected_indices]] 

        return selected_all_list
    
    def selection_k_tournament_population(self,num_individuals,population,fitness,k):
        '''
        - Select the best individuals from the population based on the k_tournament
        '''
        # Step 1: Randomly choose k individuals for each tournament
        tournament_indices = np.random.choice(population.shape[0], size=(num_individuals, k), replace=True)
        #print(f"\n tournament indices: {tournament_indices}")

        # Step 2: Get the fitness scores of the selected individuals
        tournament_fitness = fitness[tournament_indices]
        #print(f"\n Fitness of the tournament is : {tournament_fitness}")    
        

        # Step 3: Get the index of the individual with the best fitness in each tournament
        best_indices_in_tournament = np.argmin(tournament_fitness, axis=1)

        # Step 4: Use these indices to select the best individuals from each tournament
        best_selected_indices = tournament_indices[np.arange(num_individuals), best_indices_in_tournament]

        #step 5 retrieve also teh fitness of the best individuals
        best_selected_fitness = fitness[best_selected_indices]

        # Step 5: Return the selected individuals and its fitness
        return population[best_selected_indices],best_selected_fitness
    
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 4) Crossover & Mutation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    def crossover_singlepoint_population(self, selected_all_list):
        # Number of parents in the population
        #print(f"\n ----------------- Crossover Single Point Population -----------------")
        population_cluster = selected_all_list[0]
        population_startEnd = selected_all_list[1]
        population_cities = selected_all_list[2]

        num_parents = population_cluster.shape[0]
        #print(f"\n population_cluster: Parent population_cluster is : {population_cluster[1]}" )
        
        # Initialize the children population_cluster
        children_population = np.zeros_like(population_cluster)
        children_population_startEnd = np.zeros_like(population_startEnd)
        
        # Generate random crossover points
        crossover_index = np.random.randint(1, self.gen_size)
        
        # Perform crossover for each pair of parents
        for i in range(num_parents):
            # Get the indices of the parents
            parent1 = population_cluster[i]
            parent2 = population_cluster[(i + 1) % num_parents]

            parent1_startEnd = population_startEnd[i]
            parent2_startEnd = population_startEnd[(i + 1) % num_parents]

            crossover_index = np.random.randint(0, len(parent1))
            
            # Perform crossover
            child1, child2, child1_startEnd, child2_startEnd = self.crossover_singlepoint(crossover_index,parent1, parent2,parent1_startEnd,parent2_startEnd)
            
            # Add the children to the population_cluster
            children_population[i] = child1
            children_population[(i + 1) % num_parents] = child2

            children_population_startEnd[i] = child1_startEnd
            children_population_startEnd[(i + 1) % num_parents] = child2_startEnd

    
   
        children_population_cities = self.merge_paths_givenStartEnd(children_population,children_population_startEnd)
    
        offspring_all_list = [children_population,children_population_startEnd,children_population_cities]
        return offspring_all_list   
    

    def crossover_singlepoint(self, crossover_index,parent1, parent2,parent1_startEnd,parent2_startEnd):
        """
        Perform Order Crossover (OX) for valid TSP offspring.

        Parameters:
        - parent1: numpy array, representing a parent tour (e.g., [0, 1, 2, 3, ...])
        - parent2: numpy array, representing a parent tour (e.g., [3, 2, 1, 0, ...])

        Returns:
        - child1, child2: Two offspring with no duplicate cities and valid tours.
        """
        #print(f"\n ----------------- Crossover Single Point -----------------")
        #print(f"\n Parent1: {parent1}")
        #print(f"\n Parent2: {parent2}")
        #print(f"\n Parent1 StartEnd: {parent1_startEnd}")
        #print(f"\n Parent2 StartEnd: {parent2_startEnd}")
        #print(f"\n Crossover index: {crossover_index}")
        
        num_cities = len(parent1)
        
        
        
        
        # Step 2: Initialize offspring as arrays of -1 (indicating unfilled positions)
        child1 = -1 * np.ones(num_cities, dtype=int)
        child2 = -1 * np.ones(num_cities, dtype=int)
        child1_startEnd = -1 * np.ones_like(parent1_startEnd)
        child2_startEnd = -1 * np.ones_like(parent2_startEnd)
        
        # Step 3: Copy the selected segment from parent1 to child1, and from parent2 to child2
        child1[:crossover_index] = parent1[:crossover_index]
        child2[crossover_index:] = parent2[crossover_index:]
        child1_startEnd[:crossover_index] = parent1_startEnd[:crossover_index]
        child2_startEnd[crossover_index:] = parent2_startEnd[crossover_index:]
        #print(f"\n Child1: {child1}")
        #print(f"\n Child2: {child2}")
        #print(f"\n Child1 StartEnd: {child1_startEnd}")
        #print(f"\n Child2 StartEnd: {child2_startEnd}")

        # Step 4: Fill the remaining positions in child1 with cities from parent2, while avoiding duplicates
            
        child1_end,child1_end_startEnd = self.fill_child(child1, child1_startEnd, parent2, parent2_startEnd)
        child2_end,child2_end_startEnd = self.fill_child(child2, child2_startEnd, parent1, parent1_startEnd)      
   

    
        #print(f"\n Child1: {child1_end}")
        #print(f"\n Child2: {child2_end}")
        #print(f"\n Child1 StartEnd: {child1_end_startEnd}")
        #print(f"\n Child2 StartEnd: {child2_end_startEnd}")

     

        return child1_end, child2_end, child1_end_startEnd, child2_end_startEnd

    
    
    def fill_child(self,child,child_startEnd, other_parent,other_parent_startEnd):

        for idx, element1 in enumerate(child):
            if element1 == -1:
                for element2, start_end in zip(other_parent, other_parent_startEnd):
                    if element2 not in child:
                        child[idx] = element2
                        child_startEnd[idx] = start_end
                        break  # Stop once we find a valid element
        return child,child_startEnd

    
    
    def fill_child_startEnd(self,child, other_parent):
        counter = 0
        for element1 in child:
            if np.array_equal(element1, np.array([-1, -1])):
                for element2 in other_parent:
                    if element2 not in child:
                        child[counter] = element2
            counter += 1
        return child



    def mutation_singlepoint_population(self, offspring_all_list):
        #print(f"\n ----------------- Mutation Single Point Population -----------------")
        mutation_rate = self.mutation_rate
        population_cluster = offspring_all_list[0]
        population_startEnd = offspring_all_list[1]
        population_cities = offspring_all_list[2]

        #print(f"\n Mutation rate is : {mutation_rate}")
        #print(f"\n Population cluster is : {population_cluster}")
        #print(f"\n Population cities is : {population_cities}")
        #print(f"\n Population startEnd is : {population_startEnd}")

        # Number of individuals in the population
        num_individuals = population_cluster.shape[0]
        
        # Initialize the mutated population
        mutated_population_cluster = np.copy(population_cluster)
        mutated_population_cities = np.copy(population_cities)
        mutated_population_startEnd = np.copy(population_startEnd)
        
        # Perform mutation for each individual
        mutated_distance = self.calculate_fitness(mutated_population_cities)
        best_index = np.argmin(mutated_distance)
        
        for i in range(num_individuals):
            if np.random.rand() < mutation_rate and i != best_index:
                mutated_population_startEnd[i] = self.mutation_singlepoint(population_cluster[i],individual_startEnd=population_startEnd[i] ,mutation_rate=mutation_rate)
      
        merged_mutated_population = self.merge_paths_givenStartEnd(mutated_population_cluster,mutated_population_startEnd)

        offspring_mutated_all_list = [mutated_population_cluster,mutated_population_startEnd,merged_mutated_population]
        return offspring_mutated_all_list



    def mutation_singlepoint(self, individual,individual_startEnd, mutation_rate=0.8):
        #print(f"\n ----------------- Mutation Single Point -----------------")
        
     
        num_genes = len(individual)
        
        
        # Initialize the mutated individual
        mutated_individual = np.copy(individual)
        mutated_startEnd = np.copy(individual_startEnd)
        
        

        # Select how many genes to mutate
        num_mutations = np.random.randint(1, num_genes)

        # Select indices to mutate
        mutation_indices = np.random.choice(num_genes, size=num_mutations, replace=False)
        #print(f"\n Mutation indices are : {mutation_indices} from {num_genes}")

        

        
        # Perform mutation for each gene
        #print(f"\n Individual is : {individual}")
        #print(f"\n Individual StartEnd is : {individual_startEnd}")
        #print(f"\n Possible startEnd: {self.possible_entry_and_exit_points_list}")
        for idx, element1 in enumerate(individual_startEnd):
            #print(f"\n Element1 is : {element1}")
            cluster = individual[idx]
            #print(f"\n Cluster is : {cluster}")
            
         
            # Get a random index for the mutation
            if idx in mutation_indices:
                
                mutation_index = idx
                #print(f"\n Mutation index is : {mutation_index}")
                
                possible_startEnd = self.possible_entry_and_exit_points_list[cluster]

                #find if element 1 is in the possible startEnd
                if not any(np.array_equal(element1, entry) for entry in possible_startEnd):
                    print(f"\n EMERGENCY: Element 1 is not in the possible startEnd")
                    print(f"\n Element 1 is: {element1} not in {possible_startEnd}")
                    
                  
                #print(f"\n Possible startEnd is : {possible_startEnd}")
                mutation_index2 = np.random.randint(len(possible_startEnd))
                #print(f"\n Mutation index 2 is : {mutation_index2}")

                mutated_startEnd[idx] = possible_startEnd[mutation_index2]
                #print(f"\n Mutated Element 1 is : {mutated_startEnd[idx]}")
                
                
              

                #mutated_individual[i], mutated_individual[mutation_index] = mutated_individual[mutation_index], mutated_individual[i]
              
   
        #print(f"\n Mutated Individual is : {mutated_individual}")
        #print(f"\n Mutated StartEnd is : {mutated_startEnd}")
        return mutated_startEnd
    
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) Local Search ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    def calculate_total_distance_individual(self,route, distance_matrix):
        '''
        Calculate the total distance of a given route
        '''
        current_indices = route[:-1]
        next_indices = route[1:]
        distances = distance_matrix[current_indices, next_indices]
        total_distance = np.sum(distances) + distance_matrix[route[-1], route[0]]  # Return to start
        return total_distance

    def two_opt_no_loops_opt(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
        #print(f"\n ------------LOCAL SEARCH------------")
        time_start = time.time()
        best_route = np.copy(route)
        inital_fitness = self.calculate_total_distance_individual(best_route, distance_matrix)
        routes = [] 
        n = len(route)
        
        # Generate all pairs of indices i, j (i < j)
        i_indices, j_indices = np.triu_indices(n, k=2)
        
        # Calculate the old and new distances for all (i, j) pairs
        i_next = (i_indices + 1) % n
        j_next = (j_indices + 1) % n
        old_distances = (
            distance_matrix[best_route[i_indices], best_route[i_next]] +
            distance_matrix[best_route[j_indices], best_route[j_next]]
        )
        
        new_distances = (
            distance_matrix[best_route[i_indices], best_route[j_indices]] +
            distance_matrix[best_route[i_next], best_route[j_next]]
        )
        
        # Calculate delta_distances once
        delta_distances = new_distances - old_distances
        
        improvement = True
        iteration = 0
        while improvement and iteration < max_iterations:
            improvement = False
            #print(f"\n LS: Iteration number {iteration}")
            
            
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]
       
            
          
            if np.any(delta_distances[top_k_indices] < 0):
              
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
                i, j = i_indices[best_swap_index], j_indices[best_swap_index]
                #print(f"\n Route at iteration {iteration}: {best_route}")
                #print(f"\n Delta distances at iteration {iteration}: {delta_distances}")
                #print(f"\n Best swap index: {best_swap_index}")
                #print(f"\n i: {i} - j: {j}")
                
                # Perform the 2-opt swap: reverse the segment between i and j
                #print(f"Route segment before swap: {best_route[i + 1:j + 1]}")
                best_route[i + 1: j + 1] = best_route[i + 1: j + 1][::-1]
                #print(f"Route segment after swap: {best_route[i + 1:j + 1]}")
                
                
                
                #routes.append(best_route)
                routes.append(np.copy(best_route))

                improvement = True
                
                # Update the distances for the swapped pairs
                old_distances = (
                    distance_matrix[best_route[i_indices], best_route[i_next]] +
                    distance_matrix[best_route[j_indices], best_route[j_next]]
                )
                
                new_distances = (
                    distance_matrix[best_route[i_indices], best_route[j_indices]] +
                    distance_matrix[best_route[i_next], best_route[j_next]]
                )
                
                # Recalculate delta_distances after the swap
                delta_distances = new_distances - old_distances
                #print(f"\n Delta distances at iteration {iteration}: {delta_distances}")
             
            
            iteration += 1
        if len(routes) > 0:
            #print(f"\n Routes: {routes}")

            routes_np = np.array(routes)
            #check how many of the routes are unique
         
            #print(f"\n Unique Routes: {unique_routes}")

            final_fitness = self.calculate_fitness(routes_np)
            #print(f"\n Final Fitness: {final_fitness}")
            best_sol = routes_np[np.argmin(final_fitness)]
            best_fit = final_fitness[np.argmin(final_fitness)]
            print(f"\n Initial Fitness: {inital_fitness} - Final Fitness: {best_fit} at index: {np.argmin(final_fitness)}") 

            if inital_fitness > best_fit:
                best_route = best_sol
            else:
                best_route = route
                best_fit = inital_fitness
        else:
            best_route = route
            best_fit = inital_fitness

        #print(f"\n    LS Iterations: {iteration}")   
        time_end = time.time()
        time_local_search = time_end - time_start
        print(f"\n Time for the local search: {time_local_search}",flush=True)
        return best_route
   
    def local_search_population(self, population_all_list,n_best = 2, max_iterations=10):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals
        '''
        distance_matrix = self.distance_matrix
        population_clusters = population_all_list[0]
        population_startEnd = population_all_list[1]
        population_cities = population_all_list[2]


        
        # Step 1: Evaluate fitness for all individuals in the population
        distances = self.calculate_fitness(population_cities)
        
        # Step 2: Select the top `n_best` individuals
        
        best_indices = np.argsort(distances)[:n_best]
        
        # Step 3: Apply 2-opt to the selected top individuals
        for i in best_indices:
            population_cities[i] = self.two_opt_no_loops_opt(population_cities[i], distance_matrix, max_iterations,k_neighbors=10)
            #population[i] = self.two_opt_no_loops(population[i], distance_matrix, max_iterations)
        after_all_list = [population_clusters,population_startEnd,population_cities]
        return after_all_list


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) Elimnation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def eliminate_population(self, population_all_list, mutated_all_list):
        """
        Selects the best individuals from the combined population and offspring
        based on fitness.

        Parameters:
        - population: numpy array of shape (population_size, individual_size)
        - offspring: numpy array of shape (offspring_size, individual_size)

        Returns:
        - new_population: numpy array of shape (self.population_size, individual_size)
        """
        #print(f"\n ---------------- Eliminate Population ----------------")
        # Combine the original population with the offspring
        #print(f"\n Orginial --> {population}")
        #print(f"\n Offspring--> {offspring}")
        population_cluster = population_all_list[0]
        population_startEnd = population_all_list[1]
        population_cities = population_all_list[2]

        mutated_population_cluster = mutated_all_list[0]
        mutated_population_startEnd = mutated_all_list[1]
        mutated_population_cities = mutated_all_list[2]

        combined_population = np.vstack((population_cluster, mutated_population_cluster))
        combined_population_startEnd = np.vstack((population_startEnd, mutated_population_startEnd))
        combined_population_cities = np.vstack((population_cities, mutated_population_cities))

        # Calculate fitness for the combined population
        fitness_scores = self.calculate_fitness(mutated_population_cities)
        combined_fitness = np.hstack((self.fitness, fitness_scores))
        #print(f"\n Fitness V1--> {fitness_scores}")
       

        # Get the indices of the best individuals based on fitness
        unique_population_cities, unique_indices = np.unique(combined_population_cities, axis=0, return_index=True)
        # Get the fitness scores of unique individuals
        unique_fitness_scores = combined_fitness[unique_indices]
        unique_population = combined_population[unique_indices]
        unique_population_startEnd = combined_population_startEnd[unique_indices]
      
        best_indices = np.argsort(unique_fitness_scores)[:self.population_size]
        self.fitness = unique_fitness_scores[best_indices]

        # Select the best individuals
        self.population_cluster = unique_population[best_indices]
        self.population_startEnd = unique_population_startEnd[best_indices]
        self.population_cities = unique_population_cities[best_indices]  
        self.population_all_list = [self.population_cluster,self.population_startEnd,self.population_cities] 
        self.fitness = unique_fitness_scores[best_indices]
        self.hamming_distance, _ = self.calculate_hamming_distance_population(self.population_cities)


    def eliminate_population_fs_tournament(self, population_all_list, mutated_all_list, sigma, alpha, k):
        """
        Eliminates population using k-tournament selection with fitness sharing.
        
        Parameters:
        - population_all_list: list of numpy arrays [population_cluster, population_startEnd, population_cities]
        - mutated_all_list: list of numpy arrays [mutated_population_cluster, mutated_population_startEnd, mutated_population_cities]
        - sigma: float, sharing parameter for fitness sharing
        - alpha: float, exponent parameter for fitness sharing
        - k: int, size of tournament for selection
        
        Returns:
        - Updated population lists with best individuals.
        """
        # Combine populations across all representations
        population_cluster, population_startEnd, population_cities = population_all_list
        mutated_cluster, mutated_startEnd, mutated_cities = mutated_all_list

        combined_cluster = np.vstack((population_cluster, mutated_cluster))
        combined_startEnd = np.vstack((population_startEnd, mutated_startEnd))
        combined_cities = np.vstack((population_cities, mutated_cities))

        # Calculate fitness for the combined cities population
        fitness_scores = self.calculate_fitness(combined_cities)

        # Initialize survivors' indices array
        survivors_idxs = -1 * np.ones(self.population_size, dtype=int)

        # Select the best individual directly
        best_index = np.argmin(fitness_scores)
        survivors_idxs[0] = best_index

        # Exclude best individual from valid candidates
        valid_candidates = np.setdiff1d(np.arange(len(fitness_scores)), [best_index])

        # Perform k-tournament selection for the remaining survivors
        for i in range(1, self.population_size):
            new_fitness = self.fitness_sharing_individual_np(
                population=combined_cities, 
                survivors=survivors_idxs[:i],  
                population_fitness=fitness_scores, 
                sigma=sigma, 
                alpha=alpha
            )

            if len(valid_candidates) > 0:
                tournament_candidates = np.random.choice(valid_candidates, size=min(k, len(valid_candidates)), replace=False)
           
            else:
                raise ValueError("No valid candidates remain for selection.")
            
            best_in_tournament = tournament_candidates[np.argmin(new_fitness[tournament_candidates])]
            survivors_idxs[i] = best_in_tournament
            valid_candidates = valid_candidates[valid_candidates != best_in_tournament]

        # Select the final population based on the survivors' indices
        self.population_cluster = combined_cluster[survivors_idxs]
        self.population_startEnd = combined_startEnd[survivors_idxs]
        self.population_cities = combined_cities[survivors_idxs]

        self.population_all_list = [self.population_cluster, self.population_startEnd, self.population_cities]
        self.fitness = fitness_scores[survivors_idxs]
        self.hamming_distance, _ = self.calculate_hamming_distance_population(self.population_cities)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 6) Merging Paths ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    def greedyMerge_populationClusters(self,population):
        """
        Merges clusters in a greedy manner by iterating through the solution array,
        starting from the first two clusters and merging them one by one.

        Parameters:
        - solution (np.ndarray): An array representing the order of cluster indices.
        - distance_matrix (np.ndarray): A 2D numpy array where distance_matrix[i, j] gives
                                        the distance between city i and city j.

        Returns:
        - merged_cluster (np.ndarray): A single array representing the merged cluster.
        """
        # Start with the first cluster
        #print(f"\n ----------------- Greedy Merge -----------------")
        #print(f"\n Clusters solutions : {self.clusters_solution_matrix}")
        distance_matrix = self.distance_matrix
        cluster_inital = 0
        population_merged = []
        for solution in population:
            num_clusters = len(solution)
        

            # This will store the cities as they are merged
            merged_solutions = []

            #print(f"\n Solution is : {solution}")
            
            # Iterate over the clusters and merge them greedily
            for cluster_idx in range(0, num_clusters - 1):
                # Extract the current cluster's cities
                if cluster_idx == 0:
                    current_cluster_cities = self.clusters_solution_matrix[solution[cluster_idx]]
                    num_cities1 = len(current_cluster_cities)
                else:
                    current_cluster_cities = merged_solutions[-1]  # Last merged cluster
                    num_cities1 = len(current_cluster_cities)

                # Get the next cluster's cities
                next_cluster_cities = self.clusters_solution_matrix[solution[cluster_idx + 1]]
                num_cities2 = len(next_cluster_cities)
                total_cities = num_cities1 + num_cities2

                # Print for debugging
                #print(f"\n Current cluster cities: {current_cluster_cities}")
                #print(f"\n Next cluster cities: {next_cluster_cities}")

                # Compute pairwise distances between all cities in the current and next cluster
                dist_matrix = distance_matrix[np.ix_(current_cluster_cities, next_cluster_cities)]

                # Find the pair of cities that have the minimum distance
                min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                i, j = min_dist_idx  # Cities to be connected
                
                # Output the pair with the minimum distance
            #print(f"Connecting city {current_cluster_cities[i]} with city {next_cluster_cities[j]}")

                # Step 1: Remove the old edges between the closest cities in both clusters
                #print(f"\n --- Merging Clusters ---")
                #print(f"Removing edge between city {current_cluster_cities[i-1]} and city {current_cluster_cities[i]}")
                #print(f"Removing edge between city {next_cluster_cities[j-1]} and city {next_cluster_cities[j]}")

                # Step 2: Merge current cluster up to city[i], then add the connection to city[j]
                #print(f"Adding cities from current cluster up to city {current_cluster_cities[i]}")
                merged_cluster = current_cluster_cities[:i + 1]  # Include all cities before i

                # Step 3: Add the second cluster, starting from city[j]
                #print(f"Adding cities from next cluster, starting from city {next_cluster_cities[j]}")
                merged_cluster = np.concatenate([merged_cluster, next_cluster_cities[j:]])

                # Step 4: Add any remaining cities from the second cluster before city j
                remaining_cities_from_next = next_cluster_cities[:j]
                #print(f"Adding remaining cities from next cluster: {remaining_cities_from_next}")
                merged_cluster = np.concatenate([merged_cluster, remaining_cities_from_next])

                # Step 5: Add remaining cities from the first cluster after city[i]
                #print(f"Adding remaining cities from current cluster after city {current_cluster_cities[i]}")
                merged_cluster = np.concatenate([merged_cluster, current_cluster_cities[i + 1:]])

                # After merging, print the final merged cluster
                #print(f"Final merged cluster: {merged_cluster}")
                #self.cities_model.add_cities_sequence(merged_cluster)
                #self.cities_model.plot_clusters_sequence(city_sequence=True)   
                merged_solutions.append(merged_cluster)
            population_merged.append(merged_cluster)


        return np.array(population_merged)
       

    def generate_possible_entries_and_exit_points(self):
        """
        Generate all valid entry and exit points for each cluster efficiently using NumPy.
        Ensures entry != exit for all pairs, and the exit city is adjacent to the entry city (both forward and backward).
        For the last city in the list, the adjacency is cyclic.
        """
        self.possible_entry_and_exit_points_list = []

        for cluster in range(self.num_clusters):
            # Get the cities for the current cluster as a NumPy array
            cities_numpy = self.clusters_solution_matrix[cluster]
            num_cities = len(cities_numpy)

            # Create a list to store the valid (entry, exit) pairs
            valid_pairs = []

            # Iterate over each city to generate the valid entry and exit points
            for i in range(num_cities):
                # Entry point is cities_numpy[i]
                entry = cities_numpy[i]

                # Forward adjacency (exit to the next city, looping back for the last city)
                if i == num_cities - 1:
                    # Last city connects back to the first city
                    exit_forward = cities_numpy[0]
                else:
                    # Adjacent city in the list (forward direction)
                    exit_forward = cities_numpy[i + 1]

                # Backward adjacency (exit to the previous city, looping back for the first city)
                if i == 0:
                    # First city connects back to the last city
                    exit_backward = cities_numpy[num_cities - 1]
                else:
                    # Adjacent city in the list (backward direction)
                    exit_backward = cities_numpy[i - 1]

                # Add both forward and backward pairs (entry, exit)
                valid_pairs.append((entry, exit_forward))
                valid_pairs.append((entry, exit_backward))

            # Convert the list of valid pairs to a NumPy array for easier manipulation
            valid_pairs_np = np.array(valid_pairs)

            # Append to the result list
            self.possible_entry_and_exit_points_list.append(valid_pairs)

            # Print for debugging
            print(f"\nPossible entry and exit points for cluster {cluster}:")
            print(f"\nNumber of pairs: {len(valid_pairs_np)}")
            #print(f"\nNumber of cities: {num_cities}")
        #print(f"\n Possible entry and exit points: {self.possible_entry_and_exit_points_list}")

    def merge_clusters(self, population_cluster):
        """
        Reconstructs the solution for each individual in the population based on the cluster sequence
        and using the start and end cities for each cluster.

        Parameters:
        - population_cluster (np.ndarray): Array containing the order of clusters for each individual

        Returns:
        - merged_population (np.ndarray): A list of full traveling paths for each individual
        """
        merged_population = []
        start_end_population_list = []
        start_end_list = []

        for cluster_sequence in population_cluster:
            start_end_list = []
            full_solution = []  # List to store the complete solution for the individual
            visited_cities = set()  # To track cities already visited
            
            # Reconstruct each cluster path in the given order
            for cluster_idx in range(len(cluster_sequence)):
                cluster_id = cluster_sequence[cluster_idx]
                start_city, end_city = self.get_start_end_for_cluster(cluster_id)
                start_end_list.append((start_city, end_city))
                cluster_cities = self.clusters_solution_matrix[cluster_id]

                # Reconstruct the path for this cluster
                cluster_solution = self.reconstruct_cluster_path(cluster_cities, start_city, end_city, visited_cities)

                # Append this cluster's solution to the full solution
                full_solution.extend(cluster_solution)

            merged_population.append(np.array(full_solution))
            start_end_population_list.append(start_end_list)
            #print(f"\nMerged population: {merged_population}")
            #print(f"\nLength of merged population: {len(merged_population[-1])}")
            #print(f"\nStart-End list: {start_end_population_list}")

        return np.array(merged_population), np.array(start_end_population_list) 


    def reconstruct_cluster_path(self, cluster_cities, start_city, end_city, visited_cities):
        """
        Reconstructs the path for a single cluster based on its cities and the start/end points.
        Ensures cities are in the correct order and that no cities are duplicated.
        
        Parameters:
        - cluster_cities (np.ndarray): The list of cities for the cluster
        - start_city (int): The start city for the path
        - end_city (int): The end city for the path
        - visited_cities (set): A set of cities that have already been added to the solution
        
        Returns:
        - cluster_path (np.ndarray): The reconstructed path of cities for this cluster
        """
        # Create a list to store the cluster's path
        cluster_path = []
        #print(f"\n Cluster cities: {cluster_cities}")
        #print(f"\n Start city: {start_city}, End city: {end_city}")

        if start_city not in cluster_cities or end_city not in cluster_cities:
            print(f"\n Start city or end city not in cluster cities")
            print(f"\n Start city: {start_city}, End city: {end_city}")
            print(f"\n Cluster cities: {cluster_cities}")

            
        
        # We need to ensure the path starts from the start city and ends at the end city
        # The cluster's path must go through all cities in the correct order
        # First, find the index of the start city and rotate the cities list to start at start_city
        start_idx = np.where(cluster_cities == start_city)[0][0]
        cluster_cities = np.roll(cluster_cities, -start_idx)  # Rotate array to start from the start_city
        #print(f"\n Rotated cluster cities: {cluster_cities}")

        # Now the cluster_cities array starts at start_city, but we need to create a valid path for the cluster
        # Traverse the cluster cities and add them to the path in the correct order
        for city in cluster_cities:
            if city not in visited_cities and city != end_city:
                cluster_path.append(city)
                visited_cities.add(city)  # Mark the city as visited
        
        # Ensure the path ends at the end city if it isn't already the last city
        if cluster_path[-1] != end_city:
            cluster_path.append(end_city)
            visited_cities.add(end_city)
        
        return np.array(cluster_path)



    def get_start_end_for_cluster(self, cluster_id, element=None):
        """
        Retrieves the start and end cities for the given cluster.

        Parameters:
        - cluster_id (int): The identifier of the cluster
        - element (int, optional): The index of the valid entry and exit pair to select. If None, a random pair is selected.

        Returns:
        - (start_city, end_city): A tuple containing the start and end cities for the cluster
        """
        # Fetch the possible entry and exit points for the cluster from the precomputed list
        valid_pairs = self.possible_entry_and_exit_points_list[cluster_id]
        
        # If element is not provided (None), randomly select a valid pair
        if element is None:
            element = np.random.randint(0, len(valid_pairs))  # Randomly select an index
        
        #print(f"\n Valid pairs: {valid_pairs}")
        #print(f"\n Element: {element}")
        # Select the specific pair (start, end) based on the provided element index
        start_city, end_city = valid_pairs[element]
        #print(f"\n Start city: {start_city}, End city: {end_city}")
        
        return start_city, end_city
    

    def merge_paths_givenStartEnd(self, population_cluster,population_startEnd):
        """
        Reconstructs the solution for each individual in the population based on the cluster sequence
        and using the start and end cities for each cluster.

        Parameters:
        - population_cluster (np.ndarray): Array containing the order of clusters for each individual

        Returns:
        - merged_population (np.ndarray): A list of full traveling paths for each individual
        """
        #print(f"\n ----------------- Merge Paths Given Start and End -----------------")
        #print(f"\n Population cluster: {population_cluster}")
        #print(f"\n Population startEnd: {population_startEnd}")
        merged_population = []
        start_end_population_list = []
        start_end_list = []
        counter = 0

        for cluster_sequence in population_cluster:
            start_end_list = []
            full_solution = []  # List to store the complete solution for the individual
            visited_cities = set()  # To track cities already visited
            cluster_startEnd = population_startEnd[counter]   
            #print(f"\n Cluster sequence: {cluster_sequence}")
            # Reconstruct each cluster path in the given order
            for cluster_idx in range(len(cluster_sequence)):
                #print(f"\n Cluster idx: {cluster_idx}")
                cluster_id = cluster_sequence[cluster_idx]
                start_city, end_city = cluster_startEnd[cluster_idx]
                #print(f"\n Start city: {start_city}, End city: {end_city}")
                start_end_list.append((start_city, end_city))
                cluster_cities = self.clusters_solution_matrix[cluster_id]

                # Reconstruct the path for this cluster
                cluster_solution = self.reconstruct_cluster_path(cluster_cities, start_city, end_city, visited_cities)

                # Append this cluster's solution to the full solution
                full_solution.extend(cluster_solution)

            merged_population.append(np.array(full_solution))
            start_end_population_list.append(start_end_list)
            #print(f"\nMerged population: {merged_population}")
            #print(f"\nLength of merged population: {len(merged_population[-1])}")
            #print(f"\nStart-End list: {start_end_population_list}")
            counter += 1

        #print(f"\nMerged population: {merged_population}")

        return np.array(merged_population)



    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 7) Fitness Calculation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def fitness_sharing_individual_np(self, population, survivors, population_fitness, sigma, alpha):
        '''
        - Vectorized fitness sharing for TSP using `calculate_hamming_distance_individual`.
        - Computes the fitness for each individual based on its pairwise Hamming distance to the survivors.
        '''
        # Number of individuals in the population
        num_individuals = len(population)
        
        # Initialize the fitness sharing multipliers as 1
        fitness_sharing = np.ones(num_individuals)
        
        # For each survivor, apply fitness sharing to all individuals
        for survivor_idx in survivors:
            if survivor_idx == -1:
                break
            
            # Get the survivor
            survivor = population[survivor_idx]
            
            # Compute pairwise Hamming distances for each individual to the current survivor
            survivor_distances = np.array([self.calculate_hamming_distance_individual(ind, survivor) for ind in population])
            #print(f"Survivor distances: {survivor_distances}")
            
            
            
            
            # Apply the fitness sharing: if distance <= sigma, apply the sharing term (1 + alpha), else 1
            sharing_term = np.where(survivor_distances <= sigma, (1-((survivor_distances)/sigma)**alpha), 1)
            # Handle identical individuals (distance = 0) explicitly by applying the penalty
            #sharing_term[survivor_distances == 0] = 1 - (1 / sigma) ** alpha  # Apply custom penalty for identical individuals
            sharing_term[survivor_distances == 0] = 0.00000000000000000001  # Apply custom penalty for identical individuals
            #print(f"Sharing term: {sharing_term}")
            
            # Multiply the fitness sharing terms with the current fitness sharing values
            fitness_sharing *= 1/sharing_term
        
        # Compute the new fitness values by applying the sharing effect
        #print(f"Fitness sharing: {fitness_sharing}")
        #print(f"Population fitness: {population_fitness}")
        fitness_new = population_fitness * fitness_sharing
        #print(f"Fitness new: {fitness_new}")
        
        return fitness_new

    def calculate_fitness(self,population):
        '''
        - Calculate the fitness of the population
        '''
        '''
        num_individuals = population.shape[0]
        fitness = np.zeros(num_individuals)
        for i in range(num_individuals):
            fitness[i] = np.sum(self.distance_matrix[population[i],np.roll(population[i],-1)])
        #print(f"Fitness shape --> {np.shape(fitness)} \n Fitness is : {fitness[1]}" )
        '''

        num_individuals = population.shape[0]

        # Get the indices for the current and next cities
        current_indices = population[:, :-1]
        next_indices = population[:, 1:]
        
        # Calculate the distances between consecutive cities
        distances = self.distance_matrix[current_indices, next_indices]
        
        # Sum the distances for each individual
        fitness = np.sum(distances, axis=1)
        
        # Add the distance from the last city back to the first city
        last_to_first_distances =self.distance_matrix[population[:, -1], population[:, 0]]
        fitness += last_to_first_distances

        

        
        #print(f"Fitness shape --> {np.shape(fitness)} \n Fitness is : {fitness}") 
        
        return fitness

    def calculate_hamming_distance_population(self,population):
        # Number of cities (positions) in each solution
        num_cities = population.shape[1]
        
        # Compute pairwise Hamming distances
        # diff_matrix[i, j, k] -> True if city `k` in solution `i` differs from city `k` in solution `j`, else False
        diff_matrix = population[:, None, :] != population[None, :, :]
        
        # Sum the differences across city positions
        hamming_distances = np.sum(diff_matrix, axis=2)
        
        # Normalize the Hamming distance: divide each distance by the number of cities
        normalized_hamming_distances = hamming_distances / num_cities

        #print(f"Normalized Hamming distances: {normalized_hamming_distances}")
        #print(f"Normalized Hamming distances shape: {normalized_hamming_distances.shape}")

        # Calculate the mean of the normalized Hamming distances for each solution
        row_means = np.mean(normalized_hamming_distances, axis=1)
        #print(f"Row means: {row_means}")
        #print(f"Row means shape: {row_means.shape}")
        
        # Finally, compute the average of these row means
        avg_diversity = np.mean(row_means)
        #print(f"Average diversity: {avg_diversity}")
        return avg_diversity, hamming_distances
    
    def calculate_hamming_distance_individual(self,individual1,individual2):
        '''
        - Calculate the Hamming distance between two individuals
        '''
        #print(f"Individual 1: {individual1}")
        #print(f"Individual 2: {individual2}")
        # Number of cities (positions) in each solution
        num_cities = len(individual1)
        
        # Compute pairwise Hamming distances
        # diff_matrix[i, j] -> True if city `i` in solution `individual1` differs from city `i` in solution `individual2`, else False
        diff_matrix = individual1 != individual2
        
        # Sum the differences across city positions
        hamming_distance = np.sum(diff_matrix)/num_cities
        #print(f"Hamming distance: {hamming_distance}")
        
        
        
        return hamming_distance

    def calculate_add_hamming_distance(self,population,crossover=False,crossover_old= False,mutation1=False,mutation2=False,local_search=False,elimination=False): 
        '''
        - Calculate the fitness of the population
        '''
        hamming_distance,_ = self.calculate_hamming_distance_population(population)
        if crossover:
            self.hamming_distance_crossover_list.append(hamming_distance)
        elif mutation1:
            self.hamming_distance_mutation1_list.append(hamming_distance)
        elif mutation2:
            self.hamming_distance_mutation2_list.append(hamming_distance)
        elif local_search:
            self.hamming_distance_local_search_list.append(hamming_distance)
        elif elimination:
            self.hamming_distance_elimination_list.append(hamming_distance)
        elif crossover_old:
            self.hamming_distance_crossoverOld_list.append(hamming_distance) 

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 8) Plotting------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    def plot_fitness_dynamic(self):
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.best_fitness_list))),
            y=self.best_fitness_list,
            mode='lines+markers',
            name='Best Objective',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.mean_fitness_list))),
            y=self.mean_fitness_list,
            mode='lines+markers',
            name='Mean Objective',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        # Get the last iteration's best and mean objective values
        last_best_objective = self.best_fitness_list[-1]
        last_mean_objective = self.mean_fitness_list[-1]

        # Add text annotation for the last iteration's objective values
        fig_obj.add_annotation(
            x=len(self.best_fitness_list) - 1,
            y=last_best_objective,
            text=f'Best: {last_best_objective}<br>Mean: {last_mean_objective}',
            showarrow=True,
            arrowhead=1,
            ax=-10,
            ay=-40,
            bgcolor='white',
            bordercolor='black'
        )

        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Objective Distance over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Objective Distance',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='log',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the first plot
        fig_obj.show()

        # Create the second plot for Best and Mean Fitness values
        fig_fitness = go.Figure()

        # Add the best fitness trace
        fig_fitness.add_trace(go.Scatter(
            x=list(range(len(self.best_fitness_list))),
            y=self.best_fitness_list,
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean fitness trace
        fig_fitness.add_trace(go.Scatter(
            x=list(range(len(self.mean_fitness_list))),
            y=self.mean_fitness_list,
            mode='lines+markers',
            name='Mean Fitness',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        # Get the last iteration's best and mean fitness values
        last_best_fitness = self.best_fitness_list[-1]
        last_mean_fitness = self.mean_fitness_list[-1]

        # Add text annotation for the last iteration's fitness values
        fig_fitness.add_annotation(
            x=len(self.best_fitness_list) - 1,
            y=last_best_fitness,
            text=f'Best: {last_best_fitness}<br>Mean: {last_mean_fitness}',
            showarrow=True,
            arrowhead=1,
            ax=-10,
            ay=-40,
            bgcolor='white',
            bordercolor='black'
        )

        # Set the title and axis labels for the fitness plot
        fig_fitness.update_layout(
            title=f'Fitness over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Fitness',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='log',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the second plot
        fig_fitness.show()

        



        #Plotting unique solutions
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.num_unique_solutions_list))),
            y=self.num_unique_solutions_list,
            mode='lines+markers',
            name='Num Unique Solutions',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.num_repeated_solutions_list ))),
            y=self.num_repeated_solutions_list,
            mode='lines+markers',
            name=' Num Repeated Solutions',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

       
        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Number of unique solutions and repeated solution over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Number Unique and Repeated solutions',
            legend=dict(x=0, y=1),
            hovermode='x'
            
        )

        # Show the first plot
        fig_obj.show()


        #hamming distance:
         #Plotting unique solutions
        # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.hamming_distance_list))),
            y=self.hamming_distance_list,
            mode='lines+markers',
            name='Hamming distance',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        #Make me more traces for each self.hammind_distance_list
        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_crossover_list))),
            y=self.hamming_distance_crossover_list,
            mode='lines+markers',
            name='Hamming distance crossover',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_mutation1_list))),
            y=self.hamming_distance_mutation1_list,
            mode='lines+markers',
            name='Hamming distance mutation1',
            line=dict(color='green'),
            marker=dict(symbol='x')
        ))

        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_mutation2_list))),
            y=self.hamming_distance_mutation2_list,
            mode='lines+markers',
            name='Hamming distance mutation2',
            line=dict(color='red'),
            marker=dict(symbol='x')
        ))

        fig_obj.add_trace(go.Scatter
        (
            x=list(range(len(self.hamming_distance_local_search_list))),
            y=self.hamming_distance_local_search_list,
            mode='lines+markers',
            name='Hamming distance local search',
            line=dict(color='purple'),
            marker=dict(symbol='x')
        ))

    

        fig_obj.add_trace(go.Scatter
        (   
            x=list(range(len(self.hamming_distance_crossoverOld_list))),
            y=self.hamming_distance_crossoverOld_list,
            mode='lines+markers',
            name='Hamming distance crossoverOld',
            line=dict(color='pink'),
            marker=dict(symbol='x')
        ))  

       
       

       
        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Hamming distance over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Hamming distance',
            legend=dict(x=0, y=1),
            hovermode='x'
            
        )

        # Show the first plot
        fig_obj.show()
        

    def plot_timing_info(self):
        """
        Plot timing information for each stage of the genetic algorithm over iterations.
        """
        # Create a figure for the timing information
        fig_time = go.Figure()

        # Add a trace for each timing component
        timing_labels = [
            "Initialization", "Selection", "Crossover", "Mutation",
            "Elimination", "Mutation Population", "Local Search", "Total Iteration"
        ]
        timing_lists = [
            self.time_initialization_list, self.time_selection_list, self.time_crossover_list,
            self.time_mutation_list, self.time_elimination_list, self.time_mutation_population_list,
            self.time_local_search_list, self.time_iteration_list
        ]

        colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "black"]  # Assign colors to each process

        # Add each timing list to the plot
        for label, time_list, color in zip(timing_labels, timing_lists, colors):
            fig_time.add_trace(go.Scatter(
                x=list(range(len(time_list))),
                y=time_list,
                mode='lines+markers',
                name=label,
                line=dict(color=color),
                marker=dict(symbol='circle')
            ))

        # Add annotations for the last iteration's timing values
        for i, (label, time_list) in enumerate(zip(timing_labels, timing_lists)):
            if time_list:  # Ensure the list is not empty
                fig_time.add_annotation(
                    x=len(time_list) - 1,
                    y=time_list[-1],
                    text=f'{label}: {time_list[-1]:.2f}s',
                    showarrow=True,
                    arrowhead=2,
                    ax=-20 * (i + 1),  # Offset annotations for readability
                    ay=-40,
                    bgcolor='white',
                    bordercolor='black'
                )

        # Set the title and axis labels for the timing plot
        fig_time.update_layout(
            title='Timing Information Over Iterations',
            xaxis_title='Iterations',
            yaxis_title='Time (seconds)',
            legend=dict(x=1, y=1),
            hovermode='x',
        )

        # Show the timing plot
        fig_time.show()












    






#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
																			# Class Cities
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================
#==================================================================================================================================================================================


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns



class cities:
    def __init__(self,num_cities):
        np.random.seed(42)
        self.num_simulations_to_run = 2
        self.distanceMatrix = None
        self.num_cities = num_cities  
        self.cities = self.generate_cities()  
        self.clusters_list = None
        self.distance_matrix_cluster = []
        self.cities_cluster_list = []
        self.cities_sequence_list = []
        


    def print_model(self):
        '''
        - Print the model
        '''
        print("\n---------- Cities Generation: ------")
        print(f"   * Model Info:")
        print(f"       - Number of cities: {self.num_cities}")
    



    def generate_cities(self):
        '''
        - Generate the cities
        '''
        self.print_model()

        scale_factor = 100
        cities = np.random.rand(self.num_cities,2) * scale_factor
        return cities
    
    def generate_distance_matrix(self):
        '''
        - Generate the distance matrix
        '''
        distanceMatrix = np.zeros((self.num_cities,self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                distanceMatrix[i,j] = np.linalg.norm(self.cities[i]-self.cities[j])
        self.distanceMatrix = distanceMatrix
        return distanceMatrix

    def generate_distance_matrix_cluster(self):
        '''
        - Generate the distance matrix for the clusters
        '''
        print("\n---------- Generating Distance Matrix for Clusters: ------")
        #print(f"   * Model Info:")
        #print(f"       - Number of clusters: {len(self.clusters_list)}")
        #print(f"       - Clusters: {self.clusters_list}")
        #print(f"       - Cities [0]: {self.distanceMatrix.shape[0]}")
        #print(f"       - Cities [1]: {self.distanceMatrix.shape[1]}")
        #print(f"       - Cities: {self.cities}")
        #print(f"       - Distance Matrix: {self.distanceMatrix}")
        #print(f"       - Number of cities: {self.num_cities}")
        #print(f"       - Number of clusters: {len(self.clusters_list)}")
        #print(f"       - Clusters: {self.clusters_list}")

        assigned_cities_list = [cluster['assigned_cities'] for cluster in self.clusters_list]
        
        for cities in assigned_cities_list:
            self.cities_cluster_list.append(cities)
            num_cities = len(cities)
            distanceMatrix = np.zeros((num_cities,num_cities))
            #print(f"Number of cities: {num_cities}")
            #print(f"Cities: {cities}")
            row = 0
            for i in cities:
                
                column = 0
                for j in cities:
                    dist = self.distanceMatrix[i,j]
                    #print(f"Distance between city {i} and city {j}: {dist}")
                    distanceMatrix[row,column] = dist
                    column += 1
                row += 1
            #print(f"Distance matrix: {distanceMatrix}")
            self.distance_matrix_cluster.append(distanceMatrix)
        
     
        return self.distance_matrix_cluster, self.cities_cluster_list

    

    def add_clusters(self,clusters_list):
        '''
        - Add the clusters
        '''
        self.clusters_list = clusters_list

    def add_cities_sequence(self,cities_sequence_list):
        '''
        - Add the cities sequence
        '''
        self.cities_sequence_list = cities_sequence_list

    def plot_clusters(self):
        """
        Plot each cluster with a different color and highlight the medoid.
        """
        print("\n---------- Plotting Clusters: ------")
        
        # Choose a color palette with strong distinctions between colors
        num_clusters = len(self.clusters_list)
        print(f"Number of clusters: {num_clusters}")
        if num_clusters <= 10:
            colors = sns.color_palette("Dark2", num_clusters)  # 'tab10' has 10 highly distinct colors
        elif num_clusters <= 12:
            colors = sns.color_palette("Set1", num_clusters)  # 'Set1' has 9 distinct colors but works well for up to 12
        else:
            # For more than 12 clusters, we fall back on the HSV palette with high saturation
            colors = sns.color_palette("hsv", num_clusters)
        
        # Create a new figure
        plt.figure(figsize=(10, 8))
        
        # Loop through each cluster and plot the cities
        for i, cluster_info in enumerate(self.clusters_list):
            # Extract cluster details
            assigned_cities = cluster_info['assigned_cities']
            medoid_index = cluster_info['medoid']
            
            # Get the x and y coordinates of the assigned cities
            x_coords = self.cities[assigned_cities, 0]
            y_coords = self.cities[assigned_cities, 1]
            
            # Plot cities in the cluster with a unique color
            # I want large color distinction between clusters

            plt.scatter(x_coords, y_coords, label=f"Cluster {i}", color=colors[i], alpha=0.6)
            
            # Highlight the medoid in each cluster
            medoid_x = self.cities[medoid_index, 0]
            medoid_y = self.cities[medoid_index, 1]
            plt.scatter(medoid_x, medoid_y, color=colors[i], edgecolor="black", s=150, marker="X", label=f"Medoid {i}")

        # Add legend, title, and show plot
        plt.title("Clusters with Medoids Highlighted")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()
        
    
    def plot_cities(self):
        '''
        - Plot the cities
        '''
        print("\n---------- Plotting Cities: ------")
        '''
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.cities[:,0],y=self.cities[:,1],mode='markers'))
        fig.update_layout(title='Cities')
        fig.show()

        '''

        #make me the plot of teh cities using matplotlib
        #plot using seaborn

        sns.scatterplot(x=self.cities[:,0],y=self.cities[:,1])
        plt.title("Cities")
        plt.show()

    def show_clusters(self,clusters_list):
        '''
        - Show the clusters
        '''
        print("\n---------- Clusters: ------")
        self.add_clusters(clusters_list)
        self.plot_clusters()

    def plot_clusters_sequence(self, city_sequence=None):
        """
        Plot each cluster with a different color and highlight the medoid.
        Optionally, show the sequence of cities with arrows.
        Returns the figure object to allow storing and further use.
        """
        print("\n---------- Plotting Clusters: ------")
        
        # Choose a color palette with strong distinctions between colors
        num_clusters = len(self.clusters_list)
        if num_clusters <= 10:
            colors = sns.color_palette("Dark2", num_clusters)  
        elif num_clusters <= 12:
            colors = sns.color_palette("Set1", num_clusters)  
        else:
            colors = sns.color_palette("hsv", num_clusters)
        
        # Create a new figure
        plt.figure(figsize=(10, 8))
        
        # Loop through each cluster and plot the cities
        for i, cluster_info in enumerate(self.clusters_list):
            assigned_cities = cluster_info['assigned_cities']
            medoid_index = cluster_info['medoid']
            
            # Get the x and y coordinates of the assigned cities
            x_coords = self.cities[assigned_cities, 0]
            y_coords = self.cities[assigned_cities, 1]
            
            # Plot cities in the cluster with a unique color
            plt.scatter(x_coords, y_coords, label=f"Cluster {i}", color=colors[i], alpha=0.6)
            
            # Highlight the medoid in each cluster
            medoid_x = self.cities[medoid_index, 0]
            medoid_y = self.cities[medoid_index, 1]
            plt.scatter(medoid_x, medoid_y, color=colors[i], edgecolor="black", s=150, marker="X", label=f"Medoid {i}")
            
            # Add the number of cities in the cluster
            num_cities = len(assigned_cities)
            plt.text(medoid_x, medoid_y, f'{num_cities} cities', color=colors[i], ha='center', va='center', fontweight='bold')
        
        # Add labels to each city (by its index)
        for i, city in enumerate(self.cities):
            city_x, city_y = city
            plt.text(city_x, city_y, str(i), color='black', ha='center', va='center', fontweight='bold', fontsize=9)

        # If a city sequence is provided, plot arrows indicating the path
        if city_sequence:
            sequence = self.cities_sequence_list
            for j in range(len(sequence) - 1):
                start_city = sequence[j]
                end_city = sequence[j + 1]
                
                # Get start and end coordinates for the arrow
                start_x, start_y = self.cities[start_city]
                end_x, end_y = self.cities[end_city]
                
                # Plot an arrow from start to end city
                plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                        head_width=0.03, length_includes_head=True, color="black", alpha=0.8)
        
        # Add legend, title, and show plot
        plt.title("Clusters with Medoids and City Sequence")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()


