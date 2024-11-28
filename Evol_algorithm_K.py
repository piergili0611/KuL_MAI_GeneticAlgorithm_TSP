import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time
from numba import jit,njit,prange
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import heapq

class GA_K:

    def __init__(self,cities,seed=None ,mutation_prob = 0.001,elitism_percentage = 20,local_search=True,max_iterations = 50):
        #model_key_parameters
        self.cities = cities 
        self.k_tournament_k = 3
        self.population_size = 0.0
        #self.mutation_rate = mutation_prob
        self.mutation_rate = 0.5
        self.elistism = 1

        self.max_iterations = max_iterations       
        
        # Fitness Sharing
        self.sigma = 0.9
        self.alpha = 0.1


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
        
        

        self.weight_distance = 1
        self.weight_bdp = 0
        
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


        # Initial Solution
        self.initial_solution = None

        #Random seed
        if seed is not None:
            np.random.seed(seed)    
        

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
        print(f"       - Cities: {self.cities}")
        print(f"       - Distance Matrix: {self.distance_matrix}")
        print(f"   * Model Parameters:")
        print(f"       - K: {self.k_tournament_k}")
        print(f"       - Mutation rate: {self.mutation_rate}")
        print(f"       - Elitism percentage: {self.elistism} %")
        print(f"   * Running model:")
        print(f"       - Local search: {self.local_search}")
        print(f"       - Initial Fitness: {self.fitness}")
        

        

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
       

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Run ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_model(self,plot = False):
        time_start = time.time()
        #self.set_initialization()
        self.set_initialization_onlyValid_numpy_incremental(fitness_threshold=1e5)
        time_end = time.time()
        initialization_time = time_end - time_start 
        
        #self.set_initialization_onlyValid_numpy(fitness_threshold=1e5)
        yourConvergenceTestsHere = False
       
        #num_iterations = 200
        num_iterations = self.max_iterations
        iterations = 0
        while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
            '''
            meanObjective = 0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            '''
          
                
                
            time_start_iteration = time.time()
            iterations += 1
            self.iteration = iterations
            print(f"\n Iteration number {iterations}")
            
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
            offspring_mutated = self.mutation_singlepoint_population(offspring1)
            time_end = time.time()
            time_mutation = time_end - time_start
            self.calculate_add_hamming_distance(population=offspring_mutated,mutation1=True)
            


            #Mutate also population
            time_start = time.time()
            #self.population = self.mutation_singlepoint_population(self.population)
            time_end = time.time()
            time_mutation_population = time_end - time_start
            self.calculate_add_hamming_distance(population=self.population,mutation2=True)
            print(f"  Local search of teh population")
            self.population = self.local_search_population_2opt_multip(self.population,n_best = 2,max_iterations=100)

            if self.local_search:
                time_start = time.time()
                #offspring_mutated = np.vstack((offspring_mutated,self.population))
                
                
                offspring_mutated= self.local_search_population_2opt_multip(offspring_mutated,n_best = 2,max_iterations=20)
                
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
            self.eliminate_population_fs_tournament(population=self.population, offsprings=offspring_mutated, sigma=0.1, alpha=0.1, k=3)
            time_end = time.time()
            time_elimination = time_end - time_start
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
       
            
            yourConvergenceTestsHere = False
            time_end_iteration = time.time()
            diff_time_iteration = time_end_iteration - time_start_iteration
            self.update_time(time_initalization=initialization_time,time_selection=selection_time,time_crossover=time_crossover,time_mutation=time_mutation,time_elimination=time_elimination,time_mutation_population=time_mutation_population,time_local_search=time_local_search,time_iteration=diff_time_iteration)

        if plot is True:
            self.plot_fitness_dynamic()
            self.plot_timing_info()
        self.print_best_solution()
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
            print(f"\nInitial solution added to the population: {population} with fitness {fitness}")
            current_index = 1
        else:
            current_index = 0

        while current_index < population_size:
            # Initialize a new individual incrementally
            print(f"\nPopulation progress: {current_index}")
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
                    #best_city_idx = np.argmax(valid_distances)
                    best_city_idx = random_index
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
                print(f"CUrrent index is : {current_index}")
                population[current_index] = route
                fitness[current_index] = current_distance
                current_index += 1

            # Optional status print (can be removed for better performance)
            if current_index % 100 == 0:
                print(f"Population progress: {current_index}/{population_size}")

        # Ensure the population has the correct size and each solution has gen_size cities
        assert population.shape[0] == self.population_size, f"Population size mismatch: {population.shape[0]} != {self.population_size}"
        assert population.shape[1] == self.gen_size, f"Each solution should have {self.gen_size} cities, but has {population.shape[1]} cities."

        # Final population setup
        self.population = population
        self.fitness = fitness

        # Update internal distance matrix (replace any inf values with the specified value)
        self.distance_matrix = self.check_inf(self.distance_matrix, replace_value=1e8)
        
        # Recalculate the fitness for the entire population
        self.fitness = self.calculate_distance_population(self.population)
        
        # Print final status
        print(f"\nInitial Population shape: {self.population.shape}")
        print(f"\nInitial population: {self.population}")
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
            np.random.shuffle(remaining_cities)

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




    def mutation_singlepoint_population(self, population):
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
                mutated_population[i] = self.mutation_singlepoint(population[i], mutation_rate)
             
        
              

        #print(f"\n MUTATION: Children Population is : {mutated_population[1]}" )
        
        return mutated_population

    def mutation_singlepoint(self, individual, mutation_rate):
        mutated_individual = np.copy(individual)
        num_genes = len(mutated_individual)

        # Ensure valid mutation range
        max_mutations = max(1, num_genes)  # Ensure at least 1 possible mutation
        num_mutations1 = np.random.randint(1, (num_genes-1)/2)

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
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) Local Search ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Optimized Local Search Function

    # 0) Numba: Jit compilation
    def local_search_population_jit(self, population, max_iterations=10, k_neighbors=10, min_improvement_threshold=100):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals
        '''
        distance_matrix = self.distance_matrix
        
        # Step 1: Evaluate fitness for all individuals in the population
        distances = self.calculate_distance_population(population)
        
        # Step 2: Select the top `n_best` individuals
        n_best = 10  # Or set to the desired number of top individuals
        best_indices = np.argsort(distances)[:n_best]

        for i in best_indices:
            population[i] = local_search_operator_2_opt(distance_matrix, population[i])
        return population
   
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
    def local_search_population_2opt(self, population,n_best=2, max_iterations=10):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals
        '''
        distance_matrix = self.distance_matrix
        
        # Step 1: Evaluate fitness for all individuals in the population
        distances = self.calculate_distance_population(population)
        
        # Step 2: Select the best individual explicitly
        # Or set to the desired number of top individuals
        best_indices = np.argsort(distances)[:n_best]

   
        # Step 3: Apply 2-opt to the selected top individuals
        for i in best_indices:
            #print(f"\n Individual {i}")
            time_start = time.time()
            #population[i] = self.two_opt_no_loops_opt_out(population[i], distance_matrix, max_iterations,k_neighbors=10)
            population[i] = self.two_opt_better1_n(population[i], distance_matrix, max_iterations)
          
            time_end = time.time()
            time_local_search_iteration = time_end - time_start
            print(f"\n Time for the local search iteration: {time_local_search_iteration}")
            
        
        return population
    
    def two_opt_no_loops_opt_out(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
        #print(f"\n ------------LOCAL SEARCH------------")
        best_route = np.copy(route)
        initial_route = np.copy(route)
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
                
                # Perform the 2-opt swap: reverse the segment between i and j
               
                best_route[i + 1 : j + 1] = best_route[i + 1 : j + 1][::-1]
                routes.append(best_route)

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
             
            
            iteration += 1
        if len(routes) > 0:
            routes_np = np.array(routes)
            final_fitness = self.calculate_distance_population(routes_np)
            best_sol = routes_np[np.argmin(final_fitness)]
            best_fit = final_fitness[np.argmin(final_fitness)]
            print(f"\n Initial Fitness: {inital_fitness} - Final Fitness: {best_fit} ")

            if inital_fitness > best_fit:
                best_route = best_sol
            else:
                best_route = route
        else:
            best_route = route

        print(f"\n    LS Iterations: {iteration}")   
        return best_route
    
    def two_opt_better1(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
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
            time_start = time.time()
            top_k_indices = heapq.nsmallest(k_neighbors, range(len(delta_distances)), key=lambda x: delta_distances[x])
            #top_k_indices = np.argsort(delta_distances)[:k_neighbors]
            time_end = time.time()
            time_sorting = time_end - time_start
            print(f"\n Time for sorting: {time_sorting}")
            
            if np.any(delta_distances[top_k_indices] < 0):
                #print(f"\n There is an improvement")
                print(f"\n Iteration number {iteration}")
                
        
               
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
            
        return best_route
    
   
    
        iteration += 1
        print(f"\n    LS Iterations: {iteration}")
        return best_route
    

    # 2) MultiProcessors: threads
    def local_search_population_2opt_multip(self, population,n_best=5, max_iterations=10):
        '''
        Optimized local search for the population: applies 2-opt to the top individuals using threading.
        '''
        distance_matrix = self.distance_matrix

        # Step 1: Evaluate fitness for all individuals in the population
        distances = self.calculate_distance_population(population)

        # Step 2: Select the best individual explicitly
      
        best_indices = np.argsort(distances)[:n_best]
      

        # Step 3: Apply 2-opt to the selected top individuals in parallel using threads
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.local_search_for_individual, population, i, distance_matrix, max_iterations) for i in best_indices]
            
            for future in futures:
                future.result()  # Wait for the threads to finish

        return population
    
  
    def local_search_for_individual(self,population, i, distance_matrix, max_iterations=10):
        #fitness_original = self.calculate_total_distance_individual(population[i], distance_matrix) 
        #population[i] = self.two_opt_no_loops_opt(population[i], distance_matrix, max_iterations)
        population[i] = self.two_opt_no_loops_opt_out(population[i], distance_matrix, max_iterations)
        #population[i] = self.two_opt_better1_n(population[i], distance_matrix, max_iterations)
        #fitness_after = self.calculate_total_distance_individual(population[i], distance_matrix)
        #print(f"\n Fitness before: {fitness_original} - Fitness after: {fitness_after}")

        
        return population[i]


    
    

    # 3) 3-opt   
    def three_opt_no_loops_opt(self, route, distance_matrix, max_iterations=10, min_improvement_threshold=100, k_neighbors=10):
        '''
        Optimized 3-opt with reduced neighborhood search: focuses on the most promising swaps
        '''
        best_route = np.copy(route)
        best_distance = self.calculate_total_distance_individual(best_route, distance_matrix)
        n = len(route)
        
        improvement = True
        iteration = 0
        
        while improvement and iteration < max_iterations:
            improvement = False
            
            # Generate all triplets of indices (i < j < k)
            i_indices, j_indices, k_indices = zip(*combinations(range(n), 3))
            
            # Convert to numpy arrays for element-wise operations
            i_indices = np.array(i_indices)
            j_indices = np.array(j_indices)
            k_indices = np.array(k_indices)
            
            # Calculate the distance changes for all (i, j, k) triplets in parallel
            i_next = (i_indices + 1) % n
            j_next = (j_indices + 1) % n
            k_next = (k_indices + 1) % n
            
            old_distances = (
                distance_matrix[best_route[i_indices], best_route[i_next]] +
                distance_matrix[best_route[j_indices], best_route[j_next]] +
                distance_matrix[best_route[k_indices], best_route[k_next]]
            )
            
            # We now test all 3 possible ways of reconnecting the triplet of indices (i, j, k)
            new_distances_1 = (
                distance_matrix[best_route[i_indices], best_route[j_indices]] +
                distance_matrix[best_route[i_next], best_route[k_indices]] +
                distance_matrix[best_route[j_next], best_route[k_next]]
            )
            
            new_distances_2 = (
                distance_matrix[best_route[i_indices], best_route[k_indices]] +
                distance_matrix[best_route[j_indices], best_route[i_next]] +
                distance_matrix[best_route[j_next], best_route[k_next]]
            )
            
            new_distances_3 = (
                distance_matrix[best_route[i_indices], best_route[k_indices]] +
                distance_matrix[best_route[j_indices], best_route[i_next]] +
                distance_matrix[best_route[i_next], best_route[j_next]]
            )
            
            # Calculate the distance changes for each new configuration
            delta_distances = np.minimum(new_distances_1, np.minimum(new_distances_2, new_distances_3)) - old_distances
            
            # Identify the top k triplets with the largest improvement (most negative delta_distances)
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]
            
            # If there are any improving swaps in the top k, apply the best one
            if np.any(delta_distances[top_k_indices] < 0):
                improvement = True
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
                i, j, k = i_indices[best_swap_index], j_indices[best_swap_index], k_indices[best_swap_index]
                
                # Perform the 3-opt swap: apply the best of the 3 reconnections
                if delta_distances[best_swap_index] == new_distances_1[best_swap_index] - old_distances[best_swap_index]:
                    best_route[j+1:k+1] = best_route[j+1:k+1][::-1]
                elif delta_distances[best_swap_index] == new_distances_2[best_swap_index] - old_distances[best_swap_index]:
                    best_route[i+1:j+1] = best_route[i+1:j+1][::-1]
                    best_route[j+1:k+1] = best_route[j+1:k+1][::-1]
                else:
                    best_route[i+1:k+1] = best_route[i+1:k+1][::-1]
                
                best_distance += delta_distances[best_swap_index]
            
            # Stop if no improvement or improvement is very small
            if not improvement or np.min(delta_distances[top_k_indices]) > min_improvement_threshold:
                break
            
            iteration += 1

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
        '''
        - Elimination with fitness sharing for TSP using k-tournament selection, ensuring the best individual is always selected,
        and no duplicates are selected during the process.
        '''
        # 1) Combine population & calculate their fitness
        combined_population = np.vstack((population, offsprings))
        
        combined_fitness, _, _ = self.fitness_function_calculation(
            population=combined_population, 
            weight_distance=self.weight_distance, 
            weight_bdp=self.weight_bdp, 
            distance_matrix=self.distance_matrix
        )
        
        # 2) Initialize survivors
        survivors_idxs = -1 * np.ones(self.population_size, dtype=int)  # Initialize survivors
        
        # Select the best individual (lowest fitness) and store in the first slot
        best_index = np.argmin(combined_fitness)
        survivors_idxs[0] = best_index

        # Exclude the best individual from valid candidates
        valid_candidates = np.setdiff1d(np.arange(len(combined_population)), [best_index])

        # 3) Perform k-tournament selection for the remaining survivors
        for i in range(1, self.population_size):
            #print(f"Selecting survivor {i + 1}/{self.population_size}")
            
            # Compute fitness sharing for the remaining individuals
            new_fitness = self.fitness_sharing_individual_np(
                population=combined_population, 
                survivors=survivors_idxs[:i],  # Pass only the selected survivors so far
                population_fitness=combined_fitness, 
                sigma=sigma, 
                alpha=alpha
            )
            
            # Randomly select k candidates from the valid candidates
            if len(valid_candidates) > 0:  # Ensure we have valid candidates
                tournament_candidates = np.random.choice(valid_candidates, size=min(k, len(valid_candidates)), replace=False)
            else:
                raise ValueError("No valid candidates remain for selection.")
            
            # Select the individual with the best fitness from the tournament
            best_in_tournament = tournament_candidates[np.argmin(new_fitness[tournament_candidates])]
            
            # Add the best candidate to survivors
            survivors_idxs[i] = best_in_tournament
            
            # Remove the selected candidate from valid candidates
            valid_candidates = valid_candidates[valid_candidates != best_in_tournament]
            
            # If valid_candidates is empty before filling the population, break early (unlikely edge case)
            if len(valid_candidates) == 0 and i < self.population_size - 1:
                print("Warning: Not enough unique individuals to fill the population.")
                break
        
        # 4) Select the best individuals from the combined population
        self.population = combined_population[survivors_idxs]
        self.fitness = combined_fitness[survivors_idxs]
        self.distance_scores = self.calculate_distance_population(self.population)
        self.average_bpd_scores = self.average_bpd(self.population)
        self.hamming_distance, _ = self.calculate_hamming_distance_population(self.population)

    def eliminate_population_fs(self, population, offsprings, sigma, alpha):
        '''
        - Elimination with fitness sharing for TSP
        '''
        # 1) Combine population & calculate their fitness
        combined_population = np.vstack((population, offsprings))
        

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
            print(f"Counter: {counter}")
            # Compute fitness sharing for the remaining individuals
            new_fitness = self.fitness_sharing_individual_np(population=combined_population, survivors=survivors_idxs, 
                                                        population_fitness=combined_fitness, sigma=sigma, alpha=alpha)

            # Get the index of the next best individual
            best_index = np.argmin(new_fitness)
            survivors_idxs[counter] = best_index  # Add this individual to the survivors
            counter += 1

        # 4) Select the best individuals from the combined population
        self.population = combined_population[survivors_idxs]
        self.fitness = combined_fitness[survivors_idxs]
        self.distance_scores = self.calculate_distance_population(self.population)
        self.average_bpd_scores = self.average_bpd(self.population)
        self.hamming_distance,_ = self.calculate_hamming_distance_population(self.population)
        

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



    def caclulate_numberRepeatedSolution(self,population):
        '''
        - Calculate the number of repeated solutions in the population
        '''
        self.num_unique_solutions = len(np.unique(population,axis=0))
        self.num_repeated_solutions = len(population) - self.num_unique_solutions
        

    def check_stopping_criteria(self):
        '''
        - Check the stopping criteria
        '''
        
        if round(self.best_objective) == round(self.mean_objective):
            return True
        else:
            return False


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


    '''
    def plot_fitness_dynamic(self):
        # Create a plotly figure
        fig = go.Figure()

        # Add the best fitness trace
        fig.add_trace(go.Scatter(
            x=list(range(len(self.best_fitness_list))),
            y=self.best_fitness_list,
            mode='lines+markers',
            name='Best Distance',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean fitness trace
        fig.add_trace(go.Scatter(
            x=list(range(len(self.mean_fitness_list))),
            y=self.mean_fitness_list,
            mode='lines+markers',
            name='Mean Distance',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        # Get the last iteration's best and mean fitness
        last_best_fitness = self.best_fitness_list[-1]
        last_mean_fitness = self.mean_fitness_list[-1]

        # Add text annotation for the last iteration's fitness
        fig.add_annotation(
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

        # Set the title and axis labels
        fig.update_layout(
            title=f'Distance over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='Distance',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
            type='log',  # Set Y-axis to logarithmic scale
            autorange=True  # Ensure the axis is adjusted automatically
            )
        )

        # Show the plot
        fig.show()

    '''
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
@njit(nopython=True)
def perform_local_search_population_jit(population,distance_matrix,best_indices ,max_iterations=10, k_neighbors=10, min_improvement_threshold=100):
    '''
    Optimized local search for the population: applies 2-opt to the top individuals
    '''
    
    # Step 3: Apply 2-opt to the selected top individuals
    num_localsearch = len(best_indices)
    for i in range(num_localsearch):
        population[best_indices[i]] = two_opt_no_loops_optimized_jit(population[i], distance_matrix, max_iterations, k_neighbors, min_improvement_threshold)
    
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



@jit(nopython=True)
def two_opt_no_loops_optimized_jit(route, distance_matrix, max_iterations=10, k_neighbors=10, min_improvement_threshold=100):
    '''
    Optimized 2-opt with Numba for JIT compilation. Avoids unsupported advanced indexing.
    '''
    best_route = np.copy(route)
    best_distance = calculate_total_distance_individual_jit(best_route, distance_matrix)
    n = len(route)

    improvement = True
    iteration = 0

    while improvement and iteration < max_iterations:
        improvement = False

        # Generate all pairs of indices i, j (i < j)
        i_indices, j_indices = np.triu_indices(n, k=2)

        # Prepare arrays to store distance changes and candidate swaps
        delta_distances = np.empty(len(i_indices), dtype=np.float64)

        # Calculate delta distances for all (i, j) pairs using scalar indexing
        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx]
            i_next = (i + 1) % n
            j_next = (j + 1) % n

            # Calculate old and new distances for this swap
            old_dist = (
                distance_matrix[best_route[i], best_route[i_next]] +
                distance_matrix[best_route[j], best_route[j_next]]
            )
            new_dist = (
                distance_matrix[best_route[i], best_route[j]] +
                distance_matrix[best_route[i_next], best_route[j_next]]
            )

            delta_distances[idx] = new_dist - old_dist

        # Find the top k_neighbors swaps with the largest improvements
        top_k_indices = np.argsort(delta_distances)[:k_neighbors]

        # Check if any swap provides an improvement
        if np.any(delta_distances[top_k_indices] < 0):
            improvement = True
            best_swap_idx = top_k_indices[np.argmin(delta_distances[top_k_indices])]
            i, j = i_indices[best_swap_idx], j_indices[best_swap_idx]

            # Perform the 2-opt swap: reverse the segment between i and j
            best_route[i + 1 : j + 1] = best_route[i + 1 : j + 1][::-1]
            best_distance += delta_distances[best_swap_idx]

        # Stop if no improvement or improvement is below the threshold
        if not improvement or np.min(-1*delta_distances[top_k_indices]) < min_improvement_threshold:
            break

        iteration += 1

    return best_route


    


        
# Fitness function (assuming it calculates the total fitness of the solution)
@jit(nopython=True)
def fitness(distanceMatrix: np.ndarray, order: np.ndarray) -> float:
    total_distance = 0.0
    for i in range(len(order) - 1):
        total_distance += distanceMatrix[order[i], order[i + 1]]
    total_distance += distanceMatrix[order[-1], order[0]]  # Return to start
    return total_distance

# Function to calculate cumulative distances (these are precomputed for efficiency)
@jit(nopython=True)
def build_cumulatives(distanceMatrix: np.ndarray, order: np.ndarray, length: int) -> tuple:
    # Cumulative distance from 0 to each node (excluding the last one)
    cum_from_0_to_first = np.zeros(length)
    for i in range(1, length):
        cum_from_0_to_first[i] = cum_from_0_to_first[i-1] + distanceMatrix[order[i-1], order[i]]
    
    # Cumulative distance from each node to the last node
    cum_from_second_to_end = np.zeros(length)
    for i in range(length-2, -1, -1):
        cum_from_second_to_end[i] = cum_from_second_to_end[i+1] + distanceMatrix[order[i], order[i+1]]
    
    return cum_from_0_to_first, cum_from_second_to_end

# Partial fitness calculation for one value (this computes the fitness between two nodes)
@jit(nopython=True)
def partial_fitness_one_value(distanceMatrix: np.ndarray, frm: int, to: int) -> float:
    return distanceMatrix[frm, to]

# In-place 2-opt local search operator
@jit(nopython=True)
def local_search_operator_2_opt(distanceMatrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)
    best_combination = (0, 0)

    # Build cumulative arrays
    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)

    # Try swapping edges
    for first in range(1, length - 2):
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
            new_fitness = temp + fit_last_part + bridge_first + bridge_second
            
            if new_fitness < best_fitness:
                best_combination = (first, second)
                best_fitness = new_fitness

    best_first, best_second = best_combination
    if best_first == 0:  # No improvement found
        return order  # Return the original order if no better solution is found
    
    # Perform the 2-opt swap in-place
    order[best_first:best_second] = order[best_first:best_second][::-1]
    return order




    
    

    











        




    
