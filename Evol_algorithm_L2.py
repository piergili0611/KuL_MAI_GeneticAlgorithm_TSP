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

        #print(f"\n Best solution cities are : {self.best_solution_cities}")

    def print_model_info(self):
        print("\n------------- GA_Level2: -------------")
        print(f"   * Model Info:")
        print(f"       - Population Size: {self.population_size}")
        print(f"       - Number of cities: {self.gen_size}")
        print(f"       - Number of clusters: {self.num_clusters}")
        print(f"       - Cluster Soluitions: {self.clusters_solution_matrix}")
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
            self.population_size = 50
        else:
            self.population_size = 15
        
        
        self.k_tournament_k = int((3/100)*self.population_size)
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
        time_end = time.time()
        intialization_time = time_end - time_start 
        yourConvergenceTestsHere = False
        num_iterations = 300
        iterations = 0
        while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
            '''
            meanObjective = 0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            '''
            time_start_iteration = time.time()
            iterations += 1
            print(f"\n Iteration number {iterations}")
            time_start = time.time()
            parents_all_list = self.selection_k_tournament(num_individuals=self.population_size)    
            time_end = time.time()
            time_selection = time_end - time_start

            time_start = time.time()
            offspring_all_list = self.crossover_singlepoint_population(parents_all_list)
            time_end = time.time()
            time_crossover = time_end - time_start
        
            time_start = time.time()
            mutated_all_list = self.mutation_singlepoint_population(offspring_all_list)
            time_end = time.time()
            time_mutation = time_end - time_start
            
            time_start = time.time()
            self.population_all_list = self.mutation_singlepoint_population(self.population_all_list)    
            time_end = time.time()
            time_mutation_population = time_end - time_start

            if self.local_search:
                time_start = time.time()
                if iterations % 2 == 0: 
                    mutated_all_list = self.combine_all_lists(all_list1=self.population_all_list,all_list2=mutated_all_list)
                    #mutated_all_list = self.local_search_population(population_all_list=mutated_all_list)
                time_end = time.time()
                time_local_search = time_end - time_start
            else:
                time_local_search = 0

            time_start = time.time()
            self.eliminate_population(population_all_list=self.population_all_list, mutated_all_list=mutated_all_list)
            #self.eliminate_population_elitism(population=self.population, offsprings=offspring_mutated)
            time_end = time.time()
            time_elimination = time_end - time_start
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
            yourConvergenceTestsHere = False
            time_end_iteration = time.time()
            diff_time_iteration = time_end_iteration - time_start_iteration
            self.update_time(time_initalization=intialization_time,time_selection=time_selection,time_crossover=time_crossover,time_mutation=time_mutation,time_elimination=time_elimination,time_mutation_population=time_mutation_population,time_local_search=time_local_search,time_iteration=diff_time_iteration)
        self.print_best_solution()
        self.plot_fitness_dynamic()
        self.plot_timing_info()
        
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
        #self.population_merged = self.greedyMerge_populationClusters(self.population_cluster)
        #print(f"\nMerged Population: {self.population_merged}")
    
        
        # 5) Calculate fitness for the merged population
        self.fitness = self.calculate_fitness(self.population_cities)
        #print(f"\nInitial Fitness: {self.fitness}")
    
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
        '''
        Optimized 2-opt with reduced neighborhood search: focuses on the most promising swaps
        '''
        best_route = np.copy(route)
        best_distance = self.calculate_total_distance_individual(best_route, distance_matrix)
        n = len(route)
        
        improvement = True
        iteration = 0
        
        while improvement and iteration < max_iterations:
            improvement = False
            
            # Generate all pairs of indices i, j (i < j)
            i_indices, j_indices = np.triu_indices(n, k=2)
            
            # Calculate the distance changes for all (i, j) pairs in parallel
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
            
            # Identify the top k pairs with the largest improvement (most negative delta_distances)
            top_k_indices = np.argsort(delta_distances)[:k_neighbors]
            
            # If there are any improving swaps in the top k, apply the best one
            if np.any(delta_distances[top_k_indices] < 0):
                improvement = True
                best_swap_index = top_k_indices[np.argmin(delta_distances[top_k_indices])]
                i, j = i_indices[best_swap_index], j_indices[best_swap_index]
                
                # Perform the 2-opt swap: reverse the segment between i and j
                best_route[i + 1 : j + 1] = best_route[i + 1 : j + 1][::-1]
                best_distance += delta_distances[best_swap_index]
                #print(f"\n Best distance is : {best_distance}")
            
            # Stop if no improvement or improvement is very small
            if not improvement or np.min(delta_distances[top_k_indices]) > min_improvement_threshold:
                break
            
            iteration += 1

        return best_route

   
    def local_search_population(self, population_all_list, max_iterations=10):
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
        n_best = 2  # Or set to the desired number of top individuals
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



    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 8) Plotting------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
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












    
