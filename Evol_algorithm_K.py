import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

class GA_K:

    def __init__(self,cities,seed=None ,mutation_prob = 0.001,elitism_percentage = 20):
        #model_key_parameters
        self.cities = cities 
        self.k_tournament_k = 3
        self.population_size = 0.0
        self.mutation_rate = mutation_prob
        self.elistism = 25                        #Elitism rate as a percentage


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
        

        self.weight_distance = 1
        self.weight_bdp = 1

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
        

        

    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf(distance_matrix=distance_matrix,replace_value=1e8)
        self.gen_size = len(distance_matrix)
        #self.population_size = 2*self.gen_size
        self.population_size = 1*self.gen_size
        self.k_tournament_k = int((3/100)*self.population_size)
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
        return self.mean_objective,self.best_objective,best_solution
    
    def print_best_solution(self):
        '''
        - Print the best solution
        '''
        
        print(f"\n Best solution is : {self.best_objective} \n Best solution cities are : {self.best_solution_cities}")
    
    



    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Run ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_model(self):
        self.set_initialization()
        #self.set_initialization_onlyValid_numpy(fitness_threshold=1e5)
        yourConvergenceTestsHere = False
        num_iterations = 1000
        iterations = 0
        while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
            '''
            meanObjective = 0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            '''
            iterations += 1
            print(f"\n Iteration number {iterations}")
            parents = self.selection_k_tournament(num_individuals=self.population_size)    
            offspring = self.crossover_singlepoint_population(parents)
            offspring_mutated = self.mutation_singlepoint_population(offspring)

            self.eliminate_population(population=self.population, offsprings=offspring_mutated)
            #self.elimnation_population_lamdaMu(population=self.population, offsprings=offspring_mutated)
            #self.eliminate_population_elitism(population=self.population, offsprings=offspring_mutated)
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
            self.check_insert_individual(num_iterations=100,threshold = 1000)
            yourConvergenceTestsHere = False

        self.print_best_solution()
        self.plot_fitness_dynamic()
        
        return 0


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 1) Initalization ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
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


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 2) Selection ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    def selection_k_tournament(self, num_individuals, k=3):
       #print(f"\n Population size: {self.population_size}")
        #print(f"\n K size: {k}")
        #print(f"\n Number of individuals: {num_individuals}")
        # Step 1: Randomly choose k individuals for each tournament
        tournament_indices = np.random.choice(self.population_size, size=(num_individuals, k), replace=True)
        #print(f"\n tournament indices: {tournament_indices}")

        # Step 2: Get the fitness scores of the selected individuals
        tournament_fitness = self.fitness[tournament_indices]
        #print(f"\n Fitness of the tournament is : {tournament_fitness}")    
        

        # Step 3: Get the index of the individual with the best fitness in each tournament
        best_indices_in_tournament = np.argmax(tournament_fitness, axis=1)

        # Step 4: Use these indices to select the best individuals from each tournament
        best_selected_indices = tournament_indices[np.arange(num_individuals), best_indices_in_tournament]

        # Step 5: Return the selected individualsi
        return self.population[best_selected_indices]
    
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
        
        # Perform mutation for each individual

        for i in range(num_individuals):
            if np.random.rand() < mutation_rate:
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

    


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 5) Elimnation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
       
        #print(f"\n Fitness V1--> {fitness_scores}")
       

        # Get the indices of the best individuals based on fitness
        
        best_indices = np.argsort(fitness_scores)[::-1][:self.population_size]
   


        # Select the best individuals
        self.population = combined_population[best_indices]
        self.fitness = fitness_scores[best_indices]
        self.distance_scores = distances_scores[best_indices]
        self.average_bpd_scores = averageBdp_scores[best_indices]

        
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
        
        
    def check_insert_individual(self,num_iterations=100,threshold = 1000):
        '''
        - Insert the individual in the population
        '''
        # Convert lists to numpy arrays for faster indexing and operations
        best_fitness = np.array(self.best_fitness_list)
        mean_fitness = np.array(self.mean_fitness_list)
        
        # Ensure we're checking only the last `num_iterations` entries
        if len(best_fitness) < num_iterations or len(mean_fitness) < num_iterations:
            #print("Not enough data to perform the check.")
            return

        # Slice the last `num_iterations` values from both arrays
        best_fitness_last = best_fitness[-num_iterations:]
        mean_fitness_last = mean_fitness[-num_iterations:]

        # Use boolean indexing to find where mean fitness is less than threshold * best fitness
        diff_fitness = mean_fitness_last - best_fitness_last

        print(f"Best fitness: {best_fitness_last[-1]} vs Mean fitness: {mean_fitness_last[-1]} witha diff of {diff_fitness[-1]}")
        condition_met = diff_fitness <= (threshold)
        
        # If the condition is met for all iterations, add new individuals
        if np.all(condition_met):
            print(f"Condition met for {num_iterations} iterations. with threshold {threshold}") 
            print(f"Best fitness: {best_fitness_last} vs Mean fitness: {mean_fitness_last} witha diff of {mean_fitness_last - best_fitness_last}")
            
            
        
        




    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 6) Fitness Calculation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        avg_bpd = self.average_bpd(population)
        
        # Step 3: Calculate the fitness using both distance and BPD
        fitness =  ( 1 / (weight_distance * distance_fitness) + weight_bdp * avg_bpd)

        
        
        return fitness, distance_fitness, avg_bpd



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

         # Create the first plot for Best and Mean Objective values
        fig_obj = go.Figure()

        # Add the best objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.best_average_bdp_scores_list))),
            y=self.best_average_bdp_scores_list,
            mode='lines+markers',
            name='Best BPD',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # Add the mean objective trace
        fig_obj.add_trace(go.Scatter(
            x=list(range(len(self.mean_average_bdp_scores_list))),
            y=self.mean_average_bdp_scores_list,
            mode='lines+markers',
            name='Mean BPD',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

     


        # Set the title and axis labels for the objective plot
        fig_obj.update_layout(
            title=f'Objective Distance over Iterations with mutation rate {self.mutation_rate*100} %',
            xaxis_title='Iterations',
            yaxis_title='BPD (Diversity)',
            legend=dict(x=0, y=1),
            hovermode='x',
            yaxis=dict(
                type='log',  # Set Y-axis to logarithmic scale
                autorange=True  # Ensure the axis is adjusted automatically
            )
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

        






    
    


        

      



    
    

    











        




    
