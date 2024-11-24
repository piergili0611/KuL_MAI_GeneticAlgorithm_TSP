import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time

from numba import njit, prange

class GA_K:

    def __init__(self,cities,seed=None ,mutation_prob = 0.001,elitism_percentage = 20,local_search = True):
        #model_key_parameters
        self.cities = cities 
        self.k_tournament_k = 3
        self.population_size = 0.0
        self.mutation_rate = mutation_prob
        self.elistism = 100                        #Elitism rate as a percentage


        self.mean_objective = 0.0
        self.best_objective = 0.0
        self.mean_fitness_list = []
        self.best_fitness_list = [] 

        self.local_search = local_search
        

        #Random seed
        if seed is not None:
            np.random.seed(seed)    
        

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 0) Set Up: Distance Matrix------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf(distance_matrix=distance_matrix,replace_value=1e8)
        self.gen_size = len(distance_matrix)
        #self.population_size = 2*self.gen_size
        self.population_size = 2*self.gen_size
        self.k_tournament_k = int((3/100)*self.population_size)
        #print(f"Distance matrix is {self.distance_matrix}")
        #print(f"Gen size is {self.gen_size}")
    

    def check_inf(self,distance_matrix,replace_value):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''
        distance_matrix[distance_matrix == np.inf] = replace_value
        return distance_matrix
    
    def set_inf(self,distance_matrix,replace_value):
        '''
        - Set the inf values in the distance matrix
        '''
        distance_matrix[distance_matrix == replace_value] = np.inf
        return distance_matrix
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 1) Run Algorithm ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def run_model(self):
        #self.set_initialization()
        
        start_time = time.time()
        self.distance_matrix = self.set_inf(self.distance_matrix,replace_value=1e8)
        self.population, self.fitness = set_initialization_onlyValid_numpy(population_size=self.population_size, gen_size=self.gen_size, distance_matrix=self.distance_matrix, fitness_threshold=1e8)
        self.distance_matrix = self.check_inf(self.distance_matrix,replace_value=1e8)
        end_time = time.time()
        print(f"\n Time to initialize the population is : {end_time - start_time}")


        self.print_model_info()
        #self.population, self.fitness = generate_valid_population(population_size=self.population_size, gen_size=self.gen_size, distance_matrix=self.distance_matrix, fitness_threshold=1e4)
        yourConvergenceTestsHere = False
        num_iterations = 2000
        iterations = 0
        while( (yourConvergenceTestsHere is False) and iterations < num_iterations):
            '''
            meanObjective = 0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            '''
            iterations += 1
            #print(f"\n Iteration number {iterations}")
            time_start = time.time()
            parents = self.selection_k_tournament(num_individuals=self.population_size)    
            time_end = time.time()
            print(f"\n Time to select the parents is : {time_end - time_start}")

            time_start = time.time()
            offspring = crossover_singlepoint_population(population=parents,gen_size=self.gen_size)
            time_end = time.time()
            print(f"\n Time to perform the crossover is : {time_end - time_start}")

            time_start = time.time()
            offspring_mutated = mutation_singlepoint_population(population=offspring,mutation_rate=self.mutation_rate)
            time_end = time.time()
            print(f"\n Time to perform the mutation is : {time_end - time_start}")

            if self.local_search:
                time_start = time.time()
                offspring_mutated_localSearch =local_search_population(population=offspring_mutated,number_individuals= len(offspring_mutated), distance_matrix=self.distance_matrix)  
                time_end = time.time()
                time_local_search = time_end - time_start
                print(f"\n Time to perform the local search is : {time_local_search}")
            else:
                offspring_mutated_localSearch = offspring_mutated

            time_start = time.time()
            self.population, self.fitness = eliminate_population(population=self.population, offsprings=offspring_mutated_localSearch,population_fitness=self.fitness,population_size=self.population_size,distance_matrix=self.distance_matrix)
            time_end = time.time()
            print(f"\n Time to eliminate the population is : {time_end - time_start}")
            #self.eliminate_population_elitism(population=self.population, offsprings=offspring_mutated_localSearch)
            #self.eliminate_population_fs(population=self.population, offsprings=offspring_mutated_localSearch)
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
            yourConvergenceTestsHere = False

        self.print_best_solution()
        self.plot_fitness_dynamic()
        
        return 0
    

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 2) Initialization ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    def set_initialization(self):
        '''
        - Initialize the population
        '''

        self.population = np.array([np.random.permutation(self.gen_size) for _ in range(self.population_size)])
        #self.population = self.local_search_population(population=population,number_individuals= len(population))
        self.fitness = calculate_fitness(self.population,distance_matrix=self.distance_matrix)
        print(f"\n Initial Population: {self.population}")
        print(f"\n Initial Fitness: {self.fitness}")
        self.print_model_info()

    

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 3) Selection: K_tournament ------------------------------------------------------------------------------------------------
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
        best_indices_in_tournament = np.argmin(tournament_fitness, axis=1)

        # Step 4: Use these indices to select the best individuals from each tournament
        best_selected_indices = tournament_indices[np.arange(num_individuals), best_indices_in_tournament]

        # Step 5: Return the selected individualsi
        return self.population[best_selected_indices]
    
    
    #make me a function that will select the best individuals from the population based on the k_tournament, given talso the fitness of that population
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
    #--------------------------------------------------------------------- 4) Operators: Crossover ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 8.0) Extras: Fitness calculation ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 8.1) Extras: Stopping Criteria ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    def check_stopping_criteria(self):
        '''
        - Check the stopping criteria
        '''
        
        if round(self.best_objective) == round(self.mean_objective):
            return True
        else:
            return False
    



    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 8.2) Extras ------------------------------------------------------------------------------------------------
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
        print(f"       - Local Search: {self.local_search}")
        print(f"       - Distance Matrix: {self.distance_matrix}")
        print(f"   * Model Parameters:")
        print(f"       - K: {self.k_tournament_k}")
        print(f"       - Mutation rate: {self.mutation_rate}")
        print(f"       - Elitism percentage: {self.elistism} %")
        print(f"   * Running model:")
        #print(f"       - Population: {self.population}")
        #print(f"       - Len Population: {self.population.shape[0]}")
        print(f"       - Fitness: {self.fitness}")
        #print(f"       - Len Fitness: {self.fitness.shape[0]}")

    


    
    def calculate_information_iteration(self):
        '''
        - Calculate the mean and best objective function value of the population
        '''
        self.mean_objective = np.mean(self.fitness)
        self.best_objective = np.min(self.fitness)
        self.mean_fitness_list.append(self.mean_objective)
        self.best_fitness_list.append(self.best_objective)  
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index]
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
    #--------------------------------------------------------------------- 9) Plotting ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    
    def plot_distance_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.distance_matrix, cmap='viridis', annot=False)
        plt.title('Distance Matrix Heatmap')
        plt.xlabel('City Index')
        plt.ylabel('City Index')
        plt.show()


    def plot_fitness(self):
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the best and mean fitness values
        ax.plot(self.best_fitness_list, label='Best Distance', color='blue', marker='o')
        ax.plot(self.mean_fitness_list, label='Mean Distance', color='orange', marker='x')

        # Get the last iteration's best and mean fitness
        last_best_fitness = self.best_fitness_list[-1]
        last_mean_fitness = self.mean_fitness_list[-1]

        # Add text to the plot for the last iteration's fitness
        ax.text(x=len(self.best_fitness_list) - 1, 
                y=last_best_fitness, 
                s=f'Best: {last_best_fitness}\nMean: {last_mean_fitness}', 
                fontsize=10, 
                verticalalignment='bottom', 
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        # Set titles and labels
        ax.set_title(f'Distance over Iterations with mutation rate {self.mutation_rate * 100} %')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Distance')

        # Show the legend and grid
        ax.legend()
        ax.grid()

        # Return the figure object
        


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

    
        

        

 

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 10) NUMBA ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


















    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 10.1) Initialization ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def set_initialization_onlyValid_numpy(population_size, gen_size, distance_matrix, fitness_threshold=1e6):
    """
    Initialize the population with valid individuals (fitness < fitness_threshold), using fast NumPy operations.

    Parameters:
    - fitness_threshold: float
        Maximum allowed fitness for valid individuals. Higher fitness individuals are discarded.
    """
    

    

    # Pre-allocate population array and fitness list
    population = np.empty((population_size, gen_size), dtype=np.int32)
    fitness = np.empty(population_size, dtype=np.float64)

    current_index = 0

    while current_index < population_size:
        # Generate a batch of random individuals
        batch_size = population_size - current_index
        candidates = np.array([np.random.permutation(gen_size) for _ in range(batch_size)])

        # Calculate fitness for the batch
        candidate_fitness = calculate_fitness(candidates, distance_matrix=distance_matrix)

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

    return population, fitness

@njit(nopython=True)
def set_initialization_onlyValid(population_size, gen_size, distance_matrix, fitness_threshold=1e4):
    """
    Initialize the population with valid individuals (fitness < fitness_threshold), using fast NumPy operations.
    This version is optimized for Numba (no Python objects).
    """
    # Preallocate arrays for population and fitness
    population = np.empty((population_size, gen_size), dtype=np.int32)
    fitness = np.empty(population_size, dtype=np.float64)

    # Index to track valid individuals in the population
    current_index = 0

    while current_index < population_size:
        # Batch size: remaining slots in the population
        batch_size = population_size - current_index


        # Generate random candidates (batches of permutations)
        candidates = np.empty((batch_size, gen_size), dtype=np.int32)
        for i in range(batch_size):
            np.random.seed(current_index + i)
            candidates[i] = np.random.permutation(gen_size)

        # Calculate fitness for the candidates
        candidate_fitness = calculate_fitness(candidates, distance_matrix)
        #print(f"\n Candidate fitness is : {candidate_fitness}")
        #print(f"\n Fitness Threshold : {fitness_threshold}")

        # Filter valid candidates with fitness < fitness_threshold
        for i in range(batch_size):
            if candidate_fitness[i] < fitness_threshold:
                print(f"\n Candidate fitness is : {candidate_fitness[i]} < {fitness_threshold}")
                population[current_index] = candidates[i]
                fitness[current_index] = candidate_fitness[i]
                current_index += 1

        # If no valid candidates found, print a message and stop
        if current_index == population_size:
            break

    return population, fitness

@njit(nopython=True)
def generate_valid_population(population_size, gen_size, distance_matrix, fitness_threshold=1e4):
    """
    Generate a population of valid individuals (i.e., paths) that satisfy the fitness threshold.
    The population is represented as a 2D array of city indices (indices into the distance matrix).
    This version is optimized for Numba (no Python objects).
    """
    # Preallocate arrays for population and fitness
    population = np.empty((population_size, gen_size), dtype=np.int32)
    fitness = np.empty(population_size, dtype=np.float64)

    # Index to track valid individuals in the population
    current_index = 0

    for i in range(population_size):
        # Generate a random individual (path)
        individual = generate_valid_individual(distance_matrix=distance_matrix,num_cities=gen_size,threshold=1e4)
        population[i] = individual
    
    fitness = calculate_fitness(population, distance_matrix)

       

    
    return population, fitness

@njit(nopython=True)
def generate_valid_individual(distance_matrix, num_cities, threshold=1e4):
    """
    Generate a valid individual (i.e., a valid path), ensuring it doesn't break any constraints.
    The individual is represented as a list of city indices (indices into the distance matrix).
    This version is optimized for Numba (no Python objects).
    """
    while True:
        # Step 1: Start with a random city
        individual = np.empty(num_cities, dtype=np.int32)  # Preallocate an array for the individual
        visited = np.zeros(num_cities, dtype=np.bool_)  # Boolean array to track visited cities

        # Choose a random starting city
        current_city = np.random.randint(0, num_cities)  # Random city index
        individual[0] = current_city
        visited[current_city] = True

        # Step 2: Generate the path by choosing successor cities
        for i in range(1, num_cities):
            # Step 2.1: Generate all legal successor possibilities (cities that haven't been visited)
            legal_successors = np.nonzero(~visited)[0]  # Indices of cities that haven't been visited
            
            # Step 2.2: Check if there are no legal successors (i.e., all cities have been visited)
            if len(legal_successors) == 0:
                break  # No valid successor means the path can't be completed, break the inner loop

            # Step 2.3: Randomly choose the next city from the legal successors
            next_city = np.random.choice(legal_successors)
            individual[i] = next_city
            visited[next_city] = True

        # Step 3: Ensure the path is valid by checking the connection between the last and first city
        if len(legal_successors) > 0 and distance_matrix[individual[-1], individual[0]] < threshold:
            return individual  # Return the valid individual (path)


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 10.1) CROSSOVER ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit(nopython=True)
def crossover_singlepoint_population(population, gen_size):
    num_parents = population.shape[0]
    
    # Initialize the children population (ensure dtype is correctly specified)
    children_population = np.zeros_like(population, dtype=np.int32)
    
    # Generate random crossover points
    crossover_index = np.random.randint(1, gen_size)
    
    # Perform crossover for each pair of parents
    for i in range(num_parents):
        parent1 = population[i]
        parent2 = population[(i + 1) % num_parents]
        
        # Perform crossover
        child1, child2 = crossover_singlepoint(parent1, parent2, crossover_index)
        
        # Add the children to the population
        children_population[i] = child1
        children_population[(i + 1) % num_parents] = child2

    return children_population

@njit(nopython=True)
def crossover_singlepoint(parent1, parent2, crossover_index):
    num_cities = len(parent1)
    
    # Step 1: Initialize offspring as arrays of -1 (indicating unfilled positions)
    child1 = -1 * np.ones(num_cities, dtype=np.int32)
    child2 = -1 * np.ones(num_cities, dtype=np.int32)
    
    # Step 2: Copy the selected segment from parent1 to child1, and from parent2 to child2
    child1[crossover_index:] = parent1[crossover_index:]
    child2[crossover_index:] = parent2[crossover_index:]
    
    # Step 3: Fill the remaining positions in child1 with cities from parent2, avoiding duplicates
    current_idx = 0
    for i in range(num_cities):
        if child1[i] == -1:
            while parent2[current_idx] in child1:
                current_idx += 1
            child1[i] = parent2[current_idx]
    
    # Step 4: Fill the remaining positions in child2 with cities from parent1, avoiding duplicates
    current_idx = 0
    for i in range(num_cities):
        if child2[i] == -1:
            while parent1[current_idx] in child2:
                current_idx += 1
            child2[i] = parent1[current_idx]
    
    return child1, child2



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- 5) Operators: Mutation ------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@njit(nopython=True)
def mutation_singlepoint_population(population: np.ndarray, mutation_rate: float) -> np.ndarray:
    # Number of individuals in the population
    num_individuals = population.shape[0]
    
    # Initialize the mutated population
    mutated_population = np.copy(population)
    
    # Perform mutation for each individual
    for i in range(num_individuals):
        if np.random.rand() < mutation_rate:
            mutated_population[i] = mutation_singlepoint(mutated_population[i])

    return mutated_population

@njit(nopython=True)
def mutation_singlepoint(individual: np.ndarray) -> np.ndarray:
    # Number of genes in the individual
    num_genes = len(individual)
    
    # Initialize the mutated individual
    mutated_individual = np.copy(individual)

    # Select how many genes to mutate
    num_mutations = np.random.randint(1, (num_genes-1)//2 + 1)  # Make sure the upper bound is an integer
    mutation_indices1 = np.random.choice(num_genes, size=num_mutations, replace=False)
    
    # Generate mutation_indices2 (randomly select num_mutations from the remaining indices)
    all_indices = np.arange(num_genes)  # All possible indices
    mutation_indices2 = []

    # Manually find the set difference of indices to get remaining indices
    for idx in all_indices:
        if idx not in mutation_indices1:
            mutation_indices2.append(idx)
    
    # Convert mutation_indices2 to a numpy array before using np.random.choice
    mutation_indices2 = np.array(mutation_indices2)

    # Randomly sample the same number of mutations from the remaining indices
    mutation_indices2 = np.random.choice(mutation_indices2, size=num_mutations, replace=False)

    # Perform mutation for each gene
    mutated_individual[mutation_indices1], mutated_individual[mutation_indices2] = mutated_individual[mutation_indices2], mutated_individual[mutation_indices1]

    # Check if all values are unique (ensure no duplicates)
    if len(np.unique(mutated_individual)) != len(mutated_individual):
        print(f"\n EMERGENCY: Mutated Individual is not unique")

    return mutated_individual
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- 6) Operators: Local Search ------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit(nopython=True)
def local_search_population(population,distance_matrix,number_individuals = 5, ):
    """
    Perform 2-opt local search for each individual in the population.
    """
    #calculate teh fitness of the population
    orignal_size = population.shape[0]
    offspring_fitness = calculate_fitness(population)   
    best_indices = np.argsort(offspring_fitness)[:number_individuals]

    #select the best 5 individuals from the population
    selected_population = population[best_indices]
    num_individuals = selected_population.shape[0]
    
    #print(f"\n Number of individuals: {num_individuals}")
    
    # Initialize the local search population
    local_search_population = np.copy(selected_population)
    
    # Perform local search for each individual
    for i in range(num_individuals):
        local_search_population[i] = local_search_individual(local_search_population[i],distance_matrix=distance_matrix)
    
    total_population = np.vstack((population,local_search_population))  
    total_fitness = calculate_fitness(total_population)   
    best_indices = np.argsort(total_fitness)[:orignal_size]

    #select the best 5 individuals from the population
    final_population = total_population[best_indices]
    return final_population

@njit(nopython=True)
def local_search_individual( individual,distance_matrix, max_iter=10):
    """
    Perform 2-opt local search for an individual (tour) to improve it, but limit the number of iterations.
    """
    improved = True
    num_genes = len(individual)
    iteration = 0
    
    # Iterate until no more improvements are found or max_iter is reached
    while improved and iteration < max_iter:
        improved = False
        for i in range(num_genes - 1):
            for j in range(i + 2, num_genes):
                # Perform 2-opt swap between cities i and j
                new_individual = two_opt_swap(individual, i, j)
                
                # Check if the new individual is better (lower fitness)
                if calculate_fitness_individual(new_individual,distance_matrix=distance_matrix) < calculate_fitness_individual(individual,distance_matrix=distance_matrix):
                    individual = new_individual
                    improved = True
                    iteration += 1
                    break
            if improved:
                break
    
    return individual

@njit
def two_opt_swap(individual, i, j):
    """
    Perform a 2-opt swap: reverse the segment between indices i and j.
    """
    new_individual = np.copy(individual)
    
    # Reverse the order of cities between i and j (inclusive)
    new_individual[i:j+1] = np.flip(individual[i:j+1])
    
    return new_individual


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 7) Elimination ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
def eliminate_population(population, offsprings, population_fitness,population_size,distance_matrix):
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
    fitness_scores = calculate_fitness(population=offsprings, distance_matrix=distance_matrix)
    combined_fitness = np.hstack((population_fitness, fitness_scores))
    #print(f"\n Fitness V1--> {fitness_scores}")
    

    # Get the indices of the best individuals based on fitness
    
    best_indices = np.argsort(combined_fitness)[:population_size]

    

    # Select the best individuals
    new_population = combined_population[best_indices]
    new_fitness = combined_fitness[best_indices]

    return new_population, new_fitness

#make me a elimnate population that will use my elitism percentage to keep the best individuals and the rest will be compute dvia a k_tournament fucntion 

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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- 7.1) Elimnation:Fitness Sharing ------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@njit(nopython=True)
def calculate_fitness_individual(self, individual,distance_matrix):
    '''
    - Calculate the fitness of the individual (i.e., total distance of the path)
    '''
    distance = 0
    for i in range(len(individual) - 1):
        distance += distance_matrix[individual[i], individual[i + 1]]
    distance += distance_matrix[individual[-1], individual[0]]  # Returning to the start city
    return distance

def solution_similarity(self, individual, other_individual):
    """
    Measure the similarity between two solutions (individuals).
    A simple approach is to count how many cities are in the same position.
    This function could be more sophisticated depending on the problem.
    """
    similarity_count = sum(1 for i in range(len(individual)) if individual[i] == other_individual[i])
    # Return similarity as a ratio of matching cities
    return similarity_count / len(individual)

def fitness_sharing(self, population, individual, sharing_radius=0.5):
    '''
    - Fitness sharing modifies the fitness of an individual based on how similar its solution (not fitness) is to others.
    '''
    # Calculate the fitness (distance) of the current individual
    fitness = self.calculate_fitness_individual(individual)
    shared_fitness = 0.0
    
    # For each individual in the population, calculate the solution similarity to the current individual
    for other_individual in population:
        # Calculate the solution similarity (not the fitness) between individuals
        similarity = self.solution_similarity(individual, other_individual)
        #print(f"Similarity is : {similarity}")
        
        if similarity > sharing_radius:
            # Penalize individuals that are too similar in terms of the solution
            shared_fitness += 1.0 - similarity  # Sharing function: penalization increases as similarity increases
    
    # Avoid division by zero and return the adjusted shared fitness
    if shared_fitness == 0:
        return fitness  # Return original fitness if no sharing
    
    # Return the shared fitness (penalized fitness)
    return fitness / shared_fitness


def eliminate_population_fs(self, population, offsprings, sharing_radius=10.0):
    """
    Selects the best individuals from the combined population and offspring
    based on fitness with fitness sharing.

    Parameters:
    - population: numpy array of shape (population_size, individual_size)
    - offsprings: numpy array of shape (offspring_size, individual_size)
    - sharing_radius: the radius within which individuals are considered to have similar fitness (distance)

    Returns:
    - new_population: numpy array of shape (self.population_size, individual_size)
    """
    # Combine the original population with the offspring
    combined_population = np.vstack((population, offsprings))

    # Calculate shared fitness for the combined population using fitness sharing
    shared_fitness_population = []
    
    for individual in combined_population:
        shared_fitness = self.fitness_sharing(combined_population, individual, sharing_radius)
        print(f"Shared fitness is : {shared_fitness}")
        shared_fitness_population.append(shared_fitness)

    # Convert the shared fitness list into a numpy array
    shared_fitness_population = np.array(shared_fitness_population)

    # Get the indices of the best individuals based on shared fitness
    best_indices = np.argsort(shared_fitness_population)[:self.population_size]

    # Select the best individuals based on the shared fitness
    new_population = combined_population[best_indices]
    self.population = new_population
    self.fitness = self.calculate_fitness(new_population)  
    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- 7) Extras: Fitness calculation ------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@njit(nopython=True)
def calculate_fitness(population, distance_matrix):
    fitness = np.zeros(population.shape[0])
    
    # Loop through each individual (row) in the population
    for i in range(population.shape[0]):
        individual = population[i]
        
        # Initialize the fitness for this individual
        individual_fitness = 0.0
        
        # Loop through consecutive cities to calculate the total distance
        for j in range(len(individual) - 1):
            current_city = individual[j]
            next_city = individual[j + 1]
            individual_fitness += distance_matrix[current_city, next_city]
        
        fitness[i] = individual_fitness
    
    return fitness

def calculate_fitness_individual(individual,distance_matrix):
    '''
    - Calculate the fitness of the individual (tour).
    '''
    num_cities = len(individual)
    
    # Indices for current and next cities
    current_indices = individual[:-1]
    next_indices = individual[1:]
    
    # Calculate the distances between consecutive cities in the tour
    distances = distance_matrix[current_indices, next_indices]
    
    # Add the distance from the last city to the first city (closing the loop)
    closing_distance = distance_matrix[individual[-1], individual[0]]
    
    # Sum the distances
    fitness = np.sum(distances) + closing_distance
    
    return fitness





    


    
        


    
   

        

      



    
    
    











        

     

        


