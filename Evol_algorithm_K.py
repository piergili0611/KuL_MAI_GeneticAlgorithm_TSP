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
        

        #Random seed
        if seed is not None:
            np.random.seed(seed)    
        
        

    def run_model(self):
        self.set_initialization()
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
            parents = self.selection_k_tournament(num_individuals=self.population_size)    
            offspring = self.crossover_singlepoint_population(parents)
            offspring_mutated = self.mutation_singlepoint_population(offspring)

            self.eliminate_population(population=self.population, offsprings=offspring_mutated)
            #self.eliminate_population_elitism(population=self.population, offsprings=offspring_mutated)
            meanObjective, bestObjective , bestSolution  = self.calculate_information_iteration()
            yourConvergenceTestsHere = False

        self.print_best_solution()
        self.plot_fitness_dynamic()
        
        return 0


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
        #print(f"       - Distance Matrix: {self.distance_matrix}")
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
        self.population_size = 1*self.gen_size
        self.k_tournament_k = int((3/100)*self.population_size)
        #print(f"Distance matrix is {self.distance_matrix}")
        #print(f"Gen size is {self.gen_size}")



    def plot_distance_matrix(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.distance_matrix, cmap='viridis', annot=False, cbar=True)
        plt.title('Distance Matrix Heatmap')
        plt.xlabel('City Index')
        plt.ylabel('City Index')
        plt.show()


    #--------------------------------------------------------------- K_meloids clustering ---------------------------------------------------------------
    #--------------------------------------------------------------- K_meloids clustering ---------------------------------------------------------------
    #--------------------------------------------------------------- K_meloids clustering ---------------------------------------------------------------
    #--------------------------------------------------------------- Creation of points ---------------------------------------------------------------

    def set_known_locations(self, known_locations):
        # Initialize the count of valid cities and set x_new coordinate
        valid_cities = 1  # Start with the first city as valid (index 0)
        x_new = 0  # Set x_new coordinate for new cities along x-axis

        # Iterate to find exactly 3 valid cities based on the distance matrix
        while valid_cities < 3:
            # Calculate distances to the next potential city (valid_cities + 1)
            r_from_first = self.distance_matrix[0, valid_cities]
            if valid_cities > 1:
                r_from_previous = self.distance_matrix[valid_cities - 1, valid_cities]
            
            # Skip if any distance is infinite
            if np.isinf(r_from_first) or (valid_cities > 1 and np.isinf(r_from_previous)):
                valid_cities += 1
                continue
            
            # Calculate y-coordinate for the new city
            y_set = known_locations[0][1]
            y_new = y_set - np.sqrt(r_from_first**2 - (known_locations[0][0] - x_new)**2)

            # Append new location to known_locations
            known_locations = np.vstack((known_locations, [x_new, y_new]))
            valid_cities += 1
        
        print(f"\n Known locations: {known_locations}")
        self.known_cities = known_locations


    def iterative_refinement(self,distance_matrix,circles_method = False):
        

        self.set_distance_matrix(distance_matrix=distance_matrix)
        print(f"\nDistance matrix given:\n{self.distance_matrix}")
        
        known_locations = np.array([[0,0]])
        self.set_known_locations(known_locations)

        len_known = len(self.known_cities)
        n = self.distance_matrix.shape[0]
        coords = np.zeros((n, 2))
        valid_indices = np.where(self.distance_matrix!= np.inf)[0]

        
        unknown_indices = np.arange(start=len_known,stop=n,step=1)
        print(f"\n Unknown indices: {len(unknown_indices)}")

        print(f"\n Known cities: {len(self.known_cities)} \n {self.known_cities}")
        

        if circles_method:
            for unknow_city in unknown_indices:
                city1 = self.known_cities[0]
                city2 = self.known_cities[1]
                city3 = self.known_cities[2]

                r1 = self.distance_matrix[0,unknow_city]
                r2 = self.distance_matrix[1,unknow_city]
                r3 = self.distance_matrix[2,unknow_city]

                print(f"\n Unknow city {unknow_city} using: {city1}, {city2} & {city3}")

                max_iterations = 500
                tolerance = 1e-10
                x_new = 0
                steps = 200
                angles = np.linspace(0,2*np.pi,steps)
                dim = len(angles)
                pos1 = np.zeros(dim)
                print(pos1)
                pos2 = np.zeros(dim)
                pos3 = np.zeros(dim)

                pos1 = []
                pos2 = []
                pos3 = []
                counter = 0

                for angle in angles:
                    '''
                    pos1[counter] = np.array([r1*np.cos(angle),r1*np.sin(angle)])
                    pos2[counter] = np.array([r2*np.cos(angle),r2*np.sin(angle)])
                    pos3[counter] = np.array([r3*np.cos(angle),r3*np.sin(angle)])
                    '''
                    print(f"\n Angle: {angle} & r1: {r1} r2: {r2} r3: {r3}")
                    pos1.append((city1[0]+r1*np.cos(angle),city1[1]+r1*np.sin(angle)))
                    pos2.append((city2[0]+r2*np.cos(angle),city2[1]+r2*np.sin(angle)))
                    pos3.append((city3[0]+r3*np.cos(angle),city3[1]+r3*np.sin(angle)))
                    print(f"\n Positions: \n P1: {pos1[counter]} P2: {pos2[counter]} P3: {pos3[counter]}")
                    
                    counter +=1
                circles = [pos1,pos2,pos3]

                self.plot_circles(circles)

                
            

        
        if not circles_method:
            for unknown_city in unknown_indices:
                # Try to select three cities with finite distances to the unknown city
                city_indices = []
                distances = []
                for known_index, known_city in enumerate(self.known_cities):
                    distance = self.distance_matrix[known_index, unknown_city]
                    if distance != np.inf:
                        city_indices.append(known_index)
                        distances.append(distance)
                    if len(city_indices) == 3:  # Stop once we have three cities
                        break

                if len(city_indices) < 2:
                    print(f"Not enough known cities with finite distances to locate city {unknown_city}. Skipping.")
                    continue

                # Extract the selected cities and their distances
                city1, city2 = self.known_cities[city_indices[0]], self.known_cities[city_indices[1]]
                r1, r2 = distances[0], distances[1]

                # Check if a third city is available for added accuracy
                if len(city_indices) == 3:
                    city3 = self.known_cities[city_indices[2]]
                    r3 = distances[2]
                    print(f"\nUnknown city {unknown_city} using cities at indices {city_indices}: {city1}, {city2}, {city3}")
                    print(f"\nDistances: r1: {r1}, r2: {r2}, r3: {r3}")
                else:
                    print(f"\nUnknown city {unknown_city} using cities at indices {city_indices}: {city1}, {city2}")
                    print(f"\nDistances: r1: {r1}, r2: {r2}")

                # Setup iterative method parameters
                max_iterations = 10000
                tolerance = 1e-5
                learning_rate = 0.01

                # Initial guess for x_new and y_new
                x_new = 0
                y_new = city1[1] - np.sqrt(np.abs(r1**2 - (city1[0] - x_new)**2))
                previous_diff = None

                for iteration in range(max_iterations):
                    # Calculate y_new based on city1 and x_new, and x_new based on city2 and y_new
                    y_new = city1[1] - np.sqrt(np.abs(r1**2 - (city1[0] - x_new)**2))
                    x_new_calculated = city2[0] - np.sqrt(np.abs(r2**2 - (city2[1] - y_new)**2))

                    # If a third city is available, adjust x_new further to satisfy r3
                    if len(city_indices) == 3:
                        y_new_adjusted = city3[1] - np.sqrt(np.abs(r3**2 - (city3[0] - x_new)**2))
                        y_new = y_new + learning_rate * (y_new_adjusted - y_new)

                    # Update x_new with gradual adjustment to stabilize convergence
                    x_new = x_new + learning_rate * (x_new_calculated - x_new)
                    diff = np.abs(x_new_calculated - x_new)

                    # Check for convergence
                    if diff <= tolerance:
                        new_city = np.array([x_new, y_new])
                        print(f" --- CITY FOUND --> {new_city} at iteration {iteration}")
                        self.known_cities = np.vstack((self.known_cities, new_city))
                        break

                    elif previous_diff is not None and abs(diff - previous_diff) < tolerance:
                        print(f"Oscillation detected. Diff:  {diff} - {previous_diff}")

                    # Update previous_diff for oscillation detection
                    previous_diff = diff
                    # Optionally, print progress for each iteration
                    #print(f"\nIteration: {iteration} \n x_new_guess: {x_new} \n Calculated: y_new: {y_new}, x_new: {x_new_calculated} \n Diff: {diff}")

                else:
                    print("Failed to converge within the maximum iterations.")








                '''
                for unknown_city in unknown_indices:
                    city1, city2, city3 = self.known_cities[:3]

                    r1 = self.distance_matrix[0, unknown_city]
                    r2 = self.distance_matrix[1, unknown_city]
                    r3 = self.distance_matrix[2, unknown_city]

                    print(f"\n Unknown city {unknown_city} using: {city1}, {city2} & {city3}")
                    print(f"\n Distances: r1: {r1}, r2: {r2}, r3: {r3}")

                    max_iterations = 50
                    tolerance = 1e-5
                    learning_rate = 0.5  # Controls the step size to stabilize convergence

                    x_new = 0  # Initial guess
                    y_new = city1[1] - np.sqrt(np.abs(r1**2 - (city1[0] - x_new)**2))

                    previous_diff = None

                    for iteration in range(max_iterations):
                        # Calculate the new y_new and x_new based on r1 and r2
                        y_new = city1[1] - np.sqrt(np.abs(r1**2 - (city1[0] - x_new)**2))
                        x_new_calculated = city2[0] - np.sqrt(np.abs(r2**2 - (city2[1] - y_new)**2))

                        # Update with a gradual adjustment to help stabilize
                        x_new = x_new + learning_rate * (x_new_calculated - x_new)
                        diff = np.abs(x_new_calculated - x_new)

                        if diff <= tolerance:
                            # Successfully found the city location
                            new_city = np.array([x_new, y_new])
                            print(f" --- CITY FOUND --> {new_city} at iteration {iteration}")
                            self.known_cities = np.vstack((self.known_cities, new_city))
                            break
                        elif previous_diff is not None and abs(diff - previous_diff) < tolerance:
                            print("Oscillation detected, exiting.")
                            break

                        # Update previous_diff to track convergence
                        previous_diff = diff
                        print(f"\nIteration: {iteration} \n x_new_guess: {x_new} \n Calculated: y_new: {y_new}, x_new: {x_new_calculated} \n Diff: {diff}")

                    else:
                        print("Failed to converge within the maximum iterations.")

                '''


        print(f"\n Known cities: {len(self.known_cities)} \n {self.known_cities}")
        self.plot_city_locations()
        self.verify_computed_distances()
                    
                    
            

    def verify_computed_distances(self,round_to=0):
        """
        Automatically generates a distance matrix for given city positions, compares it with the original
        distance matrix, and checks for equivalency while respecting infinite values.

        Parameters:
        city_positions (numpy.ndarray): Array of shape (n, 2) representing the x, y coordinates of each city.
        original_distance_matrix (numpy.ndarray): The original distance matrix with possible np.inf values.
        round_to (int): Number of decimal places to round distances to (0 rounds to the nearest integer).

        Returns:
        bool: True if the computed distance matrix matches the original distance matrix within the specified tolerance.
        """
        # Set print options to display the full matrix
        np.set_printoptions(threshold=np.inf)
        # Step 1: Generate the computed distance matrix
        city_positions = self.known_cities
        original_distance_matrix = self.distance_matrix
        print(f"\n Original distance matrix: \n {original_distance_matrix}")
        num_cities = city_positions.shape[0]
        print(f"\n Number of Cities: {num_cities}")
        computed_distance_matrix = np.zeros((num_cities, num_cities))
        
        # Calculate Euclidean distances for each pair of cities
        for i in range(num_cities):
            for j in range(num_cities):
                if i == j:
                    computed_distance_matrix[i, j] = 0
                else:
                    distance = np.linalg.norm(city_positions[i] - city_positions[j])
                    computed_distance_matrix[i, j] = distance
        print(f"\n Computed distance matrix: \n {computed_distance_matrix}")
        # Step 2: Prepare matrices for comparison with rounding
        rounded_computed_matrix = np.where(
            np.isfinite(computed_distance_matrix),
            np.round(computed_distance_matrix, decimals=round_to),
            np.inf
        )
        rounded_original_matrix = np.where(
            np.isfinite(original_distance_matrix),
            np.round(original_distance_matrix, decimals=round_to),
            np.inf
        )
        print(f"\n Rounded computed matrix: \n {rounded_computed_matrix}")  
        print(f"\n Rounded original matrix: \n {rounded_original_matrix}")
        
        # Step 3: Create a mask of finite values to compare only the finite entries
        finite_mask = np.isfinite(original_distance_matrix)
        
        # Check if all finite values match within the specified tolerance
        finite_values_match = np.allclose(rounded_computed_matrix[finite_mask], rounded_original_matrix[finite_mask])
        
        # Check if infinite values are in the same locations in both matrices
        inf_values_match = np.array_equal(np.isinf(rounded_computed_matrix), np.isinf(rounded_original_matrix))
        
        # Output results
        if finite_values_match and inf_values_match:
            print("The computed distance matrix matches the original distance matrix.")
            return True
        else:
            print("Discrepancy found between computed and original distance matrices.")
            return False





    def plot_city_locations(self):
       
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.known_cities[:, 0], self.known_cities[:, 1], c='blue', marker='o')
        
        plt.title('City Locations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()










        #--------------------------------------------------------------- GA_Algorithm ---------------------------------------------------------------
        #--------------------------------------------------------------- GA_Algorithm ---------------------------------------------------------------
        #--------------------------------------------------------------- GA_Algorithm ---------------------------------------------------------------
        #--------------------------------------------------------------- GA_Algorithm ---------------------------------------------------------------
        #--------------------------------------------------------------- GA_Algorithm ---------------------------------------------------------------
        #--------------------------------------------------------------- GA_Algorithm ---------------------------------------------------------------

    
    def check_inf(self,distance_matrix,replace_value):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''
        #distance_matrix[distance_matrix == np.inf] = replace_value
        return distance_matrix
    

    def k_means_distanced(self,distance_matrix):
        self.set_distance_matrix(distance_matrix=distance_matrix)
        self.calculate_city_locations()
        self.plot_city_locations()  
    





    def set_initialization(self):
        '''
        - Initialize the population
        '''

        self.population = np.array([np.random.permutation(self.gen_size) for _ in range(self.population_size)])
        self.fitness = self.calculate_fitness(self.population)
        print(f"\n Initial Population: {self.population}")
        print(f"\n Initial Fitness: {self.fitness}")
        self.print_model_info()


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
            mutated_population[i] = self.mutation_singlepoint(population[i], mutation_rate)

        #print(f"\n MUTATION: Children Population is : {mutated_population[1]}" )
        
        return mutated_population

    def mutation_singlepoint(self, individual, mutation_rate=0.8):
        # Number of genes in the individual
        num_genes = len(individual)
        #print(f"\n Individual is : {individual}")
        
        # Initialize the mutated individual
        mutated_individual = np.copy(individual)
        
        # Perform mutation for each gene
        for i in range(num_genes):
            # Check if we should perform mutation for this gene
            if np.random.rand() < mutation_rate:
                # Get a random index for the mutation
                mutation_index = np.random.randint(num_genes)
                
                # Perform mutation
                mutated_individual[i], mutated_individual[mutation_index] = mutated_individual[mutation_index], mutated_individual[i]
        #print(f"\n Mutated Individual is : {mutated_individual}")
        return mutated_individual
    
    
    

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
        fitness_scores = self.calculate_fitness(offsprings)
        combined_fitness = np.hstack((self.fitness, fitness_scores))
        #print(f"\n Fitness V1--> {fitness_scores}")
       

        # Get the indices of the best individuals based on fitness
        
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.fitness = combined_fitness[best_indices]

        # Select the best individuals
        self.population = combined_population[best_indices]
        self.fitness = combined_fitness[best_indices]

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
        
        
        

   

        #print(f"---- END ELIMNATION ELITISM------")
       

        

      



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
    
    def check_stopping_criteria(self):
        '''
        - Check the stopping criteria
        '''
        
        if round(self.best_objective) == round(self.mean_objective):
            return True
        else:
            return False


    

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
    
    











    def plot_distance_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.distance_matrix, cmap='viridis', annot=False)
        plt.title('Distance Matrix Heatmap')
        plt.xlabel('City Index')
        plt.ylabel('City Index')
        plt.show()

    def plot_circles(self,circles):
        circle1 = circles[0]
        circle2 = circles[1]
        circle3 = circles[2]

        circle1_x = []
        circle1_y = []
        circle2_x = []
        circle2_y = []
        circle3_x = []
        circle3_y = []

        for i in range(len(circle1)):
            #print(circle1[i])
            circle1_x.append(circle1[i][0])
            circle1_y.append(circle1[i][1])
            circle2_x.append(circle2[i][0])
            circle2_y.append(circle2[i][1])
            circle3_x.append(circle3[i][0])
            circle3_y.append(circle3[i][1])

        
        print(f"Circle_1 x: {circle1_x}")

        
        # Create a plotly figure
        fig = go.Figure()

        # Add the best fitness trace
        fig.add_trace(go.Scatter(
            x=circle1_x,
            y=circle1_y,
            mode='lines+markers',
            name='Circle 1',
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        fig.add_trace(go.Scatter(
            x=circle2_x,
            y=circle2_y,
            mode='lines+markers',
            name='Circle 2',
            line=dict(color='orange'),
            marker=dict(symbol='x')
        ))

        fig.add_trace(go.Scatter(
            x=circle3_x,
            y=circle3_y,
            mode='lines+markers',
            name='Circle 3',
            line=dict(color='red'),
            marker=dict(symbol='x')
        ))

        


        # Set the title and axis labels
        fig.update_layout(
            title=f'Circles',
            xaxis_title='X',
            yaxis_title='Y',
            legend=dict(x=0, y=1),
            hovermode='x'
        )

        # Show the plot
        fig.show()
        




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
            hovermode='x'
        )

        # Show the plot
        fig.show()

        
