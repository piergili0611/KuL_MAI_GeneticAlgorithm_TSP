import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class GA:

    def __init__(self,seed=None,mutation_prob = 0.001,elitism_percentage = 20):
        #model_key_parameters
        self.k_tournament_k = 0
        self.population_size = 0.0
        self.mutation_rate = mutation_prob
        self.elistism = 5                        #Elitism rate as a percentage


        self.mean_objective = 0.0
        self.best_objective = 0.0
        self.mean_fitness_list = []
        self.best_fitness_list = [] 

        #Random seed
        if seed is not None:
            np.random.seed(seed)    
        
        pass

    def print_model_info(self):
        
       
        print("------ Model key parameters -------")
        print(f"\n Population size: {self.population_size} \n Mutation rate: {self.mutation_rate} \n  K_size: {self.k_tournament_k} \n")

    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf(distance_matrix=distance_matrix,replace_value=100000)
        self.gen_size = len(distance_matrix)
        #self.population_size = 2*self.gen_size
        self.population_size = 2*self.gen_size
        self.k_tournament_k = int((3/100)*self.population_size)
        #print(f"Distance matrix is {self.distance_matrix}")
        #print(f"Gen size is {self.gen_size}")

    #make me a function that will get the distnace matrix and check if a vlue is inf, and if yes will replace it by a number given like 100000
    def check_inf(self,distance_matrix,replace_value):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''
        distance_matrix[distance_matrix == np.inf] = replace_value
        return distance_matrix

    def set_initialization(self):
        '''
        - Initialize the population
        '''

        self.population = np.array([np.random.permutation(self.gen_size) for _ in range(self.population_size)])
        #print(f"Population shape --> {np.shape(self.population)} \n Population is : {self.population[1]}" )
        self.fitness = self.calculate_fitness(self.population)
        self.print_model_info()


    def selection_k_tournament(self, num_individuals, k=3):
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
        remaining_population_ordered, remaining_population_fitness_ordered = self.selection_k_tournament_population(num_individuals=remaining_size, population=remaining_population, fitness=remaining_population_fitness, k=3)


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
        return self.mean_objective,self.best_objective,best_solution
    
    

    def plot_fitness(self):
        #print(f"IS IT HERE")
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_list, label='Best Distance', color='blue', marker='o')
        plt.plot(self.mean_fitness_list, label='Mean Distance', color='orange', marker='x')

        # Get the last iteration's best and mean fitness
        last_best_fitness = self.best_fitness_list[-1]
        last_mean_fitness = self.mean_fitness_list[-1]
        
        # Add text to the plot for the last iteration's fitness
        plt.text(x=len(self.best_fitness_list) - 1, 
                y=last_best_fitness, 
                s=f'Best: {last_best_fitness}\nMean: {last_mean_fitness}', 
                fontsize=10, 
                verticalalignment='bottom', 
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        plt.title(f'Distance over Iterations with mutation rate {self.mutation_rate*100} %')
        plt.xlabel('Iterations')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid()
        plt.show()


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
