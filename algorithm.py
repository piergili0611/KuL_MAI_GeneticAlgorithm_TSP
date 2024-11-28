import numpy as np 
import matplotlib.pyplot as plt
import time 
import math
from Evol_algorithm_K import GA_K
from Evol_algorithm_L2 import GA_K_L2
from k_clusters import k_clusters
from cities import cities
from clusterdashboard import ClusterDashboard
from concurrent.futures import ProcessPoolExecutor, as_completed



class algorithm:

    def __init__(self,num_cities=None):
        self.num_city = num_cities
        self.GA_level1_model = None
        self.GA_level2_model = None
        self.K_clusters_model = None
        self.cluster_dashboard = ClusterDashboard()
        self.distanceMatrix = None
        self.clusters_list = []
        self.distance_matrix_cluster_list = []

        self.deltatime_cluster_list = []
        self.clusters_solution_list = []
        

        #To check k_clusterl model
        self.cities_model = None

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
                num_clusters = math.ceil(num_cities/50)
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
        self.distance_matrix_cluster_list,self.cities_cluster_list = self.K_clusters_model.generate_distance_matrix_cluster(self.cluster_list)


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 3) GA_Level1 ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def add_GA_level1_model(self,distance_matrix,cities,mutation_prob=0.1,local_search=True,initial_solution=None,max_iterations=50):
        '''
        - Add the GA model
        '''
        model = GA_K(cities=cities,mutation_prob=mutation_prob,seed=42,local_search=local_search,max_iterations=max_iterations)
        model.set_distance_matrix(distance_matrix)
        if initial_solution is not None:
            model.add_initialSolution(initial_solution=initial_solution)
        self.GA_level1_model = model

    def run_GA_level1_model(self,cities,plot=False):
        '''
        - Run the GA model
        '''
        self.GA_level1_model.run_model(plot=plot)

    def add_run_GA_level1_model(self,distance_matrix,cities,local_search=True,initial_solution=None,max_iterations=50,plot = False):
        '''
        - Add and run the GA model
        '''
        
        self.add_GA_level1_model(distance_matrix=distance_matrix,cities=cities,local_search=local_search,initial_solution=initial_solution,max_iterations=max_iterations)
        self.run_GA_level1_model(cities,plot=plot)

    def GA_level1_model_retrieveBestSolution(self):
        '''
        - Retrieve the best solution
        '''
        return self.GA_level1_model.best_solution_cities

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
            self.add_run_GA_level1_model(cities=np.array(cities),distance_matrix=distance_matrix,local_search=local_search,max_iterations=50,plot = True)
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
            self.add_run_k_cluster_model()
        else:
            self.add_run_k_cluster_model(num_clusters=1)

        
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
        with ProcessPoolExecutor() as executor:
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
                                         local_search=local_search,initial_solution = final_solution,max_iterations = 50,plot = True)
        else:
            final_solution = best_solution  
            print(f"Final solution: {final_solution}")
            self.check_city_solution(final_solution)

    def run_GA_Level1_task(self, cities, distance_matrix, local_search):
        '''
        Function to run GA_Level1 task. Returns the best solution and elapsed time.
        '''
        time_start = time.time()
        self.add_run_GA_level1_model(cities=cities, distance_matrix=distance_matrix, local_search=local_search)
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
    #--------------------------------------------------------------------- 7) Extras: Execution Time, Plots ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    


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







    




    
    

        
    