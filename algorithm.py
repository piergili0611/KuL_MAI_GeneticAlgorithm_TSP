import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import time 
import math
from Evol_algorithm_K import r0818807
from Evol_algorithm_L2 import GA_K_L2
from k_clusters import k_clusters
from cities import cities
from clusterdashboard import ClusterDashboard
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
        self.cluster_dashboard = ClusterDashboard()
        self.distanceMatrix = None
        self.clusters_list = []
        self.distance_matrix_cluster_list = []

        self.deltatime_cluster_list = []
        self.clusters_solution_list = []
        

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
        self.distance_matrix_cluster_list,self.cities_cluster_list = self.K_clusters_model.generate_distance_matrix_cluster(self.cluster_list)


    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------- 3) GA_Level1 ------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def add_GA_level1_model(self,distance_matrix,cities,local_search=True,initial_solution=None,max_iterations=1500, mutation_rate=0.1):
        '''
        - Add the GA model
        '''
        if self.sigma_value is not None:
            model = r0818807(cities=cities,mutation_rate=mutation_rate,seed=42,local_search=local_search,max_iterations=max_iterations,sigma_value= self.sigma_value)
        else:
             model = r0818807(cities=cities,mutation_rate=mutation_rate,seed=42,local_search=local_search,max_iterations=max_iterations)
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
        plt.legend(
            handles=handles,
            loc='upper right',  # Place it in the top-right corner inside the plot
            #bbox_to_anchor=(1.02, 0.5),  # Outside the plot on the right
            fontsize=18,
            frameon=True
        )
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







    




    
    

        
    