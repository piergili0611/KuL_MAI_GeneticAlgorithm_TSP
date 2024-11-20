import numpy as np 
import matplotlib.pyplot as plt
import time 
from Evol_algorithm_K import GA_K
from Evol_algorithm_L2 import GA_K_L2
from k_clusters import k_clusters
from cities import cities
from clusterdashboard import ClusterDashboard



class algorithm:

    def __init__(self):
        self.num_city = None
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

    def set_distance_matrix(self,distance_matrix):
        '''
        - Set the distance matrix
        '''
        self.distance_matrix = self.check_inf(distance_matrix=distance_matrix,replace_value=100000)
        self.num_city = len(self.distance_matrix)
        #print(f"Distance matrix is {self.distance_matrix}")
        
    def check_inf(self,distance_matrix, replace=False ,replace_value=1000000):
        '''
        - Check if the distance matrix has inf values and replace them with a given value
        '''

        if replace:
            distance_matrix[distance_matrix == np.inf] = replace_value
        
        return distance_matrix
    
    def add_K_clusters_model(self):
        '''
        - Add the K_clusters model
        '''
        model = k_clusters(distance_matrix=self.distance_matrix)
        self.K_clusters_model = model

    def add_GA_level1_model(self,distance_matrix,cities,mutation_prob=0.008):
        '''
        - Add the GA model
        '''
        model = GA_K(cities=cities,mutation_prob=mutation_prob,seed=42)
        model.set_distance_matrix(distance_matrix)
        self.GA_level1_model = model

    def add_GA_level2_model(self,distance_matrix,cluster_solutions_matrix,mutation_prob=0.1):
        '''
        - Add the GA model
        '''
        model = GA_K_L2(clusters_solutions_matrix=cluster_solutions_matrix,cities_model=self.cities_model,mutation_prob=mutation_prob,seed=42)
        model.set_distance_matrix(distance_matrix)
        self.GA_level2_model = model


    def run_algorithm(self,clusters=True):
        '''
        - Run the algorithm
        '''
        distance_matrix_global = self.test_k_cluster_model(clusters=clusters)
        
        print(f"-- Running the algorithm --")
        assigned_cities_list = [cluster['assigned_cities'] for cluster in self.cluster_list]
        counter = 0
        #print(f"Assigned cities list: {assigned_cities_list}")
        #print(f"Cities cluster list self: {self.cities_cluster_list}")
        #print(f"Distance matrix cluster list: {self.distance_matrix_cluster_list}")
        for cities in self.cities_cluster_list:
            #print(f"Cities: {cities}")
            #print(f"Len Distance matrix cluster list: {len(self.distance_matrix_cluster_list)}")

            # 1) Add the GA model and run it for each cluster of cities and measure the time taken
            time_start_GA_level1 = time.time()
            distance_matrix = self.distance_matrix_cluster_list[counter]
            self.add_GA_level1_model(cities=cities,distance_matrix=distance_matrix)
            self.run_GA_level1_model(cities)
            best_solution = self.GA_level1_model.best_solution_cities
            self.add_cluster_solution(best_solution)

            time_end_GA_level1 = time.time()
            delta_time = time_end_GA_level1-time_start_GA_level1    
            print(f"Time taken for GA level 1: {delta_time}")
            self.deltatime_cluster_list.append(delta_time)

            # 2) Add the cities sequence to the cities model and plot the clusters
            #self.cities_model.add_cities_sequence(self.GA_level1_model.best_solution_cities)
            #self.cities_model.plot_clusters_sequence(city_sequence=True)
            counter += 1

        self.plot_ExecutionTime_Clusters()


        # 2) Create and run Higher level GA model
        self.add_GA_level2_model(distance_matrix=distance_matrix_global,cluster_solutions_matrix=self.clusters_solution_list)
        time_start_GA_level1 = time.time()
        self.GA_level2_model.run_model()
        time_end_GA_level1 = time.time()
        delta_time = time_end_GA_level1-time_start_GA_level1
        print(f"Time taken for GA level 2: {delta_time}")
        final_solution = self.GA_level2_model.best_solution_merged

        # 2) Plot teh resulting sequence
        self.cities_model.add_cities_sequence(final_solution)
        self.cities_model.plot_clusters_sequence(city_sequence=True)
        self.check_city_solution(final_solution)

    

    #make me a function taht given a city solution will check what i sthe length (number of cities) and if each city is unique
    def check_city_solution(self,city_solution):
        '''
        - Check the city solution
        '''
        num_cities = len(city_solution)
        unique_cities = len(set(city_solution))
        print(f"Number of cities: {num_cities} & Number of unique cities: {unique_cities}")
        


        



    def run_GA_level1_model(self,cities):
        '''
        - Run the GA model
        '''
        self.GA_level1_model.run_model()

    

    

    def run_k_cluster_model(self,model=None,num_clusters=None,min_cluster_size=None):
        '''
        - Run the k_cluster also
        '''

        if model:
            num_cities = int(model.num_cities)
            if num_clusters is None:
                num_clusters = int(num_cities/20)
            if min_cluster_size is None:
                min_cluster_size = int(num_cities/60)
            model.run_model( k=num_clusters, min_cluster_size=min_cluster_size) 
            cluster_list = model.clusters_list
        else:
            num_cities = int(self.K_clusters_model.num_cities)
            num_clusters = int(num_cities/20)
            min_cluster_size = int(num_cities/10)
            self.K_clusters_model.run_model( k=num_clusters, min_cluster_size=min_cluster_size)
            cluster_list = self.K_clusters_model.clusters_list
        return cluster_list



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
    
    def add_run_cities_model(self):
        '''
        - Add and run the cities model
        '''
        self.add_cities_model()
        self.run_city_model()


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

    

        
    