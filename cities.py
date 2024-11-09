import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns



class cities:
    def __init__(self,num_cities):
        self.num_simulations_to_run = 2
        self.distanceMatrix = None
        self.num_cities = num_cities  
        self.cities = self.generate_cities()  
        self.clusters_list = None
        self.distance_matrix_cluster = []
        self.cities_cluster_list = []
        self.cities_sequence_list = []
        np.random.seed(42)


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


        
            

