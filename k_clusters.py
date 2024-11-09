import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns



class k_clusters:
    def __init__(self,distance_matrix):
        self.num_simulations_to_run = 2
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.clusters_list = []
        

    def print_info_run_model(self,k,min_cluster_size):
        '''
        - Print the information
        '''
        print("\n---------- K-Medoids Clustering: ------")
        print(f"   * Model Info:")
        print(f"       - Number of cities: {self.num_cities}")
        print(f"       - K: {k}")
        print(f"       - Min Cluster Size: {min_cluster_size}")
        print(f"   * Running model:")

        #print(f"Distance matrix: {self.distance_matrix}")

    def run_model(self,k,min_cluster_size):
        '''
        - Run the model
        '''
        self.print_info_run_model(k,min_cluster_size)
        self.k_medoids_clustering(k=k, min_cluster_size=min_cluster_size)



    def find_closest_finite_city(self,distance_matrix, city_index):
        """
        Find the closest city to the given city_index that has a finite distance.

        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix with possible np.inf values.
        - city_index (int): The index of the city for which we want to find the closest connected city.

        Returns:
        - closest_city (int): The index of the closest city with a finite distance, or None if all distances are infinite.
        """
        finite_distances = distance_matrix[city_index]
        finite_city_indices = np.where(np.isfinite(finite_distances))[0]
        
        if len(finite_city_indices) == 0:
            return None  # No reachable cities
        
        # Find the index of the closest city with a finite distance
        closest_city = finite_city_indices[np.argmin(finite_distances[finite_city_indices])]
        return closest_city
    
    def handle_infinite_distances(self,distance_matrix, labels, medoids):
        """
        Reassigns cities with infinite distances to medoids based on the cluster assignment
        of the nearest connected cities.
        
        Parameters:
        - distance_matrix: numpy.ndarray, the distance matrix with possible np.inf values.
        - labels: numpy.ndarray, initial cluster assignments.
        - medoids: list, the current medoids.
        
        Returns:
        - labels: numpy.ndarray, updated cluster assignments.
        """
        n_cities = distance_matrix.shape[0]
        for city in range(n_cities):
            # Skip cities that already have a valid cluster assignment
            if labels[city] != -1:
                continue
            
            # Use the helper function to find the closest city with a finite distance
            closest_city = self.find_closest_finite_city(distance_matrix, city)
            
            if closest_city is not None:
                # Assign the current city to the cluster of the closest connected city
                labels[city] = labels[closest_city]
                #print(f"City {city} assigned to cluster {labels[closest_city]} based on closest city {closest_city}.")
            else:
                print(f"Warning: City {city} has no reachable cities and remains unassigned.")
        
        return labels

    def find_closest_finite_city(self,distance_matrix, city):
            """
            Finds the closest city with a finite distance.
            
            Parameters:
            - distance_matrix: numpy.ndarray, the distance matrix with possible np.inf values.
            - city: int, the index of the city for which we are finding the closest city.
            
            Returns:
            - int: the index of the closest city with a finite distance, or None if none found.
            """
            n_cities = distance_matrix.shape[0]
            min_distance = np.inf
            closest_city = None
            
            for other_city in range(n_cities):
                if np.isfinite(distance_matrix[city, other_city]) and distance_matrix[city, other_city] < min_distance:
                    min_distance = distance_matrix[city, other_city]
                    closest_city = other_city
            
            return closest_city
    


    def k_medoids_clustering(self, k=4, max_iterations=20, tolerance=1e-1, distance_tolerance=100, min_cluster_size=10):
        """
        Perform k-Medoids clustering from a distance matrix.
        
        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - k (int): The number of clusters.
        - max_iterations (int): The maximum number of iterations to run the algorithm.
        - tolerance (float): The tolerance level to stop the algorithm if medoids stop changing.
        
        Returns:
        - medoids (list): The final medoids (cluster centers).
        - labels (numpy.ndarray): The cluster assignment for each city.
        """

        # Initialize medoids randomly
        
        distance_matrix = self.distance_matrix  
        n_cities = distance_matrix.shape[0]
        medoids = np.random.choice(n_cities, k, replace=False)
        prev_medoids = np.copy(medoids)
        prev_intra_cluster_distance = np.inf
        intra_cluster_distances_dict = {}  # To store intra-cluster distances
        total_distance_list = []

        for iteration in range(max_iterations):
            #print(f"\nIteration {iteration + 1}:")
            #print(f"Current Medoids: {medoids}")
            
            labels = np.full(n_cities, -1)  # Initialize all cities as unassigned
            for city in range(n_cities):
                min_distance = np.inf
                for medoid in medoids:
                    if distance_matrix[city, medoid] != np.inf:  # Avoiding inf distances
                        if distance_matrix[city, medoid] < min_distance:
                            min_distance = distance_matrix[city, medoid]
                            labels[city] = medoid

            # Handle cities with infinite distances
            labels = self.handle_infinite_distances(distance_matrix, labels, medoids)
            
            new_medoids = np.copy(medoids)
            total_intra_cluster_distance = 0  # Total distance to compute the convergence criteria

            # Update medoids by minimizing the total intra-cluster distance
            for cluster in range(k):
                cluster_cities = np.where(labels == medoids[cluster])[0]  # Get cities assigned to the current cluster
                if len(cluster_cities) > 0:  # Only proceed if the cluster is not empty
                    min_total_distance = np.inf  # Initialize to a very large number to minimize late.
                    #print(f"\nCluster {cluster}")

                    for meloid in cluster_cities:
                        best_medoid = meloid
                        #print(f"\n       Medoid:{best_medoid} cluster_cities: {cluster_cities} number of cities: {len(cluster_cities)}")
                       
                        total_distance = 0

                        for city in cluster_cities:
                        
                            
                            # Distance from the medoid to the current city
                            distance = distance_matrix[best_medoid, city]
                            #print(f"Cluster: {best_medoid}  & City {city} Total Distance: {total_distance} with distance: {distance}")
                            
                            # If the distance is finite, add it to the total distance
                            if np.isfinite(distance):
                                total_distance += distance
                            else:
                                # If the distance is infinite, find the closest city in the cluster with a finite distance
                                #print(f"City {city} has infinite distance to the medoid.")
                                closest_city = None
                                closest_distance = np.inf
                                dist_closest_to_meloid = 0
                                
                                for other_city in cluster_cities:
                                    if np.isfinite(distance_matrix[city, other_city]) and distance_matrix[city, other_city] < closest_distance and np.isfinite(distance_matrix[best_medoid, other_city]):
                                        closest_city = other_city
                                        closest_distance = distance_matrix[city, other_city]
                                        dist_closest_to_meloid = distance_matrix[best_medoid, closest_city]    
                                
                                # Add the distance to the closest city to the total distance if found
                                if closest_city is not None:
                                    #print(f"City {city} has infinite distance to the medoid. Closest city: {closest_city} Distance: {closest_distance}")
                                    total_distance += closest_distance + dist_closest_to_meloid
                                else:
                                    print(f"Warning: City {city} has no reachable cities in the cluster.")

                                        
                        # Update the best medoid if the total distance is smaller
                        if total_distance < min_total_distance:
                            #print(f"Total Distance: {total_distance} < {min_total_distance}")
                            min_total_distance = total_distance
                            best_medoid_final = best_medoid
                            #print(f"Best Medoid Final: {best_medoid_final} Total Distance: {min_total_distance}")

                    # Update the medoid for the cluster with the best one found
                    new_medoids[cluster] = best_medoid_final
                    intra_cluster_distances_dict[cluster] = min_total_distance
                    total_intra_cluster_distance += min_total_distance

            # Check for convergence
            #print(f"Total Intra-Cluster Distance: {total_intra_cluster_distance}")
            total_distance_list.append(total_intra_cluster_distance)
            #get me the previous total distance form the list

            diff_dist = np.inf
            prev_intra_cluster_distance = total_distance_list[-2] if len(total_distance_list) > 1 else np.inf
            if prev_intra_cluster_distance != np.inf:   
                diff_dist = np.abs(total_intra_cluster_distance - prev_intra_cluster_distance)
                #print(f"Change in Total Intra-Cluster Distance: {diff_dist}")
                #print(f"total_intr_cluster_distance: {total_intra_cluster_distance} prev_intra_cluster_distance: {prev_intra_cluster_distance}")    
            
            #diff = np.abs(np.sum(new_medoids - prev_medoids))
            if np.all(new_medoids == prev_medoids) or (diff_dist <= distance_tolerance):
                print(f"Converged after {iteration + 1} iterations.")
                #print me teh reason of the convergence
                #print(f"Reason of convergence: {diff_dist} <= {distance_tolerance} or {np.all(new_medoids == prev_medoids)}") 
                #print(f"Final Medoids: {new_medoids}")  
                
                break
            
            prev_medoids = np.copy(new_medoids)
            medoids = np.copy(new_medoids)
        
        # Print final clustering results
        print("\nFinal Clustering Results:")
        intra_cluster_distances = np.array(list(intra_cluster_distances_dict.values()))
        

        for cluster in range(k):
            cluster_cities = np.where(labels == medoids[cluster])[0]
            intra_cluster_distance = intra_cluster_distances_dict.get(cluster, 0)
            #print(f"\nCluster {cluster}:")
            #print(f"  Medoid: {medoids[cluster]}")
            #print(f"  Number of Cities: {len(cluster_cities)}")
            #print(f"  Assigned Cities: {cluster_cities}")
            #print(f"  Intra-Cluster Distance: {intra_cluster_distance}")
            cluster_info = {"cluster": cluster, "medoid": medoids[cluster], "num_cities": len(cluster_cities), "assigned_cities": cluster_cities, "intra_cluster_distance": intra_cluster_distance}
            self.clusters_list.append(cluster_info)
            
        self.verify_unique_cluster_assignment(labels, k)    
        self.plot_intra_cluster_distances(distance_matrix, intra_cluster_distances, labels, medoids, k)
        medoids, labels, k = self.handle_empty_or_small_clusters(distance_matrix, medoids, labels, k, min_cluster_size=min_cluster_size)
        intra_cluster_distances = self.calculate_intra_cluster_distance(distance_matrix, labels, medoids)    
        self.plot_intra_cluster_distances(distance_matrix, intra_cluster_distances, labels, medoids, k) 

        return medoids, labels, intra_cluster_distances_dict
    
    def handle_empty_or_small_clusters(self,distance_matrix, clusters, labels, k, min_cluster_size=1):
        """
        Checks for empty or small clusters and reassigns cities in these clusters to the nearest cluster.

        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - clusters (list): The list of current medoids.
        - labels (numpy.ndarray): Array of cluster assignments for each city.
        - k (int): The total number of clusters.
        - min_cluster_size (int): The minimum size a cluster can have before being considered "small" and removed.

        Returns:
        - clusters (list): Updated medoids (clusters).
        - labels (numpy.ndarray): Updated labels after reassignment.
        """
        print("\nHandling Empty or Small Clusters:")
        print(f"Number of Clusters: {k}")
        #print(f"Clusters: {clusters}")
        print(f"Number of Cities: {len(labels)}")
        #print(f"Cluster Assignments: {labels}")
        print(f"Minimum Cluster Size: {min_cluster_size}")
        

        # We will need to modify clusters, so we will create a new list for valid clusters
        new_clusters = []
        new_labels = np.copy(labels)
        
        # We need to loop over the clusters using the original `clusters` list length
        for cluster_idx in range(k):
            # Find the cities assigned to the current cluster
            cluster_cities = np.where(labels == clusters[cluster_idx])[0]
            
            # If the cluster is empty or has fewer cities than the minimum size
            if len(cluster_cities) < min_cluster_size:
                medoid = clusters[cluster_idx]
                print(f"Cluster {cluster_idx} with medoid: {medoid} is too small with number of cities {len(cluster_cities)} or empty, reassigning cities.")
                
                # For each city in this cluster, find the closest valid medoid
                for city in cluster_cities:
                    min_distance = np.inf
                    best_medoid = -1
                    
                    # Find the closest medoid that is not the current one
                    for other_cluster_idx in range(k):
                        if other_cluster_idx != cluster_idx:
                            distance_to_medoid = distance_matrix[city, clusters[other_cluster_idx]]
                            # Ensure that we don't consider infinite distances
                            if distance_to_medoid != np.inf and distance_to_medoid < min_distance:
                                min_distance = distance_to_medoid
                                best_medoid = clusters[other_cluster_idx]
                    
                    # Reassign the city to the closest cluster
                    new_labels[city] = best_medoid
                    print(f"City {city} reassigned to medoid {best_medoid} with distance {min_distance}.")
                # Do not add the current cluster to the new_clusters list (remove it)
            else:
                # If the cluster is valid (not empty or too small), keep it
                new_clusters.append(clusters[cluster_idx])
           

        # We may have fewer clusters after removal, update k accordingly
        k = len(new_clusters)
        print(f"\n New Number of Clusters: {k}")
        #print(f"New Cluster Assignments: {new_labels}")
        #print(f"New Clusters: {new_clusters}")
        print(f"New Number of Cities: {len(new_labels)}")

        # Return the updated clusters and labels
        return new_clusters, new_labels, k
    
    def calculate_intra_cluster_distance(self, distance_matrix, labels, medoids):
        """
        Calculates the intra-cluster distance for each cluster, considering cities assigned to that cluster.
        
        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - labels (numpy.ndarray): Array containing the cluster assignments for each city.
        - medoids (list): List of current medoids (cluster centers).
        
        Returns:
        - intra_cluster_distances (dict): Dictionary with cluster index as the key and the total intra-cluster distance as the value.
        """
        print("\nCalculating Intra-Cluster Distances:")

        k = len(medoids)  # Number of clusters
        intra_cluster_distances = {}  # To store the intra-cluster distances
        
        # Iterate over each cluster
        for cluster in range(k):
            cluster_cities = np.where(labels == medoids[cluster])[0]  # Get cities assigned to the current cluster
            total_intra_cluster_distance = 0  # To accumulate the total intra-cluster distance
            #print(f"\nCluster {cluster}: cluster_cities: {cluster_cities} number of cities: {len(cluster_cities)}")
            
            # Only proceed if the cluster is not empty
            if len(cluster_cities) > 0:
                for city in cluster_cities:
                    # Distance from the medoid to the current city
                    distance = distance_matrix[medoids[cluster], city]
                    #print(f"Cluster: {medoids[cluster]}  & City {city} Distance: {distance}")
                    
                    # If the distance is finite, add it to the total distance
                    if np.isfinite(distance):
                        total_intra_cluster_distance += distance
                    else:
                        # If the distance is infinite, find the closest city in the cluster with a finite distance
                        closest_city = None
                        closest_distance = np.inf
                        dist_closest_to_meloid = 0
                        
                        for other_city in cluster_cities:
                            if np.isfinite(distance_matrix[city, other_city]) and distance_matrix[city, other_city] < closest_distance and np.isfinite(distance_matrix[medoids[cluster], other_city]):
                                closest_city = other_city
                                closest_distance = distance_matrix[city, other_city]
                                dist_closest_to_meloid = distance_matrix[medoids[cluster], closest_city]    
                        
                        # Add the distance to the closest city to the total distance if found
                        if closest_city is not None:
                            total_intra_cluster_distance += closest_distance + dist_closest_to_meloid

            # Store the total intra-cluster distance for this cluster
            intra_cluster_distances[cluster] = total_intra_cluster_distance

        # Return the dictionary of intra-cluster distances
        intra_cluster_distances = np.array(list(intra_cluster_distances.values()))
        #print(f"Intra-Cluster Distances: {intra_cluster_distances}")

        return intra_cluster_distances

    

    def verify_unique_cluster_assignment(self, labels, k):
        """
        Verify that each city has been assigned to exactly one cluster.
        
        Parameters:
        - labels (numpy.ndarray): The cluster assignment for each city.
        - k (int): The number of clusters.
        
        Returns:
        - bool: True if all cities are assigned to exactly one cluster, False otherwise.
        """
        print("\nVerifying Cluster Assignments:")
        print(f"Number of Clusters: {k}")
        print(f"Number of Cities: {len(labels)}")
        print(f"Cluster Assignments: {labels}")
        # Check if the label for each city is within the valid range (0 to k-1)
        valid_assignment = np.all((labels >= 0) & (labels < len(labels)))
        
        if not valid_assignment:
            print("Error: Some cities are assigned to an invalid cluster.")
            return False
        
        # Check that each city is assigned to exactly one cluster
        unique_labels = np.unique(labels)
        
        if len(unique_labels) != k:
            print(f"Error: The number of unique clusters is {len(unique_labels)}, but expected {k}.")
            return False
        
        # If all checks pass
        print("All cities are assigned to exactly one cluster.")
        return True



    def plot_intra_cluster_distances(self,distance_matrix,inter_cluster_distance ,labels, medoids, k):
        """
        Plots the intra-cluster distances for each cluster.
        
        Parameters:
        - distance_matrix (numpy.ndarray): The distance matrix, where distance_matrix[i, j] is the distance between city i and city j.
        - labels (numpy.ndarray): The array of cluster assignments for each city.
        - medoids (numpy.ndarray): The medoids of the clusters.
        - k (int): The number of clusters.
        """
        print("\nPlotting Intra-Cluster Distances:")
        print(f"Number of Clusters: {k}")
        print(f"Number of Cities: {len(labels)}")
        print(f"Cluster Assignments: {labels}")
        print(f"Medoids: {medoids}")
        print(f"Inter-Cluster Distances: {inter_cluster_distance}")
        cluster_cities = []
        for cluster in range(k):
            cluster_cities.append(len(np.where(labels == medoids[cluster])[0]))
       
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns

        # Plot the intra-cluster distances
        axes[0].bar(range(k), inter_cluster_distance, color='skyblue')
        axes[0].set_xlabel('Cluster Index')
        axes[0].set_ylabel('Intra-Cluster Distance')
        axes[0].set_title('Intra-Cluster Distances for Each Cluster')
        axes[0].set_xticks(range(k))

        # Plot the number of cities in each cluster
        axes[1].bar(range(k), cluster_cities, color='skyblue')
        axes[1].set_xlabel('Cluster Index')
        axes[1].set_ylabel('Number of Cities')
        axes[1].set_title('Number of Cities in Each Cluster')
        axes[1].set_xticks(range(k))

        # Show both plots
        plt.tight_layout()  # Adjusts layout for better spacing
        plt.show()



