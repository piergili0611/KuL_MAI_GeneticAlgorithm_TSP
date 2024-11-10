import matplotlib.pyplot as plt
import numpy as np

class ClusterDashboard:
    def __init__(self):
        
        self.cluster_plots = {}  # Dictionary to store the pre-made sequence and fitness plots for each cluster

    def add_cluster_plots(self, cluster_id, sequence_fig, fitness_fig):
        """
        Store the pre-made sequence and fitness figures for a specific cluster.
        
        Parameters:
        - cluster_id (int): The ID of the cluster
        - sequence_fig (matplotlib.figure.Figure): Pre-made figure for the city sequence plot
        - fitness_fig (matplotlib.figure.Figure): Pre-made figure for the fitness plot
        """
        # Store both figures in the dictionary for the cluster ID
        self.cluster_plots[cluster_id] = {
            "sequence": sequence_fig,
            "fitness": fitness_fig
        }

    def show_all_cluster_plots(self):
        """
        Display sequence and fitness plots for all clusters in a grid layout using static figures.
        """
        num_clusters = len(self.cluster_plots)
        print(f"Displaying plots for {num_clusters} clusters.")
        print(f"num_clusters: {num_clusters}")   
        print("Cluster Plots: ", self.cluster_plots)  # Debug print statement
        if num_clusters == 0:
            print("No plots found for any cluster.")
            return

        # Determine grid size for subplots
        cols = 2  # One column for the sequence plot, one for the fitness plot
        rows = num_clusters  # One row per cluster

        # Create a larger figure with subplots for each cluster's sequence and fitness plots
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * num_clusters))

        # Flatten axes for easier indexing when num_clusters == 1
        if num_clusters == 1:
            axes = np.array([axes])

        for i, (cluster_id, plots) in enumerate(self.cluster_plots.items()):
            # Each cluster gets a row with two subplots (sequence and fitness)
            ax_sequence = axes[i, 0]
            ax_fitness = axes[i, 1]

            # Display the sequence plot by drawing it on the specified axes
            sequence_fig = plots["sequence"]
            sequence_fig_ax = sequence_fig.gca()  # Get the current axis from the figure
            
            if sequence_fig_ax.lines:  # Check if there are any lines in the plot
                ax_sequence.plot(*sequence_fig_ax.lines[0].get_data())  # Re-plot data on ax_sequence
            else:
                ax_sequence.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

            ax_sequence.set_title(f"Cluster {cluster_id} - City Sequence")
            ax_sequence.axis("off")

            # Display the fitness plot by drawing it on the specified axes
            fitness_fig = plots["fitness"]
            fitness_fig_ax = fitness_fig.gca()  # Get the current axis from the figure
            
            if fitness_fig_ax.lines:  # Check if there are any lines in the plot
                ax_fitness.plot(*fitness_fig_ax.lines[0].get_data())  # Re-plot data on ax_fitness
            else:
                ax_fitness.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center')

            ax_fitness.set_title(f"Cluster {cluster_id} - Fitness Evolution")
            ax_fitness.grid(True)

        plt.tight_layout()
        plt.show()