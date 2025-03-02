# Imports
from flask import Flask, render_template, request, send_from_directory
import os
import shutil
import r0818807
from r0818807 import results_sim_folder

class App:
    def __init__(self, data_folder="Data/"):
        self.data_folder = data_folder
        self.algorithm = r0818807.r0818807()
        self.app = Flask(__name__)

    def run_simulation(self, filename, clusters, local_search, generateDataSets):
        '''
        - Run the simulation
        - Gets the results from the algorithm for plots, etc
        '''

        filepath = os.path.join(self.data_folder, filename)
        self.algorithm.run(filename=filepath, clusters=clusters, local_search=local_search, generateDataSets=generateDataSets)

    def generate_plots(self):
        '''
        - Generates Plot from data
        - Copies the generated HTML plot files from the results_sim_folder to the static folder
        '''
        print("Generating Plots")
        from r0818807 import results_sim_folder

        # List of plot filenames to copy
        plot_filenames = [
            "fitness_plot.html",
            "timing_plot.html",
            "objective_distance_plot.html",
            "hamming_distance_plot.html"
        ]

        # Ensure the static/plots folder exists
        static_plots_folder = os.path.join("static", "plots")
        os.makedirs(static_plots_folder, exist_ok=True)

        plot_urls = []
        print(f"Checking for plot files in {results_sim_folder}")

        for plot_filename in plot_filenames:
            plot_path = os.path.join(results_sim_folder, plot_filename)
            print(f"Checking for plot file: {plot_path}")
            if os.path.exists(plot_path):
                static_plot_path = os.path.join(static_plots_folder, plot_filename)
                try:
                    shutil.copy(plot_path, static_plot_path)
                    plot_url = f"/static/plots/{plot_filename}"  # Generate URL for Flask
                    plot_urls.append(plot_url)
                    print(f"Plot copied to {static_plot_path} and URL added: {plot_url}")
                except Exception as e:
                    print(f"Error copying plot {plot_filename}: {e}")
            else:
                print(f"Plot file {plot_filename} not found at {plot_path}")

        if not plot_urls:
            print("No plots were generated or found.")
        return plot_urls

# Initialize Flask & Algorithm
app = Flask(__name__)
genetic_app = App()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            filename = request.form["filename"]
            clusters = "clusters" in request.form  # True if checked
            local_search = "local_search" in request.form  # True if checked
            generateDataSets = "generate_data_sets" in request.form  # True if checked

            print(f"Filename: {filename}, Clusters: {clusters}, Local Search: {local_search}, Generate Data Sets: {generateDataSets}")

            genetic_app.run_simulation(filename, clusters, local_search, generateDataSets=generateDataSets)

            # Generate and get the URLs of the HTML plot files
            plot_urls = genetic_app.generate_plots()
            print(f"Plot URLs: {plot_urls}")

            # Return the rendered template with plot URLs
            return render_template("index.html", plot_urls=plot_urls)

        except Exception as e:
            print(f"Error during simulation: {e}")
            return render_template("index.html", plot_urls=None)

    # In case of GET request or no form submission
    return render_template("index.html", plot_urls=None)

@app.route("/static/plots/<filename>")
def serve_plot(filename):
    """
    Serve the plot files from the static/plots directory.
    """
    return send_from_directory(os.path.join("static", "plots"), filename)

if __name__ == "__main__":
    app.run(debug=True)
