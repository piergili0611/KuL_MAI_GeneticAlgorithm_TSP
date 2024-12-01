from r0818807 import r0818807

def main():
    # Create an instance of the r0818807 class
    c = r0818807()

    # Define the filename for the dataset
    filename = 'tour1000.csv'

    # Run the optimization with local search enabled
    c.run(filename=filename, generateDataSets=False, clusters=False, local_search=True)

if __name__ == "__main__":
    main()