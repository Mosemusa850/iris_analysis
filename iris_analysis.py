import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.datasets import load_iris

def load_iris_data():
    """
    Load the Iris dataset from sklearn and convert to a Pandas DataFrame.
    Returns: pandas.DataFrame or None if loading fails.
    """
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        print("✓ Successfully loaded Iris dataset")
        return df
    except Exception as e:
        print(f"✗ Error loading Iris dataset: {str(e)}")
        return None

def explore_data(df, output_file="analysis_summary.txt"):
    """
    Explore the dataset (head, data types, missing values) and save results.
    Parameters:
        df (pandas.DataFrame): DataFrame to explore
        output_file (str): File to save exploration results
    """
    if df is None or df.empty:
        print("✗ Error: No data to explore.")
        return
    
    try:
        # Display first few rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        # Check data types
        print("\nData Types:")
        print(df.dtypes)
        
        # Check missing values
        missing = df.isnull().sum()
        print("\nMissing Values:")
        print(missing)
        
        # Save exploration results
        with open(output_file, "w") as f:
            f.write("Iris Dataset Exploration\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EAT')}\n\n")
            f.write("First 5 rows:\n")
            f.write(str(df.head()) + "\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes) + "\n\n")
            f.write("Missing Values:\n")
            f.write(str(missing) + "\n")
        print(f"✓ Exploration results saved to '{output_file}'")
    
    except PermissionError:
        print(f"✗ Error: Permission denied when saving to '{output_file}'.")
    except Exception as e:
        print(f"✗ Error during exploration: {str(e)}")

def analyze_data(df, output_file="analysis_summary.txt"):
    """
    Perform basic data analysis and append results to output file.
    Parameters:
        df (pandas.DataFrame): DataFrame to analyze
        output_file (str): File to append analysis results
    """
    if df is None or df.empty:
        print("✗ Error: No data to analyze.")
        return
    
    try:
        # Statistical summary
        summary = df.describe()
        print("\nStatistical Summary:")
        print(summary)
        
        # Group by species and compute mean
        grouped = df.groupby('species').mean()
        print("\nMean Values by Species:")
        print(grouped)
        
        # Findings
        findings = "Findings:\n"
        findings += "- The Iris dataset has no missing values.\n"
        findings += "- Sepal length and width vary across species, with 'virginica' having the largest average petal length.\n"
        findings += "- Petal measurements show more variation than sepal measurements.\n"
        print("\n" + findings)
        
        # Append analysis to output file
        with open(output_file, "a") as f:
            f.write("\nStatistical Summary:\n")
            f.write(str(summary) + "\n\n")
            f.write("Mean Values by Species:\n")
            f.write(str(grouped) + "\n\n")
            f.write(findings)
        print(f"✓ Analysis results appended to '{output_file}'")
    
    except PermissionError:
        print(f"✗ Error: Permission denied when appending to '{output_file}'.")
    except Exception as e:
        print(f"✗ Error during analysis: {str(e)}")

def visualize_data(df, output_dir="Visualizations"):
    """
    Create four visualizations and save to output directory.
    Parameters:
        df (pandas.DataFrame): DataFrame to visualize
        output_dir (str): Directory to save visualizations
    """
    if df is None or df.empty:
        print("✗ Error: No data to visualize.")
        return
    
    try:
        # Set Seaborn style for better visuals
        sns.set_style("whitegrid")
        os.makedirs(output_dir, exist_ok=True)
        
        # Line chart: Mean sepal length over pseudo-time (row index)
        plt.figure(figsize=(8, 6))
        plt.plot(df.index, df['sepal length (cm)'], color='blue', label='Sepal Length')
        plt.title('Sepal Length Trend Over Dataset Index')
        plt.xlabel('Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        line_plot = os.path.join(output_dir, 'line_plot.png')
        plt.savefig(line_plot)
        plt.close()
        print(f"✓ Saved line plot to '{line_plot}'")
        
        # Bar chart: Average petal length by species
        plt.figure(figsize=(8, 6))
        grouped = df.groupby('species')['petal length (cm)'].mean()
        plt.bar(grouped.index, grouped, color='green')
        plt.title('Average Petal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Petal Length (cm)')
        bar_plot = os.path.join(output_dir, 'bar_plot.png')
        plt.savefig(bar_plot)
        plt.close()
        print(f"✓ Saved bar plot to '{bar_plot}'")
        
        # Histogram: Distribution of sepal width
        plt.figure(figsize=(8, 6))
        plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        hist_plot = os.path.join(output_dir, 'histogram.png')
        plt.savefig(hist_plot)
        plt.close()
        print(f"✓ Saved histogram to '{hist_plot}'")
        
        # Scatter plot: Sepal length vs. petal length
        plt.figure(figsize=(8, 6))
        for species in df['species'].unique():
            subset = df[df['species'] == species]
            plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], label=species)
        plt.title('Sepal Length vs. Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        scatter_plot = os.path.join(output_dir, 'scatter_plot.png')
        plt.savefig(scatter_plot)
        plt.close()
        print(f"✓ Saved scatter plot to '{scatter_plot}'")
    
    except Exception as e:
        print(f"✗ Error during visualization: {str(e)}")

def main():
    print("Welcome to the Ubuntu Iris Data Analyzer")
    print("A tool for analyzing and visualizing the Iris dataset\n")
    
    try:
        # Load data
        df = load_iris_data()
        
        if df is not None:
            # Explore data
            explore_data(df)
            
            # Analyze data
            analyze_data(df)
            
            # Visualize data
            visualize_data(df)
            
            print("\nAnalysis and visualization completed. Community enriched.")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"✗ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()