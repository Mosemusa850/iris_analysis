Iris Data Analysis and VisualizationThis project demonstrates data analysis and visualization using Pandas and Matplotlib, inspired by the Ubuntu philosophy of "I am because we are." It loads the Iris dataset, performs exploratory data analysis, and creates four types of visualizations to uncover insights, fostering community through shared knowledge, respect through robust error handling, sharing via saved outputs, and practicality with a reusable tool.Project OverviewThe script (iris_analysis.py) fulfills the requirements of the Analyzing Data with Pandas and Visualizing Results with Matplotlib assignment by:Loading the Iris dataset from sklearn.datasets.
Exploring the dataset (structure, missing values) using Pandas.
Analyzing numerical columns and grouping by species for insights.
Creating four visualizations (line chart, bar chart, histogram, scatter plot) with Matplotlib and Seaborn.
Saving results to a text file and visualizations to PNG files.
Handling errors gracefully for file operations and data processing.

Filesiris_analysis.py  Main Python script that loads, analyzes, and visualizes the Iris dataset.
Performs data exploration (head, dtypes, isnull().sum), analysis (describe, groupby), and creates four visualizations.
Saves results to analysis_summary.txt and visualizations to the Visualizations directory.

requirements.txt  Lists dependencies: pandas, matplotlib, seaborn, scikit-learn.

analysis_summary.txt (generated)  Contains exploration results (first rows, data types, missing values), statistical summaries, grouped analysis, and findings.

Visualizations/ (generated directory)  Stores four PNG files:line_plot.png: Sepal length trend over dataset index.
bar_plot.png: Average petal length by species.
histogram.png: Distribution of sepal width.
scatter_plot.png: Sepal length vs. petal length by species.

