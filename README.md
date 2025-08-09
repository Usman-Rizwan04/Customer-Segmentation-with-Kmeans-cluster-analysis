# Customer Segmentation with K-Means Cluster Analysis

Interactive Customer Segmentation Tool with GUI Buttons & AI-driven Clustering

Features:
 1. Loading CSV file of customer demographics & behavior
 2. Display the first 5 records in a Tkinter table
 3. Dynamically create a button per column; click to view its distribution chart
 4. "Cluster Analysis" button to:
     a. Plot Elbow (Inertia) & Silhouette diagnostics
     b. Prompt for k and run KMeans
     c. Display PCA scatter of clusters
     d. Recommend marketing strategies per cluster
 5. Return to the GUI for further analysis without a restart

Libraries: tkinter, pandas, numpy, matplotlib, seaborn, scikit-learn
