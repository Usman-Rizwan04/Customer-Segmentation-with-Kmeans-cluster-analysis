#!/usr/bin/env python3
"""
Interactive Customer Segmentation Tool with GUI Buttons & AI-driven Clustering

Features:
 1. Load CSV of customer demographics & behavior
 2. Display first 5 records in a Tkinter table
 3. Dynamically create a button per column; click to view its distribution chart
 4. "Cluster Analysis" button to:
     a. Plot Elbow (Inertia) & Silhouette diagnostics
     b. Prompt for k and run KMeans
     c. Display PCA scatter of clusters
     d. Recommend marketing strategies per cluster
 5. Return to GUI for further analysis without restart

Dependencies: tkinter, pandas, numpy, matplotlib, seaborn, scikit-learn
"""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation Tool")
        self.root.geometry("1000x750")
        self.root.configure(bg='#f0f4f8')

        # Apply ttk theme
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f4f8')
        style.configure('TLabel', background='#f0f4f8', font=('Helvetica', 11))
        style.configure('TButton', font=('Helvetica', 10, 'bold'), padding=6)
        style.configure('Treeview', font=('Consolas', 10), rowheight=24)
        style.configure('Treeview.Heading', font=('Helvetica', 11, 'bold'))

        self.df = None

        # Layout frames using grid
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=10, padx=10)
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10)
        self.button_frame = ttk.LabelFrame(self.root, text="Attributes", padding=(10,10))
        self.button_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=10)
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=2, column=1, sticky='ew', padx=10, pady=10)

        # Configure grid weights
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Buttons
        ttk.Button(self.top_frame, text="📂 Load CSV", command=self.load_csv).pack(side='top', padx=5)
        ttk.Button(self.control_frame, text="📊 Cluster Analysis", command=self.cluster_analysis).pack(fill='x')
   
    def load_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file: return
        try:
            self.df = pd.read_csv(file)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load file: {e}")
            return
        # Show preview
        self.show_table(self.df.head())
        # Create buttons for each column
        for widget in self.button_frame.winfo_children(): widget.destroy()
        for col in self.df.columns:
            btn = ttk.Button(self.button_frame, text=col, command=lambda c=col: self.plot_attribute(c))
            btn.pack(side='left', padx=3, pady=3)

    def show_table(self, data):
        for w in self.table_frame.winfo_children(): w.destroy()
        cols = list(data.columns)
        tree = ttk.Treeview(self.table_frame, columns=cols, show='headings', height=5)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor='center')
        for _, r in data.iterrows(): tree.insert('', 'end', values=list(r))
        tree.pack(fill='x')

    def plot_attribute(self, col):
        if self.df is None: return

        data = self.df[col].dropna()
        fig, ax = plt.subplots(figsize=(6,4))  # Only create one figure

        if pd.api.types.is_numeric_dtype(self.df[col]):
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
        else:
            counts = data.value_counts()
            if len(counts) <= 5:
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                ax.set_title(f'{col} Composition')
            else:
                sns.countplot(y=data, ax=ax)
                ax.set_title(f'Count of {col}')
                ax.set_xlabel('Count')
                ax.set_ylabel(col)

        fig.tight_layout()
        plt.show()

    def cluster_analysis(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a CSV first.")
            return
        # Select numeric columns only
        numeric = self.df.select_dtypes(include='number').dropna(axis=1)
        if numeric.shape[1] < 3:
            messagebox.showwarning("Insufficient Data", "Need at least 3 numeric features.")
            return
        X = StandardScaler().fit_transform(numeric)
        # Diagnostics
        inertias, sils, ks = [], [], range(2, 11)
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42).fit(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, km.labels_))
        # Elbow & Silhouette
        fig, axs = plt.subplots(1,2,figsize=(12,5))
        axs[0].plot(ks, inertias, 'o-')
        axs[0].set(title='Elbow Method', xlabel='k', ylabel='Inertia')
        axs[1].plot(ks, sils, 'o-')
        axs[1].set(title='Silhouette Score', xlabel='k', ylabel='Score')
        plt.tight_layout(); plt.show()
        # Ask k
        k = simpledialog.askinteger("Choose k", "Enter number of clusters (2-10):", minvalue=2, maxvalue=10)
        if not k: return
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = km.labels_
        # PCA plot
        comps = PCA(2, random_state=42).fit_transform(X)
        plt.figure(figsize=(7,6))
        sns.scatterplot(x=comps[:,0], y=comps[:,1], hue=labels, palette='tab10', s=60)
        plt.title(f'K-Means Clusters (k={k})'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(title='Cluster')
        plt.tight_layout(); plt.show()
        # Recommendations
        self.recommend_strategies(numeric, labels)

    def recommend_strategies(self, df_num, labels):
        # Identify key feature columns
        cols = df_num.columns.tolist()
        income_col = next((c for c in cols if 'Income' in c), None)
        freq_col = next((c for c in cols if 'Frequency' in c), None)
        value_col = next((c for c in cols if 'Transaction' in c or 'Value' in c), None)

        df_num = df_num.copy(); df_num['Cluster'] = labels
        text = "Marketing Strategy Recommendations:" 
        for c in sorted(df_num['Cluster'].unique()):
            sub = df_num[df_num['Cluster']==c]
            inc = sub[income_col].mean() if income_col else None
            freq = sub[freq_col].mean() if freq_col else None
            val = sub[value_col].mean() if value_col else None
            text += f"\nCluster {c}:"
            # Distinct strategy per cluster
            if inc and inc > df_num[income_col].mean():
                text += "\n• Target with premium product lines & VIP perks"
            elif freq and freq > df_num[freq_col].mean():
                text += "\n• Promote subscription/bundles to leverage high engagement"
            elif val and val < df_num[value_col].mean():
                text += "\n• Upsell through curated discount offers"
            else:
                text += "\n• Send personalized recommendations and re-engagement campaigns"
        messagebox.showinfo("Recommendations", text)

if __name__ == '__main__':
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
