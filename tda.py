#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:35:00 2025

@author: stevemolloy
"""
"""
Performs Topological Data Analysis (TDA) on vector spaces.
"""
import ripser
import persim
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, cdist # For efficient distance calculation

def plot_gist_persistence(gist_matrix: np.ndarray, title: str = 'Persistence Diagram', max_dim: int = 1):
    """
    Performs persistent homology on a gist point cloud and plots the diagram.
    
    H0 (dim 0): Represents connected components (clusters).
    H1 (dim 1): Represents loops or holes.
    
    Args:
        gist_matrix (np.ndarray): An (N_points x N_dims) array of gist vectors.
        title (str): The title for the plot.
        max_dim (int): The maximum homology dimension to compute (default 1, for H0 and H1).
    """
    if gist_matrix.shape[0] < 2:
        print(f"Cannot compute persistence with {gist_matrix.shape[0]} points. Skipping.")
        return

    print(f"\n--- Computing Persistent Homology (max_dim={max_dim}) on {gist_matrix.shape[0]} points ---")
    
    # Compute persistence diagrams. 'dgms' is a list [H0, H1, ...]
    diagrams = ripser.ripser(gist_matrix, maxdim=max_dim)['dgms']
    
    print("Plotting persistence diagram...")
    persim.plot_diagrams(diagrams, show=True, title=title)
    plt.show()

def plot_betti_curves(gist_matrix: np.ndarray, title: str = 'Betti Curves', max_dim: int = 2):
    """
    Computes and plots the Betti curves for a given point cloud.
    
    Args:
        gist_matrix (np.ndarray): An (N_points x N_dims) array of gist vectors.
        title (str): The title for the plot.
        max_dim (int): The maximum homology dimension to compute.
    """
    if gist_matrix.shape[0] < 2:
        print(f"Cannot compute Betti curves with {gist_matrix.shape[0]} points. Skipping.")
        return

    print(f"\n--- Computing Betti Curves (max_dim={max_dim}) ---")
    
    diagrams = ripser.ripser(gist_matrix, maxdim=max_dim)['dgms']
    
    # Collect all unique birth and death times to define our epsilon steps for plotting
    events = []
    for dgm in diagrams:
        if dgm.size > 0:
            events.extend(dgm.flatten())
    
    # Remove duplicates, sort, and filter out infinity for a clean plot axis
    epsilon_steps = sorted(list(set(events)))
    epsilon_steps = [e for e in epsilon_steps if e != np.inf]
    
    if not epsilon_steps:
        print("No finite persistence intervals found. Cannot plot Betti curves.")
        return

    betti_numbers = [[] for _ in range(max_dim + 1)]
    num_points = gist_matrix.shape[0]

    for epsilon in epsilon_steps:
        # Calculate b0: Starts at num_points and decreases as components merge
        h0_dgm = diagrams[0]
        deaths_h0 = np.sum(h0_dgm[:, 1] <= epsilon)
        betti_numbers[0].append(num_points - deaths_h0)

        # Calculate b1, b2, ...
        for k in range(1, max_dim + 1):
            dgm = diagrams[k]
            if dgm.size > 0:
                births = np.sum(dgm[:, 0] <= epsilon)
                deaths = np.sum(dgm[:, 1] <= epsilon)
                betti_numbers[k].append(births - deaths)
            else:
                betti_numbers[k].append(0)

    # Plotting
    fig, axes = plt.subplots(max_dim + 1, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=16)

    for k in range(max_dim + 1):
        ax = axes[k]
        # Use 'post' to create a correct step function plot
        ax.step(epsilon_steps, betti_numbers[k], where='post')
        ax.set_ylabel(f'$\\beta_{k}$', fontsize=14)
        ax.set_title(f'Betti Number $\\beta_{k}$ vs. $\\epsilon$')
        ax.grid(True)

    axes[-1].set_xlabel('$\\epsilon$ (Scale)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Displaying Betti curves plot...")
    plt.show()

def plot_distance_histogram(gist_matrix: np.ndarray, title: str = 'Pairwise Distance Distribution'):
    """
    Calculates and plots a histogram of all pairwise Euclidean distances
    and a histogram of nearest neighbor distances.
    
    Args:
        gist_matrix (np.ndarray): An (N_points x N_dims) array of gist vectors.
        title (str): The title for the plot.
    """
    if gist_matrix.shape[0] < 2:
        print("Not enough points to compute a distance histogram.")
        return
        
    print(f"\n--- Calculating Pairwise Distance Distributions on {gist_matrix.shape[0]} points ---")
    
    # --- Calculate All Pairwise Distances ---
    # pdist calculates the condensed distance matrix (upper triangle), which is efficient
    all_distances = pdist(gist_matrix, 'euclidean')
    
    # --- Calculate Nearest Neighbor Distances ---
    # cdist calculates the full NxN matrix of distances
    full_distance_matrix = cdist(gist_matrix, gist_matrix, 'euclidean')
    # Set diagonal to infinity so we don't pick 0 as the min distance
    np.fill_diagonal(full_distance_matrix, np.inf)
    # Find the minimum distance in each row
    nn_distances = np.min(full_distance_matrix, axis=1)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Subplot 1: All Pairwise Distances
    axes[0].hist(all_distances, bins='auto', color='navy', alpha=0.75)
    axes[0].set_title('All Pairwise Distances (Global Structure)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, axis='y')
    mean_all = np.mean(all_distances)
    axes[0].axvline(mean_all, color='red', linestyle='dashed', linewidth=2)
    axes[0].text(mean_all * 1.05, axes[0].get_ylim()[1] * 0.9, f'Mean: {mean_all:.2f}', color='red')

    # Subplot 2: Nearest Neighbor Distances
    axes[1].hist(nn_distances, bins='auto', color='darkgreen', alpha=0.75)
    axes[1].set_title('Nearest Neighbor Distances (Local Structure)')
    axes[1].set_ylabel('Frequency (Number of Points)')
    axes[1].set_xlabel('Euclidean Distance (corresponds to $\\epsilon$ scale)')
    axes[1].grid(True, axis='y')
    mean_nn = np.mean(nn_distances)
    axes[1].axvline(mean_nn, color='red', linestyle='dashed', linewidth=2)
    axes[1].text(mean_nn * 1.05, axes[1].get_ylim()[1] * 0.9, f'Mean: {mean_nn:.2f}', color='red')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Displaying distance histograms...")
    plt.show()


    