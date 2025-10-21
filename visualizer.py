#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:13:50 2025

@author: stevemolloy
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_filler_space_tsne(filler_space):
    """
    Creates and displays a t-SNE visualization of the filler space.

    Args:
        filler_space (dict): The generated filler space.
    """
    print("Generating t-SNE visualization of the filler space...")
    
    words = list(filler_space.keys())
    vectors = np.array(list(filler_space.values()))

    if len(words) <= 1:
        print("Cannot generate t-SNE plot for one or zero words.")
        return

    # Perform t-SNE. Perplexity must be less than the number of samples.
    tsne = TSNE(n_components=2, 
                perplexity=min(5, len(words) - 1), 
                random_state=42, 
                init='pca', 
                learning_rate='auto')
    vectors_2d = tsne.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(12, 12))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
    
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10)
        
    plt.title('t-SNE Visualization of the Semantic Filler Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    print("Displaying plot...")
    plt.show()

def plot_base_space_gist(ax, corpus_gists_3d, example_gist_3d):
    """Plots the base space gist visualization on a given axis."""
    ax.scatter(corpus_gists_3d[:, 0], corpus_gists_3d[:, 1], corpus_gists_3d[:, 2], c='blue', alpha=0.1, label='Corpus Gists')
    ax.scatter(example_gist_3d[0], example_gist_3d[1], example_gist_3d[2], c='red', s=100, label='Example Gist', edgecolors='black')
    ax.set_title('Base Space (Gist)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend()

def plot_total_space_trajectory(ax, trajectory, words_added, settling_indices):
    """Plots the total space trajectory visualization on a given axis."""
    pca_traj = PCA(n_components=3)
    trajectory_3d = pca_traj.fit_transform(np.array(trajectory))
    
    # Plot the full trajectory path
    ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    
    # Plot the start and end markers
    ax.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2], c='green', s=150, label='Start', marker='o', edgecolors='black', zorder=5)
    ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2], c='black', s=150, label='End', marker='X', zorder=5)
    
    # Plot labels for each settled word at the correct point in the trajectory
    # Use the settling_indices to get the correct points from the 3D trajectory
    label_points = trajectory_3d[settling_indices]
    for i, word in enumerate(words_added):
        point = label_points[i+1] # +1 to skip the 'Start' point's index
        ax.scatter(point[0], point[1], point[2], s=80, edgecolors='black', zorder=5)
        ax.text(point[0], point[1], point[2], f'  {word}', zorder=6, fontsize=9)

    ax.set_title('Total Space (Sentence Trajectory)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend()

def plot_combined(corpus_gists_3d, example_gist_3d, trajectory, words_added, settling_indices, title_sentence):
    """Creates a combined plot with both base space and total space visualizations."""
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Dynamical Trajectory for: '{title_sentence}'", fontsize=16)

    # Base Space Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_base_space_gist(ax1, corpus_gists_3d, example_gist_3d)

    # Total Space Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_total_space_trajectory(ax2, trajectory, words_added, settling_indices)

    print("Displaying combined plot...")
    plt.show()


