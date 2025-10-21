#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:13:50 2025

@author: stevemolloy
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_base_space_gist(ax, corpus_gists_3d, example_gist_3d):
    """
    Plots the base space gist cloud and a single highlighted example gist.
    """
    # Plot the corpus cloud
    ax.scatter(corpus_gists_3d[:, 0], corpus_gists_3d[:, 1], corpus_gists_3d[:, 2], c='blue', alpha=0.1, label='Corpus Gists')
    
    # Plot the specific example gist
    ax.scatter(example_gist_3d[0], example_gist_3d[1], example_gist_3d[2], c='red', s=100, edgecolor='black', label='Example Gist')
    ax.text(example_gist_3d[0], example_gist_3d[1], example_gist_3d[2], ' Gist', color='red')
    
    ax.set_title('Base Space (Gists)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

def plot_total_space_trajectory(ax, trajectory_3d, words_added, settling_indices):
    """
    Plots the 3D projection of the sentence generation trajectory in the total space.
    It uses the settling_indices to correctly label the points where words were settled.
    """
    # Plot the full trajectory path
    ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 'b-', alpha=0.6, label='Trajectory')
    
    # Correctly label the points where words were settled using settling_indices
    labels = ['START'] + words_added
    for i, index in enumerate(settling_indices):
        if index < len(trajectory_3d):
            point = trajectory_3d[index]
            label = labels[i]
            ax.scatter(point[0], point[1], point[2], c='red', s=50)
            ax.text(point[0], point[1], point[2], f" {label}", color='red')

    # Mark the end point
    ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2], c='black', s=100, marker='x', label='END')
    
    ax.set_title('Total Space Trajectory (Sentence Generation)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()


def plot_combined(corpus_gists_3d, example_gist_3d, total_space_trajectory, words_added, settling_indices, final_sentence):
    """
    Creates a figure with two subplots: one for the base space and one for the total space trajectory.
    """
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Dynamical System Analysis\nFinal Sentence: '{final_sentence}'", fontsize=16)

    # --- First subplot: Base Space ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_base_space_gist(ax1, corpus_gists_3d, example_gist_3d)

    # --- Second subplot: Total Space Trajectory ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Use PCA to reduce the high-dimensional trajectory to 3D for plotting
    if total_space_trajectory.shape[0] > 3:
        pca_traj = PCA(n_components=3)
        trajectory_3d = pca_traj.fit_transform(total_space_trajectory)
        plot_total_space_trajectory(ax2, trajectory_3d, words_added, settling_indices)
    else:
        ax2.text(0.5, 0.5, 0.5, "Trajectory too short to plot.", ha='center', va='center')


    print("\nDisplaying combined plot...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


