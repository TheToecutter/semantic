#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:29:53 2025

@author: stevemolloy
"""
"""
Main script to orchestrate the setup and visualization of the fiber bundle model.
"""
from language_definition import ToyLanguage
from space_generator import FillerSpaceGenerator
from fiber_bundle import FiberBundle
from dynamics import generate_sentence_trajectory
from visualizer import plot_combined # This is the main function to call for plotting
import numpy as np
from sklearn.decomposition import PCA # <-- ADDED IMPORT

def main():
    # --- Step 1: Initializing Toy Language ---
    print("--- Step 1: Initializing Toy Language ---")
    language = ToyLanguage()

    # --- Step 2: Generating Shared Filler Space ---
    print("\n--- Step 2: Generating Shared Filler Space ---")
    FILLER_SPACE_DIM = 10
    space_gen = FillerSpaceGenerator(model_name='word2vec-google-news-300')
    filler_space = space_gen.generate_filler_space(language.terminals, FILLER_SPACE_DIM)

    # --- Step 3: Setting Up the Fiber Bundle ---
    print("\n--- Step 3: Setting Up the Fiber Bundle ---")
    bundle = FiberBundle(language, filler_space)
    print("Fiber bundle structure initialized.")

    # --- Step 4: Generating a sentence and its trajectory ---
    target_gist = ('cat', 'saw', 'dog')
    print(f"\n--- Step 4: Generating sentence for gist: {target_gist} ---")
    
    # The dynamical system returns all necessary data for plotting
    final_sentence, trajectory, words_added, settling_indices = generate_sentence_trajectory(target_gist, bundle)

    print(f"\nFinal Sentence: '{final_sentence}'")

    # --- Step 5: Preparing data for Base Space visualization ---
    print("\n--- Step 5: Preparing data for Base Space visualization ---")
    corpus_size = 500
    corpus_sentences = [language.generate_sentence() for _ in range(corpus_size)]
    
    # Filter out sentences that couldn't be fully represented
    valid_corpus_sentences = [s for s in corpus_sentences if all(word in filler_space for word in s.split())]
    
    corpus_gists = [bundle.project_to_base_space(bundle.represent_sentence_in_total_space(s)) for s in valid_corpus_sentences]
    corpus_gists_matrix = np.array(corpus_gists)
    
    example_total_vector = bundle.represent_sentence_in_total_space(final_sentence)
    example_gist_vector = bundle.project_to_base_space(example_total_vector)

    # Use PCA to reduce the dimensionality of the base space for plotting
    if corpus_gists_matrix.shape[0] > 3:
        pca_base = PCA(n_components=3)
        corpus_gists_3d = pca_base.fit_transform(corpus_gists_matrix)
        example_gist_3d = pca_base.transform(example_gist_vector.reshape(1, -1))[0]
    else:
        print("Corpus size too small for 3D visualization.")
        corpus_gists_3d = np.array([[0,0,0]])
        example_gist_3d = np.array([0,0,0])

    # --- Step 6: Visualizing the plots ---
    print("\n--- Step 6: Visualizing the plots ---")
    # Call the main plotting function with all the prepared data
    plot_combined(corpus_gists_3d, example_gist_3d, trajectory, words_added, settling_indices, final_sentence)

if __name__ == "__main__":
    main()


    