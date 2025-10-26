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
from visualizer import plot_combined, plot_filler_space_tsne, plot_fiber_grid_2d, plot_base_grid_3d
from tda import plot_gist_persistence, plot_betti_curves, plot_distance_histogram
import numpy as np
from sklearn.decomposition import PCA
import itertools
import random

def generate_full_gist_manifold(language: ToyLanguage, bundle: FiberBundle) -> np.ndarray:
    """
    Generates all 75 (or fewer) valid gist vectors.
    """
    print("\n--- Generating Full Gist Manifold ---")
    
    # Get all valid nouns and verbs from the filler space
    valid_nouns = [n for n in language.terminal_map['N'] if n in bundle.filler_space]
    valid_verbs = [v for v in language.terminal_map['V'] if v in bundle.filler_space]
    
    if not valid_nouns or not valid_verbs:
        print("Warning: Not enough valid nouns or verbs in filler space to generate manifold.")
        return np.array([])
        
    all_gists = []
    
    # Use itertools.product to get all combinations
    for agent, action, patient in itertools.product(valid_nouns, valid_verbs, valid_nouns):
        # We must create a "valid" sentence string that represent_sentence_in_total_space
        # can parse. The simplest is just "agent action patient".
        # This works because the parser finds the verb and splits [cite: fiber_bundle.py]
        sentence = f"{agent} {action} {patient}"
        
        try:
            total_vec = bundle.represent_sentence_in_total_space(sentence)
            base_vec = bundle.project_to_base_space(total_vec)
            all_gists.append(base_vec)
        except ValueError as e:
            print(f"Skipping gist '{sentence}': {e}")
            
    print(f"Generated {len(all_gists)} unique gist vectors.")
    return np.array(all_gists)

def generate_full_total_space_manifold(language: ToyLanguage, bundle: FiberBundle) -> np.ndarray:
    """
    Generates all 7,500 (or fewer) valid sentence vectors in the Total Space.
    """
    print("\n--- Generating Full Total Space Manifold (this may take a moment) ---")
    
    # Get all valid words from the filler space for each category
    valid_dets = [w for w in language.terminal_map['Det'] if w in bundle.filler_space]
    valid_adjs = [w for w in language.terminal_map['Adj'] if w in bundle.filler_space]
    valid_nouns = [w for w in language.terminal_map['N'] if w in bundle.filler_space]
    valid_verbs = [v for v in language.terminal_map['V'] if v in bundle.filler_space]

    if not all([valid_dets, valid_adjs, valid_nouns, valid_verbs]):
        print("Warning: Missing words in a category. The total space will be smaller than expected.")

    # 1. Generate all possible Noun Phrases (as lists of words)
    all_nps = []
    # NPs without adjectives ('Det N')
    for det, noun in itertools.product(valid_dets, valid_nouns):
        all_nps.append([det, noun])
    # NPs with adjectives ('Det Adj N')
    for det, adj, noun in itertools.product(valid_dets, valid_adjs, valid_nouns):
        all_nps.append([det, adj, noun])
        
    print(f"Generated {len(all_nps)} unique Noun Phrases.")

    # 2. Generate all sentences and their vectors
    all_sentence_vectors = []
    sentence_count = 0
    total_possible_sentences = len(all_nps) * len(valid_verbs) * len(all_nps)
    
    for np1_words, verb, np2_words in itertools.product(all_nps, valid_verbs, all_nps):
        sentence_words = np1_words + [verb] + np2_words
        sentence_str = " ".join(sentence_words)
        
        try:
            total_vec = bundle.represent_sentence_in_total_space(sentence_str)
            all_sentence_vectors.append(total_vec)
            sentence_count += 1
            if sentence_count % 1000 == 0:
                print(f"  ...processed {sentence_count}/{total_possible_sentences} sentences.")
        except ValueError as e:
            # This shouldn't happen with this deterministic generation, but good practice
            print(f"Skipping sentence '{sentence_str}': {e}")
            
    print(f"Generated {len(all_sentence_vectors)} unique total space vectors.")
    return np.array(all_sentence_vectors)

def generate_fiber_for_gist(gist: tuple, language: ToyLanguage, bundle: FiberBundle) -> np.ndarray:
    """
    Generates all syntactic variations for a single gist. This is the "fiber"
    of the fiber bundle over a single point in the base space.
    """
    agent, action, patient = gist
    print(f"\n--- Generating Fiber for Gist: ('{agent}', '{action}', '{patient}') ---")

    # Get all valid words from the filler space for each category
    valid_dets = [w for w in language.terminal_map['Det'] if w in bundle.filler_space]
    valid_adjs = [w for w in language.terminal_map['Adj'] if w in bundle.filler_space]

    # Check if the specific gist nouns are valid
    if agent not in bundle.filler_space or action not in bundle.filler_space or patient not in bundle.filler_space:
        print("Warning: One of the words in the requested gist is not in the filler space. Cannot generate fiber.")
        return np.array([])

    # 1. Generate all possible Noun Phrases for a FIXED noun
    def get_nps_for_noun(noun):
        nps = []
        # NPs without adjectives ('Det N')
        for det in valid_dets:
            nps.append([det, noun])
        # NPs with adjectives ('Det Adj N')
        for det, adj in itertools.product(valid_dets, valid_adjs):
            nps.append([det, adj, noun])
        return nps

    agent_nps = get_nps_for_noun(agent)
    patient_nps = get_nps_for_noun(patient)
    print(f"Generated {len(agent_nps)} NP variations for agent and {len(patient_nps)} for patient.")

    # 2. Generate all sentence vectors for the fixed gist
    fiber_vectors = []
    for np1_words, np2_words in itertools.product(agent_nps, patient_nps):
        sentence_words = np1_words + [action] + np2_words
        sentence_str = " ".join(sentence_words)
        try:
            total_vec = bundle.represent_sentence_in_total_space(sentence_str)
            fiber_vectors.append(total_vec)
        except ValueError as e:
            print(f"Skipping sentence '{sentence_str}': {e}")
            
    print(f"Generated {len(fiber_vectors)} unique vectors in the fiber.")
    return np.array(fiber_vectors)


def main():
    # --- Control flags for TDA ---
    RUN_TDA_ON_BASE_SPACE = True
    RUN_TDA_ON_TOTAL_SPACE = True  # Set to False to skip the long computation
    RUN_TDA_ON_SINGLE_FIBER = True # Set to False to skip the new experiment

    # --- Seed for reproducibility ---
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


    # --- Step 1: Initializing Toy Language ---
    print("--- Step 1: Initializing Toy Language ---")
    language = ToyLanguage()

    # --- Step 2: Generating Shared Filler Space ---
    print("\n--- Step 2: Generating Shared Filler Space ---")
    FILLER_SPACE_DIM = 10
    space_gen = FillerSpaceGenerator(model_name='word2vec-google-news-300')
    filler_space = space_gen.generate_filler_space(language.terminals, FILLER_SPACE_DIM)

    # --- Step 2a: Visualize the Filler Space ---
    print("\n--- Step 2a: Visualizing the Filler Space ---")
    plot_filler_space_tsne(filler_space)

    # --- Step 3: Setting Up the Fiber Bundle ---
    print("\n--- Step 3: Setting Up the Fiber Bundle ---")
    bundle = FiberBundle(language, filler_space)
    print("Fiber bundle structure initialized.")

    # --- Step 4: Generating a sentence and its trajectory ---
    target_gist = ('cat', 'saw', 'dog')
    print(f"\n--- Step 4: Generating sentence for gist: {target_gist} ---")
    
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
        
    # --- Step 5a: Topological Data Analysis (Base Space) ---
    if RUN_TDA_ON_BASE_SPACE:
        print("\n--- Step 5a: Performing Topological Data Analysis (Base Space) ---")
        HOMOLOGY_MAX_DIM = 2 # Define max dim to use for all TDA plots

        # 1. Analyze the *random sample*
        plot_gist_persistence(corpus_gists_matrix, 
                              title=f"Persistence of Random Gist Corpus (N={corpus_gists_matrix.shape[0]})",
                              max_dim=HOMOLOGY_MAX_DIM)
        
        # 2. Analyze the *full 75-point manifold*
        full_gist_manifold_matrix = generate_full_gist_manifold(language, bundle)
        plot_gist_persistence(full_gist_manifold_matrix,
                              title=f"Persistence of Full Gist Manifold (N={full_gist_manifold_matrix.shape[0]})",
                              max_dim=HOMOLOGY_MAX_DIM)

        # 3. Plot Betti Curves for the full manifold
        if full_gist_manifold_matrix.size > 0:
            plot_betti_curves(full_gist_manifold_matrix,
                              title=f"Betti Curves for Full Gist Manifold (N={full_gist_manifold_matrix.shape[0]})",
                              max_dim=HOMOLOGY_MAX_DIM)

        # 4. Plot histogram of pairwise distances
        if full_gist_manifold_matrix.size > 0:
            plot_distance_histogram(full_gist_manifold_matrix,
                                      title=f"Pairwise Distances in Full Gist Manifold (N={full_gist_manifold_matrix.shape[0]})")

        # 5. Visualize the 3D grid structure of the full manifold
        # We assume 5 nouns, 3 verbs -> 5x3x5 = 75 points
        # If this assumption is wrong, the plot will be skipped.
        if full_gist_manifold_matrix.shape[0] == 75:
            plot_base_grid_3d(full_gist_manifold_matrix, 
                              grid_dims=(5, 3, 5),
                              title="PCA Visualization of 5x3x5 Base Manifold Grid")
        elif full_gist_manifold_matrix.size > 0:
             print(f"Skipping Base Manifold grid plot: expected 75 points but found {full_gist_manifold_matrix.shape[0]}.")


    # --- Step 5b: Topological Data Analysis (Total Space) ---
    if RUN_TDA_ON_TOTAL_SPACE:
        print("\n--- Step 5b: Performing Topological Data Analysis (Total Space) ---")
        HOMOLOGY_MAX_DIM = 2 # Define max dim to use for all TDA plots
        full_total_space_matrix = generate_full_total_space_manifold(language, bundle)

        # Note: TDA on this many points will be very slow and memory-intensive.
        if full_total_space_matrix.size > 0:
            plot_gist_persistence(full_total_space_matrix,
                                  title=f"Persistence of Full Total Space Manifold (N={full_total_space_matrix.shape[0]})",
                                  max_dim=HOMOLOGY_MAX_DIM)

            plot_betti_curves(full_total_space_matrix,
                              title=f"Betti Curves for Full Total Space Manifold (N={full_total_space_matrix.shape[0]})",
                              max_dim=HOMOLOGY_MAX_DIM)

            plot_distance_histogram(full_total_space_matrix,
                                      title=f"Pairwise Distances in Full Total Space Manifold (N={full_total_space_matrix.shape[0]})")
            
            # Visualize the 3D grid structure
            if full_total_space_matrix.shape[0] == 1875:
                # We found 5 NPs for 5 nouns = 25 total NPs. 3 verbs.
                # Grid is (25 NPs) x (3 Verbs) x (25 NPs)
                plot_base_grid_3d(full_total_space_matrix, 
                                  grid_dims=(25, 3, 25),
                                  title="PCA Visualization of 25x3x25 Total Space Manifold Grid")
            elif full_total_space_matrix.size > 0:
                 print(f"Skipping Total Space grid plot: expected 1875 points but found {full_total_space_matrix.shape[0]}.")

    
    # --- Step 5c: TDA on a Single Fiber ---
    if RUN_TDA_ON_SINGLE_FIBER:
        print("\n--- Step 5c: Performing TDA on a Single Gist's Fiber ---")
        HOMOLOGY_MAX_DIM = 2
        gist_to_analyze = ('cat', 'saw', 'dog')
        
        fiber_matrix = generate_fiber_for_gist(gist_to_analyze, language, bundle)

        if fiber_matrix.size > 0:
            # Plot the 2D visualization of the grid
            plot_fiber_grid_2d(fiber_matrix, 
                               title=f"PCA Visualization of Fiber Grid for {gist_to_analyze}")

            # Plot the TDA results
            plot_gist_persistence(fiber_matrix,
                                  title=f"Persistence of Fiber for Gist: {gist_to_analyze} (N={fiber_matrix.shape[0]})",
                                  max_dim=HOMOLOGY_MAX_DIM)
            
            plot_betti_curves(fiber_matrix,
                              title=f"Betti Curves for Fiber for Gist: {gist_to_analyze} (N={fiber_matrix.shape[0]})",
                              max_dim=HOMOLOGY_MAX_DIM)
            
            plot_distance_histogram(fiber_matrix,
                                      title=f"Pairwise Distances in Fiber for Gist: {gist_to_analyze} (N={fiber_matrix.shape[0]})")


    # --- Step 6: Visualizing the plots ---
    print("\n--- Step 6: Visualizing the plots ---")
    # Call the main plotting function with all the prepared data
    plot_combined(corpus_gists_3d, example_gist_3d, trajectory, words_added, settling_indices, final_sentence)

if __name__ == "__main__":
    main()


    