#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:28:45 2025

@author: stevemolloy
"""
"""
Contains the function to generate the shared semantic filler space.
It uses a pre-trained Word2Vec model and reduces the dimensionality
of the word vectors using Principal Component Analysis (PCA).
"""
import gensim.downloader as api
from sklearn.decomposition import PCA
import numpy as np
import warnings

class FillerSpaceGenerator:
    """
    Handles the generation of the semantic filler space from a pre-trained
    Word2Vec model.
    """
    def __init__(self, model_name='word2vec-google-news-300'):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Loads the Word2Vec model."""
        if self.model is None:
            print(f"Loading pre-trained '{self.model_name}' model...")
            print("This may take a few minutes for the first download.")
            self.model = api.load(self.model_name)
            print("Model loaded successfully.")

    def generate_filler_space(self, terminals, output_dim):
        """
        Generates a dictionary mapping terminal words to vectors in a lower-dimensional space.

        Args:
            terminals (set): A set of terminal words from the toy language.
            output_dim (int): The desired dimensionality of the output vectors.

        Returns:
            dict: A dictionary mapping words to their new numpy vectors.
        """
        self._load_model()
        
        word_vectors = []
        found_words = []
        
        for word in terminals:
            if word in self.model:
                word_vectors.append(self.model[word])
                found_words.append(word)
            else:
                warnings.warn(f"Warning: '{word}' not found in Word2Vec model. Skipping.")
        
        if not found_words:
            raise ValueError("None of the terminal words were found in the Word2Vec model.")
            
        word_vectors_matrix = np.array(word_vectors)
        
        # Reduce dimensionality using PCA
        print(f"Reducing dimensionality from {self.model.vector_size} to {output_dim} using PCA...")
        pca = PCA(n_components=output_dim)
        reduced_vectors = pca.fit_transform(word_vectors_matrix)
        print("Dimensionality reduction complete.")
        
        filler_space = {word: vec for word, vec in zip(found_words, reduced_vectors)}
        
        return filler_space


