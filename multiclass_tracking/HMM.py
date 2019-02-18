#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:04:17 2018

@author: kollmitz
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class HMM:
    def __init__(self, observation_model, transition_prob):
        
        self.num_classes = 6
        
        self.observ_model = observation_model
        
        #assign small transition probabilities from each class to the other
        #classes, except background
        tr = transition_prob
        hh = 1 - (self.num_classes-2)*tr

        self.transition_model = np.array([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
                                          [0.0000, hh, tr, tr, tr, tr], 
                                          [0.0000, tr, hh, tr, tr, tr], 
                                          [0.0000, tr, tr, hh, tr, tr], 
                                          [0.0000, tr, tr, tr, hh, tr], 
                                          [0.0000, tr, tr, tr, tr, hh]])
        
        #uniform initial belief over classes
        self.belief = 1./self.num_classes * np.ones(self.num_classes)
        
    def predict(self):
        self.belief = self.belief.dot(self.transition_model)
        
    def update(self, class_obs):
        
        observation = np.zeros(self.num_classes)
        observation[class_obs] = 1
        
        observ_prob = observation.dot(np.transpose(self.observ_model))
        
        #observation model
        self.belief = np.multiply(observ_prob, self.belief)
        
        #normalize
        self.belief = self.belief / sum(self.belief)
        
    def get_max_class(self):
        return np.argmax(self.belief)
    
    def get_max_score(self):
        return np.max(self.belief)