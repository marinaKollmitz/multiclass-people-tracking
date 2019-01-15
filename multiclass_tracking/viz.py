#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:56:24 2019

@author: kollmitz
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle
import matplotlib.image as mpimg
import numpy as np

from image_projection import ImageProjection
import tf

class Visualizer:
    def __init__(self, num_classes):
        
        #TODO more than 6 classes
        self.colors_box = [[     0.0,     0.0,     0.0], #background = black
                           [     0.0, 38/255.,133/255.], #person = blue
                           [200/255.,     0.0,     0.0], #crutches = red
                           [255/255.,218/255., 23/255.], #walking frame = yellow
                           [     0.0,150/255.,     0.0], #wheelchair = green
                           [255/255.,100/255.,     0.0]] #push wheelchair = orange
        
        plt.figure(figsize=(15,10))
        self.ax1 = plt.subplot(121)
        self.ax2 = plt.subplot(222)
        self.ax3 = plt.subplot(224)
        
    #get dimensions of error ellipse corresponding to covariance sigma
    @staticmethod
    def get_error_ellipse(sigma):
        #calculate and plot covariance ellipse
        covariance = sigma[0:2,0:2]
        try:
            eigenvals, eigenvecs = np.linalg.eig(covariance)
        except np.linalg.linalg.LinAlgError as e:
            print 'cov', covariance
            print e
    
        #get largest eigenvalue and eigenvector
        max_ind = np.argmax(eigenvals)
        max_eigvec = eigenvecs[:,max_ind]
        max_eigval = eigenvals[max_ind]
        
        if max_eigval < 0.00001 or max_eigval == np.nan:
            print "max eigenval", max_eigval
            plt.waitforbuttonpress()
    
        #get smallest eigenvalue and eigenvector
        min_ind = 0
        if max_ind == 0:
            min_ind = 1
    
        min_eigval = eigenvals[min_ind]
    
        #chi-square value for sigma confidence interval
        chisquare_scale = 2.2789  
    
        #calculate width and height of confidence ellipse
        width = 2 * np.sqrt(chisquare_scale*max_eigval)
        height = 2 * np.sqrt(chisquare_scale*min_eigval)
        angle = np.arctan2(max_eigvec[1],max_eigvec[0])
        
        return width, height, angle
    
    def visualize_detections(self, image_file, trafo_cam_in_odom, 
                             trafo_cam_in_base, cam_calib,
                             detections, tracks, time_delta=0.06, step=False):
        
        trafo_odom_in_cam = np.linalg.inv(trafo_cam_in_odom)
        trafo_odom_in_base = np.dot(trafo_cam_in_base, trafo_odom_in_cam)
        rotate_robot_to_plot = np.array([[0, -1, 0],
                                         [1,  0, 0],
                                         [0,  0, 1]])
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        self.ax1.set_title("workspace view, robot perspective")
        self.ax2.set_title("detections")
        self.ax3.set_title("tracking")
        
        self.ax1.axis('equal')
        self.ax1.set_xlim(-10, 10)
        self.ax1.set_ylim(-2, 12)
        
        img = mpimg.imread(image_file)
        
        self.ax2.imshow(img)
        self.ax3.imshow(img)
        
        #plot robot pose
        robot_footprint = Circle(xy=[0.0, 0.0], radius=0.5, edgecolor='k', facecolor='none')
        self.ax1.add_artist(robot_footprint)
        #plot robot x axis
        self.ax1.arrow(0.0, 0.0, 0.0, 1.0, width=0.1, color='r')
        #plot robot y axis
        self.ax1.arrow(0.0, 0.0, -1.0, 0.0, width=0.1, color='g')
        
        #visualize detections
        for detection in detections:
            fill_rect = Rectangle((detection['bbox'][0],detection['bbox'][1]),detection['bbox'][2],detection['bbox'][3],linewidth=2,edgecolor='none', facecolor='w', alpha=0.5)
            rect = Rectangle((detection['bbox'][0],detection['bbox'][1]),detection['bbox'][2],detection['bbox'][3],linewidth=2,edgecolor=self.colors_box[detection['category_id']],facecolor='none')
            self.ax2.add_artist(fill_rect)
            self.ax2.add_artist(rect)
            
            cam_det = ImageProjection.get_cart_detection(detection, cam_calib)
            robot_det = ImageProjection.transform_detection(cam_det, trafo_cam_in_base)
            
            plot_det = np.dot(rotate_robot_to_plot, np.array([robot_det['x'], robot_det['y'], robot_det['z']]))
            
            #transform odom det into robot coordinate system
            self.ax1.plot(plot_det[0], plot_det[1], 'rx')
        
        #visualize tracks
        for track in tracks:
            
            odom_track = {}
            odom_track['x'] = track.mu[0]
            odom_track['y'] = track.mu[1]
            odom_track['z'] = track.mu[2]
            
            track_in_base = ImageProjection.transform_detection(odom_track, trafo_odom_in_base)
            base_track_array = np.squeeze(np.array([track_in_base['x'], track_in_base['y'], track_in_base['z']]))
            plot_track = np.dot(rotate_robot_to_plot, base_track_array)
            
            self.ax1.plot(plot_track[0], plot_track[1], 'x')
            
            width, height, angle = self.get_error_ellipse(track.sigma)
            r,p,y = tf.transformations.euler_from_matrix(np.linalg.inv(trafo_odom_in_base))
            
            #generate covariance ellipse
            ell = Ellipse(xy=[plot_track[0],plot_track[1]], width=width, height=height, angle=(angle-y-np.pi/2)/np.pi*180)
            ell.set_alpha(0.25)
            
            #pos_cov_plot = np.dot(rotate_robot_to_plot, pos_cov_robot)
            self.ax1.add_artist(ell)
            
            cla = track.get_class()
            
            odom_det = {}
            odom_det["x"] = track.mu[0]
            odom_det["y"] = track.mu[1]
            odom_det["z"] = track.mu[2]
            
            bbox_width = track.bbox[2]
            bbox_height = track.bbox[3]
            
            cam_det = ImageProjection.transform_detection(odom_det, trafo_odom_in_cam)
            im_bbox = ImageProjection.get_image_bbox(cam_det, cam_calib, bbox_width, bbox_height)
            
            fill_rect = Rectangle((im_bbox[0],im_bbox[1]),im_bbox[2],im_bbox[3],linewidth=2, edgecolor='none', facecolor='w', alpha=0.5)
            rect = Rectangle((im_bbox[0],im_bbox[1]),im_bbox[2],im_bbox[3],linewidth=2,edgecolor=self.colors_box[cla],facecolor='none')
            self.ax3.add_artist(fill_rect)
            self.ax3.add_artist(rect)
        
        if step:
            plt.waitforbuttonpress()
        else:
            plt.pause(time_delta)