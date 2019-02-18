#!/usr/bin/python

import numpy as np
from EKF_with_HMM import EKF_with_HMM
from scipy.stats import multivariate_normal
from image_projection import ImageProjection
import logging

logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self, ekf_sensor_noise, hmm_observation_model,
                 use_hmm=True):
        
        self.tracks = []
        self.curr_id = 0

        self.HMM_observation_model = hmm_observation_model
        self.EKF_sensor_noise = ekf_sensor_noise

        #standard tracking thresholds
        pos_cov_limit = 4.0 #threshold for track pose covariance
        chi2_thresh = 7.815 #threshold for Mahalanobis distance
        eucl_thresh = 1.0 #threshold for Eucledian distance
        self.set_thresholds(pos_cov_limit, chi2_thresh, eucl_thresh)
        
        #standard tracking config
        accel_noise = 15.0
        height_noise = 0.25
        init_vel_sigma = 1.5
        hmm_transition_prob = 2e-4
        
        self.set_tracking_config(accel_noise, height_noise, init_vel_sigma, 
                                 hmm_transition_prob, use_hmm)
    
    def set_thresholds(self, pos_cov_threshold, mahalanobis_max_dist, 
                       euclidean_max_dist):
        
        self.pos_cov_limit = pos_cov_threshold, 
        self.chi2_thresh = mahalanobis_max_dist,
        self.eucl_thresh = euclidean_max_dist
    
    def set_tracking_config(self, accel_noise, height_noise, 
                            init_vel_sigma, hmm_transition_prob,
                            use_hmm):
        
        self.accel_noise = accel_noise
        self.height_noise = height_noise
        self.init_vel_sigma = init_vel_sigma
        self.HMM_transition_prob = hmm_transition_prob
        self.use_hmm = use_hmm
        
        #reset tracks
        self.tracks = []
    
    def get_track_ids(self):
        ids = []
        for track in self.tracks:
            ids.append(track.track_id)
            
        return ids
    
    def get_track_positions(self):
        track_positions = []
        
        for track in self.tracks:
            track_pos = track.get_odom_position()
            track_positions.append(track_pos)
            
        return track_positions
    
    def get_track_velocities(self):
        track_velocities = []
        
        for track in self.tracks:
            track_vel = track.get_odom_velocity()
            track_velocities.append(track_vel)
        
        return track_velocities
    
    def get_track_detections(self, trafo_odom_in_cam):
        track_detections = []
        
        for track in self.tracks:
            track_det = track.get_track_detection(trafo_odom_in_cam)
            track_detections.append(track_det)
        
        return track_detections
    
    def get_track_covariances(self):
        track_covariances = []
        
        for track in self.tracks:
            track_cov = track.get_covariance()
            track_covariances.append(track_cov)
        
        return track_covariances
    
    def predict(self, dt):
        for track in self.tracks:
            track.predict(dt)
    
    def update(self, detections, trafo_odom_in_cam, camera_calibration):
        #calculate pairwise mahalanobis distance
        assignment_profit = np.zeros([len(self.tracks), len(detections)])
        trafo_cam_in_odom = np.linalg.inv(trafo_odom_in_cam)
        
        #sort detections with increasing confidence so the most confident
        #detection determines the track bbox
        detections = sorted(detections, key=lambda item: item['score'])
        
        for i, detection in enumerate(detections):
            
            for j,track in enumerate(self.tracks):
                z_exp = track.get_z_exp(trafo_odom_in_cam)
                H = track.get_H(trafo_odom_in_cam)

                z =  ImageProjection.get_measurement(detection)
                v = z - z_exp
                
                S = H.dot(track.sigma).dot(np.transpose(H)) + track.R
                mahalanobis_d = np.transpose(v).dot(np.linalg.inv(S)).dot(v)
                
                x = np.squeeze(v)
                mu = np.array([0.0, 0.0, 0.0])
                
                try:
                    pdf = multivariate_normal.pdf(x, mu, S)
                    assignment_profit[j,i] = pdf
                
                except np.linalg.LinAlgError as e:
                    print e
                    assignment_profit[j,i] = -1
                
                cam_det = ImageProjection.get_cart_detection(detection, camera_calibration)
                odom_det = ImageProjection.transform_detection(cam_det, trafo_cam_in_odom)
        
                eucl_distance = np.hypot(odom_det["x"] - track.mu[0], odom_det["y"] - track.mu[1])
                
                if mahalanobis_d >  self.chi2_thresh:
                    assignment_profit[j,i] = -1

                if eucl_distance > self.eucl_thresh:
                    assignment_profit[j,i] = -1
                    
        detection_assignments = -1 * np.ones(len(detections), np.int)

        #pair each detection to the closest track
        for i,odom_det in enumerate(detections):
            max_profit = 0
            for j,track in enumerate(self.tracks):
                if assignment_profit[j,i] > max_profit:
                    detection_assignments[i] = j
                    max_profit = assignment_profit[j,i]
        
        for i,detection in enumerate(detections):
            #if detection was paired, update tracker
            if detection_assignments[i] != -1:
                #detection was paired, update tracker
                track = self.tracks[detection_assignments[i]]
                track.update(detection, trafo_odom_in_cam)
            
            else: 
                #start new tracker
                track = EKF_with_HMM(detection, trafo_odom_in_cam, camera_calibration, 
                                     self.accel_noise, self.height_noise, 
                                     self.init_vel_sigma, self.EKF_sensor_noise, 
                                     self.HMM_observation_model, 
                                     self.HMM_transition_prob,self.curr_id, 
                                     self.use_hmm)
                self.curr_id += 1
                self.tracks.append(track)
                logger.debug("detection not matched, start new KF with id %d" % track.track_id)
        
        for track in self.tracks:
            #apply background detection if not detected
            if not track.updated:
                track.update_with_background()
                
            #find position uncertainty at sigma interval 
            pos_uncertainty = 0
            pos_cov = track.sigma[0:2,0:2]
            
            try:
                eigenvals, eigenvecs = np.linalg.eig(pos_cov)
            except np.linalg.linalg.LinAlgError as e:
                print e
        
            #get largest eigenvalue
            max_eigval = np.max(eigenvals)
            
            if max_eigval < 0.00001 or max_eigval == np.nan:
                pos_uncertainty = np.inf
            
            #chi-square value for sigma confidence interval
            chisquare_scale = 2.2789  
        
            #calculate width and height of confidence ellipse
            pos_uncertainty = 2 * np.sqrt(chisquare_scale*max_eigval)
            
            #check if we need to delete the track
            if pos_uncertainty > self.pos_cov_limit or track.get_class() == 0:
                logger.debug("deleting track %d" % track.track_id)
                self.tracks.remove(track)
