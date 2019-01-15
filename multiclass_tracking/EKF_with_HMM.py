import numpy as np
from HMM import HMM
from image_projection import ImageProjection

class EKF_with_HMM:

    def __init__(self, measurement, trafo_odom_in_cam, cam_calib, 
                 ekf_sensor_noise, hmm_observation_model, track_id, use_hmm=True):

        self.track_id = track_id
        self.bbox = measurement["bbox"]
        self.last_det = measurement
        
        self.accel_noise = 15.0 #noise in acceleration, in meters/sec^2
        self.height_noise = 0.25 #noise in height of person, in m
        
        trafo_cam_in_odom = np.linalg.inv(trafo_odom_in_cam)
        
        cam_det = ImageProjection.get_cart_detection(measurement, cam_calib)
        odom_det = ImageProjection.transform_detection(cam_det, trafo_cam_in_odom)
        
        #state: x_odom, y_odom, z_odom, vel_x_odom, vel_y_odom
        self.mu = np.array([[odom_det["x"]], [odom_det["y"]], [odom_det["z"]], [0.0], [0.0]])
        
        #measurement_noise
        self.R = ekf_sensor_noise
        
        #initial state uncertainty
        self.sigma = np.array([[0.0, 0.0, 0.0,    0.0,    0.0],
                               [0.0, 0.0, 0.0,    0.0,    0.0],
                               [0.0, 0.0, 0.0,    0.0,    0.0],
                               [0.0, 0.0, 0.0, 1.5**2,    0.0],
                               [0.0, 0.0, 0.0,    0.0, 1.5**2]])
        
        #initialize state uncertainty of pose from measurement uncertainty
        H = self.get_H(trafo_odom_in_cam, cam_calib)
        
        H_inv = np.linalg.inv(H[0:3,0:3])
        self.sigma[0:3,0:3] = H_inv.dot(self.R).dot(H_inv.T)
        
        self.use_hmm = use_hmm
        
        if self.use_hmm:
            #initialize HMM
            self.hmm = HMM(hmm_observation_model)
            self.hmm.update(measurement['category_id'])
        
        self.updated = True
        
    def get_H(self, trafo_odom_in_cam, cam_calib):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        
        #get the partial derivatives
        T = trafo_odom_in_cam
        T11 = T[0,0]
        T12 = T[0,1]
        T13 = T[0,2]
        T14 = T[0,3]
        T21 = T[1,0]
        T22 = T[1,1]
        T23 = T[1,2]
        T24 = T[1,3]
        T31 = T[2,0]
        T32 = T[2,1]
        T33 = T[2,2]
        T34 = T[2,3]
        
        xw = self.mu[0,0]
        yw = self.mu[1,0]
        zw = self.mu[2,0]
        
        d_imx_dx = (T11*fx)/(T34 + T31*xw + T32*yw + T33*zw) - (T31*fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imx_dy = (T12*fx)/(T34 + T31*xw + T32*yw + T33*zw) - (T32*fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imx_dz = (T13*fx)/(T34 + T31*xw + T32*yw + T33*zw) - (T33*fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        
        d_imy_dx = (T21*fy)/(T34 + T31*xw + T32*yw + T33*zw) - (T31*fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imy_dy = (T22*fy)/(T34 + T31*xw + T32*yw + T33*zw) - (T32*fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imy_dz = (T23*fy)/(T34 + T31*xw + T32*yw + T33*zw) - (T33*fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        
        d_depth_dx = T31
        d_depth_dy = T32
        d_depth_dz = T33
        
        #Jacobian of h(x)
        H = np.array([[  d_imx_dx,   d_imx_dy,   d_imx_dz, 0.0, 0.0],
                      [  d_imy_dx,   d_imy_dy,   d_imy_dz, 0.0, 0.0],
                      [d_depth_dx, d_depth_dy, d_depth_dz, 0.0, 0.0]])
        
        return H
    
    def get_z_exp(self, trafo_odom_in_cam, cam_calib):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        cx = cam_calib['cx']
        cy = cam_calib['cy']
        
        #apply sensor model
        T = trafo_odom_in_cam
        T11 = T[0,0]
        T12 = T[0,1]
        T13 = T[0,2]
        T14 = T[0,3]
        T21 = T[1,0]
        T22 = T[1,1]
        T23 = T[1,2]
        T24 = T[1,3]
        T31 = T[2,0]
        T32 = T[2,1]
        T33 = T[2,2]
        T34 = T[2,3]
        
        xw = self.mu[0,0]
        yw = self.mu[1,0]
        zw = self.mu[2,0]
        
        im_x = cx + (fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)
        im_y = cy + (fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)
        depth = T34 + T31*xw + T32*yw + T33*zw
        
        return np.array([[im_x], [im_y], [depth]])
    
    def predict(self, dt):

        #motion model
        A = np.matrix( [ [1, 0, 0, dt, 0], 
                         [0, 1, 0, 0, dt], 
                         [0, 0, 1, 0,  0],
                         [0, 0, 0, 1,  0], 
                         [0, 0, 0, 0,  1] ])
        
        #noise gain matrix
        gamma = np.array([[dt**2/2,       0, 0],
                          [      0, dt**2/2, 0],
                          [      0,       0, 1],
                          [     dt,       0, 0],
                          [      0,      dt, 0]])
        
        #individual noise params
        w = np.array([[self.accel_noise,                0,                 0],
                      [               0, self.accel_noise,                 0],
                      [               0,                0, self.height_noise]])
        
        Q = gamma.dot(w).dot(w.transpose()).dot(gamma.transpose())
        
        #this is linear in our case, can perform prediction like standard KF
        self.mu = A.dot(self.mu)
        self.sigma = A.dot(self.sigma).dot(np.transpose(A)) + Q
        
        self.updated = False
        
        if self.use_hmm:
            self.hmm.predict()
        
    def update(self, measurement, trafo_odom_in_cam, cam_calib):
        
        #measurement for Kalman filter
        z = np.array([[measurement['im_x']], [measurement['im_y']], [measurement['depth']]])
        H = self.get_H(trafo_odom_in_cam, cam_calib)
        
        # Kalman Gain
        K_tmp = H.dot(self.sigma).dot(np.transpose(H)) + self.R
        K = self.sigma.dot(np.transpose(H)).dot(np.linalg.inv(K_tmp))
        
        #update mu
        z_exp = self.get_z_exp(trafo_odom_in_cam, cam_calib) #h(mu)
        self.mu = self.mu + K.dot(z-z_exp)
        
        # update covariance
        self.sigma = ( np.eye(5) - K.dot(H) ).dot(self.sigma)
        
        # new image bbox
        self.bbox = measurement["bbox"]
        self.last_det = measurement

        #update hmm
        if self.use_hmm:
            self.hmm.update(measurement["category_id"])
            
    def get_class(self):
        if self.use_hmm:
            return self.hmm.get_max_class()
        else:
            return self.last_det["category_id"]
        
    def get_score(self):
        if self.use_hmm:
            return self.hmm.get_max_score()
        else:
            return self.last_det["score"]
        
    def update_with_background(self):
        if self.use_hmm:
            self.hmm.update(0)