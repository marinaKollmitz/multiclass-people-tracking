import numpy as np

class ImageProjection:
    
    @staticmethod
    def get_measurement(detection):
        #a measurement is an array of the form [[im_x], [im_y], [depth]]
        
        im_x, im_y = ImageProjection.get_im_coordinates(detection["bbox"])
        
        return np.array([[im_x], [im_y], [detection['depth']]])
    
    @staticmethod
    def get_im_coordinates(bbox_xyxy):
        
        im_x = (bbox_xyxy[0]+bbox_xyxy[2])/2.
        im_y = (bbox_xyxy[1]+bbox_xyxy[3])/2.
        
        return im_x, im_y
    
    #project from image into cartesian camera frame
    @staticmethod
    def get_cart_detection(detection, cam_calib):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        cx = cam_calib['cx']
        cy = cam_calib['cy']
        
        im_x, im_y = ImageProjection.get_im_coordinates(detection["bbox"])
        
        cart_det = {}
        
        #project detection into camera frame
        cart_det["x"] = (im_x - cx)/fx * detection["depth"]
        cart_det["y"] = (im_y - cy)/fy * detection["depth"]
        cart_det["z"] = detection["depth"]
        
        return cart_det
    
    #transform a detection from one frame to the other
    @staticmethod
    def transform_detection(cart_detection, trafo):
    
        #transform into world frame
        Xc = np.array([[cart_detection["x"]],[cart_detection["y"]],[cart_detection["z"]],[1]])
        Xw = trafo.dot(Xc)
        
        trafo_det = {}
        trafo_det["x"] = Xw[0,0]
        trafo_det["y"] = Xw[1,0]
        trafo_det["z"] = Xw[2,0]
        
        return trafo_det
    
    #calculate image bounding box from cartesian detection and bounding box dimensions
    @staticmethod
    def get_image_bbox(cart_detection, cam_calib, bbox_width, bbox_height):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        cx = cam_calib['cx']
        cy = cam_calib['cy']
        
        #project detection into image frame
        im_x = cart_detection["x"] / cart_detection["z"] * fx + cx
        im_y = cart_detection["y"] / cart_detection["z"] * fy + cy
        
        bbox = [im_x-bbox_width/2., 
                im_y-bbox_height/2., 
                im_x+bbox_width/2.,
                im_y+bbox_height/2.]
        
        return bbox
