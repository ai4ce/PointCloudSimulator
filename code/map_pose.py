import numpy as np
import cv2 as cv
import torch


def generate_samples(img_h,img_w,p=0.5,n_block=10):
    maps = Maps(img_h,img_w)
    maps.sampleMap(p,n_block)
    return maps

class Maps(object):
    def __init__(self,img_h,img_w):
        self.img_h = img_h
        self.img_w = img_w
        
    def sampleMap(self,p,n_block):
        """
        sample map environment
        Attribute:
            image: image representation
            line_seg: line segments that represents the edges of objects
        """
        self.image = self.oc_block(p,n_block)
        self.line_seg = torch.from_numpy(self.get_line_seg_rep())
        
    def oc_block(self,p,n_block):
        """
        sample occupied blocks in the image
        Return:
            binary image where black pixels indicate occupied positions
        """
        image = torch.ones(1,self.img_h,self.img_w)
        for i in range(n_block):
            ul_w = np.random.randint(0,self.img_w-1) # upper-left
            ul_h = np.random.randint(0,self.img_h-1)
            block_h = np.random.randint(2,p*self.img_h)
            block_w = np.random.randint(2,p*self.img_w)
            br_w = np.clip(ul_w + block_w,0,self.img_w) # bottom-right
            br_h = np.clip(ul_h + block_h,0,self.img_h)
            image[:,ul_h:br_h,ul_w:br_w] = 0
        return image.type(torch.FloatTensor)
    
    def get_line_seg_rep(self):
        """
        convert binary image representation into line segments that are boundaries of occupied blocks
        """
        img_np = self.image.numpy().squeeze(0)
        _,thresh = cv.threshold(img_np,0.5,1,cv.THRESH_BINARY_INV)
        _,contours,_ = cv.findContours(255*thresh.astype(np.uint8),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        line_segs = []
        for contour in contours:
            contour = contour.squeeze(1)
            contour_cir_shift = np.roll(contour,1,axis=0)
            contour = np.concatenate((contour,contour_cir_shift),axis=1)
            line_segs.append(contour)
            
        # add line segments for image boundary
        line_segs.append([0,0,0,self.img_h-1])
        line_segs.append([0,self.img_h-1,self.img_w-1,self.img_h-1])
        line_segs.append([self.img_w-1,self.img_h-1,self.img_w-1,0])
        line_segs.append([self.img_w-1,0,0,0])
        return np.vstack(line_segs).astype(np.float32)
