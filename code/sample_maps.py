import os
import argparse
import pickle

from PIL import Image
import numpy as np
from map_pose import generate_samples

parser = argparse.ArgumentParser()
parser.add_argument('--n_env',type=int,default=1,help='number of sampled maps')
parser.add_argument('--img_h',type=int,default=1024,help='image height')
parser.add_argument('--img_w',type=int,default=1024,help='image width')
parser.add_argument('--p',type=float,default=0.2,help='p')
parser.add_argument('--n_block',type=int,default=20,help='num of blocks')
opt = parser.parse_args()

save_dir = '../results'

for idx in range(opt.n_env):
    maps = generate_samples(opt.img_h,opt.img_w,opt.p,opt.n_block) 

    sub_save_dir = os.path.join(save_dir,'v'+str(idx))
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    
    save_name = os.path.join(sub_save_dir,'environment.png')
    
    # save image
    image_np = 255*maps.image.cpu().numpy()
    image_np = image_np.astype(np.uint8).squeeze(0)
    image = Image.fromarray(image_np)
    
    image.save(save_name)

    # save image and line seg
    save_name = os.path.join(sub_save_dir,'environment.pkl')
    with open(save_name,'wb') as f:
        pickle.dump(maps,f)
