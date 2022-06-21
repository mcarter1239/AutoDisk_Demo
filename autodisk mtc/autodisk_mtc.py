import copy
import csv
import multiprocessing as mp
import os
import time
from datetime import datetime
from tokenize import group

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats
from skimage import feature
from skimage.feature import blob_log
from skimage.transform import resize
from sqlalchemy import distinct


def readData(dname):   
    """
    Read in a 4D-STEM data file.
    
    Parameters
    ----------
    dname : str
        Name of the data file.

    Returns
    -------
    data: 4D array of int or float
        The read-in 4D-STEM data.
    
    """
    dimy = 130
    dimx = 128

    file = open(dname,'rb') 
    data = np.fromfile(file, np.float32)          
    pro_dim = int(np.sqrt(len(data)/dimx/dimy))
    
    data = np.reshape(data, (pro_dim, pro_dim, dimy, dimx))
    data = data[:,:,1:dimx+1, :]
    file.close()
    
    return data

def preProcess(data):   
    """
    Pre-process 4D-STEM data.
    
    Parameters
    ----------
    data: 4D array of int or float

    Returns
    -------
    data: 4D array of int or float
    
    """
    data[np.where(np.isnan(data)==True)] = 0

    data -= data.min() if data.min()< 0 else 0
    data += 10**(-17)
    
    return data

def visual(image,plot = True):
    """
    Convert a 2D array of int or float to an int8 array of image and visualize it.

    Parameters
    ----------
    image : 2D array of int or float
    plot : bool, optional
        True if the image need to be ploted. The default is True.

    """
    image_out = (((image - image.min()) / (image.max() - image.min())) * 255).astype(np.uint8)
    if plot==True:
        plt.imshow(image_out,cmap='gray')
        plt.show()

    pass

def drawDisks(pattern_in,disks,r):
    """
    Label the disk positions on the pattern.

    Parameters
    ----------
    pattern_in : 2D array of int or float
        The pattern to be labeled on.
    disks : 2D array of float
        Array of disk positions.
    r: float
        The radius of the disks.

    Returns
    -------
    None.

    """
    pattern = copy.deepcopy(pattern_in)
    
    for q in range (len(disks)):
        center = (int(disks[q,0]),int(disks[q,1]))
        pattern[center] = pattern.max()
    
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(pattern,cmap='gray')
    for blob in disks:
        y, x = blob
        c = plt.Circle((x, y),r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
    
    plt.show()
    
    pass

def generateAvgPattern(data):
    """
    Generate an average (sum) pattern from the 4D dataset.

    Parameters
    ----------
    data : 4D array of int or float
        Array of the 4D dataset.

    Returns
    -------
    avg_pat: 2D array of int or float
        An average (sum) difffraction pattern.

    """
    pro_y,pro_x = data.shape[:2]
    avg_pat = data[0,0]*1
    avg_pat[:,:] = 0
    for row in range (pro_y):
        for col in range (pro_x):
            avg_pat += data[row,col]
    
    return avg_pat

def detAng(ref_ctr,ctr,r): # threshold: accepted angle difference
    """
    Detect an angle to rotate the disk coordinates.

    Parameters
    ----------
    ref_ctr : 2D array of float
        Array of disk position coordinates and their corresponding weights
    ctr : 1D array of float
        Center of the zero-order disk.
    r : float
        Radius of the disks.

    Returns
    -------
    wt_ang : float
        The rotation angle.
    ref_ctr : 2D array of float
        Refined disk positions.

    """
    ctr_vec = ref_ctr[:,:2] - ctr
    ctr_diff = ctr_vec[:,0]**2 + ctr_vec[:,1]**2
    ctr_idx = np.where(ctr_diff==ctr_diff.min())[0][0]
    
    diff = ref_ctr[:,:2]-ctr
    distance = diff[:,0]**2 + diff[:,1]**2
    
    dis_copy = copy.deepcopy(distance)
    min_dis = []
    while len(min_dis) <5:
        cur_min = dis_copy.min()
        idx_rem = np.where(dis_copy==cur_min)[0]
        dis_copy = np.delete(dis_copy,idx_rem)
        idx_ctr = np.where(distance==cur_min)[0]
        if len(idx_ctr)==1:
            min_dis.append(ref_ctr[idx_ctr[0],:2])
        else:
            for each in idx_ctr:   
                min_dis.append(ref_ctr[each,:2])

    min_dis_ctr = np.array(min_dis,dtype = int)
    min_dis_ctr = np.delete(min_dis_ctr,0,axis = 0) # delete [0,0]

    vec = min_dis_ctr-ctr
       
    ang = np.arctan2(vec[:,0],vec[:,1])* 180 / np.pi
    
    for i in range (len(ang)):
        ang[i] = (180 + ang[i]) if (ang[i]<0) else ang[i]

    cand_ang_idx = np.where(ang==ang.min())[0]
    sup_pt = min_dis_ctr[cand_ang_idx] # the point retuning the smallest rotation angle


    ref_diff = ctr-sup_pt
    ini_ang = np.arctan2(ref_diff[:,0],ref_diff[:,1])*180/np.pi
    all_ref = []
    for n in range (len(ini_ang)):
        all_ref.append(np.array([ini_ang[n]]))
    if len(ref_diff)>1:
        ref_diff = ref_diff[0]

    for each_ctr in ref_ctr:
        cur_vec = each_ctr[:2] - ref_diff
        cur_diff = ref_ctr[:,:2]-cur_vec
        cur_norm = np.linalg.norm(cur_diff,axis=1)
        if cur_norm.min()<r:
            ref_idx = np.where(cur_norm==cur_norm.min())[0]
            ref_pt = ref_ctr[ref_idx]
            ref_vec = ref_pt - each_ctr
            all_ref.append(np.arctan2(ref_vec[:,0],ref_vec[:,1])* 180 / np.pi)
    
    for i in range (len(all_ref)):
        if all_ref[i]<0:
            all_ref[i] = 180 + all_ref[i]
        elif all_ref[i] >= 180:
            all_ref[i] = 180 - all_ref[i]
        
    wt_ang = np.mean(all_ref)
    ref_ctr[ctr_idx,2] = 10**38
    
    return wt_ang, ref_ctr

def rotImg(image, angle, ctr):
    """
    Rotate a pattern.

    Parameters
    ----------
    image : 2D array of int or float
        The input pattern.
    angle : float
        An angel to rotate.
    ctr : 1D array of int or float
        The rotation center.

    Returns
    -------
    result : 2D array of int or float
        The rotated pattern.

    """
    image_center = tuple(np.array([ctr[0],ctr[1]]))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

def delArti(gen_lat_pt,ref_ctr,r):
    """
    Delete any artificial lattice points.

    Parameters
    ----------
    gen_lat_pt : 2D array of float
        Array of artificial disk positions.
    ref_ctr : 2D array of float
        Array of detected disk positions.
    r : float
        Radius of the disks.

    Returns
    -------
    gen_lat_pt_up : 2D array of float
        A filtered array of disk positions.

    """
    gen_lat_pt_up = []
    for i in range (len(gen_lat_pt)):
        dif_gen_ref = np.array(gen_lat_pt[i] - ref_ctr[:,:2])
        dif_gen_ref_norm = np.linalg.norm(dif_gen_ref,axis = 1)
        if dif_gen_ref_norm.min()< r:
            gen_lat_pt_up.append(gen_lat_pt[i])
    
    gen_lat_pt_up = np.array(gen_lat_pt_up)
    
    return gen_lat_pt_up

def genLat(pattern, ret_a,ret_b, mid_ctr,r):
    """
    Generate a matrix of hypothetical lattice points.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    ret_a : 1D array of float
        The horizontal lattice vector a.
    ret_b : 1D array of float
        The non-horizontal lattice vector b.
    mid_ctr : a list of arrays of float
        a list of disk positions which are in the middle row.
    r : float
        Radius of the disks.

    Returns
    -------
    final_ctr : 2D array of float
        Disk positions in the hypothetical lattice.

    """
    img = pattern
    veca,vecb = ret_a,ret_b
    h,w = img.shape
    veca_ct = mid_ctr[:,:2].copy()
    final_ctr = []
    
    for cur_veca_ct in veca_ct:
        # one side    
        cur_h1 = cur_veca_ct[0]
        cur_w1 = cur_veca_ct[1]
        cur_ct1 = cur_veca_ct*1

        while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
            cur_h1,cur_w1 = cur_ct1-vecb
            if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                cur_ct1 = [cur_h1,cur_w1]
                final_ctr.append([cur_h1,cur_w1])
        
        # the other side
        cur_h2 = cur_veca_ct[0]
        cur_w2 = cur_veca_ct[1]
        cur_ct2 = cur_veca_ct*1

        while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
            cur_h2,cur_w2 = cur_ct2+vecb
            if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                cur_ct2 = [cur_h2,cur_w2]
                final_ctr.append([cur_h2,cur_w2])  

    ########   Check Again ########
    chk_lat_ctr= final_ctr
    
    for cur_vec2_ct in chk_lat_ctr:
        # one side    
        cur_h1 = cur_vec2_ct[0]
        cur_w1 = cur_vec2_ct[1]
        cur_ct1 = cur_vec2_ct*1
        while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                cur_h1,cur_w1 = cur_ct1-veca
                # print(cur_ct1-veca,cur_h1,cur_w1)
                if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                    cur_ct1 = [cur_h1,cur_w1]
                    dif_chk = [(ct[0]-cur_ct1[0])**2+(ct[1]-cur_ct1[1])**2 for ct in chk_lat_ctr]
                    if min(dif_chk)> r**2: 
                        final_ctr.append([cur_h1,cur_w1])
        
        # the other side
        cur_h2 = cur_vec2_ct[0]
        cur_w2 = cur_vec2_ct[1]
        cur_ct2 = cur_vec2_ct*1
        while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
            cur_h2,cur_w2 = cur_ct2+veca
            if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                cur_ct2 = [cur_h2,cur_w2]   
                dif_chk2 = [(ct[0]-cur_ct2[0])**2+(ct[1]-cur_ct2[1])**2 for ct in chk_lat_ctr]
                if min(dif_chk2)> r**2:  
                    final_ctr.append([cur_h2,cur_w2])   
                                 
    for pt in mid_ctr:
        final_ctr.append(pt)
    
    final_ctr = np.array(final_ctr)
                    
    return final_ctr

def latDist(lattice_params,refe_a,refe_b,err=0.2):
    """
    This function filters out the outliers of the lattice parameters based on the references.

    Parameters
    ----------
    lattice_params : 2D array of arrays of float
        2D array with each element as two arrays of lattice vectors.
    refe_a : 1D array of float
        The reference lattice vector a.
    refe_b : 1D array of float
        The reference lattice vector b.
    err : float, optional
        Acceptable error percentage. The default is 0.2 (20%).

    Returns
    -------
    store_whole : 3D array of float
        Array containing 3 columns, y coordinate, x coordinate, and 4 lattice vector elements
        (y of vector a, x of vector a, y of vector b, x of vector b).

    """
    arr_vec = lattice_params
    
    sm_y,sm_x = lattice_params.shape[:2]
    std_ax = refe_a[1] # vec_a[0,std_2x]
    std_ay = refe_a[0]
    std_bx = refe_b[1] # vec_b[std_1y,std_1x]
    std_by = refe_b[0]
    
    acc_ax_min = std_ax*(1-err) if std_ax>0 else std_ax*(1+err)
    acc_ax_max = std_ax*(1+err) if std_ax>0 else std_ax*(1-err)
    acc_ay_min = std_ay*(1-err) if std_ay>0 else std_ay*(1+err)
    acc_ay_max = std_ay*(1+err) if std_ay>0 else std_ay*(1-err)
    acc_bx_min = std_bx*(1-err) if std_bx>0 else std_bx*(1+err)
    acc_bx_max = std_bx*(1+err) if std_bx>0 else std_bx*(1-err)
    acc_by_min = std_by*(1-err) if std_by>0 else std_by*(1+err)
    acc_by_max = std_by*(1+err) if std_by>0 else std_by*(1-err)
    
    store_whole = np.zeros((sm_y,sm_x,4),dtype = float)

    # Delete paramater outliers
    ct = 0
    for row in range (sm_y):
        for col in range (sm_x):
            
            each = arr_vec[row,col]
        
            gax = float(each[0,1])
            gay = float(each[0,0])
            gbx = float(each[1,1])  
            gby = float(each[1,0])
            
            if gax>acc_ax_max or gax<acc_ax_min or gay>acc_ay_max or gay<acc_ay_min or gbx>acc_bx_max or gbx<acc_bx_min or gby>acc_by_max or gby<acc_by_min:
                ct += 1
                
    
            else:
                store_whole[row,col][0] = gay        
                store_whole[row,col][1] = gax
                store_whole[row,col][2] = gby
                store_whole[row,col][3] = gbx                

    return store_whole       

def calcStrain(lat_fil, refe_a,refe_b):
    """
    Compute strain maps.
    
    Parameters
    ----------
    lat_fil : 2D array of arrays of float
        2D array with each element as two lattice vectors.
    refe_a : 1D array of float 
        The reference vector a.
    refe_b : 1D array of float
        The reference vector b.

    Returns
    -------
    st_xx : 2D array of float
        Estimated strain along the x direction.
    st_yy : 2D array of float
        Estimated strain along the y direction.
    st_xy : 2D array of float
        Shear strain.
    st_yx : 2D array of float
        Shear strain.
    tha_ang : 2D array of float
        Angle of lattice rotation in deg.

    """
    sm_y,sm_x = lat_fil.shape[:2]
    
    st_xx = np.zeros((sm_y,sm_x),dtype=float)
    st_yx = np.zeros((sm_y,sm_x),dtype=float)
    st_xy = np.zeros((sm_y,sm_x),dtype=float)
    st_yy = np.zeros((sm_y,sm_x),dtype=float)
    tha_ang = np.zeros((sm_y,sm_x),dtype=float)
    
    G0_T = np.array([[refe_a[1],refe_a[0]],[refe_b[1],refe_b[0]]])
    
    for row in range (sm_y):
        for col in range (sm_x):
            if any(lat_fil[row,col]!=0):
                gay,gax,gby,gbx = lat_fil[row,col]
    
                G = np.array([[gax,gbx],[gay,gby]])
                G_T = np.transpose(G)
                G_T_n1 = np.linalg.inv(G_T)
                
                D = G_T_n1.dot(G0_T)
                theta = np.arctan2((D[1,0]-D[0,1]),(D[0,0]+D[1,1]))
                
                M = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
                
                F = M.dot(D)
                I = np.array([[1,0],[0,1]])
                
                eps = F-I
                
                st_xx[row,col] = eps[0,0]
                st_yy[row,col] = eps[1,1] 
                st_xy[row,col] = eps[0,1]
                st_yx[row,col] = eps[1,0] 
                tha_ang[row,col] = theta/np.pi*180

    return st_xx,st_yy,st_xy,st_yx,tha_ang

def generateKernel(pattern,center_disk,r,c=0.7,pad=2,pre_def = False, maxed = True, plot = False):
    """
    Generate the kernel for cross-correlation based on thee center disk.

    Parameters
    ----------
    pattern : 2D array of int or float
        An array of a diffraction pattern.
    center_disk : 1D array of float
        Array of the row and column coordinates of the center.
    r : float
        Radius of a disk.
    c : float, optional
        An coefficient to modify the kernel size. The default is 0.7.
    pad : int, optional
        A hyperparameter to change the padding size out of the feature. The default is 2.
    pre_def : bool, optional
        If True, read the pre-defined ring kernel. The default is False.
    maxed : bool, optional
        If True, all non-zero pixels in the kernal are normalized to 1. The default is True.
    plot : bool, optional
        If True, visualize the kernel after generation. The default is False.

    Returns
    -------
    kernel : 2D cd array of float
        Array of the kernel.

    """

    if pre_def == True:
        ring = np.load("kernel_cir.npy")
        f_size = int(2*r*c)
        ring = resize(ring, (f_size, f_size))
        ring = np.array(ring)
        kernel = np.zeros((len(ring)+2*pad,len(ring)+2*pad),dtype=float)
        kernel[pad:-pad,pad:-pad] = ring

        if maxed == True:
            kernel[kernel < kernel.mean()] = 0
            kernel[kernel !=0] = 1

        if plot == True:
            kernel_out = visual(kernel)

        return kernel
    
    
    y_st = int(center_disk[0]-r+0.5-pad*2)
    y_end = int(center_disk[0]+r+0.5+pad*2)
    x_st = int(center_disk[1]-r+0.5-pad*2)
    x_end = int(center_disk[1]+r+0.5+pad*2)
    # +0.5 to avoid rounding errors (always shift to right, so 0,5 is modified to 1.5)
    
    if y_end-y_st==x_end-x_st:
        ctr_disk = pattern[y_st:y_end,x_st:x_end] 
    elif y_end-y_st>x_end-x_st:
        ctr_disk = pattern[y_st+1:y_end,x_st:x_end] 
    else:
        ctr_disk = pattern[y_st:y_end,x_st+1:x_end] 
        
    edge_det = feature.canny(ctr_disk, sigma=1)
    
    dim = len(ctr_disk)
    dim_hf = dim/2
    kernel = np.zeros((dim,dim))
    for i in range (dim):
        for j in range (dim):
            if edge_det[i,j]==True:
                if (i-dim_hf)**2+(j-dim_hf)**2>int(r-2)**2 and (i-dim_hf)**2+(j-dim_hf)**2<int(r+2)**2:
                    kernel[i,j] = 1
    
    coef = int(c*r)
    f_size = 2*coef
    kernel = resize(kernel, (f_size, f_size))
    kernel = np.array(kernel)
    
    if maxed == True:
            kernel[kernel < kernel.mean()] = 0
            kernel[kernel !=0] = 1

    if plot == True:
            kernel_out = visual(kernel)

    return kernel

def crossCorr(pattern,kernel):
    """
    Cross correlate the pattern with the kernal.

    Parameters
    ----------
    pattern : 2D array of int or float
        Array of a diffraction pattern to be cross correlated.
    kernel : 2D array of float
        Array of the kernel.

    Returns
    -------
    cro_img_out : 2D array
        Cross correlated result of the input pattern.

    """
    cro_cor_img = signal.correlate2d(pattern, kernel, boundary='symm', mode='same')
    cro_img_out = np.sqrt(cro_cor_img)
    

    return cro_img_out

def ctrDet(pattern, r, kernel, n_sigma=10, thred=0.1, ovl=0):
    """
    Detect disks on a pattern.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    r : float
        Radius of a disk.
    kernel : 2D array of float
        Kernel used for cross correlation.
    n_sigma : int, optional
        The number of intermediate values of standard deviations. The default is 10.
    thred : float, optional
        The absolute lower bound for scale space maxima. The default is 0.1.
    ovl : float, optional
        Acceptable overlapping area of the blobs. The default is 0.

    Returns
    -------
    blobs : 2D array of int
         Corrdinates of the detected disk position.

    """
    adjr = r * 0.5
    f_size = len(kernel)
    img = np.empty((pattern.shape[0]+2*f_size,pattern.shape[1]+2*f_size))
    bcgd = np.mean(pattern[:f_size,f_size:])
    img[0:f_size,:] = img[-f_size:pattern.shape[0]+2*f_size,:] = img[:,0:f_size] = img[:,f_size:pattern.shape[1]+2*f_size] = bcgd
    img[f_size:pattern.shape[0]+f_size,f_size:pattern.shape[1]+f_size] = pattern 
    sh,sw = img.shape

    blobs_log = blob_log(img, 
                 min_sigma=adjr,
                 max_sigma=adjr, 
                 num_sigma=n_sigma, 
                 threshold= thred,
                 overlap = ovl)    
    
    rem = []
    f_size = len(kernel)
    for i in range (len(blobs_log)):
        if np.any(blobs_log[i,:2]<f_size+5) or np.any(blobs_log[i,0]>sh-f_size-5) or np.any(blobs_log[i,1]>sw-f_size-5):
            rem.append(i)
    
    blobs_log_out = np.delete(blobs_log, rem, axis =0)
    blobs_log_out -= f_size 
    
    blobs =  blobs_log_out[:,:2].astype(int)

    
    return blobs

def ctrRadiusIni(pattern):
    """
    Find the center coordinate and the radius of the zero-order disk.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.

    Returns
    -------
    ctr : 1D array of int or float
        Array of the center coordinates [row,col].
    r : float
        Radius of the center disk in unit of pixels.

    """
    h,w = pattern.shape
    ctr = h//2
    pix_w = pattern[ctr,:]
    pix_h = pattern[:,ctr]
    
    fir_der_w = np.abs(pix_w[:1]-pix_w[1:])
    sec_dir_w_r = np.array(fir_der_w[w//2:-1]-fir_der_w[w//2+1:])
    sec_dir_w_l = np.array(fir_der_w[1:w//2]-fir_der_w[:w//2-1])
    avg_pos1_w = np.where(sec_dir_w_r==sec_dir_w_r.max())[0][0]
    avg_pos2_w = np.where(sec_dir_w_l==sec_dir_w_l.max())[0][0]
    r_w = np.mean([avg_pos1_w+1,len(sec_dir_w_l)-avg_pos2_w])
    ctr_w =  np.mean([w//2 + avg_pos1_w + 1,avg_pos2_w + 2])
    
    fir_der_h = np.abs(pix_h[:1]-pix_h[1:])
    sec_dir_h_b = np.array(fir_der_h[h//2:-1]-fir_der_h[h//2+1:])
    sec_dir_h_u = np.array(fir_der_h[1:h//2]-fir_der_h[:h//2-1])
    avg_pos1_h = np.where(sec_dir_h_b==sec_dir_h_b.max())[0][0]
    avg_pos2_h = np.where(sec_dir_h_u==sec_dir_h_u.max())[0][0]
    r_h = np.mean([avg_pos1_h+1,len(sec_dir_h_u)-avg_pos2_h])
    ctr_h =  np.mean([h//2 + avg_pos1_h + 1, avg_pos2_h+2])
    
    r = np.mean([r_w,r_h])
    ctr = np.array([ctr_h,ctr_w])
    
    return ctr,r            

def rotCtr(pattern,ref_ctr,angle):
    """
    Rotate disk coordinates.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    ref_ctr : 2D array of float
        Array of the detected disk positions.
    angle : float
        Detected angle to rotate.

    Returns
    -------
    ctr_new : 2D array of float
        The transformed disk positions.

    """
    h,w = pattern.shape
    ctr_idx = np.where(ref_ctr[:,2]==ref_ctr[:,2].max())[0][0]
    ctr = ref_ctr[ctr_idx]
    ctr_new = []
    ang_rad = angle*np.pi/180

    for i in range (len(ref_ctr)):
        cur_cd = ref_ctr[i,:2]
        y_new = -(ctr[0] - (cur_cd[0]-ctr[0])*np.cos(ang_rad) + (cur_cd[1]-ctr[1])*np.sin(ang_rad) ) + 2*ctr[0]
        x_new = (ctr[1] + (cur_cd[0]-ctr[0])*np.sin(ang_rad) + (cur_cd[1]-ctr[1])*np.cos(ang_rad) )
        
        if y_new>0 and x_new>0 and y_new<h and x_new<w:
            ctr_new.append([y_new,x_new,ref_ctr[i,2]])
    
    ctr_new = np.array(ctr_new)    

    return ctr_new

def radGradMax(sample, blobs, r, rn=20, ra=2, n_p=40, threshold=3): 
    """
    Radial gradient Maximum process.

    Parameters
    ----------
    sample : 2D array of float or int
        The diffraction pattern.
    blobs : 2D array of int or float
        Blob coordinates.
    r : float
        Radius of the disk
    rn : int, optional
        The total number of rings. The default is 20.
    ra : int, optional
        Half of the window size. The default is 2.
    n_p : int, optional
        The number of sampling points on a ring. The default is 40.
    threshold : float, optional
        A threshold to filter out outliers. The smaller the threshold is, the more outliers are detected. The default is 3.

    Returns
    -------
    ref_ctr : 2D array of float
        Array with three columns, y component, x component and the weight of each detected disk.

    """
    ori_ctr = blobs    
    h,w = sample.shape        
    adjr = r * 1   
    r_scale = np.linspace(adjr*0.8, adjr*1.2, rn)    
    theta = np.linspace(0, 2*np.pi, n_p)     
    ref_ctr = []

    for lp in range (len(ori_ctr)):
        test_ctr = ori_ctr[lp]
        ind_list = []
        for ca in range (-ra,ra):
            for cb in range (-ra,ra):
                cur_row, cur_col = test_ctr[0]+ca, test_ctr[1]+cb
                cacb_rn = np.empty(rn)
                for i in range (rn):
                    row_coor = np.array([cur_row + r_scale[i] * np.sin(theta) + 0.5]).astype(int)
                    col_coor = np.array([cur_col + r_scale[i] * np.cos(theta) + 0.5]).astype(int)
                    
                    row_coor[row_coor>=h]=h-1
                    row_coor[row_coor<0]=0
                    col_coor[col_coor>=w]=w-1
                    col_coor[col_coor<0]=0
                    
                    int_sum = np.sum(sample[row_coor,col_coor])
                    cacb_rn[i] = int_sum
                    
                cacb_rn[:rn//2] *= np.linspace(1,rn//2,rn//2) 
                cacb_diff = np.sum(cacb_rn[:rn//2]) - np.sum(cacb_rn[rn//2:])
                ind_list.append([cur_row, cur_col,cacb_diff])
                
        
        ind_list = np.array(ind_list) 
        ind_max = np.where(ind_list[:,2]==ind_list[:,2].max())[0][0]
        ref_ctr.append(ind_list[ind_max]) 

    ref_ctr = np.array(ref_ctr)

    # Check Outliers
    z = np.abs(stats.zscore(ref_ctr[:,2]))
    outlier = np.where(z>threshold)
    if len(outlier[0])>0:
        for each in outlier[0]:
            if np.linalg.norm(ref_ctr[each,:2]-[h//2,w//2])> r:
                ref_ctr = np.delete(ref_ctr,outlier[0],axis = 0)

    return ref_ctr

def groupY (load_ctr,r):
    """
    Group disks based on their row coordinates.

    Parameters
    ----------
    load_ctr : 2D array of float
        Array of disk positions.
    r : float
        Radius of the disks.

    Returns
    -------
    g_y : a list of arrays of float
        A list with each element as a group of disk positions.

    """
    n = len(load_ctr)
    
    g_y = [[load_ctr[0,:]]]
    for i in range (1,n):        
        gy_mean = []
        for group in g_y:
            cur_mean = 0
            grp_len = len(group)
            for each in group:
                cur_mean += each[0]
            apd_mean = cur_mean/grp_len
            gy_mean.append(apd_mean)
        
        diffy = [np.abs(s-load_ctr[i,0]) for s in gy_mean]
        gy_ind = np.argmin(diffy) 
        min_diffy = np.min(diffy)
        if min_diffy>r:
            g_y.append([load_ctr[i]])
        else:
            g_y[gy_ind].append(load_ctr[i])

    return g_y

def latFit(pattern,rot_ref_ctr,r):  
    """
    Lattice fitting process.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    rot_ref_ctr : 2D array of float
        Array of the disks positionss.
    r : float
        Radius of the disks.

    Returns
    -------
    vec_a : 1D array of float
        The estimated horizontal lattice vector [y component, x component].
    vec_b_ref : 1D array of float
        The estimated non-horizontal lattice vector [y component, x component].
    result_ctr : 2D array of float
        Array of the refined disk positions.
    lat_ctr_arr : 2D array of float
        The array of the positions of disks in the middle row.
    avg_ref_ang : float
        Refined rotation angle.

    """ 
    load_ctr = rot_ref_ctr*1
    g_y = groupY(load_ctr,r)
    
    vec_a = np.array([0,0])
    vec_b_ref = np.array([0,0])
    
    result_ctr = copy.deepcopy(rot_ref_ctr)
    lat_ctr = []
    avg_ref_ang = 0
    
    ########## Sort y values in each group and refine the angle ##########
    ref_ang = []
    for ea_g in g_y:
        if len(ea_g)>1:
            ea_g_arr = np.array(ea_g)
        
            result = np.polyfit(ea_g_arr[:,1], ea_g_arr[:,0], 1)
            ref_ang.append(np.arctan2(result[0],1)* 180 / np.pi)
    
    if len(ref_ang)>0:
        avg_ref_ang =  sum(ref_ang)/len(ref_ang) 
    else:
        avg_ref_ang = 0
        
    rot_ref_ctr2 = rotCtr(pattern,load_ctr,avg_ref_ang)
    
    g_y = groupY(rot_ref_ctr2,r)

    g_y_len = [len(l) for l in g_y]
    
    if max(g_y_len)>1:
        ################ Refine y values #######################            
        n = len(rot_ref_ctr2)
        ref_y = []
        for group in g_y:
            cur_mean = 0
            sum_cur = 0
            for each in group:
                sum_cur += each[2]
            for each in group:
                cur_mean += each[0]*(each[2]/sum_cur)
            ref_y.append(cur_mean) # Weighted mean     
            
        # Change y values to the averaged y in each group    
        result_ctr = copy.deepcopy(rot_ref_ctr2)
        for j in range (n):
            cur_y = rot_ref_ctr2[j,0]
            d_y = [np.abs(s-cur_y) for s in ref_y]
            min_y_ind = np.argmin(d_y)
            result_ctr[j][0] = ref_y[min_y_ind]     
        
        ################ Vec a #######################    
        x_g = []    
        tit_diff_x = []  
        for cur_y in ref_y:
            cur_x_g = result_ctr[np.where(result_ctr[:,0]== cur_y)]
            if len(cur_x_g)>1:
                cur_x_g.sort(axis = 0)
                x_g.append(cur_x_g)
                cur_diff_x = cur_x_g[1:]-cur_x_g[:-1]
                tit_diff_x.append(cur_diff_x)
            else:
                x_g.append(cur_x_g)   
        
        ###################### Calculate average distance ################
        if len(tit_diff_x)>0:
            outl_rem_x = []
            mean_diff_x = []
            
            for i in range (len(tit_diff_x)):
                for x in tit_diff_x[i]:
                    outl_rem_x.append(x[1])
                    
            outl_rem_x = np.array(outl_rem_x)
            q1, q3= np.percentile(outl_rem_x,[25,75])
            lower_bound = 2.5*q1 - 1.5*q3
            upper_bound = 2.5*q3 - 1.5*q1
            
            for each_g in tit_diff_x:
                each_g_mod = each_g*1
                for idx in range (len(each_g)):
                    if each_g[idx,1]<lower_bound or each_g[idx,1]>upper_bound:
                        each_g_mod = np.delete(each_g,idx,axis = 0)               
                
                if len(each_g_mod)>0:
                    cur_mean = np.mean(each_g_mod[:,1],axis=0)
                    mean_diff_x.append([cur_mean,len(each_g_mod)])
                
            mean_diff_x_arr = np.array(mean_diff_x)
            
            if len(mean_diff_x_arr)>0:
                count = 0 
                sum_x = 0
                for i in range (len(mean_diff_x_arr)):
                    sum_x += mean_diff_x_arr[i,0]* mean_diff_x_arr[i,1]
                    count += mean_diff_x_arr[i,1]
                
                vec_a = np.array([0, sum_x/count])
                
                ######### Find vector b #########
                set_ct_ind = np.argmax(result_ctr[:,2])
                set_ct = result_ctr[set_ct_ind]
                
                # Find rough b
                min_nn = 10**38
                nn_vecb_rough = np.array([-1,-1,-1])
                for gn in range (len(x_g)):
                    cur_ct = x_g[gn]
                    if set_ct[0] not in cur_ct[:,0]:
                        dis_xy = cur_ct - set_ct
                        dis_norm = np.linalg.norm(dis_xy[:,:2],axis = 1)
                        xy_min = np.min(dis_norm)
                        if xy_min<=min_nn:  
                            min_nn = xy_min 
                            nn_vecb_rough = cur_ct[np.argmin(dis_norm)]   
                
                # Generate hypothetical lattice
                h,w = pattern.shape 
                lat_ctr = [set_ct[:2]]
                
                ###### Generate pts along vector a (middle row) ######
                # one side    
                cur_h1 = set_ct[0]
                cur_w1 = set_ct[1]
                cur_ct1 = set_ct[:2]*1
                while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                        cur_h1,cur_w1 = cur_ct1-vec_a
                        if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                            cur_ct1 = [cur_h1,cur_w1]
                            lat_ctr.append([cur_h1,cur_w1])
                
                # the other side
                cur_h2 = set_ct[0]
                cur_w2 = set_ct[1]
                cur_ct2 = set_ct[:2]*1.0
                while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                    cur_h2,cur_w2 = cur_ct2+vec_a
                    if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                        cur_ct2 = [cur_h2,cur_w2]
                        lat_ctr.append([cur_h2,cur_w2])                            
                        
                ######### Refine Vector b #########
                vec_b = nn_vecb_rough - set_ct
                if  vec_b[0]<0:
                    vec_b = -vec_b
            
                vec_b_rough = vec_b [:2]
            
                diff_y_ref = []   

                look_y = set_ct[0]-vec_b_rough[0]
                est_ct = lat_ctr - vec_b_rough 
                while look_y>0:
                    for each in est_ct:
                        each_diff_xy = each - result_ctr[:,:2]
                        
                        each_dis = each_diff_xy[:,0]**2+each_diff_xy[:,1]**2
                        each_dis_min = np.min(each_dis)
                        if each_dis_min<r**2:
                            cum_row = round(np.abs(np.mean(each[:][0])-set_ct[0])/vec_b_rough[0])
                            diff_y_ref.append(each_diff_xy[np.argmin(each_dis)]/cum_row)
                    look_y -= vec_b_rough[0]
                    est_ct -= vec_b_rough
        
                look_y = set_ct[0]+vec_b_rough[0]
                est_ct = lat_ctr + vec_b_rough
                while look_y<h:
                    for each in est_ct:
                        each_diff_xy = result_ctr[:,:2] - each
                        
                        each_dis = each_diff_xy[:,0]**2+each_diff_xy[:,1]**2
                        each_dis_min = np.min(each_dis)
        
                        if each_dis_min<r**2:
                            cum_row = round(np.abs(np.mean(each[:][0])-set_ct[0])/vec_b_rough[0])
                            diff_y_ref.append(each_diff_xy[np.argmin(each_dis)]/cum_row)   
                    look_y += vec_b_rough[0]
                    est_ct += vec_b_rough      
                 
                vec_b_ref = vec_b_rough*1.0
                if len(diff_y_ref)==0:
                    diff_y_ref.append([0,0])
                diff_y_ref = np.array(diff_y_ref)        
                vec_b_ref[1] = vec_b_ref[1] + np.mean(diff_y_ref[:,1])
    
    lat_ctr_arr = np.array(lat_ctr)
    return vec_a, vec_b_ref, result_ctr, lat_ctr_arr, avg_ref_ang

def latBack(refe_a,refe_b,angle):
    """
    Transform the lattice vectors to the default coordinate system.

    Parameters
    ----------
    refe_a : 1D array of float
        Array of the vector a.
    refe_b : 1D array of float
        Array of the vector b.
    angle : float
        The rotation angle.

    Returns
    -------
    a_init : 1D array of float
        Transformed array of the vector a.
    b_init : 1D array of float
        Transformed array of the vector b.

    """
    ang_init_back = angle*np.pi/180
    a_init = np.array([refe_a[1]*np.sin(ang_init_back),refe_a[1]*np.cos(ang_init_back)])
    b_init = np.array([refe_b[1]*np.sin(ang_init_back)+refe_b[0]*np.cos(ang_init_back),refe_b[1]*np.cos(ang_init_back)-refe_b[0]*np.sin(ang_init_back)])
    
    return a_init,b_init

def angle_of(disk):
    angle = np.degrees(np.arctan2(disk[0], disk[1]))
    if angle < 0:
        angle = 360 + angle
    return angle

def mtc_latfit(disks,center_disk,r):
    # Assume that the disks are already perfectly fit to the pattern they were generated from, and that the center disk position and radius is correct
    # Label disks with distance to center disk and rotation angle relative to an origin at center disk
    labeled_disks = []
    for disk in disks:
        dist_to_center = np.linalg.norm(disk - center_disk)
        if dist_to_center < 5:
            continue
        angle = angle_of(disk-center_disk)
        labeled_disk = [disk[0], disk[1], dist_to_center, angle]

        labeled_disks.append(labeled_disk)

    labeled_disks = np.array(labeled_disks)

    # Group disks with same distance, filter colinear pairs

    labeled_disks = labeled_disks[labeled_disks[:, 2].argsort()] # sort by distance to center
    n = len(labeled_disks)
    i = 0
    grouped_disks = []
    threshold = 0.5*r
    while i < n:
        dist_i = labeled_disks[i,2]
        # Check disks with greater distance until difference in distances is greater than threshold.  Then append all those disks to grouped_disks, and set i to the next highest index outside that group.
        for j in range(n):
            dist_j = labeled_disks[j,2]
            # Ignore all disks with a lower or equal distance
            if dist_j <= dist_i:
                continue
            if (dist_j - dist_i) > threshold:
                grouped_disks.append(labeled_disks[i:j-1,:])
                i = j-1
                break
        i += 1
    

    
    for group in grouped_disks:
        group = filterColinear(group)

    # Choose lowest angle nearest neighbor as vec a
    # grouped_disks structure => [group, disk, disk_properties[x,y,dist,angle]]

    vec_a = grouped_disks[0][0][0:2]-center_disk

    # If other nearest neighbor exists that is not colinear, choose next lowest angle nearest neighbor as vec b

    if len(grouped_disks[0]) > 1:
        vec_b = grouped_disks[0,1,0:2]-center_disk
        return np.array([vec_a, vec_b])

    # Else choose second nearest neighbor with the next lowest angle (other disk angles > vec b angle > vec a angle) as vec b
    else:
        vec_a_angle = grouped_disks[0,0,3]
        for disk in grouped_disks[1]:
            if disk[3] > vec_a_angle:
                vec_b = disk[0:2]-center_disk
                return np.array([vec_a, vec_b])
        return np.array([0,0])
        


def filterColinear(group, tolerance=5):
    for i in range(len(group)):
        if np.array_equal(group[i],[-1,-1,-1,-1]):
                continue
        for j in range(len(group)):
            if np.array_equal(group[i],[-1,-1,-1,-1]):
                continue
            angle_ij = np.abs(group[i,3]-group[j,3])
            if angle_ij > 180-tolerance and angle_ij < 180+tolerance:
                group[j] = [-1,-1,-1,-1]

    result = []
    for disk in group:
        if not np.array_equal(disk,[-1,-1,-1,-1]):
            result.append(disk)
    
    result = np.array(result)
    print(result)
    return result

            

def paramConstructor(PROCESSES, data, kernel, r, center_disk, angle, bckg_intensity, bckg_std):
    img_h,img_w,diff_pat_h,diff_pat_w = data.shape
    full_pattern_list = []
    for row_idx in range(img_h):
        for col_idx in range(img_w):
            full_pattern_list.append([row_idx,col_idx])
    full_pattern_list = np.array_split(full_pattern_list, PROCESSES)

    params = []
    for pattern_list in full_pattern_list:
        params.append((data,pattern_list,kernel,r,center_disk,angle,bckg_intensity,bckg_std))
    
    print('-----Params Constructed-----')
    return params

def processPatterns(data, pattern_list, kernel, r, center_disk, angle, bckg_intensity, bckg_std):
    return_val = []

    for pattern_coords in pattern_list:
        row_idx = pattern_coords[0]
        col_idx = pattern_coords[1]
        pattern = data[row_idx,col_idx]
        if isVacuum(pattern,center_disk, r, bckg_intensity, bckg_std):
            print(f'Process {os.getpid()} skipped r:{row_idx}, c: {col_idx} for being a vacuum pattern.')
            continue
        cros_map = crossCorr(pattern,kernel)   
        disks = ctrDet(cros_map, r, kernel, 10, 10,) 
        # disks = filterDisks(pattern,disks,r,bckg_intensity,bckg_std)
        if len(disks) > 5:
            ctr_cur,r_cur = ctrRadiusIni(pattern)
            if np.linalg.norm(ctr_cur-center_disk) <= 2: # 2px
                ctr = ctr_cur
            else:
                ctr = center_disk
                ctr[1] = np.round(ctr[1])
                ctr[0] = np.round(ctr[0])            
            
            ref_ctr = radGradMax(pattern, disks, r,rn=20, ra=2, n_p=40, threshold=3)           
            
            ctr_vec = ref_ctr[:,:2] - ctr
            ctr_diff = ctr_vec[:,0]**2 + ctr_vec[:,1]**2
            ctr_idx = np.where(ctr_diff==ctr_diff.min())[0][0]
            ref_ctr[ctr_idx,2] = 10**38
            rot_ref_ctr = rotCtr(pattern,ref_ctr,angle)
            ret_a,ret_b,ref_ctr2, mid_ctr,ref_ang = latFit(pattern,rot_ref_ctr,r)
            
            if any(ret_a!=0) and any(ret_b!=0):
                a_back,b_back = latBack(ret_a, ret_b, angle+ref_ang)
                return_val.append([row_idx, col_idx, a_back, b_back])
                print(f'Process {os.getpid()} returned {[row_idx, col_idx, a_back, b_back]} for r:{row_idx}, c: {col_idx}')

    return return_val


def driver_func(data, kernel, r, center_disk, angle, bckg_intensity, bckg_std):
    img_h,img_w,diff_pat_h,diff_pat_w = data.shape
    PROCESSES = mp.cpu_count()
    print(f'{PROCESSES} cores available')

    params = paramConstructor(PROCESSES, data, kernel, r, center_disk, angle, bckg_intensity, bckg_std)
    results = []
    lattice_params = np.zeros((img_h,img_w,2,2),dtype = float)
    with mp.Pool(PROCESSES) as pool:
        for p in params:
            results.append(pool.apply_async(processPatterns, p))
        n = 0
        while True:
            time.sleep(1)
            n+=1
            print(f'{n} seconds')
            # catch exception if results are not ready yet
            try:
                ready = [result.ready() for result in results]
                successful = [result.successful() for result in results]
            except Exception:
                continue
            # exit loop if all tasks returned success
            if all(successful):
                break
            # raise exception reporting exceptions received from workers
            if all(ready) and not all(successful):
                raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')
    
    for result_list in results:
        result_list = result_list.get()
        for result in result_list:
            lattice_params[result[0],result[1],0,:] = result[2]
            lattice_params[result[0],result[1],1,:] = result[3]

    return lattice_params

def radialIntensity(pattern, center_disk, r, plot=False):
    cen_x = center_disk[1]
    cen_y = center_disk[0]
    
    # Get image parameters
    a = pattern.shape[0]
    b = pattern.shape[1]

    [X, Y] = np.meshgrid(np.arange(b) - cen_x, np.arange(a) - cen_y)
    R = np.sqrt(np.square(X) + np.square(Y))

    rad = np.arange(1, np.max(R), 1)
    intensity = np.zeros(len(rad))
    index = 0

    for i in rad:
        mask = np.greater(R, i - 2) & np.less(R, i + 2)
        values = pattern[mask]
        intensity[index] = np.mean(values)
        index += 1

    if plot:
        plt.plot(rad, intensity)
        plt.title(f'r = {r}')
        plt.xlabel('Radius (px)')
        plt.ylabel('Intensity')
        plt.show()
    
    return rad, intensity

def quantifyBackground(data, center_disk, r):
    img_h,img_w,diff_pat_h,diff_pat_w = data.shape
    bckg = []
    for row in range(img_h):
        for col in range(img_w):
            pattern = data[row,col]
            masked = maskDisk(pattern,center_disk,r,-1) # mask out the center disk
            bckg.append(np.array([np.mean(masked[masked>0]),np.std(masked[masked>0])]))
    bckg = np.array(bckg)
    bckg = bckg[bckg[:, 0].argsort()]  # sort by mean intensity

    # Take the lowest 1% of mean intensities (assuming at least 1% of the image is vacuum)
    bckg = bckg[0:int(0.01*img_h*img_w),:]

    intensity = np.mean(bckg[:,0])
    std = np.mean(bckg[:,1])
    print(f'Dataset has average background intensity [{intensity}] with average standard deviation [{std}]. This was calculated from {bckg.size/2} patterns with intensity range [{bckg[0,0]}-{bckg[-1,0]}]')
    return intensity, std

def isVacuum(pattern, center_disk, r, bckg_intensity, bckg_std):
    masked = maskDisk(pattern,center_disk,r,-1)
    if np.mean(masked[masked>0]) > bckg_intensity+0.25*bckg_std:
        return False
    return True

def filterDisks(pattern, disks, r, bckg_intensity, bckg_std):
    results = []
    for disk in disks:
        disk_mask = maskDisk(pattern, disk, r, 1)
        if np.mean(disk_mask[disk_mask>0]) > bckg_intensity+0.25*bckg_std:
            results.append(disk)
    results = np.array(results)
    if disks.shape != results.shape:
        print(f'{disks.shape} reduced to {results.shape}')
    return results

def maskDisk(pattern, disk, r, factor):

        ##########################################################
        #    
        #   Defines a circular virtual detector at a given position in the pattern and radius.
        #
        #   THIS CALL REPLACES THE DEFAULT DETECTOR FOR BFDF IMAGING.
        #
        #   Returns the pattern and stores it in the object for later use
        #
        #   cenx,ceny: coordinates for the center of the circle (inside the pattern)
        #   rad: radius of the circle in pixels
        #   factor: -1 for deteting everything BUT the masked region, +1 for BF/deteting the masked region
        #   
        ##########################################################
        
        diffpatx,diffpaty = pattern.shape
        cenx = disk[1]
        ceny = disk[0]
        Y,X = np.ogrid[:diffpatx, :diffpaty]
        dist_from_center = np.sqrt((X - cenx)**2 + (Y - ceny)**2)
        
        mask = factor*dist_from_center <= factor*r*1.3

        masked = (pattern*mask)
        return masked

def saveResults(results):
    filename = str(datetime.now()).split('.')[0].replace(':','_') + '.csv'
    with open(filename,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

def testResults(results_file, test_file):
    with open(results_file, 'r') as res, open(test_file, 'r') as test:
        results = res.readlines()
        testset = test.readlines()

    for row in results:
        if row not in testset:
            print(row)



# def processPattern(data, row_idx, col_idx, kernel, r, center_disk, angle):
#     pattern = data[row_idx,col_idx]
        
#     cros_map = crossCorr(pattern,kernel)   
#     disks = ctrDet(cros_map, r, kernel, 10, 10) 
    
#     if len(disks) > 5:
#         ctr_cur,r_cur = ctrRadiusIni(pattern)
#         if np.linalg.norm(ctr_cur-center_disk) <= 2: # 2px
#             ctr = ctr_cur
#         else:
#             ctr = center_disk
#             ctr[1] = np.round(ctr[1])
#             ctr[0] = np.round(ctr[0])            
        
#         ref_ctr = radGradMax(pattern, disks, r,rn=20, ra=2, n_p=40, threshold=3)           
        
#         ctr_vec = ref_ctr[:,:2] - ctr
#         ctr_diff = ctr_vec[:,0]**2 + ctr_vec[:,1]**2
#         ctr_idx = np.where(ctr_diff==ctr_diff.min())[0][0]
#         ref_ctr[ctr_idx,2] = 10**38
#         rot_ref_ctr = rotCtr(pattern,ref_ctr,angle)
#         ret_a,ret_b,ref_ctr2, mid_ctr,ref_ang = latFit(pattern,rot_ref_ctr,r)
        
#         if any(ret_a!=0) and any(ret_b!=0):
#             a_back,b_back = latBack(ret_a, ret_b, angle+ref_ang)
#             return_val = [row_idx, col_idx, a_back, b_back]
#             print(f'Process {os.getpid()} returned {return_val} for r:{row_idx}, c: {col_idx}')
#             return return_val
