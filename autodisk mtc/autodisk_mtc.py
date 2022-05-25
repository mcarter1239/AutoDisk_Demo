import copy
from scipy import stats,signal
import numpy as np
from skimage.transform import resize
from skimage import feature
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from mtc_helpers import *

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

def process(data, row_idx, col_idx, kernel, r, center_disk, angle, lat_par, lock):
    pattern = copy.deepcopy(data[row_idx,col_idx])
        
    cros_map = crossCorr(pattern,kernel)   
    disks = ctrDet(cros_map, r, kernel, 10, 10) 
    
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
            lock.acquire()
            lat_par[row_idx,col_idx,0,:] = a_back
            lat_par[row_idx,col_idx,1,:] = b_back
            lock.release()
    