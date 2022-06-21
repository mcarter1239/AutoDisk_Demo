# Imports

from autodisk_mtc import *

if __name__ == '__main__':
    # Input and sanitize 4D-STEM file.

    data_name = 'pdpt_x64_y64.raw'
    data = readData(data_name)
    data = preProcess(data)
    img_h,img_w,diff_pat_h,diff_pat_w = data.shape
    print(img_h)
    
    # Determine center disk position and radius from sum pattern

    avg_pattern = generateAvgPattern(data)
    center_disk,r = ctrRadiusIni(avg_pattern)

    bckg_intensity, bckg_std = quantifyBackground(data,center_disk,r)
    
    # Generate kernel, then cross-correlate

    kernel = generateKernel(avg_pattern,center_disk,r)
    cros_map = crossCorr(avg_pattern,kernel)

    # Detect disks and refine positions with radial gradient maximization

    detected_disks = ctrDet(cros_map, r, kernel)
    refined_disks_weights = radGradMax(avg_pattern, detected_disks, r,ra=4)
    refined_disks_list = refined_disks_weights[:,:2]
    print(refined_disks_list)

    # Detect the angle of rotation to put disks along a horizontal axis
    angle, refined_disks_weights = detAng(refined_disks_weights,center_disk,r)

    print('Estimated rotation angle: ',angle,'(deg)')

    # Generate the coordinate of disks in the new coordinate system 
    rotated_refined_disks_weights = rotCtr(avg_pattern,refined_disks_weights,angle) 

    # Fit the rotated disks to a set of two basis vectors
    vec_a_rotated,vec_b_rotated,rotated_refined_disks_weights, middle_row_disks,angle_delta = latFit(avg_pattern,rotated_refined_disks_weights,r)

    radialIntensity(avg_pattern,center_disk,r,True)
    vecs_mtc = mtc_latfit(refined_disks_list[:,0:2],center_disk,r)
    print(vecs_mtc)

    print('Two lattice vectors: vector_a--[',vec_a_rotated[0],vec_a_rotated[1], '] and vector_b--[',vec_b_rotated[0],vec_b_rotated[1],']')

    rotated_pattern = rotImg(avg_pattern, angle+angle_delta, center_disk)
    vec_a, vec_b= latBack(vec_a_rotated, vec_b_rotated, angle)

    start = time.perf_counter()
    lattice_params = driver_func(data, kernel, r, center_disk, angle, bckg_intensity, bckg_std)
    print(f'Took {time.perf_counter()-start} seconds to complete')

    lat_fil = latDist(lattice_params,vec_a,vec_b)
    st_xx,st_yy,st_xy,st_yx,tha_ang = calcStrain(lat_fil, vec_a, vec_b)
    rdbu = plt.cm.get_cmap('RdBu')
    cmap_re = rdbu.reversed()
    
    input_min=-0.058
    input_max=0.058

    l_min = input_min*100
    l_max = input_max*100

    titles = ["$\epsilon_{xx}(\%)$","$\epsilon_{yy}(\%)$","$\epsilon_{xy}(\%)$","$\Theta$"]
    comb = [st_xx,st_yy,st_xy,tha_ang]

    fig,axs = plt.subplots(2,2,figsize = (10,10))
    i=0
    for row in range (2):
        for col in range (2):
            if row==1 and col==1:
                ax = axs[row,col]
                pcm = ax.imshow(comb[i],cmap=cmap_re,vmin=l_min,vmax=l_max)
                ax.set_title(titles[i],fontsize=28)
                ax.set_axis_off()
                fig.colorbar(pcm,ax=ax)
                i +=1
            else:    
                ax = axs[row,col]
                pcm = ax.imshow(comb[i]*100,cmap=cmap_re,vmin=l_min,vmax=l_max)
                ax.set_title(titles[i],fontsize=28)
                ax.set_axis_off()
                fig.colorbar(pcm,ax=ax)
                i +=1
        
    plt.subplots_adjust(wspace=0.25,hspace=0.25)
    plt.show()
    saveResults(lat_fil)