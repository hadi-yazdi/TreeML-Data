
import numpy as np
import os
import math
import pandas as pd
import time
from circle_fit import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import point_transformation as pt
import multiprocessing


##################################################
def TreeML_SM(PathToPointclouds,PathToTranformationMatrix,PathToSave):
    files_list = os.listdir(PathToPointclouds)
    df = pd.DataFrame(index=range(len(files_list)))


    for i in range(len(files_list)):
        if files_list[i].split('_')[-1] != 'trashtree.txt':

            pointcloud = np.loadtxt(PathToPointclouds+'/'+files_list[i], delimiter=' ', usecols = (0,1,2))
            df.loc[i,'treeID'] = files_list[i].split('.')[0]


            # noise detection

            nbrs = NearestNeighbors(n_neighbors=20).fit(pointcloud)
            distances, indices = nbrs.kneighbors(pointcloud)
            pointcloud = pointcloud[np.where(distances.mean(axis=1) < 0.3)]



            # Tree location
            
            tl_slice = pointcloud[np.where(pointcloud[:,2] < pointcloud[:,2].min()+0.3)]
            if tl_slice.shape[0] > 3:
                tl_xc, tl_yc, tl_r, sigma = taubinSVD(tl_slice[:,:2])
                tree_location = [tl_xc, tl_yc, tl_slice[:,2].min()]
                df.loc[i,'location.x'] = tl_xc
                df.loc[i,'location.y'] = tl_yc
                df.loc[i,'location.z'] = tl_slice[:,2].min()
            else:
                tl_slice = pointcloud[np.where(pointcloud[:,2] < pointcloud[:,2].min()+0.5)]                
                tl_xc, tl_yc, tl_r, sigma = taubinSVD(tl_slice[:,:2])
                tree_location = [tl_xc, tl_yc, tl_slice[:,2].min()]
                df.loc[i,'location.x'] = tl_xc
                df.loc[i,'location.y'] = tl_yc
                df.loc[i,'location.z'] = tl_slice[:,2].min()

            # Geographic location
            point_geographic = pt.point_transformation([df.loc[i,'location.x'], df.loc[i,'location.y'], df.loc[i,'location.z']], PathToTranformationMatrix)
            
            # point_geographic = np.array(point_geographic)
            df.loc[i,'location.latitude'] = point_geographic[0]
            df.loc[i,'location.longitude'] = point_geographic[1]
            df.loc[i,'location.altitude'] = point_geographic[2]


            # transforming the pointcloud to 0,0,0

            xyz_transformed = np.zeros(pointcloud.shape)
            xyz_transformed[:,0] = pointcloud[:,0] - tree_location[0]
            xyz_transformed[:,1] = pointcloud[:,1] - tree_location[1]
            xyz_transformed[:,2] = pointcloud[:,2] - tree_location[2]



            # Tree DBH
            DBH_slice = xyz_transformed[np.where((xyz_transformed[:,2] < xyz_transformed[:,2].min()+1.33) &
                                                (xyz_transformed[:,2] > xyz_transformed[:,2].min()+1.27))]
            if DBH_slice.shape[0] > 3:

                cf_Models = np.array([hyperLSQ(DBH_slice[:,:2]),
                            standardLSQ(DBH_slice[:,:2]),
                            riemannSWFLa(DBH_slice[:,:2]),
                            prattSVD(DBH_slice[:,:2]),
                            taubinSVD(DBH_slice[:,:2]),
                            hyperSVD(DBH_slice[:,:2]),
                            kmh(DBH_slice[:,:2])])

                DBH_xc, DBH_yc, DBH_r, sigma = cf_Models[np.where(cf_Models[:,3] == cf_Models[:,3].min())][0]

                df.loc[i,'DBH_m_'] = DBH_r*2
            
            else:
                DBH_slice = xyz_transformed[np.where((xyz_transformed[:,2] < xyz_transformed[:,2].min()+1.4) &
                                                    (xyz_transformed[:,2] > xyz_transformed[:,2].min()+1.2))]  
                if DBH_slice.shape[0] > 3:                
                    cf_Models = np.array([hyperLSQ(DBH_slice[:,:2]),
                                standardLSQ(DBH_slice[:,:2]),
                                riemannSWFLa(DBH_slice[:,:2]),
                                prattSVD(DBH_slice[:,:2]),
                                taubinSVD(DBH_slice[:,:2]),
                                hyperSVD(DBH_slice[:,:2]),
                                kmh(DBH_slice[:,:2])])

                    DBH_xc, DBH_yc, DBH_r, sigma = cf_Models[np.where(cf_Models[:,3] == cf_Models[:,3].min())][0]

                    df.loc[i,'DBH_m_'] = DBH_r*2                
                else:
                    df.loc[i,'DBH_m_'] = 'NaN'



            # Tree Height
            df.loc[i,'treeHeight_m_'] = xyz_transformed[:,2].max()


            # Tree crown start
            cs_size = 0.1
            cs_move = 0
            cs_slice_new = xyz_transformed[np.where((xyz_transformed[:,2] < xyz_transformed[:,2].min()+1.3+cs_size) &
                                                (xyz_transformed[:,2] > xyz_transformed[:,2].min()+1.3))]
            dif = 0
            if cs_slice_new.shape[0] > 3:
                while dif <= 0:
                    cs_slice = cs_slice_new
                    cf_Models = np.array([hyperLSQ(cs_slice[:,:2]),
                                standardLSQ(cs_slice[:,:2]),
                                riemannSWFLa(cs_slice[:,:2]),
                                prattSVD(cs_slice[:,:2]),
                                taubinSVD(cs_slice[:,:2]),
                                hyperSVD(cs_slice[:,:2]),
                                kmh(cs_slice[:,:2])])
                    cs_xc, cs_yc, cs_r, sigma = cf_Models[np.where(cf_Models[:,3] == cf_Models[:,3].min())][0]    

                    cs_move += 0.1
                    cs_slice_new = xyz_transformed[np.where((xyz_transformed[:,2] < xyz_transformed[:,2].min()+1.3+cs_size+cs_move) &
                                                        (xyz_transformed[:,2] > xyz_transformed[:,2].min()+1.3+cs_move))]
                    if cs_slice_new.shape[0] > 3:
                        cf_Models = np.array([hyperLSQ(cs_slice_new[:,:2]),
                                    standardLSQ(cs_slice_new[:,:2]),
                                    riemannSWFLa(cs_slice_new[:,:2]),
                                    prattSVD(cs_slice_new[:,:2]),
                                    taubinSVD(cs_slice_new[:,:2]),
                                    hyperSVD(cs_slice_new[:,:2]),
                                    kmh(cs_slice_new[:,:2])])
                        cs_xc_new, cs_yc_new, cs_r_new, sigma_new = cf_Models[np.where(cf_Models[:,3] == cf_Models[:,3].min())][0]   
                        dif = cs_r_new - (cs_r*1.2)
                        TreeCrownStart = 1.3+(cs_size)+cs_move
                    else:
                        dif = 1
                        TreeCrownStart = 'NaN'                        
            else:
                TreeCrownStart = 'NaN'                
            
            df.loc[i,'crownStartHeight_m_'] = TreeCrownStart



            # Crown projection area
            if TreeCrownStart == 'NaN':
                TreeCrownStart = 0
            if TreeCrownStart < xyz_transformed[:,2].max():
                hull = ConvexHull(xyz_transformed[np.where(xyz_transformed[:,2] > TreeCrownStart)][:,:2], incremental=True)
            else:
                hull = ConvexHull(xyz_transformed[:,:2], incremental=True)

            df.loc[i,'crownProjectionArea_m2_'] = hull.volume



            # Crown radius each 5 degree
            def distance_to_zero(point):
                return math.sqrt(((point[0])**2)+((point[1])**2))

            def Crown_radius(pointcloud, Degree, crown_start):
                
                # Degree = 45  # Degree must be according to degree and radian circle (270 <= degree < 90)
                # crown_start = 2
                # pointcloud is the original pointcloud as an array with three column of x,y,z and n raws
                # tree_location is a list of three value [x,y,z]
                
                #defining the slope of the line according to the degree
                Slope = math.tan(Degree * math.pi / 180)

                # finding the points close to the specific line with defined slope
                distoline = 0.05

                if Degree != -90:
                    Points_W = xyz_transformed[np.where((xyz_transformed[:,2] > crown_start) & 
                                                    (abs((Slope*xyz_transformed[:,0])-xyz_transformed[:,1])/math.sqrt((Slope*Slope)+1) < distoline) &
                                                    (xyz_transformed[:,0] < 0))]
                    Points_E = xyz_transformed[np.where((xyz_transformed[:,2] > crown_start) & 
                                                    (abs((Slope*xyz_transformed[:,0])-xyz_transformed[:,1])/math.sqrt((Slope*Slope)+1) < distoline) &
                                                    (xyz_transformed[:,0] > 0))]
                else:
                    Points_W = xyz_transformed[np.where((xyz_transformed[:,2] > crown_start) & 
                                                    (abs((Slope*xyz_transformed[:,0])-xyz_transformed[:,1])/math.sqrt((Slope*Slope)+1) < distoline) &
                                                    (xyz_transformed[:,1] < 0))]
                    Points_E = xyz_transformed[np.where((xyz_transformed[:,2] > crown_start) & 
                                                    (abs((Slope*xyz_transformed[:,0])-xyz_transformed[:,1])/math.sqrt((Slope*Slope)+1) < distoline) &
                                                    (xyz_transformed[:,1] > 0))] 
                
                if len(Points_W) == 0:
                    dis_center_W = 0
                else:
                    # W is for the west sides (90 <= degree < 270), and E is for east sides (270 <= degree < 90)

                    # defining the distance of the point to zero and adding a new colum to the data
                    temp = np.reshape(np.array([]),(-1,1))
                    for i in range(len(Points_W)):
                        temp = np.append(temp, np.reshape(distance_to_zero(Points_W[i]),(-1,1)), axis=0)
                    Points_W = np.append(Points_W, temp, axis=1)

                    # finding the minimum distance to the tree in both west and east side and round them with 2 decimals
                    Points_W[:,3] = np.around(Points_W[:,3], decimals=2)
                    dis_center_W = np.amax(Points_W[:,3])

                if len(Points_E) == 0:
                    dis_center_E = 0
                else:
                    # W is for the west sides (90 <= degree < 270), and E is for east sides (270 <= degree < 90)

                    # defining the distance of the point to zero and adding a new colum to the data
                    temp = np.reshape(np.array([]),(-1,1))
                    for i in range(len(Points_E)):
                        temp = np.append(temp, np.reshape(distance_to_zero(Points_E[i]),(-1,1)), axis=0)
                    Points_E = np.append(Points_E, temp, axis=1)        
                    
                    # finding the minimum distance to the tree in both west and east side and round them with 2 decimals
                    Points_E[:,3] = np.around(Points_E[:,3], decimals=2)
                    dis_center_E = np.amax(Points_E[:,3])
                
                return dis_center_W, dis_center_E

            
            Degrees_EW = list(range(0, 90, 5))
            Degrees_NS = list(range(-90, 0, 5))
            crownDiameter = []
            if 'crownDiameterMax_m_' not in df:
                df.insert(loc=11, column='crownDiameterMax_m_', value='')

            for degree in Degrees_EW:
                dis_center_W, dis_center_E = Crown_radius(xyz_transformed, degree, TreeCrownStart)
                df.loc[i, f'crownW_{degree}d_m_'] = dis_center_W
                df.loc[i, f'crownE_{degree}d_m_'] = dis_center_E
                crownDiameter.append(dis_center_W+dis_center_E)


            for degree in Degrees_NS:
                dis_center_N, dis_center_S = Crown_radius(xyz_transformed, degree, TreeCrownStart)
                df.loc[i, f'crownN_{90+degree}d_m_'] = dis_center_N
                df.loc[i, f'crownS_{90+degree}d_m_'] = dis_center_S
                crownDiameter.append(dis_center_N+dis_center_S)

            # Crown diameter Max
            if len(crownDiameter) != 0:
                df.loc[i, 'crownDiameterMax_m_'] = max(crownDiameter)
            else:
                df.loc[i, 'crownDiameterMax_m_'] = 'NaN'

            print ('****Tree '+ files_list[i] +' is finished****')
            
            

    df.to_csv(PathToSave+'/'+os.path.split(PathToPointclouds)[-1]+'_dataset.csv',index=False)

    return df



# RUN
##########################################################################################
################################### Path to projects #####################################

PathToPCFolder = 'J:/pointclouds/Dataset_tree'
PathToTMFolder = 'J:/pointclouds/Dataset_transformation_matrix/'
PathToSave = 'J:/pointclouds/Dataset_SM/Projects_Dataset'
project_list = os.listdir(PathToPCFolder)

##########################################################################################
##########################################################################################

def run(i):   # i should be run in range(len(project_list)) ---> projects in the folder

    start = time.time()

    PathToPointclouds = 'J:/pointclouds/Dataset_tree/'+ project_list[i]
    PathToTranformationMatrix = PathToTMFolder+project_list[i]+'.dat'
    df = TreeML_SM(PathToPointclouds,PathToTranformationMatrix,PathToSave)

    print ('****Project '+ project_list[i] +' is finished****')
    end = time.time()
    print(f'{round((end - start)/60, 2)} minutes calculation time')

    return df    
##########################################################################################
##########################################################################################

result = []
for i in range(13,len(project_list)):
    result.append(run(i))

# df_final = pd.concat(result)
# df_final.to_csv('J:/pointclouds/Dataset_SM/TreeML_dataset.csv',index=False)

# if __name__ == "__main__":
#     pool = multiprocessing.Pool(4)
#     start_time = time.perf_counter()
#     result = pool.map(run, range(len(project_list)))
#     finish_time = time.perf_counter()
#     print(f"Program finished in {(finish_time-start_time)/60} minutes")
#     #print(result)

#     df_final = pd.concat(result)
#     # save the final dataset
#     df_final.to_csv('J:/pointclouds/Dataset_SM/TreeML_dataset.csv',index=False)

