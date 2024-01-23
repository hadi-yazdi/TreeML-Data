import point_transformation as pt
import os
import numpy as np
import math
import pandas as pd
import time
import multiprocessing
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

############################### importing the dataset ####################################
############## just change here ##################
project_date = '2023-01-16'
project_name = '44'
##################################################

path = project_date+'/'+project_date+'_'+project_name
pathtobuilding = os.path.dirname(os.getcwd()) + '/Dataset_segmentation/' + path + '/'
pathtotree = os.path.dirname(os.getcwd()) + '/Dataset_tree/' + path + '/'
pathtosave = os.path.dirname(os.getcwd()) + '/Dataset_csv/'

files_list = os.listdir(path+'/treedata')

dataset = pd.concat((pd.read_csv(path+'/treedata/'+f) for f in files_list), ignore_index=True)
######################################

point_geographic = []
for i in range(len(files_list)):
    point_geographic.append(pt.point_transformation([dataset.loc[i,'location.x'], dataset.loc[i,'location.y'], dataset.loc[i,'location.z']], "matrix/"+project_date+'_'+project_name+".dat"))

point_geographic = np.array(point_geographic)
dataset.insert(loc=6, column='location.latitude', value=point_geographic[:,0])
dataset.insert(loc=7, column='location.longitude', value=point_geographic[:,1])
dataset.insert(loc=8, column='location.altitude', value=point_geographic[:,2])
dataset.insert(loc=3, column='botanical.name', value='')

dataset.to_csv(path+'/'+project_date+'_'+project_name+'_'+'dataset.csv',index=False)


############################### importing pointclouds ####################################

# importing the pointclouds
trees = []
for tree in list(dataset.loc[:,'treeID']):
    trees.append(np.loadtxt(pathtotree+tree+'.txt', delimiter=' ', usecols = (0,1,2)))
if os.path.isfile(pathtotree+project_date+'_'+project_name+'_trashtree.txt'):
    trees.append(np.loadtxt(pathtotree+project_date+'_'+project_name+'_trashtree.txt', delimiter=' ', usecols = (0,1,2)))

building = np.loadtxt(pathtobuilding+project_date+'_'+project_name+'_building.txt', delimiter=' ', usecols = (0,1,2))
###################################################################################################

def distance_to_zero(point):
    return math.sqrt(((point[0])**2)+((point[1])**2))

def distance_hight_of_obj(pointcloud, tree_location, Degree, crown_start):
    
    # Degree = 45  # Degree must be according to degree and radian circle (270 <= degree < 90)
    # crown_start = 2
    # pointcloud is the original pointcloud as an array with three column of x,y,z and n raws
    # tree_location is a list of three value [x,y,z]
    
    xyz_transformed = np.zeros(pointcloud.shape)
    xyz_transformed[:,0] = pointcloud[:,0] - tree_location[0]
    xyz_transformed[:,1] = pointcloud[:,1] - tree_location[1]
    xyz_transformed[:,2] = pointcloud[:,2] - tree_location[2]
    
    #defining the slope of the line according to the degree
    Slope = math.tan(Degree * math.pi / 180)

    # finding the points close to the specific line with defined slope (original 0.05)
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
                                        (xyz_transformed[:,1] > 0))]
        Points_E = xyz_transformed[np.where((xyz_transformed[:,2] > crown_start) & 
                                        (abs((Slope*xyz_transformed[:,0])-xyz_transformed[:,1])/math.sqrt((Slope*Slope)+1) < distoline) &
                                        (xyz_transformed[:,1] < 0))] 
    
    if len(Points_W) == 0:
        dis_to_tree_W = 'NaN'
        hight_W = 'NaN'
    else:
        # W is for the west sides (90 <= degree < 270), and E is for east sides (270 <= degree < 90)

        # defining the distance of the point to zero and adding a new colum to the data
        temp = np.reshape(np.array([]),(-1,1))
        for i in range(len(Points_W)):
            temp = np.append(temp, np.reshape(distance_to_zero(Points_W[i]),(-1,1)), axis=0)
        Points_W = np.append(Points_W, temp, axis=1)

        # finding the minimum distance to the tree in both west and east side and round them with 2 decimals
        Points_W[:,3] = np.around(Points_W[:,3], decimals=2)
        dis_to_tree_W = np.amin(Points_W[:,3])

        # removing the point further than 5 meter od the closest point to the tree
        Points_W = Points_W[np.where(Points_W[:,3] < dis_to_tree_W + 5)]

        #defining the hight of the object and round it with 2 decimals
        hight_W = np.amax(Points_W[:,2])
        hight_W = np.around(hight_W, decimals=2)

    if len(Points_E) == 0:
        dis_to_tree_E = 'NaN'
        hight_E = 'NaN'  
    else:
        # W is for the west sides (90 <= degree < 270), and E is for east sides (270 <= degree < 90)

        # defining the distance of the point to zero and adding a new colum to the data
        temp = np.reshape(np.array([]),(-1,1))
        for i in range(len(Points_E)):
            temp = np.append(temp, np.reshape(distance_to_zero(Points_E[i]),(-1,1)), axis=0)
        Points_E = np.append(Points_E, temp, axis=1)        
        
        # finding the minimum distance to the tree in both west and east side and round them with 2 decimals
        Points_E[:,3] = np.around(Points_E[:,3], decimals=2)
        dis_to_tree_E = np.amin(Points_E[:,3])

        # removing the point further than 5 meter od the closest point to the tree
        Points_E = Points_E[np.where(Points_E[:,3] < dis_to_tree_E + 5)]

        #defining the hight of the object and round it with 2 decimals
        hight_E = np.amax(Points_E[:,2])
        hight_E = np.around(hight_E, decimals=2)
    
    return dis_to_tree_W, dis_to_tree_E, hight_W, hight_E


# measuring distances

def dis(i):   # i should be run in range(dataset.shape[0])
    start = time.time()
    tree_location = [dataset.loc[i,'location.x'], dataset.loc[i,'location.y'], dataset.loc[i,'location.z']]
    crown_start = dataset.loc[i,'crownStartHeight(m)']
    pointcloud = np.concatenate(trees[:i] + trees[i+1:], axis=0)
    Degrees_EW = list(range(0, 90, 5))
    Degrees_NS = list(range(-90, 0, 5))
    df = pd.DataFrame(index=[i])
    
    for degree in Degrees_EW:
        dis_to_tree_W, dis_to_tree_E, hight_W, hight_E = distance_hight_of_obj(pointcloud, tree_location, degree, crown_start)
        df.loc[i, f'disAdjacentTreeW+{degree}d(m)'] = dis_to_tree_W
        df.loc[i, f'disAdjacentTreeE+{degree}d(m)'] = dis_to_tree_E
        df.loc[i, f'hightAdjacentTreeW+{degree}d(m)'] = hight_W
        df.loc[i, f'hightAdjacentTreeE+{degree}d(m)'] = hight_E
        
    for degree in Degrees_NS:
        dis_to_tree_N, dis_to_tree_S, hight_N, hight_S = distance_hight_of_obj(pointcloud, tree_location, degree, crown_start)
        df.loc[i, f'disAdjacentTreeN+{degree+90}d(m)'] = dis_to_tree_N
        df.loc[i, f'disAdjacentTreeS+{degree+90}d(m)'] = dis_to_tree_S
        df.loc[i, f'hightAdjacentTreeN+{degree+90}d(m)'] = hight_N
        df.loc[i, f'hightAdjacentTreeS+{degree+90}d(m)'] = hight_S

    for degree in Degrees_EW:
        dis_to_building_W, dis_to_building_E, hight_W, hight_E = distance_hight_of_obj(building, tree_location, degree, crown_start)
        df.loc[i, f'disAdjacentBuildingW+{degree}d(m)'] = dis_to_building_W
        df.loc[i, f'disAdjacentBuildingE+{degree}d(m)'] = dis_to_building_E
        df.loc[i, f'hightAdjacentBuildingW+{degree}d(m)'] = hight_W
        df.loc[i, f'hightAdjacentBuildingE+{degree}d(m)'] = hight_E

    for degree in Degrees_NS:
        dis_to_building_N, dis_to_building_S, hight_N, hight_S = distance_hight_of_obj(building, tree_location, degree, crown_start)
        df.loc[i, f'disAdjacentBuildingN+{degree}d(m)'] = dis_to_building_N
        df.loc[i, f'disAdjacentBuildingS+{degree}d(m)'] = dis_to_building_S
        df.loc[i, f'hightAdjacentBuildingN+{degree}d(m)'] = hight_N
        df.loc[i, f'hightAdjacentBuildingS+{degree}d(m)'] = hight_S    

    end = time.time()    
    print ('****Tree '+ dataset.loc[i,'treeID'] +' is finished****')    
    print(f'{round((end - start)/60, 2)} minutes calculation time')

    return df



if __name__ == "__main__":
    pool = multiprocessing.Pool(2)
    start_time = time.perf_counter()
    result = pool.map(dis, range(dataset.shape[0]))
    finish_time = time.perf_counter()
    print(f"Program finished in {(finish_time-start_time)/60} minutes")
    #print(result)

    df = pd.concat(result)
    df_final = pd.concat([dataset, df], axis=1)
    # save the final dataset
    df_final.to_csv(pathtosave+project_date+'_'+project_name+'_dataset.csv',index=False)
