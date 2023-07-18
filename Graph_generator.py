import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

####################Importing the tree cylender dataset######################
save_path = 'J:/pointclouds/Dataset_Graph'
data_list = os.listdir('J:/pointclouds/Dataset_tree/')

for project_date in data_list:
    project_names = os.listdir('J:/pointclouds/Dataset_QSM/'+project_date)

    for projectname in project_names:
        path = 'J:/pointclouds/Dataset_QSM/'+project_date+'/'+projectname

        files_list = os.listdir(path+'/optcsv')

        os.makedirs(save_path+'/'+projectname)

        for f in files_list:
            df = pd.read_csv(path+'/optcsv/'+f)
            df = df.drop(['date','projectID' , 'treeID', 'addedVirtual'], axis=1)

            ####################making the graph model######################
            G=nx.from_pandas_edgelist(df, 'cylinderID', 'parentCylID', edge_attr=True)
            df = df.drop(['childCyID', 'parentCylID'], axis = 1)

            # Iterate over df rows and set source node attributes:
            for index, row in df.iterrows():
                src_attr_dict = {k: row.to_dict()[k] for k in df.columns}    
                G.nodes[row['cylinderID']].update(src_attr_dict)
            G.remove_node(0)

            ####################exporting the graph model######################
            # save graph object to pickle file

            pickle.dump(G, open(save_path+'/'+projectname+'/'+f.split(".")[0]+'.pickle', 'wb'))

            ####################loading the graph model######################
            # load graph object from pickle file

            # G = pickle.load(open('2023-01-09_tum_campus_000003.pickle', 'rb'))
            ####################visualization######################

            # plt.figure(figsize=(10, 20))
            # coordinates = np.column_stack((df['start_x'], df['start_z']))
            # positions = dict(zip(G.nodes, coordinates))

            # subjects = list(df[df['cylinderID'].isin(list(G.nodes))]["branchOrder"])
            # nx.draw(G, positions, node_size=15, node_color=subjects, edge_color='0.5')
            # plt.show()

        print (f'-----------------------project {projectname}------------------------')
