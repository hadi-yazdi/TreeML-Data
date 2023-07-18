def point_transformation(point_project, transformation_matrix_path):
    # 1. point_project should be a list of three coordinates in project [x, y, z]
    # 2. transformation_matrix_path is the path to transformation text file
    import numpy as np
    import pyproj
    
    POP = np.loadtxt(transformation_matrix_path)
    point_project = np.array(point_project + [1])
    point_geocentric = np.matmul(POP, point_project)
    
    transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    )
    lon1, lat1, alt1 = transformer.transform(point_geocentric[0],point_geocentric[1],point_geocentric[2],radians=False)
    return [lat1, lon1, alt1]