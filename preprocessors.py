import numpy as np
import pandas as pd
from collections import Counter

from sklearn.cluster import DBSCAN

class Pipeline:
    
    """
    Pipeline to extract likely home and work location from user's
    location history. It should be initialized with the dataset
    plus the different group of variables to which we wish to apply
    the different engineering procedures.
    """
    
    def __init__(self, variables, regional_eps, regional_min_samples, 
                 regional_metric, local_eps, local_min_samples,
                 local_metric):
        
        # User location data
        self.data = None
        self.DBSCAN_data_regional = None
        self.DBSCAN_data_local = {}

        # Parameter for building DBSCAN model
        self.variables = variables
    
        # Parameters for building the regional DBSCAN model
        self.regional_eps = regional_eps
        self.regional_min_samples = regional_min_samples
        self.regional_metric = regional_metric

        # Parameters for building the local DBSCAN model
        self.local_eps = local_eps
        self.local_min_samples = local_min_samples
        self.local_metric = local_metric

        # Regional model
        self.model_regional = None

        # Local models dictionary
        self.model_local_dict = {}

        # Initiate Resulting filtered home and work locations
        self.home_list_filtered = []
        self.work_list_filtered = []

    def add_time(self, df):

        """
        Method to add Hour column to the dataset
        """

        df = df.copy()
        ts_objs = np.array([pd.Timestamp(item) for item in np.array(df['start_time(YYYYMMddHHmmZ)'])])
        df['TS_obj'] = ts_objs
        df['Hour'] = df['TS_obj'].apply(lambda d: d.hour)

        return df


    def prepare_dbscan_data(self, df, variables = None):
        
        """
        Method to extract longitude and latidute from
        the dataframe and return it as a numpy array
        for later processing by the DBSCAN algorithm
        """

        df = df.copy()
        df = df[variables]
        DBSCAN_data = df.values.astype('float32', copy=False)
        
        return DBSCAN_data


    def regional_cluster_dict(self, df):
        
        """
        Method to build a dictionary of regional clusters from
        the regional DBSCAN model and the prepared DBSCAN data
        """
        
        self.regional_dict = {}

        reg_num = len(Counter(self.model_regional.labels_)) - 1
        self.regional_dict = {n: df[self.model_regional.labels_ == n] for n in range(reg_num)}
        
        return self

    def local_cluster_dict(self):

        """
        Method to build a dictionary of local clusters from
        the local DBSCAN models dictionary and regional clusters
        dictionary
        """

        self.local_dict = {}

        for k in self.regional_dict:
            loc_num = len(Counter(self.model_local_dict[k].labels_)) - 1
            self.local_dict[k] = {n: self.regional_dict[k][self.model_local_dict[k].labels_ == n] for n in range(loc_num)}

        return self

    def construct_model_regional(self, df, variables = None):

        """
        Method to construct a regional DBSCAN model from data and regional DBSCAN parameters
        """

        data_regional = self.prepare_dbscan_data(df, variables)

        self.model_regional = DBSCAN(eps = self.regional_eps, min_samples=self.regional_min_samples, metric=self.regional_metric).fit(data_regional)

        return self

    def construct_model_local(self, data):

        """
        Method to construct local DBSCAN model from local dbscan data 
        and local DBSCAN parameters 
        """

        model = DBSCAN(eps=self.local_eps, min_samples=self.local_min_samples, metric=self.local_metric).fit(data)

        return model

    def construct_models_local_dict(self):

        """
        Method to build local models dictionary from regional dictionary 
        """

        self.DBSCAN_data_local = {k: self.prepare_dbscan_data(v, self.variables) for k,v in self.regional_dict.items()}

        self.model_local_dict = {k: self.construct_model_local(v) for k,v in self.DBSCAN_data_local.items()}

        return self

    def filter_regional_clusters(self, data):

        """
        Method to filter local clusters dictionary leaving
        only those with a parent regional cluster of more 
        than 20% of the whole dataset
        """
        self.filtered_regional_dict = {}

        for k in self.regional_dict:
            if self.regional_dict[k].shape[0]/data.shape[0] > 0.2:
                self.filtered_regional_dict[k] = self.local_dict[k]

        return self

    def filter_local_clusters(self):

        """
        Method to further filter the filtered regional clusters
        leaving only the local clusters with more than 20% data
        from data in parent regional cluster
        """

        self.filtered_local_dict = {}
        self.local_cluster_time = {}
        self.local_cluster_latlong = {}
        self.local_cluster_ratio = {}

        for r, r_v in self.filtered_regional_dict.items():

            self.filtered_local_dict[r] = {}
            self.local_cluster_time[r] = {}
            self.local_cluster_latlong[r] = {}
            self.local_cluster_ratio[r] = {}

            for l, l_v in r_v.items():
                if l_v.shape[0]/self.regional_dict[r].shape[0] > 0.2:

                    self.filtered_local_dict[r][l] = l_v
                    self.local_cluster_time[r][l] = [l_v["Hour"].mean()]
                       
                    if (11 < self.local_cluster_time[r][l][0] < 15):
                        self.local_cluster_time[r][l].append('work')
                    else:
                        self.local_cluster_time[r][l].append('home')

                    self.local_cluster_latlong[r][l] = [l_v['latitude'].mean(), l_v['longitude'].mean()]
                    self.local_cluster_ratio[r][l] = l_v.shape[0]/self.regional_dict[r].shape[0]

        return self

    def likely_loc_time(self):

        """
        Method that returns the likely time of day
        along with the likely home and work locations
        for each user (if exists)
        """

        loc_list = []
        home_list = []
        work_list = []

        for o, o_v in self.local_cluster_time.items():
            for i, i_v in o_v.items():

                loc_list.append(i_v[1])

                if i_v[1] == 'home':
                    home_list.append([o, i, self.local_cluster_latlong[o][i], self.local_cluster_ratio[o][i]])
                else:
                    work_list.append([o, i, self.local_cluster_latlong[o][i], self.local_cluster_ratio[o][i]])

        occ_home = loc_list.count('home')
        occ_work = loc_list.count('work')

        if occ_home == 0:
            self.home_list_filtered = ['no home locations detected']
        
        if occ_home == 1:
            self.home_list_filtered = home_list

        if occ_home > 1:
             
            for o in self.local_cluster_time:
                an_iterator = filter(lambda home: home[0] == o, home_list)
                each_region = list(an_iterator)

                regional_list = list(map(lambda x: x[3], each_region))
                max_value = max(regional_list)
                max_index = regional_list.index(max_value)

                self.home_list_filtered.append(regional_list[max_index])

        if occ_work == 0:
            self.work_list_filtered = ['no work locations detected']
        
        if occ_work > 0:
            self.work_list_filtered = work_list

        return self

    
    def fit(self, data):
        
        data = data.copy()
        data = self.add_time(data)
        self.construct_model_regional(data, variables = self.variables)
        self.regional_cluster_dict(data)
        self.construct_models_local_dict()
        self.local_cluster_dict()
        self.filter_regional_clusters(data)
        self.filter_local_clusters()
        self.likely_loc_time()

        return self

    def predict(self):

        return self.home_list_filtered, self.work_list_filtered
        

    
