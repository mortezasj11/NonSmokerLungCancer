
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.cluster.hierarchy as sch
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from lifelines import CoxTimeVaryingFitter
from sklearn.model_selection import GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
from gap_statistic import OptimalK


class Non_Smoker_Analysis():
    def __init__(self, z_score=True, print_info=False, path = "/Data1/non_smoker_May/pyradiomics_sybil_demog_CV.csv"):
        self.path = path
        self.n_radiomic = 113
        self.non_features_col = ['MRN', 'date', 'Date_To_Cancer']
        self.all_long_features = ['1_year_risk','2_year_risk','3_year_risk','4_year_risk','5_year_risk','6_year_risk']
        self.pre_demographic_features = ['Sex', 'Race', 'Tobacco Use', 'age']
        self.n_longitudinal = 0
        self.df , self.mrns = self.load_data_fix_age(print_info)
        self.mrn_train, self.mrn_test = self.give_train_test_mrns(print_info=print_info)
        if z_score:
            self.df = self.z_score_(print_info)
        self.df_Train, self.df_Test = self.define_df_train_test()
        #self.pre_demographic_features = self.select_demographic_features()  #### may need to be changes


    def load_data_fix_age(self, MRN_to_remove_=True, print_info=False):
        df = pd.read_csv(self.path, usecols=lambda column: column not in ['Unnamed: 0', 'Unnamed: 0.1'])
        for mrn in list(set(df["MRN"].tolist())):
            age_at_dx = df[df["MRN"]==mrn]["age at dx"].iloc[0]
            dates_to_cancer = list(df[df["MRN"]==mrn]["Date_To_Cancer"])
            for date in dates_to_cancer:
                df.loc[  (df["MRN"]==mrn) & (df["Date_To_Cancer"]==date)  , 'age'  ] = age_at_dx - date
        df = df.drop(["age at dx"], axis=1)
        df = df.drop_duplicates()
        if MRN_to_remove_:
            MRN_to_remove = ["915517", "956379", "297791", "616175", "2237068", "842261", "743544", "420022", \
                            "2429727", "2487906", "2572239", "2598366" , "2599440", "838996", \
                            "2338015B"]
            df = df[~df['MRN'].isin(MRN_to_remove)]
        mrns = list(set(df["MRN"].tolist()))
        if print_info:
            print("df.shape: " ,df.shape)
            print( "len(mrn):", len(mrns)  )
        return df , mrns


    def looking_at_data(self):
        mrns =  list(set(self.df["MRN"].tolist()))
        list_count = []
        D_count = {}
        for i in mrns:
            n_dates = self.df[self.df["MRN"]==i].shape[0]
            list_count.append(n_dates)
            if n_dates in D_count:
                D_count[n_dates].append(i)
            else:
                D_count[n_dates] = [i]
        plt.figure(figsize=(8, 4))    
        bin_edges = np.arange(min(list_count), max(list_count) + 2) - 0.5
        plt.hist(list_count, bins=bin_edges, edgecolor='black')  
        plt.xlabel("Number of longitudinal data", fontsize=12)
        plt.ylabel("Number of patients ", fontsize=12)
        plt.show()


    def give_train_test_mrns(self, same_result=False, folds=5, select_fold=0, print_info=True) :
        df = self.df
        mrns =  list(set(df["MRN"].tolist()))
        random_seed = np.random.randint(1, 1000)
        if same_result:
            random_seed = 42
        num_folds = folds
        #mrns.sort() #not necessary
        X = mrns
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed) ## Create KFold object
        train_sets = []
        test_sets = []
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train = [X[i] for i in train_index]
            X_test = [X[i] for i in test_index]
            train_sets.append(X_train)
            test_sets.append(X_test)
            #print(f"Fold {fold + 1}: Train size: {len(train_index)}, Test size: {len(test_index)}")
        # selecting first fold by default!
        self.mrn_train = train_sets[select_fold]
        self.mrn_test = test_sets[select_fold]
        if print_info:
            print('train patients:', len(self.mrn_train))
            print('test patients:',len(self.mrn_test))
            print()
            df_train__ = df[df['MRN'].isin(self.mrn_train )].copy()
            print('train CTs:',df_train__.shape[0])
            df_test__ = df[df['MRN'].isin(self.mrn_test)].copy()
            print('test CTs:',df_test__.shape[0])
        return self.mrn_train, self.mrn_test


    def z_score_(self, print_info=False):
        df = self.df
        df_train__ = self.df[self.df['MRN'].isin(self.mrn_train)].copy()
        df_test__ = self.df[self.df['MRN'].isin(self.mrn_test)].copy()
        start_idx = 3
        end_idx = -10
        columns_to_normalize = list(df.columns[start_idx:end_idx])
        mean_train = df_train__[columns_to_normalize].mean()
        std_train = df_train__[columns_to_normalize].std()
        std_train_nonzero = std_train.replace(0, 1e-5)
        df_train__[columns_to_normalize] = df_train__[columns_to_normalize].apply(zscore) # Apply z-score normalization to the selected columns in df_train
        df_test__[columns_to_normalize] = (df_test__[columns_to_normalize] - mean_train) / std_train_nonzero
        ## update the original DataFrame (df) with the normalized values for both train and test sets:
        self.df.loc[self.df['MRN'].isin(self.mrn_train), columns_to_normalize] = df_train__[columns_to_normalize]
        self.df.loc[self.df['MRN'].isin(self.mrn_test), columns_to_normalize] = df_test__[columns_to_normalize]
        if print_info:
            print("df_train: ",df_train__.shape)
            print("df_test: ", df_test__.shape)
            print('Edges columns that did not Z-Scored:    (1)',df.columns[start_idx-1], ", (2)", df.columns[end_idx])
        return self.df


    def check_mean_std_of_test(self, start_idx=3, end_idx=-10):
        df = self.df
        df_test__ = self.df[self.df['MRN'].isin(self.mrn_test)].copy()
        columns_to_normalize = list(df.columns[start_idx:end_idx])
        mean_test = df_test__[columns_to_normalize].mean()
        std_test = df_test__[columns_to_normalize].std()
        print("mean_of_all_means:" , mean_test.mean())
        print("mean_of_all_stds:" , std_test.mean())


    def define_df_train_test(self):
        df = self.df
        self.df_Train = df[df["MRN"].isin(self.mrn_train)]
        self.df_Test = df[df["MRN"].isin(self.mrn_test)]
        return self.df_Train, self.df_Test # just to see them in init


    def select_radiomic_features(self, method="ward", metric='euclidean', print_info=False, plot=False, avoid_zero = 0.2):
        # method = "ward"  #'ward'   # 'single'  'centroid'   'weighted'   'average'   'complete'   'median'
        # metric ='euclidean' #'cosine'#'manhattan'# 'euclidean'
        df = self.df
        sel_features = [item for item in df.columns.tolist() if item not in (self.all_long_features + self.pre_demographic_features + self.non_features_col)]
        #sel_features = df.columns[lower_limit:-10] #lower_limit = len(self.non_features_col)
        df_Fi_ = df[df["MRN"].isin(self.mrn_train)]
        df_Fi =  (df_Fi_.loc[:, sel_features]).T
        optimalK = OptimalK(parallel_backend='multiprocessing')
        n_clusters = optimalK(df_Fi, cluster_array=np.arange(2, 11))
        if plot:
            optimalK.plot_results()
            plt.show()
        max_gap_star = optimalK.gap_df['gap*'].idxmax()
        max_gap = optimalK.gap_df['gap_value'].idxmax()
        n_cluster_gap_star = optimalK.gap_df.at[max_gap_star, 'n_clusters']
        n_cluster_gap = optimalK.gap_df.at[max_gap, 'n_clusters']
        n_clusters = int(min((n_cluster_gap_star, n_cluster_gap)))
        if print_info:
            print('n_clusters: ', n_clusters, "   ( gap* & gap: ", n_cluster_gap_star,', ', n_cluster_gap, ' )')
        ###########################################################################################################
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=method)
        model.fit(df_Fi)
        labels = model.labels_
        dtype = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
        radiomic_features = []
        D_all = {}
        for i in range(n_clusters):
            D_i_temp = {}
            cluster_i_features = [ii for ii,v in dict(zip(df_Fi.T.columns, labels)).items() if v==i]# using df_Fi
            for cluster_i_feature in cluster_i_features:
                X_train_temp = df_Fi_[cluster_i_feature].values # using df_Fi_ NOT df_Fi   #(487,)
                X_train_temp = X_train_temp.reshape((X_train_temp.shape[0],1))   #(487, 1)
                y_train_temp = np.array([(True, x + avoid_zero) for x in df_Fi_["Date_To_Cancer"].tolist() ], dtype=dtype)
                coxphSA = CoxPHSurvivalAnalysis()
                coxphSA.fit(X_train_temp, y_train_temp)
                D_i_temp[cluster_i_feature] = coxphSA.score(X_train_temp, y_train_temp)
                D_all[cluster_i_feature] = coxphSA.score(X_train_temp, y_train_temp)
                # We can add X_test here too and check the result
                #print( cluster_i_feature, coxphSA.score(X_train_temp, y_train_temp) )
            radiomic_features.append(max(D_i_temp, key=D_i_temp.get)) # find the max value of each cluster
        if plot:
            df_Fi_label = pd.DataFrame.copy(df_Fi)
            df_Fi_label['color'] = labels
            my_palette = dict(zip(df_Fi_label["color"].unique(), ["blue","orange","purple","cyan","slategray","tan","peru","crimson","yellow","pink","green","brown"]))
            row_colors = df_Fi_label["color"].map(my_palette)
            cmap='bwr' #'seismic' 'vlag'
            plt.figure(figsize=(16,3),dpi=600)
            cm = sns.clustermap(df_Fi_label.iloc[:,:-1], metric='euclidean',method=method,cmap=cmap, figsize=(100, 100), row_colors=row_colors,linewidths=1.5)#, vmin=-3, vmax=3 col_cluster=False,
            plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=100)
            plt.show()
        if print_info:
            for i in radiomic_features:
                print(i,D_all[i] )
        self.radiomic_features = radiomic_features
        return radiomic_features


    def select_lung_feature(self, print_info=False, avoid_zero = 0.2):
        long_features = ['1_year_risk', '2_year_risk', '3_year_risk', '4_year_risk', '5_year_risk', '6_year_risk']
        dtype = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
        D_i_temp = {}
        for long_feature in long_features:
            X_train_temp = self.df_Train[long_feature].values
            X_train_temp = X_train_temp.reshape((X_train_temp.shape[0],1))
            y_train_temp = np.array([(True, x + avoid_zero) for x in self.df_Train["Date_To_Cancer"].tolist() ], dtype=dtype)
            coxphSA = CoxPHSurvivalAnalysis()
            coxphSA.fit(X_train_temp, y_train_temp)
            D_i_temp[long_feature] = coxphSA.score(X_train_temp, y_train_temp)
        if print_info:
            print(D_i_temp)
        self.lung_feature = ['1_year_risk']
        return self.lung_feature


    def select_demographic_features(self):  #######################################Maybe not needed
        self.demographic_features = ['Sex', 'Race', 'Tobacco Use', 'age']
        return self.demographic_features


    def find_df_logitudinal_(self, avoid_zero=0.2):
        df = self.df
        dtype = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
        # X_train, y_train
        X_train = self.df_Train
        y_train = np.array([(True, x + avoid_zero) for x in self.df_Train["Date_To_Cancer"].tolist() ], dtype=dtype)
        # X_test, y_test
        X_test = self.df_Test
        y_test = np.array([(True, x + avoid_zero) for x in self.df_Test["Date_To_Cancer"].tolist() ], dtype=dtype)
        ########################################################################
        selected_features = ["MRN","date","Date_To_Cancer",'age' ] + X_train.columns.tolist()[len(self.non_features_col):-len(self.pre_demographic_features)]  #116
        l_df = 121
        df_Train_variational = pd.DataFrame(columns=['id', "start", "stop", "event"] + selected_features)
        i = 0
        avoid_zero = 0.2 #year
        id_ = 0
        for mrn in self.mrn_train: 
            df_temp =  df[df["MRN"]==mrn].copy()
            DTCs_tempp = df_temp["Date_To_Cancer"].to_numpy()
            sorted_indices = np.argsort(DTCs_tempp)
            DTCs = DTCs_tempp[sorted_indices]
            DTCs_ = np.insert(DTCs + 0.1, 0, 0)
            windows = (max(DTCs_) - DTCs_)[::-1]
            DTCs_rev = DTCs[::-1]
            id_ += 1
            for idx, DTC in enumerate(DTCs_rev):  #print(DTC, windows)
                df_t = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTC)]
                t1_ = windows[idx]
                t2_ = windows[idx + 1]
                enent = True if idx==len(DTCs_rev)-1 else False
                nested_l_t = df_t[selected_features].values.tolist()
                L_t = [i for s in nested_l_t for i in s]
                df_Train_variational = df_Train_variational.append([pd.Series([id_, t1_, t2_,enent]+L_t, index = df_Train_variational.columns[0:l_df])], ignore_index=True)
        # MRN_TEST
        df_Test_variational = pd.DataFrame(columns=['id', "start", "stop", "event"] + selected_features)
        i = 0
        avoid_zero = 0.2 #year
        id_ = 0
        for mrn in self.mrn_test: #mrn_test_Fi[1:]
            df_temp =  df[df["MRN"]==mrn].copy()
            DTCs_tempp = df_temp["Date_To_Cancer"].to_numpy()
            sorted_indices = np.argsort(DTCs_tempp)
            DTCs = DTCs_tempp[sorted_indices]
            #DTCs = df_temp["Date_To_Cancer"].to_numpy()
            #DTCs.sort()# WRONGGGGGGGGGGGGGGGGGGGG
            DTCs_ = np.insert(DTCs + 0.1, 0, 0)
            windows = (max(DTCs_) - DTCs_)[::-1]
            DTCs_rev = DTCs[::-1]
            id_ += 1
            for idx, DTC in enumerate(DTCs_rev):  #print(DTC, windows)
                df_t = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTC)]
                t1_ = windows[idx]
                t2_ = windows[idx + 1]
                enent = True if idx==len(DTCs_rev)-1 else False
                nested_l_t = df_t[selected_features].values.tolist()
                L_t = [i for s in nested_l_t for i in s]
                df_Test_variational = df_Test_variational.append([pd.Series([id_, t1_, t2_,enent]+L_t, index = df_Test_variational.columns[0:l_df])], ignore_index=True)
        #self.df_logitud_train_ = df_Train_variational
        #self.df_logitud_test_ = df_Test_variational
        return df_Train_variational, df_Test_variational


    def selected_longitudinal_features(self, n_longitudinal=4, avoid_zero=0.2, summary=False):
        df_logitud_train_, df_logitud_test_ = self.find_df_logitudinal_()
        df = self.df.copy()
        # Exclude some columns
        column_to_exclude = ['Date_To_Cancer','date','MRN']
        df_tr_var = df_logitud_train_.drop(column_to_exclude, axis=1).copy()
        df_ts_var = df_logitud_test_.drop(column_to_exclude, axis=1).copy()
        # Make sure the df is fine
        df_tr_var = df_tr_var.dropna()
        df_ts_var = df_ts_var.dropna()
        D_feature_haz = {}
        for feature in  df_tr_var.columns[4:].tolist():
            col_sel = ["id","start","stop","event"]
            col_sel.append(feature)
            df_tr = df_logitud_train_.drop(column_to_exclude, axis=1).copy()
            df_ts = df_logitud_test_.drop(column_to_exclude, axis=1).copy()
            df_tr = df_tr[col_sel]
            df_ts = df_ts[col_sel]
            ctv = CoxTimeVaryingFitter(penalizer=0.05)
            ctv.fit(df_tr, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=False)
            D_feature_haz[feature] = ctv.hazard_ratios_.item()    # NOT SURE NOW
            if summary:
                ctv.print_summary(decimals=5)
                ctv.plot()
                plt.show()
        sorted_dict_high_r = dict(sorted(D_feature_haz.items(), key=lambda item: item[1], reverse=True))
        sorted_dict_high = dict(sorted(D_feature_haz.items(), key=lambda item: item[1], reverse=False))
        n_long_half = n_longitudinal//2
        features_to_hazard = list(sorted_dict_high_r.keys())[:n_long_half] + list(sorted_dict_high.keys())[:n_long_half]
        for feature_to_hazard in features_to_hazard:
            col_sel = ["id","start","stop","event"]
            col_sel.append(feature_to_hazard)
            df_tr = df_logitud_train_.drop(column_to_exclude, axis=1).copy()
            df_ts = df_logitud_test_.drop(column_to_exclude, axis=1).copy()
            # df_tr = df_tr.dropna()
            # df_ts = df_ts.dropna()
            df_tr = df_tr[col_sel]
            df_ts = df_ts[col_sel]
            ctv = CoxTimeVaryingFitter(penalizer=0.05)
            ctv.fit(df_tr, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=False)
            D_feature_haz[feature_to_hazard] = ctv.hazard_ratios_.item()    # NOT SURE NOW
            if summary:
                ctv.print_summary(decimals=5)
                ctv.plot()
                plt.show()
            ############### Adding the log Hazrard based on the feature_to_hazard  ########################
            # TRAIN
            for index, row in df_logitud_train_.iterrows():
                df_tr_temp = df_logitud_train_.iloc[[index]]
                mrn_ = df_tr_temp["MRN"].item()
                DTC_ = df_tr_temp["Date_To_Cancer"].item()
                df_tr_temp = df_tr_temp[col_sel]
                log_hazard_ = ctv.predict_log_partial_hazard(df_tr_temp).item()
                df.loc[(df["MRN"]==mrn_) & (df["Date_To_Cancer"]==DTC_), "LogHazard__"+feature_to_hazard] = log_hazard_
            # TEST
            for index, row in df_logitud_test_.iterrows():
                df_ts_temp = df_logitud_test_.iloc[[index]]
                mrn = df_ts_temp["MRN"].item()
                DTC = df_ts_temp["Date_To_Cancer"].item()
                df_ts_temp = df_ts_temp[col_sel]
                log_hazard_ = ctv.predict_log_partial_hazard(df_ts_temp).item()
                df.loc[(df["MRN"]==mrn) & (df["Date_To_Cancer"]==DTC), "LogHazard__"+feature_to_hazard] = log_hazard_
        self.df_orig = self.df
        self.df = df
        self.n_longitudinal = n_longitudinal
        self.longitudinal_features = ["LogHazard__"+i for i in features_to_hazard]
        return self.longitudinal_features


    def select_diff_features(self, method = "ward", metric ='euclidean',  plot = False, print_info = False):
        df = self.df
        upper_limit = len(self.pre_demographic_features) + self.n_longitudinal
        lower_limit = len(self.non_features_col)
        r_lu_features = df.columns[lower_limit:-upper_limit].tolist()
        d_lo_features = df.columns[-upper_limit:].tolist()
        selected_features = ['MRN', 'date', 'Date_To_Cancer'] + r_lu_features + d_lo_features  #3 + 113 + 8
        l_df = self.n_radiomic + self.n_radiomic + lower_limit + upper_limit #237
        if print_info:
            print("upper_limit: ", upper_limit)
        diff_col_name = ['d_'+i for i in r_lu_features] #113
        # MRN_TRAIN
        df_Train_diff = pd.DataFrame(columns = selected_features + diff_col_name)  #237
        for mrn in self.mrn_train: #mrn_train_Fi[1:]
            df_temp =  df[df["MRN"]==mrn].copy()
            DTCs_tempp = df_temp["Date_To_Cancer"].to_numpy()
            sorted_indices = np.argsort(DTCs_tempp)
            DTCs = DTCs_tempp[sorted_indices]                     # [0.  , 0.54, 0.84, 1.09, 1.86, 2.12]            
            diffs = np.array(list(-(DTCs[:-1] - DTCs[1:])) + [1e-6]) # [0.54, 0.29, 0.25, 0.77, 0.26, 0   ]
            #DTCs_rev = DTCs[::-1]
            for idx in range(len(DTCs)):
                df_t_d_lo = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx]  )][d_lo_features]
                df_t   = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx]  )][r_lu_features]
                if idx != len(DTCs)-1:
                    df_t_1 = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx+1])][r_lu_features]
                elif idx == len(DTCs)-1:
                    df_t_1 = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx])][r_lu_features]
                delta_t = diffs[idx]
                df_t = df_t.reset_index(drop=True)
                df_t_1 = df_t_1.reset_index(drop=True)
                #diff_df = df_t.sub(df_t_1, fill_value=0).iloc[0].tolist()
                diff_df = (df_t.sub(df_t_1, fill_value=0).iloc[0] / delta_t).tolist()
                orig_col_values = df_t.iloc[0].tolist() # n=113
                orig_d_lo_values = df_t_d_lo.iloc[0].tolist() # n=8
                date = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx]  )]['date'].item() 
                df_Train_diff = df_Train_diff.append([pd.Series([mrn, date, DTCs[idx]]+orig_col_values+orig_d_lo_values+diff_df, index = df_Train_diff.columns[0:l_df])], ignore_index=True)
        # MRN_TRAIN
        df_Test_diff = pd.DataFrame(columns = selected_features + diff_col_name)  #229
        for mrn in self.mrn_test: #mrn_train_Fi[1:]
            df_temp =  df[df["MRN"]==mrn].copy()
            DTCs_tempp = df_temp["Date_To_Cancer"].to_numpy()
            sorted_indices = np.argsort(DTCs_tempp)
            DTCs = DTCs_tempp[sorted_indices]                     # [0.  , 0.54, 0.84, 1.09, 1.86, 2.12]            
            diffs = np.array(list(-(DTCs[:-1] - DTCs[1:])) + [1e-6]) # [0.54, 0.29, 0.25, 0.77, 0.26, 0   ]
            #DTCs_rev = DTCs[::-1]
            for idx in range(len(DTCs)):
                df_t_d_lo = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx]  )][d_lo_features]
                df_t   = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx]  )][r_lu_features]
                if idx != len(DTCs)-1:
                    df_t_1 = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx+1])][r_lu_features]
                elif idx == len(DTCs)-1:
                    df_t_1 = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx])][r_lu_features]
                delta_t = diffs[idx]
                df_t = df_t.reset_index(drop=True)
                df_t_1 = df_t_1.reset_index(drop=True)
                #diff_df = df_t.sub(df_t_1, fill_value=0).iloc[0].tolist()
                diff_df = (df_t.sub(df_t_1, fill_value=0).iloc[0] / delta_t).tolist()
                orig_col_values = df_t.iloc[0].tolist() # n=113
                orig_d_lo_values = df_t_d_lo.iloc[0].tolist() # n=8
                date = df_temp[(df_temp["MRN"]==mrn) & (df_temp["Date_To_Cancer"]==DTCs[idx]  )]['date'].item() 
                df_Test_diff = df_Test_diff.append([pd.Series([mrn, date, DTCs[idx]]+orig_col_values+orig_d_lo_values+diff_df, index = df_Test_diff.columns[0:l_df])], ignore_index=True)
        ################################################################################################################################################################################
        # Defining the number of clusters
        sel_features = diff_col_name
        df_Fi_ = df_Train_diff
        df_Fi =  (df_Fi_.loc[:, sel_features]).T
        optimalK = OptimalK(parallel_backend='multiprocessing')
        n_clusters = optimalK(df_Fi, cluster_array=np.arange(2, 11))
        if plot:
            optimalK.plot_results()
            plt.show()
        max_gap_star = optimalK.gap_df['gap*'].idxmax()
        max_gap = optimalK.gap_df['gap_value'].idxmax()
        n_cluster_gap_star = optimalK.gap_df.at[max_gap_star, 'n_clusters']
        n_cluster_gap = optimalK.gap_df.at[max_gap, 'n_clusters']
        if print_info:
            print(n_cluster_gap_star, n_cluster_gap)
        n_clusters_d_ = int(min((n_cluster_gap_star, n_cluster_gap)))
        ################################################################################################################################################################################
        sel_features = diff_col_name # not considering the SYBIL features which are the last 6
        df_Fi_ = df_Train_diff
        df_Fi =  (df_Fi_.loc[:, sel_features]).T
        model = AgglomerativeClustering(n_clusters=n_clusters_d_, affinity=metric, linkage=method)
        model.fit(df_Fi)
        labels = model.labels_
        df_Fi_label = pd.DataFrame.copy(df_Fi)
        df_Fi_label['color'] = labels
        my_palette = dict(zip(df_Fi_label["color"].unique(), ["blue","orange","purple","cyan","slategray","tan","peru","crimson","yellow","pink","green","brown"]))
        row_colors = df_Fi_label["color"].map(my_palette)
        cmap='bwr' #'seismic' 'vlag'
        if plot:
            plt.figure(figsize=(16,3),dpi=600)
            cm = sns.clustermap(df_Fi_label.iloc[:,:-1], metric='euclidean',method=method,cmap=cmap, figsize=(100, 100), row_colors=row_colors,linewidths=1.5)#, vmin=-3, vmax=3 col_cluster=False,
            plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=100)
            plt.show()
        # select the feature that provides the highest coxph in each cluster
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        D_all = {}
        avoid_zero = 0.2
        dtype = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
        d_radiomic_features = []
        for i in range(n_clusters_d_):
            D_i_temp = {}
            cluster_i_features = [ii for ii,v in dict(zip(df_Fi.T.columns, labels)).items() if v==i]# using df_Fi
            for cluster_i_feature in cluster_i_features:
                X_train_temp = df_Fi_[cluster_i_feature].values # using df_Fi_ NOT df_Fi   #(487,)
                X_train_temp = X_train_temp.reshape((X_train_temp.shape[0],1))   #(487, 1)
                y_train_temp = np.array([(True, x + avoid_zero) for x in df_Fi_["Date_To_Cancer"].tolist() ], dtype=dtype)
                coxphSA = CoxPHSurvivalAnalysis()
                coxphSA.fit(X_train_temp, y_train_temp)
                D_i_temp[cluster_i_feature] = coxphSA.score(X_train_temp, y_train_temp)
                D_all[cluster_i_feature] = coxphSA.score(X_train_temp, y_train_temp)
            d_radiomic_features.append(max(D_i_temp, key=D_i_temp.get)) # find the max value of each cluster
        n_diff_features = n_clusters_d_
        sorted_keys = sorted(D_all, key=lambda x: D_all[x], reverse=True)
        self.diff_features = [key for key in sorted_keys if key in d_radiomic_features][:n_diff_features]
        self.df = pd.concat([df_Train_diff, df_Test_diff], ignore_index=True)
        return self.diff_features


    def get_dummies_pre_rsf(self): # should not be applied soon
        # Check if the columns have already been converted to dummies
        if not set(['Sex', 'Race', 'Tobacco Use']).issubset(self.df.columns):
            return self.demog_features
        df_temp = pd.get_dummies(self.df, columns=['Sex', 'Race', 'Tobacco Use'], drop_first=True)
        self.demog_features = df_temp.columns.difference(self.df.columns).to_list()
        self.df = pd.get_dummies(self.df, columns=['Sex', 'Race', 'Tobacco Use'], drop_first=True)
        return self.demog_features


    def define_features(self, show=True):
        # self.radiomic_features
        # self.lung_feature
        # self.demog_features
        # self.longitudinal_features
        if show:
            print("1. RADIOMIC ({}) :   {}".format(len(self.radiomic_features ),  self.radiomic_features))
            print()
            print("#"*125)
            print("2. Lung ({}) :   {}".format(len(self.lung_feature ),  self.lung_feature))
            print()
            print("#"*125)
            print("3. DEMOGRAPHIC ({}) :   {}".format(len(self.demog_features ),  self.demog_features))
            print()
            print("#"*125)
            print("4. LONGITUDINAL ({}) :   {}".format(len(self.longitudinal_features ),  self.longitudinal_features))
            print()
            print("#"*125)
            print("5. DIFFERENCE ({}) :   {}".format(len(self.diff_features ),  self.diff_features))


    def give_one_from_each_mrn(self, df):
        df_copy = df.copy()
        df_new = df_copy.groupby('MRN').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
        return df_new
    

    def random_survival_forest_gs(self, features , label=None, itter=0, avoid_zero=0.2, cv=5, n_jobs=-1, inner_distinct=10):
        label = label+'_' if label else ''
        df = self.df
        performance = {}
        perform_distinct = {}
        feature_types = features
        dtype = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
        ################################# X_train, y_train##################################
        df_train = df.loc[df['MRN'].isin(self.mrn_train)]
        X_train = df_train[feature_types]
        y_train = np.array([(True, x + avoid_zero) for x in df_train["Date_To_Cancer"].tolist() ], dtype=dtype)
        param_grid = {
            'n_estimators': [50,  200,  600],
            'min_samples_split': [3, 5, 6, 8],
            'min_samples_leaf': [2, 3, 5],
            'max_features': [None, "sqrt", "log2"]}
        rsf_r = RandomSurvivalForest(random_state=20)
        grid_search = GridSearchCV(rsf_r, param_grid, cv=cv, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        ################################## X_test, y_test ###################################
        df_test = df.loc[df['MRN'].isin(self.mrn_test)]
        X_test = df_test[feature_types] 
        y_test = np.array([(True, x + avoid_zero) for x in df_test["Date_To_Cancer"].tolist() ], dtype=dtype)
        performance['tr_' + label+str(itter)] = grid_search.score(X_train, y_train)
        performance['ts_' +label+str(itter)] = grid_search.score(X_test, y_test)
        ################################# Test on Distinct MRNs #############################
        score_distinct_tr = []
        for _ in range(inner_distinct):
            df_train_distinct = self.give_one_from_each_mrn(df_train)
            X_train_distinct = df_train_distinct[feature_types]
            y_train_distinct = np.array([(True, x + avoid_zero) for x in df_train_distinct["Date_To_Cancer"].tolist() ], dtype=dtype)
            score_distinct_tr.append(grid_search.score(X_train_distinct, y_train_distinct))
        perform_distinct['tr_' +label+str(itter)] = np.mean(score_distinct_tr)
        score_distinct_ts = []
        for _ in range(inner_distinct):
            df_test_distinct = self.give_one_from_each_mrn(df_test)
            X_test_distinct = df_test_distinct[feature_types]
            y_test_distinct = np.array([(True, x + avoid_zero) for x in df_test_distinct["Date_To_Cancer"].tolist() ], dtype=dtype)
            score_distinct_ts.append(grid_search.score(X_test_distinct, y_test_distinct))
        perform_distinct['ts_' +label+str(itter)] = np.mean(score_distinct_ts)
        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return performance,  perform_distinct




