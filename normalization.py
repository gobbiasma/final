
from sklearn import preprocessing
import numpy as np
import argparse

import pandas as pd
import os

def Normalization(path_data, out):

    #df = pd.read_csv(os.path.join(path_target,'target.csv'))
    housing = pd.read_csv(os.path.join(path_data,"data.csv"))
    df = housing.drop(['0','1','2','3','4','5','6','7'], axis=1)
    x_array = housing.drop(['index','img','target'], axis=1)
    
    #column_names = ['index','img','target']
    #df = pd.read_csv(os.path.join(path_data,"data.csv"), names=column_names)
    
    scaler = preprocessing.MinMaxScaler()
    names = x_array.columns
    d = scaler.fit_transform(x_array)
    #d = preprocessing.normalize(x_array)
    print("Normalization Done")

    scaled_df = pd.DataFrame(d, columns=names)
    final_df = pd.concat([df,scaled_df],axis=1)

    final_df.to_csv(os.path.join(out,'data_normaliz_2class.csv'),index=False)
    print("Save data_normaliz.csv Sucssefuly")

def get_args():

    parser = argparse.ArgumentParser(description = "U-net Lung Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path_data',type=str,default='E:/2 MASTER/Memoire/07-06-2021 (croped)')
    parser.add_argument('--path_target', type=str, default='E:/2 MASTER/Memoire/07-06-2021 (croped)/covid_croped/dataset')
    parser.add_argument('--out', type=str, default='./')
    
    return parser.parse_args()

def main():
    args = get_args()
    Normalization(args.path_data, args.out)

if __name__ == '__main__':

    main()
