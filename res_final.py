import os
import numpy as np
import pandas as pd
import argparse
import pickle

from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from csv import reader,writer
from tqdm import tqdm
from process_data import process_image

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path_img',type=str,default='E:/2 MASTER/Memoire/07-06-2021 (croped)/test/pneumonia') 
    # arguments for training
    parser.add_argument('--loader_model_svm', type = str , default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)/linear_svc_model.sav')
    parser.add_argument('--data_not_normaliz', type=str, default='E:/2 MASTER/Memoire/07-06-2021 (croped)/data_concat_non_normaliz.csv')
    parser.add_argument('--out', type=str, default='E:/2 MASTER/Memoire/07-12-2021 (file csv)/data csv')

    return parser.parse_args()

args = get_args()

# create path out save
if not os.path.exists(args.out):
    os.mkdir(args.out)

# create data frame vide to image input
df_image_vide = pd.DataFrame(columns = ['index','img','target','0','1','2','3','4','5','6','7'])
file_csv = pd.concat([df_image_vide])

file_csv.to_csv(os.path.join(args.out,'data.csv'),index=False)

# copy data no normalization to data frame image input
with open(args.data_not_normaliz, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Iterate over each row in the csv using reader object
    with open(os.path.join(args.out,'data.csv'), 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        if header != None:
            for row in tqdm(csv_reader):
                csv_writer.writerow(row)

# Process image (get features)
data = []
img_list = os.listdir(args.path_img)
img = np.array(Image.open(os.path.join(args.path_img,os.path.basename(img_list[0]))).convert('L'))
features = process_image(img)        
data.append(features)
feature = pd.DataFrame(data)

# create annotation and effect new row target
annotation = pd.DataFrame(columns = ['index','img','target'])
new_row = {'index':17227, 'img':'image_test', 'target':0}
#append row to the dataframe
annotation = annotation.append(new_row, ignore_index=True)

# concatinate annotation end feature to final df
final_df = pd.concat([annotation,feature],axis=1)
final_df.to_csv(os.path.join(args.out,'data_add.csv'),index=False)

# add data_add row to data final
with open(os.path.join(args.out,'data_add.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Iterate over each row in the csv using reader object
    with open(os.path.join(args.out,'data.csv'), 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        if header != None:
            for row in tqdm(csv_reader):
                csv_writer.writerow(row)


# normalization data final and effect to data_normalization
df = pd.read_csv(os.path.join(args.out,'data.csv'))
x_array = df.drop(['index','img','target'], axis=1)
#info = df['img']
names = x_array.columns
scaler = MinMaxScaler()
scaler.fit(x_array)
d = scaler.transform(x_array)
scaled_df = pd.DataFrame(d, columns=names)
final_df = pd.concat([scaled_df],axis=1)
final_df.to_csv(os.path.join(args.out,'data_normalization.csv'),index=False)
print("Normalization Done")

# find row image_test normalization
with open(os.path.join(args.out,'data_normalization.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    i = 0
    N = 17227
    for row in csv_reader:
        if i == N:
            print("This is the line.")
            print(row)
            features_image_test = row
            break

        i += 1

# remove file csv
os.remove(os.path.join(args.out,'data.csv'))
os.remove(os.path.join(args.out,'data_add.csv'))
os.remove(os.path.join(args.out,'data_normalization.csv'))

# loaded model SVM to classification image
loaded_model = pickle.load(open(args.loader_model_svm, 'rb'))
result = loaded_model.predict([features_image_test])
if result == [0]:    
    print("image input is NORMAL")
if result == [1]:    
    print("image input is COVID")
if result == [2]:    
    print("image input is PNEUMONIA")