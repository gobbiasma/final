
import os
import pandas as pd
import argparse



def create_annotation(path, a):
   

   
    #images_path = "../input/mamo-croping/crop/*.png"
    #masks_path = os.path.join(path,'Ground-truths')
    
   images = os.listdir(path)
    #masks = os.listdir(masks_path)
   Calc_images = []
   Mass_images = []
#print(images[0])

   for img in images: 
       if "Calc" in img:
           Calc_images.append(img)
   for img in images: 
       if "Mass" in img:
           Mass_images.append(img)
       

    #clac_images =[image for image in Calc]
    #mass_images =[image for image in Mass]

   Calc = pd.DataFrame(columns=['img','target'])
   
   Mass = pd.DataFrame(columns=['img','target'])

   Calc['img'] = Calc_images
   Calc['target'] = a
   Mass['img'] = Mass_images
   Mass['target'] = 1

   annotation = pd.concat([Calc], [Mass])

   annotation = annotation.reset_index()

   return annotation

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='../input/mamo-croping/crop')
    parser.add_argument('--a', type=str, default = '0')
    parser.add_argument('--folder_image_name', type=str, default='crop')
    # arguments for training
    #parser.add_argument('--img_size', type = int , default = 800)

    #parser.add_argument('--load_model', type=str, default='best_checkpoint.pt', help='.pth file path to load model')

    parser.add_argument('--out', type=str, default='./')
    return parser.parse_args()

def main():
    
    args = get_args()

    if not os.path.exists(args.out):
        print("path created")
        os.mkdir(args.out+'/target')
    
    df = create_annotation(args.path, args.a)

    df.to_csv(os.path.join(args.out,'target.csv'),index=False)

if __name__ == '__main__':

    main()
