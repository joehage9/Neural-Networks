# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:48:32 2019

@author: joe hage
"""

from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
classifier = load_model("dogAndCatCNNModel2.h5")

# To check the class of one pic
def checkSpecie(path):   
    test_image = image.load_img(path = path, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    # /255 to correct the color channels and if you put the show image after the expand method, you ll have to squeeze it again
    plt.imshow(test_image/255) 
    plt.show()
    test_image = np.expand_dims(test_image, axis = 0)   
    result = classifier.predict(test_image)     
    if result[0][0] == 1: 
        print('It is adog\n')
    else:
        print('it is cat\n')    

                

while True:    
    case=input('Do you want to test:\n1-An image\n2-A whole directory\n3-Sort a directory\n0-Exit\nPlease choose: ')
    if case == '0':        
        break
       
    elif case == '1':        
        while True:
            directory = input('Please specify the path of the image\n')
            if os.path.isfile(directory):
                checkSpecie(directory)
               
                break
            else:
                print('Please insert a valid path.\n')
                
    elif case == '2':  
        while True:
            count = 0
            animalNb = 0
            dogNb = 0
            catNb = 0
            
            directory = input('Please specify the directory, or more than one separated by a space\n')
            directory = directory.split()         
            
            for i in range(len(directory)):
                if os.path.isdir(directory[i]):                       
                    count+=1
                else:
                    print ('Entry number: ' + str(i) + ': ' + directory[i] + ' is not a valid directory')   
                         
            if count == len(directory): 
                print('1 directory found') if count == 1 else  print(str(count) + ' directories found')
                for j in range(len(directory)): 
                    print('testing '+directory[j]+'...')                    
                    for filename in tqdm(os.listdir(directory[j])):
                        test_image=image.load_img(path=directory[j] + '/' + filename, target_size=(128, 128))
                        test_image=image.img_to_array(test_image)
                        test_image=np.expand_dims(test_image, axis=0)   
                        result=classifier.predict(test_image)       
                        animalNb+=1                         
                        if result[0][0]==1:           
                            dogNb+=1                            
                        else:           
                            catNb+=1   
                print('animal count is : ' + str(animalNb) + '\ncat count is: ' + str(catNb) + '\ndog count is: ' + str(dogNb))  
                break
            
    elif case == '3':  
        import shutil
        while True:
            directory = input('Please specify the directory you want to sort\n')
            
            if os.path.isdir(directory):  
                if os.path.isdir(directory+'/dog') == False :
                    os.mkdir(directory+'/dog')
                    
                if os.path.isdir(directory+'/cat') == False :
                    os.mkdir(directory+'/cat')
                    
                for j in range(len(directory)): 
                    print('Processing and moving '+directory+'...\n')                    
                    for filename in tqdm(os.listdir(directory)):
                        file = directory + '/' + filename
                        if os.path.isdir(file) == False :
                            test_image=image.load_img(path = file, target_size=(128, 128))
                            test_image=image.img_to_array(test_image)
                            test_image=np.expand_dims(test_image, axis=0)   
                            result=classifier.predict(test_image)     
                                  
                            if result[0][0]==1:           
                                shutil.move(file, directory + '/dog/' + filename)
                            else:          
                                shutil.move(file, directory + '/cat/' + filename)
                print('Done') 
                break
            else:
                print ('Not a valid directory')               
        
    else:
        print('invalid entry')

