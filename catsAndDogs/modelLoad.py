# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:48:32 2019

@author: joe hage
"""

from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import time 
classifier=load_model("dogAndCatCNNModel.h5")

# To check the class of one pic
def checkSpecie(path):   
    test_image=image.load_img(path=path, target_size=(128, 128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image, axis=0)   
    result=classifier.predict(test_image)  
    if result[0][0]==1:
        print('dog\n')
    else:
        print('cat\n') 
        
def stopWatch(value):
    valueD = (((value/365)/24)/60)
    Days = int (valueD)    
    valueH = (valueD-Days)*365
    Hours = int(valueH)    
    valueM = (valueH - Hours)*24
    Minutes = int(valueM)    
    valueS = (valueM - Minutes)*60
    Seconds = int(valueS)  
    if Days>1:
        print('Time elapsed: '+str(Days)+' days '+str(Hours)+' hours '+str(Minutes)+'minutes '+str(Seconds)+' seconds')
    elif Hours>1:
        print('Time elapsed: '+str(Hours)+' hours '+str(Minutes)+'minutes '+str(Seconds)+' seconds')
    elif Minutes>1:
        print('Time elapsed: '+str(Minutes)+'minutes '+str(Seconds)+' seconds')
    elif Seconds>1:
        print('Time elapsed: '+str(Seconds)+' seconds')
    elif Seconds<1:
        print('Time elapsed is less than 1 second')
        
# Stopwatch starts
start = time.time() 
       
animalNb=0
dogNb=0
catNb=0
# 
#directory = ['dataset/test_set/cat','dataset/training_set/cat','dataset/test_set/dog','dataset/training_set/dog']  
directory=['dataset/single_prediction']
#directory = ['dataset/test_set/cat','dataset/test_set/dog']  

for i in range(len(directory)):
    print('testing '+directory[i]+'...')
    for filename in os.listdir(directory[i]):
        test_image=image.load_img(path=directory[i]+'/'+filename, target_size=(128, 128))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image, axis=0)   
        result=classifier.predict(test_image)       
        animalNb+=1  
        if animalNb%1000==0:
            print(str(animalNb)+' pictures done.')
        if result[0][0]==1:           
            dogNb+=1
        else:           
            catNb+=1
            
print('animal count is : '+str(animalNb)+'\ncat count is: '+str(catNb)+'\ndog count is: '+str(dogNb))

# Stopwatch ends
end = time.time()   
# Check time elapsed   
stopWatch(end-start)
