import cv2
import numpy as np
from matplotlib import pyplot as plt
import string
from xml.dom import minidom
import os.path
#---training
img = []
gray = []
alphabet = list(string.ascii_uppercase)
for char in alphabet:
    img.append("")
    gray.append("")
    for i in range(0,100):
        j = alphabet.index(char)
        #print j
        if (i==0):
            img[j] = [cv2.imread('./training/training/upper/'+char+'/'+str(i)+'.jpg')]
            gray[j] = [cv2.cvtColor(img[j][0],cv2.COLOR_BGR2GRAY)]
        else:
            img[j].append(cv2.imread('./training/training/upper/'+char+'/'+str(i)+'.jpg'))
            gray[j].append(cv2.cvtColor(img[j][-1],cv2.COLOR_BGR2GRAY))
for char in alphabet:
    img.append("")
    gray.append("")
    for i in range(0,100):
        j = alphabet.index(char)+26
        #print j
        if (i==0):
            img[j] = [cv2.imread('./training/training/lower/'+char+'/'+str(i)+'.jpg')]
            gray[j] = [cv2.cvtColor(img[j][0],cv2.COLOR_BGR2GRAY)]
        else:
            #if (os.path.exists('./training/training/lower/'+char+'/'+str(i)+'.jpg')):
            img[j].append(cv2.imread('./training/training/lower/'+char+'/'+str(i)+'.jpg'))
            gray[j].append(cv2.cvtColor(img[j][-1],cv2.COLOR_BGR2GRAY))
            #else:
            #    pass

#print gray[0]
#print len(gray)
#print len(gray[0][0])
x = np.array(gray)
print len(x)
print len(x[0])
print len(x[0][0])
#print x
train = x[:,:].reshape(-1,400).astype(np.float32)
#print len(train)
#print len(train[0])
#print train[0][0]
#print gray[0][0][0]
k = np.arange(65,117)
print k
#train_labels = np.arange(0)
#for i in range(0,26):
#    print i
#    train_labels = np.concatenate((train_labels,np.repeat(k[i],lenTrain[i])))
train_labels = np.repeat(k,100)
print train_labels
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)


#---reading xml
response = []
doc = minidom.parse("./word_2/word.xml")
images = doc.getElementsByTagName("image")
for image in images:
    fileDir = image.getAttribute("file")
    word = image.getAttribute("tag")
    if 'word/12' not in fileDir:
        response.append(word)
        
#---testing and recognition
correct = 0
number = 0
result = []
for j in range(1,12):
    for i in range(1,101):
        number += 1
        img = cv2.imread('./word_2/word/'+str(j)+'/'+str(((j-1)*100)+i)+'.jpg')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        tret3,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tret4,threshInv = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        threshInv_color = cv2.cvtColor(threshInv,cv2.COLOR_GRAY2BGR)
        #thresh = cv2.dilate(thresh,None,iterations = 1)
        thresh = cv2.erode(thresh,None,iterations = 1)
        threshInv = cv2.erode(threshInv,None,iterations = 1)
        thresh,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        threshInv,contoursInv,hierarchyInv = cv2.findContours(threshInv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            rects.append([x,y,w,h])
        rects.sort(key = lambda rects:rects[0])
        #print rects
        word = ''
        for cnt in rects:
            x,y,w,h = cnt
            im = thresh_color[y:y+h, x:x+w]
            im3 = im.copy()
            height, width, depth = im3.shape
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            #thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)   
            roi = blur[0:height, 0:width]
            roi = cv2.resize(roi,(20,20))
            roi = roi.reshape((1,400))
            roi = np.float32(roi)
            retval, results, neigh_resp, dists = knn.findNearest(roi, k = 1)
            if (results>90):
                word += chr(results+6)
            if (results<=90):
                word += chr(results)


        rectsInv = []
        for cnt in contoursInv:
            x,y,w,h = cv2.boundingRect(cnt)
            rectsInv.append([x,y,w,h])
        rectsInv.sort(key = lambda rectsInv:rectsInv[0])
        #------BWInv
        wordInv = ''
        for cnt in rectsInv:
            x,y,w,h = cnt
            im = thresh_color[y:y+h, x:x+w]
            im3 = im.copy()
            height, width, depth = im3.shape
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            #thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)   
            roi = blur[0:height, 0:width]
            roi = cv2.resize(roi,(20,20))
            roi = roi.reshape((1,400))
            roi = np.float32(roi)
            retval, results, neigh_resp, dists = knn.findNearest(roi, k = 1)
            if (results>90):
                wordInv += chr(results+6)
            if (results<=90):
                wordInv += chr(results)
                    
        if (word == response[((j-1)*100)+i-1] or wordInv == response[((j-1)*100)+i-1]):
            result.append(True)
            print word+' '+str(((j-1)*100)+i)+' Or '+wordInv+' '+str(((j-1)*100)+i)
            correct += 1
        else:
            result.append(False)


print correct
print number
print (correct*100.0/number)

