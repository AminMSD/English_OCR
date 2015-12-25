import numpy as np
import cv2
from matplotlib import pyplot as plt
import string

img = []
gray = []
alphabet = list(string.ascii_uppercase)
for char in alphabet:
    img.append("")
    gray.append("")
    for i in range(0,200):
        j = alphabet.index(char)
        if (i==0):
            img[j] = [cv2.imread('./training/training/upper/'+char+'/'+str(i)+'.jpg')]
            gray[j] = [cv2.cvtColor(img[j][0],cv2.COLOR_BGR2GRAY)]
        else:
            img[j].append(cv2.imread('./training/training/upper/'+char+'/'+str(i)+'.jpg'))
            gray[j].append(cv2.cvtColor(img[j][-1],cv2.COLOR_BGR2GRAY))

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(gray)

print len(x[0][0])
#print x
print x[8][0][0]

# Now we prepare train_data and test_data.

train = x[:,:100].reshape(-1,400).astype(np.float32)
test = x[:,100:200].reshape(-1,400).astype(np.float32) # Size = (2500,400)
print len(train)
# Create labels for train and test data
k = np.arange(26)
train_labels = np.repeat(k,100)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=1)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
print correct
accuracy = correct*100.0/result.size
print accuracy
