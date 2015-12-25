from pytesser import *
import string

img = []
result = []
correct = 0
number = 0
alphabet = list(string.ascii_uppercase)
for char in alphabet:
    for i in range(100,200):
        number += 1
        image = Image.open('./training/training/upper/'+char+'/'+str(i)+'.jpg')
        response = image_to_string(image).strip()

        if (response == char ):
            result.append(True)
            correct += 1
        else:
            result.append(False)


print correct
print (correct*100.0/number)
