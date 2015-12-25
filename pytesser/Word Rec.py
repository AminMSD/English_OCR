from pytesser import *
from xml.dom import minidom


response = []
doc = minidom.parse("./word_2/word.xml")
images = doc.getElementsByTagName("image")
for image in images:
    fileDir = image.getAttribute("file")
    word = image.getAttribute("tag")
    if 'word/12' not in fileDir:
        response.append(word)
        

print response

correct = 0
number = 0
result = []
for j in range(1,12):
    for i in range(1,101):
        number += 1
        image = Image.open('./word_2/word/'+str(j)+'/'+str(((j-1)*100)+i)+'.jpg')
        response2 = image_to_string(image).strip()
        if (response2 == response[((j-1)*100)+i-1]):
            print response2+str(((j-1)*100)+i)
            result.append(True)
            correct += 1
        else:
            result.append(False)

print correct
print number
print (correct*100.0/number)
