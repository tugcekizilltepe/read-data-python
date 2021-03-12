# Data Types

Here, I will show ways to load, read, write or open these files. There are many ways to do some operations. I will only show some of them. You can check the references for more information or operations.

I also want to show each process as a separate piece. That's why I put the import parts repeatedly.

* [Tabular, Spreadsheet and Interchange Data Formats](#1)
* [Data File Formats](#2)
* [Image Data Formats](#3)
* [Video Data Formats](#4)
* [Audio Data Formats](#5)
* [Text Data Formats](#6)
* [References](#7)

<a id = 1></a>
# Tabular, Spreadsheet and Interchange Data Formats

1. Tabular Text Formats
  * "Table" — generic tabular data (.dat)

  * "CSV" — comma-separated values (.csv)

  * "TSV" — tab-separated values (.tsv)

2. Spreadsheet Formats
  * "XLS" — Excel spreadsheet (.xls)

  * "XLSX" — Excel 2007 format (.xlsx)

  * "ODS" — OpenDocument spreadsheet (.ods)

  * "SXC" — OpenOffice 1.0 spreadsheet file (.sxc)

  * "DIF" — VisiCalc data interchange format (.dif)

2. Data Interchange Formats
  * "RawJSON" — JSON with objects as associations

  * "JSON" — JSON with objects as rule lists (.json)

  * "UBJSON" — Universal Binary JSON (.ubj)

## Tabular Text Formats
You can use this in general to read a file:

```python
import os

with open(filename, 'r') as file:
    text = file.read()
    print(text)
```

Load the data in the .dat file into a numpy array
```python

import numpy as np

data = np.loadtxt(filename)
```

Read .csv file 

```python

import pandas as pd

df =  pd.read_csv(filename)
``` 

Read .tsv file
```python

import pandas as pd

df = pd.read_csv(filename, sep='\t')
``` 
## Spreadsheet Formats

In below codes, there is parameter called sheet_name. You can find sheet name by looking at the bottom left of excel file.

```python
import xlrd

# Load .xls file
#You need to install,and import xlrd
dfs = pd.read_excel(filename, sheet_name= sheet_name, engine="openpyxl")

# Load .xlsx file
dfs = pd.read_excel(filename, sheet_name= sheet_name)

# Load .ods file
# Returns DataFrame
from pandas_ods_reader import read_ods

dods = read_ods(filename, sheet_name)

``` 
```python
# Load .dif file
import DataInterchangeFormat

# as a list of tuples
dif = DataInterchangeFormat.DIFReader(filename, first_row_keys=True)
dif.sheet 

# as a list of dictionaries
dif = DataInterchangeFormat.DIFDictReader(filename, first_row_keys=True)
dif.sheet
``` 
## Data Interchange Formats

```python
import json

with open (filename) as f:
  data = json.load(f)
``` 

<a id = 2></a>
# Data File Formats
PKL – Pickle format, HDF5, Zip, SQL, MAT, NPY, NPZ

Pickling is a way to convert a python object into a character stream.

```python
import pickle

#Store Data
#Open with binary mode
data = {}
file = open(filename, 'ab')
pickle.dump(data, file)
file.close()

#Load Data
#Open with binary mode
file = open(filename, 'rb')
data = pickle.load(file)
file.close()
``` 
```python
#Load hd5py files
import h5py

#Read a file
#h5py.File acts like a Python dictionary
f = h5py.File(filename, 'r')

list(f.keys())

#If there is one data set called my_dataset
dset = f['my_dataset']
dset.shape 
dset.dtype

#Creating a file

import h5py
import numpy as np
f = h5py.File(filename, "w")

dset = f.create_dataset("mydataset", (100,), dtype='i')
``` 
```python
#Read zip file
from zipfile import ZipFile 
  
with ZipFile(filename) as zip:
    zip.printdir() # print all the contents of the zip file
    with zip.open(filename2) as myfile:
        print(myfile.read())

#Write on zip file
with ZipFile(filename, 'w') as zip:
    zip.write(filename2)
``` 
```python
#Establish connection to a server
def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

connection = create_server_connection("localhost", "root", pw)
#It will print "MySQL Database connection successful" if there is no error.

#Create a new Database

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

 create_database(connection, "CREATE DATABASE Library")       
``` 
```python
#For matlab up to 7.1 can be read usin scipy.ip
from scipy.io import loadmat

data = loadmat(filename)

#For matlab 7.3 and greater
import tables
file = tables.openFile(filename)
``` 

An NPY file is a NumPy array file which is created using Python and NumPy library. 
```python
import numpy as np

#Save .npy file
np.save(filename, array)

#Load .npy file
array = np.load(filename)
``` 
An NPZ file is a file which is an array using gzip compression.

```python
import numpy as np

#Save as a compression
np.savez(filename)

#Load npz file
npzfile = np.load(filename)

``` 
<a id = 3></a>
# Image Data Formats
* JPG, PNG, BMP, TIFF 

```python
from PIL import Image

image = Image.open(filename)
print(image.size)
image.show()

from matplotlib import image

data = image.imread(filename)
plt.imshow(data)
plt.show()
``` 
<a id = 4></a>
# Video Data Formats
* MP4, AVI, MPEG

```python
pip install cv
``` 
```python
import cv2
cap = cv2.VideoCapture(filename)

while(cap.isOpened()):
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow("frame", gray)

  if cv2.waitKey(1) & 0XFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
``` 
### Convert one video format to other
```python
pip install PythonVideoConverter
``` 
```python
from converter import Converter
conv = Converter()

info = conv.probe(filename)

convert = conv.convert(filename1, filename2, {
    'format': 'mp4',
    'audio': {
        'codec': 'aac',
        'samplerate': 11025,
        'channels': 2
    },
    'video': {
        'codec': 'hevc',
        'width': 720,
        'height': 400,
        'fps': 25
    }})

for timecode in convert:
    print(f'\rConverting ({timecode:.2f}) ...')
``` 
<a id = 5></a>
# Audio Data Formats
* MP3, MIDI, WAV 
```python
pip install playsound
``` 
```python
from playsound import playsound

playsound(filename)
``` 
<a id = 6></a>
# Text Data Formats
* TXT, PDF, DOCX
## TXT
```python
import os

#Open txt file
f = open(filename, 'r')

#Print Lines
for line in f:
  print(line)
``` 
## PDF
```python
pip install PyPDF2
``` 
```python
import PyPDF2

#Create an object:
file = open(filename, 'rb')

#Ceate a pdf reader object
fileReader = PyPDF2.PdfFileReader(file)

#Get document info
print(fileReader.documentInfo)

#Get page as a text
print(fileReader.getPage(0).extractText())

#Get number of pages in pdf file
fileReader.numPages
``` 
## DOCX
```python
pip install python-docx
``` 
```python
import docx

doc = docx.Document(filename)

#get paragraphs
paragraphs = doc.paragraphs
``` 
```python
import docx2txt

doc = docx2txt.process(filename)

#Print whole page
print(text)
``` 
<a id = 7></a>
# References

* For .dif files: https://github.com/lysdexia/python-dif  
* For .json files: https://www.geeksforgeeks.org/json-load-in-python/  
* For .hd5py files: https://docs.h5py.org/en/stable/quick.html  
* For .zip files: https://docs.python.org/3/library/zipfile.html  
* For SQL files: https://github.com/thecraigd/Python_SQL  
* For .npz files: https://numpy.org/doc/stable/reference/generated/numpy.savez.html   
* For video converter: https://pypi.org/project/PythonVideoConverter/  
* For .pdf files: https://pythonhosted.org/PyPDF2/  
* For .docx files: https://python-docx.readthedocs.io/en/latest/
