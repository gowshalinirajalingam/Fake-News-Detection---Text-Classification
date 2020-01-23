#reference https://www.mkyong.com/python/python-how-to-list-all-files-in-a-directory/

import os
import json
import pandas as pd



def createDF(path,tweetid,label):
    # r=root, d=directories, f = files

    folders=[]
    #read folders inside the path
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(os.path.join(r, folder))

    files = []

    for f in folders:
        if 'source-tweet' in f:
            # print(f)
            #read files inside source-tweet folder
            for r, d, f1 in os.walk(f):
                for file in f1:
                    if '.json' in file:
                        files.append(os.path.join(r, file))
    df = pd.DataFrame()
    for f in files: 
        #read json file
        with open(f, 'r+') as f:
            data = json.load(f)
            words = str(f).split('\\')
            for word in words:
                # if tweetid in str(word):
                if word.startswith(tweetid):
                    id =word
                    break
                else:
                    id=1
            if (id ==1):        #when iterating through all folders if the word is not in the words array it gives error.so added this to avoid.
                continue
            textdf = data['text'].replace("\n"," ")
            df1 =pd.DataFrame({'tweet_id':'"'+str(id)+'"','text': textdf}, index=[0])
            df = df.append(df1,sort=False) 
    df['label'] = label
    return df


rdf = pd.DataFrame()
#------------------------------------------------ charliehebdo
path1 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\charliehebdo\\non-rumours'
rdf1 = createDF(path1,'55','REAL')
print('charliehebdo REAL success')

path2 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\charliehebdo\\rumours'
rdf2 = createDF(path2,'55','FAKE')
print('charliehebdo FAKE success')


# ---------------------------------------------------ferguson
path3 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\ferguson\\non-rumours'
rdf3 = createDF(path3,'49','REAL')
rdf4 = createDF(path3,'50','REAL')
print('ferguson REAL success')


path4 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\ferguson\\rumours'
rdf5 = createDF(path4,'49','FAKE')
rdf6 = createDF(path4,'50','FAKE')
print('ferguson FAKE success')


# ----------------------------------------------------germanwings-crash
path5 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\germanwings-crash\\non-rumours'
rdf7 = createDF(path5,'58','REAL')
print('germanwings-crash REAL success')


path6 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\germanwings-crash\\rumours'
rdf8 = createDF(path6,'58','FAKE')
print('germanwings-crash FAKE success')


#-----------------------------------------------------ottawashooting
path7 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\ottawashooting\\non-rumours'
rdf9 = createDF(path7,'52','REAL')
print('ottawashooting REAL success')


path8 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\ottawashooting\\rumours'
rdf10 = createDF(path8,'52','FAKE')
print('ottawashooting FAKE success')


# -------------------------------------------------------sydneysiege
path9 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\sydneysiege\\non-rumours'
rdf11 = createDF(path9,'54','REAL')
print('sydneysiege REAL success')


path10 = 'G:\\sliit DS\\4th year 1st seme\\thawes research\\thawes-20190512T131638Z-001\\thawes\\Fake_Media_Rich_News_detection_A_Srvey\\PHEME dataset\\pheme-rnr-dataset\\sydneysiege\\rumours'
rdf12 = createDF(path10,'54','FAKE')
print('sydneysiege FAKE success')





dfresult = rdf1.append(rdf1).append(rdf2).append(rdf3).append(rdf4).append(rdf5).append(rdf6).append(rdf7).append(rdf8).append(rdf9).append(rdf10).append(rdf11).append(rdf12)


dfresult.to_csv("G:\\sliit DS\\4th year 1st seme\\thawes research\\model building fake real\\pheme_dataset.csv")
    

