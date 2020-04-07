import numpy as np
from matplotlib import image
from matplotlib import pyplot


def loadData(trainPercent=0.94, relative_path='../Images/', format='.jpg', size=750):
    maleTrainData = []
    femaleTrainData = []
    maleTestData = []
    femaleTestData = []
    trainSize = size * trainPercent
    for i in range(size):
        imgMale = image.imread(relative_path + 'CM' + str(i + 1) + format)
        imgFemale = image.imread(relative_path + 'CF' + str(i + 1) + format)
        if i + 1 < trainSize:
            maleTrainData.append(imgMale)
            femaleTrainData.append(imgFemale)
        else:
            maleTestData.append(imgMale)
            femaleTestData.append(imgFemale)
    return maleTrainData, femaleTrainData, maleTestData, femaleTestData


#maleTrainData, femaleTrainData,maleTestData,femaleTestData = loadData()
#print(np.shape(maleTrainData))
