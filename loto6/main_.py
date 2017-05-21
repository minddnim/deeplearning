import csv
import numpy as np
from math import sqrt

from LotoNN import run, PAST_NUMBER, MAX_NUMBER

import PIL.Image
import skimage.color

NUMBER = 1120

def flatten(nested_list):
  return [e for inner_list in nested_list for e in inner_list]

def readData():
  with open('loto6.csv', newline='') as f:
    reader = csv.reader(f)
    inputDataList = []
    for row in reader:
      hitNumbers = [np.int32(x) for x in row]
      inputData = np.array([np.float32(0.0)]*(10*5))
      inputData = inputData.reshape(10, 5)
      for i in hitNumbers:
        index = i - 1
        r = index % 10
        c = np.int32(index / 10)
        inputData[r][c] = np.float32(1.0)
      inputData[9][4] = inputData[8][4] = inputData[7][4] = np.float32(0.0)
      inputData[6][4] = inputData[5][4] = inputData[4][4] = inputData[3][4] = np.float32(0.0)
      inputDataList.append(inputData)
  return inputDataList

def createTestData(inputDataList):
  mini_batch_size = NUMBER - PAST_NUMBER
  in_channels = PAST_NUMBER
  XS = np.zeros((mini_batch_size, in_channels, 10, 5)).astype(np.float32)
  YS = np.zeros((mini_batch_size, 10, 5)).astype(np.int32)

  for i in range(mini_batch_size):
    for j in range(0, PAST_NUMBER):
      img = inputDataList[i + j]
      XS[i, j, :, :] = img
    img_label = inputDataList[i+PAST_NUMBER].astype(np.int32)
    YS[i, :, :] = img_label
  return XS, YS

def main():
  inputDataList = readData()
  XS, YS = createTestData(inputDataList)
  run(XS, YS)

if __name__ == '__main__':
  main()