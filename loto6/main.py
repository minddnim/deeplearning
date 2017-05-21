import csv
import numpy as np
from math import sqrt

from LotoNN import run, PAST_NUMBER, MAX_NUMBER

import PIL.Image
import skimage.color

def flatten(nested_list):
  return [e for inner_list in nested_list for e in inner_list]

def readData():
  with open('loto6.csv', newline='') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
      hitNumbers = [np.int32(x) for x in row]
      canvas = np.array(PIL.Image.new('RGB', (5, 10), (0, 0, 0)))
      for cnt, i in enumerate(hitNumbers):
        index = i - 1
        r = index % 10
        c = np.int32(index / 10)
        if cnt == len(hitNumbers)-1:
          canvas[r][c] = [255, 0, 0]
        else:
          canvas[r][c] = [0, 0, 255]
      canvas[9][4] = canvas[8][4] = canvas[7][4] = [255, 255, 255]
      canvas[6][4] = canvas[5][4] = canvas[4][4] = canvas[3][4] = [255, 255, 255]
      img2 = PIL.Image.fromarray(np.uint8(canvas))
      img2.save("./correct/{i}.png".format(i=(idx + 1 - 10)))
      img2.save("./org/{i}.png".format(i=(idx + 1)))

def main():
  readData()

if __name__ == '__main__':
  main()