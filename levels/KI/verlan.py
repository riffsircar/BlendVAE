import sys
import os
import random
import json

filename = str(sys.argv[1])

f = open(filename, "rb")
s = f.readlines()
f.close()
f = open(filename, "wb")
s.reverse()
for item in s:
  f.write(item)
f.close()
