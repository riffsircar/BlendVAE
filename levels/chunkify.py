import sys
import os
import random
import json



# folder = sys.argv[1]

generic_mapping = {
    # already generic
    "-": "-",
    
    # mario to generic
    "X": "X",
    "E": "E",
    "Q": "Q",
    "?": "?",
    "<": "<",
    ">": ">",
    "[": "[",
    "]": "]",
    "o": "o",
    "S": "S",

    # icarus to generic
    "#": "X",
    "H": "E",
    "T": "T",
    "M": "M",
    "D": "D"
}

mapping = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "T": 11,
    "M": 12,
    "D": 13,
    "#": 14,
    "H": 15
    }



levels = []

mario_data = open('mario-1-1.txt','r').read().splitlines()
mario_data = [line.replace('\r\n','') for line in mario_data]
print mario_data

data = []

for offset in range(0,len(mario_data[0])-15):
    temp_data = []
    for line in mario_data:
        temp_data.append(line[offset:offset+16])
    data.append(temp_data)

print len(data), len(data[0]), len(data[0][0])

icarus_data = open('kidicarus_5.txt','r').read().splitlines()
icarus_data = [line.replace('\r\n','') for line in icarus_data]
print icarus_data

print len(icarus_data)

idat = []
for offset in range(0,len(icarus_data)-15):
    temp_data = []
    for line in icarus_data[offset:offset+16]:
        temp_data.append(line)
    idat.append(temp_data)

print len(idat), len(idat[0]), len(idat[0][0])
#print len(data), len(data[0]), len(data[0][0])
#for d in data:
    #print d


for (i, line) in enumerate(data):
    outfile = open('chunks_ng/smb_chunk_' + str(i) + '.txt', 'w')
    temp = []
    for d in line:
        outfile.write(d + '\n')
    outfile.close()

for (i, line) in enumerate(idat):
    outfile = open('chunks_ng/ki_chunk_' + str(i) + '.txt', 'w')
    temp = []
    for d in line:
        outfile.write(d + '\n')
    outfile.close()
        # temp.append(list(d))
        #d_list = list(d)
        #print d_list
        #d_list_map = [mapping[x] for x in d_list]
        # print d, d_list_map
        #temp.append(d_list_map)
    #outdata.append(temp)
# print outdata[0], outdata[1]
# print outdata
