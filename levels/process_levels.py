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
    "H": 15,
    "*": 16,
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


outdata = []
for line in data:
    temp = []
    for d in line:
        # temp.append(list(d))
        d_list = list(d)
        d_list_map = [mapping[x] for x in d_list]
        # print d, d_list_map
        temp.append(d_list_map)
    outdata.append(temp)
#print "0: ", outdata[0], "1: ", outdata[1]

#with open('levels_smb.json','w') as outfile:
#    json.dump(outdata, outfile)

"""
for offset_h in range(0,len(mario_data[0])-7):
    #print "H: ", offset_h
    for offset_v in range(0,9):
        temp_data = []
        #print "V: ", offset_v
        for line in mario_data[offset_v:offset_v+8]:
            #print "line: ", line
            temp_data.append(line[offset_h:offset_h+8])
            #print "Line: ", line[offset_h:offset_h+8]
        data.append(temp_data)
"""
print len(data), len(data[0]), len(data[0][0])

icarus_data = open('kidicarus_5.txt','r').read().splitlines()
icarus_data = [line.replace('\r\n','') for line in icarus_data]
print icarus_data

print len(icarus_data)

#data = []
for offset in range(0,len(icarus_data)-15):
    temp_data = []
    for line in icarus_data[offset:offset+16]:
        temp_data.append(line)
    data.append(temp_data)
"""
for offset_v in range(0,len(icarus_data)-7):
    for offset_h in range(0,9):
        temp_data = []
        for line in icarus_data[offset_v:offset_v+8]:
            temp_data.append(line[offset_h:offset_h+8])
        data.append(temp_data)

"""
"""
print len(idat), len(idat[0]), len(idat[0][0])
for i in range(50):
    print idat[i]
sys.exit()
"""
print len(data), len(data[0]), len(data[0][0])
#for d in data:
    #print d

outdata = []
for line in data:
    temp = []
    for d in line:
        # temp.append(list(d))
        d_list = list(d)
        d_list_map = [mapping[x] for x in d_list]
        # print d, d_list_map
        temp.append(d_list_map)
    outdata.append(temp)
#print "0: ", outdata[0], "1: ", outdata[1]

with open('levels_both.json','w') as outfile:
    json.dump(outdata, outfile)
