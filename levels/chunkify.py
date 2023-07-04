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

"""
images = {
    # TODO: Get T, D, M tiles from Icarus
    "E": Image.open('../tiles/E.png'),
    "H": Image.open('../tiles/H.png'),
    "G": Image.open('../tiles/G.png'),
    "M": Image.open('../tiles/M.png'),
    "o": Image.open('../tiles/o.png'),
    "S": Image.open('../tiles/S.png'),
    "T": Image.open('../tiles/T.png'),
    "?": Image.open('../tiles/Q.png'),
    "Q": Image.open('../tiles/Q.png'),
    "X": Image.open('../tiles/X1.png'),
    "#": Image.open('../tiles/X.png'),
    "-": Image.open('../tiles/-.png'),
    "0": Image.open('../tiles/0.png'),
    "D": Image.open('../tiles/D.png'),
    "<": Image.open('../tiles/PTL.png'),
    ">": Image.open('../tiles/PTR.png'),
    "[": Image.open('../tiles/[.png'),
    "]": Image.open('../tiles/].png')
    }
"""


levels = []
i = 0
#mario_levels = ['mario-1-1.txt','mario-1-2.txt','mario-1-3.txt','mario-2-1.txt','mario-3-1.txt','mario-3-3.txt','mario-4-1.txt','mario-4-2.txt','mario-5-1.txt','mario-5-3.txt','mario-6-1.txt','mario-6-2.txt','mario-6-3.txt','mario-7-1.txt','mario-8-1.txt']
mario_levels = ['mario-1-1.txt']
for ml in mario_levels:
    mario_data = open('smb/' + ml,'r').read().splitlines()
    mario_data = [line.replace('\r\n','') for line in mario_data]
    print("mario data: ", mario_data)

    data = []

    for offset in range(0,len(mario_data[0])-15):
        temp_data = []
        for line in mario_data:
            temp_data.append(line[offset:offset+16])
        data.append(temp_data)

    print(len(data), len(data[0]), len(data[0][0]))

    for (_, line) in enumerate(data):
        outfile = open('chunks_mario/smb_chunk_' + str(i) + '.txt', 'w')
        temp = []
        for d in line:
            outfile.write(d + '\n')
        outfile.close()
        i += 1

#sys.exit()

j = 0
#ki_levels = ['kidicarus_1.txt', 'kidicarus_2.txt','kidicarus_3.txt','kidicarus_4.txt','kidicarus_5.txt','kidicarus_6.txt']
ki_levels = ['kidicarus_1.txt']
for kl in ki_levels:
    icarus_data = open('ki/' + kl,'r').read().splitlines()
    icarus_data = [line.replace('\r\n','') for line in icarus_data]
    print(icarus_data)
    print(len(icarus_data))

    idat = []
    for offset in range(0,len(icarus_data)-15):
        temp_data = []
        for line in icarus_data[offset:offset+16]:
            temp_data.append(line)
        idat.append(temp_data)

    print(len(idat), len(idat[0]), len(idat[0][0]))
    #print len(data), len(data[0]), len(data[0][0])
    #for d in data:
        #print d

    for (_, line) in enumerate(idat):
        outfile = open('chunks_icarus/ki_chunk_' + str(j) + '.txt', 'w')
        temp = []
        for d in line:
            d = d.replace("-","*")
            #print "D: ", d
            outfile.write(d + '\n')
        outfile.close()
        j += 1
            # temp.append(list(d))
            #d_list = list(d)
            #print d_list
            #d_list_map = [mapping[x] for x in d_list]
            # print d, d_list_map
            #temp.append(d_list_map)
        #outdata.append(temp)
    # print outdata[0], outdata[1]
    # print outdata



for (i, line) in enumerate(data):
    outfile = open('chunks_new/smb_chunk_' + str(i) + '.txt', 'w')
    temp = []
    for d in line:
        outfile.write(d + '\n')
    outfile.close()

for (i, line) in enumerate(idat):
    outfile = open('chunks_new/ki_chunk_' + str(i) + '.txt', 'w')
    temp = []
    for d in line:
        d = d.replace("-","*")
        #print "D: ", d
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
