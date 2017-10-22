# how to design the size of vertexes?
# 8 16 32 64 128 256 512 1024 2048 4096 8192
#import random
import numpy as np
foldername = './testdata/'
upperbound = 1000001
#numOfFiles = 2
vertex_size_list = [8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
#vertex_size_list = [3,5]
for vertexes in vertex_size_list:
    filename = foldername+str(vertexes)+'.in'
    print 'writing '+filename

    output = open(filename, 'w')
    a = np.random.randint(1,upperbound,size=[vertexes,vertexes])
    #low inclusive, high exlusive
    #np.savetxt(filename,a)
    #change the diagno value
    for i in range(vertexes):
        a[i][i] = 0
    output.write(str(vertexes))
    output.write('\n')
    for i in range(len(a)):
        output.write(str(a[i])[1:-1])
        output.write('\n')

    output.close()
