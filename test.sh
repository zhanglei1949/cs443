#/bin/bash

make

for i in 8 16 32 64 128 256 512 1024 2048 4096 8192
#for i in 8192
do
	echo $i
	for p in 1 2 4:
	do	
		mpiexec -n $p ./dijkstra-file ./testdata/$i.in
	done
done
