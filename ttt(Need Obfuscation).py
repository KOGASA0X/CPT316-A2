def splited_list(ls,n):
	if not isinstance(ls,list) or not isinstance(n,int):
		return []
	ls_len = len(ls)
	if n<=0 or 0==ls_len:
		return []
	if n > ls_len:
		return []
	elif n == ls_len:
		return [[i] for i in ls]
	else:
		j = ls_len//n
		k = ls_len%n
		### j,j,j,...(前面有n-1个j),j+k
		#步长j,次数n-1
		ls_return = []
		for i in range(0,(n-1)*j,j):
			ls_return.append(ls[i:i+j])
		#算上末尾的j+k
		ls_return.append(ls[(n-1)*j:])
		return ls_return

def merge_sort(unsorted_array): # Base case: if the array is empty or has one element, it is sorted
    if len(unsorted_array) > 1: # Recursive case: divide the array into two sub-arrays
        mid = len(unsorted_array) // 2  # Finding the mid of the array
        left = unsorted_array[:mid]  # Dividing the array elements
        right = unsorted_array[mid:]  # into 2 halves

        merge_sort(left) # Sorting the first half
        merge_sort(right) # Sorting the second half

        i = j = k = 0

        #  data to temp arrays L[] and R[]
        while i < len(left) and j < len(right): # Copy data to temp arrays L[] and R[]
            if left[i] < right[j]:
                unsorted_array[k] = left[i] # Copy the data to temp arrays L[] and R[]
                i += 1
            else:
                unsorted_array[k] = right[j] # Copy the remaining elements of L[], if there are any
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left):
            unsorted_array[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            unsorted_array[k] = right[j]
            j += 1
            k += 1

# Code to print the list
def print_list(array1): # Traverse through all array elements
    for i in range(len(array1)):# Printing the data
        print(array1[i])

from mpi4py import MPI
import numpy as np
import time
import tracemalloc

com=MPI.COMM_WORLD
rank=com.Get_rank()

# driver code to test the above code
if __name__ == '__main__': # Creating an array of size 10
    f = []# empty list
    if rank ==0:
        file = open("sgb-words.txt","r") # Open the file
    
        for x in file: # read each line in the file
            f.append(x)

        f=splited_list(f,8)
    
    # calculate time used by the function
        start_time = time.time() #start time
    
    # starting the monitoring
        tracemalloc.start()

    
    f=com.scatter(f,root=0)
    merge_sort(f) # sort the list

    f=com.gather(f,root=0)

    if rank ==0:
        for i in range(1,len(f)):
            f[0].extend(f[i])
        f=f[0]
        merge_sort(f)
        current_size, peak_size = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = (time.time() - start_time)*1000
    
    
        print("Sorted array is: ", end="\n")
        print(len(f))
        print_list(f)
    
        # end time
        print("Elapsed time: %s milliseconds" % elapsed_time )
    
        # memory used by the function
        import sys
        print("Used memory: ",peak_size/(1024*1024), "MB") # memory used by the function
