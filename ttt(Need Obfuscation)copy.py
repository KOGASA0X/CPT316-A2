def split_list(ls, n):
    # Check if input list has at least n elements
    if not isinstance(ls, list) or not isinstance(n, int) or len(ls) < n:
        return []
    # Determine how many elements each sublist should contain
    els_per_sublist = len(ls) // n
    # Create list of sublists
    return [ls[i:i + els_per_sublist] for i in range(0, n * els_per_sublist, els_per_sublist)]

def merge_sort(unsorted_array): 
    if len(unsorted_array) > 1: 
        mid = len(unsorted_array) // 2  
        left = unsorted_array[:mid]  
        right = unsorted_array[mid:]  

        merge_sort(left) 
        merge_sort(right) 

        i = j = k = 0

        
        while i < len(left) and j < len(right): 
            if left[i] < right[j]:
                unsorted_array[k] = left[i] 
                i += 1
            else:
                unsorted_array[k] = right[j] 
                j += 1
            k += 1

        while i < len(left):
            unsorted_array[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            unsorted_array[k] = right[j]
            j += 1
            k += 1

# Function to print the list
def print_list(array1): #define a function called print_list that takes an argument called array1
    for i in range(len(array1)): #for each item in the array1 list
        print(array1[i]) #print the item

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

        f=split_list(f,8)
    
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
        print("Time used: ", elapsed_time, "ms")
