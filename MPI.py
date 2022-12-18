from mpi4py import MPI
import time

com=MPI.COMM_WORLD
rank=com.Get_rank()

def split_list(ls: list[int], n: int) -> list[list[int]]:
    # Check if input list has at least n elements
    if not isinstance(ls, list) or not isinstance(n, int) or len(ls) < n:
        return []
    # Determine how many elements each sublist should contain
    els_per_sublist = len(ls) // n
    # Create list of sublists
    return [ls[i:i + els_per_sublist] for i in range(0, n * els_per_sublist, els_per_sublist)]

def merge_sort(array: list[int]) -> list[int]: #function to sort array
    if len(array) > 1: #if array length is greater than 1
        mid = len(array) // 2 #find the middle of the array
        left = array[:mid] #left half of the array
        right = array[mid:] #right half of the array

        merge_sort(left) #sort the left half of the array
        merge_sort(right) #sort the right half of the array

        i = j = k = 0
        
        while i < len(left) and j < len(right): #go through the left and right half of the array
            if left[i] < right[j]: #if left half is less than right half
                array[k] = left[i] #add the left half to the array
                i += 1
            else:
                array[k] = right[j] #add the right half to the array
                j += 1
            k += 1

        while i < len(left): #if the left half is anything but the middle, add it to the array
            array[k] = left[i]
            i += 1
            k += 1

        while j < len(right): #if the right half is anything but the middle, add it to the array
            array[k] = right[j]
            j += 1
            k += 1

if __name__ == '__main__':
    string_list = []
    if rank ==0:
        file = open("sgb-words.txt","r") 
    
        for x in file: 
            string_list.append(x)

        string_list=split_list(string_list,8)
        start_time = time.time() 
    
    string_list=com.scatter(string_list,root=0)
    merge_sort(string_list) 
    string_list=com.gather(string_list,root=0)

    if rank ==0:
        for i in range(1,len(string_list)):
            string_list[0].extend(string_list[i])
        string_list=string_list[0]
        merge_sort(string_list)

        run_time = (time.time() - start_time)*1000
    
        for i in range(len(string_list)): 
            print(string_list[i]) 
        print("Time used: ", run_time, "ms")

# The code above does the following, explained in English:
# 1. The code first imports the required libraries and modules, and then sets the number of processes to be used in the parallelization.
# 2. The code then opens the file that contains the list of words, and reads it line by line.
# 3. The code then splits the list of words into eight equal parts, one for each process. 
# 4. The code then starts the timer. 
# 5. The code then uses the scatter function to distribute the list of words into eight parts, one for each process. 
# 6. The code then sorts each part using merge sort. 
# 7. The code then uses the gather function to collect the sorted parts.
# 8. The code then merges all the parts into one list of words. 
# 9. The code then sorts the list of words using merge sort.
# 10. The code then stops the timer.
# 11. The code then prints the list of words (in alphabetical order) and the time taken to sort the list of words. 