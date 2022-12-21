//cuda merge sort

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <time.h>

using namespace std;

// Merges two sorted arrays and stores the result in a third array
__global__ void mergeTwoSortedArrays(char** a, char** b, char** result, int sizeA, int sizeB){
    int i = 0; // Index for a
    int j = 0; // Index for b
    int k = 0; // Index for result
    while(i < sizeA && j < sizeB){
        if(strcmp(a[i], b[j]) < 0){
            result[k] = a[i];
            i++;
        }else{
            result[k] = b[j];
            j++;
        }
        k++;
    }
    while(i < sizeA){
        result[k] = a[i];
        i++;
        k++;
    }
    while(j < sizeB){
        result[k] = b[j];
        j++;
        k++;
    }
}

__global__ void mergeSort(char** a, int size){  
    // Base case
    if(size <= 1){
        return;
    }
    // Divide the array in half
    int leftSize = size/2;    
    int rightSize = size - leftSize;    
    // Create left and right arrays
    char** left = (char**)malloc(leftSize * sizeof(char*));
    char** right = (char**)malloc(rightSize * sizeof(char*));
    // Iterate through the original array and copy elements into the left and right arrays
    int i = 0;
    for(int i = 0; i < size; i++){
        if(i < leftSize){
            left[i] = a[i];
        }else{
            right[i - leftSize] = a[i];
        }  
    }
    // Recursively sort the left and right arrays
    mergeSort(left, leftSize);
    mergeSort(right, rightSize);
    // Merge the two sorted arrays
    mergeTwoSortedArrays(left, right, a, leftSize, rightSize);
}

// readfile reads the file named filename and returns a list<string>
// containing each line of the file as a list element.
list<string> readfile(string filename){
    ifstream file(filename);  // open the file
    string str;               // a string to hold each line
    list<string> a;           // the list to return
    while (getline(file, str)) // read each line
    {
        a.push_back(str);    // add the line to the list
    }
    return a;                // return the list
}


//This function writes the contents of a list to a file
//The list is passed to the function as a reference, so the original list is changed
//The filename is passed as a string
void writetofile(list<string> &a, string filename){
    //Open the file for writing
    ofstream file(filename);
    //Loop through the list
    for(list<string>::iterator it = a.begin(); it != a.end(); it++){
        //Write each list element to the file
        file << *it << endl;
    }
}

int main(){
    //Read in the file and store each line in a list.
    list<string> a = readfile("test1.txt");
    //Convert the list of strings into an array of strings for CUDA.
    char** a1 = (char**)malloc(a.size() * sizeof(char*));
    int i = 0;
    for(list<string>::iterator it = a.begin(); it != a.end(); it++){
        a1[i] = (char*)malloc((*it).size() * sizeof(char));
        strcpy(a1[i], (*it).c_str());
        i++;
    }
    //Allocate memory on the GPU and copy the array of strings over.
    char** a2;
    cudaMalloc((void**)&a2, a.size() * sizeof(char*));
    cudaMemcpy(a2, a1, a.size() * sizeof(char*), cudaMemcpyHostToDevice);
    //Call the kernel on the GPU.
    mergeSort<<<1, 1>>>(a2, a.size());
    //Copy the sorted array of strings back to the CPU.
    cudaMemcpy(a1, a2, a.size() * sizeof(char*), cudaMemcpyDeviceToHost);
    //Convert the array of strings back into a list of strings.
    list<string> result;
    for(int i = 0; i < a.size(); i++){
        result.push_back(a1[i]);
    }
    //Write the sorted list to a file.
    writetofile(result, "test1_result.txt");
    return 0;
}
