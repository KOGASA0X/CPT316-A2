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

__global__ void mergeTwoSortedArrays(char** a, char** b, char** result, int sizeA, int sizeB){
    int i = 0;
    int j = 0;
    int k = 0;
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
    if(size <= 1){
        return;
    }
    int leftSize = size/2;    
    int rightSize = size - leftSize;    
    char** left = (char**)malloc(leftSize * sizeof(char*));
    char** right = (char**)malloc(rightSize * sizeof(char*));
    int i = 0;
    for(int i = 0; i < size; i++){
        if(i < leftSize){
            left[i] = a[i];
        }else{
            right[i - leftSize] = a[i];
        }  
    }
    mergeSort(left, leftSize);
    mergeSort(right, rightSize);
    mergeTwoSortedArrays(left, right, a, leftSize, rightSize);
}

list<string> readfile(string filename){
    ifstream file(filename);
    string str;
    list<string> a;
    while (getline(file, str))
    {
        a.push_back(str);
    }
    return a;
}

void writetofile(list<string> a, string filename){
    ofstream file(filename);
    for(list<string>::iterator it = a.begin(); it != a.end(); it++){
        file << *it << endl;
    }
}

int main(){
    list<string> a = readfile("test1.txt");
    char** a1 = (char**)malloc(a.size() * sizeof(char*));
    int i = 0;
    for(list<string>::iterator it = a.begin(); it != a.end(); it++){
        a1[i] = (char*)malloc((*it).size() * sizeof(char));
        strcpy(a1[i], (*it).c_str());
        i++;
    }
    char** a2;
    cudaMalloc((void**)&a2, a.size() * sizeof(char*));
    cudaMemcpy(a2, a1, a.size() * sizeof(char*), cudaMemcpyHostToDevice);
    mergeSort<<<1, 1>>>(a2, a.size());
    cudaMemcpy(a1, a2, a.size() * sizeof(char*), cudaMemcpyDeviceToHost);
    list<string> result;
    for(int i = 0; i < a.size(); i++){
        result.push_back(a1[i]);
    }
    writetofile(result, "test1_result.txt");
    return 0;
}
