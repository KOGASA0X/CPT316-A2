#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <time.h>

/* This structure will hold a doubly-linked list. */
typedef struct node {
    char data[5];         /* A buffer to hold some data. */
    struct node *prev;    /* A pointer to the previous node. */
    struct node *next;    /* A pointer to the next node. */
} node_t;

/* print_list
 * Prints the data from each node in the list to stdout
 * head: pointer to the first node in the list
 */
void print_list(node_t *head) {
    node_t *current = head; // current node pointer

    // loop through the list
    while (current != NULL) {
        printf("%s ", current->data); // print the data
        current = current->next; // move to the next node
    }
}

void push(node_t **head_ref, char new_data[]) {
    // Allocate memory for new node
    node_t *new_node = (node_t *) malloc(sizeof(node_t));

    // Put in the data
    strcpy(new_node->data, new_data);

    // Make next of new node as head and previous as NULL
    new_node->prev = NULL;
    new_node->next = (*head_ref);

    // Change previous of head node to new node
    if ((*head_ref) != NULL)
        (*head_ref)->prev = new_node;

    // Move the head to point to the new node
    (*head_ref) = new_node;
}

void insertAfter(node_t *prev_node, char new_data[]) {
    // Check if the given prev_node is NULL
    if (prev_node == NULL) {
        printf("the given previous node cannot be NULL");
        return;
    }

    // Allocate new node
    node_t *new_node = (node_t *) malloc(sizeof(node_t));

    // Put in the data
    strcpy(new_node->data, new_data);

    // Make next of new node as next of prev_node
    new_node->next = prev_node->next;

    // Make the next of prev_node as new_node
    prev_node->next = new_node;

    // Make prev_node as previous of new_node
    new_node->prev = prev_node;

    // Change previous of new_node's next node
    if (new_node->next != NULL)
        new_node->next->prev = new_node;
}

void append(node_t **head_ref, char new_data[]) {
    // Allocate memory for the new node, and put data in it.
    node_t *new_node = (node_t *) malloc(sizeof(node_t));
    strcpy(new_node->data, new_data);

    // This new node is going at the end, so make next point to NULL.
    new_node->next = NULL;

    // If the Linked List is empty, then make the new node head
    node_t *last = *head_ref; // Used in step 5
    if (*head_ref == NULL) {
        new_node->prev = NULL;
        *head_ref = new_node;
        return;
    }

    // Else traverse till the last node
    while (last->next != NULL)
        last = last->next;

    // Change the next of last node
    last->next = new_node;

    // Make last node as previous of new node
    new_node->prev = last;
    return;
}

void deleteNode(node_t **head_ref, node_t *del) {
    if (*head_ref == NULL || del == NULL)
        return;

    // If node to be deleted is head node
    if (*head_ref == del)
        *head_ref = del->next;

    // Change next only if node to be deleted
    // is NOT the last node
    if (del->next != NULL)
        del->next->prev = del->prev;

    // Change prev only if node to be deleted
    // is NOT the first node
    if (del->prev != NULL)
        del->prev->next = del->next;

    free(del);
    return;
}

void deleteList(node_t **head_ref) {
    // save current head
    node_t *current = *head_ref;
    node_t *next;

    while (current != NULL) {
        // save next node
        next = current->next;
        // delete current node
        free(current);
        current = next;
    }

    // set head to NULL
    *head_ref = NULL;
}

node_t* mergeTwoLists(node_t* l1, node_t* l2) {
    node_t* head = NULL; // head of the list to return
    node_t* tail = NULL; // the last node in the list we're building
    while (l1 != NULL && l2 != NULL) {
        // if the first node in l1 comes before the first node in l2
        if (strcmp(l1->data, l2->data) < 0) {
            // if the head of the list we're building is empty
            if (head == NULL) {
                // set the head to l1
                head = l1;
                // set the tail to l1
                tail = l1;
            }
            else {
                // otherwise, set the tail's next to l1
                tail->next = l1;
                // set l1's prev to the tail
                l1->prev = tail;
                // set the tail to l1
                tail = l1;
            }
            // set l1 to the next node
            l1 = l1->next;
        }
        else {
            // otherwise, if the head of the list we're building is empty
            if (head == NULL) {
                // set the head to l2
                head = l2;
                // set the tail to l2
                tail = l2;
            }
            else {
                // otherwise, set the tail's next to l2
                tail->next = l2;
                // set l2's prev to the tail
                l2->prev = tail;
                // set the tail to l2
                tail = l2;
            }
            // set l2 to the next node
            l2 = l2->next;
        }
    }
    // if l1 is not empty
    if (l1 != NULL) {
        // set the tail's next to l1
        tail->next = l1;
        // set l1's prev to the tail
        l1->prev = tail;
    }
    // if l2 is not empty
    if (l2 != NULL) {
        // set the tail's next to l2
        tail->next = l2;
        // set l2's prev to the tail
        l2->prev = tail;
    }
    // return the head
    return head;
} 

node_t* mergeSort(node_t* head) // This function is used to sort the linked list
{
    if (head == NULL || head->next == NULL) // If the head is null or the next node is null, return the head
    {
        return head;
    }
    node_t* slow = head; // Define slow pointer
    node_t* fast = head; // Define fast pointer
    while (fast->next != NULL && fast->next->next != NULL) // While the fast pointer's next node is not null and the next node's next node is not null
    {
        slow = slow->next; // Slow pointer moves to the next node
        fast = fast->next->next; // Fast pointer moves to the next node's next node
    }
    node_t* head2 = slow->next; // Define head2 as the slow pointer's next node
    slow->next = NULL; // Set the slow pointer's next node to null
    #pragma omp parallel sections // Parallelize the mergeSort function
    {
        #pragma omp section // Parallelize the first mergeSort function
        {
            head = mergeSort(head); // Sort the first half of the linked list
        }
        #pragma omp section // Parallelize the second mergeSort function
        {
            head2 = mergeSort(head2); // Sort the second half of the linked list
        }
    }
    return mergeTwoLists(head, head2); // Merge the sorted list
}

// readfile reads the contents of a file and returns a linked list
// of the words in the file.  The list will be in the same order as
// the words appeared in the file.
node_t* readfile(char* filename)
{
    // open the file for reading
    FILE* fp = fopen(filename, "r");

    // if the file doesn't exist, print an error message and return NULL
    if (fp == NULL)
    {
        printf("Cannot open file %s", filename);
        return NULL;
    }

    // start with an empty list
    node_t* head = NULL;

    // read words from the file into a buffer until we reach the end of file
    char buffer[5];
    while (fscanf(fp, "%s", buffer) != EOF)
    {
        // add the word to the list
        append(&head, buffer);
    }

    // close the file
    fclose(fp);

    // return the list
    return head;
}

void writetofile(char* filename, node_t* head)
{
    // open the file for writing
    FILE* fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Cannot open file %s", filename);
        return;
    }
    // loop through the linked list and write each node to the file
    while (head != NULL)
    {
        fprintf(fp, "%s\n", head->data);
        head = head->next;
    }
    // close the file
    fclose(fp);
}

void parallalmersort(node_t* head)
{
    // Base case: list is empty or has only one element
    if (head == NULL || head->next == NULL)
    {
        return;
    }
    // Find middle of list
    node_t* slow = head;
    node_t* fast = head;
    while (fast->next != NULL && fast->next->next != NULL)
    {
        slow = slow->next;
        fast = fast->next->next;
    }
    // Split list into two halves
    node_t* head2 = slow->next;
    slow->next = NULL;
    // Recursive calls to sort the two halves
    #pragma omp parallel sections
    {
        #pragma omp section
            parallalmersort(head);
        #pragma omp section
            parallalmersort(head2);
    }
    // Merge the two halves
    head = mergeTwoLists(head, head2);
}


int main(int argc, char const *argv[])
{
    // Read the file into a linked list
    node_t* head = readfile("sgb-words.txt");

    // Start the timer
    clock_t start = clock();

    // Sort the linked list
    head = mergeSort(head);

    // Stop the timer
    clock_t end = clock();

    // Calculate the time taken
    double time = (double)(end - start) / CLOCKS_PER_SEC;

    // Print the time taken
    printf("Time taken: %f", time);

    // Write the sorted linked list to a file
    writetofile("output.txt", head);


    return 0;
}
