#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <time.h>

typedef struct node {
    char data[5];
    struct node *prev;
    struct node *next;
} node_t;

void print_list(node_t *head) {
    node_t *current = head;

    while (current != NULL) {
        printf("%s ", current->data);
        current = current->next;
    }
}

void push(node_t **head_ref, char new_data[]) {
    node_t *new_node = (node_t *) malloc(sizeof(node_t));

    strcpy(new_node->data, new_data);
    new_node->prev = NULL;
    new_node->next = (*head_ref);

    if ((*head_ref) != NULL)
        (*head_ref)->prev = new_node;

    (*head_ref) = new_node;
}

void insertAfter(node_t *prev_node, char new_data[]) {
    if (prev_node == NULL) {
        printf("the given previous node cannot be NULL");
        return;
    }

    node_t *new_node = (node_t *) malloc(sizeof(node_t));

    strcpy(new_node->data, new_data);
    new_node->next = prev_node->next;
    prev_node->next = new_node;
    new_node->prev = prev_node;

    if (new_node->next != NULL)
        new_node->next->prev = new_node;
}

void append(node_t **head_ref, char new_data[]) {
    node_t *new_node = (node_t *) malloc(sizeof(node_t));
    node_t *last = *head_ref;

    strcpy(new_node->data, new_data);
    new_node->next = NULL;

    if (*head_ref == NULL) {
        new_node->prev = NULL;
        *head_ref = new_node;
        return;
    }

    while (last->next != NULL)
        last = last->next;

    last->next = new_node;
    new_node->prev = last;
    return;
}

void deleteNode(node_t **head_ref, node_t *del) {
    if (*head_ref == NULL || del == NULL)
        return;

    if (*head_ref == del)
        *head_ref = del->next;

    if (del->next != NULL)
        del->next->prev = del->prev;

    if (del->prev != NULL)
        del->prev->next = del->next;

    free(del);
    return;
}

void deleteList(node_t **head_ref) {
    node_t *current = *head_ref;
    node_t *next;

    while (current != NULL) {
        next = current->next;
        free(current);
        current = next;
    }

    *head_ref = NULL;
}

node_t* mergeTwoLists(node_t* l1, node_t* l2) {
    node_t* head = NULL;
    node_t* tail = NULL;
    while (l1 != NULL && l2 != NULL) {
        if (strcmp(l1->data, l2->data) < 0) {
            if (head == NULL) {
                head = l1;
                tail = l1;
            }
            else {
                tail->next = l1;
                l1->prev = tail;
                tail = l1;
            }
            l1 = l1->next;
        }
        else {
            if (head == NULL) {
                head = l2;
                tail = l2;
            }
            else {
                tail->next = l2;
                l2->prev = tail;
                tail = l2;
            }
            l2 = l2->next;
        }
    }
    if (l1 != NULL) {
        tail->next = l1;
        l1->prev = tail;
    }
    if (l2 != NULL) {
        tail->next = l2;
        l2->prev = tail;
    }
    return head;
} 

node_t* mergeSort(node_t* head)
{
    if (head == NULL || head->next == NULL)
    {
        return head;
    }
    node_t* slow = head;
    node_t* fast = head;
    while (fast->next != NULL && fast->next->next != NULL)
    {
        slow = slow->next;
        fast = fast->next->next;
    }
    node_t* head2 = slow->next;
    slow->next = NULL;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            head = mergeSort(head);
        }
        #pragma omp section
        {
            head2 = mergeSort(head2);
        }
    }
    return mergeTwoLists(head, head2);
}

node_t* readfile(char* filename)
{
    FILE* fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Cannot open file %s", filename);
        return NULL;
    }
    node_t* head = NULL;
    char buffer[5];
    while (fscanf(fp, "%s", buffer) != EOF)
    {
        append(&head, buffer);
    }
    fclose(fp);
    return head;
}

void writetofile(char* filename, node_t* head)
{
    FILE* fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Cannot open file %s", filename);
        return;
    }
    while (head != NULL)
    {
        fprintf(fp, "%s\n", head->data);
        head = head->next;
    }
    fclose(fp);
}

void parallalmersort(node_t* head)
{
    if (head == NULL || head->next == NULL)
    {
        return;
    }
    node_t* slow = head;
    node_t* fast = head;
    while (fast->next != NULL && fast->next->next != NULL)
    {
        slow = slow->next;
        fast = fast->next->next;
    }
    node_t* head2 = slow->next;
    slow->next = NULL;
    #pragma omp parallel sections
    {
        #pragma omp section
            parallalmersort(head);
        #pragma omp section
            parallalmersort(head2);
    }
    head = mergeTwoLists(head, head2);
}

int main(int argc, char const *argv[])
{

    node_t* head = readfile("sgb-words.txt");

    clock_t start = clock();
    head = mergeSort(head);
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %f", time);

    writetofile("output.txt", head);


    return 0;
}
