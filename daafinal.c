/*time o(v+e)
Type: Graph traversal, level-order traversal.*/
#include <stdio.h>  
#include <time.h> 
#define MAX 100  
int adj[MAX][MAX], visited[MAX], queue[MAX], front = 0, rear = -1, n;  

void bfs(int start) {  
    visited[start] = 1;   
    queue[++rear] = start;  
    while (front <= rear) {  
        int city = queue[front++];  
        printf("%d ", city);  
        for (int i = 0; i < n; i++)  
            if (adj[city][i] && !visited[i]) 
                queue[++rear] = i, visited[i] = 1;  
    }  }  

int main() {  
    int start;  
    printf("Enter number of cities: ");  
    scanf("%d", &n);  
    printf("Enter adjacency matrix:\n");  
    for (int i = 0; i < n; i++)  
        for (int j = 0; j < n; j++)  
            scanf("%d", &adj[i][j]);  

    printf("Enter starting city: ");  
    scanf("%d", &start);  
    clock_t start_time = clock();  // Start timing  
    printf("Cities reachable from %d are: ", start);  
    bfs(start);  
    printf("\nTime taken for BFS: %f seconds\n", 
           (double)(clock() - start_time) / CLOCKS_PER_SEC);  
    return 0;  
}

/*Enter number of cities: 5
Enter adjacency matrix:
0 1 1 1 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
Enter starting city: 0
Cities reachable from 0 are: 0 1 2 3 4
Time taken for BFS: 0.002000 seconds*/




/*time o(v+e)
 Graph traversal, backtracking.*/
#include <stdio.h>  
#include <time.h>  
#define MAX 100  

int adj[MAX][MAX], visited[MAX], n;  

void dfs(int start) {  
    printf("%d ", start);  
    visited[start] = 1;  
    for (int i = 0; i < n; i++)  
        if (adj[start][i] && !visited[i])  
            dfs(i);  
}  

int main() {  
    int start;  
    printf("Enter number of lands: ");  
    scanf("%d", &n);  

    printf("Enter adjacency matrix:\n");  
    for (int i = 0; i < n; i++)  
        for (int j = 0; j < n; j++)  
            scanf("%d", &adj[i][j]);  

    printf("Enter starting land: ");  
    scanf("%d", &start);  
    clock_t start_time = clock();  
    printf("Lands reachable from %d are: ", start);  
    dfs(start);  

    printf("\nTime taken for DFS: %f seconds\n", 
           (double)(clock() - start_time) / CLOCKS_PER_SEC);  

    return 0;  
}

/*Enter number of lands: 5
Enter adjacency matrix:
0 1 1 1 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0 
Enter starting land: 0
Lands reachable from 0 are: 0 1 4 2 3*/




/*Time Complexity: O(log n)
Divide-and-conquer, searching.*/
#include <stdio.h>  
#include <time.h>  

int binarySearch(int arr[], int size, int key) {  
    int low = 0, high = size - 1;  
    while (low <= high) {  
        int mid = (low + high) / 2;  
        if (arr[mid] == key) return mid;  
        (arr[mid] < key) ? (low = mid + 1) : (high = mid - 1);  
    }  
    return -1;  
}  
int main() {  
    int n, productID;  
    printf("Enter number of products: ");  
    scanf("%d", &n);  

    int products[n];  
    printf("Enter sorted product IDs:\n");  
    for (int i = 0; i < n; i++) scanf("%d", &products[i]);  

    printf("Enter product ID to search: ");  
    scanf("%d", &productID);  

    clock_t start_time = clock();  
    int result = binarySearch(products, n, productID);  
    printf(result == -1 ? "Product not available\n" : "Product available at index %d\n", result + 1);  
    printf("Time taken: %f seconds\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);  

    return 0;  
}

/*
Enter the number of Products: 10

Sorted Product ID List is:  
 1  2   3   4   5   6   7   10  15  20

Enter the Product ID to be searched: 15
Product found!!
Product 15 found in position 9

Time taken to search is 0.934521
*/
/*Average-case: O(n^2)
Comparison-based sorting, incremental.*/
#include <stdio.h>  
#include <stdlib.h>  
#include <time.h>  



void insertionSort(long long arr[], int n) {  
    for (int i = 1; i < n; i++) {  
        long long key = arr[i];  
        int j = i - 1;  
        while (j >= 0 && arr[j] > key) arr[j + 1] = arr[j--];  
        arr[j + 1] = key;  
    }  }  
    
int main() {  
    int n;  
    printf("Enter number of phone numbers: ");  
    scanf("%d", &n);  
    long long phones[n];  

    srand(time(0));  
    for (int i = 0; i < n; i++)  
        phones[i] = (7 + rand() % 3) * 1000000000LL + rand() % 100000000;  

    clock_t start_time = clock();  
    insertionSort(phones, n);  
    printf("Sorted phone numbers:\n");  
    for (int i = 0; i < n; i++) printf("%lld\n", phones[i]);  
    printf("Time taken: %f seconds\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);  

    return 0;  
}

/*Enter the number of phone numbers: 6
Unsorted phone numbers:
8000016741
9000026698
7000006773
9000021437
7000032707
9000008423

Sorted phone numbers:
7000006773
7000032707
8000016741
9000008423
9000021437
9000026698

Time taken for sorting: 0.000000 seconds*/



/*Average-case: O(n log n),*/
/*Worst-case: O(n^2)
Type: Divide-and-conquer, comparison-based sorting.*/
#include <stdio.h> 
#include <stdlib.h>  
#include <time.h>  

int partition(int a[], int low, int high) {  
    int pivot = a[low], i = low + 1, j = high;  
    while (1) {  
        while (i <= high && a[i] <= pivot) i++;  
        while (j > low && a[j] > pivot) j--;  
        if (i < j) { int temp = a[i]; a[i] = a[j]; a[j] = temp; }  
        else { a[low] = a[j]; a[j] = pivot; return j; }  
    }  }  

void quick_sort(int a[], int low, int high) {  
    if (low < high) {  
        int j = partition(a, low, high);  
        quick_sort(a, low, j - 1);  
        quick_sort(a, j + 1, high);  
    }  }  

int main() {  
    int n, a[200000];  
    printf("Enter number of student records: ");  
    scanf("%d", &n);  
    if (n > 200000) return 1; // Error check

    srand(time(NULL));  
    for (int i = 0; i < n; i++) a[i] = rand() % 100;  
    printf("Roll numbers: ");  
    for (int i = 0; i < n; i++) printf("%d ", a[i]);  

    clock_t start = clock();  
    quick_sort(a, 0, n - 1);  
    printf("\nSorted roll numbers: ");  
    for (int i = 0; i < n; i++) printf("%d ", a[i]);  
    printf("\nRun time: %f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);  

    return 0;  
}

/*Enter the number of student records (max 200000): 
5
The roll numbers are:
94 43 44 41 36 

Sorted roll numbers are:
36 41 43 44 94 
The run time is 0.000000 seconds*/



/*Time Complexity: O(n log n)
Comparison-based sorting, divide-and-conquer.*/
#include <stdio.h>  
#include <time.h>  

void heapify(int h[], int n, int i) {  
    int largest = i, left = 2 * i + 1, right = 2 * i + 2;  
    if (left < n && h[left] > h[largest]) largest = left;  
    if (right < n && h[right] > h[largest]) largest = right;  
    if (largest != i) {  
        int temp = h[i]; h[i] = h[largest]; h[largest] = temp;  
        heapify(h, n, largest);  
    }  }  

void heapsort(int h[], int n) {  
    for (int i = n / 2 - 1; i >= 0; i--) heapify(h, n, i);  
    for (int i = n - 1; i > 0; i--) {  
        int temp = h[0]; h[0] = h[i]; h[i] = temp;  
        heapify(h, i, 0);  
    }  }  

int main() {  
    int n, h[20];  
    printf("Enter number of resumes: ");  
    scanf("%d", &n);  
    if (n > 20) return 1;  
   printf("Enter resumes: ");  
    for (int i = 0; i < n; i++) scanf("%d", &h[i]);  

    clock_t start = clock();  
    heapsort(h, n);  
    printf("Sorted ranks:\n");  
    for (int i = 0; i < n; i++) printf("%d\t", h[i]);  
    printf("\nRun time: %f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);  

    return 0;  
}

/*Enter the number of resumes (max 20): 6
Enter the rank for candidate 1: 2
Enter the rank for candidate 2: 4
Enter the rank for candidate 3: 3
Enter the rank for candidate 4: 5
Enter the rank for candidate 5: 78
Enter the rank for candidate 6: 54

The ranks in sorted order:
2       3       4       5       54      78
The run time is 0.000000 seconds*/



/*Horspool's Algorithm is an
 efficient string-matching algorithm*/
#include <stdio.h>
#include <string.h>
#define MAX 256

void shifttable(char p[], int t[]) {
    for (int i = 0; i < MAX; i++) t[i] = strlen(p);
    for (int j = 0; j < strlen(p) - 1; j++) t[(unsigned char)p[j]] = strlen(p) - 1 - j;
}
int horspool(char src[], char p[], int t[]) {
    for (int i = strlen(p) - 1; i < strlen(src);) {
        int k = 0;
        while (k < strlen(p) && p[strlen(p) - 1 - k] == src[i - k]) k++;
        if (k == strlen(p)) return i - k + 1;
        i += t[(unsigned char)src[i]];
    }
    return -1;
}
int main() {
    char src[100], p[100];
    int t[MAX];

    // Input text
    printf("Enter text: ");
    fgets(src, 100, stdin);
    src[strcspn(src, "\n")] = '\0';  // Remove newline character from input

    // Input pattern
    printf("Enter pattern: ");
    fgets(p, 100, stdin);
    p[strcspn(p, "\n")] = '\0';  // Remove newline character from input

    // Build shift table and perform the search
    shifttable(p, t);
    int pos = horspool(src, p, t);

    // Output the result
    if (pos >= 0)
        printf("Pattern found at position %d\n", pos + 1);
    else
        printf("Pattern not found\n");

    return 0;
}

/*Enter the text in which pattern is to be searched:
TTATAGATCTCGTATTCTTTTATAGATCTCCTATTCTT.
Enter the pattern to be searched:
TATT

The desired pattern was found starting from position 13 */



#include <stdio.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}
void knapsack(int W, int wt[], int val[], int n) {
    int K[n + 1][W + 1];

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0)
                K[i][w] = 0;
            else if (wt[i - 1] <= w)
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
            else
                K[i][w] = K[i - 1][w];
        }}
    printf("Maximum value of products that can be collected: %d\n", K[n][W]);

    printf("Selected products (product index starts from 1):\n");
    int w = W;
    for (int i = n; i > 0 && w > 0; i--) {
        if (K[i][w] != K[i - 1][w]) {
            printf("Product %d (Value: %d, Weight: %d)\n", i, val[i - 1], wt[i - 1]);
            w -= wt[i - 1];
        }}}

int main() {
    int n, W;

    printf("Enter the number of products: ");
    scanf("%d", &n);

    printf("Enter the weight capacity of the basket (in kg): ");
    scanf("%d", &W);

    int val[n], wt[n];

    printf("Enter the values and weights of the products:\n");
    for (int i = 0; i < n; i++) {
        printf("Product %d - Value and Weight: ", i + 1);
        scanf("%d %d", &val[i], &wt[i]);
    }
    knapsack(W, wt, val, n);

    return 0;
}
/*Enter the number of products: 5
Enter the weight capacity of the basket (in kg): 15
Enter the values and weights of the products:
Product 1 - Value and Weight: 1000 5
Product 2 - Value and Weight: 500 4
Product 3 - Value and Weight: 1500 6
Product 4 - Value and Weight: 700 3
Product 5 - Value and Weight: 900 2

Maximum value of products that can be collected: 2900
Selected products (product index starts from 1):
Product 5 (Value: 900, Weight: 2)
Product 4 (Value: 700, Weight: 3)
Product 3 (Value: 1500, Weight: 6)
*/







/*greedy algo
Time Complexity:
Sorting the edges takes O(E log E)*/
#include <stdio.h>  
typedef struct { int u, v, cost; } edge;  

int find(int v, int parent[]) {  
    return (parent[v] == v) ? v : find(parent[v], parent);  
}  
void union_ij(int i, int j, int parent[]) {  
    parent[j] = i;  
}  
void kruskal(int n, edge e[], int m) {  
    int parent[n], sum = 0, count = 0, t[n - 1][2];  
    for (int i = 0; i < n; i++) parent[i] = i;  
 
    for (int i = 0; i < m - 1; i++)  
        for (int j = 0; j < m - 1 - i; j++)  
            if (e[j].cost > e[j + 1].cost) {  
                edge temp = e[j]; e[j] = e[j + 1]; e[j + 1] = temp;}  

    for (int i = 0; count < n - 1 && i < m; i++) {  
        int root_u = find(e[i].u, parent), root_v = find(e[i].v, parent);  
        if (root_u != root_v) {  
            t[count][0] = e[i].u; t[count][1] = e[i].v;  
            union_ij(root_u, root_v, parent); 
            sum += e[i].cost; 
            count++;  
        }  }  
    if (count == n - 1) {  
        printf("Spanning tree exists\nEdges:\n");  
        for (int i = 0; i < n - 1; i++) printf("%d %d\n", t[i][0], t[i][1]);  
        printf("Cost: %d\n", sum);  
    } else {  
        printf("Spanning tree does not exist\n");  
    }  
}  

int main() {  
    int n, m;  
    edge e[20];  
    printf("Enter vertices and edges: ");  
    scanf("%d %d", &n, &m);  
    printf("Enter edges (u v cost):\n");  
    for (int i = 0; i < m; i++) scanf("%d %d %d", &e[i].u, &e[i].v, &e[i].cost);  
    kruskal(n, e, m);  
    return 0;  
}

/*Enter number of vertices and edges: 4 5
Enter the edge list (u v cost):
0 1 10
0 2 6
0 3 5
1 3 15
2 3 4
Spanning tree exists
The spanning tree is:
2  3    0  3    0  1
Cost of the spanning tree: 19*/


/*Dijkstra's Algorithm is a famous algorithm used to find the shortest path from a source node to all other 
nodes in a graph with non-negative edge weights Using Priority Queue: O((V + E) log V), */
#include <stdio.h>  
#include <time.h>  
#define MAX 10  
#define INF 999  

int choose(int dist[], int s[], int n) {  
    int min = INF, j = -1;  
    for (int w = 1; w <= n; w++)  
        if (dist[w] < min && !s[w]) { min = dist[w]; j = w; }  
    return j;   
}  
void spath(int v, int cost[][MAX], int dist[], int n) {  
    int s[MAX] = {0};  
    for (int i = 1; i <= n; i++) dist[i] = cost[v][i];  
    s[v] = 1; dist[v] = 0;  

    for (int num = 2; num <= n; num++) {  
        int u = choose(dist, s, n);  
        s[u] = 1;  
        for (int w = 1; w <= n; w++)  
            if (dist[u] + cost[u][w] < dist[w] && !s[w])  
                dist[w] = dist[u] + cost[u][w];  
    }}  

int main() {  
    int cost[MAX][MAX], dist[MAX], n, v;  
    clock_t starttime, endtime;  

    printf("Enter number of vertices: ");  
    scanf("%d", &n);  
    printf("Enter adjacency matrix:\n");  
    for (int i = 1; i <= n; i++)  
        for (int j = 1; j <= n; j++)  
            scanf("%d", &cost[i][j]);  

    printf("Enter source vertex: ");  
    scanf("%d", &v);  

    starttime = clock();  
    spath(v, cost, dist, n);  
    endtime = clock();  

    printf("Shortest distances from vertex %d:\n", v);  
    for (int i = 1; i <= n; i++)  
        printf("%d to %d = %d\n", v, i, dist[i]);  

    printf("Time taken: %f seconds\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);  
    return 0;  
}

/*Enter number of vertices: 5
Enter the cost of adjacency matrix:
999     4       2       999     8
999     999     999     4       5
999     999     999     1       999
999     999     999     999     3
999     999     999     999     999
Enter the source vertex: 1
Shortest distances from vertex 1:
1 to 1 = 0
1 to 2 = 4
1 to 3 = 2
1 to 4 = 3
1 to 5 = 6
The time taken is 0.000000 seconds*/
/*backtracking
The worst-case time complexity is O(N!)    */



#include <stdio.h>  
#include <stdlib.h>  
int x[20];  
int place(int k, int i) {  
    for (int j = 1; j < k; j++)  
        if (x[j] == i || abs(x[j] - i) == abs(j - k)) return 0;  
    return 1;  
}  
void nqueens(int k, int n) {  
    for (int i = 1; i <= n; i++) {  
        if (place(k, i)) {  
            x[k] = i;  
            if (k == n) {  
                for (int a = 1; a <= n; a++) printf("%d\t", x[a]);  
                printf("\n");  
            } else nqueens(k + 1, n);  
        } }}  

int main() {  
    int n;  
    printf("Enter number of queens: ");  
    scanf("%d", &n);  
    nqueens(1, n);  
    return 0;  
}

/*Enter the number of queens: 4
The solutions to the N Queens problem are:
2       4       1       3
3       1       4       2*/


/*backtracking
Time complexity: O(2^n)*/
#include <stdio.h>  
int count = 0, w[10], d, x[10];  
void subset(int cs, int k) {  
    if (cs == d) {  
        printf("\nSubset solution = %d\n", ++count);  
        for (int i = 0; i < k; i++)  
            if (x[i]) printf("%d\n", w[i]);  
        return;  
    }  
    if (k < 10) {  
        if (cs + w[k] <= d) {  
            x[k] = 1; subset(cs + w[k], k + 1);  
        }  
        x[k] = 0; subset(cs, k + 1);  
    }  }  

int main() {  
    int n, sum = 0;  
    printf("Enter number of elements: ");  
    scanf("%d", &n);  
    printf("Enter elements in ascending order:\n");  
    for (int i = 0; i < n; i++) {  
        scanf("%d", &w[i]);  
        sum += w[i];  
    }  
    printf("Enter required sum: ");  
    scanf("%d", &d);  

    if (sum < d) {  
        printf("No solution exists.\n");  
    } else {  
        printf("The solution is:\n");  
        subset(0, 0);  
    }  
    return 0;  
}

/*Enter number of elements: 5
Enter elements in ascending order:
1 2 3 4 5
Enter required sum: 6
The solution is:

Subset solution = 1
1
2
3

Subset solution = 2
1
5

Subset solution = 3
2
4*/



/*divide conuer
Best, Worst, and Average Case: O(n log n)  */
#include <stdio.h>  
#include <time.h>  

void merge(int arr[], int l, int m, int r) {  
    int n1 = m - l + 1, n2 = r - m, L[n1], R[n2];  
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];  
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];  

    for (int i = 0, j = 0, k = l; k <= r; k++) {  
        if (j == n2 || (i < n1 && L[i] <= R[j])) arr[k] = L[i++];  
        else arr[k] = R[j++];  
    }  }  
void mergeSort(int arr[], int l, int r) {  
    if (l < r) {  
        int m = l + (r - l) / 2;  
        mergeSort(arr, l, m);  
        mergeSort(arr, m + 1, r);  
        merge(arr, l, m, r);  
    }  }  
int main() {  
    int n;  
    printf("Enter number of employees: ");  
    scanf("%d", &n);  
    int ids[n];  

    printf("Enter employee IDs:\n");  
    for (int i = 0; i < n; i++) {  
        printf("ID for Employee %d: ", i + 1);  
        scanf("%d", &ids[i]);  
    }  

    clock_t start_time = clock();  
    mergeSort(ids, 0, n - 1);  
    double time_taken = (double)(clock() - start_time) / CLOCKS_PER_SEC;  

    printf("Sorted employee IDs:\n");  
    for (int i = 0; i < n; i++) printf("%d\n", ids[i]);  
    printf("Time taken for sorting: %f seconds\n", time_taken);  
    return 0;  
}

/*Enter number of employees: 3
Enter employee IDs:
ID for Employee 1: 34
ID for Employee 2: 53
ID for Employee 3: 43
Sorted employee IDs:
34
43
53
Time taken for sorting: 0.000000 seconds */

