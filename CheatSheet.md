# <font face='楷体'>CheatSheet

<font size=3>

## 考试注意

- 思考时
  1. 确定题型
  2. 考虑时间复杂度
  3. 若无更简单的方法，则先试一试笨方法
  4. 当前题目较难时先跳过
- Debug时
  1. 确认看清楚题目
  2. 优先看细节是否出错
  3. 若细节无问题则考虑算法是否出错
  4. 多在本地上跑一跑，本地运行时尽量把过程显现出来
- 常见细节错误：
  1. 数组越界
  2. 图中有重边
  3. 没有按要求格式输出
  4. 可能有多组数据
  5. 某一处打字错误

## 排序

1.BubbleSort

```python
# Optimized Python program for implementation of Bubble Sort
def bubbleSort(arr):
	n = len(arr)
	
    # Traverse through all array elements
    for i in range(n):
	swapped = False
		
        # Last i elements are already in place
		for j in range(0, n - i - 1):
			
            # Traverse the array from 0 to n-i-1
			# Swap if the element found is greater
			# than the next element
			if arr[j] > arr[j + 1]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]
				swapped = True
        if (swapped == False):
			break
```

2.QuickSort

```python
def quicksort(arr, left, right):
	if left < right:
		partition_pos = partition(arr, left, right)
		quicksort(arr, left, partition_pos - 1)
		quicksort(arr, partition_pos + 1, right)

def partition(arr, left, right):
	i = left
	j = right - 1
	pivot = arr[right]
	while i <= j:
		while i <= right and arr[i] < pivot:
			i += 1
		while j >= left and arr[j] >= pivot:
			j -= 1
		if i < j:
			arr[i], arr[j] = arr[j], arr[i]
	if arr[i] > pivot:
		arr[i], arr[right] = arr[right], arr[i]
    return i
```

3.MergeSort

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2
        
        L = arr[:mid] # Dividing the array elements
        R = arr[mid:] # Into 2 halves
        
        mergeSort(L) # Sorting the first half
        mergeSort(R) # Sorting the second half
        
        i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
				k += 1
		
        # Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1
		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
```

## 前缀和

### 构造乘积矩阵

下面代码利用==前缀和==和==后缀和==算出一个矩阵中子矩阵的所有数的乘积。

```python
class Solution(object):
    def constructProductMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: List[List[int]]
        """
        mod = 12345
        n = len(grid)
        m = len(grid[0])
        N = n * m
        # 将二维矩阵展平成一维数组
        arr = []
        for row in grid:
            arr.extend(row)

        # 计算前缀乘积数组
        prefix = [0] * N
        prefix[0] = arr[0] % mod
        for i in range(1, N):
            prefix[i] = (prefix[i - 1] * arr[i]) % mod

        # 计算后缀乘积数组
        suffix = [0] * N
        suffix[-1] = arr[-1] % mod
        for i in range(N - 2, -1, -1):
            suffix[i] = (suffix[i + 1] * arr[i]) % mod

        # 计算结果数组：对于位置 i, 结果为 (前缀[i-1] * 后缀[i+1]) % mod
        res = [0] * N
        for i in range(N):
            left = prefix[i - 1] if i > 0 else 1
            right = suffix[i + 1] if i < N - 1 else 1
            res[i] = (left * right) % mod

        # 将结果数组还原成 n*m 的矩阵
        ans = []
        idx = 0
        for i in range(n):
            row = []
            for j in range(m):
                row.append(res[idx])
                idx += 1
            ans.append(row)
        return ans
```

## KMP算法

```python
""""
compute_lps 函数用于计算模式字符串的LPS表。LPS表是一个数组，
其中的每个元素表示模式字符串中当前位置之前的子串的最长前缀后缀的长度。
该函数使用了两个指针 length 和 i，从模式字符串的第二个字符开始遍历。
"""
def compute_lps(pattern):
    """
    计算pattern字符串的最长前缀后缀（Longest Proper Prefix which is also Suffix）表
    :param pattern: 模式字符串
    :return: lps表
    """

    m = len(pattern)
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]    # 跳过前面已经比较过的部分
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length
    return lps


def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    lps = compute_lps(pattern)
    matches = []

    j = 0  # j是pattern的索引
    for i in range(n):  # i是text的索引
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)
            j = lps[j - 1]
    return matches
```

### 前缀中的周期

本题寻找一个字符串各个位前的最大循环节。

```python
"""
这是一个字符串匹配问题，通常使用KMP算法（Knuth-Morris-Pratt算法）来解决。
使用了 Knuth-Morris-Pratt 算法来寻找字符串的所有前缀，并检查它们是否由重复的子串组成，如果是的话，就打印出前缀的长度和最大重复次数。
"""

# 得到字符串s的前缀值列表
def kmp_next(s):
  	# kmp算法计算最长相等前后缀
    next = [0] * len(s)
    j = 0
    for i in range(1, len(s)):
        while s[i] != s[j] and j > 0:
            j = next[j - 1]
        if s[i] == s[j]:
            j += 1
        next[i] = j
    return next


def main():
    case = 0
    while True:
        n = int(input().strip())
        if n == 0:
            break
        s = input().strip()
        case += 1
        print("Test case #{}".format(case))
        next = kmp_next(s)
        for i in range(2, len(s) + 1):
            k = i - next[i - 1]		# 可能的重复子串的长度
            if (i % k == 0) and i // k > 1:
                print(i, i // k)
        print()


if __name__ == "__main__":
    main()
```



## 差分数组

### 零数组变换I

```python
class Solution(object):
    def isZeroArray(self, nums, queries):
        """
        :type nums: List[int]
        :type queries: List[List[int]]
        :rtype: bool
        """
        n = len(nums)
        delta = [0] * (n + 1)
        for l, r in queries:
            delta[l] += 1
            delta[r + 1] += -1
        
        prefix = [0] * n
        prefix[0] = delta[0]
        if prefix[0] < nums[0]:
            return False
        for i in range(1, n):
            prefix[i] = prefix[i - 1] + delta[i]
            if prefix[i] < nums[i]:
                return False
        return True
```

## 中序表达式转后序表达式

```python
def infix_to_postfix(expression):
    stack=[]
    postfix=[]
    number=''

    for char in expression:
        if char not in '+-*/()':
            number+=char
        else:
            if number:
                postfix.append(number)
                number=''

            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char]<=precedence[stack[-1]]:
                    symbol=stack.pop()
                    postfix.append(symbol)
                stack.append(char)
            elif char=='(':
                stack.append(char)
            elif char==')':
                while stack and stack[-1]!='(':
                    symbol=stack.pop()
                    postfix.append(symbol)
                stack.pop()
    
    if number:
        postfix.append(number)

    while stack:
        symbol=stack.pop()
        postfix.append(symbol)
    
    return ' '.join(postfix)

n=int(input())
for _ in range(n):
    expression=input()
    print(infix_to_postfix(expression))
```

## 单调栈

### 柱状图中最大矩阵

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        n = len(heights)
        left, right = [0] * n, [0] * n

        stack = []
        for i in range(n):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if stack:
                left[i] = stack[-1] + 1
            else:
                left[i] = 0
            stack.append(i)
        
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if stack:
                right[i] = stack[-1] - 1
            else:
                right[i] = n - 1
            stack.append(i)
        
        maxi = 0
        for i in range(n):
            maxi = max(maxi, heights[i] * (right[i] - left[i] + 1))
        
        return maxi
```

## 链表

1.单链表反转

```python
def ReverseLinkedList(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```

2.合并两个有序链表

```python
def MergeSortedLists(l1,l2):
    dummy = Node(0) # 设置一个虚拟表头，最后返回dummy.next即可
    cur = dummy
    while l1 and l2:
        if l1.val >= l2:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if l1:
        cur.next = l1
    elif l2:
        cur.next = l2
    return dummy.next
```

3.查找链表中间节点

```python
def find_middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

## Huffman编码树

```python
import heapq

class Node:
    
    def __init__(self, char, frequency):
        self.char = char
        self.freq = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, another_node):
        return self.freq < another_node.freq
    

def huffman_encoding(char_freq):
    """
    char_freq[dict]:字典，储存键值对char-freq,表示键char的出现频率freq
    """
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]
```

## 通用深度优先搜索

```python
from collections import defaultdict

class Vertex:
    
    def __init__(self, key):
        self.key = key
        self.color = 'white' # 用顶点颜色记录访问情况
        self.previous = None
        self.discovery_time = 0 # 顶点开始搜索的时间
        self.closing_time = 0 # 顶点结束搜索的时间

class DFSGraph:
    def __init__(self):
        self.vertices = dict() # 储存所有顶点，键为key，值为Vertex
        self.adjList = defaultdict(list) # 邻接表表示图
        self.time = 0
    
    def dfs(self):
        for vertex in self.vertices.values():
            vertex.color = 'white'
            vertex.previous = None
        for vertex in self.vertices.values():
            if vertex.color = 'white':
                self.dfs_visit(vertex)
    
    def dfs_visit(self, start):
        start.color = 'gray'
        self.time += 1
        start.discovery_time = self.time
        for neighbor in self.adjList[start]:
            if neighbor.color == 'white':
                neighbor.previous = start
                self.dfs_visit(neighbor)
        start.color = 'black'
        self.time += 1
        start.closing_time = self.time
```

## Trie

### 电话号码

本题查询一个电话号码是否是前面号码的前缀。

```python
class Node:

    def __init__(self):
        self.children = dict()

class Trie:

    def __init__(self):
        self.root = Node()
    
    def InsertNumbers(self, number):
        current = self.root
        for x in number:
            if x not in current.children:
                current.children[x] = Node()
            current = current.children[x]
    
    def Search(self, number):
        current = self.root
        for x in number:
            if x not in current.children:
                return False
            current = current.children[x]
        
        return True

T = int(input())
for _ in range(T):
    n = int(input())
    numbers = [input() for _ in range(n)]
    numbers.sort(reverse = True)

    trie = Trie()
    flag = True
    for number in numbers:
        if trie.Search(number):
            flag = False
            break

        trie.InsertNumbers(number)
    
    if flag:
        print('YES')
    else:
        print('NO')
```

## Prim(最小生成树)

```python
def prim(N, graph):
    connected = set()
    heap = []
    
    connected.add(0)
    for j in range(1, N):
        heapq.heappush(heap, (graph[0][j], 0, j))
    
    cnt, s = 1, 0
    while cnt < N:
        w, u, v = heapq.heappop(heap)
        if v not in connected:
            connected.add(v)
            cnt += 1
            s += w
        for neighbor in range(0, N):
            if neighbor in connected:
                continue
            heapq.heappush(heap, (graph[v][neighbor], v, neighbor))
    
    return s
```

## Disjoint Set

```python
class DisjointSet:
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1

# 用并查集实现Kruskal算法解决最小生成树问题
def kruskal(graph):
    num_vertices = len(graph)
    edges = []

    # 构建边集
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    # 按照权重排序
    edges.sort(key=lambda x: x[2])

    # 初始化并查集
    disjoint_set = DisjointSet(num_vertices)

    # 构建最小生成树的边集
    minimum_spanning_tree = []

    for edge in edges:
        u, v, weight = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            minimum_spanning_tree.append((u, v, weight))

    return minimum_spanning_tree
```

## 强连通单元(Kosaraju算法)

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)

def kosaraju(graph):
    # Step 1: Perform first DFS to get finishing times
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    
    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    
    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs
```

