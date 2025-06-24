# <font face='楷体' size=4>

## 1.快慢指针查找链表中间位置

### 回文链表
linked-list, https://leetcode.cn/problems/palindrome-linked-list/

给你一个单链表的头节点 `head` ，请你判断该链表是否为

回文链表（**回文** 序列是向前和向后读都相同的序列。如果是，返回 `true` ；否则，返回 `false` 。



 

**示例 1：**

<img src="https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg" alt="img" style="zoom:67%;" />

```
输入：head = [1,2,2,1]
输出：true
```

**示例 2：**

<img src="https://assets.leetcode.com/uploads/2021/03/03/pal2linked-list.jpg" alt="img" style="zoom:67%;" />

```
输入：head = [1,2]
输出：false
```

 

**提示：**

- 链表中节点数目在范围`[1, 105]` 内
- `0 <= Node.val <= 9`

**进阶**:你能否用 `O(n)` 时间复杂度和 `O(1)` 空间复杂度解决此题？

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        
        # 1. 使用快慢指针找到链表的中点
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # 2. 反转链表的后半部分
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node
        
        # 3. 对比前半部分和反转后的后半部分
        left, right = head, prev
        while right:  # right 是反转后的链表的头
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True

```

## 2.二分查找法

### 月度开销
http://cs101.openjudge.cn/practice/04135

**描述**
农夫约翰是一个精明的会计师。他意识到自己可能没有足够的钱来维持农场的运转了。他计算出并记录下了接下来 N (1 ≤ N ≤ 100,000) 天里每天需要的开销。

约翰打算为连续的M (1 ≤ M ≤ N) 个财政周期创建预算案，他把一个财政周期命名为fajo月。每个fajo月包含一天或连续的多天，每天被恰好包含在一个fajo月里。

约翰的目标是合理安排每个fajo月包含的天数，使得开销最多的fajo月的开销尽可能少。

**输入**
第一行包含两个整数N,M，用单个空格隔开。
接下来N行，每行包含一个1到10000之间的整数，按顺序给出接下来N天里每天的开销。

**输出**
一个整数，即最大月度开销的最小值。
样例输入

>7 5
>100
>400
>300
>100
>500
>101
>400

样例输出
>500

提示
若约翰将前两天作为一个月，第三、四两天作为一个月，最后三天每天作为一个月，则最大月度开销为500。其他任何分配方案都会比这个值更大。

思路:
>用二分查找解决。若可以让最大开支不超过$y$,那么对$x \geq y$,也可以让最大开支不超过$x$.初始化答案上下界mini,maxi=max(costs),10000*N.计算每个月不超过mid需要将的月数，若不超过$M$,则令maxi=mid,ans=mid.否则说明mid大于答案，令mini=mid+1.当mini>=maxi时返回ans即可。

代码：

```python
N,M=map(int,input().split())
costs=[int(input()) for _ in range(N)]

def check(max_cost):
    months=1
    current=0
    for c in costs:
        if current+c > max_cost:
            months+=1
            current=c
        else:
            current+=c
    return months<=M

mini,maxi=max(costs),10000*N
ans=maxi
while mini < maxi:
    mid=(mini+maxi)//2
    if check(mid):
        maxi=mid
        ans=maxi
    else:
        mini=mid+1

print(ans)
```

### Minimum Limit of Balls in a Bag
 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/

You are given an integer array nums where the ith bag contains nums[i] balls. You are also given an integer maxOperations.

You can perform the following operation at most maxOperations times:

Take any bag of balls and divide it into two new bags with a positive number of balls.
For example, a bag of 5 balls can become two new bags of 1 and 4 balls, or two new bags of 2 and 3 balls.
Your penalty is the maximum number of balls in a bag. You want to minimize your penalty after the operations.

Return the minimum possible penalty after performing the operations.

Example 1:
Input:
>nums = [9], maxOperations = 2

Output:
>3

Explanation: 
- Divide the bag with 9 balls into two bags of sizes 6 and 3. [9] -> [6,3].
- Divide the bag with 6 balls into two bags of sizes 3 and 3. [6,3] -> [3,3,3].
The bag with the most number of balls has 3 balls, so your penalty is 3 and you should return 3.

Example 2:
Input:
>nums = [2,4,8,2], maxOperations = 4

Output:
>2

Explanation:
- Divide the bag with 8 balls into two bags of sizes 4 and 4. [2,4,8,2] -> [2,4,4,4,2].
- Divide the bag with 4 balls into two bags of sizes 2 and 2. [2,4,4,4,2] -> [2,2,2,4,4,2].
- Divide the bag with 4 balls into two bags of sizes 2 and 2. [2,2,2,4,4,2] -> [2,2,2,2,2,4,2].
- Divide the bag with 4 balls into two bags of sizes 2 and 2. [2,2,2,2,2,4,2] -> [2,2,2,2,2,2,2,2].
The bag with the most number of balls has 2 balls, so your penalty is 2, and you should return 2.

Constraints:
`1 <= nums.length <= 105`
`1 <= maxOperations, nums[i] <= 109`

思路:
>这个题用二分查找做，因为若nums中的数可以分为不超过$y$的数，那么一定也可以分为不超过$x$的数($x \geq y$).初始化答案的上下界mini,maxi=1,max(nums).然后依次检验mid=(mini+maxi)//2是否可行，若可行(即将所有数分为不超过mid的数所需步骤之和不超过maxOpertions)，则将maxi改为mid，否则将mini改为mid+1.当mini>=maxi时返回maxi即可得到答案。

代码：

```python
class Solution(object):
    def minimumSize(self, nums, maxOperations):
        """
        :type nums: List[int]
        :type maxOperations: int
        :rtype: int
        """
        def count_operations(x,y):
            return (x-1)//y
        
        def sum_operations(nums,M):
            cnt=0
            for num in nums:
                cnt+=count_operations(num,M)
            return cnt
        
        def BinarySearch(nums,mini,maxi):
            if mini>=maxi:
                return maxi
            
            mid=(mini+maxi)//2
            if sum_operations(nums,mid) > maxOperations:
                return BinarySearch(nums,mid+1,maxi)
            else:
                return BinarySearch(nums,mini,mid)
        
        mini,maxi=1,max(nums)
        return BinarySearch(nums,mini,maxi)
```

### 河中跳房子
http://cs101.openjudge.cn/2025sp_routine/08210

描述

每年奶牛们都要举办各种特殊版本的跳房子比赛，包括在河里从一个岩石跳到另一个岩石。这项激动人心的活动在一条长长的笔直河道中进行，在起点和离起点L远 (1 ≤ L≤ 1,000,000,000) 的终点处均有一个岩石。在起点和终点之间，有N (0 ≤ N ≤ 50,000) 个岩石，每个岩石与起点的距离分别为Di (0 < Di < L)。

在比赛过程中，奶牛轮流从起点出发，尝试到达终点，每一步只能从一个岩石跳到另一个岩石。当然，实力不济的奶牛是没有办法完成目标的。

农夫约翰为他的奶牛们感到自豪并且年年都观看了这项比赛。但随着时间的推移，看着其他农夫的胆小奶牛们在相距很近的岩石之间缓慢前行，他感到非常厌烦。他计划移走一些岩石，使得从起点到终点的过程中，最短的跳跃距离最长。他可以移走除起点和终点外的至多M (0 ≤ M ≤ N) 个岩石。

请帮助约翰确定移走这些岩石后，最长可能的最短跳跃距离是多少？



输入
第一行包含三个整数L, N, M，相邻两个整数之间用单个空格隔开。
接下来N行，每行一个整数，表示每个岩石与起点的距离。岩石按与起点距离从近到远给出，且不会有两个岩石出现在同一个位置。

输出
一个整数，最长可能的最短跳跃距离。

样例输入
>25 5 2
>2
>11
>14
>17
>21

样例输出
>4

提示

在移除位于2和14的两个岩石之后，最短跳跃距离为4（从17到21或从21到25）。

> 放这个题主要是要说明使用二分查找时要注意:
> (i)mid是取上整还是下整
> (ii)判断结果后left和right应该做出什么样的变化
> (iii)最终是否能够退出循环并且输出正确答案

代码:
```python
L,N,M=map(int,input().split())
rocks=[0]+[int(input()) for _ in range(N)]+[L]

def check(d):
    cnt=0
    pre=0
    
    for i in range(1,N+2):
        if rocks[i]-rocks[pre] < d:
            cnt+=1
        else:
            pre=i
    
    return cnt<=M

left,right=1,L
ans=0
while left<=right:
    mid=(left+right+1)//2
    if check(mid):
        ans=mid
        left=mid+1
    else:
        right=mid-1
print(ans)
```

### 当前队列中位数
http://cs101.openjudge.cn/2025sp_routine/27256/

**描述**
中位数是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

现在，假设我有一个空的`list`，我会对他进行以下三种操作：

在`list`最后添加一个数：`add x`

删除当前`list`的第一个数：`del`

查询当前`list`的中位数：`query`

**输入**
输入为若干行，第一行为一个整数n(n <= 100000)，表示操作的次数，接下来n行表示n次操作。数据保证在删除或查询的时候，队列长度大于0

**输出**
针对于每次query，输出当前的中位数

**样例**
> sample1 input:
> 5
> add 1
> add 2
> query
> del
> query

> sample1 out:
> 1.5
> 2

> sample2 input:
> 5
> add 1
> query
> add 3
> query
> query

> sample2 out:
> 1
> 2
> 2

```python
from collections import deque
queue=deque([])
sorted_queue=[]
cnt=0

def ins(array,cnt,x):
    left,right=0,cnt-1
    if x>=array[right]:
        return cnt
    
    while left < right:
        mid=(left+right)//2
        if array[mid]<=x:
            left=mid+1
        elif array[mid] > x:
            right=mid
    
    return right

n=int(input())
for _ in range(n):
    operation=input()
    
    if operation=='del':
        x=queue.popleft()
        sorted_queue.remove(x)
        cnt-=1
    
    elif operation=='query':
        if cnt%2==1:
            print(sorted_queue[cnt//2])
        else:
            double_mid=sorted_queue[cnt//2-1]+sorted_queue[cnt//2]
            if double_mid%2==0:
                print(double_mid//2)
            else:
                print(double_mid/2)
    
    else:
        x=int(operation[4:])
        if not queue:
            queue.append(x)
            sorted_queue.append(x)
        else:
            queue.append(x)
            sorted_queue.insert(ins(sorted_queue,cnt,x),x)
        cnt+=1
```

## 3.利用栈模拟递归

### 八皇后
http://cs101.openjudge.cn/practice/02754

**描述**

会下国际象棋的人都很清楚：皇后可以在横、竖、斜线上不限步数地吃掉其他棋子。如何将8个皇后放在棋盘上（有8 * 8个方格），使它们谁也不能被吃掉！这就是著名的八皇后问题。

对于某个满足要求的8皇后的摆放方法，定义一个皇后串a与之对应，即a=b1b2...b8，其中bi为相应摆法中第i行皇后所处的列数。已经知道8皇后问题一共有92组解（即92个不同的皇后串）。
给出一个数b，要求输出第b个串。串的比较是这样的：皇后串x置于皇后串y之前，当且仅当将x视为整数时比y小。

**输入**

第1行是测试数据的组数n，后面跟着n行输入。每组测试数据占1行，包括一个正整数b(1 <= b <= 92)

**输出**

输出有n行，每行输出对应一个输入。输出应是一个正整数，是对应于b的皇后串。

**样例输入**
>2
>1
>92

**样例输出**
>15863724
>84136275

代码:
```python
# 定义一个检验当前摆放方式是否符合规则的函数is_valid
def is_valid(row,col,cols):
    for r in range(0,row):
        if cols[r]==col or abs(row-r)==abs(col-cols[r]):
            return False
    return True

# 用栈模拟递归生成所有n皇后的排序
def QueenStack(n):
    Stack=[(0,[])] # 栈暂时储存可能的结果
    Solutions=[] # 储存所有合法的排序

    while Stack:
        row,cols=Stack.pop()
        if row==n:
            Solutions.append(cols)
        else:
            for col in range(1,n+1):
                if is_valid(row,col,cols):
                    Stack.append((row+1,cols+[col]))
    return Solutions

# 得到第b个八皇后排序
def get_queen_string(b):
    queens=QueenStack(8)
    queens=queens[::-1]
    Queen=list(map(str,queens[b-1]))
    return ''.join(Queen)

n=int(input())
for _ in range(n):
    b=int(input())
    print(get_queen_string(b))
```

### 棋盘问题
http://cs101.openjudge.cn/25dsapre/01321/

**描述**

在一个给定形状的棋盘（形状可能是不规则的）上面摆放棋子，棋子没有区别。要求摆放时任意的两个棋子不能放在棋盘中的同一行或者同一列，请编程求解对于给定形状和大小的棋盘，摆放k个棋子的所有可行的摆放方案C。

**输入**

输入含有多组测试数据。
每组数据的第一行是两个正整数，n k，用一个空格隔开，表示了将在一个n*n的矩阵内描述棋盘，以及摆放棋子的数目。`n <= 8 , k <= n`
当为`-1 -1`时表示输入结束。
随后的n行描述了棋盘的形状：每行有n个字符，其中`#`表示棋盘区域，`.`表示空白区域（数据保证不出现多余的空白行或者空白列）。

**输出**
对于每一组数据，给出一行输出，输出摆放的方案数目C（数据保证C<2^31）。

**样例输入**
>2 1
>#.
>.#
>4 4
>...#
>..#.
>.#..
>#...
>-1 -1

**样例输出**
>2
>1

代码:
```python
def check(i,j,points):
    for x,y in points:
        if i==x or j==y:
            return False
    return True

def dfs(n,k,board):
    stack=[]
    cnt=0

    stack.append((0,[]))
    while stack:
        num,points=stack.pop()
        if num==k:
            cnt+=1
            continue
        if points:
            for i in range(points[-1][0]+1,n):
                for j in range(0,n):
                    if board[i][j]=='#' and check(i,j,points):
                        stack.append((num+1,points+[(i,j)]))
        else:
            for i in range(0,n):
                for j in range(0,n):
                    if board[i][j]=='#' and check(i,j,points):
                        stack.append((num+1,points+[(i,j)]))
    return cnt

while True:
    n,k=map(int,input().split())
    if n==-1 and k==-1:
        break
    board=[list(input()) for _ in range(n)]
    print(dfs(n,k,board))
```

## 4.双指针

### 盛最多水的容器
https://leetcode.cn/problems/container-with-most-water/description/

You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:

Input:`height = [1,8,6,2,5,4,8,3,7]`

Output:`49`

Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

Example 2:

Input:`height = [1,1]`

Output: `1`

Constraints:
`n == height.length`
`2 <= n <= 105`
`0 <= height[i] <= 104`


```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n=len(height)
        left,right=0,n-1
        L,R=height[left],height[right]
        water=0

        while left < right:
            water=max(water,(right-left)*min(L,R))

            if height[left]<=height[right]:
                while left < right and height[left]<=L:
                    left+=1
                L=height[left]
            elif height[left] > height[right]:
                while right > left and height[right]<=R:
                    right-=1
                R=height[right]
        return water
```



## 5.前缀和

### 构造乘积矩阵
https://leetcode.cn/problems/construct-product-matrix/

Given a **0-indexed** 2D integer matrix `grid` of size `n * m`, we define a 0-indexed 2D matrix `p` of size `n * m` as the product matrix of `grid` if the following condition is met:

- Each element `p[i][j]` is calculated as the product of all elements in grid except for the element `grid[i][j]`. This product is then taken modulo `12345`.
Return the product matrix of grid.

 

**Example 1**:

**Input**: `grid = [[1,2],[3,4]]`
**Output**: `[[24,12],[8,6]]`
**Explanation**: p[0][0] = grid[0][1] * grid[1][0] * grid[1][1] = 2 * 3 * 4 = 24
p[0][1] = grid[0][0] * grid[1][0] * grid[1][1] = 1 * 3 * 4 = 12
p[1][0] = grid[0][0] * grid[0][1] * grid[1][1] = 1 * 2 * 4 = 8
p[1][1] = grid[0][0] * grid[0][1] * grid[1][0] = 1 * 2 * 3 = 6
So the answer is `[[24,12],[8,6]]`.

**Example 2**:

**Input**: grid = `[[12345],[2],[1]]`
**Output**: `[[2],[0],[0]]`
**Explanation**: p[0][0] = grid[0][1] * grid[0][2] = 2 * 1 = 2.
p[0][1] = grid[0][0] * grid[0][2] = 12345 * 1 = 12345. 12345 % 12345 = 0. So p[0][1] = 0.
p[0][2] = grid[0][0] * grid[0][1] = 12345 * 2 = 24690. 24690 % 12345 = 0. So p[0][2] = 0.
So the answer is `[[2],[0],[0]]`.


**Constraints**:

`1 <= n == grid.length <= 105`
`1 <= m == grid[i].length <= 105`
`2 <= n * m <= 105`
`1 <= grid[i][j] <= 109`

这个题不仅用了前缀，还用了“后缀”。

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

## 6.Data Structrue

### 设置内存分配器
https://leetcode.cn/problems/design-memory-allocator/description/

You are given an integer n representing the size of a 0-indexed memory array. All memory units are initially free.

You have a memory allocator with the following functionalities:

Allocate a block of size consecutive free memory units and assign it the id mID.
Free all memory units with the given id mID.
Note that:

Multiple blocks can be allocated to the same mID.
You should free all the memory units with mID, even if they were allocated in different blocks.
Implement the Allocator class:

Allocator(int n) Initializes an Allocator object with a memory array of size n.
int allocate(int size, int mID) Find the leftmost block of size consecutive free memory units and allocate it with the id mID. Return the block's first index. If such a block does not exist, return -1.
int freeMemory(int mID) Free all memory units with the id mID. Return the number of memory units you have freed.


Example 1:

Input
> ["Allocator", "allocate", "allocate", "allocate", "freeMemory", "allocate", "allocate", "allocate", "freeMemory", "allocate", "freeMemory"]
[ [10], [1, 1], [1, 2], [1, 3], [2], [3, 4], [1, 1], [1, 1], [1], [10, 2], [7]]

Output
> [null, 0, 1, 2, 1, 3, 1, 6, 3, -1, 0]

Explanation
>Allocator loc = new Allocator(10); // Initialize a memory array of size 10. All memory units are initially free.
> loc.allocate(1, 1); // The leftmost block's first index is 0. The memory array becomes [1,_,_,_,_,_,_,_,_,_]. We return 0.
> loc.allocate(1, 2); // The leftmost block's first index is 1. The memory array becomes [1,2,_,_,_,_,_,_,_,_]. We return 1.
> loc.allocate(1, 3); // The leftmost block's first index is 2. The memory array becomes [1,2,3,_,_,_,_,_,_,_]. We return 2.
> loc.freeMemory(2); // Free all memory units with mID 2. The memory array becomes [1,_, 3,_,_,_,_,_,_,_]. We return 1 since there is only 1 unit with mID 2.
> loc.allocate(3, 4); // The leftmost block's first index is 3. The memory array becomes [1,_,3,4,4,4,_,_,_,_]. We return 3.
> loc.allocate(1, 1); // The leftmost block's first index is 1. The memory array becomes [1,1,3,4,4,4,_,_,_,_]. We return 1.
> loc.allocate(1, 1); // The leftmost block's first index is 6. The memory array becomes [1,1,3,4,4,4,1,_,_,_]. We return 6.
> loc.freeMemory(1); // Free all memory units with mID 1. The memory array becomes [_,_,3,4,4,4,_,_,_,_]. We return 3 since there are 3 units with mID 1.
> loc.allocate(10, 2); // We can not find any free block with 10 consecutive free memory units, so we return -1.
> loc.freeMemory(7); // Free all memory units with mID 7. The memory array remains the same since there is no memory unit with mID 7. We return 0.


Constraints:

- 1 <= `n, size, mID` <= 1000
- At most 1000 calls will be made to allocate and freeMemory.

```python
class Allocator(object):

    def __init__(self, n):
        """
        :type n: int
        """
        self.length=n
        self.items=[None]*n

    def allocate(self, size, mID):
        """
        :type size: int
        :type mID: int
        :rtype: int
        """
        cnt=0
        for i in range(0,self.length):
            if not self.items[i]:
                cnt+=1
                if cnt==size:
                    for j in range(i,i-size,-1):
                        self.items[j]=mID
                    return i+1-size
            else:
                cnt=0
        return -1

    def freeMemory(self, mID):
        """
        :type mID: int
        :rtype: int
        """
        cnt=0
        for i in range(0,self.length):
            if self.items[i]==mID:
                self.items[i]=None
                cnt+=1
        return cnt

# Your Allocator object will be instantiated and called as such:
# obj = Allocator(n)
# param_1 = obj.allocate(size,mID)
# param_2 = obj.freeMemory(mID)
```

### 实现堆结构

http://cs101.openjudge.cn/practice/04078/

**描述**
定义一个数组，初始化为空。在数组上执行两种操作：

1、增添1个元素，把1个新的元素放入数组。

2、输出并删除数组中最小的数。

使用堆结构实现上述功能的高效算法。

**输入**
第一行输入一个整数`n`，代表操作的次数。
每次操作首先输入一个整数`type`。
当type=1，增添操作，接着输入一个整数`u`，代表要插入的元素。
当type=2，输出删除操作，输出并删除数组中最小的元素。
`1<=n<=100000`。

**输出**
每次删除操作输出被删除的数字。

**样例输入**
> 4
> 1 5
> 1 1
> 1 7
> 2

**样例输出**
> 1

```python
### 用二叉搜索树实现堆结构
class Node:
    
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

def InsertNode(root, node):
    """
    root[Node]:树的根节点
    node[Node]:要插入的节点
    """
    if node == None:
        return
    
    if node.val <= root.val:
        if not root.left:
            root.left = node
            return
        
        InsertNode(root.left, node)
        return
    
    elif node.val > root.val:
        if not root.right:
            root.right = node
            return
        
        InsertNode(root.right, node)
        return

def DeleteMinimum(root):
    """
    root[Node]:树的根节点
    """
    node = root
    depth = 0

    while node.left and node.left.left:
        node = node.left
        depth += 1
    
    if depth == 0 and not node.left:
        return node.right, node.val
    
    mini = node.left.val
    R = node.left.right
    node.left = None
    InsertNode(root, R)
    return root, mini

n = int(input())
root = None
for _ in range(n):
    operation = list(map(int,input().split()))

    if operation[0] == 1:
        u = operation[1]
        if root == None:
            root = Node(u)
        else:
            InsertNode(root, Node(u))
    
    elif operation[0] == 2:
        root, mini = DeleteMinimum(root)
        print(mini)
```

## 7.递归 Recursion

### 有多少种合法的出栈顺序

http://cs101.openjudge.cn/2025sp_routine/27217/

**描述**

栈是计算机中经典的数据结构，简单的说，栈就是限制在一端进行插入删除操作的线性表。

栈有两种最重要的操作，即 pop（从栈顶弹出一个元素）和 push（将一个元素进栈）。

栈的重要性不言自明，任何一门数据结构的课程都会介绍栈。宁宁同学在复习栈的基本概念时，想到了一个书上没有讲过的问题，而他自己无法给出答案，所以需要你的帮忙。

![alt text](image.png)

宁宁考虑的是这样一个问题：一个操作数序列，1,2,...,n（图示为 1 到 3 的情况），栈 A 的深度大于 n。

现在可以进行两种操作，

将一个数，从操作数序列的头端移到栈的头端（对应数据结构栈的 push 操作）

将一个数，从栈的头端移到输出序列的尾端（对应数据结构栈的 pop 操作）

使用这两种操作，由一个操作数序列就可以得到一系列的输出序列，下图所示为由 1 2 3 生成序列 2 3 1 的过程。

![alt text](image-1.png)


（原始状态如上图所示）

你的程序将对给定的 n，计算并输出由操作数序列 1,2,...,n 经过操作可能得到的输出序列的总数。

**输入**
输入文件只含一个整数 n（1 <= n <= 1000）。

**输出**
输出文件只有一行，即可能输出序列的总数目。

**样例输入**
> 3

**样例输出**
> 5


**方法一：==记忆化搜索==**
```python
"""
递归/记忆化搜索，https://www.luogu.com.cn/problem/solution/P1044
1）二维数组f[i,j]，用下标 i 表示队列里还有几个待排的数，j 表示栈里有 j 个数，
f[i,j]表示此时的情况数
2）那么，更加自然的，只要f[i,j]有值就直接返回；
3）然后递归如何实现呢？首先，可以想到，要是数全在栈里了，就只剩1种情况了，所以：i=0时，返回1；
4）然后，有两种情况：一种栈空，一种栈不空：在栈空时，我们不可以弹出栈里的元素，只能进入，
所以队列里的数−1，栈里的数+1，即加上 f[i−1,j+1] ；另一种是栈不空，
那么此时有出栈1个或者进1个再出1个 2种情况，分别加上 f[i−1,j+1] 和 f[i,j−1] 
"""

import sys
sys.setrecursionlimit(1<<30)

def dfs(i, j, f):
    if f[i][j] != -1:
        return f[i][j]
    
    if i == 0:
        f[i][j] = 1
        return 1
    
    if j == 0:
        f[i][j] = dfs(i - 1, j + 1, f)
        return f[i][j]
    
    f[i][j] = dfs(i - 1, j + 1, f) + dfs(i, j - 1, f)
    return f[i][j]

n = int(input())
f = [[-1] * (n + 1) for _ in range(n + 1)]

result = dfs(n, 0, f)
print(result)
```

**方法二：==卡特兰数==**

这个问题可以通过卡特兰数（Catalan number）来解决。对于一个操作数序列 1, 2, ..., n，经过栈的 push 和 pop 操作得到的输出序列的总数就是第 n 个卡特兰数。 卡特兰数的递推公式为：
$$C_0 = 1$$
$$C_n = \sum_{i=0}^{n - 1} C_n C_{n - 1 -i}, n > 0$$
也可以直接利用公式:
$$C_n = \frac{C_{2n}^{n}}{n + 1} = \frac{(2n)!}{n!(n + 1)!}$$

```python
n = int(input())
# 初始化卡特兰数数组
catalan = [0] * (n + 1)
# 初始化 C_0 为 1
catalan[0] = 1

# 递推计算卡特兰数
for i in range(1, n + 1):
    for j in range(i):
        catalan[i] += catalan[j] * catalan[i - 1 - j]

# 输出第 n 个卡特兰数
print(catalan[n])
```

### 分解因数

http://cs101.openjudge.cn/2025sp_routine/02749/

**描述**
给出一个正整数a，要求分解成若干个正整数的乘积，即a = a1 * a2 * a3 * ... * an，并且1 < a1 <= a2 <= a3 <= ... <= an，问这样的分解的种数有多少。注意到a = a也是一种分解。

**输入**
第1行是测试数据的组数n，后面跟着n行输入。每组测试数据占1行，包括一个正整数a (1 < a < 32768)

**输出**
n行，每行输出对应一个输入。输出应是一个正整数，指明满足要求的分解的种数

**样例输入**
> 2
> 2
> 20

**样例输出**
> 1
> 4

```python
def dfs(a,mini):
    if a == 1:
        return 1
    cnt = 0
    for i in range(mini,a + 1):
        if a % i == 0:
            cnt += dfs(a // i,i)
    return cnt

n = int(input())
for _ in range(n):
    a = int(input())
    print(dfs(a,2))
```

## 8.Trie Data Structrue

### 电话号码

http://cs101.openjudge.cn/practice/04089/

**描述**
给你一些电话号码，请判断它们是否是一致的，即是否有某个电话是另一个电话的前缀。比如：

Emergency 911
Alice 97 625 999
Bob 91 12 54 26

在这个例子中，我们不可能拨通Bob的电话，因为Emergency的电话是它的前缀，当拨打Bob的电话时会先接通Emergency，所以这些电话号码不是一致的。

**输入**
第一行是一个整数`t`，`1 ≤ t ≤ 40`，表示测试数据的数目。
每个测试样例的第一行是一个整数`n`，`1 ≤ n ≤ 10000`，其后`n`行每行是一个不超过10位的电话号码。

**输出**
对于每个测试数据，如果是一致的输出“YES”，如果不是输出“NO”。

**样例输入**
> 2
> 3
> 911
> 97625999
> 91125426
> 5
> 113
> 12340
> 123440
> 12345
> 98346

**样例输出**
> NO
> YES


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

## 9.单调栈

### 柱状图中最大的矩阵

https://leetcode.cn/problems/largest-rectangle-in-histogram/description/

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

**示例 1:**
![alt text](image-3.png)
**输入：**`heights = [2,1,5,6,2,3]`
**输出：**`10`
**解释：** 最大的矩形为图中红色区域，面积为 10

**示例 2：**
![alt text](image-4.png)
**输入：** `heights = [2,4]`
**输出：** `4`


**提示：**
- `1 <= heights.length <=105`
- `0 <= heights[i] <= 104`

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

## 10.拓扑排序

### 有向图中最大颜色值

https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/

给你一个 有向图 ，它含有`n`个节点和`m`条边。节点编号从`0 到 n - 1`。

给你一个字符串`colors`，其中`colors[i]`是小写英文字母，表示图中第 `i` 个节点的 颜色 （下标从 `0` 开始）。同时给你一个二维数组 `edges` ，其中 `edges[j] = [aj, bj]` 表示从节点 `aj` 到节点 `bj` 有一条 有向边 。

图中一条有效 路径 是一个点序列 `x1 -> x2 -> x3 -> ... -> xk` ，对于所有 `1 <= i < k` ，从 `xi` 到 `xi+1` 在图中有一条有向边。路径的 颜色值 是路径中 出现次数最多 颜色的节点数目。

请你返回给定图中有效路径里面的 最大颜色值 。如果图中含有环，请返回 `-1` 。

 

示例 1：
![alt text](image-5.png)
> 输入：colors = "abaca", edges = `[[0,1],[0,2],[2,3],[3,4]]`
> 输出：3
> 解释：路径 0 -> 2 -> 3 -> 4 含有 3 个颜色为 "a" 的节点（上图中的红色节点）。

示例 2：
![alt text](image-6.png)
> 输入：colors = "a", edges = `[[0,0]]`
> 输出：-1
> 解释：从 0 到 0 有一个环。


提示：
- `n == colors.length`
- `m == edges.length`
- `1 <= n <= 105`
- `0 <= m <= 105`
- `colors`只含有小写英文字母。
- `0 <= aj, bj < n`

```python
from collections import deque

class Vertex:

    def __init__(self, i, color):
        self.index = i
        self.color = color
        self.indegree = 0
        self.neighbors = []

class Solution(object):
    def largestPathValue(self, colors, edges):
        """
        :type colors: str
        :type edges: List[List[int]]
        :rtype: int
        """
        n = len(colors)
        graph = [Vertex(i, colors[i]) for i in range(n)]
        for s, e in edges:
            graph[s].neighbors.append(graph[e])
            graph[e].indegree += 1
        
        cnt = 0
        dp = [[0] * 26 for _ in range(n)]
        queue = deque()
        for vertex in graph:
            if vertex.indegree == 0:
                queue.append(vertex)
        
        while queue:
            v = queue.popleft()
            cnt += 1
            dp[v.index][ord(v.color) - ord('a')] += 1
            for neighbor in v.neighbors:
                neighbor.indegree -= 1
                if neighbor.indegree == 0:
                    queue.append(neighbor)
                for c in range(26):
                    dp[neighbor.index][c] = max(dp[neighbor.index][c], dp[v.index][c])
        
        if cnt != n:
            return -1
        
        maxi = 0
        for i in range(n):
            maxi = max(maxi, max(dp[i]))
        return maxi
```

<font color='red'>注：<font color='black'>==不存在环等价于拓扑排序正好遍历了$n$个点。==

## 11.KMP算法

关于该算法，可以参考[字符串匹配的KMP算法](https://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html)。

### 前缀中的周期

http://cs101.openjudge.cn/2025sp_routine/01961/

**描述**
一个字符串的前缀是从第一个字符开始的连续若干个字符，例如"abaab"共有5个前缀，分别是a, ab, aba, abaa,  abaab。

我们希望知道一个N位字符串S的前缀是否具有循环节。换言之，对于每一个从头开始的长度为 i （i 大于1）的前缀，是否由重复出现的子串A组成，即 AAA...A （A重复出现K次,K 大于 1）。如果存在，请找出最短的循环节对应的K值（也就是这个前缀串的所有可能重复节中，最大的K值）。

**输入**
输入包括多组测试数据。每组测试数据包括两行。
第一行包括字符串S的长度N（2 <= N <= 1 000 000）。
第二行包括字符串S。
输入数据以只包括一个0的行作为结尾。

**输出**
对于每组测试数据，第一行输出 "Test case #“ 和测试数据的编号。
接下来的每一行，输出前缀长度i和重复测数K，中间用一个空格隔开。前缀长度需要升序排列。
在每组测试数据的最后输出一个空行。

**样例输入**
> 3
> aaa
> 12
> aabaabaabaab
> 0

**样例输出**
> Test case #1
> 2 2
> 3 3
> 
> Test case #2
> 2 2
> 6 2
> 9 3
> 12 4

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

## 12.Dijkstra算法

### Subway

http://cs101.openjudge.cn/2025sp_routine/02502/

**描述**
You have just moved from a quiet Waterloo neighbourhood to a big, noisy city. Instead of getting to ride your bike to school every day, you now get to walk and take the subway. Because you don't want to be late for class, you want to know how long it will take you to get to school.
You walk at a speed of 10 km/h. The subway travels at 40 km/h. Assume that you are lucky, and whenever you arrive at a subway station, a train is there that you can board immediately. You may get on and off the subway any number of times, and you may switch between different subway lines if you wish. All subway lines go in both directions.

**输入**
Input consists of the x,y coordinates of your home and your school, followed by specifications of several subway lines. Each subway line consists of the non-negative integer x,y coordinates of each stop on the line, in order. You may assume the subway runs in a straight line between adjacent stops, and the coordinates represent an integral number of metres. Each line has at least two stops. The end of each subway line is followed by the dummy coordinate pair -1,-1. In total there are at most 200 subway stops in the city.

**输出**
Output is the number of minutes it will take you to get to school, rounded to the nearest minute, taking the fastest route.

**样例输入**
> 0 0 10000 1000
> 0 200 5000 200 7000 200 -1 -1 
> 2000 600 5000 600 10000 600 -1 -1

**样例输出**
> 21

```python
import math
import heapq

# 计算两点之间的欧几里得距离
def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 读取起点（家）和终点（学校）坐标
sx, sy, ex, ey = map(int, input().split())

# min_time: 记录从起点到每个地铁站/终点的最短时间（单位：小时）
min_time = {}

# rails: 记录所有地铁连接（双向）
rails = set()

# 读取所有地铁线路
while True:
    try:
        rail = list(map(int, input().split()))
        if rail == [-1, -1]:
            break
        # 解析当前地铁线路的所有站点
        stations = [(rail[2 * i], rail[2 * i + 1]) for i in range(len(rail) // 2 - 1)]

        for j, station in enumerate(stations):
            # 初始化所有地铁站点的最短时间为无穷大
            min_time[station] = float('inf')
            # 添加地铁线路中相邻站点的双向连接
            if j != len(stations) - 1:
                rails.add((station, stations[j + 1]))
                rails.add((stations[j + 1], station))
    except EOFError:
        break  # 输入结束

# 把起点和终点加入时间表中
min_time[(sx, sy)] = 0  # 起点时间为 0
min_time[(ex, ey)] = float('inf')  # 终点初始化为无穷大

# 使用小根堆实现 Dijkstra 算法，按时间升序处理节点
min_heap = [(0, sx, sy)]  # (当前耗时, 当前x, 当前y)

while min_heap:
    curr_time, x, y = heapq.heappop(min_heap)

    # 如果当前耗时不是最短路径中记录的值，说明已经被更新，跳过
    if curr_time > min_time[(x, y)]:
        continue

    # 如果已经到达终点，提前结束
    if (x, y) == (ex, ey):
        break

    # 遍历所有可达点（隐式图）
    for position in min_time.keys():
        if position == (x, y):
            continue  # 自己跳过
        nx, ny = position

        # 计算当前位置到下一个点的距离
        dis = get_distance(x, y, nx, ny)

        # 判断是否为地铁连接：地铁速度是步行的4倍
        rail_factor = 4 if ((position, (x, y)) in rails or ((x, y), position) in rails) else 1

        # 计算到该点的所需时间（单位：小时）
        new_time = curr_time + dis / (10000 * rail_factor)

        # 如果时间更短，则更新并加入堆中
        if new_time < min_time[position]:
            min_time[position] = new_time
            heapq.heappush(min_heap, (new_time, nx, ny))

# 输出从起点到终点的最短时间，转换为分钟并四舍五入
print(round(min_time[(ex, ey)] * 60))
```

## 13.差分数组

### 零数组变换I

https://leetcode.cn/problems/zero-array-transformation-i/submissions/634039152/

给定一个长度为 `n` 的整数数组 `nums` 和一个二维数组 `queries`，其中 `queries[i] = [li, ri]`。

对于每个查询 `queries[i]`：

在 `nums` 的下标范围 `[li, ri]` 内选择一个下标 子集。
将选中的每个下标对应的元素值减 `1`。
零数组 是指所有元素都等于 `0` 的数组。

如果在按顺序处理所有查询后，可以将 `nums` 转换为 零数组 ，则返回 `true`，否则返回 `false`。

 

**示例 1**：

**输入**： `nums = [1,0,1], queries = [[0,2]]`

**输出**： `true`

**解释**：

对于 `i = 0`：
选择下标子集 `[0, 2]` 并将这些下标处的值减 `1`。
数组将变为 `[0, 0, 0]`，这是一个零数组。

**示例 2**：

**输入**： `nums = [4,3,2,1], queries = [[1,3],[0,2]]`

**输出**： `false`

**解释**：

对于 `i = 0`： 
选择下标子集 `[1, 2, 3]` 并将这些下标处的值减 `1`。
数组将变为 `[4, 2, 1, 0]`。
对于 `i = 1`：
选择下标子集 `[0, 1, 2]` 并将这些下标处的值减 `1`。
数组将变为 `[3, 1, 0, 0]`，这不是一个零数组。


**提示**：

- `1 <= nums.length <= 105`
- `0 <= nums[i] <= 105`
- `1 <= queries.length <= 105`
- `queries[i].length == 2`
- `0 <= li <= ri < nums.length`

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