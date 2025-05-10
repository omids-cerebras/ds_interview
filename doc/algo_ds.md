
# Algorithm and Data Structure Problems

## Problem 1: Real-Time Rolling Average (Sliding Window)
### Task:
You are processing real-time sensor data and need to calculate the rolling average of the last `k` data points. Implement a function to maintain the rolling average efficiently.

### Code Skeleton:
```python
from collections import deque

class RollingAverage:
    def __init__(self, k):
        self.window = deque()
        self.k = k
        self.total = 0

    def add(self, value):
        # TODO: Add a new value to the window
        # Update the total and remove the oldest value if needed
        pass

    def get_average(self):
        # TODO: Return the current rolling average
        pass

# Example usage
ra = RollingAverage(3)
ra.add(5)
ra.add(10)
ra.add(15)
print(ra.get_average())  # Should print the average of [5, 10, 15]
ra.add(20)
print(ra.get_average())  # Should print the average of [10, 15, 20]
```
**Instructions:**
1. Implement the `add` method to update the rolling average.
2. Implement the `get_average` method to calculate the current average.
3. Use a deque for efficient removal of the oldest element.

---

## Problem 2: Anomaly Detection using Hash Maps
### Task:
Given a list of sensor readings, find the first reading that appears more than once.  
- Use a hash map to track occurrences.  
- Return the first duplicate reading.  

### Code Skeleton:
```python
def find_first_duplicate(readings):
    seen = set()
    for reading in readings:
        # TODO: Check if the reading has been seen before
        pass
    return None

# Example usage
readings = [12, 15, 9, 12, 20, 15]
print(find_first_duplicate(readings))  # Should print 12
```
**Instructions:**
1. Use a hash map (set) to track seen readings.
2. Return the first duplicate if found, otherwise return None.

---

## Problem 3: Dependency Resolution (Topological Sorting)
### Task:
Given a list of tasks and their dependencies, determine the order in which to complete the tasks.  
- Use topological sorting to resolve dependencies.  
- Tasks are represented as a directed graph.  

### Code Skeleton:
```python
from collections import deque, defaultdict

def topological_sort(tasks, dependencies):
    graph = defaultdict(list)
    indegree = defaultdict(int)

    # Build graph and calculate in-degrees
    for task, dep in dependencies:
        graph[dep].append(task)
        indegree[task] += 1

    # Initialize the queue with tasks having zero in-degree
    queue = deque([task for task in tasks if indegree[task] == 0])
    order = []

    while queue:
        current = queue.popleft()
        order.append(current)
        # TODO: Update in-degrees and enqueue zero in-degree tasks
        pass

    return order if len(order) == len(tasks) else []

# Example usage
tasks = ['A', 'B', 'C', 'D']
dependencies = [('B', 'A'), ('C', 'A'), ('D', 'B'), ('D', 'C')]
print(topological_sort(tasks, dependencies))  # Should print a valid topological order
```
**Instructions:**
1. Implement the topological sort algorithm.
2. Use a queue to maintain tasks with zero in-degree.
3. Check for cycles (return an empty list if found).

---

## Problem 4: Optimal Resource Allocation (Greedy)
### Task:
You are given a list of tasks with their durations and a fixed number of machines.  
- Distribute tasks to minimize the maximum workload on any machine.  
- Use a greedy algorithm to assign tasks.  

### Code Skeleton:
```python
import heapq

def minimize_max_workload(tasks, machines):
    workloads = [0] * machines
    heapq.heapify(workloads)

    for task in sorted(tasks, reverse=True):
        # TODO: Assign the current task to the machine with the least workload
        pass

    return max(workloads)

# Example usage
tasks = [5, 3, 8, 2, 7, 4]
machines = 3
print(minimize_max_workload(tasks, machines))  # Should print the minimum max workload
```
**Instructions:**
1. Use a min-heap to track workloads.
2. Assign the heaviest remaining task to the machine with the smallest workload.

---

## Problem 5: Stream Median Calculation (Heaps)
### Task:
Implement a data structure that supports adding numbers from a stream and efficiently retrieving the median of the numbers seen so far.  
- Use two heaps to maintain the lower and upper halves of the numbers.  

### Code Skeleton:
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.low = []  # Max heap (invert values for min-heap behavior)
        self.high = []  # Min heap

    def add_number(self, num):
        # TODO: Add the number to the correct heap and balance
        pass

    def find_median(self):
        # TODO: Return the current median
        pass

# Example usage
mf = MedianFinder()
mf.add_number(1)
mf.add_number(5)
mf.add_number(2)
print(mf.find_median())  # Should print 2
mf.add_number(10)
print(mf.find_median())  # Should print 3.5
```
**Instructions:**
1. Add numbers to the appropriate heap.
2. Balance the heaps to maintain the median.
3. Implement the `find_median` method to return the current median.
