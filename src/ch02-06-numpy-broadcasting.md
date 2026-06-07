## NumPy ব্রডকাস্টিং ও অপারেশন

### কেন ব্রডকাস্টিং শিখবো?

"ব্রডকাস্টিং" শুনলে মনে হতে পারে এটা কোনো টিভি চ্যানেলের কথা। কিন্তু NumPy তে ব্রডকাস্টিং হলো একটা অত্যন্ত শক্তিশালী ফিচার, যেটা আলাদা আলাদা শেপের array নিয়ে কাজ করতে দেয়। এটা না বুঝলে, অনেক এরর আসবে আর বুঝতেও পারা যাবে না কেন আসছে।

### এলিমেন্ট-ওয়াইজ অপারেশন

প্রথমে দেখি, NumPy তে কিভাবে গাণিতিক অপারেশন কাজ করে:

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Element-wise operations
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
print(f"a % b = {a % b}")
```

Output:
```
a + b = [11 22 33 44 55]
a - b = [ -9 -18 -27 -36 -45]
a * b = [ 10  40  90 160 250]
a / b = [0.1 0.1 0.1 0.1 0.1]
a ** 2 = [ 1  4  9 16 25]
a % b = [1 2 3 4 5]
```

এগুলো সব এলিমেন্ট-ওয়াইজ, মানে প্রতিটা এলিমেন্টে আলাদা আলাদা অপারেশন হচ্ছে। Python list এ এটা হতো না, সেখানে loop লাগতো।

### স্কেলার অপারেশন

একটা array আর একটা সংখ্যা (স্কেলার) এর মধ্যে অপারেশন:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(f"arr + 10 = {arr + 10}")
print(f"arr * 3 = {arr * 3}")
print(f"arr / 2 = {arr / 2}")
print(f"arr ** 2 = {arr ** 2}")
```

Output:
```
arr + 10 = [11 12 13 14 15]
arr * 3 = [ 3  6  9 12 15]
arr / 2 = [0.5 1.  1.5 2.  2.5]
arr ** 2 = [ 1  4  9 16 25]
```

এটাই হলো সবচেয়ে সহজ ব্রডকাস্টিং। স্কেলার ১০ কে "ব্রডকাস্ট" করে পুরো array এ, মানে প্রতিটা এলিমেন্টে ১০ যোগ হয়।

### ব্রডকাস্টিং কি?

ব্রডকাস্টিং হলো NumPy এর একটা নিয়ম, যেটা আলাদা শেপের array নিয়ে অপারেশন করতে দেয়। ছোট array কে স্বয়ংক্রিয়ভাবে "স্ট্রেচ" করে বড় array এর শেপে আনে। কিন্তু সব শেপে এটা কাজ করে না, নির্দিষ্ট নিয়ম আছে।

**ব্রডকাস্টিং এর তিনটা নিয়ম:**

1. দুইটা array এর ndim আলাদা হলে, ছোট array এর শেপের বামে 1 যোগ হয়
2. কোনো ডাইমেনশনে শেপ 1 হলে, সেটা বড় array এর শেপ পর্যন্ত স্ট্রেচ হয়
3. কোনো ডাইমেনশনে শেপ আলাদা আর 1 ও না হলে, এরর আসবে

```python
import numpy as np

# Example 1: 1D + scalar
a = np.array([1, 2, 3])
b = 5
print(f"{a.shape} + scalar = {(a + b).shape}")
print(f"{a} + {b} = {a + b}")

# Example 2: 2D + 1D
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])
print(f"\na shape: {a.shape}, b shape: {b.shape}")
print(f"a + b:\n{a + b}")

# Example 3: Column broadcasting
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([[10],
              [20]])
print(f"\na shape: {a.shape}, b shape: {b.shape}")
print(f"a + b:\n{a + b}")
```

Output:
```
(3,) + scalar = (3,)
[1 2 3] + 5 = [6 7 8]

a shape: (2, 3), b shape: (3,)
a + b:
[[11 22 33]
 [14 25 36]]

a shape: (2, 3), b shape: (2, 1)
a + b:
[[11 12 13]
 [24 25 26]]
```

### ব্রডকাস্টিং কাজ না করার উদাহরণ

```python
import numpy as np

# Broadcasting will not work in this case
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)
b = np.array([10, 20])     # shape: (2,)

try:
    result = a + b
except ValueError as e:
    print(f"Error: {e}")
    print(f"a shape: {a.shape}, b shape: {b.shape}")
    print("Solution: reshape b to (2,1)")

# Solution
b_reshaped = b.reshape(2, 1)
print(f"\nb reshaped: {b_reshaped.shape}")
print(f"a + b_reshaped:\n{a + b_reshaped}")
```

Output:
```
Error: operands could not be broadcast together with shapes (2,3) (2,)
a shape: (2, 3), b shape: (2,)
Solution: reshape b to (2,1)

b reshaped: (2, 1)
a + b_reshaped:
[[11 12 13]
 [24 25 26]]
```

### তুলনামূলক অপারেশন

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 2, 1, 4, 5])

print(f"a > b: {a > b}")
print(f"a < b: {a < b}")
print(f"a == b: {a == b}")
print(f"a != b: {a != b}")
print(f"a >= b: {a >= b}")

# Which positions satisfy the condition
print(f"\nnp.where(a > b): {np.where(a > b)}")
print(f"np.where(a > 3): {np.where(a > 3)}")
```

Output:
```
a > b: [False False  True False False]
a < b: [ True False False False False]
a == b: [False  True False  True  True]
a != b: [ True False  True False False]
a >= b: [False  True  True  True  True]

np.where(a > b): (array([2]),)
np.where(a > 3): (array([3, 4]),)
```

### লজিক্যাল অপারেশন

```python
import numpy as np

a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(f"a & b (AND): {a & b}")
print(f"a | b (OR): {a | b}")
print(f"~a (NOT): {~a}")

# Logical operations on numbers
x = np.array([1, 5, 8, 3, 9, 2, 7])
print(f"\nnp.logical_and(x > 3, x < 8): {np.logical_and(x > 3, x < 8)}")
print(f"np.logical_or(x < 3, x > 8): {np.logical_or(x < 3, x > 8)}")
print(f"np.logical_not(x > 5): {np.logical_not(x > 5)}")
```

Output:
```
a & b (AND): [ True False False False]
a | b (OR): [ True  True  True False]
~a (NOT): [False False  True  True]

np.logical_and(x > 3, x < 8): [False  True  True False False False  True]
np.logical_or(x < 3, x > 8): [ True False False False  True  True False]
np.logical_not(x > 5): [ True  True False  True False  True False]
```

### ম্যাট্রিক্স অপারেশন

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
print(f"A * B (element-wise):\n{A * B}")

# Matrix multiplication (dot product)
print(f"\nA @ B (matrix multiplication):\n{A @ B}")
print(f"\nnp.dot(A, B):\n{np.dot(A, B)}")

# Matrix transpose
print(f"\nA.T:\n{A.T}")

# Determinant
print(f"\nnp.linalg.det(A): {np.linalg.det(A)}")

# Inverse
print(f"\nnp.linalg.inv(A):\n{np.linalg.inv(A)}")
```

Output:
```
A * B (element-wise):
[[ 5 12]
 [21 32]]

A @ B (matrix multiplication):
[[19 22]
 [43 50]]

np.dot(A, B):
[[19 22]
 [43 50]]

A.T:
[[1 3]
 [2 4]]

np.linalg.det(A): -2.0000000000000004

np.linalg.inv(A):
[[-2.   1. ]
 [ 1.5 -0.5]]
```

### সারসংক্ষেপ

| অপারেশন | সিনট্যাক্স | ব্যাখ্যা |
|---|---|---|
| এলিমেন্ট-ওয়াইজ | `a + b`, `a * b` | প্রতি এলিমেন্টে আলাদা |
| স্কেলার | `a + 5` | সব এলিমেন্টে যোগ |
| ব্রডকাস্টিং | `a + b` (আলাদা shape) | স্বয়ংক্রিয় স্ট্রেচ |
| তুলনামূলক | `a > b` | বুলিয়ান array |
| ম্যাট্রিক্স গুণ | `A @ B` | ডট প্রোডাক্ট |
| ট্রান্সপোজ | `A.T` | রো/কলাম অদলবদল |

ব্রডকাস্টিং হলো NumPy এর সবচেয়ে শক্তিশালী ফিচার, কিন্তু এটা না বুঝলে সবচেয়ে বেশি এরর আসবে। "shapes not aligned" এরর আসলে বোঝা যাবে, ব্রডকাস্টিং নিয়ম মিলছে না। এরর আসলে shape প্রিন্ট করতে হবে, ব্রডকাস্টিং নিয়ম মেলাতে হবে, দরকার হলে reshape করতে হবে।

#NumPy #Broadcasting #MatrixOperations #DataScience #PythonForML
