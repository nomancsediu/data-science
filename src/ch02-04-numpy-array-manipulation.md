## NumPy Array Manipulation

### Reshape

অ্যারের শেপ পরিবর্তন করা। মোট এলিমেন্ট সংখ্যা বদলানো যাবে না, শুধু সাজানোর ধরন বদলায়। ML এ ডেটা লোড করে শেপ ঠিক করতে সবচেয়ে বেশি reshape লাগে।

```python
import numpy as np

arr = np.arange(12)
print(f"Original: {arr}, shape: {arr.shape}")

# 1D to 2D
print(f"\nreshape(3,4):\n{arr.reshape(3, 4)}")
print(f"\nreshape(4,3):\n{arr.reshape(4, 3)}")

# 1D to 3D
print(f"\nreshape(2,2,3):\n{arr.reshape(2, 2, 3)}")

# -1 দিয়ে অটো ক্যালকুলেশন (বড় ডেটাসেটে খুব কাজের)
auto = arr.reshape(3, -1)
print(f"\nreshape(3,-1): shape={auto.shape}\n{auto}")

# ভুল shape - মোট এলিমেন্ট মিলতে হবে
try:
    arr.reshape(3, 5)
except ValueError as e:
    print(f"\nreshape(3,5) error: {e}")
```

Output:
```
Original: [ 0  1  2  3  4  5  6  7  8  9 10 11], shape: (12,)

reshape(3,4):
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

reshape(4,3):
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

reshape(2,2,3):
[[[ 0  1  2]
  [ 3  4  5]]

 [[ 6  7  8]
  [ 9 10 11]]]

reshape(3,-1): shape=(3, 4)
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

reshape(3,5) error: cannot reshape array of size 12 into shape (3,5)
```

**reshape vs resize:** reshape মোট এলিমেন্ট বদলাতে পারে না। resize পারে, বেশি হলে repeat করে, কম হলে truncate করে।

```python
import numpy as np

arr = np.arange(6)
print(f"Original: {arr}")

# reshape - এলিমেন্ট মিলতে হবে
print(f"reshape(2,3):\n{arr.reshape(2, 3)}")

# resize - এলিমেন্ট না মিললেও চলে
print(f"\nresize(3,3) - repeat করে:\n{np.resize(arr, (3, 3))}")
print(f"\nresize(2,2) - truncate করে:\n{np.resize(arr, (2, 2))}")
```

Output:
```
Original: [0 1 2 3 4 5]
reshape(2,3):
[[0 1 2]
 [3 4 5]]

resize(3,3) - repeat করে:
[[0 1 2]
 [3 4 5]
 [0 1 2]]

resize(2,2) - truncate করে:
[[0 1]
 [2 3]]
```

**Row vector আর Column vector:** Scikit-learn এ 1D array কে row বা column vector বানাতে প্রায়ই লাগে।

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(f"Original shape: {arr.shape}")

row = arr.reshape(1, -1)       # বা arr[np.newaxis, :]
col = arr.reshape(-1, 1)       # বা arr[:, np.newaxis]

print(f"Row vector: shape={row.shape}, {row}")
print(f"Column vector: shape={col.shape}\n{col}")
```

Output:
```
Original shape: (5,)
Row vector: shape=(1, 5), [[1 2 3 4 5]]
Column vector: shape=(5, 1)
[[1]
 [2]
 [3]
 [4]
 [5]]
```

### Transpose

ম্যাট্রিক্সের সারি আর কলাম অদলবদল। Linear Regression, Neural Network এর ম্যাট্রিক্স মাল্টিপ্লিকেশনে সবখানে Transpose লাগে। 1D array তে .T কাজ করে না, আগে 2D বানাতে হবে।

```python
import numpy as np

# 2D transpose
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(f"Original shape: {arr.shape}")
print(f"Transpose shape: {arr.T.shape}")
print(f"Transpose:\n{arr.T}")

# 1D তে .T কাজ করে না
arr_1d = np.array([1, 2, 3])
print(f"\n1D .T shape: {arr_1d.T.shape} (কিছু হয়নি)")

# 1D কে column বানাতে
col = arr_1d[:, np.newaxis]
print(f"1D -> column: shape={col.shape}")
```

Output:
```
Original shape: (2, 3)
Transpose shape: (3, 2)
Transpose:
[[1 4]
 [2 5]
 [3 6]]

1D .T shape: (3,) (কিছু হয়নি)
1D -> column: shape=(3, 1)
```

**3D Transpose:** axis কোনটা কোথায় যাবে সেটা transpose() এর axes প্যারামিটার দিয়ে কন্ট্রোল করা যায়।

```python
import numpy as np

arr_3d = np.arange(24).reshape(2, 3, 4)
print(f"Original shape: {arr_3d.shape}")

# axes swap: (0,1,2) -> (1,0,2)
print(f"transpose(1,0,2) shape: {arr_3d.transpose(1, 0, 2).shape}")

# swapaxes - শুধু দুইটা axis অদলবদল
print(f"swapaxes(0,2) shape: {arr_3d.swapaxes(0, 2).shape}")

# .T দিয়ে 3D তে reverse order হয়
print(f".T shape: {arr_3d.T.shape}")
```

Output:
```
Original shape: (2, 3, 4)
transpose(1,0,2) shape: (3, 2, 4)
swapaxes(0,2) shape: (4, 3, 2)
.T shape: (4, 3, 2)
```

**Matrix multiplication এ Transpose কেন লাগে:** ম্যাট্রিক্স গুণের নিয়ম, প্রথম ম্যাট্রিক্সের column সংখ্যা = দ্বিতীয় ম্যাট্রিক্সের row সংখ্যা। না মিললে transpose করে মেলাতে হয়।

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
B = np.array([[10, 20], [30, 40]])        # (2, 2)

result = A @ B
print(f"A({A.shape}) @ B({B.shape}) = result({result.shape})")
print(result)

# যদি B এর shape উল্টা থাকে
C = np.array([[10, 30], [20, 40]])  # (2, 2) - উল্টা
# A @ C হবে না কারণ A columns=2, C rows=2, কিন্তু result আলাদা হবে
print(f"\nA @ C:\n{A @ C}")
print(f"A @ C.T:\n{A @ C.T}")
```

Output:
```
A(3, 2) @ B(2, 2) = result(3, 2)
[[ 70 100]
 [150 220]
 [230 340]]

A @ C:
[[ 50 110]
 [110 250]
 [170 390]]
A @ C.T:
[[ 70 100]
 [150 220]
 [230 340]]
```

### Flatten ও Ravel

মাল্টিডাইমেনশনাল array কে 1D তে নামিয়ে আনা। flatten() নতুন copy বানায়, ravel() আসল array এর view দেয়। Neural Network এ image flatten করে Dense Layer এ পাঠানো হয়।

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D:\n{arr}")

flat = arr.flatten()
rav = arr.ravel()
print(f"\nflatten(): {flat}")
print(f"ravel():   {rav}")

# পার্থক্য: flatten modify করলে original বদলায় না
flat[0] = 100
print(f"\nflatten modify -> original:\n{arr}")

# ravel modify করলে original বদলে যায়
rav2 = arr.ravel()
rav2[0] = 100
print(f"ravel modify -> original:\n{arr}")
```

Output:
```
2D:
[[1 2 3]
 [4 5 6]]

flatten(): [1 2 3 4 5 6]
ravel():   [1 2 3 4 5 6]

flatten modify -> original:
[[1 2 3]
 [4 5 6]]

ravel modify -> original:
[[100   2   3]
 [  4   5   6]]
```

**C order vs F order:** C order row ধরে যায় (default), F order column ধরে যায়।

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"C order (row-wise): {arr.flatten(order='C')}")
print(f"F order (col-wise): {arr.flatten(order='F')}")

# 3D flatten
arr_3d = np.arange(24).reshape(2, 3, 4)
print(f"\n3D shape: {arr_3d.shape}")
print(f"Flattened: {arr_3d.flatten()}")
```

Output:
```
C order (row-wise): [1 2 3 4 5 6]
F order (col-wise): [1 4 2 5 3 6]

3D shape: (2, 3, 4)
Flattened: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
```

### Concatenation

দুই বা তার বেশি অ্যারে জোড়া লাগানো। axis=0 দিলে row যোগ, axis=1 দিলে column যোগ। ডেটাসেটে নতুন feature বা নতুন data যোগ করতে এটা লাগে। জোড়া লাগানোর সময় জোড়া লাগানোর দিক ছাড়া বাকি ডাইমেনশন মিলতে হবে।

```python
import numpy as np

# 1D concatenation
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"1D concat: {np.concatenate([a, b])}")

# 2D - axis=0 (row যোগ, column সংখ্যা মিলতে হবে)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
print(f"\naxis=0 (row-wise):\n{np.concatenate([arr1, arr2], axis=0)}")

# 2D - axis=1 (column যোগ, row সংখ্যা মিলতে হবে)
arr3 = np.array([[5], [6]])
print(f"\naxis=1 (col-wise):\n{np.concatenate([arr1, arr3], axis=1)}")
```

Output:
```
1D concat: [1 2 3 4 5 6]

axis=0 (row-wise):
[[1 2]
 [3 4]
 [5 6]]

axis=1 (col-wise):
[[1 2 5]
 [3 4 6]]
```

**vstack, hstack, dstack:** concatenate এর শর্টকাট। vstack=উল্লম্বে(row), hstack=অনুভূমিকে(column), dstack=depth-wise।

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"vstack:\n{np.vstack([a, b])}")
print(f"\nhstack: {np.hstack([a, b])}")
print(f"\ndstack shape: {np.dstack([a, b]).shape}")
print(f"dstack:\n{np.dstack([a, b])}")
```

Output:
```
vstack:
[[1 2 3]
 [4 5 6]]

hstack: [1 2 3 4 5 6]

dstack shape: (1, 3, 2)
dstack:
[[[1 4]
  [2 5]
  [3 6]]]
```

**2D তে vstack আর hstack:**

```python
import numpy as np

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"vstack shape {np.vstack([arr1, arr2]).shape}:\n{np.vstack([arr1, arr2])}")
print(f"\nhstack shape {np.hstack([arr1, arr2]).shape}:\n{np.hstack([arr1, arr2])}")
```

Output:
```
vstack shape (4, 2):
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

hstack shape (2, 4):
[[1 2 5 6]
 [3 4 7 8]]
```

**column_stack:** 1D arrays কে column হিসেবে জোড়ে। Feature engineering এ নতুন feature যোগের মতো কাজ।

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

result = np.column_stack([a, b, c])
print(f"column_stack:\n{result}")
print(f"shape: {result.shape}")
```

Output:
```
column_stack:
[[1 4 7]
 [2 5 8]
 [3 6 9]]
shape: (3, 3)
```

**Shape mismatch error:** জোড়া লাগানোর দিক ছাড়া বাকি dimension মিলতে হবে।

```python
import numpy as np

arr1 = np.array([[1, 2], [3, 4]])   # (2, 2)
arr2 = np.array([[5, 6, 7]])         # (1, 3)

try:
    np.concatenate([arr1, arr2], axis=0)
except ValueError as e:
    print(f"axis=0 error: {e}")

# Transpose করলে axis=1 তে জোড়া লাগানো যায়
result = np.concatenate([arr1, arr2.T], axis=1)
print(f"\narr2.T দিয়ে axis=1: shape={result.shape}")
print(result)
```

Output:
```
axis=0 error: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 2 and the array at index 1 has size 3

arr2.T দিয়ে axis=1: shape=(2, 5)
[[1 2 5]
 [3 4 6]]
```

### Split

একটা বড় অ্যারেকে ছোট ছোট অংশে ভাগ করা। Concatenation এর উল্টো। split() সমান ভাগ করে, না পারলে error দেয়। array_split() uneven ভাগও পারে।

```python
import numpy as np

arr = np.arange(12)
print(f"Original: {arr}")

# 4 সমান ভাগ
parts = np.split(arr, 4)
for i, p in enumerate(parts):
    print(f"Part {i}: {p}")

# নির্দিষ্ট indices এ ভাগ
parts2 = np.split(arr, [3, 7])
print(f"\nSplit at [3,7]:")
for i, p in enumerate(parts2):
    print(f"Part {i}: {p}")
```

Output:
```
Original: [ 0  1  2  3  4  5  6  7  8  9 10 11]
Part 0: [0 1 2]
Part 1: [3 4 5]
Part 2: [6 7 8]
Part 3: [ 9 10 11]

Split at [3,7]:
Part 0: [0 1 2]
Part 1: [3 4 5 6]
Part 2: [ 7  8  9 10 11]
```

**split vs array_split:**

```python
import numpy as np

arr = np.arange(10)

# split - সমান ভাগ না হলে error
try:
    np.split(arr, 3)
except ValueError as e:
    print(f"split(10,3) error: {e}")

# array_split - uneven ভাগও করতে পারে
result = np.array_split(arr, 3)
print(f"\narray_split(10,3):")
for i, p in enumerate(result):
    print(f"Part {i}: {p}, shape: {p.shape}")
```

Output:
```
split(10,3) error: array split does not result in an equal division

array_split(10,3):
Part 0: [0 1 2 3], shape: (4,)
Part 1: [4 5 6], shape: (3,)
Part 2: [7 8 9], shape: (3,)
```

**hsplit আর vsplit:** hsplit=অনুভূমিকে ভাগ (column ধরে), vsplit=উল্লম্বে ভাগ (row ধরে)। Training/Testing data ভাগ করতে এই concept লাগে।

```python
import numpy as np

arr = np.arange(16).reshape(4, 4)
print(f"Original:\n{arr}")

# hsplit - column ধরে ভাগ
h_parts = np.hsplit(arr, 2)
print(f"\nhsplit(2):")
for i, p in enumerate(h_parts):
    print(f"Part {i}:\n{p}")

# vsplit - row ধরে ভাগ
v_parts = np.vsplit(arr, 2)
print(f"\nvsplit(2):")
for i, p in enumerate(v_parts):
    print(f"Part {i}:\n{p}")
```

Output:
```
Original:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

hsplit(2):
Part 0:
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
Part 1:
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]

vsplit(2):
Part 0:
[[0 1 2 3]
 [4 5 6 7]]
Part 1:
[[ 8  9 10 11]
 [12 13 14 15]]
```

**Specific indices এ split:**

```python
import numpy as np

arr = np.arange(20).reshape(4, 5)
print(f"Original shape: {arr.shape}")

# hsplit at columns [2,4]
h = np.hsplit(arr, [2, 4])
print(f"\nhsplit([2,4]):")
for i, p in enumerate(h):
    print(f"Part {i} shape: {p.shape}\n{p}")

# vsplit at rows [1,3]
v = np.vsplit(arr, [1, 3])
print(f"\nvsplit([1,3]):")
for i, p in enumerate(v):
    print(f"Part {i} shape: {p.shape}\n{p}")
```

Output:
```
Original shape: (4, 5)

hsplit([2,4]):
Part 0 shape: (4, 2)
[[ 0  1]
 [ 5  6]
 [10 11]
 [15 16]]
Part 1 shape: (4, 2)
[[ 2  3]
 [ 7  8]
 [12 13]
 [17 18]]
Part 2 shape: (4, 1)
[[ 4]
 [ 9]
 [14]
 [19]]

vsplit([1,3]):
Part 0 shape: (1, 5)
[[0 1 2 3 4]]
Part 1 shape: (2, 5)
[[ 5  6  7  8  9]
 [10 11 12 13 14]]
Part 2 shape: (1, 5)
[[15 16 17 18 19]]
```

**dsplit - 3D depth-wise split:**

```python
import numpy as np

arr_3d = np.arange(24).reshape(2, 3, 4)
print(f"3D shape: {arr_3d.shape}")

d = np.dsplit(arr_3d, 2)
for i, p in enumerate(d):
    print(f"Part {i} shape: {p.shape}")
```

Output:
```
3D shape: (2, 3, 4)
Part 0 shape: (2, 3, 2)
Part 1 shape: (2, 3, 2)
```

### Append, Insert আর Delete

append=শেষে যোগ, insert=নির্দিষ্ট ইনডেক্সে যোগ, delete=নির্দিষ্ট ইনডেক্স মুছে ফেলা। এগুলো প্রতিটাই নতুন array return করে, NumPy array fixed size, in-place change হয় না।

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(f"append [6,7]: {np.append(arr, [6, 7])}")
print(f"insert at 2:  {np.insert(arr, 2, [10, 20])}")
print(f"delete 1,3:   {np.delete(arr, [1, 3])}")
```

Output:
```
append [6,7]: [1 2 3 4 5 6 7]
insert at 2:  [ 1  2 10 20  3  4  5]
delete 1,3:   [1 3 5]
```

**2D তে append, insert, delete:**

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original:\n{arr}")

# নতুন row যোগ
print(f"\nappend row:\n{np.append(arr, [[7, 8, 9]], axis=0)}")

# নতুন column যোগ
print(f"\nappend col:\n{np.append(arr, [[10], [20]], axis=1)}")

# নির্দিষ্ট row তে insert
print(f"\ninsert at row 1:\n{np.insert(arr, 1, [[7, 8, 9]], axis=0)}")

# row বা column delete
print(f"\ndelete row 0:\n{np.delete(arr, 0, axis=0)}")
print(f"\ndelete col 1:\n{np.delete(arr, 1, axis=1)}")
```

Output:
```
Original:
[[1 2 3]
 [4 5 6]]

append row:
[[1 2 3]
 [4 5 6]
 [7 8 9]]

append col:
[[ 1  2  3 10]
 [ 4  5  6 20]]

insert at row 1:
[[1 2 3]
 [7 8 9]
 [4 5 6]]

delete row 0:
[[4 5 6]]

delete col 1:
[[1 3]
 [4 6]]
```

### Unique, Repeat আর Tile

unique=ইউনিক ভ্যালু বের করা, repeat=প্রতিটা এলিমেন্ট repeat, tile=পুরো array repeat। repeat এলিমেন্ট-wise, tile array-wise কাজ করে।

```python
import numpy as np

# unique with counts
arr = np.array([3, 1, 2, 3, 2, 1, 4, 5, 4])
uniq, counts = np.unique(arr, return_counts=True)
print(f"Original: {arr}")
print(f"Unique: {uniq}")
print(f"Counts: {counts}")

# repeat vs tile
arr2 = np.array([1, 2, 3])
print(f"\nrepeat(2): {np.repeat(arr2, 2)}")
print(f"repeat([1,3,2]): {np.repeat(arr2, [1, 3, 2])}")
print(f"tile(2): {np.tile(arr2, 2)}")
print(f"tile(2,3):\n{np.tile(arr2, (2, 3))}")
```

Output:
```
Original: [3 1 2 3 2 1 4 5 4]
Unique: [1 2 3 4 5]
Counts: [2 2 2 2 1]

repeat(2): [1 1 2 2 3 3]
repeat([1,3,2]): [1 2 2 2 3 3]
tile(2): [1 2 3 1 2 3]
tile(2,3):
[[1 2 3 1 2 3 1 2 3]
 [1 2 3 1 2 3 1 2 3]]
```

### Flip, Roll আর Rot90

flip=উল্টে দেয়, roll=circular shift, rot90=90 degree rotate। ডেটা augmentation বা sequence reverse করতে এগুলো কাজে লাগে।

```python
import numpy as np

# 1D flip আর roll
arr = np.arange(10)
print(f"Original: {arr}")
print(f"flip: {np.flip(arr)}")
print(f"roll(3): {np.roll(arr, 3)}")
print(f"roll(-2): {np.roll(arr, -2)}")

# 2D flip
arr_2d = np.arange(6).reshape(2, 3)
print(f"\n2D:\n{arr_2d}")
print(f"flip(axis=0):\n{np.flip(arr_2d, axis=0)}")
print(f"flip(axis=1):\n{np.flip(arr_2d, axis=1)}")

# rot90
arr_sq = np.arange(9).reshape(3, 3)
print(f"\nOriginal:\n{arr_sq}")
print(f"rot90:\n{np.rot90(arr_sq)}")
print(f"rot90(k=2):\n{np.rot90(arr_sq, k=2)}")
```

Output:
```
Original: [0 1 2 3 4 5 6 7 8 9]
flip: [9 8 7 6 5 4 3 2 1 0]
roll(3): [7 8 9 0 1 2 3 4 5 6]
roll(-2): [2 3 4 5 6 7 8 9 0 1]

2D:
[[0 1 2]
 [3 4 5]]
flip(axis=0):
[[3 4 5]
 [0 1 2]]
flip(axis=1):
[[2 1 0]
 [5 4 3]]

Original:
[[0 1 2]
 [3 4 5]
 [6 7 8]]
rot90:
[[2 5 8]
 [1 4 7]
 [0 3 6]]
rot90(k=2):
[[8 7 6]
 [5 4 3]
 [2 1 0]]
```

### সারসংক্ষেপ

| মেথড | কাজ |
|---|---|
| `reshape()` | শেপ পরিবর্তন (এলিমেন্ট সমান) |
| `resize()` | শেপ পরিবর্তন (repeat/truncate) |
| `np.newaxis` | ডাইমেনশন যোগ |
| `.T` | Transpose |
| `transpose()` | Transpose with axes control |
| `swapaxes()` | দুইটা axis অদলবদল |
| `flatten()` | 1D কপি |
| `ravel()` | 1D ভিউ |
| `concatenate()` | axis দিয়ে জোড়া |
| `vstack()` | row-wise জোড়া |
| `hstack()` | column-wise জোড়া |
| `dstack()` | depth-wise জোড়া |
| `column_stack()` | 1D কে column হিসেবে জোড়া |
| `split()` | সমান ভাগ |
| `array_split()` | uneven ভাগও সম্ভব |
| `hsplit()` | column ধরে ভাগ |
| `vsplit()` | row ধরে ভাগ |
| `dsplit()` | depth-wise ভাগ |
| `append()` | শেষে যোগ |
| `insert()` | নির্দিষ্ট ইনডেক্সে যোগ |
| `delete()` | নির্দিষ্ট ইনডেক্স মুছে ফেলা |
| `unique()` | ইউনিক ভ্যালু |
| `repeat()` | এলিমেন্ট-wise repeat |
| `tile()` | array-wise repeat |
| `flip()` | উল্টে দেয় |
| `roll()` | circular shift |
| `rot90()` | 90 degree rotate |
