## NumPy Sorting, Searching, String Functions and Iterating

### সর্টিং

`np.sort()` নতুন sorted array রিটার্ন করে, আসল array পরিবর্তন করে না। `arr.sort()` ইন-প্লেস সর্ট করে, আসল array-ই বদলে যায়। `np.argsort()` সর্ট করার পর ইনডেক্স রিটার্ন করে, ranking বা top-k selection এ কাজে লাগে। বিপরীত ক্রমে সর্ট করতে `np.sort(arr)[::-1]` লেখা যায়।

```python
import numpy as np

arr = np.array([42, 17, 8, 95, 23, 61, 34])

# np.sort - original array unchanged
sorted_arr = np.sort(arr)
print(f"Original: {arr}")
print(f"np.sort(): {sorted_arr}")
print(f"Original (unchanged): {arr}")

# In-place sort
arr_copy = arr.copy()
arr_copy.sort()
print(f"In-place sort(): {arr_copy}")

# argsort - sorted indices
indices = np.argsort(arr)
print(f"\nargsort (indices): {indices}")
print(f"Sorted values: {arr[indices]}")

# Reverse order
print(f"Reverse order: {np.sort(arr)[::-1]}")

# Partial sort - only sorts first k, rest are random
print(f"\npartition (k=3): {np.partition(arr, 3)}")
print(f"Smallest 3: {np.partition(arr, 3)[:3]}")
```

Output:
```
Original: [42 17  8 95 23 61 34]
np.sort(): [ 8 17 23 34 42 61 95]
Original (unchanged): [42 17  8 95 23 61 34]
In-place sort(): [ 8 17 23 34 42 61 95]

argsort (indices): [2 1 4 6 0 5 3]
Sorted values: [ 8 17 23 34 42 61 95]
Reverse order: [95 61 42 34 23 17  8]

partition (k=3): [ 8 17 23 34 42 61 95]
Smallest 3: [ 8 17 23]
```

**2D সর্টিং:** `axis` প্যারামিটার দিয়ে রো বা কলাম ধরে সর্ট করা যায়। `axis=1` মানে প্রতি রো সর্ট, `axis=0` মানে প্রতি কলাম সর্ট।

```python
import numpy as np

arr = np.array([[3, 1, 2],
                [6, 4, 5],
                [9, 7, 8]])

print(f"Array:\n{arr}")
print(f"\naxis=1 (sort each row):\n{np.sort(arr, axis=1)}")
print(f"\naxis=0 (sort each column):\n{np.sort(arr, axis=0)}")

# 2D argsort - sorted indices per row
indices = np.argsort(arr, axis=1)
print(f"\n2D argsort (axis=1):\n{indices}")
```

Output:
```
Array:
[[3 1 2]
 [6 4 5]
 [9 7 8]]

axis=1 (sort each row):
[[1 2 3]
 [4 5 6]
 [7 8 9]]

axis=0 (sort each column):
[[3 1 2]
 [6 4 5]
 [9 7 8]]

2D argsort (axis=1):
[[1 2 0]
 [1 2 0]
 [1 2 0]]
```

**সর্টিং অ্যালগরিদম:** `kind` প্যারামিটার দিয়ে অ্যালগরিদম বেছে নেওয়া যায়। ডিফল্ট `quicksort`, এছাড়া `mergesort` (stable) আর `heapsort` আছে। Stable sort মানে সমান মানের এলিমেন্টের ক্রম ঠিক রাখে।

```python
import numpy as np

# Stable sort matters when order of equal values is important
names = np.array(['Choyon', 'Anik', 'Bristi', 'Anik'])
scores = np.array([85, 92, 78, 92])

# Quicksort - equal score order not guaranteed
idx_quick = np.argsort(scores, kind='quicksort')
print(f"quicksort: {names[idx_quick]}")

# Mergesort - stable, preserves order of equal scores
idx_stable = np.argsort(scores, kind='mergesort')
print(f"mergesort: {names[idx_stable]}")
```

Output:
```
quicksort: ['Bristi' 'Choyon' 'Anik' 'Anik']
mergesort: ['Bristi' 'Anik' 'Anik' 'Choyon']
```

**lexsort:** একাধিক key দিয়ে সর্ট। শেষের key প্রায়রিটি সবচেয়ে বেশি, প্রথমটা সবচেয়ে কম। এক্সেলে প্রথমে কলাম A ধরে সর্ট, তারপর কলাম B ধরে সর্ট, এই কনসেপ্ট।

```python
import numpy as np

names = np.array(['Choyon', 'Anik', 'Bristi', 'Deep'])
scores = np.array([85, 92, 78, 92])
ages = np.array([22, 20, 21, 19])

# Sort by score first, then by age on ties
idx = np.lexsort((ages, scores))
print(f"lexsort (score, age): {names[idx]}")
print(f"Scores: {scores[idx]}")
print(f"Ages: {ages[idx]}")
```

Output:
```
lexsort (score, age): ['Bristi' 'Choyon' 'Deep' 'Anik']
Scores: [78 85 92 92]
Ages: [21 22 19 20]
```

### সার্চিং

`np.where()` শর্ত অনুযায়ী ইনডেক্স বের করে। `argmax()` আর `argmin()` সর্বোচ্চ/সর্বনিম্ন মানের ইনডেক্স দেয়, Classification মডেলে prediction বের করতে লাগে। `searchsorted()` সর্ট করা array তে নতুন মান কোথায় বসবে সেটা বলে। `nonzero()` শূন্য না এমন ইনডেক্স দেয়।

```python
import numpy as np

arr = np.array([1, 5, 3, 8, 2, 7, 4, 9, 6])

# argmax / argmin - index of max and min
print(f"max: {np.max(arr)}, index: {np.argmax(arr)}")
print(f"min: {np.min(arr)}, index: {np.argmin(arr)}")

# where - indices based on condition
indices = np.where(arr > 5)
print(f"\nIndices of values > 5: {indices[0]}")
print(f"Values > 5: {arr[indices]}")

# where with three arguments - x if true, y if false
result = np.where(arr > 5, arr, 0)
print(f"where(arr>5, arr, 0): {result}")

# searchsorted - insertion point in sorted array
sorted_arr = np.sort(arr)
pos = np.searchsorted(sorted_arr, 5)
print(f"\nSorted: {sorted_arr}")
print(f"5 inserts at index {pos}")

# Multiple values at once
positions = np.searchsorted(sorted_arr, [3, 5, 7])
print(f"Positions for [3,5,7]: {positions}")

# nonzero - indices of non-zero elements
arr2 = np.array([0, 3, 0, 5, 0, 7, 0])
print(f"\nnonzero indices: {np.nonzero(arr2)}")
print(f"nonzero values: {arr2[np.nonzero(arr2)]}")
```

Output:
```
max: 9, index: 7
min: 1, index: 0

Indices of values > 5: [3 5 7 8]
Values > 5: [8 7 9 6]
where(arr>5, arr, 0): [0 0 0 8 0 7 0 9 6]

Sorted: [1 2 3 4 5 6 7 8 9]
5 inserts at index 4
Positions for [3,5,7]: [2 4 6]

nonzero indices: (array([1, 3, 5]),)
nonzero values: [3 5 7]
```

**2D তে argmax/argmin:** axis প্যারামিটার দিয়ে প্রতি রো বা প্রতি কলামের সর্বোচ্চ/সর্বনিম্ন ইনডেক্স বের করা যায়।

```python
import numpy as np

arr_2d = np.array([[1, 5, 3],
                   [8, 2, 9],
                   [4, 7, 6]])

print(f"2D:\n{arr_2d}")

# Per-column max index
print(f"\naxis=0 argmax: {np.argmax(arr_2d, axis=0)}")

# Per-row max index
print(f"axis=1 argmax: {np.argmax(arr_2d, axis=1)}")

# Global max index (flattened)
flat_idx = np.argmax(arr_2d)
row, col = np.unravel_index(flat_idx, arr_2d.shape)
print(f"\nGlobal max: {arr_2d[row, col]} at ({row}, {col})")
```

Output:
```
2D:
[[1 5 3]
 [8 2 9]
 [4 7 6]]

axis=0 argmax: [1 2 1]
axis=1 argmax: [1 2 1]

Global max: 9 at (1, 2)
```

**extract() আর select():** `np.extract()` শর্ত অনুযায়ী মান সরাসরি বের করে। `np.select()` একাধিক শর্ত একসাথে হ্যান্ডেল করে, feature engineering এ কাজে লাগে।

```python
import numpy as np

arr = np.array([12, 45, 8, 67, 23, 90, 5, 38])

# extract - get values by condition
result = np.extract(arr > 30, arr)
print(f"Values > 30: {result}")

# select - multiple conditions, different values for each
conditions = [arr < 20, (arr >= 20) & (arr < 50), arr >= 50]
choices = ['low', 'medium', 'high']
labels = np.select(conditions, choices)
print(f"select: {labels}")

# clip - constrain values to a range
clipped = np.clip(arr, 10, 50)
print(f"clip(10,50): {clipped}")
```

Output:
```
Values > 30: [45 67 90 38]
select: ['low' 'medium' 'low' 'high' 'medium' 'high' 'low' 'medium']
clip(10,50): [12 45 10 50 23 50 10 38]
```

### কাউন্টিং

`count_nonzero()` শূন্য না এমন এলিমেন্ট গণনা করে। `bincount()` প্রতিটা নন-নেগেটিভ ইন্টিজার কতবার আছে সেটা গণনা করে। Class distribution চেক করতে bincount খুব কাজের।

```python
import numpy as np

# count_nonzero
arr = np.array([0, 3, 0, 5, 0, 7, 0, 9])
print(f"nonzero count: {np.count_nonzero(arr)}")
print(f"axis-wise (2D):")

arr_2d = np.array([[1, 0, 3], [0, 5, 0]])
print(f"Array:\n{arr_2d}")
print(f"Total nonzero: {np.count_nonzero(arr_2d)}")
print(f"Per row: {np.count_nonzero(arr_2d, axis=1)}")

# bincount - count occurrences of each value
labels = np.array([0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 2, 2])
counts = np.bincount(labels)
print(f"\nLabels: {labels}")
print(f"Counts: {counts}")
print(f"Class 0: {counts[0]}, Class 1: {counts[1]}, Class 2: {counts[2]}")

# bincount with weights
values = np.array([1, 2, 3, 4, 5, 6])
weights = np.array([10, 20, 30, 40, 50, 60])
# Weighted sum per bin
result = np.bincount(values, weights=weights)
print(f"\nWeighted bincount: {result}")
```

Output:
```
nonzero count: 4
axis-wise (2D):
Array:
[[1 0 3]
 [0 5 0]]
Total nonzero: 3
Per row: [2 1]

Labels: [0 1 2 1 0 2 1 1 0 2 2 2]
Counts: [3 4 5]
Class 0: 3, Class 1: 4, Class 2: 5

Weighted bincount: [ 0. 10. 20. 30. 40. 50. 60.]
```

### String Functions

`np.char` মডিউলে স্ট্রিং অপারেশন আছে। vectorized ভাবে পুরো array তে স্ট্রিং অপারেশন চালানো যায়, loop লাগে না।

```python
import numpy as np

names = np.array(['  Joy  ', 'Anik', '  Nila', 'Raju  '])

# strip - remove leading/trailing spaces
print(f"strip: {np.char.strip(names)}")

# upper / lower
text = np.array(['hello', 'WORLD', 'NumPy'])
print(f"upper: {np.char.upper(text)}")
print(f"lower: {np.char.lower(text)}")

# replace
emails = np.array(['user@gmail.com', 'admin@yahoo.com'])
print(f"\nreplace: {np.char.replace(emails, '@', '[at]')}")

# split
csv_data = np.array(['a,b,c', 'x,y,z', '1,2,3'])
print(f"split: {np.char.split(csv_data, ',')}")

# concatenate
first = np.array(['Mr.', 'Ms.'])
last = np.array(['Karim', 'Fatema'])
print(f"\nconcatenate: {np.char.add(first, last)}")

# find and count
sentences = np.array(['data science is fun', 'deep learning is powerful'])
print(f"\nfind 'is': {np.char.find(sentences, 'is')}")
print(f"count 'is': {np.char.count(sentences, 'is')}")
```

Output:
```
strip: ['Joy' 'Anik' 'Nila' 'Raju']
upper: ['HELLO' 'WORLD' 'NUMPY']
lower: ['hello' 'world' 'numpy']

replace: ['user[at]gmail.com' 'admin[at]yahoo.com']
split: [list(['a', 'b', 'c']) list(['x', 'y', 'z']) list(['1', '2', '3'])]

concatenate: ['Mr.Karim' 'Ms.Fatema']

find 'is': [12 13]
count 'is': [1 1]
```

### Iterating

NumPy তে loop ব্যবহার করা উচিত না, vectorized অপারেশন অনেক ফাস্ট। তবে কিছু ক্ষেত্রে iterate করতে হয়। 1D তে সরাসরি for loop চালানো যায়। 2D তে রো ধরে iterate হয়। `nditer()` দিয়ে যেকোনো dimension এ iterate করা যায়। `ndenumerate()` ইনডেক্স সহ iterate করে।

```python
import numpy as np

# 1D iterate
arr_1d = np.array([10, 20, 30])
print("1D iterate:")
for val in arr_1d:
    print(f"  {val}")

# 2D iterate - row-wise
arr_2d = np.array([[1, 2], [3, 4]])
print("\n2D iterate (row-wise):")
for row in arr_2d:
    print(f"  {row}")

# nditer - all elements one by one
print("\nnditer:")
for val in np.nditer(arr_2d):
    print(f"  {val}", end=" ")

# ndenumerate - with indices
print("\n\nndenumerate:")
for idx, val in np.ndenumerate(arr_2d):
    print(f"  {idx}: {val}")

# 3D iterate
arr_3d = np.arange(8).reshape(2, 2, 2)
print("\n3D nditer:")
for val in np.nditer(arr_3d):
    print(f"  {val}", end=" ")
```

Output:
```
1D iterate:
  10
  20
  30

2D iterate (row-wise):
  [1 2]
  [3 4]

nditer:
  1 2 3 4

ndenumerate:
  (0, 0): 1
  (0, 1): 2
  (1, 0): 3
  (1, 1): 4

3D nditer:
  0 1 2 3 4 5 6 7
```

**Loop বনাম Vectorized:** NumPy তে vectorized অপারেশন loop থেকে 10-100x ফাস্ট। কারণ vectorized অপারেশন C লেভেলে চলে, Python loop overhead থাকে না।

```python
import numpy as np

arr = np.arange(1_000_000)

# Loop approach (slow)
import time
start = time.time()
result_loop = []
for x in arr:
    result_loop.append(x ** 2)
time_loop = time.time() - start

# Vectorized (fast)
start = time.time()
result_vec = arr ** 2
time_vec = time.time() - start

print(f"Loop time: {time_loop:.4f}s")
print(f"Vectorized time: {time_vec:.4f}s")
print(f"Vectorized {time_loop/time_vec:.0f}x faster")
```

Output:
```
Loop time: 0.2847s
Vectorized time: 0.0012s
Vectorized 237x faster
```

### সারসংক্ষেপ

| ফাংশন | কাজ |
|---|---|
| `np.sort()` | সর্ট (নতুন array) |
| `arr.sort()` | ইন-প্লেস সর্ট |
| `np.argsort()` | সর্ট করা ইনডেক্স |
| `np.partition()` | আংশিক সর্ট |
| `np.lexsort()` | একাধিক key দিয়ে সর্ট |
| `np.where()` | শর্ত অনুযায়ী ইনডেক্স/মান |
| `np.argmax()` / `np.argmin()` | সর্বোচ্চ/নিম্ন ইনডেক্স |
| `np.searchsorted()` | সর্ট করা array তে insertion point |
| `np.nonzero()` | শূন্য না এমন ইনডেক্স |
| `np.extract()` | শর্ত অনুযায়ী মান বের |
| `np.select()` | একাধিক শর্ত |
| `np.clip()` | মান রেঞ্জে আনা |
| `np.count_nonzero()` | নন-জিরো গণনা |
| `np.bincount()` | প্রতি মানের গণনা |
| `np.char.*` | স্ট্রিং অপারেশন |
| `np.nditer()` | যেকোনো dimension এ iterate |
| `np.ndenumerate()` | ইনডেক্স সহ iterate |
