## NumPy ইনডেক্সিং ও স্লাইসিং

### কেন ইনডেক্সিং শিখবো?

ডেটা সায়েন্স এ সবচেয়ে বেশি যেটা করতে হয়, সেটা হলো ডেটা থেকে নির্দিষ্ট অংশ বের করা। একটা ডেটাসেটে হাজার রো আর কলাম থাকে, সবটা একসাথে দেখা হয় না, নির্দিষ্ট কলাম বা রো বের করা হয়। আর সেটা করতে হলে ইনডেক্সিং আর স্লাইসিং জানতে হবে।

### ১D Array ইনডেক্সিং

১D array এর ইনডেক্সিং Python list এর মতোই, কিন্তু আরো ফিচার আছে:

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Single element access
print(f"arr[0] = {arr[0]}")
print(f"arr[4] = {arr[4]}")
print(f"arr[-1] = {arr[-1]}")
print(f"arr[-3] = {arr[-3]}")
```

Output:
```
arr[0] = 10
arr[4] = 50
arr[-1] = 100
arr[-3] = 80
```

### ১D Array স্লাইসিং

স্লাইসিং এর সিনট্যাক্স: `arr[start:stop:step]`

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Basic slicing
print(f"arr[2:5] = {arr[2:5]}")
print(f"arr[:5] = {arr[:5]}")
print(f"arr[5:] = {arr[5:]}")
print(f"arr[::2] = {arr[::2]}")
print(f"arr[::-1] = {arr[::-1]}")
print(f"arr[1:8:3] = {arr[1:8:3]}")
```

Output:
```
arr[2:5] = [30 40 50]
arr[:5] = [10 20 30 40 50]
arr[5:] = [60 70 80 90 100]
arr[::2] = [10 30 50 70 90]
arr[::-1] = [100 90 80 70 60 50 40 30 20 10]
arr[1:8:3] = [20 50 80]
```

### ২D Array ইনডেক্সিং

২D array তে ইনডেক্সিং হলো `[row, column]`:

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"Array:\n{arr}")
print(f"\narr[0, 0] = {arr[0, 0]}")
print(f"arr[1, 2] = {arr[1, 2]}")
print(f"arr[2, 3] = {arr[2, 3]}")
print(f"arr[-1, -1] = {arr[-1, -1]}")

# Entire row
print(f"\narr[0] = {arr[0]}")
print(f"arr[1] = {arr[1]}")

# Entire column
print(f"\narr[:, 0] = {arr[:, 0]}")
print(f"arr[:, 2] = {arr[:, 2]}")
```

Output:
```
Array:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

arr[0, 0] = 1
arr[1, 2] = 7
arr[2, 3] = 12
arr[-1, -1] = 12

arr[0] = [1 2 3 4]
arr[1] = [5 6 7 8]

arr[:, 0] = [1 5 9]
arr[:, 2] = [3 7 11]
```

### ২D Array স্লাইসিং

২D স্লাইসিং হলো `[row_start:row_stop, col_start:col_stop]`:

```python
import numpy as np

arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20]])

print(f"Array (4x5):\n{arr}")

# First two rows, first three columns
print(f"\narr[:2, :3]:\n{arr[:2, :3]}")

# Middle section
print(f"\narr[1:3, 2:4]:\n{arr[1:3, 2:4]}")

# Last two rows
print(f"\narr[2:, :]:\n{arr[2:, :]}")

# Every second column
print(f"\narr[:, ::2]:\n{arr[:, ::2]}")
```

Output:
```
Array (4x5):
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]
 [16 17 18 19 20]]

arr[:2, :3]:
[[1 2 3]
 [6 7 8]]

arr[1:3, 2:4]:
[[ 8  9]
 [13 14]]

arr[2:, :]:
[[11 12 13 14 15]
 [16 17 18 19 20]]

arr[:, ::2]:
[[ 1  3  5]
 [ 6  8 10]
 [11 13 15]
 [16 18 20]]
```

### বুলিয়ান ইনডেক্সিং

এটা অত্যন্ত শক্তিশালী ফিচার। শর্ত দিয়ে এলিমেন্ট ফিল্টার করা যায়:

```python
import numpy as np

arr = np.array([15, 25, 8, 45, 60, 12, 35, 50, 5, 30])

# Which elements are greater than 30?
mask = arr > 30
print(f"mask: {mask}")
print(f"arr[mask]: {arr[mask]}")

# Multiple conditions
mask2 = (arr > 20) & (arr < 50)
print(f"\nBetween 20-50: {arr[mask2]}")

mask3 = (arr < 10) | (arr > 50)
print(f"Less than 10 or greater than 50: {arr[mask3]}")

# Modify values based on condition
arr_modified = arr.copy()
arr_modified[arr_modified > 30] = 30
print(f"\nValues > 30 capped to 30: {arr_modified}")
```

Output:
```
mask: [False False False  True  True False  True  True False False]
arr[mask]: [45 60 35 50]

Between 20-50: [25 45 35]

Less than 10 or greater than 50: [ 8 60  5]

Values > 30 capped to 30: [15 25  8 30 30 12 30 30  5 30]
```

### ফ্যান্সি ইনডেক্সিং

একাধিক নির্দিষ্ট ইনডেক্স থেকে এলিমেন্ট বের করা:

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70])

# Specific indices
indices = [0, 2, 5]
print(f"arr[0,2,5]: {arr[indices]}")

# Fancy indexing in 2D
arr_2d = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

rows = [0, 1, 2]
cols = [2, 1, 0]
print(f"\narr_2d[rows, cols]: {arr_2d[rows, cols]}")
# Returns values at (0,2), (1,1), (2,0)

# Specific rows with all columns
print(f"arr_2d[[0,2]]:\n{arr_2d[[0, 2]]}")
```

Output:
```
arr[0,2,5]: [10 30 60]

arr_2d[rows, cols]: [3 5 7]
arr_2d[[0,2]]:
[[1 2 3]
 [7 8 9]]
```

### ভিউ বনাম কপি

NumPy স্লাইসিং এ একটা গুরুত্বপূর্ণ বিষয় হলো ভিউ আর কপি:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Slice is a view (reference to original)
view_slice = arr[1:4]
print(f"Original: {arr}")
print(f"View: {view_slice}")

view_slice[0] = 100
print(f"\nAfter modifying view:")
print(f"Original: {arr}")
print(f"View: {view_slice}")

# Copy (brand new array)
arr2 = np.array([1, 2, 3, 4, 5])
copy_slice = arr2[1:4].copy()
copy_slice[0] = 100
print(f"\nAfter modifying copy:")
print(f"Original: {arr2}")
print(f"Copy: {copy_slice}")
```

Output:
```
Original: [1 2 3 4 5]
View: [2 3 4]

After modifying view:
Original: [  1 100   3   4   5]
View: [100   3   4]

After modifying copy:
Original: [1 2 3 4 5]
Copy: [100   3   4]
```

এটা অনেক বাগ এর কারণ হতে পারে। ভাবা হচ্ছে আসল array পরিবর্তন হচ্ছে না, কিন্তু ভিউ পরিবর্তন করলে আসল array ও বদলে যাচ্ছে। সেফ কাজ করতে চাইলে `.copy()` ব্যবহার করতে হবে।

### সারসংক্ষেপ

| মেথড | ব্যবহার | উদাহরণ |
|---|---|---|
| বেসিক ইনডেক্সিং | একটা এলিমেন্ট | `arr[2]` |
| বেসিক স্লাইসিং | একাংশ | `arr[1:5]` |
| ২D ইনডেক্সিং | রো, কলাম | `arr[0, 2]` |
| ২D স্লাইসিং | সাব-ম্যাট্রিক্স | `arr[:2, 1:3]` |
| বুলিয়ান | শর্ত দিয়ে ফিল্টার | `arr[arr > 30]` |
| ফ্যান্সি | নির্দিষ্ট ইনডেক্স | `arr[[0, 3, 5]]` |
| `.copy()` | নতুন মেমরি | `arr[1:4].copy()` |

ইনডেক্সিং আর স্লাইসিং হলো সেই জিনিস যেটা প্রায় প্রতিদিন ব্যবহার হয়। ২D স্লাইসিং এ গুলিয়ে যাওয়া সাধারণ, কোনটা রো, কোনটা কলাম। একটা সহজ নিয়ম মনে রাখতে হবে: প্রথমটা রো, দ্বিতীয়টা কলাম। `arr[রো, কলাম]`। আর বুলিয়ান ইনডেক্সিং হলো সবচেয়ে বেশি ব্যবহৃত, কারণ রিয়েল ডেটায় সবসময় শর্ত দিয়ে ফিল্টার করতে হয়।
