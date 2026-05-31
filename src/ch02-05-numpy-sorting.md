## NumPy সর্টিং, সার্চিং ও Array Manipulation

### কেন এই বিষয়গুলো শিখবো?

রিয়েল ওয়ার্ল্ড ডেটা কখনো সাজানো থাকে না। ডেটা সর্ট করতে হয়, নির্দিষ্ট মান খুঁজে বের করতে হয়, array জয়েন করতে হয়, বা ভাগ করতে হয়। এই সব কাজের জন্য NumPy তে অনেক ফাংশন আছে। এগুলো না জানলে, কাজ করতে গিয়ে অনেক সময় নষ্ট হবে।

### সর্টিং

```python
import numpy as np

# ১D সর্টিং
arr = np.array([42, 17, 8, 95, 23, 61, 34])
print(f"আসল: {arr}")
print(f"np.sort(): {np.sort(arr)}")
print(f"আসল (অপরিবর্তিত): {arr}")

# ইন-প্লেস সর্ট
arr_copy = arr.copy()
arr_copy.sort()
print(f"ইন-প্লেস sort(): {arr_copy}")

# সর্ট করা ইনডেক্স
indices = np.argsort(arr)
print(f"\nargsort (ইনডেক্স): {indices}")
print(f"সর্ট করা মান: {arr[indices]}")

# বিপরীত ক্রম
print(f"বিপরীত ক্রম: {np.sort(arr)[::-1]}")
```

Output:
```
আসল: [42 17  8 95 23 61 34]
np.sort(): [ 8 17 23 34 42 61 95]
আসল (অপরিবর্তিত): [42 17  8 95 23 61 34]
ইন-প্লেস sort(): [ 8 17 23 34 42 61 95]

argsort (ইনডেক্স): [2 1 4 6 0 5 3]
সর্ট করা মান: [ 8 17 23 34 42 61 95]
বিপরীত ক্রম: [95 61 42 34 23 17  8]
```

### ২D সর্টিং

```python
import numpy as np

arr = np.array([[3, 1, 2],
                [6, 4, 5],
                [9, 7, 8]])

print(f"Array:\n{arr}")

# প্রতি রো সর্ট
print(f"\naxis=1 (প্রতি রো):\n{np.sort(arr, axis=1)}")

# প্রতি কলাম সর্ট
print(f"\naxis=0 (প্রতি কলাম):\n{np.sort(arr, axis=0)}")
```

Output:
```
Array:
[[3 1 2]
 [6 4 5]
 [9 7 8]]

axis=1 (প্রতি রো):
[[1 2 3]
 [4 5 6]
 [7 8 9]]

axis=0 (প্রতি কলাম):
[[3 1 2]
 [6 4 5]
 [9 7 8]]
```

### সার্চিং

```python
import numpy as np

arr = np.array([1, 5, 3, 8, 2, 7, 4, 9, 6])

# সর্বোচ্চ আর সর্বনিম্ন
print(f"max: {np.max(arr)}, index: {np.argmax(arr)}")
print(f"min: {np.min(arr)}, index: {np.argmin(arr)}")

# শর্ত অনুযায়ী ইনডেক্স
indices = np.where(arr > 5)
print(f"\n৫ এর বেশি মানের ইনডেক্স: {indices[0]}")
print(f"৫ এর বেশি মান: {arr[indices]}")

# searchsorted - সর্ট করা array তে নতুন মান কোথায় বসবে
sorted_arr = np.sort(arr)
new_val = 5
pos = np.searchsorted(sorted_arr, new_val)
print(f"\nসর্ট করা: {sorted_arr}")
print(f"{new_val} বসবে ইনডেক্স {pos} তে")

# nonzero - শূন্য না এমন ইনডেক্স
arr2 = np.array([0, 3, 0, 5, 0, 7, 0])
print(f"\nnonzero ইনডেক্স: {np.nonzero(arr2)}")
```

Output:
```
max: 9, index: 7
min: 1, index: 0

৫ এর বেশি মানের ইনডেক্স: [3 5 7 8]
৫ এর বেশি মান: [8 7 9 6]

সর্ট করা: [1 2 3 4 5 6 7 8 9]
5 বসবে ইনডেক্স 4 তে

nonzero ইনডেক্স: (array([1, 3, 5]),)
```

### Array জয়েন (Concatenation)

```python
import numpy as np

# ১D কনক্যাটেনেশন
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"np.concatenate([a, b]): {np.concatenate([a, b])}")
print(f"np.hstack([a, b]): {np.hstack([a, b])}")

# ২D কনক্যাটেনেশন
a2d = np.array([[1, 2], [3, 4]])
b2d = np.array([[5, 6], [7, 8]])

print(f"\naxis=0 (রো যোগ):\n{np.concatenate([a2d, b2d], axis=0)}")
print(f"\naxis=1 (কলাম যোগ):\n{np.concatenate([a2d, b2d], axis=1)}")

# vstack আর hstack
print(f"\nvstack:\n{np.vstack([a2d, b2d])}")
print(f"\nhstack:\n{np.hstack([a2d, b2d])}")
```

Output:
```
np.concatenate([a, b]): [1 2 3 4 5 6]
np.hstack([a, b]): [1 2 3 4 5 6]

axis=0 (রো যোগ):
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

axis=1 (কলাম যোগ):
[[1 2 5 6]
 [3 4 7 8]]

vstack:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

hstack:
[[1 2 5 6]
 [3 4 7 8]]
```

### Array স্প্লিট

```python
import numpy as np

# ১D স্প্লিট
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

result = np.split(arr, 3)
print(f"৩ ভাগে স্প্লিট: {result}")

result2 = np.array_split(arr, 4)
print(f"৪ ভাগে স্প্লিট (অসমান): {result2}")

# নির্দিষ্ট পজিশনে স্প্লিট
result3 = np.split(arr, [3, 5])
print(f"ইনডেক্স ৩ আর ৫ তে স্প্লিট: {result3}")

# ২D স্প্লিট
arr2d = np.arange(16).reshape(4, 4)
print(f"\n২D Array:\n{arr2d}")

top, bottom = np.vsplit(arr2d, 2)
print(f"\nvsplit উপর:\n{top}")
print(f"vsplit নিচ:\n{bottom}")

left, right = np.hsplit(arr2d, 2)
print(f"\nhsplit বাম:\n{left}")
print(f"hsplit ডান:\n{right}")
```

Output:
```
৩ ভাগে স্প্লিট: [array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
৪ ভাগে স্প্লিট (অসমান): [array([1, 2, 3]), array([4, 5]), array([6, 7]), array([8, 9])]
ইনডেক্স ৩ আর ৫ তে স্প্লিট: [array([1, 2, 3]), array([4, 5]), array([6, 7, 8, 9])]

২D Array:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

vsplit উপর:
[[0 1 2 3]
 [4 5 6 7]]
vsplit নিচ:
[[ 8  9 10 11]
 [12 13 14 15]]

hsplit বাম:
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
hsplit ডান:
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]
```

### Array ম্যানিপুলেশন

```python
import numpy as np

# ট্রান্সপোজ
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"আসল:\n{arr}")
print(f"ট্রান্সপোজ:\n{arr.T}")

# রিশেপ
arr1d = np.arange(12)
print(f"\n১D: {arr1d}")
print(f"reshape(3,4):\n{arr1d.reshape(3, 4)}")
print(f"reshape(4,3):\n{arr1d.reshape(4, 3)}")
print(f"reshape(2,6):\n{arr1d.reshape(2, 6)}")

# টাইল (পুনরাবৃত্তি)
arr_small = np.array([[1, 2], [3, 4]])
tiled = np.tile(arr_small, (2, 3))
print(f"\nটাইল (2,3):\n{tiled}")

# রিপিট
arr_simple = np.array([1, 2, 3])
repeated = np.repeat(arr_simple, 3)
print(f"\nrepeat(3): {repeated}")

# ফ্লিপ
arr = np.array([1, 2, 3, 4, 5])
print(f"\nflip: {np.flip(arr)}")

# রোল
print(f"roll(2): {np.roll(arr, 2)}")
```

Output:
```
আসল:
[[1 2 3]
 [4 5 6]]
ট্রান্সপোজ:
[[1 4]
 [2 5]
 [3 6]]

১D: [ 0  1  2  3  4  5  6  7  8  9 10 11]
reshape(3,4):
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
reshape(4,3):
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
reshape(2,6):
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]]

টাইল (2,3):
[[1 2 1 2 1 2]
 [3 4 3 4 3 4]
 [1 2 1 2 1 2]
 [3 4 3 4 3 4]]

repeat(3): [1 1 1 2 2 2 3 3 3]

flip: [5 4 3 2 1]
roll(2): [4 5 1 2 3]
```

### ইউনিক আর বিনকাউন্ট

```python
import numpy as np

# ইউনিক মান
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique_vals, counts = np.unique(arr, return_counts=True)
print(f"ইউনিক মান: {unique_vals}")
print(f"গণনা: {counts}")

# bincount
arr2 = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
print(f"\nbincount: {np.bincount(arr2)}")
```

Output:
```
ইউনিক মান: [1 2 3 4]
গণনা: [1 2 3 4]

bincount: [1 2 3 4]
```

### সারসংক্ষেপ

| ফাংশন | কাজ |
|---|---|
| `np.sort()` | সর্ট (নতুন array) |
| `arr.sort()` | ইন-প্লেস সর্ট |
| `np.argsort()` | সর্ট করা ইনডেক্স |
| `np.where()` | শর্ত অনুযায়ী ইনডেক্স |
| `np.argmax()` / `np.argmin()` | সর্বোচ্চ/নিম্ন ইনডেক্স |
| `np.concatenate()` | জয়েন |
| `np.vstack()` / `np.hstack()` | উল্লম্ব/সমান্তরাল জয়েন |
| `np.split()` | ভাগ |
| `np.vsplit()` / `np.hsplit()` | উল্লম্ব/সমান্তরাল ভাগ |
| `np.unique()` | ইউনিক মান |
| `np.tile()` / `np.repeat()` | পুনরাবৃত্তি |
| `np.flip()` / `np.roll()` | উল্টানো / ঘোরানো |

সর্টিং আর সার্চিং মনে রাখা সহজ, কিন্তু concatenate আর split এ axis গুলিয়ে ফেলা সাধারণ। একটা ট্রিক মনে রাখতে হবে: vstack মানে vertical, রো যোগ হয় (axis=0)। hstack মানে horizontal, কলাম যোগ হয় (axis=1)। vsplit আর hsplit ও একই নিয়ম।

---
