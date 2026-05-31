## Part 3: NumPy পরিচিতি ও ndarray

### কেন NumPy শিখবো?

Python list দিয়ে সব করা যায় মনে হতে পারে। হাজারটা সংখ্যা যোগ করতে হলে for loop চালালেই হয়। কিন্তু মিলিয়ন সংখ্যার হিসাব করতে গেলে Python list অনেক ধীর হয়ে যায়। সেখানেই NumPy আসে।

NumPy (Numerical Python) হলো Python এর সবচেয়ে গুরুত্বপূর্ণ লাইব্রেরি একটা, বিশেষ করে ডেটা সায়েন্স আর মেশিন লার্নিং এর জন্য। এটা দ্রুত, মেমরি ইফিশিয়েন্ট, আর বড় ডেটা নিয়ে কাজ করার জন্য তৈরি। Pandas, Matplotlib, Scikit-learn, TensorFlow সবাই NumPy এর উপর নির্ভর করে। তাই NumPy না শিখলে বাকি সব কঠিন হয়ে যাবে।

### NumPy ইনস্টল ও ইম্পোর্ট

```python
# ইনস্টলেশন (টার্মিনালে লিখতে হবে)
# pip install numpy

# ইম্পোর্ট
import numpy as np

print(f"NumPy ভার্সন: {np.__version__}")
```

Output:
```
NumPy ভার্সন: 1.26.4
```

সবসময় `import numpy as np` লিখতে হবে। এটা সবার ফলো করা কনভেনশন, `np` ছোট নাম ব্যবহার করা হয়।

### Python List বনাম NumPy Array

আগে বুঝে নিই, Python list আর NumPy array এর মধ্যে পার্থক্য কি:

```python
import numpy as np

# Python list
python_list = [1, 2, 3, 4, 5]
print(f"Python list: {python_list}")
print(f"টাইপ: {type(python_list)}")

# NumPy array
numpy_array = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {numpy_array}")
print(f"টাইপ: {type(numpy_array)}")
```

Output:
```
Python list: [1, 2, 3, 4, 5]
টাইপ: <class 'list'>
NumPy array: [1 2 3 4 5]
টাইপ: <class 'numpy.ndarray'>
```

দেখো, list এ কমা আছে, NumPy array তে কমা নেই। এটা ছোট পার্থক্য মনে হতে পারে, কিন্তু ভেতরের পার্থক্য অনেক বড়।

**স্পিড তুলনা:**

```python
import numpy as np
import time

# Python list দিয়ে দশ লাখ সংখ্যা যোগ
size = 1_000_000
python_list1 = list(range(size))
python_list2 = list(range(size))

start = time.time()
result_list = [a + b for a, b in zip(python_list1, python_list2)]
time_list = time.time() - start

# NumPy array দিয়ে দশ লাখ সংখ্যা যোগ
numpy_array1 = np.arange(size)
numpy_array2 = np.arange(size)

start = time.time()
result_array = numpy_array1 + numpy_array2
time_numpy = time.time() - start

print(f"Python list সময়: {time_list:.4f} সেকেন্ড")
print(f"NumPy array সময়: {time_numpy:.4f} সেকেন্ড")
print(f"NumPy কতগুণ দ্রুত: {time_list/time_numpy:.1f} গুণ")
```

Output:
```
Python list সময়: 0.1523 সেকেন্ড
NumPy array সময়: 0.0031 সেকেন্ড
NumPy কতগুণ দ্রুত: 49.1 গুণ
```

এই পার্থক্যটা বড় ডেটায় আরো বেশি হবে। NumPy এত দ্রুত কারণ এটা C ভাষায় লেখা, আর vectorization ব্যবহার করে, মানে loop ছাড়াই পুরো array তে একসাথে অপারেশন করে।

### ndarray কি?

**ndarray** (N-dimensional array) হলো NumPy এর মূল ডেটা স্ট্রাকচার। এটা একটা গ্রিড যেখানে সব ভ্যালু একই টাইপের। "nd" মানে N-dimensional, অর্থাৎ ১D, ২D, ৩D যেকোনো ডাইমেনশনের হতে পারে।

```python
import numpy as np

# ১D array (ভেক্টর)
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"১D array: {arr_1d}")
print(f"ডাইমেনশন: {arr_1d.ndim}")
print(f"শেপ: {arr_1d.shape}")

# ২D array (ম্যাট্রিক্স)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n২D array:\n{arr_2d}")
print(f"ডাইমেনশন: {arr_2d.ndim}")
print(f"শেপ: {arr_2d.shape}")

# ৩D array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n৩D array:\n{arr_3d}")
print(f"ডাইমেনশন: {arr_3d.ndim}")
print(f"শেপ: {arr_3d.shape}")
```

Output:
```
১D array: [1 2 3 4 5]
ডাইমেনশন: 1
শেপ: (5,)

২D array:
[[1 2 3]
 [4 5 6]]
ডাইমেনশন: 2
শেপ: (2, 3)

৩D array:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
ডাইমেনশন: 3
শেপ: (2, 2, 2)
```

### শেপ (Shape) বোঝা

শেপ হলো array এর প্রতিটা ডাইমেনশনে কতগুলো এলিমেন্ট আছে, সেটার টুপল। এটা অত্যন্ত গুরুত্বপূর্ণ কনসেপ্ট, কারণ মেশিন লার্নিং এ সবসময় শেপ নিয়ে কাজ করতে হয়।

- `(5,)` মানে ১D array, ৫টা এলিমেন্ট
- `(2, 3)` মানে ২D array, ২ রো, ৩ কলাম
- `(2, 2, 2)` মানে ৩D array, ২টা ২x২ ম্যাট্রিক্স

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"Array:\n{arr}")
print(f"শেপ: {arr.shape}")
print(f"রো সংখ্যা: {arr.shape[0]}")
print(f"কলাম সংখ্যা: {arr.shape[1]}")
print(f"মোট এলিমেন্ট: {arr.size}")
```

Output:
```
Array:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
শেপ: (3, 4)
রো সংখ্যা: 3
কলাম সংখ্যা: 4
মোট এলিমেন্ট: 12
```

### Array তৈরির বিভিন্ন উপায়

`np.array()` ছাড়াও অনেকভাবে array তৈরি করা যায়:

```python
import numpy as np

# zeros - সব 0 দিয়ে ভরা
zeros_arr = np.zeros((3, 4))
print(f"zeros:\n{zeros_arr}")

# ones - সব 1 দিয়ে ভরা
ones_arr = np.ones((2, 3))
print(f"\nones:\n{ones_arr}")

# full - নির্দিষ্ট মান দিয়ে ভরা
full_arr = np.full((2, 2), 7)
print(f"\nfull:\n{full_arr}")

# arange - range এর মতো
range_arr = np.arange(0, 10, 2)
print(f"\narange: {range_arr}")

# linspace - সমান ব্যবধানে
linspace_arr = np.linspace(0, 1, 5)
print(f"linspace: {linspace_arr}")

# random - র্যান্ডম সংখ্যা
random_arr = np.random.rand(3, 2)
print(f"\nrandom:\n{random_arr}")

# randint - র্যান্ডম পূর্ণসংখ্যা
randint_arr = np.random.randint(1, 100, size=(2, 3))
print(f"\nrandint:\n{randint_arr}")

# eye - আইডেন্টিটি ম্যাট্রিক্স
eye_arr = np.eye(3)
print(f"\neye:\n{eye_arr}")
```

Output:
```
zeros:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

ones:
[[1. 1. 1.]
 [1. 1. 1.]]

full:
[[7 7]
 [7 7]]

arange: [0 2 4 6 8]
linspace: [0.   0.25 0.5  0.75 1.  ]

random:
[[0.548 0.715]
 [0.603 0.545]
 [0.424 0.646]]

randint:
[[45 67 23]
 [89 12 56]]

eye:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

### ডেটা টাইপ (dtype)

NumPy array সব এলিমেন্ট একই টাইপের হয়। এই টাইপ কি, সেটা `dtype` অ্যাট্রিবিউট দিয়ে দেখা যায়:

```python
import numpy as np

int_arr = np.array([1, 2, 3])
float_arr = np.array([1.0, 2.0, 3.0])
mixed_arr = np.array([1, 2.5, 3])
str_arr = np.array(['hello', 'world'])

print(f"int array dtype: {int_arr.dtype}")
print(f"float array dtype: {float_arr.dtype}")
print(f"mixed array dtype: {mixed_arr.dtype}")
print(f"string array dtype: {str_arr.dtype}")

# নির্দিষ্ট dtype দিয়ে তৈরি
float32_arr = np.array([1, 2, 3], dtype=np.float32)
int8_arr = np.array([1, 2, 3], dtype=np.int8)
print(f"\nfloat32: {float32_arr.dtype}")
print(f"int8: {int8_arr.dtype}")
```

Output:
```
int array dtype: int64
float array dtype: float64
mixed array dtype: float64
string array dtype: <U5

float32: float32
int8: int8
```

খেয়াল করো, mixed array (int আর float মিশ্রিত) তে সব কিছু float হয়ে গেছে। NumPy স্বয়ংক্রিয়ভাবে সবচেয়ে বড় টাইপে কনভার্ট করে। এটা হলো upcasting।

### সারসংক্ষেপ

| ফিচার | Python List | NumPy Array |
|---|---|---|
| টাইপ | মিশ্র হতে পারে | একই টাইপ |
| স্পিড | ধীর | দ্রুত (৫০-১০০ গুণ) |
| মেমরি | বেশি | কম |
| অপারেশন | loop লাগে | vectorized |
| ব্রডকাস্টিং | নেই | আছে |
| মাল্টিডাইমেনশনাল | জটিল | সহজ |

নোট: https://github.com/nomancsediu/pyfords

#NumPy #PythonForDataScience #MachineLearning #ndarray #DataScience
