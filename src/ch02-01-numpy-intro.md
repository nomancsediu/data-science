## NumPy পরিচিতি ও ndarray

### কেন NumPy শিখবো?

Python list দিয়ে সব করা যায় মনে হতে পারে। হাজারটা সংখ্যা যোগ করতে হলে for loop চালালেই হয়। কিন্তু মিলিয়ন সংখ্যার হিসাব করতে গেলে Python list অনেক ধীর হয়ে যায়। সেখানেই NumPy আসে।

NumPy (Numerical Python) হলো Python এর সবচেয়ে গুরুত্বপূর্ণ লাইব্রেরি একটা, বিশেষ করে ডেটা সায়েন্স আর মেশিন লার্নিং এর জন্য। এটা দ্রুত, মেমরি ইফিশিয়েন্ট, আর বড় ডেটা নিয়ে কাজ করার জন্য তৈরি। Pandas, Matplotlib, Scikit-learn, TensorFlow সবাই NumPy এর উপর নির্ভর করে। তাই NumPy না শিখলে বাকি সব কঠিন হয়ে যাবে।

### NumPy ইনস্টল ও ইম্পোর্ট

```python
# Installation (run in terminal)
# pip install numpy

# Import
import numpy as np

print(f"NumPy version: {np.__version__}")
```

Output:
```
NumPy version: 1.26.4
```

সবসময় `import numpy as np` লিখতে হবে। এটা সবার ফলো করা কনভেনশন, `np` ছোট নাম ব্যবহার করা হয়।

### Python List বনাম NumPy Array

আগে বুঝে নিই, Python list আর NumPy array এর মধ্যে পার্থক্য কি:

```python
import numpy as np

# Python list
python_list = [1, 2, 3, 4, 5]
print(f"Python list: {python_list}")
print(f"Type: {type(python_list)}")

# NumPy array
numpy_array = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {numpy_array}")
print(f"Type: {type(numpy_array)}")
```

Output:
```
Python list: [1, 2, 3, 4, 5]
Type: <class 'list'>
NumPy array: [1 2 3 4 5]
Type: <class 'numpy.ndarray'>
```

দেখো, list এ কমা আছে, NumPy array তে কমা নেই। এটা ছোট পার্থক্য মনে হতে পারে, কিন্তু ভেতরের পার্থক্য অনেক বড়।

**স্পিড তুলনা:**

```python
import numpy as np
import time

# Adding 1 million numbers with Python list
size = 1_000_000
python_list1 = list(range(size))
python_list2 = list(range(size))

start = time.time()
result_list = [a + b for a, b in zip(python_list1, python_list2)]
time_list = time.time() - start

# Adding 1 million numbers with NumPy array
numpy_array1 = np.arange(size)
numpy_array2 = np.arange(size)

start = time.time()
result_array = numpy_array1 + numpy_array2
time_numpy = time.time() - start

print(f"Python list time: {time_list:.4f} sec")
print(f"NumPy array time: {time_numpy:.4f} sec")
print(f"NumPy speedup: {time_list/time_numpy:.1f}x")
```

Output:
```
Python list time: 0.1523 sec
NumPy array time: 0.0031 sec
NumPy speedup: 49.1x
```

NumPy এত দ্রুত কারণ এটা C ভাষায় লেখা, আর vectorization ব্যবহার করে, মানে loop ছাড়াই পুরো array তে একসাথে অপারেশন করে।

### ndarray কি?

**ndarray** (N-dimensional array) হলো NumPy এর মূল ডেটা স্ট্রাকচার। এটা একটা গ্রিড যেখানে সব ভ্যালু একই টাইপের। "nd" মানে N-dimensional, অর্থাৎ 1D, 2D, 3D যেকোনো ডাইমেনশনের হতে পারে।

Python list এ প্রতিটা এলিমেন্ট আলাদা অবজেক্ট, তাই লুপ চালালে অনেক স্লো হয়। অন্যদিকে ndarray মেমরিতে কনটিগুয়াসলি স্টোর হয়, আর সব এলিমেন্ট একই টাইপের হয়। তাই NumPy C লেভেলে অপারেশন করতে পারে, আর এত ফাস্ট হয়।

```python
import numpy as np

# 1D array (vector)
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D array: {arr_1d}")
print(f"Dimension: {arr_1d.ndim}")
print(f"Shape: {arr_1d.shape}")

# 2D array (matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D array:\n{arr_2d}")
print(f"Dimension: {arr_2d.ndim}")
print(f"Shape: {arr_2d.shape}")

# 3D array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D array:\n{arr_3d}")
print(f"Dimension: {arr_3d.ndim}")
print(f"Shape: {arr_3d.shape}")
```

Output:
```
1D array: [1 2 3 4 5]
Dimension: 1
Shape: (5,)

2D array:
[[1 2 3]
 [4 5 6]]
Dimension: 2
Shape: (2, 3)

3D array:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
Dimension: 3
Shape: (2, 2, 2)
```

