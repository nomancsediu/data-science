## NumPy পরিচিতি ও ndarray

মেশিন লার্নিং মডেল ডেটা কিভাবে প্রসেস করে? টেক্সট হিসেবে? ইমেজ হিসেবে? না। সবকিছুকে নম্বরে রূপান্তর করে ম্যাট্রিক্স আর ভেক্টর হিসেবে প্রসেস করে। একটা ছবি আসলে হাজার হাজার নম্বরের একটা ম্যাট্রিক্স। একটা ডেটাসেট হলো সারি আর কলামের একটা বিশাল ম্যাট্রিক্স। আর এই ম্যাট্রিক্স অপারেশনগুলো দ্রুত আর এফিশিয়েন্টলি করার জন্য যে টুল লাগে, সেটাই NumPy।

### NumPy কি?

NumPy এর পুরো নাম Numerical Python। Python এর সবচেয়ে বেশি ব্যবহৃত লাইব্রেরি গুলোর একটা। Pandas, Scikit-learn, TensorFlow সবার নিচে NumPy কাজ করে। মানে তুমি যখন Pandas দিয়ে ডেটা প্রসেস করছ, আসলে ভেতরে NumPy রান হচ্ছে। যখন Scikit-learn দিয়ে মডেল ট্রেইন করছ, ভেতরে ম্যাট্রিক্স অপারেশনগুলো NumPy দিয়েই হচ্ছে।

```python
import numpy as np
print(f"NumPy version: {np.__version__}")
```

Output:
```
NumPy version: 1.26.4
```

### ndarray

NumPy এর সবচেয়ে বড় শক্তি হলো **ndarray** (N-dimensional array)। Python এর সাধারণ list এর চেয়ে ndarray কয়েকশ গুণ ফাস্ট। কারণ ndarray মেমরিতে কনটিগুয়াসলি স্টোর হয়, আর সব এলিমেন্ট একই টাইপের হয়। Python list এ প্রতিটা এলিমেন্ট আলাদা অবজেক্ট, তাই লুপ চালালে অনেক স্লো হয়। অন্যদিকে NumPy C লেভেলে অপারেশন করে, তাই এত ফাস্ট।

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D array: {arr_1d}")

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D array:\n{arr_2d}")
```

Output:
```
1D array: [1 2 3 4 5]
2D array:
[[1 2 3]
 [4 5 6]]
```

### ML এ NumPy

ধরো তুমি একটা ডেটাসেট নিয়ে কাজ করছ যেখানে ১০ লাখ সারি আর ১০০ কলাম আছে। Python list দিয়ে এই ডেটাতে কোনো ম্যাথ অপারেশন করলে মিনিট লেগে যাবে। কিন্তু NumPy দিয়ে সেকেন্ডের ভগ্নাংশে হয়ে যাবে। এই স্পিড ছাড়া ML মডেল ট্রেইন করা অসম্ভব।

### Array তৈরির উপায়

```python
import numpy as np

zeros_arr = np.zeros((2, 3))
ones_arr = np.ones((2, 3))
range_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 5)
random_arr = np.random.rand(2, 2)

print(f"zeros:\n{zeros_arr}")
print(f"ones:\n{ones_arr}")
print(f"arange: {range_arr}")
print(f"linspace: {linspace_arr}")
print(f"random:\n{random_arr}")
```

Output:
```
zeros:
[[0. 0. 0.]
 [0. 0. 0.]]
ones:
[[1. 1. 1.]
 [1. 1. 1.]]
arange: [0 2 4 6 8]
linspace: [0.   0.25 0.5  0.75 1.  ]
random:
[[0.548 0.715]
 [0.603 0.545]]
```

### ডেটা টাইপ (dtype)

NumPy array সব এলিমেন্ট একই টাইপের হয়। এই টাইপ কি, সেটা `dtype` অ্যাট্রিবিউট দিয়ে দেখা যায়:

```python
import numpy as np

int_arr = np.array([1, 2, 3])
float_arr = np.array([1.0, 2.0, 3.0])
print(f"int dtype: {int_arr.dtype}")
print(f"float dtype: {float_arr.dtype}")
```

Output:
```
int dtype: int64
float dtype: float64
```

### সারসংক্ষেপ

| ফিচার | Python List | NumPy Array |
|---|---|---|
| টাইপ | মিশ্র হতে পারে | একই টাইপ |
| স্পিড | ধীর | দ্রুত (৫০-১০০ গুণ) |
| মেমরি | বেশি | কম |
| অপারেশন | loop লাগে | vectorized |
| মাল্টিডাইমেনশনাল | জটিল | সহজ |
