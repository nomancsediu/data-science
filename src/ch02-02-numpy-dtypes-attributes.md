## NumPy ডেটা টাইপ ও Array Attributes

### কেন ডেটা টাইপ বুঝতে হবে?

মেশিন লার্নিং এ ডেটা টাইপ অত্যন্ত গুরুত্বপূর্ণ। একটা ছবি ০ থেকে ২৫৫ এর int হিসেবে থাকে, কিন্তু নিউরাল নেটওয়ার্ক ০ থেকে ১ এর float চায়। টাইপ ভুল হলে মডেল কাজ করবে না। ডেটা টাইপ ভুল হলে দিনের পর দিন এরর খুঁজতে হতে পারে, আর শেষে দেখা যাবে সমস্যা শুধু ডেটা টাইপ। তাই এই বিষয়টা ভালো করে বুঝতে হবে।

### Array তৈরির বিভিন্ন উপায়

`np.array()` ছাড়াও অনেকভাবে array তৈরি করা যায়:

```python
import numpy as np

# zeros - filled with 0
zeros_arr = np.zeros((3, 4))
print(f"zeros:\n{zeros_arr}")

# ones - filled with 1
ones_arr = np.ones((2, 3))
print(f"\nones:\n{ones_arr}")

# full - filled with specific value
full_arr = np.full((2, 2), 7)
print(f"\nfull:\n{full_arr}")

# arange - like range()
range_arr = np.arange(0, 10, 2)
print(f"\narange: {range_arr}")

# linspace - equal spacing
linspace_arr = np.linspace(0, 1, 5)
print(f"linspace: {linspace_arr}")

# random - random numbers
random_arr = np.random.rand(3, 2)
print(f"\nrandom:\n{random_arr}")

# randint - random integers
randint_arr = np.random.randint(1, 100, size=(2, 3))
print(f"\nrandint:\n{randint_arr}")

# eye - identity matrix
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

### NumPy এর ডেটা টাইপসমূহ

NumPy তে অনেক ধরনের ডেটা টাইপ আছে, কিন্তু বেশি ব্যবহৃত গুলো হলো:

```python
import numpy as np

# Common data types
int_arr = np.array([1, 2, 3])
print(f"int64: {int_arr.dtype}")

float_arr = np.array([1.5, 2.5, 3.5])
print(f"float64: {float_arr.dtype}")

bool_arr = np.array([True, False, True])
print(f"bool: {bool_arr.dtype}")

complex_arr = np.array([1+2j, 3+4j])
print(f"complex128: {complex_arr.dtype}")

# Specific size types
int32_arr = np.array([1, 2, 3], dtype=np.int32)
print(f"int32: {int32_arr.dtype}")

float32_arr = np.array([1.0, 2.0], dtype=np.float32)
print(f"float32: {float32_arr.dtype}")

int8_arr = np.array([1, 2, 3], dtype=np.int8)
print(f"int8: {int8_arr.dtype}")
```

Output:
```
int64: int64
float64: float64
bool: bool
complex128: complex128
int32: int32
float32: float32
int8: int8
```

### ডেটা টাইপের সাইজ ও রেঞ্জ

```python
import numpy as np

# Range of different int types
print("Integer Types:")
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo(dtype)
    print(f"  {dtype.__name__:8s}: {info.min:>20} to {info.max:<20} ({info.bits} bits)")

print("\nFloat Types:")
for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"  {dtype.__name__:8s}: min={info.min:.2e}, max={info.max:.2e} ({info.bits} bits)")
```

Output:
```
Integer Types:
  int8    :                -128 to 127                 (8 bits)
  int16   :              -32768 to 32767                (16 bits)
  int32   :          -2147483648 to 2147483647           (32 bits)
  int64   : -9223372036854775808 to 9223372036854775807  (64 bits)

Float Types:
  float16 : min=-6.55e+04, max=6.55e+04 (16 bits)
  float32 : min=-3.40e+38, max=3.40e+38 (32 bits)
  float64 : min=-1.80e+308, max=1.80e+308 (32 bits)
```

কেন এটা গুরুত্বপূর্ণ? কারণ ডিপ লার্নিং এ float32 ব্যবহার হয়, float64 না। কারণ float64 দ্বিগুণ মেমরি লাগে, আর GPU তে float32 দ্রুত কাজ করে। তাই ডেটা টাইপ সম্পর্কে জানা দরকার।

### টাইপ কনভার্শন (Type Casting)

এক টাইপ থেকে আরেক টাইপে কনভার্ট করা প্রায়ই দরকার হয়:

```python
import numpy as np

# Type conversion with astype()
int_arr = np.array([1, 2, 3, 4, 5])
print(f"Original: {int_arr}, dtype: {int_arr.dtype}")

float_arr = int_arr.astype(np.float64)
print(f"float64: {float_arr}, dtype: {float_arr.dtype}")

int32_arr = int_arr.astype(np.int32)
print(f"int32: {int32_arr}, dtype: {int32_arr.dtype}")

# float to int (decimal truncated)
float_data = np.array([1.7, 2.3, 3.9, 4.1])
int_data = float_data.astype(np.int32)
print(f"\nfloat to int: {float_data} -> {int_data}")
```

Output:
```
Original: [1 2 3 4 5], dtype: int64
float64: [1. 2. 3. 4. 5.], dtype: float64
int32: [1 2 3 4 5], dtype: int32

float to int: [1.7 2.3 3.9 4.1] -> [1 2 3 4]
```

খেয়াল করো, float থেকে int এ কনভার্ট করলে রাউন্ড হয় না, দশমিক অংশ কেটে যায়। ১.৭ হয়ে যায় ১, ৩.৯ হয়ে যায় ৩। এটা অনেক সময় বাগ এর কারণ হতে পারে।

### Array Attributes

NumPy array তে অনেক গুরুত্বপূর্ণ অ্যাট্রিবিউট আছে:

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"Array:\n{arr}")
print(f"\nndim (number of dimensions): {arr.ndim}")
print(f"shape (size per dimension): {arr.shape}")
print(f"size (total elements): {arr.size}")
print(f"dtype (data type): {arr.dtype}")
print(f"itemsize (bytes per element): {arr.itemsize}")
print(f"nbytes (total bytes): {arr.nbytes}")
print(f"T (transpose):\n{arr.T}")
```

Output:
```
Array:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

ndim (number of dimensions): 2
shape (size per dimension): (3, 4)
size (total elements): 12
dtype (data type): int64
itemsize (bytes per element): 8
nbytes (total bytes): 96
T (transpose):
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
```

**গুরুত্বপূর্ণ অ্যাট্রিবিউট ব্যাখ্যা:**

- **ndim:** কত ডাইমেনশনের array। ১D হলে ১, ২D হলে ২, ইত্যাদি।
- **shape:** প্রতি ডাইমেনশনে কতগুলো এলিমেন্ট। (৩, ৪) মানে ৩ রো, ৪ কলাম।
- **size:** মোট এলিমেন্ট সংখ্যা। shape এর সব গুণফল।
- **dtype:** এলিমেন্টগুলোর ডেটা টাইপ।
- **itemsize:** প্রতিটা এলিমেন্ট কত বাইট জায়গা নেয়। int64 হলে ৮ বাইট, float32 হলে ৪ বাইট।
- **nbytes:** পুরো array কত বাইট জায়গা নেয়। size * itemsize।
- **T:** ট্রান্সপোজ, রো আর কলাম অদলবদল।

### শেপ রিশেপ (Reshape)

অনেক সময় array এর শেপ পরিবর্তন করতে হয়। মেশিন লার্নিং এ এটা প্রায়ই দরকার হয়:

```python
import numpy as np

# Reshape 1D to 2D
arr = np.arange(12)
print(f"Original 1D: {arr}, shape: {arr.shape}")

arr_2d = arr.reshape(3, 4)
print(f"2D (3,4):\n{arr_2d}")

arr_2d_alt = arr.reshape(4, 3)
print(f"\n2D (4,3):\n{arr_2d_alt}")

arr_3d = arr.reshape(2, 2, 3)
print(f"\n3D (2,2,3):\n{arr_3d}")

# -1 for automatic calculation
arr_auto = arr.reshape(3, -1)  # -1 means calculate automatically
print(f"\nAuto (3,-1):\n{arr_auto}")
print(f"shape: {arr_auto.shape}")
```

Output:
```
Original 1D: [ 0  1  2  3  4  5  6  7  8  9 10 11], shape: (12,)

2D (3,4):
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

2D (4,3):
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

3D (2,2,3):
[[[ 0  1  2]
  [ 3  4  5]]

 [[ 6  7  8]
  [ 9 10 11]]]

Auto (3,-1):
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
shape: (3, 4)
```

reshape এ একটা নিয়ম আছে: মোট এলিমেন্ট সংখ্যা বদলানো যাবে না। ১২টা এলিমেন্ট আছে, তাহলে (৩,৪), (৪,৩), (২,৬), (৬,২), (২,২,৩) সব সম্ভব, কিন্তু (৩,৫) সম্ভব না, কারণ ৩*৫=১৫, ১২ না।

### Flatten ও Ravel

মাল্টিডাইমেনশনাল array কে ১D তে নামিয়ে আনা:

```python
import numpy as np

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D array:\n{arr_2d}")

# flatten - creates a new array
flat = arr_2d.flatten()
print(f"\nflatten: {flat}")

# ravel - returns a view (memory efficient)
rav = arr_2d.ravel()
print(f"ravel: {rav}")

# Difference between flatten and ravel
flat[0] = 100
print(f"\nAfter modifying flatten, original array:\n{arr_2d}")

rav2 = arr_2d.ravel()
rav2[0] = 100
print(f"\nAfter modifying ravel, original array:\n{arr_2d}")
```

Output:
```
2D array:
[[1 2 3]
 [4 5 6]]

flatten: [1 2 3 4 5 6]
ravel: [1 2 3 4 5 6]

After modifying flatten, original array:
[[1 2 3]
 [4 5 6]]

After modifying ravel, original array:
[[100   2   3]
 [  4   5   6]]
```

`flatten()` নতুন মেমরি তে কপি করে, `ravel()` আসল array এর ভিউ দেয়। তাই ravel এ পরিবর্তন করলে আসল array ও বদলে যায়। মেমরি সেভ করতে চাইলে ravel, সেফ কাজ করতে চাইলে flatten ব্যবহার করতে হবে।

### সারসংক্ষেপ

| অ্যাট্রিবিউট | কাজ |
|---|---|
| `ndim` | ডাইমেনশন সংখ্যা |
| `shape` | প্রতি ডাইমেনশনে সাইজ |
| `size` | মোট এলিমেন্ট |
| `dtype` | ডেটা টাইপ |
| `itemsize` | প্রতি এলিমেন্টে বাইট |
| `nbytes` | মোট বাইট |
| `T` | ট্রান্সপোজ |
| `reshape()` | শেপ পরিবর্তন |
| `flatten()` | ১D কপি |
| `ravel()` | ১D ভিউ |
| `astype()` | টাইপ কনভার্ট |

ডেটা টাইপ আর শেপ হলো মেশিন লার্নিং এর দুইটা সবচেয়ে গুরুত্বপূর্ণ বেসিক কনসেপ্ট। প্রায় প্রতিদিন shape mismatch এরর দেখা যায়। "ValueError: shapes (100,3) and (4,) not aligned" এই ধরনের এরর মানে হলো, array এর শেপ মিলছে না। তাই shape, reshape, dtype এগুলো ভালো করে প্র্যাকটিস করতে হবে।
