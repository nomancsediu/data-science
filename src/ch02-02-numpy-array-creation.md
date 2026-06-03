## NumPy অ্যারে তৈরি ও ডেটা টাইপ

### np.array() দিয়ে অ্যারে তৈরি

সবচেয়ে বেশি ব্যবহৃত মেথড `np.array()`। Python list পাস করে দিলেই NumPy array বানিয়ে দেবে:

```python
import numpy as np

# Python list থেকে 1D array
py_list = [1, 2, 3, 4, 5]
arr_1d = np.array(py_list)
print(f"1D array: {arr_1d}")
print(f"shape: {arr_1d.shape}, ndim: {arr_1d.ndim}")

# Python list থেকে 2D array
py_list_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(py_list_2d)
print(f"\n2D array:\n{arr_2d}")
print(f"shape: {arr_2d.shape}, ndim: {arr_2d.ndim}")

# Python list থেকে 3D array
py_list_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
arr_3d = np.array(py_list_3d)
print(f"\n3D array:\n{arr_3d}")
print(f"shape: {arr_3d.shape}, ndim: {arr_3d.ndim}")

# Tuple থেকেও array বানানো যায়
arr_from_tuple = np.array((10, 20, 30))
print(f"\nTuple থেকে: {arr_from_tuple}")
```

Output:
```
1D array: [1 2 3 4 5]
shape: (5,), ndim: 1

2D array:
[[1 2 3]
 [4 5 6]]
shape: (2, 3), ndim: 2

3D array:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
shape: (2, 2, 2), ndim: 3

Tuple থেকে: [10 20 30]
```

### অ্যারে তৈরির অন্যান্য উপায়

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

**rand, randn, randint এর পার্থক্য:**

```python
import numpy as np

# rand - uniform distribution (0 to 1)
uniform = np.random.rand(5)
print(f"rand (uniform 0-1): {uniform}")

# randn - normal distribution (mean=0, std=1)
normal = np.random.randn(5)
print(f"randn (normal):     {normal}")

# randint - random integers in range
ints = np.random.randint(10, 50, size=5)
print(f"randint (10-50):    {ints}")
```

Output:
```
rand (uniform 0-1): [0.374 0.852 0.123 0.667 0.941]
randn (normal):     [ 0.312 -1.245  0.876 -0.432  1.567]
randint (10-50):    [34 12 47 23 38]
```

### ডেটা টাইপসমূহ

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
  float64 : min=-1.80e+308, max=1.80e+308 (64 bits)
```

**মেমরি হিসাব:** একটা 1000x1000 array তে int64 ব্যবহার করলে 8,000,000 বাইট লাগবে। int32 ব্যবহার করলে 4,000,000 বাইট। অর্ধেক মেমরি বাঁচবে। ডিপ লার্নিং এ float32 ব্যবহার হয় এই কারণেই, GPU তেও float32 দ্রুত কাজ করে।

### টাইপ কনভার্শন (Type Casting)

```python
import numpy as np

# astype() দিয়ে টাইপ কনভার্ট
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

খেয়াল করো, float থেকে int এ কনভার্ট করলে রাউন্ড হয় না, দশমিক অংশ কেটে যায়। 1.7 হয়ে যায় 1, 3.9 হয়ে যায় 3। এটা অনেক সময় বাগ এর কারণ হতে পারে।

### Array Attributes

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

- **ndim:** কত ডাইমেনশনের array। 1D হলে 1, 2D হলে 2, ইত্যাদি।
- **shape:** প্রতি ডাইমেনশনে কতগুলো এলিমেন্ট। (3, 4) মানে 3 রো, 4 কলাম।
- **size:** মোট এলিমেন্ট সংখ্যা। shape এর সব গুণফল।
- **dtype:** এলিমেন্টগুলোর ডেটা টাইপ।
- **itemsize:** প্রতিটা এলিমেন্ট কত বাইট জায়গা নেয়। int64 হলে 8 বাইট, float32 হলে 4 বাইট।
- **nbytes:** পুরো array কত বাইট জায়গা নেয়। size * itemsize।
- **T:** ট্রান্সপোজ, রো আর কলাম অদলবদল।

### সারসংক্ষেপ

| মেথড/অ্যাট্রিবিউট | কাজ |
|---|---|
| `np.array()` | list/tuple থেকে array |
| `np.zeros()` | 0 দিয়ে ভরা array |
| `np.ones()` | 1 দিয়ে ভরা array |
| `np.full()` | নির্দিষ্ট মান দিয়ে ভরা |
| `np.arange()` | range এর মতো |
| `np.linspace()` | সমান ব্যবধান |
| `np.random.rand()` | uniform 0-1 |
| `np.random.randn()` | normal distribution |
| `np.random.randint()` | র্যান্ডম ইন্টিজার |
| `np.eye()` | identity matrix |
| `ndim` | ডাইমেনশন সংখ্যা |
| `shape` | প্রতি ডাইমেনশনে সাইজ |
| `size` | মোট এলিমেন্ট |
| `dtype` | ডেটা টাইপ |
| `itemsize` | প্রতি এলিমেন্টে বাইট |
| `nbytes` | মোট বাইট |
| `T` | ট্রান্সপোজ |
| `astype()` | টাইপ কনভার্ট |
