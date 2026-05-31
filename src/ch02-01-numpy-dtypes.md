## NumPy ডেটা টাইপ ও Array Attributes

### কেন ডেটা টাইপ বুঝতে হবে?

মেশিন লার্নিং এ ডেটা টাইপ অত্যন্ত গুরুত্বপূর্ণ। একটা ছবি ০ থেকে ২৫৫ এর int হিসেবে থাকে, কিন্তু নিউরাল নেটওয়ার্ক ০ থেকে ১ এর float চায়। টাইপ ভুল হলে মডেল কাজ করবে না। ডেটা টাইপ ভুল হলে দিনের পর দিন এরর খুঁজতে হতে পারে, আর শেষে দেখা যাবে সমস্যা শুধু ডেটা টাইপ। তাই এই বিষয়টা ভালো করে বুঝতে হবে।

### NumPy এর ডেটা টাইপসমূহ

NumPy তে অনেক ধরনের ডেটা টাইপ আছে, কিন্তু বেশি ব্যবহৃত গুলো হলো:

```python
import numpy as np

# সাধারণ ডেটা টাইপ
int_arr = np.array([1, 2, 3])
print(f"int64: {int_arr.dtype}")

float_arr = np.array([1.5, 2.5, 3.5])
print(f"float64: {float_arr.dtype}")

bool_arr = np.array([True, False, True])
print(f"bool: {bool_arr.dtype}")

complex_arr = np.array([1+2j, 3+4j])
print(f"complex128: {complex_arr.dtype}")

# নির্দিষ্ট সাইজের টাইপ
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

# বিভিন্ন int টাইপের রেঞ্জ
print("Integer Types:")
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo(dtype)
    print(f"  {dtype.__name__:8s}: {info.min:>20} থেকে {info.max:<20} ({info.bits} bits)")

print("\nFloat Types:")
for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"  {dtype.__name__:8s}: min={info.min:.2e}, max={info.max:.2e} ({info.bits} bits)")
```

Output:
```
Integer Types:
  int8    :                -128 থেকে 127                 (8 bits)
  int16   :              -32768 থেকে 32767                (16 bits)
  int32   :          -2147483648 থেকে 2147483647           (32 bits)
  int64   : -9223372036854775808 থেকে 9223372036854775807  (64 bits)

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

# astype() দিয়ে টাইপ কনভার্ট
int_arr = np.array([1, 2, 3, 4, 5])
print(f"আসল: {int_arr}, dtype: {int_arr.dtype}")

float_arr = int_arr.astype(np.float64)
print(f"float64: {float_arr}, dtype: {float_arr.dtype}")

int32_arr = int_arr.astype(np.int32)
print(f"int32: {int32_arr}, dtype: {int32_arr.dtype}")

# float থেকে int (দশমিক কাটা যাবে)
float_data = np.array([1.7, 2.3, 3.9, 4.1])
int_data = float_data.astype(np.int32)
print(f"\nfloat থেকে int: {float_data} -> {int_data}")
```

Output:
```
আসল: [1 2 3 4 5], dtype: int64
float64: [1. 2. 3. 4. 5.], dtype: float64
int32: [1 2 3 4 5], dtype: int32

float থেকে int: [1.7 2.3 3.9 4.1] -> [1 2 3 4]
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
print(f"\nndim (ডাইমেনশন সংখ্যা): {arr.ndim}")
print(f"shape (প্রতি ডাইমেনশনে সাইজ): {arr.shape}")
print(f"size (মোট এলিমেন্ট): {arr.size}")
print(f"dtype (ডেটা টাইপ): {arr.dtype}")
print(f"itemsize (প্রতি এলিমেন্টে বাইট): {arr.itemsize}")
print(f"nbytes (মোট বাইট): {arr.nbytes}")
print(f"T (ট্রান্সপোজ):\n{arr.T}")
```

Output:
```
Array:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

ndim (ডাইমেনশন সংখ্যা): 2
shape (প্রতি ডাইমেনশনে সাইজ): (3, 4)
size (মোট এলিমেন্ট): 12
dtype (ডেটা টাইপ): int64
itemsize (প্রতি এলিমেন্টে বাইট): 8
nbytes (মোট বাইট): 96
T (ট্রান্সপোজ):
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

# ১D কে ২D তে রিশেপ
arr = np.arange(12)
print(f"আসল ১D: {arr}, shape: {arr.shape}")

arr_2d = arr.reshape(3, 4)
print(f"২D (3,4):\n{arr_2d}")

arr_2d_alt = arr.reshape(4, 3)
print(f"\n২D (4,3):\n{arr_2d_alt}")

arr_3d = arr.reshape(2, 2, 3)
print(f"\n৩D (2,2,3):\n{arr_3d}")

# -1 দিয়ে স্বয়ংক্রিয় হিসাব
arr_auto = arr.reshape(3, -1)  # -1 মানে স্বয়ংক্রিয়ভাবে হিসাব করো
print(f"\nস্বয়ংক্রিয় (3,-1):\n{arr_auto}")
print(f"shape: {arr_auto.shape}")
```

Output:
```
আসল ১D: [ 0  1  2  3  4  5  6  7  8  9 10 11], shape: (12,)

২D (3,4):
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

২D (4,3):
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

৩D (2,2,3):
[[[ 0  1  2]
  [ 3  4  5]]

 [[ 6  7  8]
  [ 9 10 11]]]

স্বয়ংক্রিয় (3,-1):
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
print(f"২D array:\n{arr_2d}")

# flatten - নতুন array তৈরি করে
flat = arr_2d.flatten()
print(f"\nflatten: {flat}")

# ravel - ভিউ রিটার্ন করে (মেমরি সেভ)
rav = arr_2d.ravel()
print(f"ravel: {rav}")

# flatten আর ravel এর পার্থক্য
flat[0] = 100
print(f"\nflatten পরিবর্তনের পর, আসল array:\n{arr_2d}")

rav2 = arr_2d.ravel()
rav2[0] = 100
print(f"\nravel পরিবর্তনের পর, আসল array:\n{arr_2d}")
```

Output:
```
২D array:
[[1 2 3]
 [4 5 6]]

flatten: [1 2 3 4 5 6]
ravel: [1 2 3 4 5 6]

flatten পরিবর্তনের পর, আসল array:
[[1 2 3]
 [4 5 6]]

ravel পরিবর্তনের পর, আসল array:
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

---
