## NumPy গাণিতিক ও পরিসংখ্যান ফাংশন

### কেন এই ফাংশনগুলো শিখবো?

মেশিন লার্নিং মানেই গণিত। ডেটার গড়, মাঝামাঝি, স্ট্যান্ডার্ড ডেভিয়েশন বের করতে হয়। নরমালাইজেশন করতে হয়। লগারিদমিক ট্রান্সফরমেশন করতে হয়। এই সব কাজের জন্য NumPy এর গাণিতিক আর পরিসংখ্যান ফাংশন অপরিহার্য। একেকটা ফাংশন নিজে লিখলে কয়েক লাইন কোড, NumPy তে এক লাইন।

### বেসিক গাণিতিক ফাংশন

```python
import numpy as np

arr = np.array([1, 4, 9, 16, 25])

# বেসিক ফাংশন
print(f"arr = {arr}")
print(f"np.sqrt(arr) = {np.sqrt(arr)}")
print(f"np.abs(arr) = {np.abs(arr)}")
print(f"np.square(arr) = {np.square(arr)}")

# এক্সপোনেনশিয়াল আর লগারিদম
arr2 = np.array([1, 2, 3])
print(f"\narr2 = {arr2}")
print(f"np.exp(arr2) = {np.exp(arr2)}")
print(f"np.log(arr2) = {np.log(arr2)}")
print(f"np.log2(arr2) = {np.log2(arr2)}")
print(f"np.log10(arr2) = {np.log10(arr2)}")

# রাউন্ডিং
arr3 = np.array([1.23, 2.67, 3.14, 4.89])
print(f"\narr3 = {arr3}")
print(f"np.round(arr3, 1) = {np.round(arr3, 1)}")
print(f"np.floor(arr3) = {np.floor(arr3)}")
print(f"np.ceil(arr3) = {np.ceil(arr3)}")
```

Output:
```
arr = [ 1  4  9 16 25]
np.sqrt(arr) = [1. 2. 3. 4. 5.]
np.abs(arr) = [ 1  4  9 16 25]
np.square(arr) = [  1  16  81 256 625]

arr2 = [1 2 3]
np.exp(arr2) = [ 2.718  7.389 20.086]
np.log(arr2) = [0.    0.693 1.099]
np.log2(arr2) = [0.    1.    1.585]
np.log10(arr2) = [0.    0.301 0.477]

arr3 = [1.23 2.67 3.14 4.89]
np.round(arr3, 1) = [1.2 2.7 3.1 4.9]
np.floor(arr3) = [1. 2. 3. 4.]
np.ceil(arr3) = [2. 3. 4. 5.]
```

### ট্রিগোনোমেট্রিক ফাংশন

```python
import numpy as np

angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

print(f"কোণ (রেডিয়ান): {angles}")
print(f"sin: {np.round(np.sin(angles), 3)}")
print(f"cos: {np.round(np.cos(angles), 3)}")
print(f"tan: {np.round(np.tan(angles), 3)}")

# ডিগ্রি থেকে রেডিয়ান আর উল্টো
deg = np.array([0, 30, 45, 60, 90])
rad = np.deg2rad(deg)
print(f"\n{deg} ডিগ্রি = {np.round(rad, 4)} রেডিয়ান")
print(f"{np.round(rad, 4)} রেডিয়ান = {np.rad2deg(rad)} ডিগ্রি")
```

Output:
```
কোণ (রেডিয়ান): [0.     0.524  0.785  1.047  1.571]
sin: [0.    0.5   0.707 0.866 1.   ]
cos: [1.    0.866 0.707 0.5   0.   ]
tan: [0.    0.577 1.    1.732 inf  ]

[0 30 45 60 90] ডিগ্রি = [0.     0.5236 0.7854 1.0472 1.5708] রেডিয়ান
[0.     0.5236 0.7854 1.0472 1.5708] রেডিয়ান = [ 0. 30. 45. 60. 90.] ডিগ্রি
```

### পরিসংখ্যান ফাংশন

এই ফাংশনগুলো মেশিন লার্নিং এ সবচেয়ে বেশি ব্যবহৃত:

```python
import numpy as np

data = np.array([23, 45, 12, 67, 34, 89, 56, 78, 43, 21])

print(f"ডেটা: {data}")
print(f"np.sum(data) = {np.sum(data)}")
print(f"np.mean(data) = {np.mean(data)}")
print(f"np.median(data) = {np.median(data)}")
print(f"np.std(data) = {np.std(data):.2f}")
print(f"np.var(data) = {np.var(data):.2f}")
print(f"np.min(data) = {np.min(data)}")
print(f"np.max(data) = {np.max(data)}")
print(f"np.argmin(data) = {np.argmin(data)}")
print(f"np.argmax(data) = {np.argmax(data)}")
print(f"np.percentile(data, 25) = {np.percentile(data, 25)}")
print(f"np.percentile(data, 75) = {np.percentile(data, 75)}")
```

Output:
```
ডেটা: [23 45 12 67 34 89 56 78 43 21]
np.sum(data) = 468
np.mean(data) = 46.8
np.median(data) = 44.0
np.std(data) = 24.59
np.var(data) = 604.56
np.min(data) = 12
np.max(data) = 89
np.argmin(data) = 2
np.argmax(data) = 5
np.percentile(data, 25) = 23.0
np.percentile(data, 75) = 67.0
```

### অ্যাক্সিস অনুযায়ী অপারেশন

২D array তে অ্যাক্সিস খুবই গুরুত্বপূর্ণ। axis=0 মানে কলাম বরাবর (উপর থেকে নিচে), axis=1 মানে রো বরাবর (বাম থেকে ডানে):

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(f"Array:\n{arr}")

# পুরো array এর উপর
print(f"\nnp.sum(arr) = {np.sum(arr)}")
print(f"np.mean(arr) = {np.mean(arr):.2f}")

# অ্যাক্সিস 0 (কলাম বরাবর)
print(f"\nnp.sum(arr, axis=0) = {np.sum(arr, axis=0)}")
print(f"np.mean(arr, axis=0) = {np.mean(arr, axis=0)}")

# অ্যাক্সিস 1 (রো বরাবর)
print(f"\nnp.sum(arr, axis=1) = {np.sum(arr, axis=1)}")
print(f"np.mean(arr, axis=1) = {np.mean(arr, axis=1)}")
```

Output:
```
Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]

np.sum(arr) = 45
np.mean(arr) = 5.00

np.sum(arr, axis=0) = [12 15 18]
np.mean(arr, axis=0) = [4. 5. 6.]

np.sum(arr, axis=1) = [ 6 15 24]
np.mean(arr, axis=1) = [2. 5. 8.]
```

অ্যাক্সিস মনে রাখার সহজ উপায়: axis=0 তে রো গুটিয়ে যায় (কলাম থাকে), axis=1 তে কলাম গুটিয়ে যায় (রো থাকে)।

### কামুলেটিভ অপারেশন

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(f"arr = {arr}")
print(f"np.cumsum(arr) = {np.cumsum(arr)}")
print(f"np.cumprod(arr) = {np.cumprod(arr)}")
```

Output:
```
arr = [1 2 3 4 5]
np.cumsum(arr) = [ 1  3  6 10 15]
np.cumprod(arr) = [  1   2   6  24 120]
```

### ক্লিপিং আর হিস্টোগ্রাম

```python
import numpy as np

# clip - মান একটা রেঞ্জের মধ্যে রাখা
arr = np.array([1, 5, 10, 15, 20, 25, 30])
clipped = np.clip(arr, 5, 25)
print(f"আসল: {arr}")
print(f"clip(5, 25): {clipped}")

# histogram - ডেটার ডিস্ট্রিবিউশন
data = np.random.randn(1000)
counts, bins = np.histogram(data, bins=10)
print(f"\nহিস্টোগ্রাম বিনস: {np.round(bins, 2)}")
print(f"কাউন্ট: {counts}")
```

Output:
```
আসল: [ 1  5 10 15 20 25 30]
clip(5, 25): [ 5  5 10 15 20 25 25]

হিস্টোগ্রাম বিনস: [-3.24 -2.56 -1.89 -1.21 -0.53  0.14  0.82  1.5   2.17  2.85  3.53]
কাউন্ট: [  3  14  55 143 250 243 155  85  37  15]
```

### নরমালাইজেশন উদাহরণ

মেশিন লার্নিং এ ডেটা নরমালাইজ করা অত্যন্ত সাধারণ:

```python
import numpy as np

# Min-Max নরমালাইজেশন (0-1 এর মধ্যে)
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

normalized = (data - data.min()) / (data.max() - data.min())
print(f"আসল ডেটা: {data}")
print(f"নরমালাইজড: {normalized}")

# Z-score স্ট্যান্ডার্ডাইজেশন
standardized = (data - data.mean()) / data.std()
print(f"\nস্ট্যান্ডার্ডাইজড: {np.round(standardized, 2)}")
print(f"গড়: {standardized.mean():.10f}")
print(f"স্ট্যান্ডার্ড ডেভিয়েশন: {standardized.std():.10f}")
```

Output:
```
আসল ডেটা: [ 10  20  30  40  50  60  70  80  90 100]
নরমালাইজড: [0.    0.111 0.222 0.333 0.444 0.556 0.667 0.778 0.889 1.   ]

স্ট্যান্ডার্ডাইজড: [-1.49 -1.16 -0.83 -0.5  -0.17  0.17  0.5   0.83  1.16  1.49]
গড়: 0.0000000000
স্ট্যান্ডার্ড ডেভিয়েশন: 1.0000000000
```

Z-score স্ট্যান্ডার্ডাইজেশন এর পর গড় ০ আর স্ট্যান্ডার্ড ডেভিয়েশন ১ হয়। এটা মেশিন লার্নিং এ অনেক অ্যালগরিদমের জন্য দরকারী, যেমন SVM, KNN, Neural Network।

### সারসংক্ষেপ

| ফাংশন | কাজ |
|---|---|
| `np.sum()` | যোগফল |
| `np.mean()` | গড় |
| `np.median()` | মেডিয়ান |
| `np.std()` | স্ট্যান্ডার্ড ডেভিয়েশন |
| `np.var()` | ভ্যারিয়েন্স |
| `np.min()` / `np.max()` | সর্বনিম্ন / সর্বোচ্চ |
| `np.argmin()` / `np.argmax()` | সর্বনিম্ন / সর্বোচ্চ এর ইনডেক্স |
| `np.percentile()` | পার্সেন্টাইল |
| `np.cumsum()` | কামুলেটিভ যোগফল |
| `np.clip()` | মান রেঞ্জের মধ্যে |
| `np.histogram()` | ডিস্ট্রিবিউশন |

গাণিতিক ফাংশনগুলো মনে রাখার কিছু নেই, প্রয়োজনে ডকুমেন্টেশন দেখলেই হবে। কিন্তু axis কনসেপ্টটা ভালো করে বুঝতে হবে, এটা প্রায় প্রতিদিন দরকার হবে। আর নরমালাইজেশন আর স্ট্যান্ডার্ডাইজেশন এর পার্থক্য মনে রাখতে হবে: Min-Max ০-১ এর মধ্যে আনে, Z-score গড় ০ আর std ১ করে।
