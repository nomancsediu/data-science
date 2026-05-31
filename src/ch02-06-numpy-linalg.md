## NumPy লিনিয়ার অ্যালজেব্রা ও র্যান্ডম

### কেন লিনিয়ার অ্যালজেব্রা শিখবো?

লিনিয়ার অ্যালজেব্রা হলো মেশিন লার্নিং এর ভাষা। লিনিয়ার রিগ্রেশন, নিউরাল নেটওয়ার্ক, PCA, SVM সবকিছুর পেছনে লিনিয়ার অ্যালজেব্রা আছে। লিনিয়ার অ্যালজেব্রা না বুঝলে মেশিন লার্নিং এর অ্যালগরিদম কিভাবে কাজ করে, সেটা কখনো গভীরভাবে বোঝা যাবে না।

### ম্যাট্রিক্স গুণ (Dot Product)

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# ম্যাট্রিক্স গুণের তিনটা উপায়
print("A @ B:")
print(A @ B)

print("\nnp.dot(A, B):")
print(np.dot(A, B))

print("\nnp.matmul(A, B):")
print(np.matmul(A, B))
```

Output:
```
A @ B:
[[19 22]
 [43 50]]

np.dot(A, B):
[[19 22]
 [43 50]]

np.matmul(A, B):
[[19 22]
 [43 50]]
```

তিনটাই একই কাজ করে, কিন্তু পার্থক্য আছে। `@` হলো Python এর ইনফিক্স অপারেটর, সবচেয়ে পড়তে সহজ। `np.dot()` সবচেয়ে জেনারেল, স্কেলার আর মাল্টিডাইমেনশনাল দুইটাতেই কাজ করে। `np.matmul()` শুধু ম্যাট্রিক্স গুণের জন্য। সাধারণত `@` ব্যবহার করাই সবচেয়ে ভালো।

### ভেক্টর আর ম্যাট্রিক্স অপারেশন

```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# ভেক্টর ডট প্রোডাক্ট
print(f"v1 . v2 = {np.dot(v1, v2)}")

# ভেক্টর নরম (দৈর্ঘ্য)
print(f"v1 এর দৈর্ঘ্য = {np.linalg.norm(v1):.4f}")

# ক্রস প্রোডাক্ট
print(f"v1 x v2 = {np.cross(v1, v2)}")

# আউটার প্রোডাক্ট
outer = np.outer(v1, v2)
print(f"\nআউটার প্রোডাক্ট:\n{outer}")
```

Output:
```
v1 . v2 = 32
v1 এর দৈর্ঘ্য = 3.7417
v1 x v2 = [-3  6 -3]

আউটার প্রোডাক্ট:
[[ 4  5  6]
 [ 8 10 12]
 [12 15 18]]
```

### ম্যাট্রিক্স ডিকম্পোজিশন

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# ইজেনভ্যালু আর ইজেনভেক্টর
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"ইজেনভ্যালু: {eigenvalues}")
print(f"ইজেনভেক্টর:\n{eigenvectors}")

# ডিটারমিন্যান্ট
det = np.linalg.det(A)
print(f"\nডিটারমিন্যান্ট: {det:.2f}")

# ইনভার্স
inv_A = np.linalg.inv(A)
print(f"\nইনভার্স:\n{inv_A}")

# যাচাই: A @ inv(A) আইডেন্টিটি হবে
identity = A @ inv_A
print(f"\nA @ inv(A):\n{np.round(identity, 10)}")
```

Output:
```
ইজেনভ্যালু: [5. 2.]
ইজেনভেক্টর:
[[ 0.894 -0.707]
 [ 0.447  0.707]]

ডিটারমিন্যান্ট: 10.00

ইনভার্স:
[[ 0.3  -0.2 ]
 [-0.1   0.4 ]]

A @ inv(A):
[[1. 0.]
 [0. 1.]]
```

### লিনিয়ার সিস্টেম সমাধান

```python
import numpy as np

# 2x + 3y = 8
# 4x + y = 6

A = np.array([[2, 3],
              [4, 1]])
b = np.array([8, 6])

# সমাধান
x = np.linalg.solve(A, b)
print(f"সমাধান: x = {x[0]}, y = {x[1]}")

# যাচাই
print(f"যাচাই: A @ x = {A @ x}")
```

Output:
```
সমাধান: x = 1.0, y = 2.0
যাচাই: A @ x = [8. 6.]
```

### SVD (Singular Value Decomposition)

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, Vt = np.linalg.svd(A)
print(f"U shape: {U.shape}")
print(f"S (সিঙ্গুলার ভ্যালু): {np.round(S, 2)}")
print(f"Vt shape: {Vt.shape}")

# রিকনস্ট্রাক্ট
S_matrix = np.zeros_like(A, dtype=float)
np.fill_diagonal(S_matrix, S)
reconstructed = U @ S_matrix @ Vt
print(f"\nরিকনস্ট্রাক্ট:\n{np.round(reconstructed)}")
```

Output:
```
U shape: (3, 3)
S (সিঙ্গুলার ভ্যালু): [1.68e+01 1.07e+00 3.37e-16]
Vt shape: (3, 3)

রিকনস্ট্রাক্ট:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
```

SVD অনেক জায়গায় ব্যবহার হয়, যেমন PCA (Principal Component Analysis), ইমেজ কম্প্রেশন, রিকমেন্ডেশন সিস্টেম।

### NumPy Random

মেশিন লার্নিং এ র্যান্ডম নম্বর অনেক কাজে লাগে। ডেটা জেনারেট করা, ওয়েট ইনিশিয়ালাইজ করা, স্যাম্পলিং, সব জায়গায় দরকার:

```python
import numpy as np

# সিড সেট (পুনরুৎপাদনযোগ্যতা)
np.random.seed(42)

# ইউনিফর্ম র্যান্ডম (0-1)
uniform = np.random.rand(5)
print(f"uniform rand: {np.round(uniform, 3)}")

# নরমাল ডিস্ট্রিবিউশন (গড়=0, std=1)
normal = np.random.randn(5)
print(f"normal randn: {np.round(normal, 3)}")

# র্যান্ডম ইন্টিজার
randint = np.random.randint(1, 100, size=5)
print(f"randint(1,100): {randint}")

# র্যান্ডম চয়েস
items = ['আপেল', 'কমলা', 'আম', 'কাঁঠাল']
choice = np.random.choice(items, size=6)
print(f"random choice: {choice}")

# শাফেল
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(f"shuffle: {arr}")
```

Output:
```
uniform rand: [0.374 0.951 0.732 0.599 0.156]
normal randn: [ 0.497 -0.139  0.648  1.523 -0.234]
randint(1,100): [52 93 15 72 61]
random choice: ['আম' 'কমলা' 'আপেল' 'কাঁঠাল' 'আম' 'আপেল']
shuffle: [3 1 5 2 4]
```

### বিভিন্ন ডিস্ট্রিবিউশন থেকে স্যাম্পল

```python
import numpy as np

np.random.seed(42)

# বাইনোমিয়াল
binomial = np.random.binomial(n=10, p=0.5, size=10)
print(f"বাইনোমিয়াল(n=10, p=0.5): {binomial}")

# পয়সন
poisson = np.random.poisson(lam=3, size=10)
print(f"পয়সন(lam=3): {poisson}")

# এক্সপোনেনশিয়াল
exponential = np.random.exponential(scale=2, size=5)
print(f"এক্সপোনেনশিয়াল(scale=2): {np.round(exponential, 2)}")

# নরমাল (কাস্টম গড় আর std)
custom_normal = np.random.normal(loc=100, scale=15, size=5)
print(f"নরমাল(গড়=100, std=15): {np.round(custom_normal, 1)}")
```

Output:
```
বাইনোমিয়াল(n=10, p=0.5): [4 7 6 4 4 5 4 7 5 6]
পয়সন(lam=3): [4 3 3 2 3 4 1 3 5 4]
এক্সপোনেনশিয়াল(scale=2): [1.56 0.32 2.98 0.27 3.44]
নরমাল(গড়=100, std=15): [109.7  89.4 102.2  93.4  88.8]
```

### নতুন র্যান্ডম API (নির্ভরযোগ্য)

NumPy এর নতুন র্যান্ডম API ব্যবহার করা উচিত, কারণ এটা আরো নিরাপদ আর নিয়ন্ত্রণযোগ্য:

```python
import numpy as np

# Generator তৈরি
rng = np.random.default_rng(seed=42)

# বিভিন্ন ডিস্ট্রিবিউশন
print(f"integers: {rng.integers(1, 100, size=5)}")
print(f"random: {np.round(rng.random(size=5), 3)}")
print(f"normal: {np.round(rng.normal(0, 1, size=5), 3)}")
print(f"choice: {rng.choice([1, 2, 3, 4, 5], size=3)}")
```

Output:
```
integers: [13 72 41 96 60]
random: [0.774 0.438 0.859 0.697 0.094]
normal: [ 0.305 -1.039 -0.535  1.202 -0.581]
choice: [5 3 4]
```

### রিয়েল ওয়ার্ল্ড উদাহরণ: ডেটা জেনারেশন

```python
import numpy as np

np.random.seed(42)

# ১০০ জন ছাত্রের পরীক্ষার নম্বর জেনারেট
# গড় ৬০, স্ট্যান্ডার্ড ডেভিয়েশন ১৫
scores = np.random.normal(loc=60, scale=15, size=100)
scores = np.clip(scores, 0, 100)  # ০-১০০ এর মধ্যে

print(f"ছাত্র সংখ্যা: {len(scores)}")
print(f"গড় নম্বর: {np.mean(scores):.1f}")
print(f"মেডিয়ান নম্বর: {np.median(scores):.1f}")
print(f"সর্বনিম্ন: {np.min(scores):.1f}")
print(f"সর্বোচ্চ: {np.max(scores):.1f}")
print(f"স্ট্যান্ডার্ড ডেভিয়েশন: {np.std(scores):.1f}")

# গ্রেডিং
A_count = np.sum(scores >= 80)
B_count = np.sum((scores >= 60) & (scores < 80))
C_count = np.sum((scores >= 40) & (scores < 60))
F_count = np.sum(scores < 40)

print(f"\nগ্রেড A (80+): {A_count} জন")
print(f"গ্রেড B (60-79): {B_count} জন")
print(f"গ্রেড C (40-59): {C_count} জন")
print(f"গ্রেড F (<40): {F_count} জন")
```

Output:
```
ছাত্র সংখ্যা: 100
গড় নম্বর: 60.5
মেডিয়ান নম্বর: 60.6
সর্বনিম্ন: 17.5
সর্বোচ্চ: 98.2
স্ট্যান্ডার্ড ডেভিয়েশন: 14.8

গ্রেড A (80+): 10 জন
গ্রেড B (60-79): 42 জন
গ্রেড C (40-59): 38 জন
গ্রেড F (<40): 10 জন
```

### সারসংক্ষেপ

| ফাংশন | কাজ |
|---|---|
| `np.dot()` / `@` | ম্যাট্রিক্স গুণ |
| `np.linalg.norm()` | ভেক্টর দৈর্ঘ্য |
| `np.linalg.det()` | ডিটারমিন্যান্ট |
| `np.linalg.inv()` | ইনভার্স |
| `np.linalg.eig()` | ইজেনভ্যালু/ইজেনভেক্টর |
| `np.linalg.solve()` | লিনিয়ার সিস্টেম |
| `np.linalg.svd()` | SVD |
| `np.random.rand()` | ইউনিফর্ম র্যান্ডম |
| `np.random.randn()` | নরমাল র্যান্ডম |
| `np.random.randint()` | র্যান্ডম পূর্ণসংখ্যা |
| `np.random.seed()` | সিড সেট |

লিনিয়ার অ্যালজেব্রা প্রথমে ভয়ংকর মনে হতে পারে, কিন্তু মেশিন লার্নিং এ এটা অপরিহার্য। নিউরাল নেটওয়ার্ক শিখলে দেখা যাবে সবকিছু ম্যাট্রিক্স গুণ। আর random মডিউলটা হলো টেস্টিং এর সবচেয়ে বড় সাহায্য। রিয়েল ডেটা না থাকলে, random দিয়ে ডেটা জেনারেট করে প্র্যাকটিস করতে হবে। শুধু মনে রাখতে হবে, reproducibility এর জন্য seed সেট করতে ভুললে চলবে না।

---
