**Part 9**

NumPy Linear Algebra: ML এর হৃদপিণ্ড

Machine Learning এর ভেতরে যা কিছু হচ্ছে, সবকিছুর পেছনে Linear Algebra আছে। Linear Regression এ ম্যাট্রিক্স মাল্টিপ্লিকেশন, Neural Network এ weight আর input এর গুণফল, PCA তে Eigenvalue ডিকম্পোজিশন, SVD তে ম্যাট্রিক্স ফ্যাক্টরাইজেশন, সবকিছুই Linear Algebra। আর NumPy এর linalg মডিউল দিয়ে এই সব অপারেশন করা যায়।

Dot Product: সবচেয়ে বেশি ব্যবহৃত অপারেশন। দুইটা ভেক্টরের মধ্যে গুণ করার নিয়ম। শুধু ভেক্টর না, ম্যাট্রিক্সের মধ্যেও গুণ করা যায়। np.dot() বা @ অপারেটর দিয়ে করা হয়। Linear Regression এ y = Xw + b, এখানে X আর w এর মধ্যে dot product হচ্ছে। Neural Network এ প্রতিটা লেয়ারে input আর weight এর মধ্যে dot product হচ্ছে। একটা Neural Network ট্রেইন করতে গেলে লাখ লাখ বার dot product হয়।

Matrix Multiplication: দুইটা ২D ম্যাট্রিক্সের গুণফল। np.matmul() বা @ অপারেটর দিয়ে করা হয়। এটা dot product থেকে আলাদা, কারণ ম্যাট্রিক্স মাল্টিপ্লিকেশনে ডাইমেনশনের নিয়ম কঠোর। প্রথম ম্যাট্রিক্সের কলাম সংখ্যা আর দ্বিতীয় ম্যাট্রিক্সের সারি সংখ্যা সমান হতে হবে।

Determinant আর Inverse: একটা ম্যাট্রিক্সের determinant বের করা যায় np.linalg.det() দিয়ে। Inverse বের করা যায় np.linalg.inv() দিয়ে। Linear Regression এর Closed-form Solution এ ম্যাট্রিক্স Inverse লাগে। যে সমীকরণটা দিয়ে সরাসরি weight বের করা যায়, সেখানে ম্যাট্রিক্স Inverse ক্যালকুলেট করতে হয়।

Eigenvalues আর Eigenvectors: একটা ম্যাট্রিক্সের Eigenvalue আর Eigenvector বের করা যায় np.linalg.eig() দিয়ে। PCA (Principal Component Analysis) তে এই কনসেপ্ট সরাসরি ব্যবহার হয়। PCA হলো ডাইমেনশনালিটি রিডাকশনের একটা টেকনিক, যেখানে বড় ডেটাসেটকে ছোট করে ফেলা হয় মূল ইনফরমেশন হারানো ছাড়াই। আর এই কাজটা Eigenvalue ডিকম্পোজিশন দিয়ে হয়।

Singular Value Decomposition (SVD): যেকোনো ম্যাট্রিক্সকে তিনটা ম্যাট্রিক্সে ভাগ করা। np.linalg.svd() দিয়ে করা হয়। SVD শুধু PCA তেই না, Recommender System, Image Compression, Natural Language Processing এও ব্যবহার হয়। Google এর PageRank অ্যালগরিদমেও SVD এর ব্যবহার আছে।

Linear Equation Solve: যখন Ax = b টাইপ সমীকরণ সমাধান করতে হয়, তখন np.linalg.solve() ব্যবহার করা যায়। Inverse বের করে সমাধান করার চেয়ে solve() ব্যবহার করা বেশি এফিশিয়েন্ট আর নির্ভরযোগ্য।

Norm: ভেক্টর বা ম্যাট্রিক্সের দৈর্ঘ্য বা ম্যাগনিটিউড মাপা। np.linalg.norm() দিয়ে করা হয়। Regularization এ L1 আর L2 Norm ব্যবহার হয় মডেল Overfitting হওয়া থেকে বাঁচাতে। Ridge Regression এ L2 Norm, Lasso Regression এ L1 Norm কাজে লাগে।

নোট: [নোটের লিংক]

পরের পার্টে থাকবে, NumPy এর সব কনসেপ্ট একসাথে কীভাবে কাজ করে, সেটা দেখা হবে একটা Capstone প্রজেক্টে।

#MachineLearning #AllAboutSemicolons #NumPy #LinearAlgebra #DataScience #Python