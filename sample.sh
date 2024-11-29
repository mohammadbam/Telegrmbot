import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# تولید داده نمونه (می‌توانید داده واقعی خود را جایگزین کنید)
np.random.seed(42)
data = np.random.rand(200, 2)  # 200 نقطه دو بعدی تصادفی

# نمایش داده
plt.scatter(data[:, 0], data[:, 1], s=30, cmap='viridis')
plt.title('Initial Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# تعیین تعداد خوشه‌ها با استفاده از روش Elbow
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

# نمایش نمودار Elbow
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# اعمال KMeans با تعداد خوشه انتخابی
optimal_k = 3  # تعداد خوشه بهینه را مشخص کنید
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(data)

# نمایش نتایج خوشه‌بندی
plt.scatter(data[:, 0], data[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title(f'Clustering with k={optimal_k}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# محاسبه امتیاز سیلوئت
silhouette_avg = silhouette_score(data, labels)
print(f'Silhouette Score: {silhouette_avg}')


