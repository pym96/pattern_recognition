'''
层次聚类是一种树形的聚类方法，它通过计算数据点之间的距离来创建一个聚类的树状图或称为“树状图”(dendrogram)。这种方法的主要优点是我们不需要事先知道要形成的聚类的数量。

有两种主要的层次聚类方法：

凝聚型：开始时，每个数据点都是一个聚类。然后，在每一步中，两个最近的簇被合并成一个新的簇。这一过程一直持续到所有的数据点都在一个簇中为止。
分裂型：开始时，所有数据点都在一个簇中。然后，在每一步中，一个簇被分裂成两个子簇。这个过程持续到每个簇只包含一个数据点为止。
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 创建模拟数据
X, _ = make_blobs(n_samples=150, centers=4, random_state=42)

methods = ['ward', 'single', 'complete', 'average']
plt.figure(figsize=(20, 15))

for i, method in enumerate(methods, 1):
    linked = linkage(X, method=method)
    
    plt.subplot(2, 2, i)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title(f"Hierarchical Clustering - {method} method")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    
plt.tight_layout()
plt.show()

'''
以下是层次聚类的一些常见缺点：

计算复杂性：对于基于距离的层次聚类，其时间复杂性通常是 m^2 log2N 或更差，这使得它不适合大型数据集。

不可逆：一旦合并了两个聚类，就不能再拆分它们。这意味着，如果早期的合并选择是不恰当的，它可能会影响最终的结果，而没有办法进行修正。

对噪声和离群点敏感：特别是使用“single-linkage”方法时，层次聚类很容易受到噪声和离群点的影响。

固定的链接准则：一旦选择了链接准则（例如，Ward、complete、average等），就不能在同一层次结构中组合多种准则。

内存使用：层次聚类需要维护距离矩阵，该矩阵的大小为n^2。这对于大型数据集可能会非常消耗内存。

解释性：虽然树状图为聚类提供了一个视觉表示，但对于非专家来说，解释和确定如何选择合适的聚类数量可能并不直观。

无法进行增量聚类：如果你有新的数据点要加入，你必须重新执行整个层次聚类过程，而不能只对新数据点进行增量更新。

尽管有上述的限制，但在某些情况下，层次聚类仍然是一个有价值的工具，特别是当：

数据集的大小是可管理的。
当你希望在多个尺度上探索数据的结构。
当你对数据的层次结构特别感兴趣。
'''