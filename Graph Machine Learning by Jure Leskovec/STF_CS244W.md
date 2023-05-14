# 1 引入：图机器学习



## 1.1 先修知识点

1. 机器学习
2. 算法和图论
3. 概率论与数理统计



## 1.2 为什么是图

**图机器学习中的常用工具**：NetworkX, PyTorch Geometric, DeepSNAP, GraphGym, SNAP.PY



**选择图的原因**：图是用于描述并分析有关联/互动的实体的一种普适语言。它不将实体视为一系列孤立的点，而认为其互相之间有关系。它是一种很好的描述领域知识的方式。



**网络与图的分类**：

1. **networks / natural graphs**：自然表示为图
   - **社交网络**：社会是七十亿个体的网络。
   - **交流与交易**：电子设备、电话，金融交易。
   - **生物制药**：基因或蛋白质之间互动从而调节生理活动的过程。
   - **神经连接**：我们的想法隐藏于神经元的连接之中。
2. **graphs**：作为一种表示方法
   - 信息/知识被组织或连接。
   - 软件可以被图的方式表达出来。
   - 相似网络，数据点之间的连接相似。
   - 关系结构，分子、场景图、3D形状、基于粒子的物理模拟。



复杂领域会有丰富的关系结构，可以被表示为**关系图**relational graph，通过显式地建模关系，可以获得更好的表现。

但是现代深度学习工具常用于建模简单的序列sequence（如文本、语音等具有线性结构的数据）和grid（图片具有平移不变性，可以被表示为fixed size grids或fixed size standards）。

这些传统工具很难用于图的建模，其难点在于网络的复杂：

- 任意的大小和复杂的拓扑结构。
- 没有基准点，没有节点固定的顺序。没有那种上下左右的方向。
- 经常出现动态的图，而且会有多模态的特征。



**本课程中讲述如何将神经网络模型适用范围拓展到图上**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217215728237.png" alt="image-20221217215728237" style="zoom:80%;" />



**有监督机器学习全流程图**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217215811352.png" alt="image-20221217215811352" style="zoom:80%;" />

在传统机器学习流程中，我们需要对原始数据进行**特征工程**（feature engineering）（比如手动提取特征等），但是现在我们使用**表示学习**（representation learning）的方式来自动学习到数据的特征，直接应用于下流预测任务。



**图的表示学习**：大致来说就是将原始的节点（或链接、或图）表示为向量（嵌入embedding），图中相似的节点会被embed得靠近（指同一实体，在节点空间上相似，在向量空间上就也应当相似）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217220011032.png" alt="image-20221217220011032" style="zoom:80%;" />



## 1.3 图机器学习的应用

**图机器学习任务分成四类**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217220101725.png" alt="image-20221217220101725" style="zoom:80%;" />

- **节点级别（node level）**：预测结点属性，对结点进行聚类。例如，对联机用户/项目进行分类。
- **边级别（edge level）**：预测两个结点之间是否存在连接。例如，知识图谱构建。
- **社区 / 子图级别（community/subgraph level）**：对不同图进行分类。例如，分子属性预测。
- **图级别，包括预测任务（graph-level prediction）和图生成任务（graph generation）**：例如，药物发现、物理模拟。



## 1.4 表示图的选择

**图的组成成分**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217220730964.png" alt="image-20221217220730964" style="zoom:80%;" />

- **节点**（$N$/$V$）
- **连接/边**（$E$）
- **网络/图**（G）



图是一种解决关系问题时的通用语言，各种情况下的统一数学表示。将问题抽象成图，可以用同一种机器学习算法解决所有问题。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217220921668.png" alt="image-20221217220921668" style="zoom:80%;" />



建图时需要考虑以什么作为节点，以什么作为边。对某一领域或问题选择合适的网络表示方法会决定我们能不能成功使用网络：

- 有些情况下只有唯一的明确做法
- 有些情况下可以选择很多种做法
- 设置连接的方式将决定研究问题的本质



**有向图与无向图**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217221141993.png" alt="image-20221217221141993" style="zoom:80%;" />



**度（degree）**：与结点相连边的个数。

- **无向图**：$Avg.degree=\bar{k}=<K>=\frac{1}{N}\sum_{i=1}^Nk_i=\frac{2E}{N}$ 
- **有向图**：分为入度（in-degree）与出度（out-degree），度是两者之和。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217221529457.png" alt="image-20221217221529457" style="zoom:80%;" />



**二部图（Bipartite Graph）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217221656686.png" alt="image-20221217221656686" style="zoom:80%;" />



**折叠/投影二部分图（Folded/Projected Bipartite Graphs）**：就是将一个bipartite图的两个节点子集分别投影，projection图上两个节点之间有连接，这两个节点在folded/projected bipartite graphs上至少有一个共同邻居。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217221904224.png" alt="image-20221217221904224" style="zoom:80%;" />



**表示图（Representing Graphs）**：

- **邻接矩阵（Adjacency Matrix）**：每一行/列代表一个节点，如果节点之间有边就是1，没有就是0。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222102072.png" alt="image-20221217222102072" style="zoom:80%;" />

  无向图的邻接矩阵天然对称。

  网络的邻接矩阵往往是稀疏矩阵，矩阵的密度为 $\frac{E}{N^2}$ 。

- **边表（Edge List）**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222222008.png" alt="image-20221217222222008" style="zoom:80%;" />

  这种方式常用于深度学习框架中，因为可以将图直接表示成一个二维矩阵。这种表示方法的问题在于很难进行图的操作和分析，就算只是计算图中点的度数都会很难。

- **邻接表（Adjacency List）**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222323622.png" alt="image-20221217222323622" style="zoom:80%;" />

  对图的分析和操作更方便。



**节点和边的属性，可选项**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222436395.png" alt="image-20221217222436395" style="zoom:80%;" />



**有权/无权**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222505229.png" alt="image-20221217222505229" style="zoom:80%;" />



**自环/多重图**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222625951.png" alt="image-20221217222625951" style="zoom:80%;" />

这个multigraph有时也可被视作是weighted graph，就是说将多边的地方视作一条边的权重（在邻接矩阵上可看出效果是一样的）。但有时也可能就是想要分别处理每一条边，这些边上可能有不同的property和attribute。



**连通性（Connectivity）**：

- **无向图的连通性**：

  1. 连通：任意两个节点都有路径相通。
  2. 不连通：由2至多个连通分量（connected components）构成。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217222910418.png" alt="image-20221217222910418" style="zoom:80%;" />

- **有向图的连通性**：

  1. **强连通有向图**：具有从每个节点到每个其他节点的路径，反之亦然（例如，A-B路径和B-A路径）

  2. **弱连通有向图**：如果忽视边的方向则是连通的。

  3. **强连通分量**：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221217223209833.png" alt="image-20221217223209833" style="zoom:80%;" />



# 2 图上的传统机器学习方法



## 2.1 章节前言

传统图机器学习流程可分为以下四步：

1. 第一步是根据不同的下游任务为节点/链接/图的人工设计特征（hand-designed features）
2. 第二步是将我们设计特征构建为训练数据集

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218134244446.png" alt="image-20221218134244446" style="zoom:80%;" />

3. 第三步是使用训练数据集训练一个机器学习模型，常见的有随机森林，SVM和神经网络等。
4. 第四步是使用训练好的的模型完成新样本的预测任务。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218134328792.png" alt="image-20221218134328792" style="zoom:80%;" />



从上面的步骤中我们可以发现，在传统图机器学习中**模型性能好坏很大程度上取决于人工设计的图数据特征（hand-designed features）**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218134501395.png" alt="image-20221218134501395" style="zoom:80%;" />



## 2.2 传统基于特征的方法：节点

**半监督学习任务**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218133721593.png" alt="image-20221218133721593" style="zoom:80%;" />

任务是预测灰点属于红点还是绿点。区分特征是度数（红点度数是1，绿点度数是2）。



**特征抽取目标**：找到能够描述节点在网络中结构与位置的特征。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218133827653.png" alt="image-20221218133827653" style="zoom:67%;" />



**节点度数**（node degree）：缺点在于将节点的所有邻居视为同等重要的。



**节点中心性（node centrality）** $c_v$：考虑了节点的重要性。

- **特征向量中心性（eigenvector centrality）**：认为如果节点邻居重要，那么节点本身也重要。因此节点 $v$ 的centrality是邻居centrality的加总：$c_v=\frac1\lambda\sum_{u\in N(v)}c_u$ ，其中 $\lambda$ 是一个正的常值。

  这是个递归式，解法是将其转换为矩阵形式 $\lambda\bold{c}=\bold{A}\bold{c}$ ，其中 $\bold{A}$ 是邻接矩阵，$\bold{c}$ 是centrality向量。

  从而发现centrality就是特征向量。根据Perron-Frobenius Theorem可知最大的特征值 $\lambda_{max}$ 总为正且唯一，对应的leading eigenvector $c_{max}$ 就是centrality向量。

- **中介中心性（betweenness cent）**：认为如果一个节点处在很多节点对的最短路径上，那么这个节点是重要的。（衡量一个节点作为bridge或transit hub的能力。就对我而言直觉上感觉就像是新加坡的马六甲海峡啊，巴拿马运河啊，埃及的苏伊士运河啊，什么君士坦丁堡，上海，香港……之类的商业要冲的感觉）
  $$
  c_v=\sum_{s\ne v\ne t}\frac{\#(s和t之间包含v的最短路径)}{\#(s和t之间的最短路径)}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218215301790.png" alt="image-20221218215301790" style="zoom:80%;" />

- **紧密中心性（closeness centrality）**：认为如果一个节点距其他节点之间距离最短，那么认为这个节点是重要的。
  $$
  c_v=\frac{1}{\sum_{u\ne v}u和v之间的最短距离}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218215607008.png" alt="image-20221218215607008" style="zoom:80%;" />



**聚类系数（clustering coefficient）**：衡量节点邻居的连接程度，描述节点的局部结构信息。
$$
e_v=\frac{\#(相邻结点之间的边数)}{(_2^{k_v})}\in[0,1]
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218220847344.png" alt="image-20221218220847344" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218221019992.png" alt="image-20221218221019992" style="zoom:80%;" />

所以这个式子代表 $v$ 邻居所构成的节点对，即潜在的连接数。整个公式衡量节点邻居的连接有多紧密。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218221143420.png" alt="image-20221218221143420" style="zoom:80%;" />

在社交网络之中会有很多这种三角形，因为可以想象你的朋友可能会经由你的介绍而认识，从而构建出一个这样的三角形/三元组。





**有根连通异构子图（graphlet）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218221306456.png" alt="image-20221218221306456" style="zoom:80%;" />

对于某一给定节点数 $k$ ，会有 $n_k$ 个连通的异构子图。就是说，这些图首先是connected的，其次这些图有 $k$ 个节点，第三它们异构。节点数为2-5情况下一共能产生如图所示73种graphlet。这73个graphlet的核心概念就是不同的形状，不同的位置。

<font color=red>（注意这里的graphlet概念和后文图的graphlet kernel的概念不太一样）</font>

**Graphlet Degree Vector（GDV）**：节点基于graphlet的特征，是以给定节点为根的graphlet的计数向量。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218222430581.png" alt="image-20221218222430581" style="zoom:80%;" />

<font color=red>（如图所示，对四种graphlet，$v$ 的每一种graphlet的数量作为向量的一个元素。注意：graphlet $c$ 的情况不存在，是因为像graphlet $b$ 那样中间那条线连上了。这是因为graphlet是induced subgraph，所以那个边也存在，所以 $c$ 情况不存在）</font>

**GDV与其他两种描述节点结构的特征的区别**：

- **节点的度**：计算节点连接出去的边的个数。
- **聚类系数**：计算节点连接出去三角形的个数。
- **GDV**：计算节点连接出去的graphlet的个数。

考虑2-5个节点的graphlets，我们得到一个长度为73个坐标coordinate（就前图所示一共73种graphlet）的向量GDV，描述该点的局部拓扑结构topology of node’s neighborhood，可以捕获距离为4 hops的互联性interconnectivities。相比节点度数或clustering coefficient，GDV能够描述两个节点之间更详细的节点局部拓扑结构相似性local topological similarity。



**节点的特征可以分为两类**：

1. **Importance-based features**：捕获节点在图中的重要性。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218223041456.png" alt="image-20221218223041456" style="zoom:80%;" />

2. **Structure-based features**：捕获节点附近的拓扑属性。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218223226293.png" alt="image-20221218223226293" style="zoom:80%;" />



**结语**：传统节点特征只能识别出结构上的相似，不能识别出图上空间、距离上的相似。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218223407376.png" alt="image-20221218223407376" style="zoom:80%;" />



## 2.3 传统基于特征的方法：连接

预测任务是基于已知的边，预测新链接的出现。测试模型时，将每一对无链接的点对进行排序，取存在链接概率最高的K个点对，作为预测结果。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218223501448.png" alt="image-20221218223501448" style="zoom:80%;" />

有时你也可以直接将两个点的特征合并concatenate起来作为点对的特征，来训练模型。但这样做的缺点就在于失去了点之间关系的信息。



**链接预测任务的两种类型**：随机缺失边、随时间演化边。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218225339807.png" alt="image-20221218225339807" style="zoom:80%;" />



**基于相似性进行链接预测**：计算两点间的相似性得分（如用共同邻居衡量相似性），然后将点对进行排序，得分最高的n组点对就是预测结果，与真实值作比较。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218225453697.png" alt="image-20221218225453697" style="zoom: 80%;" />



**基于距离的特征**：两点间最短路径的长度。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218225817449.png" alt="image-20221218225817449" style="zoom: 80%;" />

这种方式的问题在于没有考虑两个点邻居的重合度the degree of neighborhood overlap，如B-H有2个共同邻居，B-E和A-B都只有1个共同邻居。



**局部邻域重合（local neighborhood overlap）**：捕获节点的共同邻居数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218225945387.png" alt="image-20221218225945387" style="zoom:80%;" />

common neighbors的问题在于度数高的点对就会有更高的结果，Jaccard’s coefficient是其归一化后的结果。Adamic-Adar index在实践中表现得好。在社交网络上表现好的原因：有一堆度数低的共同好友比有一堆名人共同好友的得分更高。



**全局邻域重合（global neighborhood overlap）**：局部邻域重合的限制在于，如果两个点没有共同邻居，值就为0。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218230211593.png" alt="image-20221218230211593" style="zoom:80%;" />

但是这两个点未来仍有可能被连接起来。所以我们使用考虑全图的global neighborhood overlap来解决这一问题。



**Katz指标（Katz index）**：计算点对之间所有长度路径的条数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218230630564.png" alt="image-20221218230630564" style="zoom:80%;" />

<font color=red>（discount factor β会给比较长的距离以比较小的权重）</font>

**证明**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218230859091.png" alt="image-20221218230859091" style="zoom:80%;" />

计算方式：邻接矩阵求幂，邻接矩阵的 $k$ 次幂结果，每个元素就是对应点对之间长度为 $k$ 的路径的条数。

**证明**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218230456274.png" alt="image-20221218230456274" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218230522511.png" alt="image-20221218230522511" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218230600989.png" alt="image-20221218230600989" style="zoom:80%;" />



**结语**：

1. **基于距离的特征**：使用两节点之间的最短路径长度，但是无法捕捉邻域重叠。
2. **局部邻域重合**：
   - 捕捉两节点共有的邻居节点的数量。
   - 当两节点没有共有节点时为0.
3. **全局邻域重合**：
   - 使用全局图结构计算两节点之间的值。
   - Katz 指标计算两节点之间所有长度的路径数量。



## 2.4 传统基于特征的方法：图

图级别特征构建目标：找到能够描述全图结构的特征。



**背景**：核方法（Kernel Methods）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218231837307.png" alt="image-20221218231837307" style="zoom:80%;" />

**总结来说**：两个图的核 $K(G,G')$ 用标量衡量相似度，存在特征表示 $\Phi(\cdot)$ 使得 $K(G,G')=\Phi(G)^T\Phi(G')$ ， 定义好核后就可以直接应用核SVM之类的传统机器学习模型。



**概述**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218232204506.png" alt="image-20221218232204506" style="zoom:80%;" />



**关键思想**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218232327572.png" alt="image-20221218232327572" style="zoom:80%;" />

bag-of-words相当于是把文档表示成一个向量，每个元素代表对应word出现的次数。此处讲述的特征抽取方法也将是bag-of-something的形式，将图表示成一个向量，每个元素代表对应something出现的次数（这个something可以是node, degree, graphlet, color）

光用node不够的话，可以设置一个degree kernel，用bag-of-degrees来描述图特征。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218232450026.png" alt="image-20221218232450026" style="zoom:80%;" />



**graphlet 特征**：计算图中不同 graphlet 的数量。

注意这里对graphlet的定义跟上文节点层面特征抽取里的graphlet不一样。

**区别在于**：这里 graphlets 中的节点不需要相连，可以有相互隔离的节点。

**对每一种节点数，可选的graphlet**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218232729065.png" alt="image-20221218232729065" style="zoom:80%;" />

**graphlet count vector**：每个元素是图中对应graphlet的数量。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221218232825950.png" alt="image-20221218232825950" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219000743745.png" alt="image-20221219000743745" style="zoom:80%;" />



**graphlet kernel**就是直接点积两个图的graphlet count vector得到相似性。对于图尺寸相差较大的情况需进行归一化。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219000821496.png" alt="image-20221219000821496" style="zoom:80%;" />

**graphlet kernel的限制：计算昂贵**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219000853944.png" alt="image-20221219000853944" style="zoom:80%;" />



**WL 核（Weisfeiler-Lehman Kernel）**：相比graphlet kernel代价较小，效率更高。用节点邻居结构迭代地来扩充节点信息。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001010095.png" alt="image-20221219001010095" style="zoom:80%;" />

**实现算法：Weisfeiler-Lehman graph isomorphism test=color refinement**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001044116.png" alt="image-20221219001044116" style="zoom:80%;" />

**color refinement示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001211334.png" alt="image-20221219001211334" style="zoom:80%;" />

**对聚集后颜色取哈希值**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001244335.png" alt="image-20221219001244335" style="zoom:80%;" />

**把邻居颜色聚集起来**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001326913.png" alt="image-20221219001326913" style="zoom:80%;" />

**对聚集后颜色取哈希值**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001347114.png" alt="image-20221219001347114" style="zoom:80%;" />

**进行 $K$ 次迭代后，用整个迭代过程中颜色出现的次数作为Weisfeiler-Lehman graph feature**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001438688.png" alt="image-20221219001438688" style="zoom:80%;" />

**用上图的向量点积计算相似性，得到WL kernel**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001502204.png" alt="image-20221219001502204" style="zoom:80%;" />



**WL kernel的优势在于计算成本低**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001526302.png" alt="image-20221219001526302" style="zoom:80%;" />



**结语**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219001617452.png" alt="image-20221219001617452" style="zoom:80%;" />





# 3 节点嵌入



## 3.1 章节前言

**图表示学习（graph representation learning）**：学习到图数据用于机器学习的、与下游任务无关的特征，我们希望这个向量能够抓住数据的结构信息。

这个数据被称作**特征表示（feature representation）**或**嵌入（embedding）**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219172121812.png" alt="image-20221219172121812" style="zoom:80%;" />



**为什么要嵌入**：将节点映射到embedding space

- embedding的相似性可以反映原节点在网络中的相似性，比如定义有边连接的点对为相似的点，则这样的点的embedding应该离得更近。
- embedding编码网络信息。
- embedding可用于下游预测任务。



**node embedding举例**：二维节点嵌入可视化（将不同类的节点很好地分开了）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219172336089.png" alt="image-20221219172336089" style="zoom:80%;" />



## 3.2 节点嵌入：编码与解码

图 $G$ ，节点集合 $V$ ，邻接矩阵 $A$ （二元的）（简化起见：不考虑节点的特征或其他信息）

**节点嵌入**：目标是将节点编码为embedding space中的向量，使embedding的相似度（如点积）近似于图中节点的相似度（需要被定义）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219172553040.png" alt="image-20221219172553040" style="zoom:80%;" />



**编码（Encoder）**：将节点映射为embedding。

定义一个衡量节点相似度的函数（如衡量在原网络中的节点相似度）

**解码（Decoder）**：将embedding对映射为相似度得分。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219172839330.png" alt="image-20221219172839330" style="zoom:80%;" />



**两个关键组件**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219172953393.png" alt="image-20221219172953393" style="zoom:80%;" />



**浅编码（shallow encoding）**：最简单的编码方式，编码器只是一个**嵌入查找（embedding-lookup）**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219173245755.png" alt="image-20221219173245755" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219173456512.png" alt="image-20221219173456512" style="zoom:80%;" />

$Z$ 每列是一个节点所对应的embedding向量。$v$ 是一个其他元素都为0，对应节点位置的元素为1的向量。通过矩阵乘法的方式得到结果。

这种方式就是将每个节点直接映射为一个embedding向量，我们的学习任务就是直接优化这些embedding。

**缺点**：参数多，很难scale up到大型图上。

**优点**：如果获得了 $Z$，各节点就能很快得到embedding。

<font color=red>（有很多种方法：如DeepWalk，node2vec等）</font>



**节点相似的不同定义**：

- 有边
- 共享邻居
- 有相似的结构特征
- **随机游走random walk**定义的节点相似度



## 3.3 节点嵌入：随机行走方法

**统一符号表示notation**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219173831673.png" alt="image-20221219173831673" style="zoom:80%;" />

$P(v|\bold{z}_u)$ 是从 $u$ 开始随机游走能得到 $v$ 的概率，衡量 $u$ 和 $v$ 的相似性，用节点embedding向量相似性算概率。
用于计算预测概率的非线性函数：softmax会将一组数据归一化为和为1的形式，最大值的结果会几乎为1。sigmoid会将实数归一化到 (0,1) 上。



**随机行走（random walk）**：从某一节点开始，每一步按照概率选一个邻居，走过去，重复。停止时得到一系列节点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219174226577.png" alt="image-20221219174226577" style="zoom:80%;" />



**随机行走嵌入（random walk embeddings）**：
$$
\bold{z}_u^T\bold{z}_v\approx u和v在一次随机游走中(以u为起点)同时出现的概率
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219174503012.png" alt="image-20221219174503012" style="zoom:80%;" />



**为什么选择随机行走方法**：

表达性：灵活的节点相似性随机定义，结合了局部和高阶邻域信息。

想法：如果从节点 𝒖 开始随机行走访问 𝒗 的概率很高，则 𝒖 和 𝒗 相似（高阶多跳信息）

效率：训练时不需要考虑所有节点对；只需要考虑随机行走中同时出现的节点对。



**无监督特征学习**：

直觉：查找节点嵌入𝑑-维空间的向量，可以保持相似性。

想法：学习节点嵌入，使网络中邻接的节点相互靠近。

【**问题**】**给定节点 $u$ ，如何定义邻接节点**？

$N_R(u)$ 用于表示节点 $u$ 通过一些随机行走策略 $R$ 得到的邻居。



**作为优化的特征学习**：

目标是使每个节点 $u$ ，$N_R(u)$ 的节点与 $\bold{z}_u$ 靠近，也就是 $P(N_R(u)|\bold{z}_u)$ 值很大。
$$
f:u\rightarrow R^d,f(u)=\bold{z}_u
$$
**极大似然目标函数**：
$$
\max_{f}\sum_{u\in V}logP(N_R(u)|\bold{z}_u)
$$
对这个目标函数的理解是：对节点 $u$ ，我们希望其表示向量对其random walk neighborhood $N_R(u)$ 的节点是predictive的（可以预测到它们的出现）



**随机行走的优化**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219193517916.png" alt="image-20221219193517916" style="zoom:80%;" />

把最大似然估计翻过来，拆开，就成了需被最小化的损失函数 $L$ ：
$$
L=\sum_{u\in V}\sum_{v\in N_R(u)}-log(P(v|\bold{z}_u))
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219194929453.png" alt="image-20221219194929453" style="zoom:80%;" />

这个计算概率 $P(v|\bold{z}_u)$ 选用 softmax 的 intuition 就是前文所提及的，softmax 会使最大元素输出靠近1，也就是在节点相似性最大时趋近于1。

将 $P(v|\bold{z}_u)$ 代入 $L$ 得到：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219195620507.png" alt="image-20221219195620507" style="zoom:80%;" />

优化 random walk embeddings 就是找到 $\bold{z}$ 使得 $L$ 最小。

但是计算这个公式代价很大，因为需要内外加总2次所有节点，复杂度达 $O(|V|^2)$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219195759374.png" alt="image-20221219195759374" style="zoom:80%;" />

我们发现问题就在于用于softmax归一化的这个分母：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219195937991.png" alt="image-20221219195937991" style="zoom:80%;" />

为了解决这个分母，我们使用negative sampling的方法：简单来说就是原本我们是用所有节点作为归一化的负样本，现在我们只抽出一部分节点作为负样本，通过公式近似减少计算。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219200111480.png" alt="image-20221219200111480" style="zoom:80%;" />

这个随机分布不是uniform random，而是random in a biased way：概率与其度数成比例。

负样本个数 $k$ 的考虑因素：

- 更高的 $k$ 会使估计结果更鲁棒robust（我的理解是方差）
- 更高的k会使负样本上的偏差bias更高（其实我没搞懂这是为什么）
- 实践上的k常用值：5-20



**优化目标函数（最小化损失函数）的方法**：随机梯度下降 $SGD$



**随机行走总结**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219201101939.png" alt="image-20221219201101939" style="zoom:80%;" />



**随机行走策略**：

1. **DeepWalk**：仅使用固定长度，无偏地从起始节点开始进行随机行走。但是相似度概念受限。

2. **node2vec**：有弹性的网络邻居 $N_R(u)$ 定义使 $u$ 的embedding更丰富，因此使用有偏的二阶随机游走策略以产生 $N_R(u)$ 。

   用有弹性、有偏的随机游走策略平衡local（$BFS$）和global（$DFS$）的节点网络结构信息。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219201709451.png" alt="image-20221219201709451" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219201736693.png" alt="image-20221219201736693" style="zoom:80%;" />

   **有偏定长随机游走的参数**：

   - return parameter $p$ ：返回上一个节点的概率
   - in-out parameter $q$ ：向外走（DFS）VS 向内走（BFS），相比于DFS，选择BFS的概率。

   **有偏随机游走举例**：上一步是 $(s_1,w)$

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219201938196.png" alt="image-20221219201938196" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219202002229.png" alt="image-20221219202002229" style="zoom:80%;" />

   BFS-like walk会给p较低的值，DFS-like walk会给q较低的值。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219202159734.png" alt="image-20221219202159734" style="zoom:80%;" />

   **node2vec算法**：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219202250294.png" alt="image-20221219202250294" style="zoom:80%;" />

   线性时间复杂度是因为节点邻居数量是固定的。

3. **其他随机游走方法**：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219202645170.png" alt="image-20221219202645170" style="zoom:80%;" />



## 3.4 嵌入整个图

任务目标：嵌入子图或整个图 $G$ ，得到表示向量 $\bold{z}_G$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219212744133.png" alt="image-20221219212744133" style="zoom:80%;" />

**任务示例**：

- 分类有毒/无毒分子
- 识别异常图

**也可以视为对节点的一个子集的嵌入**：

1. **方法1**：聚合（加总或求平均）节点的嵌入
   $$
   \bold{z}_G=\sum_{v\in G}\bold{z}_v
   $$

2. **方法2**：创造一个假节点（virtual node），用这个节点嵌入来作为图嵌入。这个virtual node和它想嵌入的节点子集（比如全图）相连。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219213017940.png" alt="image-20221219213017940" style="zoom:80%;" />

3. **方法3**： anonymous walk embeddings 以节点第一次出现的序号（是第几个出现的节点）作为索引。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219213117191.png" alt="image-20221219213117191" style="zoom:80%;" />

   这种做法会使具体哪些节点被游走到这件事不可知（因此匿名）。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219213159593.png" alt="image-20221219213159593" style="zoom:80%;" />

   **anonymous walks的个数随walk长度指数级增长**：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219213248336.png" alt="image-20221219213248336" style="zoom:80%;" />

   **anonymous walks的应用**：

   - **模拟图上长为 $l$ 的匿名随机游走**：将图表示为walks的概率分布向量（我感觉有点bag-of-anonymous walks、有点像GDV那些东西，总之都是向量每个元素代表其对应object的出现概率/次数）

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219213443924.png" alt="image-20221219213443924" style="zoom:80%;" />

     **sampling anonymous walks**：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219213618797.png" alt="image-20221219213618797" style="zoom:80%;" />

   - **行走嵌入（walk embeddings）**：$\bold{Z}_G$ 和匿名游走嵌入 $\bold{z}_i$​ 。得到 $\bold{Z}_G$ 后可用于预测任务，可以视其内积为核，照第二章所介绍的核方法来进行机器学习；也可以照后续课程将介绍的神经网络方法来进行机器学习。

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219214105993.png" alt="image-20221219214105993" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219214133760.png" alt="image-20221219214133760" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219214414223.png" alt="image-20221219214414223" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221219214446149.png" alt="image-20221219214446149" style="zoom:80%;" />





# 4 链接分析：网页排名（图转换为矩阵）



## 4.1 图转换为矩阵

本节课研究矩阵角度的图分析和学习。这里的矩阵就是指邻接矩阵。

将图视为矩阵形式，可以通过**随机游走**的方式定义节点重要性（即PageRank），通过**矩阵分解**matrix factorization (MF)来获取**节点嵌入**，将其他节点嵌入（如node2vec）也视作MF。



## 4.2 网页排名/谷歌算法

PageRank是谷歌搜索用的算法，用于对网页的重要性进行排序。在搜索引擎应用中，可以对网页重要性进行排序，从而辅助搜索引擎结果的网页排名。



在现实世界中，将整个互联网视作图：将网页视作节点，将网页间的超链接视作边

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220202911330.png" alt="image-20221220202911330" style="zoom:80%;" />



**一个网页之间互相链接的情况的示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220203041167.png" alt="image-20221220203041167" style="zoom:80%;" />

老一点的网页超链接都是navigational纯导航到其他页面的，当代的很多链接则是transactional用于执行发布、评论、点赞、购买等功能事务的。本课程中主要仅考虑那种网页之间互相链接的情况。



**将网页看作有向图，以链接指向作为边的方向（这个网页/节点能直接跳转到的网页就作为其下一个节点successor）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220203145465.png" alt="image-20221220203145465" style="zoom:80%;" />



**其他可表现为有向图形式的信息网络示例：论文引用，百科全书中词条间的互相引用**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220203226222.png" alt="image-20221220203226222" style="zoom:80%;" />



在图中，我们想要定义节点的**重要性**importance，通过网络图链接结构来为网页按重要性分级rank。以下将介绍3种用以计算图中节点重要性的方法：

- **PageRank**
- **Personalized PageRank (PPR)**
- **Random Walk with Restarts (RWR)**



**衡量节点重要性**：认为一个节点的链接越多，那么这个节点越重要。

有向图有in-coming links和out-going links两种情况。可以想象，in-links比较不容易造假，比较靠谱，所以用in-links来衡量一个节点的重要性。可以认为一个网页链接到下一网页，相当于对该网页重要性投了票（vote）。所以我们认为一个节点的in-links越多，那么这个节点越重要。同时，我们认为来自更重要节点的in-links，在比较重要性时的权重vote更大。

这就成了一个递归recursive的问题——要计算一个节点的重要性就要先计算其predecessors的重要性，计算这些predecessors的重要性又要先计算它们predecessors的重要性……

### 4.2.1 网页排名：流模型

链接权重与其source page的重要性成正比例。

如果网页 $i$ 的重要性是 $r_i$ ，有 $d_i$ 个out-links，那么每个边的权重就是 $r_i/d_i$ 。

网页 $j$ 的重要性 $r_j$ 是其 in-links 上权重的总和。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220203653110.png" alt="image-20221220203653110" style="zoom:80%;" />

从而得到对节点 $j$ 的级别 $r_j$ 的定义：
$$
r_j=\sum_{i\rightarrow j}\frac{r_i}{d_i},其中d_i是节点i的出度
$$


**以这个1839年的网络为例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220203848249.png" alt="image-20221220203848249" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220203859467.png" alt="image-20221220203859467" style="zoom:80%;" />

在直觉上我们好像可以用高斯消元法Gaussian elimination来解这个线性方程组，但这种方法不scalable。所以我们寻找更scalable的矩阵形式解法。

### 4.2.2 网页排名：矩阵表达

建立**随机邻接矩阵（stochastic adjacency matrix）** $M$ ，网页 $j$ 有 $d_j$ 条 out-links，如果 $j\rightarrow i$ ，$M_{ij}=\frac{1}{d_j}$ 。

$M$ 是列随机矩阵column stochastic matrix（列和为1），$M$ 的第 $j$ 列可以视作 $j$ 在邻居节点上的概率分布。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220204738807.png" alt="image-20221220204738807" style="zoom:80%;" />

rank vector $r$ ：每个网页 $i$ 的重要性得分 $r_i$ ，$\sum_i{r_i}=1$ ，所以 $r$ 也可被视为是网络中所有节点的概率分布。

flow equations可以被写成：$\bold{r}=M\cdot \bold{r}$ 。

回忆一下原公式 $r_j=\sum_{i\rightarrow j}\frac{r_i}{d_i}$ ，$M$ 的第 $j$ 行被指向的节点对应的列 $i$ 的元素就是 $\frac{1}{d_i}$ ，该列对应的是 $r_i$ ，加总起来就得到上个公式。



**flow equation和矩阵对比的举例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220205313554.png" alt="image-20221220205313554" style="zoom:80%;" />

### 4.2.3 连接到随机行走

假想一个web surfer的随机游走过程，在 $t$ 时间在网页 $i$ 上，在 $t+1$ 时间从 $i$ 的out-links中随机选一条游走。如此重复过程。

向量 $\mathbf{p}(t)$ 的第 $i$ 个坐标是 $t$ 时间web surfer在网页 $i$ 的概率，因此向量 $\mathbf{p}(t)$ 是网页间的概率分布向量。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220213526523.png" alt="image-20221220213526523" style="zoom:80%;" />



**平稳分布（stationary distribution）**：
$$
\bold{p}(t+1)=M\cdot\bold{p}(t)
$$
$M$ 是web surfer的转移概率，这个公式的逻辑感觉和 $\mathbf{r}=M\cdot \mathbf{r}$ 其实类似。

如果达到这种条件，即下一时刻的概率分布向量与上一时刻的相同：
$$
\bold{p}(t+1)=M\cdot\bold{p}(t)=\bold{p}(t)
$$
则 $\mathbf{p}(t)$ 是随机游走的stationary distribution。

$\bold{r}=M\cdot\bold{r}$ ，所以 $\mathbf{r}$ 是随机游走的stationary distribution。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220214051982.png" alt="image-20221220214051982" style="zoom:80%;" />

### 4.2.4 特征向量表达

无向图的邻接矩阵的特征向量是节点特征eigenvector centrality，而PageRank定义在有向图的随机邻接矩阵上。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220214345976.png" alt="image-20221220214345976" style="zoom:80%;" />
$$
1\cdot \bold{r}=M\cdot \bold{r}
$$
rank vector $\mathbf{r}$ 是随机邻接矩阵 $M$ 的特征向量，对应特征值为1。

从任一向量 $\mathbf{u}$ 开始，极限 $M(M(...M(M\mathbf{u})))$ 是web surfer的长期分布，也就是 $\mathbf{r}$（意思是无论怎么开局，最后结果都一样，这个感觉可以比较直觉地证明），PageRank=极限分布=M的principal eigenvector。

根据这个思路，我们就能找到PageRank的求解方式：power iteration

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220214704747.png" alt="image-20221220214704747" style="zoom:80%;" />

**极限分布（limiting distribution）**：相当于是random surfer一直随机游走，直至收敛，到达稳定状态。这个 $M$ 的叠乘可以让人联想到Katz index叠乘邻接矩阵 $A$ 。相比高斯消元法，power iteration是一种scalable的求解PageRank方法。

### 4.2.5 网页排名总结

1. 通过网络链接结构衡量图中节点的重要性。
2. 用随机邻接矩阵M建立web surfer随机游走模型。
3. PageRank解方程：$\bold{r}=M\cdot \bold{r}$ ，$\bold{r}$ 可被视作 $M$ 的principle eigenvector，也可被视作图中随机游走的stationary distribution。



## 4.3 网页排名：如何求解？

对每个节点赋初始PageRank。重复计算每个节点的PageRank $r_{j}^{(t+1)}=\sum_{i\rightarrow j}\frac{r_i^{(t)}}{d_i}$ （$d_i$ 是节点 $i$ 的出度）直至收敛 $\sum_i|r_i^{(t+1)}-r_i^t|<\epsilon$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220215618577.png" alt="image-20221220215618577" style="zoom:80%;" />



**幂迭代法（power iteration method）**：初始化 $r^0=[1/N,...,1/N]^T$ ，迭代 $\bold{r}^{(t+1)}=M\cdot\bold{r}^t$ 直到 $|\bold{t}^{(t+1)}-\bold{t}^{(t)}|<\epsilon$ ，约50次迭代即可逼近极限。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221628582.png" alt="image-20221220221628582" style="zoom:80%;" />



**power iteration示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221711084.png" alt="image-20221220221711084" style="zoom:80%;" />

![image-20221220221730356](C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221730356.png)



**PageRank的问题及其解决方案**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221807968.png" alt="image-20221220221807968" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221821136.png" alt="image-20221220221821136" style="zoom:80%;" />



- **spider trap**：所有出边都在一个节点组内，会吸收所有重要性。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221858736.png" alt="image-20221220221858736" style="zoom:80%;" />

  **spider traps解决方案**：random jumps or teleports（teleport（通常见于科幻作品）（被）远距离传送，大概我就翻译成直接跳了）

  random surfer每一步以概率 $\beta$ 随机选择一条链接（用 $M$ ），以概率 $1-\beta$ 随机跳到一个网页上（ $\beta$ 一般在0.8-0.9之间）。

  这样surfer就会在几步后跳出spider trap。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220222204222.png" alt="image-20221220222204222" style="zoom:80%;" />

- **dead end**：没有出边，造成重要性泄露。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220221928095.png" alt="image-20221220221928095" style="zoom:80%;" />

  **dead ends解决方案**：random jumps or teleports。

  从dead-ends出发的web surfer随机跳到任意网页（相当于从dead end出发，向所有节点连了一条边）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220222254455.png" alt="image-20221220222254455" style="zoom:80%;" />



spider-traps在数学上不是个问题，但是无法得到我们想要的PageRank结果；因此要在有限步内跳出spider traps。

dead-ends在数学上就是问题（其随机邻接矩阵列和不为0，初始假设直接不成立），因此要直接调整随机邻接矩阵，让web surfer无路可走时可以直接teleport。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220222321376.png" alt="image-20221220222321376" style="zoom:80%;" />



**整体解决方案**：random teleport

random surfer每一步以概率 $\beta$ 随机选择一条链接（$M$），以概率 $1-\beta$ 随机跳到一个网页上。
$$
r_j=\sum_{i\rightarrow j}\beta\frac{r_i}{d_i}+(1-\beta)\frac1N,d_i是节点i的出度
$$
$M$ 需要没有dead ends，可以通过直接去除所有dead ends或显式将dead ends跳到随机节点的概率设置到总和为1。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220222542255.png" alt="image-20221220222542255" style="zoom:80%;" />

**The Google Matrix $G$**：
$$
G=\beta M+(1-\beta)[\frac1N]_{N\times N}
$$
其中 $[\frac1N]_{N\times N}$ 是每个元素都是 $\frac{1}{N}$ 的 $N\times N$ 的矩阵。

现在 $ \mathbf{r}=G\cdot\mathbf{r}$ 又是一个迭代问题，power iteration方法仍然可用，$\beta$ 在实践中一般用0.8或0.9（平均5步跳出一次）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220223448467.png" alt="image-20221220223448467" style="zoom:80%;" />



**random teleports举例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220223537910.png" alt="image-20221220223537910" style="zoom:80%;" />

$M$ 是个spider trap，所以加上random teleport links，$G$ 也是一个转移概率矩阵。

**PageRank结果示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220223640729.png" alt="image-20221220223640729" style="zoom:80%;" />



**PageRank求解部分总结**：

用power iteration方法求解 $\mathbf{r}=G\cdot\mathbf{r}$（$G$ 是随机邻接矩阵）。

用random uniform teleporation解决dead-ends和spider-traps问题。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220223752244.png" alt="image-20221220223752244" style="zoom:80%;" />



## 4.4 重启动随机行走&个性化网页排名

**举例**：推荐问题（一个由user和item两种节点组成的bipartite graph）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220223848088.png" alt="image-20221220223848088" style="zoom:80%;" />



**Bipartite User-Item Graph**：

求解目标：图节点间相似性（针对与item Q交互的user，应该给他推荐什么别的item？）

可以直觉地想到，如果item Q和P都与相似user进行过交互，我们就应该推荐Q。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220223956812.png" alt="image-20221220223956812" style="zoom:80%;" />

**但是我们如何量化这个相似性呢**？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224017377.png" alt="image-20221220224017377" style="zoom:80%;" />



**衡量节点相似性的问题**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224043034.png" alt="image-20221220224043034" style="zoom:80%;" />

A A’比B B’近可以因为它们之间距离较短；但是A A’和C C’距离相等，C C’却有着共同用户，这又如何比较呢？如果引入shared neighbors作为标准，D D’和C C’有相同的share neighbors，但是D D’的共同用户之间的相似性却很低，这又如何衡量呢？



**图上的相似性**：Random Walks with Restarts

- PageRank用重要性来给节点评级，随机跳到网络中的任何一个节点。

- Personalized PageRank衡量节点与teleport nodes $\mathbf{S}$ 中的节点的相似性。

- 用于衡量其他item与item Q的相似性：Random Walks with Restarts只能跳到起始节点：$\bold{S}=\{\bold{Q}\}$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224317518.png" alt="image-20221220224317518" style="zoom:80%;" />



**Random Walks**：每个节点都有重要性，在其边上均匀分布，传播到邻居节点。对 query_nodes 模拟随机游走：

1. 随机游走到一个邻居，记录走到这个节点的次数（visit count）
2. 以 alpha 概率从 query_nodes 中某点重启
3. 结束随机游走后，visit count最高的节点就与 query_nodes 具体最高的相似性（直觉上这就是 query_nodes 最容易走到、最近的点了）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224502107.png" alt="image-20221220224502107" style="zoom:80%;" />



**以 Q 作为示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224525882.png" alt="image-20221220224525882" style="zoom:80%;" />

**算法伪代码（从item随机游走到另一个item，记录visit_count；以一定概率返回query_nodes）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224543503.png" alt="image-20221220224543503" style="zoom:80%;" />

**结果示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224618281.png" alt="image-20221220224618281" style="zoom:80%;" />

在示例中是模拟的random walk，但其实也可以用power iteration的方式来做。



RWR的优点在于，这种相似性度量方式考虑到了网络的如下复杂情况：

1. multiple connections
2. multiple paths
3. direct and indirect connections
4. degree of the node



**对不同PageRank变体的总结**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224745983.png" alt="image-20221220224745983" style="zoom:80%;" />



**总结**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224812537.png" alt="image-20221220224812537" style="zoom:80%;" />



## 4.5 矩阵表示与节点嵌入

**回忆上一章讲到的embedding lookup的encoder**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220224905233.png" alt="image-20221220224905233" style="zoom:80%;" />



**将节点嵌入视作矩阵分解**：

假设有边的节点是相似节点，则 $\mathbf{z}_v^T\mathbf{z}_u=A_{u,v}$（A是邻接矩阵）(如有边连接，则节点嵌入相似性直接取到1)，则 $\mathbf{Z}^T\mathbf{Z}=A$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225117796.png" alt="image-20221220225117796" style="zoom:80%;" />



**矩阵分解问题**：节点表示向量维度远低于节点数。如上一序号，将节点嵌入视作矩阵分解问题，严格的矩阵分解 $A=\mathbf{Z}^T\mathbf{Z}$ 很难实现（因为没有那么多维度来表示），因此通过最小化 $A$ 和 $\mathbf{Z}^T\mathbf{Z}$ 间的L2距离（元素差的平方和）来近似目标。

在Lecture 3中我们使用softmax来代替L2距离完成目标。

所以用边连接来定义节点相似性的inner product decoder等同于A的矩阵分解问题。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225327876.png" alt="image-20221220225327876" style="zoom:80%;" />



**矩阵分解问题**：基于random walk定义的相似性

DeepWalk和node2vec有基于random walk定义的更复杂的节点相似性，但还是可以视作矩阵分解问题，不过矩阵变得更复杂了。（相当于是把上面的 $A$ 给换了）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225427965.png" alt="image-20221220225427965" style="zoom:80%;" />



**通过矩阵分解和随机游走进行节点嵌入的限制**：

- 无法获取不在训练集中的节点嵌入，每次新增节点都要对全部数据集重新计算嵌入。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225500161.png" alt="image-20221220225500161" style="zoom:80%;" />

- 无法捕获结构相似性：比如图中节点1和节点11在结构上很相似，但是节点嵌入会差别很大（随机游走走不过去）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225542607.png" alt="image-20221220225542607" style="zoom:80%;" />

- 无法使用节点、边和图上的特征信息。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225629513.png" alt="image-20221220225629513" style="zoom:80%;" />



## 4.6 总结

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221220225659555.png" alt="image-20221220225659555" style="zoom:80%;" />



# 5. Colab1

本colab以无向图 Karate Club Network1 （有34个节点，78条边）为例，探索该数据集的相关统计量，并将从NetworkX下载的数据集转换为PyTorch的Tensor格式，用边连接作为节点相似性度量指标实现shallow encoder（以 nn.Embedding 为embedding-lookup）的节点嵌入代码。



节点嵌入训练概览：

- 用图中原本的边作为正值，从不存在的边中抽样作为负值，将对应边/节点对的点积结果用sigmoid归一化后视作输出值，将1视为正值的标签，0视为负值的标签。用BCELoss计算损失函数。
- 将nn.Embedding作为参数，用PyTorch在神经网络中以随机梯度下降的方式进行训练。
- 最后通过PCA将nn.Embedding.weight（即embedding-lookup的值）降维到二维上，通过可视化的方式直观检验训练效果。



## 5.0 Python 包导入

```python
import networkx as nx

import torch
import torch.nn as nn
from torch.optim import SGD

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import random
```





























# 6. 消息传递与节点分类

**本章主要内容**：
我们的任务是：已知图中一部分节点的标签，用图中节点之间的关系来将标签分配到所有节点上。属于半监督学习任务。

本节课我们学习message passing方法来完成这一任务。对某一节点的标签进行预测，需要其本身特征、邻居的标签和特征。

message passing的假设是图中相似的节点之间会存在链接，也就是相邻节点有标签相同的倾向。这种现象可以用homophily（**相似节点倾向于聚集**）、influence（**关系会影响节点行为**）、confounding（**环境影响行为和关系**）来解释。

collective classification给所有节点同时预测标签的概率分布，基于马尔科夫假设（某一点标签仅取决于其邻居的标签）。

local classifier（用节点特征预测标签）→ relational classifier（用邻居标签 和/或 特征，预测节点标签）→ collective inference（持续迭代）



**本节课讲如下三种collective classification的实现技术**：

- relational classification：用邻居标签概率的加权平均值来代表节点标签概率，循环迭代求解。
- iterative classification：在训练集上训练用 (节点特征) 和 (节点特征，邻居标签summary $z$ ) 两种自变量预测标签的分类器 $\phi_1$ 和 $\phi_2$，在测试集上用 $\phi_1$ 赋予初始标签，循环迭代求解 $z\rightleftharpoons$ 用 $\phi_2$ 重新预测标签
- belief propagation：在边上传递节点对邻居的标签概率的置信度（belief）的message/estimate，迭代计算边上的message，最终得到节点的belief。有环时可能出现问题。



## 6.1 消息传递与节点分类

**本章重要问题**：给定网络中部分节点的标签，如何用它们来分配整个网络中节点的标签？（举例：已知网络中有一些诈骗网页，有一些可信网页，如何找到其他诈骗和可信的网页节点？）

训练数据中一部分有标签，剩下的没标签，这种就是半监督学习。

对这个问题的一种解决方式：node embeddings。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221171451122.png" alt="image-20221221171451122" style="zoom:80%;" />



**本章我们介绍一个解决上述问题的方法**：**message passing**

使用message passing基于“网络中存在关系correlations”这一直觉，亦即相似节点间存在链接。

message passing是通过链接传递节点的信息，感觉上会比较类似于 PageRank 将节点的vote通过链接传递到下一节点，但是在这里我们更新的不再是重要性的分数，而是对节点标签预测的概率。

核心概念 **collective classification**：同时将标签分配到网络中的所有节点上。



**本章将讲述三种实现技术**：

- **关系分类（relational classification）**
- **迭代分类（iterative classification）**
- **信念传播（belief propagation）**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221171629966.png" alt="image-20221221171629966" style="zoom:80%;" />



举例：节点分类

半监督学习问题：给出一部分节点的标签（如图中给出了一部分红色节点、一部分绿色节点的标签），预测其他节点的标签

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221171655526.png" alt="image-20221221171655526" style="zoom:80%;" />



网络中存在关系correlations：

相似的行为在网络中会互相关联。

correlation：相近的节点会具有相同标签

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221171733119.png" alt="image-20221221171733119" style="zoom:80%;" />



导致correlation的两种解释：

homophily（同质性，趋同性，同类相吸原则）：个体特征影响社交连接

influence：社交连接影响个体特征

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221171813381.png" alt="image-20221221171813381" style="zoom:80%;" />

homophily：相似节点会倾向于交流、关联（物以类聚，人以群分）

在网络研究中得到了大量观察

举例：同领域的研究者更容易建立联系，因为他们参加相同的会议、学术演讲……等活动

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221171901981.png" alt="image-20221221171901981" style="zoom:80%;" />

homophily举例：一个在线社交网络，以人为节点，友谊为边，通过人们的兴趣将节点分为多类（用颜色区分）。

从图中可以看出，各种颜色都分别聚在一起，亦即有相同兴趣（同色）的人们更有聚集在一起的倾向。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172000457.png" alt="image-20221221172000457" style="zoom:80%;" />

influence：社交链接会影响个人行为。

举例：用户将喜欢的音乐推荐给朋友。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172155889.png" alt="image-20221221172155889" style="zoom:80%;" />



既然知道了网络中关系的影响机制，我们就希望能够通过网络中的链接关系来辅助预测节点标签。

如图举例，我们希望根据已知的绿色（label 1）和红色（label 0）节点来预测灰色（标签未知）节点的标签。将各节点从 1-9 标注上node-id。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172236285.png" alt="image-20221221172236285" style="zoom:80%;" />



**解决分类问题的逻辑**：我们已知相似节点会在网络中更加靠近，或者直接相连。

因此根据关联推定guilt-by-association：如果我与具有标签 $X$ 的节点相连，那么我也很可能具有标签 $X$（基于马尔科夫假设）

举例：互联网中的恶意/善意网页：恶意网页往往会互相关联，以增加曝光，使其看起来更可靠，并在搜索引擎中提高排名。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172359112.png" alt="image-20221221172359112" style="zoom:80%;" />



**预测节点 $v$ 的标签需要**：

- $v$ 的特征
- $v$ 邻居的标签
- $v$ 邻居的特征

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172448435.png" alt="image-20221221172448435" style="zoom:80%;" />



**半监督学习**：

任务：假设网络中存在homophily，根据一部分已知标签（红/绿）的节点预测剩余节点的标签

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172543671.png" alt="image-20221221172543671" style="zoom:80%;" />

示例任务：$A$ 是 $n\times n$ 的邻接矩阵，$Y=\{0,1\}^n$ 是标签向量，目标是预测未知标签节点属于哪一类。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172647058.png" alt="image-20221221172647058" style="zoom:80%;" />

**解决方法**：**collective classification**



**collective classification的应用**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172734825.png" alt="image-20221221172734825" style="zoom:80%;" />



**collective classification概述**：使用网络中的关系同时对相连节点进行分类，使用概率框架（propabilistic framework），基于马尔科夫假设，节点 $v$ 的标签 $Y_v$ 取决于其邻居 $N_v$ 的标签 $P(Y_v)=P(Y_v|N_v)$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221172948796.png" alt="image-20221221172948796" style="zoom:80%;" />



**集体分类法（collective classification）分成三步**：

1. **局部分类器（local classifier）**：分配节点的初始标签（基于节点特征建立标准分类，不使用网络结构信息）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221173624945.png" alt="image-20221221173624945" style="zoom:80%;" />

2. **关系分类器（relational classifier）**：捕获关系（基于邻居节点的标签 和/或 特征，建立预测节点标签的分类器模型）（应用了网络结构信息）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221173801304.png" alt="image-20221221173801304" style="zoom:80%;" />

3. **集体推论（collective inference）**：传播关系（在每个节点上迭代relational classifier，直至邻居间标签的不一致最小化。网络结构影响最终预测结果）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221173933729.png" alt="image-20221221173933729" style="zoom:80%;" />



**问题设置**：预测无标签（灰色）节点 $v$ 的标签 $Y_v$ 。所有节点 $v$ 具有特征向量 $f_v$ 。部分节点的标签已给出（绿色是1，红色是0）

**任务**：求解 $P\left(Y_v\right)$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221174133477.png" alt="image-20221221174133477" style="zoom:80%;" />



**对本章后文内容的概述**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221174204356.png" alt="image-20221221174204356" style="zoom:80%;" />



## 6.2 关系分类/概率关系分类器

**基础思想**：节点 $v$ 的类概率 $Y_v$ 是其邻居类概率的加权平均值。

- 对有标签节点 $v$ ，固定 $Y_v$ 为真实标签（ground-truth label）$Y_v^*$ 。
- 对无标签节点 $v$ ，初始化 $Y_v$ 为 0.5。

以随机顺序（不一定要是随机顺序，但实证上发现最好是。这个顺序会影响结果，尤其对小图而言）更新所有无标签节点，直至收敛或到达最大迭代次数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221174529819.png" alt="image-20221221174529819" style="zoom:80%;" />



**更新公式**：
$$
P(Y_v=c)=\frac{1}{\sum_{(v,u)\in E}A_{v,u}}\sum_{(v,u)\in E}A_{v,u}P(Y_u=c)
$$
邻接矩阵 $A_{v,u}$ 可以带权；分母是节点 $v$ 的度数或入度；$P(Y_u=c)$ 是节点 $v$ 标签为 $c$ 的概率。

**问题**：

- 不一定能收敛
- 模型无法使用节点特征信息

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221180752099.png" alt="image-20221221180752099" style="zoom:80%;" />



**举例**：

1. 初始化：迭代顺序就是node-id的顺序。有标签节点赋原标签，无标签节点赋0，$P_{Y_v}=P(Y_v=1)$ 。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221180909037.png" alt="image-20221221180909037" style="zoom:80%;" />

2. **第一轮迭代**：

   - 更新节点3：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181055581.png" alt="image-20221221181055581" style="zoom:80%;" />

   - 更新节点4：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181114707.png" alt="image-20221221181114707" style="zoom:80%;" />

   - 更新节点5：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181136914.png" alt="image-20221221181136914" style="zoom:80%;" />

   - 更新完所有无标签节点：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181206063.png" alt="image-20221221181206063" style="zoom:80%;" />

3. 第二轮迭代：结束后发现节点9收敛

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181226480.png" alt="image-20221221181226480" style="zoom:80%;" />

4. 第三轮迭代：结束后发现节点8收敛

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181247116.png" alt="image-20221221181247116" style="zoom:80%;" />

5. 第四轮迭代：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181304019.png" alt="image-20221221181304019" style="zoom:80%;" />

6. 收敛后：预测概率>0.5的节点为1，<0.5的为0

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181323446.png" alt="image-20221221181323446" style="zoom:80%;" />



## 6.3 迭代分类

关系分类器（relational classifiers）没有使用节点特征信息，所以我们使用新方法迭代分类（iterative classification）。

**迭代分类（iterative classification）主思路**：基于节点特征及邻居节点标签对节点进行分类

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181431690.png" alt="image-20221221181431690" style="zoom:80%;" />



**迭代分类的方法**：训练两个分类器

- $\phi_1(f_v)$ 基于节点特征向量 $f_v$ 预测节点标签。
- $\phi_2(f_v,z_v)$ 基于节点特征向量 $f_v$ 和邻居节点标签summary $z_v$ 预测节点标签。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181617467.png" alt="image-20221221181617467" style="zoom:80%;" />



**计算summary $z_v$ **：$z_v$ 是个向量，可以是邻居标签的直方图（各标签数目或比例），邻居标签中出现次数最多的标签，邻居标签的类数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221181809818.png" alt="image-20221221181809818" style="zoom:80%;" />



**迭代分类器iterative classifier的结构**：

1. **第一步：基于节点特征建立分类器**。在训练集上训练如下分类器（可以用线性分类器、神经网络等算法）：
   - $\phi_1(f_v)$ 基于 $f_v$ 预测 $Y_v$ 。
   - $\phi_2(f_v,z_v)$ 基于 $f_v$ 和 $z_v$ 预测 $Y_v $ 。
2. **第二步：迭代至收敛**。在测试集上，用 $\phi_1$ 预测标签，根据 $\phi_1$ 计算出的标签计算 $z_v$ 并用 $\phi_2$ 预测标签。对每个节点重复迭代计算 $z_v$ ，用 $\phi_2$ 预测标签这个过程，直至收敛或到达最大迭代次数（10, 50, 100……这样，不能太多）。注意：模型不一定能收敛。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221183247793.png" alt="image-20221221183247793" style="zoom:80%;" />



**计算举例：网页分类问题**。

节点是网页，链接是超链接，链接有向。节点特征简化设置为2维二元向量。预测网页主题。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221183329676.png" alt="image-20221221183329676" style="zoom:80%;" />

1. **基于节点特征训练分类器 $\phi_1$ **：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221183441703.png" alt="image-20221221183441703" style="zoom:80%;" />

   可以假设分类器以特征第一个元素作为分类标准，于是对中间节点分类错误。

2. **根据 $\phi_1$ 得到的结果计算 $z_v$**：此处设置 $z_v$ 为四维向量，四个元素分别为指向该节点的节点中标签为0和1的数目、该节点指向的节点中标签为0和1的数目。在这一步应用了网络结构信息。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221183604339.png" alt="image-20221221183604339" style="zoom:80%;" />



**过程举例**：

1. 第一步：在训练集上训练 $\phi_1$ 和 $\phi_2$ 。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221183722738.png" alt="image-20221221183722738" style="zoom:80%;" />

2. 第二步：在测试集上预测标签。

   - 用 $\phi_1$ 预测 $Y_v$ ：

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221184625794.png" alt="image-20221221184625794" style="zoom:80%;" />

   - 循环迭代：

     - 用 $Y_v$ 计算 $z_v$ 

       <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221184733190.png" alt="image-20221221184733190" style="zoom:80%;" />

     - 用 $\phi_2$ 预测标签

       <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221184820178.png" alt="image-20221221184820178" style="zoom:80%;" />

     - 迭代直至收敛

       <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221184847146.png" alt="image-20221221184847146" style="zoom:80%;" />

3. 结束迭代（收敛或达到最大迭代次数），得到最终预测结果。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221184924443.png" alt="image-20221221184924443" style="zoom:80%;" />



**对relational classification和iterative classification的总结**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221184952533.png" alt="image-20221221184952533" style="zoom:80%;" />



## 6.4 循环信念传播

名字叫loopy是因为loopy BP方法会应用在有很多环的情况下。



**信念传播（belief propagation）**是一种动态规划方法，用于回答图中的概率问题（比如节点属于某类的概率）。

邻居节点之间迭代传递信息pass message（如传递相信对方属于某一类的概率），直至达成共识（大家都这么相信），计算最终置信度（也有翻译为信念的）belief。

问题中节点的状态并不取决于节点本身的belief，而取决于邻居节点的belief。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185200522.png" alt="image-20221221185200522" style="zoom:80%;" />



**消息传递（message passing）**：

例子介绍：

任务：计算图中的节点数（注意，如果图中有环会出现问题，后文会讲有环的情况。在这里不考虑）

限制：每个节点只能与邻居交互（传递信息）

举例：path graph（节点排成一条线）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185300133.png" alt="image-20221221185300133" style="zoom:80%;" />

**算法**：

1. 定义节点顺序（生成一条路径）
2. 基于节点顺序生成边方向，从而决定message passing的顺序
3. 按节点顺序，计算其对下一节点的信息（至今数到的节点数），将该信息传递到下一节点

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185654710.png" alt="image-20221221185654710" style="zoom:80%;" />

每个节点**接收邻居信息，更新信息，传递信息**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185714363.png" alt="image-20221221185714363" style="zoom:80%;" />

将path graph的情况泛化到树上：从叶子节点到根节点传递信息

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185728028.png" alt="image-20221221185728028" style="zoom:80%;" />

在树结构上更新置信度：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185749246.png" alt="image-20221221185749246" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185803252.png" alt="image-20221221185803252" style="zoom:80%;" />



**Loopy BP Algorithm**：从 $i$ 传递给 $j$ 的信息，取决于 $i$ 从邻居处接收的信息。每个邻居给 $i$ 对其状态的置信度的信息。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221185852946.png" alt="image-20221221185852946" style="zoom:80%;" />



**一些记号说明**：

- label-label potential matrix $\psi$ ：节点及其邻居间的dependency。$\psi(Y_i,Y_j)$ 表示，节点 $i$ 是节点 $j$ 的邻居，已知 $i$ 属于类 $Y_i$ ， $\psi\left(Y_i,Y_j\right)$ 与 $j$ 属于类 $Y_j$ 的概率成比例。
- prior belief $\phi$ ：$\phi(Y_i)$ 与节点 $i$ 属于类 $Y_i$ 的概率成比例。
- $m_{i\rightarrow j}(Y_j)$ ：$i$ 对 $j$ 属于类 $Y_j$ 的message/estimate。
- $L$ ：所有类/标签的集合。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221190514429.png" alt="image-20221221190514429" style="zoom:80%;" />



**Loopy BP Algorithm**：

1. 将所有信息初始化为1。

2. 对每个节点重复：
   $$
   m_{i\rightarrow j}(Y_j)=\sum_{Y_i\in L}\psi(Y_i,Y_j)\phi_i(Y_i)\prod_{k\in N_i/\ j}m_{i\rightarrow j}(Y_i),\forall Y_j\in L
   $$
   <font color=red>（反斜杠指除了 $j$ ）</font>

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221191715111.png" alt="image-20221221191715111" style="zoom:80%;" />

3. 收敛后，计算 $b_i(Y_i) = 节点i属于类Y_i的置信度$ 
   $$
   b_i(Y_i)=\phi_i(Y_i)\prod_{j\in N_i}m_{k\rightarrow i}(Y_i)
   $$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192012933.png" alt="image-20221221192012933" style="zoom:80%;" />



**示例**：

1. 现在我们考虑图中有环的情况，节点没有顺序了。我们采用上述算法，从随机节点开始，沿边更新邻居节点。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192045313.png" alt="image-20221221192045313" style="zoom:80%;" />

2. 由于图中有环，来自各子图的信息就不独立了。信息会在圈子里加强（就像 PageRank 里的 spider trap）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192116341.png" alt="image-20221221192116341" style="zoom:80%;" />



**可能出现的问题：置信度不收敛（如图，信息在环里被加强了）**

但是由于现实世界的真实复杂图会更像树，就算有环也会有弱连接，所以还是能用Loopy BP

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192211909.png" alt="image-20221221192211909" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192227095.png" alt="image-20221221192227095" style="zoom:80%;" />



**置信度传播方法的特点**：

- **优点**：
  1. 易于编程及同步运算
  2. 可泛化到任何形式potentials（如高阶）的图模型上
- **挑战**：不一定能收敛（参考：尤其在闭环多的情况下）
- potential functions (parameters) (label-label potential matrix) 需要训练来估计

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192410139.png" alt="image-20221221192410139" style="zoom: 80%;" />



## 6.5 总结

学习了如何利用图中的关系来对节点做预测。



主要技术：

1. relational classification
2. iterative classification
3. loopy belief propagation

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221221192527683.png" alt="image-20221221192527683" style="zoom:80%;" />





# 7. 图神经网络1：GNN模型

**本章主要内容**：

介绍深度学习基础。

介绍GNN思想：聚合邻居信息。

每一层都产生一种节点嵌入。将上一层的邻居信息聚合起来，连接本节点上一层信息，产生新的节点嵌入。

第一层节点嵌入就是节点特征。

GCN：用平均值作为聚合函数。

GraphSAGE：用各种聚合函数。



## 7.1 图神经网络1：GNN模型

回忆一下节点嵌入任务。其目的在于将节点映射到 $d$ 维向量，使得在图中相似的节点在向量域中也相似。

我们已经学习了 “Shallow” Encoding 的方法来进行映射过程，也就是使用一个大矩阵直接储存每个节点的表示向量，通过矩阵与向量乘法来实现嵌入过程。

**这种方法的缺陷在于**：

- 需要 $O(|V|)$​ 复杂度（矩阵的元素数，即表示向量维度 $d$ ×节点数 $|V|$）的参数，太多了节点间参数不共享，每个节点的表示向量都是完全独特的。
- 直推式（transductive）：无法获取在训练时没出现过的节点的表示向量。
- 无法应用节点特征信息。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222161147906.png" alt="image-20221222161147906" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222161219308.png" alt="image-20221222161219308" style="zoom:80%;" />



本节课将介绍deep graph encoders，也就是用图神经网络GNN来进行节点嵌入。

映射函数，即之前讲过的node embedding中的encoder：$ENC\left(v\right)=$ 基于图结构的多层非线性转换。

（对节点相似性的定义仍然可以使用之前Lecture 3中的DeepWalk、node2vec等方法）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222161350942.png" alt="image-20221222161350942" style="zoom:80%;" />



**一个GNN网络的结构如图**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222161413975.png" alt="image-20221222161413975" style="zoom:80%;" />



**通过网络可以解决的任务有**：

- 节点分类：预测节点的标签
- 链接预测：预测两点是否相连
- 社区发现：识别密集链接的节点簇
- 网络相似性：度量图/子图间的相似性

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222161510817.png" alt="image-20221222161510817" style="zoom:80%;" />



**传统机器学习难以应用在图结构上**：图上的节点不全是独立同分布的,因此传统的机器学习无法直接运用到图上。



## 7.2 深度学习基础

**机器学习：一个优化任务**

有监督学习：输入自变量 $x$，预测标签 $y$ 。

**将该任务视作一个优化问题**：
$$
\min_{\Theta}L(y,f(x))
$$
$\Theta$ 是参数集合，优化对象，可以是一至多个标量、向量或矩阵。如在shallow encoder中 $\Theta=\{Z\}$ （就是embedding lookup，那个大矩阵）。

$L$ 是目标函数/损失函数。

举例：$L2$ loss（回归任务），$L(y,f(x))=||y-f(x)||_2$ 。

其他常见损失函数：L1 loss、huber loss、max margin（hinge loss）、交叉熵（后文将详细介绍）……等。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222162653597.png" alt="image-20221222162653597" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222162709352.png" alt="image-20221222162709352" style="zoom:80%;" />



损失函数举例：常用于分类任务的**交叉熵（cross entropy）**

标签 $y$ 是一个独热编码（所属类别索引的元素为1，其他元素为0）的分类向量，如 $y=[0,0,1,0,0]$ 。

输出结果 $f(x)$ 是经过softmax的概率分布向量，即 $f(x)=Softmax(g(x))$ ，如 $f(x)=[0.1,0.3,0.4,0.1,0.1]$ 。
$$
CE(y,f(x))=-\sum_{i=1}^C(y_ilogf(x)_i)
$$
其中 $C$ 是类别总数，下标 $i$ 代表向量中第 $i$ 个元素。$CE$ 越低越好，越低说明预测值跟真实值越近。

**在所有训练集数据上的总交叉熵**：
$$
L=\sum_{(x,y)\in T}CE(y,f(x))
$$
其中 $T$ 是所有训练集数据。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222163510716.png" alt="image-20221222163510716" style="zoom:80%;" />



梯度向量 $\nabla_\Theta L$ ：函数增长最快的方向和增长率，每个元素是对应参数在损失函数上的偏微分。

方向导数：函数在某个给定方向上的变化率。

梯度是函数增长率最快的方向的方向导数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222164025474.png" alt="image-20221222164025474" style="zoom:80%;" />



**梯度下降**：

迭代：将参数向梯度负方向更新：$\Theta\leftarrow \Theta-\eta\nabla_\Theta L$ 

学习率learning rate $\eta$ 是一个需要设置的超参数，控制梯度下降每一步的步长，可以在训练过程中改变（有时想要学习率先快后慢：LR scheduling）

理想的停止条件是梯度为0，在实践中一般则是用“验证集上的表现不再提升”作为停止条件。（据我的经验一般是设置最大迭代次数，如果在验证集上表现不再增加就提前停止迭代（early stopping））

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222164326445.png" alt="image-20221222164326445" style="zoom:80%;" />



**随机梯度下降stochastic gradient descent (SGD)**：

每一次梯度下降都需要计算所有数据集上的梯度，耗时太久，因此我们使用SGD的方法，将数据分成多个minibatch，每次用一个minibatch来计算梯度。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222164421619.png" alt="image-20221222164421619" style="zoom:80%;" />



**minibatch SGD**： SGD是梯度的无偏估计，但不保证收敛，所以一般需要调整学习率。

对SGD的改进优化器：Adam，Adagrad，Adadelta，RMSprop……等。

**一些概念**：

- batch size：每个minibatch中的数据点数
- iteration：在一个minibatch上做一次训练
- epoch：在整个数据集上做一次训练（在一个epoch中iteration的数量是 $\frac{dataset\_size}{batch\_size}$ ）

 <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222164913495.png" alt="image-20221222164913495" style="zoom:80%;" />



**神经网络目标函数**：
$$
\min_{\Theta}L(y,f(x))
$$
深度学习中的 $f$ 可能非常复杂，为了简化，先假设一个线性函数：$f(x)=W\cdot x$ ，$\Theta=\{W\}$ 。

- 如果 $f$ 返回一个标量，则 $W$ 是一个可学习的向量 $\nabla_Wf=(\frac{\part f}{\part w_1},\frac{\part f}{\part w_2},\frac{\part f}{\part w_3},...)$ 。
- 如果 $f$ 返回一个向量，则 $W$ 是一个权重矩阵 $\nabla_Wf=f的雅可比矩阵$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222165930939.png" alt="image-20221222165930939" style="zoom:80%;" />



**反向传播（Back Propagation）**：

对更复杂的函数，如 $f(x)=W_2(W_1x)$ ，$\Theta=\{W_1,W_2\}$ 。

将该函数视为：$h(x)=W_1x$ ，$f(h)=W_2h$ 。

应用链式法则计算梯度：$\nabla_xf=\frac{\part f}{\part h}\cdot\frac{\part h}{\part x}=\frac{\part f}{\part(W_1x)}\cdot\frac{\part(W_1x)}{\part x}$ 。

反向传播就是应用链式法则反向计算梯度，最终得到 $L$ 关于参数的梯度。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222170333487.png" alt="image-20221222170333487" style="zoom:80%;" />



**神经网络举例**：

简单两层线性网络 $f(x)=g(h(x))=W_2(W_1x)$

在一个minibatch上的 $L2$ loss ：$L_{(x,y)\in \Beta}=||y-f(x)||_2$  

**隐藏层**：$x$ 的中间表示向量。这里我们用 $h(x)=W_1x$ 来表示隐藏层，$f(x)=W_2h(x)$ 。

**前向传播**：从输入计算输出，用输出计算loss。

**反向传播**：计算梯度 $\frac{\part L}{\part W_2}=\frac{\part L}{\part f}\cdot\frac{\part f}{\part W_2}$ ，$\frac{\part L}{\part W_1}=\frac{\part L}{\part f}\cdot\frac{\part f}{\part W_2}\cdot\frac{\part W_2}{\part W_1}$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222171311032.png" alt="image-20221222171311032" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222171324296.png" alt="image-20221222171324296" style="zoom:80%;" />



**非线性**:

- ReLU：$ReLU(x)=max(x,0)$ 
- Sigmoid：$\sigma(x)=\frac1{1+e^{-x}}$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222171617277.png" alt="image-20221222171617277" style="zoom:80%;" />



**多层感知器Multi-layer Perceptron (MLP)**：

MLP每一层都是线性转换和非线性结合 
$$
x^{(l+1)}=\sigma(W_lx^{(l)}+b^l)
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222171739438.png" alt="image-20221222171739438" style="zoom:80%;" />



**总结**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222171806304.png" alt="image-20221222171806304" style="zoom:80%;" />



## 7.3 图上的深度学习

**本节内容**：

1. local network neighborhoods
   - 聚合策略
   - 计算图
2. 叠层
   - 模型、参数、训练
   - 如何学习？
   - 无监督和有监督学习举例



**Setup**：图 $G$ ，节点集 $V$ ，邻接矩阵 $A$ （二元，无向无权图。这些内容都可以泛化到其他情况下），节点特征矩阵 $X\in \R^{m\times |V|}$ ，一个节点 $v$ ，$v$ 的邻居集合 $N(v) $ 。如果数据集中没有节点特征，可以用指示向量indicator vectors（节点的独热编码），或者所有元素为常数1的向量。有时也会用节点度数来作为特征。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174124278.png" alt="image-20221222174124278" style="zoom:80%;" />



我们可能很直接地想到，将邻接矩阵和特征合并在一起应用在深度神经网络上（如图，直接一个节点的邻接矩阵+特征合起来作为一个观测）。这种方法的问题在于：

- 需要 $O(|V|)$ 的参数。
- 不适用于不同大小的图。
- 对节点顺序敏感（我们需要一个即使改变了节点顺序，结果也不会变的模型）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174532677.png" alt="image-20221222174532677" style="zoom:80%;" />



**将网格上的卷积神经网络泛化到图上，并应用到节点特征数据**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174605568.png" alt="image-20221222174605568" style="zoom:80%;" />

**图上无法定义固定的locality或滑动窗口，而且图是permutation invariant的（节点顺序不固定）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174631342.png" alt="image-20221222174631342" style="zoom:80%;" />

**从image到graph：聚合邻居信息**

过程：转换邻居信息 $W_ih_i$ ，将其加总 $\sum_iW_ih_i$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174727932.png" alt="image-20221222174727932" style="zoom:80%;" />

**图卷积神经网络（Graph Convolutional Networks）**：通过节点邻居定义其计算图，传播并转换信息，计算出节点表示（可以说是用邻居信息来表示一个节点）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174818838.png" alt="image-20221222174818838" style="zoom:80%;" />

**核心思想：通过聚合邻居来生成节点嵌入**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174854018.png" alt="image-20221222174854018" style="zoom:80%;" />

**直觉：通过神经网络聚合邻居信息**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222174928931.png" alt="image-20221222174928931" style="zoom:80%;" />



**直觉：通过节点邻居定义计算图（它的邻居是子节点，子节点的邻居又是子节点们的子节点……）**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222175024028.png" alt="image-20221222175024028" style="zoom:80%;" />



**深度模型就是有很多层**。

节点在每一层都有不同的表示向量，每一层节点嵌入是邻居上一层节点嵌入再加上它自己（相当于添加了自环）的聚合。

第0层是节点特征，第k层是节点通过聚合k hop邻居所形成的表示向量。

在这里就没有收敛的概念了，直接选择跑有限步（k）层。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222175131090.png" alt="image-20221222175131090" style="zoom:80%;" />



**邻居信息聚合（neighborhood aggregation）**：不同聚合方法的区别就在于如何跨层聚合邻居节点信息。neighborhood aggregation方法必须要order invariant或者说permutation invariant。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222180914907.png" alt="image-20221222180914907" style="zoom:80%;" />

**基础方法：从邻居获取信息求平均，再应用神经网络**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222181003476.png" alt="image-20221222181003476" style="zoom:80%;" />

**这种deep encoder的数学公式**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222181045810.png" alt="image-20221222181045810" style="zoom:80%;" />



**如何训练模型：需要定义节点嵌入上的损失函数**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222181222611.png" alt="image-20221222181222611" style="zoom:80%;" />

$h_v^l$ 是 $l$ 层 $v$ 的隐藏表示向量。

模型上可以学习的参数有 $W_l$ （neighborhood aggregation的权重）和 $B_l$ （转换节点自身隐藏向量的权重），注意，每层参数在不同节点之间是共享的。

可以通过将输出的节点表示向量输入损失函数中，运行SGD来训练参数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222181504948.png" alt="image-20221222181504948" style="zoom:80%;" />



**矩阵形式**：很多种聚合方式都可以表示为（稀疏）矩阵操作的形式，如这个基础方法可以表示成图中这种形式

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222181625945.png" alt="image-20221222181625945" style="zoom:80%;" />

补充：向量点积/矩阵乘法就是逐元素相乘然后累加，对邻接矩阵来说相当于对存在边的元素累加

对整个公式的矩阵化也可以实现：

这样就可以应用有效的稀疏矩阵操作。

同时，也要注意，当aggregation函数过度复杂时，GNN可能无法被表示成矩阵形式。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222183040537.png" alt="image-20221222183040537" style="zoom:80%;" />



**如何训练GNN**：节点嵌入 $\mathbf{z}_v$ 。

- **监督学习**：优化目标 $\min_{\Theta}L(\bold{y},f(\bold{z}_v))$

  如回归问题可以用L2 loss，分类问题可以用交叉熵。

  比如二分类交叉熵：$L=-\sum_{v\in V}(y_vlog(\sigma(z_v^T\theta))+(1-y_v)log(1-\sigma(z_v^T\theta)))$ ，其中 $z_v$ 是encoder输出的节点嵌入向量，$\theta$ 是classification weight。这个式子中，前后两个加数只有一个会被用到（y要么是1要么是0）

- **无监督学习**：用图结构作为学习目标

  比如节点相似性（随机游走、矩阵分解、图中节点相似性……等）

  损失函数为 $L=\sum_{z_u,z_v}CE(y_{u,v},DEC(z_u,z_v))$ ，如果 $u$ 和 $v$ 相似，则 $y_{u,v} =1$ ，$CE$ 是交叉熵，$DEC$ 是节点嵌入的 decoder 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184621318.png" alt="image-20221222184621318" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184635473.png" alt="image-20221222184635473" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184650877.png" alt="image-20221222184650877" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184718779.png" alt="image-20221222184718779" style="zoom:80%;" />



**模型设计：overview**

1. 定义邻居聚合函数
2. 定义节点嵌入上的损失函数
3. 在节点集合（如计算图的batch）上做训练
4. 训练后的模型可以应用在训练过与没有训练过的节点上

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184755694.png" alt="image-20221222184755694" style="zoom:80%;" /><img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184843152.png" alt="image-20221222184843152" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222184902677.png" alt="image-20221222184902677" style="zoom:80%;" />



**归约能力（inductive capability）**：因为聚合邻居的参数在所有节点之间共享，所以训练好的模型可以应用在没见过的节点/图上。比如动态图就有新增节点的情况。模型参数数量是亚线性sublinear于 $|V|$ 的（仅取决于嵌入维度和特征维度）（矩阵尺寸就是下一层嵌入维度×上一层嵌入维度，第0层嵌入维度就是特征维度）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222185153611.png" alt="image-20221222185153611" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222185220611.png" alt="image-20221222185220611" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222185236488.png" alt="image-20221222185236488" style="zoom:80%;" />



**结语**：通过聚合邻居信息产生节点嵌入，本节阐述了这一总思想下的一个基本变体。具体GNN方法的区别在于信息如何跨层聚合。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222185411347.png" alt="image-20221222185411347" style="zoom:80%;" />



## 7.4 图卷积神经网络与GraphSAGE

**GraphSAGE**：这个聚合函数可以是任何将一组向量（节点邻居的信息）映射到一个向量上的可微函数
$$
h_v^{(l+1)}=\sigma([W_l\cdot AGG(\{h_u^{(l)},\forall u\in N(v)\}),B_lh_v^{(l)}])
$$
**aggregation变体**：

- Mean：
  $$
  AGG=\sum_{u\in N(v)}\frac{h_u^{(l)}}{|N(v)|}
  $$

- Pool：
  $$
  AGG=\gamma(\{MLP(h_u^{(l)},\forall u\in N(u))\})
  $$
  对邻居信息向量做转换，再应用对称向量函数。

- LSTM：
  $$
  AGG=LSTM([h_u^{(l)},\forall u\in \pi(N(v))])
  $$
  在reshuffle的邻居上应用LSTM。

在每一层的节点嵌入上都可以做 $L2$ 归一化 $h_v^k\leftarrow\frac{h_v^k}{||h_v^k||_2}$ ，有时可以提升模型效果。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222190745231.png" alt="image-20221222190745231" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222190805939.png" alt="image-20221222190805939" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222190830859.png" alt="image-20221222190830859" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222190901794.png" alt="image-20221222190901794" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222190918944.png" alt="image-20221222190918944" style="zoom:80%;" />



**GCN 与 GraphSAGE**：核心思想都是基于local neighborhoods产生节点嵌入，用神经网络聚合邻居信息。

- GCN：邻居信息求平均，叠网络层。
- GraphSAGE：泛化neighborhood aggregation所采用的函数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222191018885.png" alt="image-20221222191018885" style="zoom:80%;" />



## 7.5 总结

**本节课中介绍了**：

1. 神经网络基础：损失函数loss，优化optimization，梯度gradient，随机梯度下降SGD，非线性non-linearity，多层感知器MLP
2. 图深度学习思想：
   - 多层嵌入转换
   - 每一层都用上一层的嵌入作为输入
   - 聚合邻居和本身节点
3. GCN Graph Convolutional Network：用求平均的方式做聚合，可以用矩阵形式来表示。
4. GraphSAGE：更有弹性的聚合函数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221222191138990.png" alt="image-20221222191138990" style="zoom:80%;" />



# 8. Colab2









































# 9. 图神经网络2：设计空间

**本章主要内容**：

本章主要介绍了GNN的**设计空间design space**，也就是设计一个GNN模型中的各种选择条件。



本章首先讲了GNN单层的设计选择。

一层GNN包含信息转换和信息聚合两个部分。

讲了三种典型实例GCN、GraphSAGE、GAT。

- GCN相当于用权重矩阵和节点度数归一化实现信息转换，用邻居节点求平均的方式实现聚合。
- GraphSAGE可选用多种聚合方式来聚合邻居信息，然后聚合邻居信息和节点本身信息。在GraphSAGE中可以选用L2正则化。
- GAT使用注意力机制为邻居对节点的信息影响程度加权，用线性转换后的邻居信息加权求和来实现节点嵌入的计算。注意力权重用一个可训练的模型attention mechanism计算出两节点间的attention coefficient，归一化得到权重值。此外可以应用多头机制增加鲁棒性。

在GNN层中还可以添加传统神经网络模块，如Batch Normalization、Dropout、非线性函数（激活函数）等。



然后讲了GNN多层堆叠方式。

叠太多层会导致过平滑问题。感受野可以用来解释这一问题。

对于浅GNN，可以通过增加单层GNN表现力、增加非GNN层来增加整体模型的表现力。

可以应用skip connections实现深GNN。skip connections可以让隐节点嵌入只跳一层，也可以全部跳到最后一层。



## 9.1 图形神经网络的一般观点

对design space的理解大概就是在设计模型的过程中，可以选择的各种实现方式所组成的空间。比如说可以选择怎么卷积，怎么聚合，怎么将每一层网络叠起来，用什么激活函数、用什么损失函数……用这些选项组合出模型实例，构成的空间就是design space。



**通用GNN框架**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223193115310.png" alt="image-20221223193115310" style="zoom:80%;" />

对GNN的一个网络层：要经历message（**信息转换**）和aggregation（**信息聚合**）两个环节，不同的实例应用不同的设计方式（如GCN，GraphSAGE，GAT……）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223192745267.png" alt="image-20221223192745267" style="zoom:80%;" />

**连接GNN网络层**：可以逐层有序堆叠，也可以添加skip connections

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223192836961.png" alt="image-20221223192836961" style="zoom:80%;" />

**图增强graph sugmentation**：使原始输入图和应用在GNN中的计算图不完全相同（即对原始输入进行一定处理后，再得到GNN中应用的计算图）。

图增强分为：图特征增强 / 图结构增强

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223192959353.png" alt="image-20221223192959353" style="zoom:80%;" />

**学习目标**：有监督/无监督目标，节点/边/图级别目标

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223193045223.png" alt="image-20221223193045223" style="zoom:80%;" />



## 9.2 GNN的单独层网络

GNN 单层网络的设计空间：message transformation + message aggregation

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223193330618.png" alt="image-20221223193330618" style="zoom:80%;" />



GNN单层网络的**目标**是将一系列向量（上一层的自身和邻居的message）压缩到一个向量中（新的节点嵌入）

完成这个目标分成两步：**信息处理**，**信息聚合**（这里的聚合方法是需要ordering invariant的，也就是邻居节点信息聚合，聚合的顺序应当和结果无关）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223193438493.png" alt="image-20221223193438493" style="zoom:80%;" />



**信息处理（message computation）**：
$$
\bold{m}_u^{(l)}=MSG^{(l)}(\bold{h}_u^{(l-1)})
$$
直觉：用每个节点产生一个信息，传播到其他节点上

示例：线性层 $\bold{m}_u^{(l)}=\bold{W}^{(l)}\bold{h}_u^{(l-1)}$ ，$\bold{W}^{(l)}$ 为权重矩阵

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223193737622.png" alt="image-20221223193737622" style="zoom:80%;" />



**信息聚合（message aggregation）**：
$$
\bold{h}_v^{(l)}=AGG^{(l)}(\{\bold{m}_u^{(l)},u\in N(v)\})
$$
直觉：对每个节点，聚合其邻居的节点信息

举例：求和，求平均，求极大值

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223194019871.png" alt="image-20221223194019871" style="zoom:80%;" />

这种message aggregation会导致节点自身的信息丢失，因为对 $\bold{h}_v^{(l)}$ 的计算不直接依赖于 $\bold{h}_v^{(l-1)}$ 。

**对此问题的解决方式**：在计算 $\bold{h}_v^{(l)}$ 时包含 $\bold{h}_v^{(l-1)}$ 。

- message computation：对节点本身及其邻居应用不同的权重矩阵
  $$
  \bold{m}_u^{(l)}=\bold{W}^{(l)}\bold{h}_u^{(l-1)}\\
  \bold{m}_v^{(l)}=\bold{B}^{(l)}\bold{h}_v^{(l-1)}
  $$

- message aggregation：聚合邻居信息，再将邻居信息与节点自身信息进行聚合（用concatenation或加总）
  $$
  \bold{h}_v^{(l)}=CONCAT(AGG^{(l)}(\{\bold{m}_u^{(l)},u\in N(v)\}),\bold{m}_v^{(l)})
  $$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223194636204.png" alt="image-20221223194636204" style="zoom:80%;" />



GNN单层网络就是合并上述两步：对每个节点，先计算出其自身与邻居的节点信息，然后计算其邻居与本身的信息聚合。

在这两步上都可以用非线性函数（激活函数）来增加其表现力：激活函数常写作 $\sigma(\cdot)$ ，如 $ReLU(\cdot)$ ，$Sigmoid(\cdot)$ ......

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223194840656.png" alt="image-20221223194840656" style="zoom:80%;" />



**经典GNN层（公式中橘色部分为信息聚合，红色部分为信息转换）**：

1. **图卷积网络GCN**：
   $$
   \bold{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}\bold{W}^{(l)}\frac{\bold{h}_u^{(l-1)}}{|N(v)|})
   $$
   信息转换：对上一层的节点嵌入用本层的权重矩阵进行转换，用节点度数进行归一化（在不同GCN论文中会应用不同的归一化方式）

   信息聚合：加总邻居信息，应用激活函数

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223195320311.png" alt="image-20221223195320311" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223195336034.png" alt="image-20221223195336034" style="zoom:80%;" />

2. **GraphSAGE**：
   $$
   \bold{h}_v^{(l)}=\sigma(\bold{W}^{(l)}\cdot CONCAT(\bold{h}_v^{(l-1)},AGG(\{\bold{h}_u^{(l-1)},\forall u\in N(v) \})))
   $$
   信息转换在 $AGG$ 过程中顺带着实现。

   信息聚合分为两步：

   - 第一步：聚合邻居节点信息，$\bold{h}_{N(v)}^{(l)}\leftarrow AGG(\{\bold{h}_u^{(l-1)},\forall u\in N(v) \})$ 
   - 第二步：将上一步信息与节点本身信息进行聚合，$\bold{h}_v^{(l)}\leftarrow\sigma(\bold{W}^{(l)}\cdot CONCAT(\bold{h}_v^{(l-1)},\bold{h}_{N(v)}^{l}))$ 

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223200220778.png" alt="image-20221223200220778" style="zoom:80%;" />

   **GraphSAGE聚合邻居的方式**：

   - Mean：邻居的加权平均值
     $$
     AGG=\sum_{u\in N(v)}\frac{\bold{h}_u^{(l-1)}}{|N(v)|}
     $$

   - Pool：对邻居向量做转换，再应用对称向量函数，如求和 $Mean(\cdot)$ 或求最大值 $Max(\cdot)$ 。
     $$
     AGG=Mean(\{MLP(\bold{h}_u^{(l-1)}),\forall u\in N(v)  \})
     $$

   - LSTM：在reshuffle的邻居上应用LSTM
     $$
     AGG=LSTM([\bold{h}_u^{(l-1)},\forall u \in \pi(N(v))])
     $$

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223200747845.png" alt="image-20221223200747845" style="zoom:80%;" />

   **在GraphSAGE每一层上都可以做 L2 归一化**：
   $$
   \bold{h}_v^{(l)}\leftarrow\frac{\bold{h}_v^{(l)}}{||\bold{h}_v^{l} ||_2}
   $$
   经过归一化后，所有向量都具有了相同的L2范式

   有时可以提升模型的节点嵌入效果

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223200932802.png" alt="image-20221223200932802" style="zoom:80%;" />

3. **GAT**：
   $$
   \bold{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}\alpha_{vu}\bold{W}^{(l)}\bold{h}_u^{(l-1)})
   $$
   $\alpha_{vu}$ 是注意力权重attention weights，衡量 $u$ 的信息对 $v$ 的重要性（越重要，权重越高，对 $v$ 的信息计算结果影响越大）。

   在GCN和GraphSAGE中 $\alpha_{vu}=\frac1{|N(v)|}$  ，直接基于图结构信息（节点度数）显式定义注意力权重，相当于认为节点的所有邻居都同样重要（注意力权重一样大）。

   在GAT中注意力权重的计算方法是：通过attention mechanism $a$ （ 一个可训练的模型）用两个节点上一层的节点嵌入计算其attention coefficient $e_{vu}$ ，用 $e_{vu}$ 计算 $\alpha_{vu}$ 。

   **注意力机制（attention mechanism）**：
   $$
   e_{vu}=a(\bold{W}^{(l)}\bold{h}_u^{(l-1)},\bold{W}^{(l)}\bold{h}_v^{(l-1)})
   $$
   这个 $a$ 随便选（可以是不对称的），比如用单层神经网络，则 $a$ 有可训练参数（线性层中的权重）：$e_{vu}=Linear(Concat(\bold{W}^{(l)}\bold{h}_u^{(l-1)},\bold{W}^{(l)}\bold{h}_v^{(l-1)}))$ 。

   **注意力系数（attention coefficient）**：$e_{vu}$ 表示了 $u$ 的信息对 $v$ 的重要性。

   将 attention coefficient $e_{vu}$ 归一化，得到最终的**注意力权重（attention weight）**：
   $$
   \alpha_{vu}=\frac{exp(e_{vu})}{\sum_{k\in N(v)}exp(e_{vk})}
   $$
   用softmax函数使 $\sum_{u\in N(v)}\alpha_{vu}=1$ ，也就是邻居对该节点的注意力权重和为1。

   GAT节点信息的计算方式就是基于 attention weight $\alpha_{vu}$ 加权求和：
   $$
   \bold{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}\alpha_{vu}\bold{W}^{(l)}\bold{h}_u^{(l-1)})
   $$
   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223204927018.png" alt="image-20221223204927018" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223204942472.png" alt="image-20221223204942472" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223204954415.png" alt="image-20221223204954415" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223205010357.png" alt="image-20221223205010357" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223205050667.png" alt="image-20221223205050667" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223205104042.png" alt="image-20221223205104042" style="zoom:80%;" />

   **GAT的多头注意力机制multi-head attention**：增加模型鲁棒性，使模型不卡死在奇怪的优化空间，在实践上平均表现更好。

   **用不同参数建立多个 attention 模型**：

   $\bold{h}_v^{(l)}[1]=\sigma(\sum_{u\in N(v)}\alpha_{vu}^1\bold{W}^{(l)}\bold{h}_u^{(l-1)})$

   $\bold{h}_v^{(l)}[2]=\sigma(\sum_{u\in N(v)}\alpha_{vu}^2\bold{W}^{(l)}\bold{h}_u^{(l-1)})$

   $\bold{h}_v^{(l)}[3]=\sigma(\sum_{u\in N(v)}\alpha_{vu}^3\bold{W}^{(l)}\bold{h}_u^{(l-1)})$

   **将输出进行聚合（通过concatenation或加总）**：$\bold{h}_v^{(l)}= AGG(\bold{h}_v^{(l)}[1],\bold{h}_v^{(l)}[2],\bold{h}_v^{(l)}[3])$ 

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223205523112.png" alt="image-20221223205523112" style="zoom:80%;" />

   **注意力机制的优点**：核心优点：隐式定义节点信息对邻居的importance value $\alpha_{vu}$

   - computationally efficient：对attentional coefficients的计算可以在图中所有边上同步运算，聚合过程可以在所有节点上同步运算
   - storage efficient：稀疏矩阵运算需要存储的元素数不超过 $O(V+E)$ ，参数数目固定（ $a$ 的可训练参数尺寸与图尺寸无关）
   - localized：仅对本地网络邻居赋予权重
   - inductive capability：边间共享机制，与全局图结构无关（我的理解就是算出 attention mechanism 的参数之后就完全可以正常对新节点运算，不需要再重新计算什么参数了）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223210854454.png" alt="image-20221223210854454" style="zoom:80%;" />



**GAT示例：Cora Citation Net**

attention机制可应用于多种GNN模型中。在很多案例中表现出了结果的提升。

如图显示，将节点嵌入降维到二维平面，节点不同颜色表示不同类，边的宽度表示归一化后的attention coefficient（8个attention head计算后求和）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211008640.png" alt="image-20221223211008640" style="zoom:80%;" />

t-SNE是一种降维技术。



实践应用中的GNN网络层：往往会应用传统神经网络模块，如在信息转换阶段应用Batch Normalization（使神经网络训练稳定）、Dropout（预防过拟合）、Attention / Gating（控制信息重要性）等。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211054939.png" alt="image-20221223211054939" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211124209.png" alt="image-20221223211124209" style="zoom:80%;" />

- **Batch Normalization（使神经网络训练稳定）**：对一个batch的输入数据（节点嵌入）进行归一化，使其平均值为0，方差为1。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211234527.png" alt="image-20221223211234527" style="zoom:80%;" />

- **Dropout（预防过拟合）**：在训练阶段，以概率p随机将神经元置为0；在测试阶段，用所有的神经元来进行计算。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211347273.png" alt="image-20221223211347273" style="zoom:80%;" />

  **GNN中的Dropout：应用在信息转换过程中的线性网络层上**

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211432761.png" alt="image-20221223211432761" style="zoom:80%;" />



**非线性函数 / 激活函数**：用于激活嵌入向量 $x$ 的第 $i$ 维

最常用 $ReLU=max(x_i,0)$ 。

Sigmoid $\sigma(x_i)=\frac1{1+e^{-x_i}}$ ，用于希望限制嵌入范围。

Parametric ReLU $PReLU(x_i)=max(x_i,0)+a_i min(x_i,0)$ ，$a_i$ 是可学习的参数，实证表现优于ReLU。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211818230.png" alt="image-20221223211818230" style="zoom:80%;" />



可以通过 [GraphGym](https://github.com/snap-stanford/GraphGym) 来测试不同的GNN设计实例。



## 9.3 GNN的堆叠层

**连接GNN网络层部分**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223211933447.png" alt="image-20221223211933447" style="zoom:80%;" />



连接GNN网络层的标准方式：**按序堆叠**

输入原始节点特征，输出 $L$ 层后计算得到的节点嵌入向量。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212142464.png" alt="image-20221223212142464" style="zoom:80%;" />



**过平滑问题the over-smoothing problem**：如果GNN层数太多，不同节点的嵌入向量会收敛到同一个值（如果我们想用节点嵌入做节点分类任务，这就凉了）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212213531.png" alt="image-20221223212213531" style="zoom:80%;" />

GNN的层跟别的神经网络的层不一样，GNN的层数说明的是它聚集多少跳邻居的信息。



**GNN的感受野（receptive field）**：决定该节点嵌入的节点组成的集合。

对K层GNN，每个节点都有一个K跳邻居的感受野。如图可见K越大，感受野越大。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212354087.png" alt="image-20221223212354087" style="zoom:80%;" />

对两个节点来说，K变大，感受野重合部分会迅速变大。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212459218.png" alt="image-20221223212459218" style="zoom:80%;" />

节点嵌入受其感受野决定，两个节点间的感受野越重合，其嵌入就越相似。

堆叠很多GNN网络层→节点具有高度重合的感受野→节点嵌入高度相似→过平滑问题

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212519707.png" alt="image-20221223212519707" style="zoom:80%;" />



由于过平滑问题，我们需要谨慎考虑增加GNN层。

- 第一步：分析解决问题所需的必要感受野（如测量图的直径）
- 第二步：设置GNN层数 L 略大于我们想要的感受野

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212600701.png" alt="image-20221223212600701" style="zoom:80%;" />

**既然GNN层数不能太多，那么我们如何使一个浅的GNN网络更具有表现力呢**？

- 方法1：增加单层GNN的表现力，如将信息转换/信息聚合过程从一个简单的线性网络变成深度神经网络（如3层MLP）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212638717.png" alt="image-20221223212638717" style="zoom:80%;" />

- 方法2：添加不是用来传递信息的网络层，也就是非GNN层，如对每个节点应用MLP（在GNN层之前或之后均可，分别叫 pre-process layers 和 post-process layers）

  - pre-processing layers：如果节点特征必须经过编码就很重要（如节点表示图像/文字时）
  - post-processing layers：如果在节点嵌入的基础上需要进行推理和转换就很重要（如图分类、知识图谱等任务中）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212728136.png" alt="image-20221223212728136" style="zoom:80%;" />



如果实际任务还是需要很多层GNN网络，那么可以在GNN模型中增加**skip connections**。

通过对过平滑问题进行观察，我们可以发现，靠前的GNN层可能能更好地区分节点。

因此我们可以在最终节点嵌入中增加靠前GNN层的影响力，实现方法是在GNN中直接添加捷径，保存上一层节点的嵌入向量（看后文应该是指在激活函数前，在聚合后的结果的基础上再加上前一层的嵌入向量）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223212855282.png" alt="image-20221223212855282" style="zoom:80%;" />



**skip connections原理**：相当于制造了多个模型（如图所示），$N$ 个skip connections就相当于创造了 $2^N$ 条路径，每一条路径最多有 $N$ 个模块。

这些路径都会对最终的节点嵌入产生影响，相当于自动获得了一个浅GNN和深GNN的融合模型。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223214323586.png" alt="image-20221223214323586" style="zoom:80%;" />



**skip connections示例：在GCN中的应用**

**标准GCN层**：
$$
\bold{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}\bold{W}^{(l)}\frac{\bold{h}_u^{(l-1)}}{|N(v)|})
$$
**带skip connections的GCN层**：
$$
\bold{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}\bold{W}^{(l)}\frac{\bold{h}_u^{(l-1)}}{|N(v)|}+\bold{h}_v^{(l-1)})
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223214927138.png" alt="image-20221223214927138" style="zoom:80%;" />



skip connections也可以跨多层，直接跨到最后一层，在最后一层聚合之前各层的嵌入（通过concat / pooling / LSTM）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221223215015166.png" alt="image-20221223215015166" style="zoom:80%;" />





# 10. 图神经网络的应用

**本章主要内容**：

本章继续上一章内容，讲design space剩下的两部分：图增强，如何训练一个GNN模型（GNN训练全流程）。



在图增强方面：首先介绍图增强的原因和分类。然后分别介绍：

- graph feature augmentation的方法：使用常数特征、独热编码、图结构信息
- graph structure augmentation的方法：对稀疏图，增加虚拟边或虚拟节点；对稠密图，节点邻居抽样



接下来讲GNN模型训练的学习目标。

首先介绍不同粒度任务下的prediction head（将节点嵌入转换为最终预测向量）：节点级别的任务可以直接进行线性转换。链接级别的任务可以将节点对的嵌入进行concatenation或点积后进行线性转换。图级别的任务是将图中所有节点嵌入作池化操作，可以通过hierarchical global pooling方法来进行优化（实际应用：DiffPool）。


接下来介绍了预测值和标签的问题：有监督/无监督学习情况下的标签来源。

然后介绍损失函数：分类常用交叉熵，回归任务常用MSE（L2 loss）。

接下来介绍评估指标：回归任务常用RMSE和MAE，分类任务常用accuracy和ROC AUC。

最后讲了设置GNN预测任务（将图数据拆分为训练/验证/测试集）的方法，分为transductive和inductive两种。



## 10.1 GNN 的图增强方法

回顾一遍在 Lecture 7第一节中讲过的GNN图增强部分：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224231234679.png" alt="image-20221224231234679" style="zoom:80%;" />



**为什么要进行图增强？**

我们在之前的学习过程中都假设原始数据和应用于GNN的计算图一致，但很多情况下原始数据可能不适宜于GNN：

- 特征层面：输入图可能缺少特征（也可能是特征很难编码）→特征增强
- 结构层面：
  - 图可能过度稀疏→导致message passing效率低（边不够嘛）
  - 图可能过度稠密→导致message passing代价太高（每次做message passing都需要对好几个节点做运算）
  - 图可能太大→GPU装不下
- 事实上输入图很难恰好是适宜于GNN（图数据嵌入）的最优计算图

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224231429835.png" alt="image-20221224231429835" style="zoom:80%;" />



**图增强方法**：

- 图特征：输入图**缺少特征**→特征增强
- 图结构：
  - 图**过于稀疏**→增加虚拟节点/边
  - 图**过于稠密**→在message passing时抽样邻居
  - 图**太大**→在计算嵌入时抽样子图（在后续课程中会专门介绍如何将GNN方法泛化到大型数据上scale up）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224231536712.png" alt="image-20221224231536712" style="zoom:80%;" />

### 10.1.1 图特征增强Feature Augmentation

**应对图上缺少特征的问题（比如只有邻接矩阵），标准方法**：

- constant：给每个节点赋常数特征

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224231628336.png" alt="image-20221224231628336" style="zoom:80%;" />

- one-hot：给每个节点赋唯一ID，将ID转换为独热编码向量的形式（即ID对应索引的元素为1，其他元素都为0）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224231652003.png" alt="image-20221224231652003" style="zoom:80%;" />



**两种方法的比较**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224231804094.png" alt="image-20221224231804094" style="zoom:80%;" />



**用于应对GNN很难学到特定图结构的问题（如果不用特征专门加以区分，GNN就学不到这些特征）**：

举例：节点所处环上节点数cycle count这一属性

问题：因为度数相同（都是2），所以无论环上有多少个节点，GNN都会得到相同的计算图（二叉树），无法分别。

解决方法：加上cycle count这一特征（独热编码向量，节点数对应索引的元素为1，其他元素为0）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232003943.png" alt="image-20221224232003943" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232013412.png" alt="image-20221224232013412" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232025133.png" alt="image-20221224232025133" style="zoom:80%;" />

其他常用于数据增强的特征：clustering coefficient，centrality（及任何 Lecture 2 中讲过的特征），PageRank等。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232057991.png" alt="image-20221224232057991" style="zoom:80%;" />

### 10.1.2 图结构增强Structure Augmentation

**对稀疏图：增加虚拟边virtual nodes或虚拟节点virtual edges**

- 虚拟边：在2-hop邻居之间增加虚拟边

  直觉：在GNN计算时不用邻接矩阵 $A$ ，而用 $A+A^2$ 

  适用范例：bipartite graphs

  如作者-论文组成的bipartite graph，增加虚拟边可以在合作作者或者同作者论文之间增加链接。

  这样GNN可以浅一些，训练也会更快一些（因为在同类节点之间可以直接交互了），但如果添的边太多了也会增加复杂性。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232310250.png" alt="image-20221224232310250" style="zoom:80%;" />

- 虚拟节点：增加一个虚拟节点，这个虚拟节点与图（或者一个从图中选出的子图）上的所有节点相连

  这会导致所有节点最长距离变成2（节点A-虚拟节点-节点B）

  优点：稀疏图上message passing大幅提升

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232405394.png" alt="image-20221224232405394" style="zoom:80%;" />



**对稠密图：节点邻居抽样node neighborhood sampling**

在message passing的过程中，不使用一个节点的全部邻居，而改为抽样一部分邻居。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232447070.png" alt="image-20221224232447070" style="zoom:80%;" />

举例来说，对每一层，在传播信息时随机选2个邻居，计算图就会从上图变成下图：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232510947.png" alt="image-20221224232510947" style="zoom:80%;" />

优点：计算图变小

缺点：可能会损失重要信息



**可以每次抽样不同的邻居，以增加模型鲁棒性**：

我们希望经抽样后，结果跟应用所有邻居的结果类似，但还能高效减少计算代价（在后续课程中会专门介绍如何将GNN方法泛化到大型数据上scale up）。

实践证明效果很好。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232654873.png" alt="image-20221224232654873" style="zoom:80%;" />



## 10.2 学习目标

**回顾一遍在 Lecture 7第一节中讲过的学习目标部分：我们如何训练一个GNN模型**？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232802005.png" alt="image-20221224232802005" style="zoom:80%;" />



**GNN训练pipeline**：

输入数据→用GNN训练数据→得到节点嵌入→prediction head（在不同粒度的任务下，将节点嵌入转换为最终需要的预测向量）→得到预测向量和标签→选取损失函数→选取评估指标

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224232851861.png" alt="image-20221224232851861" style="zoom:80%;" />



### 10.2.1 预测头

不同粒度下的prediction head：节点级别，边级别，图级别

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224233007940.png" alt="image-20221224233007940" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224233026307.png" alt="image-20221224233026307" style="zoom:80%;" />



**节点级别：直接用节点嵌入做预测**

GNN得到的节点嵌入 $\bold{h}_v^{(L)}$ d维。

预测目标向量 k 维。

- 分类任务：在k个类别之间做分类
- 回归任务：在k个目标target / characteristic 上做回归

$$
\hat{\bold{y}}_v=Head_{node}(\bold{h}_v^{(L)})=\bold{W}^{(H)}\bold{h}_v^{(l)}
$$

其中 $\bold{W}^{(H)}\in \R^{k\times d}$ ，将d维嵌入映射到k维输出。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224233408718.png" alt="image-20221224233408718" style="zoom:80%;" />



**边级别：用节点嵌入对来做预测**
$$
\hat{\bold{y}}_{uv}=Head_{edge}(\bold{h}_u^{(L)},\bold{h}_v^{(L)})
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224233609522.png" alt="image-20221224233609522" style="zoom:80%;" />

$Head_{edge}(\bold{h}_u^{(L)},\bold{h}_v^{(L)})$ **的可选方法**：

- **concatenation+linear**：这种方法在讲GAT的时候介绍过，注意力机制 $a$ 可以用这种方法将节点对信息转换为注意力系数 $e$ 。
  $$
  \hat{\bold{y}}_{uv}=Linear(Concat(\bold{h}_u^{(L)},\bold{h}_v^{(L)}))
  $$
  $Linear(\cdot)$ 将 $2d$ 维嵌入映射到 $k$ 维输出。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224233859824.png" alt="image-20221224233859824" style="zoom:80%;" />

- **点积**：
  $$
  \hat{\bold{y}}_{uv}=(\bold{h}_u^{(L)})^T\bold{h}_v^{(L)}
  $$
  这种方法只能应用于1-way prediction（因为点积输出结果就一维嘛），例如链接预测任务（预测边是否存在）。

  应用到k-way prediction上：跟GAT中的多头注意力机制类似，多算几组然后合并（公式中的 $\bold{W}^{(1)},...,\bold{W}^{(k)}$ 是可学习的参数）：
  $$
  \hat{\bold{y}}_{uv}^{(1)}=(\bold{h}_u^{(L)})^T\bold{W}^{(1)}\bold{h}_v^{(L)}\\
  \hat{\bold{y}}_{uv}^{(2)}=(\bold{h}_u^{(L)})^T\bold{W}^{(2)}\bold{h}_v^{(L)}\\
  ...\\
  \hat{\bold{y}}_{uv}^{(k)}=(\bold{h}_u^{(L)})^T\bold{W}^{(k)}\bold{h}_v^{(L)}
  $$

  $$
  \hat{\bold{y}}_{uv}=Concat(\hat{\bold{y}}_{uv}^{(1)},\hat{\bold{y}}_{uv}^{(2)},...,\hat{\bold{y}}_{uv}^{(k)})\in\R^k
  $$

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224234435283.png" alt="image-20221224234435283" style="zoom:80%;" />



**图级别：用图中所有节点的嵌入向量来做预测**
$$
\hat{\bold{y}}_G=Head_{graph}(\{\bold{h}_v^{(L)}\in\R^d,\forall v\in G  \})
$$
$Head_{graph}(\cdot)$ 与GNN单层中的 $AGG(\cdot)$ 类似，都是将若干嵌入聚合为一个嵌入。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224235322312.png" alt="image-20221224235322312" style="zoom:80%;" />

$Head_{graph}(\bold{h}_v^{(L)}\in\R^d,\forall v\in G)$ **的可选方法**：

- global mean pooling：$\hat{\bold{y}}_G=Mean(\{\bold{h}_v^{(L)}\in\R^d,\forall v\in G  \})$ 
- global max pooling：$\hat{\bold{y}}_G=Max(\{\bold{h}_v^{(L)}\in\R^d,\forall v\in G  \})$
- global sum pooling：$\hat{\bold{y}}_G=Sum(\{\bold{h}_v^{(L)}\in\R^d,\forall v\in G  \})$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221224235746361.png" alt="image-20221224235746361" style="zoom:80%;" />

如果想比较不同大小的图，mean方法可能比较好（因为结果不受节点数量的影响）；如果关心图的大小等信息，sum方法可能比较好。

这些方法都在小图上表现很好，但是在大图上的global pooling方法可能会面临丢失信息的问题。

**举例：使用一维节点嵌入**。

$G_1$ 的节点嵌入为 $\{-1,-2,0,1,2\}$ ，$G_2$ 的的节点嵌入为 $\{-10,-20,0,10,20\}$ ，显然两个图的节点嵌入差别很大，图结构很不相同。

但是经过global sum pooling后结果均为 0 。

就这两个图的表示向量一样了，无法做出区分，这是不行的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225000008403.png" alt="image-20221225000008403" style="zoom:80%;" />

为了解决这一问题，解决方法是**hierarchical global pooling**：**分层聚合节点嵌入**。

举例：使用 $ReLU(Sum(\cdot))$ 做聚合，先分别聚合前两个节点和后三个节点的嵌入，然后再聚合这两个嵌入。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225000504169.png" alt="image-20221225000504169" style="zoom:80%;" />

这样我们就可以将 $G_1$ 和 $G_2$ 作出区分了。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225000731426.png" alt="image-20221225000731426" style="zoom:80%;" />

一个hierarchical pooling的实际应用：$DiffPool$ 

大致来说，就是每一次先用一个GNN计算节点嵌入，然后用另一个GNN（这两个GNN可以同步运算）（两个GNN联合训练jointly train）计算节点属于哪一类，然后按照每一类对图进行池化。每一类得到一个表示向量，保留类间的链接，产生一个新的图。重复这一过程，直至得到最终的表示向量。

将图池化问题与社区发现问题相结合，用节点嵌入识别社区→聚合社区内的节点得到。

community embeddings→用community embeddings识别supercommunity→聚合supercommunity内的节点得到supercommunity embeddings……

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225001517008.png" alt="image-20221225001517008" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225001528115.png" alt="image-20221225001528115" style="zoom:80%;" />

### 10.2.2 预测&标签

**有监督问题的标签 & 无监督问题的信号**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225001619300.png" alt="image-20221225001619300" style="zoom:80%;" />

有监督学习supervise learning：直接给出标签（如一个分子图是药的概率）

无监督学习unsupervised learning / self-supervised learning：使用图自身的信号（如链接预测：预测两节点间是否有边）

有时这两种情况下的分别比较模糊，在无监督学习任务中也可能有“有监督任务”，如训练GNN以预测节点clustering coefficient

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225001710948.png" alt="image-20221225001710948" style="zoom:80%;" />



**有监督学习的标签：按照实际情况而来**

举例：

- 节点级别——引用网络中，节点（论文）属于哪一学科
- 边级别——交易网络中，边（交易）是否有欺诈行为
- 图级别——图（分子）是药的概率

建议将无监督学习任务规约到三种粒度下的标签预测任务，因为这种预测任务有很多已做过的工作可资参考，会好做些。

例如聚类任务可视为节点属于某一类的预测任务。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225001811438.png" alt="image-20221225001811438" style="zoom:80%;" />



**无监督学习的信号**：

在没有外部标签时，可以使用图自身的信号来作为有监督学习的标签。举例来说，GNN可以预测：

- 节点级别：节点统计量（如clustering coefficient，PageRank等）
- 边级别：链接预测（隐藏两节点间的边，预测此处是否存在链接）
- 图级别：图统计量（如预测两个图是否同构）

这些都是不需要外部标签的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225002501015.png" alt="image-20221225002501015" style="zoom:80%;" />



### 10.2.3 损失函数

**分类任务常用交叉熵，回归任务常用MSE**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225002545864.png" alt="image-20221225002545864" style="zoom:80%;" />



用 $\hat{\bold{y}}^{(i)}$ 和 $\bold{y}^{(i)}$ 来统一指代各级别的预测值和标签（ $i$ 是观测编号）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225002716886.png" alt="image-20221225002716886" style="zoom:80%;" />



分类任务的标签 $\bold{y}^{(i)}$ 是离散数值，如节点分类任务的标签是节点属于哪一类。

回归任务的标签 $\bold{y}^{(i)}$ 是连续数值，如预测分子图是药的概率。

两种任务都能用GNN。其区别主要在于损失函数和评估指标。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225002828842.png" alt="image-20221225002828842" style="zoom:80%;" />



**分类任务的损失函数交叉熵**：
$$
CE(\bold{y}^{(i)},\hat{\bold{y}}^{(i)})=-\sum_{j=1}^{K}(\bold{y}_j^{(i)}log\hat{\bold{y}}_j^{(i)})\\
Loss=\sum_{i=1}^NCE(\bold{y}^{(i)},\hat{\bold{y}}^{(i)})
$$
其中 $i$ 是观测序号，$j$ 是类别对应的维度索引。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225003228430.png" alt="image-20221225003228430" style="zoom:80%;" />



**回归任务的损失函数MSE / L2 loss**：
$$
MSE(\bold{y}^{(i)},\hat{\bold{y}}^{(i)})=\sum_{j=1}^K(\bold{y}^{(i)}-\hat{\bold{y}}^{(i)})^2\\
Loss=\sum_{i=1}^N MSE(\bold{y}^{(i)},\hat{\bold{y}}^{(i)})
$$
其中 $i$ 是观测序号，$j$ 是类别对应的维度索引。

MSE的优点：连续、易于微分……等。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225003503734.png" alt="image-20221225003503734" style="zoom:80%;" />



此外还有其他损失函数，如maximum margin loss，适用于我们关心节点顺序、不关心具体数值而关心其排行的情况。

### 10.2.4 评估指标

**evaluation metrics： Accuracy 和 ROC AUC**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225003647346.png" alt="image-20221225003647346" style="zoom:80%;" />



**回归任务**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225003744464.png" alt="image-20221225003744464" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225003803489.png" alt="image-20221225003803489" style="zoom:80%;" />



**分类任务**：

- **多分类任务**：
  $$
  accuracy=\frac{1[arg\ max(\hat{\bold{y}}^{(i)})=\bold{y}^{(i)}]}{N}
  $$

- **二分类任务**：对分类阈值敏感的评估指标。因为数据不平衡时可能会出现accuracy虚高的情况。比如99%的样本都是负样本，那么分类器只要预测所有样本为负就可以获得99%的accuracy，但这没有意义。所以需要其他评估指标来解决这一问题。

  **对分类阈值不敏感的评估指标：ROC AUC** 

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004054052.png" alt="image-20221225004054052" style="zoom:80%;" />

  accuracy（分类正确的观测占所有观测的比例）

  precision（预测为正的样本中真的为正（预测正确）的样本所占比例）

  recall（真的为正的样本中预测为正（预测正确）的样本所占比例）

  F1-Score（precision和recall的调和平均值，信息抽取、文本挖掘等领域常用）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004237270.png" alt="image-20221225004237270" style="zoom:80%;" />

  **ROC曲线：TPR（recall）和FPR之间的权衡（对角斜线说明是随机分类器）**

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004305467.png" alt="image-20221225004305467" style="zoom:80%;" />

  **ROC AUC** ：ROC曲线下面积。越高越好，0.5是随机分类器，1是完美分类器。

  **直觉**：随机抽取一个正样本和一个负样本，正样本被识别为正样本的概率比负样本被识别为正样本的概率高的概率。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004444155.png" alt="image-20221225004444155" style="zoom:80%;" />

### 10.2.5 切分数据集

**将数据集切分为训练集、验证集、测试集**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004522694.png" alt="image-20221225004522694" style="zoom:80%;" />



**fixed / random split**：

- fixed split：只切分一次数据集，此后一直使用这种切分方式
- random split：随机切分数据集，应用多次随机切分后计算结果的平均值



**我们希望三部分数据之间没有交叉，即留出法hold-out data**

但由于图结构的特殊性，如果直接像普通数据一样切分图数据集，我们可能不能保证测试集隔绝于训练集：就是说，测试集里面的数据可能与训练集里面的数据有边相连，在message passing的过程中就会互相影响，导致信息泄露。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004620814.png" alt="image-20221225004620814" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004640409.png" alt="image-20221225004640409" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004652097.png" alt="image-20221225004652097" style="zoom:80%;" />



**解决方式1： transductive setting**，输入全图在所有split中可见。仅切分（节点）标签。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004806487.png" alt="image-20221225004806487" style="zoom:80%;" />

**解决方式2： inductive setting**，去掉各split之间的链接，得到多个互相无关的图。这样不同split之间的节点就不会互相影响。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225004840645.png" alt="image-20221225004840645" style="zoom:80%;" />



**transductive setting / inductive setting**：

- **transductive setting**：
  1. 测试集、验证集、训练集在同一个图上，整个数据集由一张图构成
  2. 全图在所有split中可见。
  3. 仅适用于节点/边预测任务。
- **inductive setting**：
  1. 测试集、验证集、训练集分别在不同图上，整个数据集由多个图构成。
  2. 每个split只能看到split内的图。成功的模型应该可以泛化到没见过的图上。
  3. 适用于节点/边/图预测任务。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005010829.png" alt="image-20221225005010829" style="zoom:80%;" />



**示例：节点分类任务**

- transductive：各split可见全图结构，但只能观察到所属节点的标签
- inductive：切分多个图，如果没有多个图就将一个图切分成3部分、并去除各部分之间连接的边

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005107234.png" alt="image-20221225005107234" style="zoom:80%;" />



**示例：图预测任务**

只适用inductive setting，将不同的图划分到不同的split中。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005153178.png" alt="image-20221225005153178" style="zoom:80%;" />



**示例：链接预测任务**

任务目标：预测出缺失的边。

这是个 unsupervised / self-supervised 任务，需要自行建立标签、自主切分数据集。

需要隐藏一些边，然后让GNN预测边是否存在。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005226960.png" alt="image-20221225005226960" style="zoom:80%;" />

**在切分数据集时，我们需要切分两次**：

1. **第一步**：在原图中将边分为**message edges**（用于GNN message passing）和**supervision edges**（作为GNN的预测目标）。只留下message edges，不将supervision edges传入GNN。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005333630.png" alt="image-20221225005333630" style="zoom:80%;" />

2. **第二步**：切分数据集。

   - **方法1： inductive link prediction split** ，划分出3个不同的图组成的split，每个split里的边按照第一步分成message edges和supervision edges

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005618258.png" alt="image-20221225005618258" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225005628352.png" alt="image-20221225005628352" style="zoom:80%;" />

   - **方法2： transductive link prediction split** ，链接预测任务的默认设置方式。在一张图中进行切分：在训练时要留出验证集/测试集的边，而且注意边既是图结构又是标签，所以还要留出supervision edges。

     **具体来说**：

     - 训练：用 **training message edges** 预测 **training supervision edges**
     - 验证：用 training message edges 和 training supervision edges 预测 **validation edges**
     - 测试：用 training message edges 和 training supervision edges 和 validation edges 预测 **test edges**

     是个链接越来越多，图变得越来越稠密的过程。这是因为在训练过程之后，supervision edges就被GNN获知了，所以在验证时就要应用 supervision edges 来进行 message passing（测试过程逻辑类似）

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010152424.png" alt="image-20221225010152424" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010207619.png" alt="image-20221225010207619" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010221443.png" alt="image-20221225010221443" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010242105.png" alt="image-20221225010242105" style="zoom:80%;" />



<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010257914.png" alt="image-20221225010257914" style="zoom:80%;" />

### 10.2.6 GNN训练流水线

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010357573.png" alt="image-20221225010357573" style="zoom:80%;" />



## 10.3 GNN design space 总结

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225010419746.png" alt="image-20221225010419746" style="zoom:80%;" />





# 11 图神经网络的理论

**本章主要内容**：

本章主要学习GNN模型的表达能力expressive power，即将不同图数据表示为不同嵌入向量的能力。

我们主要考虑图中节点的局部邻居结构 **local neighborhood structure** 信息，GNN通过计算图 **computational graph** 捕获节点的局部邻居结构。

因此，GNN无法区分具有相同计算图的节点。

如果GNN能将具有不同计算图的节点区分开来（即形成一个**单射rejective**函数，不同计算图的节点被表示为不同的嵌入），则我们称该GNN具有强表达能力，即计算图中的信息得到了完全的保留。要实现这样的区分度，需要GNN的**聚合**函数是单射的。



已知聚合函数表达能力越强，GNN表达能力越强，单射聚合函数的GNN表达能力最强。我们设计具有最强表达能力的GNN模型：

邻居聚合过程可被抽象为一个输入为**multi-set**的函数。

均值池化（GCN）无法区分各类占比相同的multi-set，最大池化（GraphSAGE）无法区分具有相同的不同类的multi-set，因此都不单射，不够具有表达能力。

为了建立在信息传递message-passing框架下的最强GNN，我们需要设计multi-set上的单射邻居聚合函数。根据Xu et al. ICLR 2019的定理，我们可以设计一个含 $\Phi$ 和 $f$ 两个未知函数、并应用sum-pooling的函数来表示这样的单射函数，根据universal approximation theorem可知 $\Phi$ 和 $f$ 可以通过MLP拟合得到。从而建立 **Graph Isomorphism Network (GIN)** 模型。

GIN是WL graph kernel的神经网络版。GIN和WL graph kernel都可以区分大部分真实图。

在表达能力上，sum（multiset） > mean（distribution）> max（set）



## 11.1 图神经网络的表达能力

本节课主要探讨，在已经提出GCN、GAT、GraphSAGE、design space等众多GNN模型的前提下，各种模型的表示能力（区分不同图结构的能力）如何？我们如何设计这样一种表示能力最强的模型？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225155725110.png" alt="image-20221225155725110" style="zoom:80%;" />



**GNN模型实例**：

- GCN：mean-pool + Linear + ReLU non-linearity

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225155856103.png" alt="image-20221225155856103" style="zoom:80%;" />

- GraphSAGE（以最大池化为例）：MLP + max-pool

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225155919426.png" alt="image-20221225155919426" style="zoom:80%;" />



本课程中用节点颜色指代特征feature，同色即特征相同。

如图中举例图中所有节点的特征都相同（黄色）。

图上的信息分为结构信息和特征信息，因为特征相同，所以无法仅凭特征来区分节点了（如果特征全都不一样，只需要看特征向量就能将节点区分开了）。让所有特征相同可以更好看出GNN如何捕捉结构信息。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160008259.png" alt="image-20221225160008259" style="zoom:80%;" />

**局部结构信息**：我们感兴趣于量化节点的局部结构信息。

- 例子1：节点1和节点5，因其度数不同而具有不同的局部结构信息。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160121749.png" alt="image-20221225160121749" style="zoom:80%;" />

- 例子2：节点1和节点4，具有相同度数，但到其两跳邻居的信息上，可以区分两点：其邻居的度数不同。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160140800.png" alt="image-20221225160140800" style="zoom:80%;" />

- 例子3：节点1和节点2，具有相同的邻居结构，因为在图中是对称的。（不可区分）（无论多少跳邻居上，都具有相同的局部结构）（位置同构）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160237673.png" alt="image-20221225160237673" style="zoom:80%;" />



我们接下来就要分析GNN节点嵌入能否区分不同节点的局部邻居结构，在什么情况下会区分失败。

GNN通过计算图得到局部邻居结构。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160356580.png" alt="image-20221225160356580" style="zoom:80%;" />



**计算图**：

- GNN每一层聚合邻居信息（节点嵌入），即通过其邻居得到的计算图产生节点嵌入。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160427092.png" alt="image-20221225160427092" style="zoom:80%;" />

- 节点1和节点2的计算图：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160442745.png" alt="image-20221225160442745" style="zoom:80%;" />

- 上图两个计算图本质上是一样的：GNN只能识别特征，不能识别ID

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160500068.png" alt="image-20221225160500068" style="zoom:80%;" />

  因为计算图相同，这两个节点将被嵌入到同一个表示向量（即在表示域重叠，GNN无法区分这两个节点）。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160534656.png" alt="image-20221225160534656" style="zoom:80%;" />

- 一般来说，不同的局部邻居会得到不同的计算图：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225160556302.png" alt="image-20221225160556302" style="zoom:80%;" />

- 计算图和对应节点的有根子树结构rooted subtree structure相同，通过从根节点逐邻居展开计算图而得到。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225162938086.png" alt="image-20221225162938086" style="zoom:80%;" />

- GNN节点嵌入捕获这个rooted subtree structures，表示能力最强的GNN模型将不同的rooted subtree结构映射到不同的节点嵌入中（图中表示为不同颜色）：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163002377.png" alt="image-20221225163002377" style="zoom:80%;" />

- 单射injective函数：将不同自变量映射为不同的因变量，这样可以完整保留输入数据中的信息。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163100806.png" alt="image-20221225163100806" style="zoom:80%;" />

- 表示能力最强的GNN就应该单射地映射子树到节点嵌入（即不同的子树映射为不同的嵌入）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163125080.png" alt="image-20221225163125080" style="zoom:80%;" />

- 同深度的子树可以从叶节点到根节点迭代表示信息，来进行区分

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163153093.png" alt="image-20221225163153093" style="zoom:80%;" />

- 如果GNN每一步聚合都可以保留全部邻居信息，那么所产生的节点嵌入就可以区分不同的有根子树，也就达成了GNN具有最强表示能力的效果。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163250252.png" alt="image-20221225163250252" style="zoom:80%;" />

- 所以表示能力最强的GNN就是每一步都使用单射邻居聚合函数（保留全部信息），把不同的邻居映射到不同的嵌入上。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163316301.png" alt="image-20221225163316301" style="zoom:80%;" />



**总结**：为得到节点嵌入，GNN使用计算图（以节点为根的子树），如果每层都使用单射的聚合函数，就可以达成区分不同子树的效果。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163353056.png" alt="image-20221225163353056" style="zoom:80%;" />



## 11.2 设计最有力的图神经网络

GNN的表示能力取决于其应用的邻居聚合函数。聚合函数表达能力越强，GNN表达能力越强，单射聚合函数的GNN表达能力最强。

接下来本课程将理论分析各聚合函数的表示能力。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163448786.png" alt="image-20221225163448786" style="zoom:80%;" />



邻居聚合过程可以被抽象为multi-set（一个元素可重复的集合，在此处指节点的邻居集合，元素为节点，节点特征可重复）上的函数。如图中以圆点集合作例，点同色指特征相同：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163524200.png" alt="image-20221225163524200" style="zoom:80%;" />



**接下来我们分析GCN和GraphSAGE的聚合函数**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163552776.png" alt="image-20221225163552776" style="zoom:80%;" />

- **GCN：mean-pool** 

  mean-pool + Linear + ReLU non-linearity

  根据Xu et al. ICLR 2019得到定理：GCN的聚合函数无法区分颜色占比相同的multi-set。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163729057.png" alt="image-20221225163729057" style="zoom:80%;" />

  **假设不同颜色的特征是独热编码特征，如图所示黄蓝二色特征**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163749242.png" alt="image-20221225163749242" style="zoom:80%;" />

  **GCN无法区分不同multi-set的一个实例**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163825854.png" alt="image-20221225163825854" style="zoom:80%;" />

- **GraphSAGE：max-pool**

  MLP + max-pool

  根据Xu et al. ICLR 2019得到定理：GraphSAGE的聚合函数无法区分具有相同的不同颜色，即具有一样的多种颜色，或者不重复颜色组成的集合相同。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225163938743.png" alt="image-20221225163938743" style="zoom:80%;" />

  一个失败案例：假设上一层节点嵌入经过一个单射的MLP函数形成不同的独热编码向量，经逐元素最大池化后得到相同输出：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225164006585.png" alt="image-20221225164006585" style="zoom:80%;" />



**根据上文对GNN表示能力的分析，我们得出的主要结论takeaway为**：

- GNN的表示能力由其邻居聚合函数决定
- 邻居聚合是个multi-set上的函数，multi-set是一个元素可重复的集合
- GCN和GraphSAGE的聚合函数都不能区分某些基本的multi-set，因此都不单射，不够具有表达能力。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225164205462.png" alt="image-20221225164205462" style="zoom:80%;" />



我们的目标是设计信息传递框架下表示能力最强的GNN，这要求我们设计出multi-set上的单射邻居聚合函数。

在本课程中，我们用神经网络建模单射的multi-set函数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225164252163.png" alt="image-20221225164252163" style="zoom:80%;" />



**单射的multi-set函数**：

根据Xu et al. ICLR 2019得到定理：任一单射的multi-set函数都可以表示为
$$
\Phi(\sum_{x\in S}f(x))
$$
$\Phi$ 和 $f$ 是非线性函数，$\sum_{x\in S} $ 在multi-set上求和。

**如图所示**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225164454573.png" alt="image-20221225164454573" style="zoom:80%;" />



**一个作为证明的直觉举例**：

$f$ 得到颜色的独热编码，对其求和就能得到输入multi-set的全部信息（每类对应向量的一个索引，每个索引的求和结果就对应该类的节点数量，就可以区分不同类任何个数的情况）

**如图所示**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225165606485.png" alt="image-20221225165606485" style="zoom:80%;" />



**全局近似定理（universal approximation theorem）**：

为了建模 $\Phi(\sum_{x\in S}f(x))$ 中的 $\Phi$ 和 $f$ 我们使用MLP：因为根据 universal approximation theorem，只有一个隐藏层的MLP只要隐藏层维度够宽，并有合适的非线性函数 $\sigma(\cdot)$ （包括ReLU和sigmoid），就可以任意精度逼近任何连续函数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225165751912.png" alt="image-20221225165751912" style="zoom:80%;" />



**应用MLP，我们可以用神经网络建模出任一单射的multiset函数，其形式即变为**：
$$
MLP_{\Phi}(\sum_{x\in S}MLP_f(x))
$$
在实践中，MLP的隐藏层维度在100-500就够用了。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225165854807.png" alt="image-20221225165854807" style="zoom:80%;" />



**图同构网络（Graph Isomorphism Network，GIN）**：
$$
MLP_{\Phi}(\sum_{x\in S}MLP_f(x))
$$
其聚合函数是单射的，没有区分失败的案例，是信息传递类GNN中表示能力最强的GNN。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225170137143.png" alt="image-20221225170137143" style="zoom:80%;" />



**GIN与WL graph kernel的关系**：

我们通过将GIN与WL graph kernel（获得图级别特征的传统方法）做关联，来全面了解GIN模型。

GIN可以说是WL graph kernel的神经网络版。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225170238111.png" alt="image-20221225170238111" style="zoom:80%;" />



**WL graph kernel就像个硬编码的图神经网络**：

算法：color refinement

迭代公式：
$$
c^{(k+1)}(v)=HASH(c^{(k)}(v),\{c^{(k)}(u))\}_{u\in N(v)})
$$
迭代至稳定后，如果两个图的颜色集相同，说明它们同构。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225170501247.png" alt="image-20221225170501247" style="zoom:80%;" />

**GIN就用一个神经网络来建模这个单射的哈希HASH函数**：

单射函数建立在这个元组tuple上：$(c^{(k)}(v),\{c^{(k)}(u))\}_{u\in N(v)})$（根节点特征，邻居节点颜色）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225170609912.png" alt="image-20221225170609912" style="zoom:80%;" />

根据Xu et al. ICLR 2019得到定理：这样一个元组上的单射函数可以被建模为：
$$
MLP_{\Phi}((1+\epsilon)\cdot MLP_{f}(c^{(k)}(v))+\sum_{u\in N(v)}MLP_f(c^{(k)}(u)))
$$
其中 $\epsilon$ 是一个可训练的标量。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225170830005.png" alt="image-20221225170830005" style="zoom:80%;" />



如果输入特征（即初始颜色）是独热编码，那么直接加总就是单射的（跟上面的例子一样），我们就仅需 $\Phi$ 来确保函数的单射，它需要产生独热编码来作为下一层的输入特征。即：
$$
GINConv(c^{(k)}(v),\{c^{(k)}(u)\}_{u\in N(v)})=MLP_\Phi((1+\epsilon)\cdot c^{(k)}(v)+\sum_{u\in N(v)}c^{(k)}(u))
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225171324443.png" alt="image-20221225171324443" style="zoom:80%;" />



**GIN的节点嵌入更新过程**：

- 分配节点的初始向量 $c^{(0)}(v)$
- 迭代更新：$c^{(k+1)}(v)=GINConv(\{c^{(k)}(v),\{c^{(k)}(u)\}_{u\in N(v)}\})$ ，GINConv相当于可微的color HASH函数，将不同输入映射到不同嵌入中（即单射）
- 经过K次迭代，$c^{(K)}(v)$ 总结得到节点 $v$ 的 $K$ 跳邻居结构信息。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225171700990.png" alt="image-20221225171700990" style="zoom:80%;" />



**GIN和WL graph kernel**： GIN相当于可微神经网络版的WL graph kernel

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225171752610.png" alt="image-20221225171752610" style="zoom:80%;" />

**GIN相较于WL graph kernel的优点在于：**

- 节点嵌入是低维的，因此可以捕获到细粒度的节点相似性
- GINConv的参数可被学习得到并应用于下流任务



**GIN的表示能力**：

由于GIN与WL graph kernel之间的关系，二者的表示能力是相同的，也就是对于一样的图，要么二者都能区分，要么都不能区分。

在理论上和实践上，WL graph kernel都能区分现实世界的大部分图，因此GIN也能区分现实世界的大部分图。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225171912746.png" alt="image-20221225171912746" style="zoom:80%;" />



**本节课总结**：

1. 我们设计了一个可以建模单射的multi-set函数的神经网络
2. 我们用这个神经网络来聚合邻居信息，得到GIN：表示能力最强的GNN模型
3. 关键在于用element-wise sum pooling代替mean-/max-pooling
4. GIN与WL graph kernel有关
5. GIN和WL graph kernel都能区分现实世界的大部分图

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225171959649.png" alt="image-20221225171959649" style="zoom:80%;" />



**各种池化方法的能力**：sum能区分整个multiset，mean只能区分不同的分布，max只能区分元素类型集合

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221225172039235.png" alt="image-20221225172039235" style="zoom:80%;" />



**增加GNN的表示能力**：

对于类似“节点处于不同环中”这种问题，GNN仍然无法区分（因为计算图相同）。解决方法可以是添加可区分节点的feature，也可以使用reference node来区分相同计算图等。后续课程将会讲述具体做法。





# 12 知识图嵌入

**本章主要内容**：

本章首先介绍了 **异质图heterogeneous graph** 和 **relational GCN (RGCN)**。

接下来介绍了 **知识图谱补全knowledge graph completion** 任务，以及通过图嵌入方式的四种实现方式及其对关系表示的限制：**TransE**，**TransR**，**DistMult**，**ComplEx**。



## 12.1 异质图与RGCN

**本节课任务**：

之前课程的内容都囿于一种边类型，本节课拓展到有向、多边类型的图（即异质图）上。

介绍RGCN，知识图谱，知识图谱补全任务的表示方法。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104400505.png" alt="image-20221226104400505" style="zoom:80%;" />

**图的节点和边都可以是异质的**。



**异质图**：节点集 $V$ ，节点 $v_i\in V$ ；边集 $E$ ，边 $(v_i,r,v_j)\in E$ ；节点类型 $T(v_i)$ ，边类型集合 $R$ ，边类型 $r\in R$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104621829.png" alt="image-20221226104621829" style="zoom:80%;" />

异质图举例：生物医学知识图谱或事件图

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104649527.png" alt="image-20221226104649527" style="zoom:80%;" />



**Relational GCN**：将GCN拓展到异质图上。

**从只有一种边类型的有向图开始**：通过GCN学习节点A的表示向量，即沿其入边形成的计算图进行信息传播（message + aggregation）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104813005.png" alt="image-20221226104813005" style="zoom:80%;" />

**对于有多种边类型的情况**：在信息转换时，对不同的边类型使用不同的权重 $W$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104846745.png" alt="image-20221226104846745" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104858447.png" alt="image-20221226104858447" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226104910967.png" alt="image-20221226104910967" style="zoom:80%;" />



**Relation GCN 定义**：
$$
\bold{h}_v^{(l+1)}=\sigma(\sum_{r\in R}\sum_{u\in N_v^r}\frac1{c_{v,r}}\bold{W}_r^{(l)}\bold{h}_u^{(l)}+\bold{W}_r^{(0)}\bold{h}_v^{(l)})
$$
其中 $c_{v,r}=|N_v^r|$ 用节点in-degree进行归一化。

**message**：
$$
对特定关系的邻居:\bold{m}_{u,r}^{(l)}=\frac1{c_{v,r}}\bold{W}_r^{(l)}\bold{h}_u^{(l)}\\
对自环:\bold{m}_v^{(l)}=\bold{W}_r^{(0)}\bold{h}_v^{(l)}
$$
**aggregation**：加总邻居和自环信息，应用激活函数
$$
\bold{h}_v^{(l+1)}=\sigma(Sum(\{\bold{m}_{u,r}^{(l)},u\in\{N(v)\}\cup\{v\} \}))
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226105708440.png" alt="image-20221226105708440" style="zoom:80%;" />



**RGCN的scalability**：每种关系都需要 $L$ （层数）个权重矩阵 $\bold{W}_r^{(1)},\bold{W}_r^{(2)},...,\bold{W}_r^{(L)}$ ，每个权重矩阵的尺寸为 $d^{(l+1)}\times d^{(l)}$ （$d^{(l)}$ 是第 $l$ 层的隐嵌入维度）。

参数量随关系类数迅速增长，易产生过拟合问题。

2种规则化权重矩阵的方法：**block diagonal matrices** 和 **basis/dictionary learning**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226110004819.png" alt="image-20221226110004819" style="zoom:80%;" />

- **块对角矩阵（block diagonal matrices）**：使权重矩阵变稀疏，减少非0元素数量。做法就是如图所示，让权重矩阵成为这样对角块的形式。如果用 $B$ 个低维矩阵，参数量就会从 $d^{(l+1)}\times d^{(l)}$ 减少到 $B\times \frac{d^{(l+1)}}{B}\times \frac{d^{(l)}}{B}$ 。这种做法的限制在于，这样就只有相邻神经元/嵌入维度可以通过权重矩阵交互了。要解决这一限制，需要多加几层神经网络，或者用不同的block结构，才能让不在一个block内的维度互相交流。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226110504560.png" alt="image-20221226110504560" style="zoom:80%;" />

- **基础学习（basis learning）**：在不同关系之间共享权重参数。

  做法：将特定关系的权重矩阵表示为 基变换basis transformations 的 线性组合linear combination 的形式：
  $$
  \bold{W}_r=\sum_{b=1}^Ba_{rb}\cdot \bold{V}_b
  $$
  其中 $\bold{V}_b$ 在关系间共享，$\bold{V}_b$ 是基础矩阵或字典，$a_{rb}$ 是 $\bold{V}_b$ 的权重。

  这样我们对每个关系就只需要学习 $\{a_{rb} \}_{b=1}^B$ 这 $B$ 个标量了。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226110934699.png" alt="image-20221226110934699" style="zoom:80%;" />



**示例**：

1. **实体/节点分类**：

   目标：预测节点标签。

   RGCN使用最后一层产生的表示向量。

   举例：k-way分类任务，使用节点 $A$ 最后一层（prediction head）的输出 $\bold{h}_A^{(L)}\in \R^k$ （比如可能是经softmax输出），每个元素表示对应节点属于对应类的概率

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226112607693.png" alt="image-20221226112607693" style="zoom:80%;" />

2. **链接预测**：

   在异质图中，将每种关系对应的边都分成 training message edges, training supervision edges, validation edges, test edges 四类（切分每种关系所组成的同质图）

   这么分是因为有些关系类型的边可能很少，如果全部混在一起四分的话可能有的就分不到（比如分不到验证集里……之类的）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226112728562.png" alt="image-20221226112728562" style="zoom:80%;" />

   **RGCN在链接预测任务上的应用**：

   假定 $(E,r_3,A)$ 是 training supervision edge，则其他边都是 training message edges。

   用RGCN给 $(E,r_3,A)$ 打分：

   首先得到最后一层节点 $E$ 和节点 $A$ 的输出向量：$\bold{h}_E^{(L)}$ 和 $\bold{h}_A^{(L)}$ 

   然后应用 relation-specific 的打分函数 $f_r$ ：$\R^d \times \R^d\rightarrow \R$ 

   例如，$f_{r_1}(\bold{h}_E,\bold{h}_A)=\bold{h}_E^T\bold{W}_{r_1}\bold{h}_A$ ，$\bold{W}_{r_1}\in \R^{d\times d}$ 

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226113407409.png" alt="image-20221226113407409" style="zoom:80%;" />

   - **训练阶段**：用 training message edges 预测 training supervision edges

     1. 用RGCN给 training supervision edge $(E,r_3,A)$ 打分。

     2. 通过 扰乱perturb supervision edge 得到 **negative edge**：corrupt $(E,r_3,A)$ 的尾节点，举例得到 $(E,r_3,B)$ 。

        注意 negative edge 不能属于 training message edges 或 training supervision edges。

     3. 用GNN模型给 negative edge 打分。

     4. 优化交叉熵损失函数，使 training supervision edge 上得分最大，negative edge 上得分最低：
        $$
        l=-log\sigma(f_{r_3}(h_E,h_A))-log(1-\sigma(f_{r_3}(h_E,h_B)))
        $$
        其中 $\sigma$ 是sigmoid函数。

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226114837723.png" alt="image-20221226114837723" style="zoom:80%;" />

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226114855911.png" alt="image-20221226114855911" style="zoom:80%;" />

   - **验证阶段（测试阶段类似）**：

     用 training message edges 和 training supervision edges 预测 validation edges：$(E,r_3,D)$ 的得分应该比所有 negative edges 的得分更高。

     negative edges：尾节点不在 training message edges 和 training supervision edges 中的以 $E$ 为头节点、$r_3$ 为关系的边，如 $(E,r_3,B)$ 。

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115102389.png" alt="image-20221226115102389" style="zoom:80%;" />

     **具体步骤**：

     1. 计算 $(E,r_3,D)$ 的得分
     2. 计算所有 negative edges：$\{(E,r_3,v)|v\in\{B,F \} \}$ 的得分
     3. 获得 $(E,r_3,D)$ 的排名 ranking RK
     4. 计算指标：
        - $Hits@k:1[EK\le k]$ 的次数越高越好
        - Reciprocal Rank： $\frac1{RK}$ 越高越好

     <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115511849.png" alt="image-20221226115511849" style="zoom:80%;" />



**总结**：

1. Relational GCN：用于异质图的图神经网络模型
2. 可用于实体分类和链接预测任务
3. 类似思想可以扩展到其他RGNN模型上（如RGraphSAGE，RGAT等）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115547352.png" alt="image-20221226115547352" style="zoom:80%;" />



## 12.2 知识图：KG嵌入完成

**知识图谱 Knowledge Graphs (KG)**：以图形式呈现的知识，捕获实体entity（节点）、类型（节点标签）、关系relationship（边）。

**一种异质图实例**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115710842.png" alt="image-20221226115710842" style="zoom:80%;" />



**示例**：

- **bibliographic networks**：通过定义节点类型、关系类型及其之间的关系，得到如图所示的schema

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115810213.png" alt="image-20221226115810213" style="zoom:80%;" />

- **bio knowledge graphs**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115835933.png" alt="image-20221226115835933" style="zoom:80%;" />



**知识图谱应用实例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226115946071.png" alt="image-20221226115946071" style="zoom:80%;" />



**公开可用的知识图谱有：FreeBase, Wikidata, Dbpedia, YAGO, NELL, etc.**

这些知识图谱的共同特点是：大，不完整（缺少很多真实边）

对于一个大型KG，几乎不可能遍历所有可能存在的事实，因此需要预测可能存在却缺失的边

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226120113498.png" alt="image-20221226120113498" style="zoom:80%;" />



**举例：Freebase**

大量信息缺失

有 complete 的子集供研究KG模型

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226120145491.png" alt="image-20221226120145491" style="zoom:80%;" />



## 12.3 知识图补全：TransE，TransR，DistMult，ComplEx

**知识图谱补全（KG Completion Task）**：已知 (head, relation)，预测 tails（注意，这跟链接预测任务有区别，链接预测任务是啥都不给，直接预测哪些链接最有可能出现）

举例：已知（JK罗琳，作品流派），预测 tail “科幻小说”

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226121348545.png" alt="image-20221226121348545" style="zoom:80%;" />



在本节课中使用 shallow encoding 的方式来进行图表示学习，也就是用固定向量表示图数据（虽然这里不用GNN，但是如果愿意的话也可以用）



**知识图谱表示**：边被表示为三元组的形式 $(h,r,t)$ ，将实体和边表示到嵌入域/表示域 $\R^d$ 中。

给出一个真实的三元组 $(h,r,t)$ ，目标是对 $(h,r)$ 的嵌入应靠近 $t$ 的嵌入。

以下介绍如何嵌入 $(h,r)$ ，如何定义相似性。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226121621942.png" alt="image-20221226121621942" style="zoom:80%;" />



**TransE（translate embeddings）**：给出三元组 $(h,r,t)$ ，将实体和关系都映射到 $\R^d$ 上，使三元组为真时 $\bold{h}+\bold{r}\approx\bold{t}$ ，反之 $\bold{h}+\bold{r}\ne \bold{t}$ 。

**scoring function**：$f_r(h,t)=-||\bold{h}+\bold{r}-\bold{t}||$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226121938929.png" alt="image-20221226121938929" style="zoom:80%;" />

**TransE的算法**：

<img src="https://img-blog.csdnimg.cn/20210702180649244.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70#pic_center" alt="img" style="zoom:50%;" />

更新参数时使用的是 contrastive loss，总之大意就是最小化真三元组的距离（也就是最大化真三元组的score或相似性）、最大化假三元组的距离

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226122116049.png" alt="image-20221226122116049" style="zoom:80%;" />



**KG中关系的模式（Connectivity Patterns in KG）**：在KG中，关系可能有多种属性，我们接下来就要探讨KG嵌入方法（如TransE等）能否建模、区分开这些关系模式

- **symmetric relations（如室友关系）**：
  $$
  r(h,t)\Rightarrow r(t,h)
  $$

- **antisymmetric relations（如上位词关系）**：
  $$
  r(h,t)\Rightarrow \neg r(t,h)
  $$

- **inverse relations（如导师-学生关系）**：
  $$
  r_2(h,t)\Rightarrow r_1(t,h)
  $$

- **composition (transitive) relations（如母亲-姐姐-姨母关系）**：
  $$
  r_1(x,y)\and r_2(y,z)\Rightarrow r_3(x,z)
  $$

- **1-to-N relations（如 的学生 关系）**：
  $$
  r(h,t_1),r(h,t_2),...,r(h,t_n) 同时为真
  $$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226122634593.png" alt="image-20221226122634593" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226122652434.png" alt="image-20221226122652434" style="zoom:80%;" />



- **TransE： Antisymmetric Relations** $\checkmark$
  $$
  \bold{h}+\bold{r}=\bold{t},\bold{t}+\bold{r}\ne\bold{h}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226122911182.png" alt="image-20221226122911182" style="zoom:80%;" />

- **TransE： Inverse Relations** $\checkmark$
  $$
  \bold{h}+\bold{r_2}=\bold{t},设置\bold{r_2}=-\bold{r_1} ,即可实现 \bold{t}+\bold{r_1}=\bold{h}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226123137746.png" alt="image-20221226123137746" style="zoom:80%;" />

- **TransE： Composition** $\checkmark$
  $$
  \bold{x}+\bold{r_1}=\bold{y},\bold{y}+\bold{r_2}=\bold{z},若\bold{r_3}=\bold{r_1}+\bold{r_2}，则有\bold{x}+\bold{r_3}=\bold{z}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226123447242.png" alt="image-20221226123447242" style="zoom:80%;" />

- **TransE： Symmetric Relations** $×$  
  $$
  欲使 ||\bold{h}+\bold{r}-\bold{t} ||=0 和 ||\bold{t}+\bold{r}-\bold{h} ||=0同时成立，需\bold{r}=0且\bold{h}=\bold{t}
  $$
  但这样把两个实体嵌入到同一点上就没有意义了，所以是不行的

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226123836326.png" alt="image-20221226123836326" style="zoom:80%;" />

- **TransE： 1-to-N Relations** $×$
  $$
  欲使 ||\bold{h}+\bold{r}-\bold{t_1} ||=0 和 ||\bold{h}+\bold{r}-\bold{t_2} ||=0同时成立，需\bold{t_1}=\bold{t_2}
  $$
  但这样把两个实体嵌入到同一点上就没有意义了，所以是不行的

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226124020368.png" alt="image-20221226124020368" style="zoom:80%;" />



**TransR**：将实体映射为实体空间 $\R^d$ 上的向量，关系映射为关系空间上的向量 $\bold{r}\in \R^{k}$ ，且有 relation-specific 的 projection matrix $\bold{M}_r\in \R^{k\times d}$ 。

用 projection matrix 将实体从实体域投影到空间域上：$\bold{h}_{⊥}=\bold{M}_r\bold{h}$ ，$\bold{t}_=\bold{M}_r\bold{t}$ 。

**scoring function** ：$f_r(h,t)=-||\bold{h}_⊥+\bold{r}-\bold{t}_⊥ ||$ 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226130325127.png" alt="image-20221226130325127" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226130349011.png" alt="image-20221226130349011" style="zoom:80%;" />



- **TransR： Symmetric Relations** $\checkmark$
  $$
  \bold{r}=0,\bold{h}_⊥=\bold{M}_r\bold{h}=\bold{M}_r\bold{t}=\bold{t}_⊥
  $$
  即可以使在实体域上不同的 $h$ 和 $t$ 可以在 $r$ 的关系域上相同。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226130716471.png" alt="image-20221226130716471" style="zoom:80%;" />

- **TransR： Antisymmetric Relations ** $\checkmark$ 
  $$
  \bold{r}\ne0,\bold{M}_r\bold{h}+\bold{r}=\bold{M}_r\bold{t},则 \bold{M}_r\bold{t}+\bold{r}\ne \bold{M}_r\bold{h}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226131008403.png" alt="image-20221226131008403" style="zoom:80%;" />

- **TransR： 1-to-N Relations** $\checkmark$
  $$
  学习合适的\bold{M}_r,使\bold{t}_⊥=\bold{M}_r\bold{t}_1=\bold{M}_r\bold{t}_2
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226131308940.png" alt="image-20221226131308940" style="zoom:80%;" />

- **TransR： Inverse Relations** $\checkmark$ 
  $$
  \bold{r}_2=-\bold{r}_1,\bold{M}_{r_1}=\bold{M}_{r_2},则 \bold{M}_{r_1}\bold{t}+\bold{r}_1=\bold{M}_{r_1}\bold{h}且\bold{M}_{r_2}\bold{h}+\bold{r}_2=\bold{M}_{r_2}\bold{t}
  $$
  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226131702901.png" alt="image-20221226131702901" style="zoom:80%;" />

- **TransR： Composition Relations** $×$

  每个关系都有独立的空间域，不能直接自然组合

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226131833907.png" alt="image-20221226131833907" style="zoom:80%;" />



**New Idea: Bilinear Modeling**

在至今学习过的TransE和TransR中，scoring function $f_r(h,t)$ 都是L1或L2距离的相反数。另一种做法是选用 bilinear modeling。

**DistMult**：实体和关系都表示为 $\R^k$ 的向量（在这一点上有点像TransE）

**score function**：$f_r(h,t)=<h,r,t>=\sum_ih_i\cdot r_i\cdot t_i$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226135554575.png" alt="image-20221226135554575" style="zoom:80%;" />

**DistMult**：score function 可以直觉地被视作 $\bold{h}\cdot\bold{r}$ 和 $\bold{t}$ 之间的 cosine similarity。即，使 $\bold{h}\cdot\bold{r}$ 和 $\bold{t}$ 同侧、靠近时，score 较高。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226135806820.png" alt="image-20221226135806820" style="zoom:80%;" />

- **DistMult： 1-to-N Relations** $\checkmark$

  当知识图谱中存在 $(h,r,t_1)$ 和 $(h,r,t_2)$ ，如图所示，DistMult可以建模使 $t_1$ 和 $t_2$ 在 $h\cdot r$ 上的投影等长，即使二者与 $h\cdot r$ 的点积相等，即 $<h,r,t_1>=<h,r,t_2>$ ，符合要求。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226140142793.png" alt="image-20221226140142793" style="zoom:80%;" />

- **DistMult： Symmetric Relations** $\checkmark$ 

  DistMult天然建模symmetric relations：$f_r(h,t)=<h,r,t>=\sum_ih_i\cdot r_i\cdot t_i=<t,r,h>=f_r(t,h)$ 

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226140521340.png" alt="image-20221226140521340" style="zoom:80%;" />

- **DistMult： Antisymmetric Relations** $×$

  $f_r(h,t)==\sum_ih_i\cdot r_i\cdot t_i==f_r(t,h)$ 永远成立，不符要求

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226140756492.png" alt="image-20221226140756492" style="zoom:80%;" />

- **DistMult： Inverse Relations** $×$

  如果要建模inverse relations，即使 $f_{r_2}(h,t)=<h,r_2,t>=<t,r_1,h>=f_{r_2}(t,h)$ 必须使 $r_1=r_2$ ，这显然是没有意义的。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226141825218.png" alt="image-20221226141825218" style="zoom:80%;" />

- **DistMult： Composition Relations** $×$

  DistMult对每个 (head,relation) 定义了一个超平面，对多跳关系产生的超平面的联合（如 $(r_1,r_2)$ ）无法用单一超平面 $(r_3)$ 来表示。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226141955429.png" alt="image-20221226141955429" style="zoom:80%;" />



**ComplEx**：基于DistMult，ComplEx在复数向量域complex vector space $\C^k$ 中表示实体和关系。

$\bold{u}=\bold{a}+\bold{b}i$ （其中 $\bold{u}\in\C^k$ ，实部 $a\in\R^k$ ，虚部 $b\in\R^k$）

$\bar{\bold{u}}=\bold{a}-\bold{b}i$ （共轭） 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226144920440.png" alt="image-20221226144920440" style="zoom:80%;" />

**score function **：$f_r(h,t)=Re(\sum_ih_i\cdot r_i\cdot\bar{t_i})$ 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226145118894.png" alt="image-20221226145118894" style="zoom:80%;" />

- **ComplEx： Antisymmetric Relations** $\checkmark$ 

  ComplEx可以建模使 $f_r(h,t)=Re(\sum_ih_i\cdot r_i\cdot \bar{t_i})$ 与 $f_r(t,h)=RE(\sum_i t_i\cdot r_i\cdot \bar{h_i})$ 不同，因为这一不对称建模使用的是复数共轭。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226145444625.png" alt="image-20221226145444625" style="zoom:80%;" />

- **ComplEx： Symmetric Relations** $\checkmark$ 

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226150000607.png" alt="image-20221226150000607" style="zoom:80%;" />

- **ComplEx： Inverse Relations** $\checkmark$

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226150113590.png" alt="image-20221226150113590" style="zoom:80%;" />

- ComplEx：Composition Relations $×$ 1-to-N Relations $\checkmark$

  和DistMult一样



**所有模型的表示能力对比**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226150236644.png" alt="image-20221226150236644" style="zoom:80%;" />



**知识图谱嵌入问题的实践应用**：

1. 不同知识图谱可能会有很不同的关系模式
2. 因此没有适合所有KG的嵌入方法，可用上表来辅助选择
3. 可以先试用TransE来迅速获得结果（如果目标KG没有过多symmetric relations的话）
4. 然后再用更有表示能力的模型，如ComplEx或RotatE（复数域的TransE）等

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221226150314902.png" alt="image-20221226150314902" style="zoom:80%;" />



## 12.4 总结

1. 链接预测或图补全任务是知识图谱领域的重要研究任务
2. 介绍了不同嵌入域和不同表示能力的模型
   1. TransE
   2. TransR
   3. DistMult
   4. ComplEx





# 13 Colab3





# 14 知识图上的推理

**本章主要内容**：

本章将介绍知识图谱上的推理任务。

目标是回答 多跳查询multi-hop queries，包括path queries和conjunctive queries。

介绍query2box方法以解决predictive queries问题。



## 14.1 知识图上的推理

回忆：知识图谱补全任务。

本章主旨：介绍如何实现知识图谱上的多跳推理任务。

- 回答多跳查询问题，包括path queries和conjunctive queries。在某种程度上也可以说是在做知识图谱预测问题，在对任意predictive queries做预测。
- 介绍query2box方法。

![image-20221231111944846](C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231111944846.png)



**知识图谱示例：Biomedicine**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231112049928.png" alt="image-20221231112049928" style="zoom:80%;" />



**KG上的predictive queries**：

任务目标：在一个incomplete的大型KG上做多跳推理（如回答复杂查询问题）。

对于某一类查询，我们可以自然语言的形式（绿色字）、formula/logical structure（棕色字）的形式或者graph structure（蓝色节点是查询中出现的实体，绿色节点是查询结果）的形式来表示它。

本节课仅讨论有formula/logical structure或graph structure后如何进行工作，从自然语言转换到对应形式的工作不在本课程讲解。



**查询类型及示例**：

1. **one-hop queries**： What adverse event is caused by Fulvestrant?

   (e:Fulvestrant，(r:Causes))

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231112441440.png" alt="image-20221231112441440" style="zoom:80%;" />

2. **path queries**： What protein is associated with the adverse event caused by Fulvestrant?

   (e:Fulvestrant, (r:Causes, r:Assoc))

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231112539290.png" alt="image-20221231112539290" style="zoom:80%;" />

3. **conjunctive queries**： What is the drug that treats breast cancer and caused headache?

   ((e:BreastCancer, (r:TreatedBy))，(e:Migraine, (r:CausedBy))

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231112631098.png" alt="image-20221231112631098" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231112706959.png" alt="image-20221231112706959" style="zoom:80%;" />



**predictive one-hop queries**：

知识图谱补全任务可以formulate成回答one-hop queries问题

KG补全任务：链接 $(h,r,t)$ 在KG中是否存在？

one-hop query： $t$ 是否是查询 $(h,(r))$ 的答案？

举例：What side effects are caused by drug Fulvestrant ?

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231112937105.png" alt="image-20221231112937105" style="zoom:80%;" />



**path queries**： one-hop queries可以视作path queries的特殊情况，one-hop queries在路径上增加更多关系就成了path queries。

一个n-hop query $q$ 可表示为 $q=(v_a,(r_1,...,r_n))$

$v_a$ 是 anchor entity ，查询结果可表示为 $[q]_G$ 。

$q$ 的query plan（一个链）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231123357403.png" alt="image-20221231123357403" style="zoom:80%;" />

**path queries示例**：What proteins are associated with adverse events caused by Fulvestrant?

$v_a$ 是 e:Fulvestrant，$(r_1,r_2)$ 是 (r:Causes, r:Assoc)，query：(e:Fulvestrant, (r:Causes, r:Assoc))

**query plan**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231124833668.png" alt="image-20221231124833668" style="zoom:80%;" />



**那么我们应该如何回答KG上的path query问题呢**？

如果图是complete的话，那么我们只需要沿query plan直接traverse（遍历）KG就可以。

1. **从anchor node（Fulverstrant）开始**：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231124938642.png" alt="image-20221231124938642" style="zoom:80%;" />

2. **从anchor node（Fulverstrant）开始，遍历关系“Causes”，到达实体{“Brain Bleeding”, “Short of Breath”, “Kidney Infection”, “Headache”}**

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231125016649.png" alt="image-20221231125016649" style="zoom:80%;" />

3. **从实体{“Brain Bleeding”, “Short of Breath”, “Kidney Infection”, “Headache”}开始，遍历关系“Assoc”，到达实体{“CASP8”, “BIRC2”, “PIM1”}，即所求答案**

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231125058346.png" alt="image-20221231125058346" style="zoom:80%;" />



**但由于KG是incomplete的，所以如果仅traverse KG，可能会缺失一些关系，从而无法找到全部作为答案的实体。**

我们可能很直觉地会想，那能不能直接先用KG补全技术，将KG补全为completed (probabilistic) KG，然后再traverse KG？

但这样不行，KG被补全后就会是一个稠密图，因为KG补全后很多关系存在的概率都非0，所以KG上会有很多关系，在traverse时要过的边太多，其复杂度与路径长度 $L$ 呈指数增长：$O(d_{max}^L)$ ，复杂度过高，无法实现。

**因此我们就需要进行预测任务：predictive queries**

目标：在incomplete KG上回答path-based queries

我们希望这一方法能够回答任意查询问题，同时隐式地impute或补全KG，实现对KG中缺失信息和噪音的鲁棒性。

对链接预测任务的泛化：从one-step link prediction task（就以前讲过的那种）到multi-step link prediction task（path queries）



## 14.2 在知识图上回答预测查询

**idea: traversing KG in vector space**

核心思想：嵌入query

相当于把TransE泛化到multi-hop reasoning任务上：使query embedding $\bold{q}$ （相当于一个实体加关系的嵌入 $\bold{q}=\bold{h}+\bold{r}$ ）与answer embedding $\bold{t}$ （一个实体）靠近，$f_q(t)=-||\bold{q}-\bold{t}||$ 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231125608491.png" alt="image-20221231125608491" style="zoom:80%;" />

对path query $q=(v_a,(r_1,...,r_n))$ ，其嵌入就是 $\bold{q}=\bold{v}_a + \bold{r}_1+...+\bold{r}_n$

嵌入过程仅包含向量相加，与KG中总实体数无关。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231125803766.png" alt="image-20221231125803766" style="zoom:80%;" />

**path query示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231125834165.png" alt="image-20221231125834165" style="zoom:80%;" />

可以训练TransE来优化KG补全目标函数。

因为TransE天然可以处理composition relations，所以也能处理path queries，在隐空间通过叠加relation嵌入来表示多跳。

TransR / DistMult / ComplEx无法处理composition relations，因此很难像TransE这样轻易扩展到path queries上。



**conjunctive queries**：

示例：What are drugs that cause Short of Breath and treat diseases associated with protein ESR2?

((e:ESR2, (r:Assoc, r:TreatedBy)), (e:Short of Breath, (r:CausedBy))

**query plan**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130045557.png" alt="image-20221231130045557" style="zoom:80%;" />

**同样，如果KG是complete的话，直接traverse KG就行**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130154906.png" alt="image-20221231130154906" style="zoom: 80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130216474.png" alt="image-20221231130216474" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130244802.png" alt="image-20221231130244802" style="zoom:80%;" />

**同样，如果KG有关系缺失了，有些答案就会找不到**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130334500.png" alt="image-20221231130334500" style="zoom:80%;" />



**我们希望通过嵌入方法来隐式impute KG中缺失的关系 (ESR2, Assoc, Breast Cancer)**：

如图所示，ESR2与BRCA1和ESR1都有interact关系，这两个实体都与breast cancer有assoc关系：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130455709.png" alt="image-20221231130455709" style="zoom:80%;" />



再回顾一遍query plan，注意图中的中间节点都代表实体，我们也需要学习这些实体的表示方法。此外我们还需要定义在隐空间的intersection操作。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130544575.png" alt="image-20221231130544575" style="zoom:80%;" />



## 14.3 Query2box: 基于盒嵌入的KGs推理

**box embeddings**：用 hyper-rectangles (boxes) 来建模query $\bold{q}=(Center(q),Offset(q))$ 。

一个多维长方形，用中心和corner（偏移）来定义。

如图所示：在理想状态下，一个box里包含了所有query（Fulverstrant副作用）的回答的实体。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130750868.png" alt="image-20221231130750868" style="zoom:80%;" />



**key insight**: box就是组合之后还是box，就很好定义节点集的intersection。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231130836784.png" alt="image-20221231130836784" style="zoom:80%;" />



**embed with box embedding**：

1. 实体嵌入：zero-volume boxes，参数量 $d|V|$
2. 关系嵌入：从盒子投影到盒子，参数量 $2d|R|$ 
3. intersection operator $f$ ：从盒子投影到盒子，建模box的intersection操作

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231133026547.png" alt="image-20221231133026547" style="zoom:80%;" />



**projection operator** $P$ ： 用当前box作为输入，用关系嵌入来投影和扩展box，得到一个新的box

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231131204108.png" alt="image-20221231131204108" style="zoom:80%;" />



**用box embedding，用projection operator，沿query plan求解**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231131314353.png" alt="image-20221231131314353" style="zoom:80%;" />



**接下来我们的问题就在于：如何定义box上的intersection？**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231131340894.png" alt="image-20221231131340894" style="zoom:80%;" />

有一种对intersection的定义比较严格，就是定义为数学上的intersection，类似于维恩图。我们想要更flexible一点的定义，就如下文所介绍。



**geometric intersection operator** $J$ ：

输入：多个box

输出：intersection box

$J:Box\times Box\times...\times Box\rightarrow Box$ 

**直觉**：

- 输出box的center应该靠近输入boxes的centers
- offset (box size) 应该收缩（因为intersected set应该比所有input set的尺寸都小）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231131645650.png" alt="image-20221231131645650" style="zoom:80%;" />



**intersection operator公式**：

- **center**：
  $$
  Cen(q_{inter})=\sum_{i}\bold{w}_i\odot Cen(q_i)\\
  \bold{w}_i=\frac{exp(f_{cen}(Cen(q_i)))}{\sum_j exp(f_{cen}(Cen(q_j)))}
  $$
  其中 $\odot$ 是哈达玛积，即逐元素乘积。$Cen(q_i)\in\R^d$ ，$\bold{w}_i\in\R^d$ 。

  直觉解读：center应该在下图红色区域内（原center之间）

  应用：center是原center的加权求和

  $\bold{w}_i\in\R^d$ 通过含可训练参数的神经网络 $f_{cen}$ 计算得到，代表每个输入 $Cen(q_i)$ 的self-attention得分。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231133429243.png" alt="image-20221231133429243" style="zoom:80%;" />

- **offset**：
  $$
  Off(q_{inter})=min(Off(q_1),...,Off(q_n))\odot \sigma(f_{off}(Off(q_1),...,Off(q_n)))
  $$
  保证offset收缩：$\sigma$ 表示sigmoid函数，把输出压缩到 (0,1) 之间。$f_{off}$ 是一个含可训练参数的神经网络，提取input boxes的表示向量以增强表示能力。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231133743688.png" alt="image-20221231133743688" style="zoom:80%;" />



**通过定义intersection operator，现在我们可以完成使用box embedding沿query plan的求解**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231133830191.png" alt="image-20221231133830191" style="zoom:80%;" />



**entity-to-box distance**：

定义score function $f_q(v)$ （query box $q$ 和 entity embedding $v$ （也是个box）距离的相反数）
$$
f_q(v)=-d_{box}(\bold{q},\bold{v})\\
d_{box}(\bold{q},\bold{v})=d_{out}(\bold{q},\bold{v})+\alpha\cdot d_{in}(\bold{q},\bold{v}),\alpha\in(0,1)
$$
直觉：如果实体在盒子里面，距离权重就应该较小。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231134309293.png" alt="image-20221231134309293" style="zoom:80%;" />



**extending to union operation**：

析取问题示例：What drug can treat breast cancer or lung cancer?

conjunctive queries + disjunction被叫做Existential Positive First-order (EPFO) queries，也叫AND-OR queries。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231134358801.png" alt="image-20221231134358801" style="zoom:80%;" />



**在低维向量空间可以嵌入AND-OR queries吗？**

不能，在任意查询上的union操作必须要高维嵌入。

**举例：三个查询和对应的答案实体集合** $[q_1]=\{v_1\},[q_2]=\{v_2\},[q_3]=\{v_3\}$

如果我们允许union操作，可以将其嵌入到二维平面上吗？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231134547585.png" alt="image-20221231134547585" style="zoom:80%;" />

以下图示中，红点（答案）是我们希望在box中的实体，蓝点（负答案）是我们希望在box外的实体：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231134613734.png" alt="image-20221231134613734" style="zoom:80%;" />

对三个点来说，二维是足够的。

**举例：四个查询和对应的答案实体集合** $[q1]={v1},[q2]={v2},[q3]={v3},[q_4]=\{v_4\}$

如果我们允许union操作，可以将其嵌入到二维平面上吗？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231134738946.png" alt="image-20221231134738946" style="zoom:80%;" />

答案是不能，举例来说，如下图所示，我们希望设计一个 $\bold{q}_2\or \bold{q}_4$ 的box embedding，即 $v_2$ 和 $v_4$ 在box里，$v_1$ 和 $v_3$ 在box外。显然不行。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231134910480.png" alt="image-20221231134910480" style="zoom:80%;" />

结论：任何 $M$ 个conjunctive queries $q_1,...,q_M$ ，各自答案不重叠，我们需要 $\Theta(M)$ 维来处理所有 OR queries。这可能就很大。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231135117957.png" alt="image-20221231135117957" style="zoom:80%;" />



因为我们无法在低维空间嵌入 AND-OR queries，所以对这类问题，我们的处理思路就是把所有query plan前面的union操作单拎出来，只在最后一步进行union操作。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231135217178.png" alt="image-20221231135217178" style="zoom:80%;" />



**disjunctive normal form (DNF)**：AND-OR query可以表述成DNF的形式，例如conjunctive queries的disjunction $q=q_1\or q_2\or...\or q_m$ ，$q_i$ 是 conjunctive query.

这样的话我们就可以先嵌入所有的 $q_i$ ，然后在最后一步聚集起来。



**实体嵌入 $v$ 和DNF $q$ 之间的距离定义为**：
$$
d_{box}(\bold{q},\bold{v})=min(d_{box}(\bold{q}_1,\bold{v}),...,d_{box}(\bold{q}_m,\bold{v}))
$$
**直觉**:

- 只要 $v$ 是一个conjuctive query $q_i$ 的答案，$v$ 就是 $q$ 的答案。
- $v$ 离一个conjuctive query $q_i$ 的答案越近，$v$ 就应该离 $q$ 的嵌入域越近。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231135830676.png" alt="image-20221231135830676" style="zoom:80%;" />



**嵌入AND-OR query $q$ 的过程**：

1. 将 $q$ 转换为 equivalent DNF $q_1\or q_2\or...\or q_m$ 
2. 嵌入 $q_1$ 至 $q_m$ 
3. 计算 (box) distance $d_{box}(\bold{q}_i,\bold{v})$ 
4. 计算所有distance的最小值
5. 得到最终score $f_q(v)=-d_{box}(\bold{q},\bold{v})$ 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231140325474.png" alt="image-20221231140325474" style="zoom:80%;" />



**training overview**：

1. **overview and intuition（类似于KG补全问题）**：

   已知query embedding $\bold{q}$ 

   目标：最大化答案 $v\in[q]$ 上的得分 $f_q(v)$ ，最小化负答案 $v'\notin [q]$ 上的得分 $f_q(v')$

2. **可训练参数**：

   - 实体嵌入参数量 $d|V|$ 
   - 关系嵌入参数量 $2d|R|$
   - intersection operator

3. 接下来的问题就在于：如何从KG中获取query、query对应的答案和负答案来训练参数？如何划分KG？



**训练流程**：

1. 从训练图 $G_{train}$ 中随机抽样一个query $q$ ，及其答案 $v\in[q]$ 和一个负答案样本 $v'\notin[q]$ ，负答案样本：在KG中存在且和 $v$ 同类但非 $q$ 答案的实体。

2. 嵌入query $\bold{q}$ 

3. 计算得分 $f_q(v)$ 和 $f_q(v')$ 

4. 优化损失函数 $l$ 以最大化 $f_q(v)$ 并最小化 $f_q(v')$ ：
   $$
   l = -log\sigma(f_q(v))-log(1-\sigma(f_q(v')))
   $$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141040778.png" alt="image-20221231141040778" style="zoom:80%;" />



**抽样query：从templates生成**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141109348.png" alt="image-20221231141109348" style="zoom:80%;" />

生成复杂query的流程：从query template开始，通过实例化query template中的变量为KG中实际存在的实体和关系来生成query（如实例化Anchor1为KG节点ESR2，Rel1为KG边Assoc）。

query template可以视作是query的抽象。
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141236083.png" alt="image-20221231141236083" style="zoom:80%;" />

**实例化query template的具体方法：从实例化答案节点开始，迭代实例化其他边和节点，直至到达所有anchor nodes**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141318075.png" alt="image-20221231141318075" style="zoom:80%;" />



**实例化query template示例**：

从query template的根节点开始：从KG中随机选择一个实体作为根节点，例如我们选择了Fulverstrant

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141347010.png" alt="image-20221231141347010" style="zoom:80%;" />

然后我们看intersection：实体集的intersection是Fulverstrant，则两个实体集自然都应包含Fulverstrant

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141529565.png" alt="image-20221231141529565" style="zoom:50%;" />

我们通过随机抽样一个连接到当前实体Fulverstrant的关系，来实例化对应的projection edge。举例来说，我们选择关系TreatedBy，检查通过TreatedBy关系连接到Fulverstrant的实体：{Breast Cancer}

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141617392.png" alt="image-20221231141617392" style="zoom:80%;" />

以此类推，完成一条支路：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141649946.png" alt="image-20221231141649946" style="zoom:80%;" />

类似地，完成另一条支路：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141731621.png" alt="image-20221231141731621" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141754316.png" alt="image-20221231141754316" style="zoom:80%;" />

现在我们就得到了一个查询 $q$ ：((e:ESR2, (r:Assoc, r:TreatedBy)), (e:Short of Breath, (r:CausedBy))

$q$ 在KG上必有答案，而且其答案之一就是实例化的答案节点：Fulverstrant。

我们可以通过KG traversal获得全部答案集合 $[q]_G$ 。

抽样回答不了这个answer的节点作为 non-answer负样本 $v'\notin[q]_G$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231141922406.png" alt="image-20221231141922406" style="zoom:80%;" />



**在嵌入域可视化查询答案**：

示例：List male instrumentalists who play string instruments

用t-SNE将嵌入向量降维到2维

**可视化节点嵌入和query plan**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142025533.png" alt="image-20221231142025533" style="zoom:80%;" />

**anchor node的嵌入**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142106418.png" alt="image-20221231142106418" style="zoom:50%;" />

**投影**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142137293.png" alt="image-20221231142137293" style="zoom: 50%;" />

**投影**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142206935.png" alt="image-20221231142206935" style="zoom: 50%;" />

**anchor node的嵌入**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142253453.png" alt="image-20221231142253453" style="zoom:50%;" />

**投影**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142402746.png" alt="image-20221231142402746" style="zoom:50%;" />

**intersection**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231142426115.png" alt="image-20221231142426115" style="zoom:50%;" />



## 14.4 总结

本章介绍了在大型KG上回答predictive queries。

关键思想在嵌入查询，通过可学习的operator来实现嵌入。在嵌入域中，query的嵌入应该靠近其答案的嵌入。





# 15 基于GNN的频繁子图挖掘

**本章主要内容**：

本章首先介绍了图中motif / subgraph的概念，以及对motif significance的定义（即定义图中的subgraph要比null model多/少出多少才算显著，以及如何生成null model）。

接下来介绍了神经网络模型下的subgraph matching方法（同时也是subgraph的表示方法）。

最后介绍如何找到图中出现频率较高的motif / subgraph。



## 15.1 识别和计算网络中的主题motifs

**subgraph**：subgraph是网络的组成部分，可用于识别和区分不同的网络（可以说是不同种类网络会具有不同特征的subgraph）。

使用传统的discrete type matching方法代价很大，本文会介绍使用神经网络解决subgraph matching问题的方法。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231210433299.png" alt="image-20221231210433299" style="zoom:80%;" />



以下图分子式为例：含有羧基（subgraph）的分子（graph）是羧酸（group）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231210514622.png" alt="image-20221231210514622" style="zoom:80%;" />



## 15.2 Subgraph 和 Motifs

### 15.2.1 Subgraph 和 Motifs 的定义

**subgraph定义**：对于图 $G=(V,E)$ 有两种定义其subgraph的方式：

- **node-induced subgraph / induced subgraph**：常用定义

  图中的一个节点子集+原图中两个节点都在该节点子集内的边（即edges induced by the nodes）

  $G'=(V',E')$ 是 node induced subgraph，当且仅当 $V'\sube V$ ，$E'=\{(u,v)\in E|u,v\in V' \}$ 。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231210840521.png" alt="image-20221231210840521" style="zoom:80%;" />

- **edge-induced subgraph / non-induced subgraph / subgraph**：

  图中的一个边子集+该子集的对应节点

  $G'=(V',E')$ 是 edge induced subgraph，当且仅当 $E'\sube E$ ，$V'=\{v\in V|(v,u)\in E' forsome\ u \}$ 。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231211103378.png" alt="image-20221231211103378" style="zoom:80%;" />



**具体使用哪种定义取决于问题领域**：

如在化学领域中常使用node-induced概念（官能团），在知识图谱中常用edge-induced概念（我们关心的是代表逻辑关系的边）。



前文对subgraph的定义都需要 $V'\sube V$ 且 $E'\sube E$ ，即 $V'$ 和 $E'$ 都出自原图。如果节点和边出自不同的图但仍有对应关系，如下图所示，我们称这种情况为 $G_1$ is contained in $G_2$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231211424287.png" alt="image-20221231211424287" style="zoom:80%;" />



**图同构（graph isomorphism）**：如果 $G_1$ 和 $G_2$ 存在双射关系 $f:V_1\rightarrow V_2$ ，使得当且仅当 $(f(u),f(v))\in E_2$ 时，$(u,v)\in E_1$ （即 $G_1$ 中的节点能一一映射到 $G_2$ 中的节点，使节点之间对应的边关系也能同时映射到另一个图所对应的节点之间）。我们称两个图同构。

**如下图左图所示（节点颜色表示映射关系）**：因为节点没有固定顺序，所以我们不知道节点之间是怎么映射的，所以我们需要遍历所有可能。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231211736371.png" alt="image-20221231211736371" style="zoom:80%;" />

检验图是否同构的问题是否NP-hard未知，但至今没有提出polynomial algorithm。



**子图同构（subgraph isomorphism）**：如果 $G_2$ 的子图与 $G_1$ 同构，我们称 $G_2$ is subgraph-isomorphic to $G_1$ 。

我们可以使用node-induced或edge-induced subgraph定义。

这一问题是NP-hrad的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231211931515.png" alt="image-20221231211931515" style="zoom:80%;" />

节点之间的映射不必唯一。



**subgraph举例**：

1. 所有非同构的、connected、无向的4个节点的图：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212027401.png" alt="image-20221231212027401" style="zoom:80%;" />

2. 所有非同构的、connected、有向的3个节点的图：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212105383.png" alt="image-20221231212105383" style="zoom: 80%;" />

   一般最多也就4-5个节点了。



**network motifs 定义**：recurring, significant patterns of interconnections。

- pattern：小的node-induced subgraph
- recurring：出现很多次，即出现频率高（以下将介绍如何定义频率）
- significant：比预期（如在随机生成的图中）出现的频率高（以下将介绍如何定义随机图）

**motif举例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212318976.png" alt="image-20221231212318976" style="zoom:80%;" />

如图所示：左上角就是我们所感兴趣的induced subgraph（motif）。蓝三角内的induced subgraph符合要求，红三角内不是induced subgraph不符合要求。



**motif的意义**：

1. 帮助我们了解图的工作机制。
2. 帮助我们基于图数据集中某种subgraph的出现和没有出现来做出预测。

**举例**：

1. feed-forward loops：神经元网络中用于中和biological noise

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212435261.png" alt="image-20221231212435261" style="zoom:80%;" />

2. parallel loops：食物链中（就两种生物以同一种生物为食并是同一种生物的猎物嘛）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212459651.png" alt="image-20221231212459651" style="zoom:80%;" />

3. single-input modules：基因控制网络中

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212530187.png" alt="image-20221231212530187" style="zoom:80%;" />



<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212606797.png" alt="image-20221231212606797" style="zoom:80%;" />



**subgraph frequency**：

- **图级别的subgraph frequency定义**：

  设 $G_Q$ 是一个小图，$G_T$ 是目标图数据集。

  $G_Q$ 在 $G_T$ 中的频率：$G_T$ 不同的节点子集 $V_T$ 的数目（ $V_T$ induce的 $G_T$ 的subgraph与 $G_Q$ 同构）

  如下左图中frequency为2（红圈中的两种节点子集），右图中的frequency为 $C_{100}^6$ 

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231212926373.png" alt="image-20221231212926373" style="zoom:80%;" />

- **节点级别的subgraph frequency定义**：

  设 $G_Q$ 是一个小图，$v$ 是其一个节点（anchor），$G_T$ 是目标图数据集。

  $G_Q$ 在 $G_T$ 中的频率：$G_T$ 中节点 $u$ 的数目（ $G_T$ 的subgraph与 $G_Q$ 同构，其同构映射 $u$ 到 $v$ 上。

  $(G_Q,v)$ 叫node-anchored subgraph 。

  这种定义对异常值比较鲁棒。如在图例中，star subgraph以中心节点为anchor，其在 $G_T$ 中的frequency就是1；若以其外围节点作为anchor，则其frequency就是100。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231213319147.png" alt="image-20221231213319147" style="zoom:80%;" />

- 如果数据集中包含多个图，我们可将其视为一整个大图 $G_T$ （包含disconnected components，各组成部分对应单个图）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231213441757.png" alt="image-20221231213441757" style="zoom:80%;" />



### 15.2.2 确定主题Motifs意义

**我们首先需要定义null-model**

核心思想：在真实网络中比在随机网络中出现更频繁的subgraph有functional significance。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231213615694.png" alt="image-20221231213615694" style="zoom:80%;" />



**定义随机图**：**Erdős–Rényi (ER) random graphs**

$G_{n,p}$ ：$n$ 个节点的无向图，每个边 $(u,v)$ 以频率 $p$ 独立同分布出现。

可以是disconnected：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231213749638.png" alt="image-20221231213749638" style="zoom:80%;" />



**新模型：configuration model**

目标：按照给定度数序列 $k_1,k_2,...,k_N$ 生成随机图。

作为网络的null model很有用，可以将真实图和与其具有相同度数序列的随机图作比。

configuration model流程如图所示：对节点上的边进行两两配对，得到最终的结果图（如果出现重边（multiple edges）或自环的情况，由于其罕见，所以可以直接忽略。如图中A-B节点之间出现了double edge，但在最后的结果图中就忽略了，仅作为一条边来处理）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231215502609.png" alt="image-20221231215502609" style="zoom:80%;" />



**除了像上图那种节点辐条的做法，还可以使用switching方法**：

对给定图 $G$ ，重复switching步骤 $Q\cdot |E|$ 次：

1. 随机选取一对边 $A\rightarrow B$ ，$C\rightarrow D$ 
2. 交换端点使 $A\rightarrow D$ ，$C\rightarrow B$ （仅在无重边或自环被产生时进行交换操作）

得到randomly rewired graph（与原图的节点度数相同）

$Q$ 需足够大（如100）以使这个过程收敛

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231215801979.png" alt="image-20221231215801979" style="zoom:80%;" />



**motif significance overview**：

检验motif在真实网络中是否比在随机图中overrepresent的步骤：

1. step1：在真实图中数motif的个数
2. step2：产生多个与真实图有相同统计量（如节点数、边数、度数序列等）的随机图，在这些随机图中数motif的个数
3. step3：使用统计指标衡量每一motif是否显著，用Z-score

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231215956391.png" alt="image-20221231215956391" style="zoom:80%;" />



**Z-score for statistical significance**：

$Z_i$ 捕获motif $i$ 的statistical significance：
$$
Z_i=\frac{N_i^{real}-\bar{N}_i^{rand}}{std(N_i^{rand})}
$$
其中 $N_i^{real}$ 是真实图中motif $i$ 的个数，$\bar{N}_i^{rand}$ 是随机图实例中motif $i$ 的平均个数。

0就是说真实图和随机图中motif出现的一样多。绝对值大于2时就算显著地多或者显著地少。

**network significance profile (SP)**：
$$
SP_i=\frac{Z_i}{\sqrt{\sum_jZ_j^2}}
$$
SP 是归一化的Z-score向量，其维度取决于我们考虑的motif的数量。

SP 强调subgraph的相对重要性：在比较不同大小的网络时很重要，因为一般来说，大图会出现更高的Z-score。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231220420790.png" alt="image-20221231220420790" style="zoom:80%;" />



**显著性分布（significance profile）**：

对每个subgraph，Z-score指标可用于分类subgraph significance：负数意味着under-representation，整数意味着over-representation。

network significance profile是具有所有subgraph大小上的值的feature vector。

接下来就可以比较随机图和不同图上的profile了。

**不同网络举例**：

- 基因调控网络[4](https://blog.csdn.net/PolarisRisingWar/article/details/119107608#fn4)
- 神经网络（突触连接）
- 万维网（网页超链接）
- 社交网络
- language networks (word adjacency)

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231220704592.png" alt="image-20221231220704592" style="zoom:80%;" />



**significance profile示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231220733141.png" alt="image-20221231220733141" style="zoom:80%;" />

相似领域的网络会具有相似的SP。可以通过motif frequency来判断网络功能。如社交网络中的subgraph6少、但是subgraph13多，因为一个人很难同时与两人保持紧密好友关系而这两个人不互相认识，每周还要出来喝2次咖啡。毕竟如果他们认识以后就可以一周出来只约1次咖啡了。


**检测motif总结**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231220830795.png" alt="image-20221231220830795" style="zoom:80%;" />



**motif概念的变体**：

衍生：有向/无向，colored/uncolored（应该指的是节点类型，如下图中右上角045算motif出现、345不算motif出现的情况），动态/static motif

概念上的变体：不同的frequency概念、不同的significance指标、under-representation (anti-motifs)（如下图中右下角所示）、不同的null models

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231220922148.png" alt="image-20221231220922148" style="zoom:80%;" />



**motif总结**：

- subgraph和motif是图的组成部分，子图同构和技术问题是NP-hrad。
- 理解数据集中motif的频繁或显著出现，可以使我们了解该领域的独有特征。
- 使用随机图作为null model来通过Z-score衡量motif的显著性。



## 15.3 神经子图匹配/表示

**subgraph matching**：

给出大的target graph（可以是disconnected），query graph（connected）

问题：query graph是否是target graph中的子图？

示例如下图（节点颜色表示映射关系）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231222853870.png" alt="image-20221231222853870" style="zoom:80%;" />



在本课程中我们不用combinatorial matching、逐边检查，而将该问题视作预测任务，使用机器学习方法来解决这一问题。

直觉：利用嵌入域的几何形状来捕获子图同构的属性

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231222936543.png" alt="image-20221231222936543" style="zoom:80%;" />



**task setup**：

二元预测问题：返回query是否与target graph子图同构

（注意在这里我们只关注该预测问题的最终决策，即是不是。而具体的节点之间如何一一对应的关系本课程中不讲）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223023535.png" alt="image-20221231223023535" style="zoom:80%;" />



**overview of the approach**：

整体流程如图所示：将target graph拆成很多neighborhoods，嵌入neighborhoods和query，将每个neighborhood与query做匹配，判断其是否子图同构：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223142372.png" alt="image-20221231223142372" style="zoom:80%;" />



**neural architecture for subgraphs**：

- 我们将使用node-anchored定义，用anchor的嵌入来判断是否同构：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223234813.png" alt="image-20221231223234813" style="zoom:80%;" />

- 使用node-anchored neighborhoods：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223319987.png" alt="image-20221231223319987" style="zoom:80%;" />

  用GNN基于anchor的邻居得到其嵌入，预测 $u$ 的邻居是否与 $v$ 的邻居同构（图中应该是用了二阶邻居的意思）：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223411971.png" alt="image-20221231223411971" style="zoom:80%;" />



**为什么要使用anchor呢？**

回忆node-level frequency definition。这是因为我们可以用GNN来获得 $u$ 和 $v$ 对应的嵌入，从而可以得知 $u$ 的邻居是否与 $v$ 的邻居同构，这样就可以预测是否存在anchor的映射关系并识别出对应的特定节点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223554656.png" alt="image-20221231223554656" style="zoom:80%;" />



**将 $G_T$ 分解为neighborhoods**：

对 $G_T$ 中的每个节点（准anchor），获取其k跳邻居（可以通过BFS获取）。k是一个超参，k越大，模型代价越高。

同样的过程也应用于 $G_Q$ ，同样得到neighborhoods。

我们通过GNN得到anchor node基于neighborhood的embedding，也就是得到了这些neighborhoods的嵌入。



**order embeddings space**：

将图 $A$ 映射到高维（如64维）嵌入域的点 $z_A$ ，使 $z_A$ 所有维度元素都非负。

可以捕获partial ordering（关系可传递）（具体见图）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231223907913.png" alt="image-20221231223907913" style="zoom:80%;" />

总之可以用嵌入各维元素全部小于等于的关系来表示subgraph。



**subgraph order embedding space**：

如图：在order embedding space中全部维度小于target graph的anchor嵌入的就是其subgraph的anchor嵌入（在二维嵌入域中，就是在target graph的左下角）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231224028645.png" alt="image-20221231224028645" style="zoom:80%;" />



**为什么要使用order embedding space？**

因为subgraph isomorphism relationship可以很好地在order embedding space中编码，也就是说order embedding可以在向量域表示图域中subgraph的关系：

- **transitivity**：

  图域：如果 $G_1$ 是 $G_2$ 的subgraph，$G_2$ 是 $G_3$ 的subgraph，则 $G_1$ 是 $G_3$ 的subgraph

  嵌入域：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231224317454.png" alt="image-20221231224317454" style="zoom:80%;" />

- **anti-symmetry**：

  图域：如果 $G_1$ 是 $G_2$ 的subgraph，$G_2$ 是 $G_1$ 的subgraph，则 $G_1$ 与 $G_2$ 同构。

  嵌入域：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231224459878.png" alt="image-20221231224459878" style="zoom:80%;" />

- **closure under intersection**：

  图域：1个节点的图是所有图的subgraph

  嵌入域：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231224600807.png" alt="image-20221231224600807" style="zoom:80%;" />



**order constraint**：

我们用GNN来嵌入neighborhoods并保持其order embedding结构，因此我们需要学习一种可以学习反映subgraph关系的order embedding的损失函数。

我们基于order constraint设计损失函数。order constraint规定了理想的可反映subgraph关系的order embedding属性：
$$
\forall_{i=1}^Dz_q[i]\le z_t[i] \ iff\ G_Q\sube G_T
$$
其中 $z_q$ 是query embedding，$z_u$ 是target embedding，$i$ 是embedding dimension（D应该是嵌入维度），$\sube$ 是subgraph关系。

order constraint用max-margin loss来训练。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231225905894.png" alt="image-20221231225905894" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231225943264.png" alt="image-20221231225943264" style="zoom:80%;" />



**损失函数**：

GNN嵌入通过最小化max-margin loss学得。

定义图 $G_q$ 和 $G_t$ 之间的margin（penalty或violation）：
$$
E(G_q,G_t)=\sum_{i=1}^D(max(0,z_q[i]-z_t[i]))^2
$$
当margin=0时 $z_q[i]<z_t[i]$ 恒成立，即 $G_q$ 是 $G_t$ 的subgraph；当margin＞0时 $G_q$ 不是 $G_t$ 的subgraph。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231230418320.png" alt="image-20221231230418320" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231230459967.png" alt="image-20221231230459967" style="zoom:80%;" />



**training neural subgraph matching**：

为了学习这种嵌入，需要约束训练集样本 $(G_q,G_t)$ 中一半 $G_q$ 是 $G_t$ 的subgraph，另一半不是。

- **对正样本**：最小化 $E(G_q,G_t)$ 
- **对负样本**：最小化 $max(0,\alpha-E(G_q,G_t))$ 

max-margin loss使正样本中 $z_q[i]<z_t[i]$ ，使全式为0；负样本中的 $E(G_q,G_t)$ 大于 $\alpha$ （使 $\alpha-E(G_q,G_t)<0$ 使全式为0）。但都不强化其差异，使嵌入向量之间不需要隔很远。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231231425717.png" alt="image-20221231231425717" style="zoom:80%;" />



**training example construction**：

我们需要从数据集 $G$ 中生成训练样本：query $G_Q$ 和 target $G_T$ 。

得到 $G_T$ ：随机选取anchor $v$ ，获取其全部 $K$ 阶邻居。

得到 $G_Q$：用BFS抽样，从 $G_T$ 中抽样induced subgraph：

- step1：初始化 $S=\{v \}$ ，$V=\empty$ 。
- step2： $N(S)$ 为 $S$ 中节点的所有邻居。每一步抽样10%在 $N(S)$ 中但不在 $V$ 的节点放入 $S$ 中，并将其余节点放在 $V$ 中。
- step3：$K$ 步后，获取 $G$ 的 induced by $S$ anchored at $v$ 的subgraph。

对负样本（ $G_Q$ 不是 $G_T$ 的subgraph）：corrupt $G_Q$ 增加/移动节点/边使它不再是一个subgraph。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231232419820.png" alt="image-20221231232419820" style="zoom:80%;" />



**训练细节**：

- **我们需要抽样出多少training examples？**

  每次迭代，我们都需要抽样新的target-query对。

  这样做的好处是每次模型都会看到不同的subgraph例子，提升表现结果、避免过拟合（毕竟有指数级的可能subgraph可供抽样）

- **BFS抽样应该多深？**

  这是一个需要平衡运行时间和表现结果的超参数，一般设置为3-5，也需要取决于数据集的大小。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231232706208.png" alt="image-20221231232706208" style="zoom:80%;" />



**在新图上预测一个图是否是另一个图的subgraph**：

已知：query graph $G_q$ anchored at node $q$ ，target graph $G_t$ anchored at node $t$ 。

目标：输出query是否是target的node-anchored subgraph

过程：如果 $E(G_q,G_t)<\epsilon$ ，预测为真；反之预测为假（$\epsilon$ 为超参数）

为了检验 $G_Q$ 是否与 $G_T$ subgraph isomorphism，对所有 $q\in G_Q,t\in G_T$  重复上述流程。此处的 $G_q$ 是 $q\in G_Q$ 附近的neighborhood。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231233054902.png" alt="image-20221231233054902" style="zoom:80%;" />



**neural subgraph matching总结**：

neural subgraph matching使用基于机器学习的方法学习NP-hard的subgraph isomorphism问题：

1. 将query和target图都嵌入order embedding space
2. 用这些嵌入向量计算 $E(G_q,G_t)$ 以判断query是否是target的subgraph

用order embedding space嵌入图使subgraph isomorphism可以高效表示并由图嵌入的相对位置进行检验。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231233215941.png" alt="image-20221231233215941" style="zoom:80%;" />



## 15.4 挖掘/查找频繁的Motifs/子图

**finding frequent subgraphs**:

找最频繁的大小为k的motif需要解决两个挑战：

1. 迭代所有大小为k的connected subgraph
2. 数每一类subgraph的出现次数

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231233341291.png" alt="image-20221231233341291" style="zoom:80%;" />



这个问题难在仅确定一个特定subgraph是否存在于图中就已是计算代价很高的问题（subgraph isomorphism是NP-complete），计算用时随subgraph指数级增长（因此传统方法可行的motif尺寸都相对较小，如3到7）。

可以说这是两个指数级增长的问题梦幻联动（特定大小有多少motif（组合爆炸combinatorial explosion）+找出每个motif的frequency（subgraph isomorphism and subgraph counting））

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231233443008.png" alt="image-20221231233443008" style="zoom:80%;" />



**使用表示学习的方法来解决问题**：

表示学习通过search space（每次增加一个节点，累积到size k的subgraph上，详情见下文。注意我们仅关心高频subgraph）解决组合爆炸问题，通过GNN预测解决subgraph isomorphism问题（就是本章第三节所讲述的知识）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231233535410.png" alt="image-20221231233535410" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231233911101.png" alt="image-20221231233911101" style="zoom:80%;" />



**problem setup: frequent motif mining**

给出target graph（数据集）$G_T$ ，subgraph大小参数 $k$ 。

所需结果数量 $f$ 

目标：从所有大小为 $k$ 个节点的图中，识别出 $r$ 个在 $G_T$ 中出现频率最高的图。

我们使用node-level subgraph定义。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231234056923.png" alt="image-20221231234056923" style="zoom:80%;" />



**SPMiner overview**：

**SPMiner：识别高频motifs的神经网络模型**

步骤：将输入图 $G_T$ decompose为重复的node-anchored neighborhoods，将subgraph嵌入到order embedding space（上述两步和neural subgraph matching是一样的）；然后进行search procedure，策略是不遍历所有可能的subgraph，而直接从2个节点的图开始增长出一个所需节点数的subgraph，在增长的同时尽量保证频率高。
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231234243997.png" alt="image-20221231234243997" style="zoom:80%;" />



**SPMiner：核心思想**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231234314686.png" alt="image-20221231234314686" style="zoom:80%;" />

order embedding的核心优势：可以迅速找到特定subgraph $G_Q$ 的频率

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231234351760.png" alt="image-20221231234351760" style="zoom:80%;" />



**motif频率估计**：

已知：一系列 $G_T$ 的subgraph（node-anchored neighborhoods）$G_{N_i}$ （通过随机抽样得到）。

核心思想：估计 $G_Q$ 的频率：通过数符合要求的 $G_{N_i}$ 的个数（$z_Q\le z_{N_i}$） 

这是由order embedding space的属性得到的结论：如下图所示，红色节点（motif）右上角的红色区域就是其super-graph region，红色节点是所有落在该区域的节点（ $G_T$ 的neighborhoods）的subgraph。

这样的好处就是算得快。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231234715976.png" alt="image-20221231234715976" style="zoom:80%;" />



**SPMiner search procedure**：

1. 开始：从一个从target graph中随机选择的起点 $u$ 开始：设 $S=\{u\}$

   所有neighborhoods都在一个点的右上方区域，即都包含这个subgraph。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231234850147.png" alt="image-20221231234850147" style="zoom:80%;" />

2. 迭代：每次选一个 $S$ 中节点的邻居，加到 $S$ 中，如此逐渐增长motif的尺寸。

   目标：在 $k$ 步后最大化红色阴影区域中的neighborhoods数目。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231235010705.png" alt="image-20221231235010705" style="zoom:80%;" />

3. 停止：达到预期motif尺寸后，选取the subgraph of the target graph induced by $S$ 

   我们找到的motif就是预期尺寸备选subgraph嵌入中有最多target graph neighborhoods（蓝点）在红色区域的subgraph。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231235106738.png" alt="image-20221231235106738" style="zoom:80%;" />

4. 每一步如何选取节点？

   定义subgraph $G$ 的total violation：不包含 $G$ 的neighborhoods数量。即不满足 $z_Q≼z_{N_i}$ 的neighborhoods $G_{N_i}$ 数量。

   最小化total violation就是最大化频率。

   我们采用贪心算法，每一步都选择时当前total violation最小的节点。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231235505663.png" alt="image-20221231235505663" style="zoom:80%;" />



**实验结果**：

1. **小motif**：

   ground truth：通过代价高昂的BF迭代算法（暴力破解）找到10个出现频率最高的motif。

   在大小为5-6的motif上，SPMiner可以识别出top 10中前9/8个，识别出的频率接近真实值：

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231235609580.png" alt="image-20221231235609580" style="zoom:80%;" />

2. **大motif**：

   SPMiner比baseline多识别10-100倍。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231235648835.png" alt="image-20221231235648835" style="zoom:80%;" />



## 15.5 总结

**总结**：

1. subgraph和motif是可用于深入了解图结构的重要概念，对其计数可用作节点或图的特征。
2. 本章介绍一种预测subgraph isomorphism关系的神经网络方法。
3. order embeddings的特性可用于编码subgraph关系。
4. order embedding space上的neural embedding-guided search让我们有了一种比传统方法能识别出更高motif频率的机器学习模型。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20221231235807185.png" alt="image-20221231235807185" style="zoom:80%;" />





# 16 网络中的社区检测

**本章主要内容**：

本章首先介绍了网络中社区community（或cluster / group）的概念，以及从社会学角度来证明了网络中社区结构的存在性。

接下来介绍了modularity概念衡量community识别效果。

然后介绍了Louvain算法识别网络中的community。

对于overlapping communiteis，本章介绍了BigCLAM方法来进行识别。



## 16.1 网络中的社区检测

图中的社区识别任务就是对节点进行聚类。



**networks & communities**：网络会长成图中这样：由多个内部紧密相连、互相只有很少的边连接的community组成。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101201331831.png" alt="image-20230101201331831" style="zoom:50%;" />



**从社会学角度理解这一结构**：在社交网络中，用户是被嵌入的节点，信息通过链接（长链接或短链接流动）。

以工作信息为例，Mark Granovetter 在其60年代的博士论文中提出，人们通过人际交往来获知信息，但这些交际往往在熟人（长链接）而非密友（短链接）之间进行，也就是真找到工作的信息往往来自不太亲密的熟人。这与我们所知的常识相悖，因为我们可能以为好友之间会更容易互相帮助。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101201558415.png" alt="image-20230101201558415" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101201534579.png" alt="image-20230101201534579" style="zoom:80%;" />



**Granovetter认为友谊（链接）有两种角度**：

- 一是structural视角，友谊横跨网络中的不同部分。
- 二是interpersonal视角，两人之间的友谊是强或弱的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101201655755.png" alt="image-20230101201655755" style="zoom:80%;" />



**Granovetter在一条边的social和structural视角之间找到了联系**：

1. structure：structurally/well embeded / tightly-connected 的边也socially strong，长距离的、横跨社交网络不同部分的边则socially weak。

   也就是community内部、紧凑的短边更strong，community之间、稀疏的长边更weak。

2. information：长距离边可以使人获取网络中不同部分的信息，从而得到新工作。structurally embedded边在信息获取方面则是冗余的。

   也就是说，你跟好友之间可能本来就有相近的信息源，而靠外面的熟人才可能获得更多新信息。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101202057262.png" alt="image-20230101202057262" style="zoom:80%;" />



**triadic closure**： triadic（三元; 三色系; 三色; 三重; 三合一）

community (tightly-connected cluster of nodes) 形成原因：如果两个节点拥有相同的邻居，那么它们之间也很可能有边。（如果网络中两个人拥有同一个朋友，那么它们也很可能成为朋友）

**triadic closure = high clustering coefficient**

**triadic closure产生原因**：如果B和C拥有一个共同好友A，那么B更可能遇见C（因为他们都要和A见面），B和C会更信任彼此，A也有动机带B和C见面（因为要分别维持两段关系比较难），从而B和C也更容易成为好友。

Bearman和Moody的实证研究证明，clustering coefficient低的青少年女性更容易具有自杀倾向。



多年来Granovetter的理论一直没有得到验证，但是我们现在有了大规模的真实交流数据（如email，短信，电话，Facebook等），从而可以衡量真实数据中的edge strength。

举例：数据集Onnela et al. 2007

20%欧盟国家人口的电话网络，以打电话的数量作为edge weight

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101202443796.png" alt="image-20230101202443796" style="zoom:80%;" />



**edge overlap**：两个节点 $i$ 、$j$ 之间的edge overlap，是除本身节点之外，两个节点共同邻居占总邻居的比例
$$
O_{ij}=\frac{|(N(i)\cap N(j) )-\{i,j\} |}{|(N(i)\cup N(j) )-\{i,j\}|}
$$
overlap=0 时这条边是local bridge。

当两个节点well-embeded或structurally strong时，overlap会较高。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101202832843.png" alt="image-20230101202832843" style="zoom:80%;" />



在电话网络中，edge strength（电话数）和edge overlap之间具有正相关性：图中蓝色是真实数据，红色是重新排列edge strength之后的数据（对照组）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101202951271.png" alt="image-20230101202951271" style="zoom:80%;" />

**在真实图中，更embeded（密集）的部分edge strength也更高（红）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101203626235.png" alt="image-20230101203626235" style="zoom:80%;" />

**相比之下，strength被随机shuffle之后的结果就没有这种特性**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101203700042.png" alt="image-20230101203700042" style="zoom:80%;" />

**从strength更低的边开始去除，能更快disconnect网络（相当于把community之间的边去掉）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101203736250.png" alt="image-20230101203736250" style="zoom:80%;" />

**从overlap更低的边开始，disconnect更快**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101204546287.png" alt="image-20230101204546287" style="zoom:80%;" />



从而，我们得到网络的概念图：由structurally embeded的很多部分组成，其内部链接更强，其之间链接更弱：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101204637222.png" alt="image-20230101204637222" style="zoom:80%;" />



## 16.2 网络社区

network communites就是这些部分（也叫cluster，group，module）：一系列节点，其内部有很多链接，与网络其他部分的外部链接很少

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101204713163.png" alt="image-20230101204713163" style="zoom:80%;" />



我们的目标就是给定一个网络，由算法自动找到网络中的**communities（densely connected groups of nodes）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101204936881.png" alt="image-20230101204936881" style="zoom:80%;" />



以Zachary’s Karate club network来说，通过社交关系创造的图就可以正确预判出成员冲突后会选择哪一边：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101205302755.png" alt="image-20230101205302755" style="zoom:80%;" />



在付费搜索领域中，也可以通过社区识别来发现微众市场：举例来说，节点是advertiser和query/keyword，边是advertiser在该关键词上做广告。在赌博关键词中我们可以专门找到sporting betting这一小社区（微众市场）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101205450853.png" alt="image-20230101205450853" style="zoom:80%;" />



**NCAA Football Network**：节点是球队，边是一起打过比赛。通过社区识别算法也可以以较高的准确度将球队划分到不同的会议中

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101205536997.png" alt="image-20230101205536997" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101205648972.png" alt="image-20230101205648972" style="zoom:80%;" />



定义**modularity** $Q$ 衡量网络的一个社区划分partitioning（将节点划分到不同社区）的好坏程度：

已知一个划分，网络被划分到disjoint groups $s\in S$ 中
$$
Q\propto \sum_{s\in S}[(\#edges\ within\ group\ s)-(expected\ \#edges\ within\ group\ s)]
$$
其中expected # edges within group s需要一个null model

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101210215891.png" alt="image-20230101210215891" style="zoom:80%;" />



**null model: configuration model**

通过configuration model生成的 $G'$ 是个multigraph（即存在重边，即节点对间可能存在多个边）。

真实图共 $n$ 个节点，$m$ 条边。在 $G'$ 中，节点 $i$ （度数为 $k_i$）和节点 $j$ （度数为 $k_j$）之间的期望边数为 $k_i\cdot\frac{k_j}{2m}=\frac{k_ik_j}{2m}$ 。

（一共有 $2m$ 个有向边，所以一共有 $2m$ 个节点上的spoke。相当于对节点 $i$ 的 $k_i$ 个spoke每次做随机选择，每次选到 $j$ 上的spoke的概率为 $\frac{k_j}{2m}$ ）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101212457685.png" alt="image-20230101212457685" style="zoom:80%;" />

则 $G'$ 中的理想边数为：
$$
=\frac12\sum_{i\in N}\sum_{j\in N}\frac{k_ik_j}{2m}\\
=\frac12\cdot \frac1{2m}\sum_{i\in N} k_i(\sum_{j\in N}k_j)\\
=\frac1{4m}2m\cdot 2m\\
=m
$$
所以在null model中，真实图中的度数分布和总边数都得到了保留。

（注意：这一模型可应用于有权和无权网络。对有权网络就是用有权度数（边权求和））

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101213106966.png" alt="image-20230101213106966" style="zoom:80%;" />



**所以modularity**
$$
Q(G,S)=\frac1{2m}\sum_{s\in S}\sum_{i\in s}\sum_{j\in s}(A_{ij}-\frac{k_ik_j}{2m})\\
A_{ij}=\begin{cases}
1,if\ i\rightarrow j\\
0,otherwise
\end{cases}
$$
其中 $\frac1{2m}$ 是归一化常数，用于使 $-1\le Q\le1$ 如果 $G$ 有权，$A_{ij}$ 就是边权。

如果组内边数超过期望值，modularity就为正。当 $q$ 大于0.3-0.7时意味着这是个显著的社区结构。

**注意modularity概念同时可应用于有权和无权的网络**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101214408556.png" alt="image-20230101214408556" style="zoom:80%;" />



**modularity也可以写作**：
$$
Q=\frac{1}{2m}\sum_{ij}[A_{ij}-\frac{k_ik_j}{2m}]\delta(c_i,c_j)
$$
其中 $A_{ij}$ 是节点 $i$ 和 $j$ 之间的边权，$k_i$ 和 $k_j$ 分别是节点 $i$ 和 $j$ 入度，$2m$ 是图中边权总和，$c_i$ 和 $c_j$ 是节点所属社区，$\delta(c_i,c_j)$ 在 $c_i=c_j$ 时为1，其他为 0 。



**我们可以通过最大化modularity来识别网络中的社区**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101214819815.png" alt="image-20230101214819815" style="zoom:80%;" />



## 16.3 Louvain Algorithm

**Louvain Algorithm**：是用于社区发现的贪心算法，$O(nlogn)$ 运行时间。

支持带权图和hierarchical社区（就可以形成如图所示的树状图dendrogram）。

广泛应用于研究大型网络，快，迅速收敛，输出的modularity高（也就意味着识别出的社区更好）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101215013112.png" alt="image-20230101215013112" style="zoom:80%;" />



**Louvain算法每一步分成两个阶段**：

- 第一阶段：仅通过节点-社区隶属关系的局部改变来优化modularity。
- 第二阶段：将第一步识别出来的communities聚合为super-nodes，从而得到一个新网络。
- 返回第一阶段，重复直至modularity不可能提升。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101215126391.png" alt="image-20230101215126391" style="zoom:80%;" />



**Louvain算法将图视作带权的**。

原始图可以是无权的（例如边权全为1）。

随着社区被识别并聚合为super-nodes，算法就创造出了带权图（边权是原图中的边数）。

在计算过程中应用带权版本的modularity。



**phase1 (partitioning)**：

首先将图中每个节点放到一个单独社区中（每个社区只有一个节点）。

对每个节点 $i$ ，计算其放到邻居 $j$ 所属节点中的 $\Delta Q$ ，将其放到 $\Delta Q$ 最大的 $j$ 所属的社区中。

phase1运行至调整节点社区无法得到modularity的提升（local maxima）。

注意在这个算法中，节点顺序会影响结果，但实验证明节点顺序对最终获得的总的modularity没有显著影响。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101215433212.png" alt="image-20230101215433212" style="zoom:80%;" />



**modularity gain**：

将节点 $i$ 从社区 $D$ 转移到 $C$ 的modularity gain：
$$
\Delta Q(D\rightarrow i\rightarrow C)=\Delta Q(D\rightarrow i)+\Delta Q(i\rightarrow C)
$$

- $\Delta Q(D\rightarrow i)$ ：将 $i$ 从 $D$ 中移出，单独作为一个社区。
- $\Delta Q(i\rightarrow C)$ ：将 $i$ 从单独一个社区放入 $C$ 中。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101220429043.png" alt="image-20230101220429043" style="zoom:80%;" />



接下来我们计算：$\Delta Q(i\rightarrow C)$

我们首先计算modularity winthin $C$ 的 $Q(C)$ ：

**定义**：

- $\sum_{in}=\sum_{i,j\in C}A_{ij}$ ：$C$ 中节点间的边权总和。
- $\sum_{out}=\sum_{i\in C}k_i$ ：$C$ 中节点的边权总和。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101220746658.png" alt="image-20230101220746658" style="zoom:80%;" />

**从而得到**：
$$
Q(C)=\frac1{2m}\sum_{i,j\in C}[A_{ij}-\frac{k_ik_j}{2m}]\\
=\frac{\sum_{i,j\in C}A_{ij}}{2m}-\frac{(\sum_{i\in C}k_i)(\sum_{j\in C}k_j)}{(2m)^2}\\
=\frac{\sum_{in}}{2m}-(\frac{\sum_{tot}}{2m})^2
$$
当绝大多数链接都在社区内部时，$Q(C)$ 大。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101221212019.png" alt="image-20230101221212019" style="zoom:80%;" />

接下来我们定义：

- $k_{i,in}=\sum_{j\in C}A_{ij}+\sum_{j\in C}A_{ji}$ ：节点 $i$ 和社区 $c$ 间边权加总。
- $k_i$ ：节点 $i$ 边权加总（如度数）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101221459444.png" alt="image-20230101221459444" style="zoom:80%;" />

- $i$ 放入 $C$ 前：
  $$
  Q_{before}=Q(C)+Q(\{i\})=[\frac{\sum_{in}}{2m}-(\frac{\sum_{tot}}{2m})^2]+[0-(\frac{k_i}{2m})^2]
  $$

- $i$ 放入 $C$ 后：
  $$
  Q_{after}=Q(C+\{i\})=\frac{\sum_{in}+k_{i,in}}{2m}-(\frac{\sum_{tot}+k_i}{2m})^2
  $$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101222032252.png" alt="image-20230101222032252" style="zoom:80%;" />



**从而得到总的modularity gain**：
$$
\Delta Q(i\rightarrow C)=Q_{after}-Q_{before}=[\frac{\sum_{in}+k_{i,in}}{2m}-(\frac{\sum_{tot}+k_i}{2m})^2]-[\frac{\sum_{in}}{2m}-(\frac{\sum_{tot}}{2m})^2-(\frac{k_i}{2m})^2]
$$
类似地，可以得到 $\Delta Q(D\rightarrow i)$ ，从而得到总的 $\Delta Q(D\rightarrow i\rightarrow C)$ 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101222439157.png" alt="image-20230101222439157" style="zoom:80%;" />



**phase1总结**：

迭代直至没有节点可以移动到新社区：

对每个节点，计算其最优社区 $C'$ ：$C'=\arg max_{C'}\Delta Q(C\rightarrow i \rightarrow C')$

如果 $\Delta Q(C\rightarrow i\rightarrow C')>0$ ，更新社区：

$C\leftarrow C-\{i\}$ 

$C'\leftarrow C'+\{i\}$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101222849396.png" alt="image-20230101222849396" style="zoom:80%;" />



**phase2 (restructing)**：

将通过上一步得到的社区收缩到super-nodes，创建新网络，如果原社区之间就有节点相连，就在对应super-nodes之间建边，边权是原对应社区之间的所有边权加总。

然后在该super-node网络上重新运行phase1。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101223624837.png" alt="image-20230101223624837" style="zoom:80%;" />



**图示如下**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101223754574.png" alt="image-20230101223754574" style="zoom:80%;" />



**Louvain Algorithm整体图示**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101223810264.png" alt="image-20230101223810264" style="zoom:80%;" />



**在Belgian mobile phone network中，通过Louvain algorithm，可以有效划分出法语和德语社区**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101223901547.png" alt="image-20230101223901547" style="zoom:80%;" />



**本节总结**：

1. modularity：对将图划分到社区的partitioning的质量的评估指标，用于决定社区数
2. Louvain modularity maximization：贪心策略，表现好，能scale到大网络上

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101224038078.png" alt="image-20230101224038078" style="zoom:80%;" />



## 16.4 检测重叠社区：BigCLAM

**overlapping communities**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101224157049.png" alt="image-20230101224157049" style="zoom:80%;" />



**Facebook Ego-network**：

其节点是用户，边是友谊

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101224321005.png" alt="image-20230101224321005" style="zoom:80%;" />

在该网络中社区之间有重叠。通过BigCLAM算法，我们可以直接通过没有任何背景知识的网络正确分类出很多节点（图中实心节点是分类正确的节点）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101224814319.png" alt="image-20230101224814319" style="zoom:80%;" />



**protein-protein interactions**：

在protein-protein interactions网络中，节点是蛋白质，边是interaction。其functional modules在网络中也是重叠的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101224931547.png" alt="image-20230101224931547" style="zoom:80%;" />



![image-20230101224954042](C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101224954042.png)



**重叠与非重叠的社区在图中和在邻接矩阵中的图示**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101225031445.png" alt="image-20230101225031445" style="zoom:80%;" />



**BigCLAM方法流程**：

- 第一步：定义一个基于节点-社区隶属关系生成图的模型（community affiliation graph model (AGM)）
- 第二步：给定图，假设其由AGM生成，找到最可能生成该图的AGM

通过这种方式，我们就能发现图中的社区。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101225421681.png" alt="image-20230101225421681" style="zoom:80%;" />



**community affiliation graph model (AGM)**：

通过节点-社区隶属关系（下图左图）生成相应的网络（下图右图）。

参数为节点 $V$ ，社区 $C$ ，成员关系 $M$ ，每个社区 $c$ 有个概率 $p_c$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101225715314.png" alt="image-20230101225715314" style="zoom:80%;" />



**AGM生成图的流程**：

给定参数 $(V,C,M,\{p_c\})$ ，每个社区 $c$ 内的节点以概率 $p_c$ 互相链接。

对同属于多个社区的节点对，其相连概率就是：
$$
p(u,v)=1-\prod_{c\in M_u\cap M_v}(1-p_c)
$$
（注意：如果节点 $u$ 和 $v$ 没有共同社区，其相连概率就是0。为解决这一问题，我们会设置一个 background “epsilon” 社区，每个节点都属于该社区，这样每个节点都有概率相连）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101230432814.png" alt="image-20230101230432814" style="zoom:80%;" />



**AGM可以生成稠密重叠的社区**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101230619245.png" alt="image-20230101230619245" style="zoom:80%;" />



**AGM有弹性，可以生成各种社区结构：non-overlapping, overlapping, nested**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101230657739.png" alt="image-20230101230657739" style="zoom:80%;" />



**通过AGM发现社区：给定图，找到最可能产生出该图的AGM模型参数（最大似然估计）**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101230728273.png" alt="image-20230101230728273" style="zoom:80%;" />



**graph fitting**

我们需要得到 $F=\arg max_F P(G|F)$

为解决这一问题，我们需要高效计算 $P(G|F)$ ，优化 $F$ （如通过梯度下降）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101230847824.png" alt="image-20230101230847824" style="zoom:80%;" />



graph likelihood $P(G|F)$ 

通过 $F$ 得到边产生概率的矩阵 $P(u,v)$ ，$G$ 具有邻接矩阵，从而得到 $P(G|F)=\prod_{(u,v)\in G}P(u,v)\prod_{(u,v)\notin G}(1-P(u,v))$

即通过AGM产生的图，原图中边存在、非边不存在的概率连乘，得到图就是原图的概率。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101231232065.png" alt="image-20230101231232065" style="zoom:80%;" />



“Relaxing” AGM: Towards $P(u,v)$

成员关系具有strength（如图所示）：$F_{uA}$ 是节点 $u$ 属于社区 $A$ 的成员关系的strength（如果 $F_{uA}=0$ ，说明没有成员关系）（strength非负）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101231438780.png" alt="image-20230101231438780" style="zoom:80%;" />

对社区 $C$ ，其内部节点链接概率为：$P_c(u,v)=1-exp(-F_{uC}\cdot F_{vC})$ ，$0\le P_c(u,v)\le1$ （是个valid probability）

$P_c(u,v)=0$ （节点之间无链接） 当且仅当 $F_{uC}\cdot F_{vC}=0$ （至少有一个节点对 $C$ 没有成员关系）

$P_c(u,v)\approx 0$ （节点之间有链接） 当且仅当 $F_{uC}\cdot F_{vC}$ 很大（两个节点都对 $C$ 有强成员关系strength）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101231917829.png" alt="image-20230101231917829" style="zoom:80%;" />



节点对之间可以通过多个社区相连，在至少一个社区中相连，节点对就相连：$P(u,v)=1-\prod_{C\in T} (1-P_c(u,v))$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101232028559.png" alt="image-20230101232028559" style="zoom:80%;" />



<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101232051640.png" alt="image-20230101232051640" style="zoom:80%;" />

（$F_u$ 是 $\{F_{uC}\}_{C\in T}$ 的向量）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101232221507.png" alt="image-20230101232221507" style="zoom:80%;" />



**BigCLAM model**：
$$
P(G|F)=\prod_{(u,v)\in E}P(u,v)\prod_{(u,v)\notin E}(1-P(u,v))\\
=\prod_{(u,v)\in E}(1-exp(-F_u^TF_v))\prod_{(u,v)\notin E}exp(-F_u^TF_v)
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101232528756.png" alt="image-20230101232528756" style="zoom:80%;" />

但是直接用概率的话，其值就是很多小概率相乘，数字小会导致numerically unstable的问题，所以要用log likelihood（因为log是单调函数，所以可以）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101232601818.png" alt="image-20230101232601818" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101232618229.png" alt="image-20230101232618229" style="zoom:80%;" />



优化 $l(F)$ ：从随机成员关系 $F$ 开始，迭代直至收敛：

对每个节点 $u$ ，固定其它节点的membership、更新 $F_u$ 。我们使用梯度提升的方法，每次对 $F_u$ 做微小修改，使log-likelihood提升。

对 $F_u$ 的偏微分：
$$
∇l(F)=\sum_{v\in N(u)}(\frac{exp(-F_u^TF_v)}{1-exp(-F_u^TF_v)})\cdot F_v-\sum_{v\notin N(u)}F_v
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101233107031.png" alt="image-20230101233107031" style="zoom:80%;" />

在梯度提升的过程中，$\sum_{v\in N(u)}\frac{exp(-F_u^TF_v)}{1-exp(-F_u^TF_v)}\cdot F_v$ 与 $u$ 的度数线性相关（快），$\sum_{v\notin N(u)}F_v$ 与图中节点数线性相关（慢）。

因此我们将后者进行分解：$\sum_{v\notin N(u)}F_v=\sum_vF_v-F_u-\sum_{v\in N(u)}F_v$ 

右式第一项可以提前计算并cache好，每次直接用；后两项与 $u$ 的度数线性相关。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101233455054.png" alt="image-20230101233455054" style="zoom:80%;" />



**BigCLAM总结**：

BigCLAM定义了一个模型，可生成重叠社区结构的网络。给定图，BigCLAM的参数（每个节点的membership strength）可以通过最大化对数似然估计得到。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230101233528527.png" alt="image-20230101233528527" style="zoom:80%;" />





# 17 图的传统生成模型

**本章主要内容**：

本章首先介绍了图生成模型generative models for graphs的基本概念和意义。

接下来介绍了一些真实世界网络的属性（度数分布、聚集系数、connected component、path length等，可参考）（也是图生成模型希望可以达到的要求）。

最后介绍了一些传统的图生成模型（Erdös-Renyi graphs, small-world graphs, Kronecker graphs）。



## 17.1 图的（传统）生成模型

**图生成模型问题的研究动机**：

我们此前的学习过程中，都假设图是已知的；但我们也会想通过graph generative model人工生成与真实图类似的synthetic graph，这可以让我们

1. 了解图的形成过程。
2. 预测图的演化。
3. 生成新的图实例。
4. 异常检测：检测一个图是否异常。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102145101709.png" alt="image-20230102145101709" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102145320646.png" alt="image-20230102145320646" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102145344246.png" alt="image-20230102145344246" style="zoom:80%;" />



**本课程对图生成模型的介绍流程**：

在本章介绍真实图的属性（生成图需要符合的特性）和传统图生成模型（每种模型都源自对图形成过程的不同假设）。

在下一章介绍深度图生成模型，从原始数据直接学习到图生成过程。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102145501708.png" alt="image-20230102145501708" style="zoom:80%;" />



## 17.2 真实世界图的性质

**衡量真实图数据的属性有**：

- degree distribution
- clustering coefficient
- connected components
- path length

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102145613984.png" alt="image-20230102145613984" style="zoom:50%;" />



**degree distribution**：

$P(k)$ ：一个随机节点度数为 $k$ 的概率

用 $N_k$ 表示度数为 $k$ 的节点数，则 $P(k)=\frac{N_k}{N}$

相当于节点度数的归一化直方图：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102145753202.png" alt="image-20230102145753202" style="zoom:80%;" />



**clustering coefficient衡量节点邻居的连接紧密程度**：

节点 $i$ 的度数为 $k_i$ ，邻居间边数为 $e_i$ ，则其clustering coefficient为
$$
C_i=\frac{e_i}{C_{k_i}^2}=\frac{2e_i}{k_i(k_i-1)}
$$
即实际存在的邻居上的边数占所有邻居上可能存在的边数 $C_{k_i}^2$ 的比例。大小范围为 $[0,1]$ 。

整个图上的clustering coefficient就是对每个节点的clustering coefficient取平均
$$
C=\frac1N\sum_i^NC_i
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102150117925.png" alt="image-20230102150117925" style="zoom:80%;" />



**connectivity**是largest connected component（任意两个节点都有路径相连的最大子图）的大小。

largest component=giant component

找到connected components（连通分量）的方法：随机选取节点跑BFS，标记所有被访问到的节点；如果所有节点都能访问到，说明整个网络都是连通的；否则就选一个没有访问过的节点重复BFS过程。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102150335313.png" alt="image-20230102150335313" style="zoom:80%;" />



path length：一条路径的长度。

节点对之间最短路径长度称为距离。

diameter：图中最大的节点对间最短路径。

connected graph 或 strongly connected directed graph 上的average path length：
$$
\bar{h}=\frac1{2E_{max}}\sum_{i,j\ne i}h_{ij}
$$
其中 $h_{ij}$ 是节点 $i$ 到 $j$ 之间的距离，$E_{max}$ 是最大边数（即节点对数）$\frac{n(n-1)}{2}$ 

我们往往只在相连的节点对上求平均，即忽略infinite的路径长度。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102150748186.png" alt="image-20230102150748186" style="zoom:80%;" />



**案例研究：MSN Graph（社交网络）**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102150813957.png" alt="image-20230102150813957" style="zoom:80%;" />

- **degree distribution**：

  用线性坐标轴绘制就基本啥都看不清

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102152358798.png" alt="image-20230102152358798" style="zoom:80%;" />

  用log-log双对数坐标绘制（数据不变，但是坐标轴换成对数的）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102152435847.png" alt="image-20230102152435847" style="zoom:80%;" />

- **clustering coefficient按度数分布**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102152822844.png" alt="image-20230102152822844" style="zoom:80%;" />

- **weakly connected component大小的分布**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102153033363.png" alt="image-20230102153033363" style="zoom:80%;" />

- **path length的分布**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102153144241.png" alt="image-20230102153144241" style="zoom:80%;" />

  small world phenomenon：虽然图很大但是平均最短路径很小（6.6）。
  随机选择一个节点，90%节点都可以在8跳BFS内达到。



**这些核心属性在图上的最终结果**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102153251782.png" alt="image-20230102153251782" style="zoom:80%;" />

这些值是否超乎预料，需要通过图生成模型来检验。



## 17.3 Erdös-Renyi 随机图

**Erdös-Renyi Random Graphs是最简单的图生成模型，有两种变体**：

- $G_{np}$：有 $n$ 个节点的无向图，每条边 $(u,v)$ 以概率 $p$ 独立同分布生成。
- $G_{nm}$：有 $n$ 个节点的无向图，随机生成 $m$ 条边。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102153524379.png" alt="image-20230102153524379" style="zoom:80%;" />



$G_{np}$：图由随机过程生成，因此同样的 $n$ 和 $p$ 可以不同的图实例

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102153622385.png" alt="image-20230102153622385" style="zoom:80%;" />



$G_{np}$ 的图属性值：

- **degree distribution**：

  一个节点的度数为 $k$ 的概率，即图中度数为 $k$ 的节点所占比例的期望值服从二项分布
  $$
  P(k)=C_{n-1}^{k}p^k(1-p)^{n-1-k}
  $$
  （在除这个节点之外的 $n-1$ 个节点中有 $k$ 个节点与该节点以 $p$ 的概率相连，这一事件发生的概率）
  $$
  \bar{k}=p(n-1)\\
  \sigma^2=p(1-p)(n-1)
  $$
  ER随机图的度数分布类似于一个高斯分布的离散模拟。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102154012673.png" alt="image-20230102154012673" style="zoom:80%;" />

- **clustering coefficient**：

  每个节点邻居中边数的期望值为
  $$
  E(e_i)=p\cdot C_{k_i}^2=p\cdot\frac{k_i(k_i-1)}{2}
  $$
  其中 $e_i$ 是节点 $i$ 邻居节点之间的边数，$k_i$ 是节点 $i$ 的度数。

  节点 $i$ 期望的clustering coefficient
  $$
  E(C_i)=\frac{2E(e_i)}{k_i(k_i-1)}=\frac{p\cdot k_i(k_i-1)}{k_i(k_i-1)}=p=\frac{\bar{k}}{n-1}\approx \frac{\bar{k}}{n}
  $$
  random graph的clustering coefficient很小，如果我们保持平均度数 $k$ 不变、增大图尺寸（指固定 $p=k\cdot\frac1n$ ），$C$ 会随图尺寸 $n$ 减小。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102154930347.png" alt="image-20230102154930347" style="zoom:80%;" />

- **connected components**：

  随着 $p$ 从0到1变化，CC会出现如图中数轴所示的变化情况。

  $\bar{k}=\frac{2E}{n}$ ，$p=\frac{\bar{k}}{n-1}$  

  - $\bar{k}=1-\epsilon$ 时所有CC的尺寸都是 $\ohm(logn)$ 
  - $\bar{k}=1+\epsilon$ 时出现一个 $\ohm(n)$ 的CC，其他CC的尺寸还是 $\ohm(logn)$ ，每一个节点都至少期望有一条边

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102164032838.png" alt="image-20230102164032838" style="zoom:80%;" />

  这种平均度数在1上下会突然出现largest connected component的转变被称为phase transition behavior，如图所示，平均度数达到3的时候已经几乎所有节点都属于largest connected component了：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102164115451.png" alt="image-20230102164115451" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102164217275.png" alt="image-20230102164217275" style="zoom:80%;" />



**定义图 $G(V,E)$ 上的概念expansion** $\alpha$ ：对任意节点子集 $S$ ，伸出 $S$ 的边数（如图所示，指 $S$ 和 $V- S$ 之间的边）大于等于 $\alpha\cdot min(|S|,|V-S|)$ （这个 $min$ 只是考虑到 $|S|$ 超过 $\frac12|V|$ 的可能性，如果 $|S|$ 是小部分的话，可以直接大于等于 $\alpha|S|$ ）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102165218323.png" alt="image-20230102165218323" style="zoom:80%;" />

expansion是用来衡量鲁棒性的：为了disconnect $l$ 个节点（让一个CC中 $l$ 个节点不再属于这个CC），需要割断至少 $\alpha \cdot l$ 条边。

（为什么是 $l$ 而不是 $min(\,n-l)$ 呢，因为 $n-l$ 要是比 $l$ 还小这就不太对劲了，就不是这 $l$ 个节点被disconnect了而是对面被disconnect了对吧……在上文也说了一般这部分是小部分，所以可以直接用 $l$ 的）

如图所示，expansion越低的图越容易被disconnect。而社交网络就是在社区内部expansion高，在社区之间expansion低：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102165538107.png" alt="image-20230102165538107" style="zoom:80%;" />



**random graphs的expansion**：

事实：对于一个有 $n$ 个节点，expansion为 $\alpha$ 的图，节点对间存在长度为 $O((logn)/\alpha)$ 的路径。

对随机图 $G_{np}$ ：对 $logn>np>c$ ，$diam(G_{np})=O(logn/lognp)$ 

如图所示，随机图有很好的expansion，所以BFS需要经对数级步数访问所有节点：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102165854689.png" alt="image-20230102165854689" style="zoom:80%;" />



如果我们固定 $\bar{k}=np$ ，我们就可以得到图节点间最长的最短路径 $diam(G_{np})=O(logn)$ ，Erdös-Renyi Random Graphs可以让节点在迅速增加时，shortest path length仍然增长很慢，如图所示：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170016713.png" alt="image-20230102170016713" style="zoom:80%;" />



得到 $G_{np}$ 上的所有属性后，与MSN的属性对比，发现：

MSN的degree distribution很偏，但 $G_{np}$ 是高斯分布。不相似。

两种图的avg. path length都很短。相似。

$G_{np}$ 的聚集系数远小于MSN，失去了局部结构。不相似。

MSN绝大多数节点都属于GCC，$G_{np}$ 在 $\bar{k}>1$ （示例中约等于14）时也存在GCC。在存在GCC方面相似，但真实图中的giant component并不通过phase transition出现。

可见真实世界的图不是随机图。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170245192.png" alt="image-20230102170245192" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170301723.png" alt="image-20230102170301723" style="zoom:80%;" />



## 17.4 小世界模型

发明小世界模型的动机：我们有高聚集系数、高diameter的regular lattice graph，也有低聚集系数、低diameter的 $G_{np}$ 随机图，但真实图是低diameter、高聚集系数的，我们希望找到一个能生成这种图的模型。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170439506.png" alt="image-20230102170439506" style="zoom:80%;" />



**在同节点数、同平均度数的随机图的对比下，各种真实图都展示出了类似的特质**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170508088.png" alt="image-20230102170508088" style="zoom:80%;" />



**在随机图和regular lattice graph中，这两个属性间却存在着矛盾**：

由于expansion的缘故，在固定平均度数时，随机图中的short paths为 $O(logn)$ 长但聚集系数也很低。

而有局部结构的网络regular lattice graph有很多社交网络中常有的triadic closure（我朋友的朋友还是我的朋友），即高聚集系数，但diameter也很高。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170622603.png" alt="image-20230102170622603" style="zoom:80%;" />



**我们希望在两种图间进行插值，得到结合二者特性，高聚集系数、低diameter的small-world graph**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170654366.png" alt="image-20230102170654366" style="zoom:80%;" />



**small-world model 的方法**：

1. 从一个低维regular lattice（这里表示成ring）开始，这个图有很高的聚集系数和diameter。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170802125.png" alt="image-20230102170802125" style="zoom:80%;" />

2. rewire：新增随机捷径，将本来较远的部分连接起来

   对每个边，以 $p$ 的概率将一端移到一个随机节点上

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170903161.png" alt="image-20230102170903161" style="zoom:80%;" />

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102170920089.png" alt="image-20230102170920089" style="zoom:80%;" />



在下图中绿色箭头指向的区域，就是小世界模型适宜的参数区域：（能够得到这个合适区域的直觉理解：需要很多随机性才能破坏聚集系数，但仅需一点随机性就能产生捷径，所以就能得到高聚集系数、低diameter的中间的图）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171023283.png" alt="image-20230102171023283" style="zoom:80%;" />



**总结**：小世界模型提供了一个在clustering和small-world（指diameter小）之间交互的视角，捕获到了真实图的结构，解释了真实图中的高聚集系数，但其度数分布仍然并不符合真实图的情况。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171132088.png" alt="image-20230102171132088" style="zoom:80%;" />



## 17.5 Kronecker 图模型

**Kronecker Graph Model的idea：迭代式的图生成**

self-similarity：物体自身总是与其部分相似。我们模仿图/社区的迭代式增长，如图所示不断重复同样的图的生成过程。

Kronecker product克罗内积就是一种生成self-similar矩阵的方式。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171227720.png" alt="image-20230102171227720" style="zoom:80%;" />



**Kronecker graph**：从小矩阵 $(K)$ 开始，通过克罗内积 $\otimes$ 生成大的邻接矩阵

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171324769.png" alt="image-20230102171324769" style="zoom:80%;" />



**克罗内积定义**：矩阵 $A$ 和 $B$ 的克罗内积

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171414771.png" alt="image-20230102171414771" style="zoom:80%;" />

图的克罗内积是对两个图的邻接矩阵求克罗内积，以其值作为新图的邻接矩阵。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171445679.png" alt="image-20230102171445679" style="zoom:80%;" />



Kronecker graph就通过对initiator matrix $K_1$ 连续迭代做克罗内积，逐渐增长式得到最终结果（也可以用多个尺寸、元素都可以不同的initiator matrics）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171604649.png" alt="image-20230102171604649" style="zoom:80%;" />



**Kronecker initiator matrices示例**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102171643614.png" alt="image-20230102171643614" style="zoom:80%;" />



**stochastic Kronecker graphs**：

步骤如图所示

1. 创建尺寸为 $[N_1,N_1]$ 的probability matrix $\Theta_1$ 
2. 计算得到 $k$ 阶克罗内积 $\Theta_k$ 
3. 生成实例矩阵 $K_k$ ：对 $\Theta_k$ 中的每个元素 $p_{uv}$ ，按概率选择是否生成 $K_k$ 对应的元素1（也就是按概率 $p_{uv}$ 生成对应的边 $(u,v)$ ）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102172319800.png" alt="image-20230102172319800" style="zoom:80%;" />



**Generation of Kronecker Graphs**：

根据上面提到的方法，如果想要生成一个有向的stochastic Kronecker graph，需要计算 $n^2$ 次概率，太慢。

利用Kronecker graphs的递归特性，我们还有一种更快的方法ball dropping / edge dropping：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102172431894.png" alt="image-20230102172431894" style="zoom:80%;" />



对Kronecker图的概率的理解，如图所示，既可以按照中间那种算出每一个元素的概率然后计算每一条边是否存在，也可以理解为右边那种形式，就很多层，每一层都是原 $\Theta$ ，最底层就是每个元素就是最终Kronecker graph的元素：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102172533260.png" alt="image-20230102172533260" style="zoom:80%;" />

快速Kronecker generator algorithm：从一个尺寸为 $[2,2]$ 的矩阵开始，迭代 $m$ 次，就能得到一个节点数为 $n=2^m$ 的图 $G$ 。



如图所示，一种比较快的方式是按层（从大象限到小象限）依次按概率选择一个象限，$a\rightarrow d\rightarrow a/b/c/d$ ，像这样选到最底层的一个格子，即邻接矩阵的一个元素、图的一条边，就相当于按照原来的边生成概率来建立新的边。

这样的问题就是可能会出现边冲突，也就是多次选择到了同一个格子。发生这种情况就直接忽略并再选一次。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102172743781.png" alt="image-20230102172743781" style="zoom:80%;" />



**Fast Kronecker generator algorithm中插入一条有向边的算法**：

1. 对矩阵进行归一化（就让它的元素总和为1，也就是变成概率矩阵）。
2. 逐层根据概率选择象限并挪动对应的坐标。在选到最后一层时就把这条边添上。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102172837535.png" alt="image-20230102172837535" style="zoom:80%;" />



在Epinions图上的估计结果：如图所示，用 $\Theta_1$ 作为initiator matrix生成的Kronecker graph在各项属性上都与真实图的类似：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102172949903.png" alt="image-20230102172949903" style="zoom:80%;" />



## 17.6 总结

介绍了传统的图生成模型，每种模型都对图生成过程提出了不同的先验假设。

下一章将介绍直接从原始数据学习图生成过程的深度图生成模型。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230102173035873.png" alt="image-20230102173035873" style="zoom:80%;" />





# 18 Colab4





# 19 图的深层生成模型

**本章主要内容**：

首先介绍了深度图生成模型的基本情况，然后介绍了直接从图数据集中学习的GraphRNN模型，最后介绍了医药生成领域的GCPN模型。



## 19.1 图的深层生成模型

对深度图生成模型，有两种看待问题的视角：

- 第一种是说，图生成任务很重要，我们此前已经学习过传统图生成模型，接下来将介绍在图表示学习框架下如何用深度学习的方法来实现图生成任务。
- 另一种视角是将其视为图表示学习任务的反方向任务。



课程此前学习过的图表示学习任务 deep graph encoders：输入图数据，经图神经网络输出节点嵌入：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104210202544.png" alt="image-20230104210202544" style="zoom:80%;" />

而深度图生成模型可以说是deep graph decoders：输入little noise parameter或别的类似东西，输出图结构数据

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104210300024.png" alt="image-20230104210300024" style="zoom:80%;" />



## 19.2 用于图生成的机器学习

**图生成任务分为两种**：

- **realistic graph generation**：生成与给定的一系列图相似的图
- **goal-directed graph generation**：生成优化特定目标或约束的图（举例：生成/优化药物分子）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104210529243.png" alt="image-20230104210529243" style="zoom:80%;" />



**图生成模型**：给定一系列图（抽样自一个冥冥中注定的数据分布 $p_{data}(G)$ ）

**目标**：

1. 学到分布 $p_{momdel}(G)$ 
2. 从 $p_{model}(G)$ 中抽样，得到新的图

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104210736864.png" alt="image-20230104210736864" style="zoom:80%;" />



**生成模型基础**：

我们想从一系列数据点（如图数据）$\{x_i\}$ 中学到一个生成模型：

- $p_{data}(x)$ 是数据分布，不可知，但我们已经抽样出了 $x_i\sim p_{data}(x)$ 
- $p_{model}(x;\theta)$ 是模型，以 $\theta$ 为参数，用于近似 $p_{data}(x)$

**学习目标**：

1. density estimation: 使 $p_{model}(x;\theta)$ 近似 $p_{data}(x)$ 
2. sampling: 从 $p_{model}(x;\theta)$ 中抽样，生成数据（图）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104211711636.png" alt="image-20230104211711636" style="zoom:80%;" />



**density estimation**：

使 $p_{model}(x;\theta)$ 近似 $p_{data}(x)$ 

主要原则：极大似然（建模分布的基本方法）
$$
\theta^*=\arg \max_{\theta} E_{x\sim p_{data}}\log p_{model}(x;\theta)
$$
即找到使被观察到的数据点 $x_i\sim p_{data}$ 最有可能在 $p_{model}$ 下生成（即 $\prod_{i}p_{model}(x_i;\theta^*)$ 最大，即 $\log \prod_{i}p_{model}(x_i;\theta^*)$ 最大，即 $\sum_{i}\log p_{model}(x_i;\theta^*)$ 最大）的 $p_{model}$ 的参数 $\theta^*$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104213332545.png" alt="image-20230104213332545" style="zoom:80%;" />



**sampling**：

从 $p_{model}(x;\theta)$ 中抽样

从复杂分布中抽样的常用方法：

首先从一个简单noise distribution $N(0,1)$ 中抽样出 $z_i$ 

然后将 $z_i$ expand到图数据上，即将它通过函数 $f(\cdot)$ 进行转换 $x_i=f(z_i;\theta)$ ，这样 $x_i$ 就能服从于一个复杂的分布。

$f(\cdot)$ 通过已知数据，用深度神经网络进行学习。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104213853677.png" alt="image-20230104213853677" style="zoom:80%;" />



**auto-regressive models** ：$p_{model}(x;\theta)$ 同时用于density estimation和sampling。

（一些其他模型，如Variational Auto Encoders (VAEs), Generative Adversarial Nets (GANs) 有二至多个模型来分别完成任务）

**核心思想：链式法则。**

联合分布是条件分布的连乘结果：
$$
p_{model}(x;\theta)=\prod_{t=1}^np_{model}(x_t|x_1,...,x_{t-1};\theta)
$$
例如：如果 $\bold{x}$ 是向量，$x_t$ 是其第 $t$ 维元素；$\bold{x}$ 是句子，$x_t$ 是其第 $t$ 个单词。

在我们的案例中，$x_t$ 是第 $t$ 个行动（如增加一个节点或增加一条边）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104214503526.png" alt="image-20230104214503526" style="zoom:80%;" />



## 19.3 GraphRNN：生成真实图

GraphRNN的优点在于它不需要任何inductive bias assumptions，就可以直接实现图生成任务。



**GraphRNN的思想**：sequentially增加节点和边，最终生成一张图。

如图所示：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104214701067.png" alt="image-20230104214701067" style="zoom:80%;" />



**将图建模为序列**：

给定图 $G$ 及其对应的node ordering $\pi$ ，我们可以将其唯一映射为一个node and edge additions的序列 $S^{\pi}$ 

如图所示，序列 $S^{\pi}$ 的每个元素都是加一个节点和这个节点与之前节点连接的边：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104214924152.png" alt="image-20230104214924152" style="zoom:80%;" />

$S^{\pi}$ 是一个sequence的sequence，有两个级别：节点级别每次添加一个节点，边级别每次添加新节点与之前节点之间的边。

- **节点级别**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104215049813.png" alt="image-20230104215049813" style="zoom:80%;" />

- **节点级别的每一步是一个边级别的序列**：每一个元素是是否与该节点添加一条边，即形成一个如图所示的0-1变量序列

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104215152507.png" alt="image-20230104215152507" style="zoom:80%;" />



这里的node ordering是随机选的，随后我们会讨论这一问题。

如图所示，每一次是生成邻接矩阵（黄色部分）中的一个节点（向右），每个节点生成一列边（向下）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104215359003.png" alt="image-20230104215359003" style="zoom:80%;" />

这样我们就将图生成问题转化为序列生成问题。



**我们需要建模两个过程**：

1. 生成一个新节点的state（节点级别序列）
2. 根据新节点state生成它与之前节点相连的边（边级别序列）

**方法：用Recurrent Neural Networks (RNNs) 建模这些过程**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104215505958.png" alt="image-20230104215505958" style="zoom:80%;" />



**RNN**： RNNs是为序列数据所设计的，它sequentially输入序列数据以更新其hidden states，其hidden states包含已输入RNN的所有信息。更新过程由RNN cells实现。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104215653877.png" alt="image-20230104215653877" style="zoom:80%;" />



**RNN cell**：

- $s_t$ ：RNN在第 $t$ 步之后的state
- $x_t$ ：RNN在第 $t$ 步的输入
- $y_t$ ：RNN在第 $t$ 步的输出

（在我们的例子中，上述三个值都是标量）

**RNN cell: 可训练参数** $W,U,V$

1. 第一步：根据输入和上一步state更新hidden state $s_t=\sigma(W\cdot x_t+U\cdot s_{t-1})$ 
2. 第二步：根据state进行输出 $y_t=V\cdot s_t$

还有更具有表现力的cells：GRU，LSTM等

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104220050087.png" alt="image-20230104220050087" style="zoom:50%;" />



**GraphRNN: Two levels of RNN**

GraphRNN有一个节点级别RNN和一个边级别RNN，节点级别RNN生成边级别RNN的初始state，边级别RNN sequentially预测这个新节点与每一个之前的节点是否相连。

**如图所示，边级别RNN预测新加入的节点是否与之前各点相连**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104220208234.png" alt="image-20230104220208234" style="zoom: 80%;" />



**接下来将介绍如何用这个RNN生成序列**：

1. 用RNN生成序列：用前一个cell的输出作为下一个cell的输入（$x_{t+1}=y_t$）
2. 初始化输入序列：用 start of sequence token (SOS) 作为初始输入。SOS常是一个全0或全1的向量。
3. 结束生成任务：用 end of sequence token (EOS) 作为RNN额外输出。

**如果输出EOS=0，则RNN继续生成；如果过输出EOS=1，则RNN停止生成。**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104220513741.png" alt="image-20230104220513741" style="zoom:80%;" />



**模型如图所示**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104220926612.png" alt="image-20230104220926612" style="zoom:80%;" />

**这样的问题在于模型是确定的，但我们需要生成的是分布，所以需要模型具有随机性。**

我们的目标就是用RNN建模 $\prod_{k=1}^np_{model}(x_t|x_1,...,x_{t-1};\theta)$

所以我们让 $y_t=p_{model}(x_t|x_1,...,x_{t-1};\theta)$，然后从 $y_t$ 中抽样 $x_{t+1}$ ，即 $x_{t+1}\sim y_t$ ：RNN每一步产生一条边的生成概率，我们依此抽样并将抽样结果输入下一步。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104221240126.png" alt="image-20230104221240126" style="zoom:80%;" />



**RNN at Test Time**：我们假设已经训练好了模型，$y_t$ 是 $x_{t+1}$ 是否为1这一遵从伯努利分布事件的概率，从而根据模型我们可以从输入输出 $y_t$ ，从而抽样出 $x_{t+1}$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104221543092.png" alt="image-20230104221543092" style="zoom:80%;" />

**RNN at Training Time**：在训练过程中，我们已知的数据就是序列 $y^*$ （该节点与之前每一节点是否相连的0-1元素组成的序列）。我们使用teacher forcing的方法，将每一个输入都从前一个节点的输出换成真实序列值，而用真实序列值与模型输出值来计算损失函数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104221821856.png" alt="image-20230104221821856" style="zoom:80%;" />

这一问题的损失函数使用binary cross entropy，即最小化下式损失函数：
$$
L=-[y_1^*\log(y_1)+(1-y_1^*)\log(1-y_1)]
$$
对每一个输出，上式右式左右两项同时只能存在一个：

- 如果边存在，即 $y_1^*=1$ ，则我们需要最小化 $-\log(y_1)$ ，即使 $y_1$ 增大。
- 如果边不存在，即 $y_1^*=0$ ，我们需要最小化 $-\log(1-y_1)$ ，即使 $y_1$ 减小。

这样就使 $y_1$ 靠近data samples $y_1^*$ 。

$y_1$ 是由RNN计算得到的，通过这一损失函数，使用反向传播就能对应调整RNN参数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104222413191.png" alt="image-20230104222413191" style="zoom:80%;" />



**Putting Things Together**：

1. 增加一个新节点：跑节点RNN，用其每一步输出来初始化边RNN
2. 为新节点增加新边：跑边RNN，预测新节点是否与每一之前节点相连
3. 增加另一个新节点：用边RNN最后的hidden state来跑下一步的节点RNN
4. 停止图生成任务：如果边RNN在第一步输出EOS，则我们知道新节点上没有任何一条边，即不再与之前的图有连接，从而停止图生成过程。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104222534775.png" alt="image-20230104222534775" style="zoom:80%;" />



**训练过程**：

假设节点1已在图中，现在添加节点2：输入SOS到节点RNN中

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104222639925.png" alt="image-20230104222639925" style="zoom:50%;" />

边RNN预测节点2是否会与节点1相连：输入SOS到边RNN中，输出节点2是否会与节点1相连的概率0.5

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104222806487.png" alt="image-20230104222806487" style="zoom: 50%;" />

用边RNN的hidden state更新节点RNN

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104222858084.png" alt="image-20230104222858084" style="zoom: 50%;" />

边RNN预测节点3是否会与节点1、2相连：输入SOS到边RNN中，输出节点3是否会与节点2相连的概率0.6；输入节点3与节点2不相连的真实值0到下一个cell中，输出节点3是否会与节点2相连的概率0.4

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104222952858.png" alt="image-20230104222952858" style="zoom:50%;" />

用边RNN的hidden state更新节点RNN

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223031179.png" alt="image-20230104223031179" style="zoom: 50%;" />

我们已知节点4不与任何之前节点相连，所以停止生成任务：输入SOS到边RNN中，没看懂这里是不是用teacher forcing强制停止的意思。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223156920.png" alt="image-20230104223156920" style="zoom: 50%;" />

每一步我们都用真实值作为监督，如图所示，就跟右上角的图形式或邻接矩阵形式一样的真实值

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223309095.png" alt="image-20230104223309095" style="zoom:50%;" />

通过时间反向传播，随time step累积梯度，如图所示：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223347263.png" alt="image-20230104223347263" style="zoom:80%;" />



**测试阶段**：

1. 根据预测出来的边分布抽样边
2. 用GraphRNN自己的预测来代替每一步输入（就类似训练阶段如果不用tearcher forcing的那种效果）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223441706.png" alt="image-20230104223441706" style="zoom: 80%;" />



**GraphRNN总结**：

通过生成一个2级序列来生成一张图，用RNN来生成序列。如图中所示，节点级别RNN向右预测，边级别RNN向下预测。

接下来我们要使RNN tractable，以及对其效果进行评估。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223533970.png" alt="image-20230104223533970" style="zoom:80%;" />



**tractability**：

在此前的模型中，每一个新节点都可以与其前任何一个节点相连，这需要太多步边生成了，需要产生一整个邻接矩阵（如上图所示），也有太多过长的边依赖了（不管已经有了多少个节点，新节点还要考虑是否与最前面的几个节点有边连接关系）。

如果我们使用随机的node ordering，那我们对每个新生成的节点就是都要考虑它与之前每一个节点是否有边（图中左下角所示）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223747715.png" alt="image-20230104223747715" style="zoom:80%;" />



**BFS**：但是如果我们换成一种BFS的node ordering，那么在对每个边考虑它可能相连的之前节点的过程如图所示，我们只需要考虑在BFS时它同层和上一层的节点（因为再之前的节点跟它不会有邻居关系），即只需要考虑2步的节点而非 $n-1$ 步的节点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104223851869.png" alt="image-20230104223851869" style="zoom:80%;" />

**这样的好处有二**：

- 减少了可能存在的node ordering数量（从 $O(n!)$ 减小到不同BFS ordering的数量）
- 减少了边生成的步数（因为不需要看之前所有节点了，只需要看一部分最近的节点即可）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224011950.png" alt="image-20230104224011950" style="zoom:80%;" />

**在运行GraphRNN时仅需考虑该节点及其之前的一部分节点，如图所示**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224146123.png" alt="image-20230104224146123" style="zoom:80%;" />



**对生成图的评估**：我们的数据集是若干图，输出也是若干图，我们要求评估这两组图之间的相似性。有直接从视觉上观察其相似性和通过图统计指标来衡量其相似性两种衡量方式。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224233656.png" alt="image-20230104224233656" style="zoom:80%;" />

- **visual similarity**：就直接看，能明显地发现在grid形式的图上，GraphRNN跟输入数据比传统图生成模型（主要用于生成网络而非这种grid图）要更像很多

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224329166.png" alt="image-20230104224329166" style="zoom:80%;" />

  即使在传统图生成模型应用的有社区的社交网络上，GraphRNN也表现很好，如图所示。这体现了GraphRNN的可泛化能力。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224426417.png" alt="image-20230104224426417" style="zoom:80%;" />

- **graph statistics similarity**：我们想找到一些比目测更精确的比较方式，但直接在两张图的结构之间作比较很难（同构性检测是NP的），因此我们选择比较图统计指标。

  **典型的图统计指标包括**：

  - degree distribution (Deg.)
  - clustering coefficient distribution (Clus.)
  - orbit count statistics

  注意：每个图统计指标都是一个概率分布。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224555312.png" alt="image-20230104224555312" style="zoom:80%;" />

  所以我们一要比较两种图统计指标（两个概率分布），解决方法是earth mover distance (EMD)；二要比较两个图统计指标的集合（两个概率分布的集合），解决方法是基于EMD的maximum mean discrepancy (MMD)。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224657809.png" alt="image-20230104224657809" style="zoom:80%;" />

  - **earth mover distance (EMD)**：用于比较两个分布之间的相似性。在直觉上就是衡量需要将一种分布编程另一种分布所需要移动的最小“泥土量”（面积）。总之这里有个公式，但是我也没仔细看具体怎么搞的。

    <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224812534.png" alt="image-20230104224812534" style="zoom:80%;" />

  - **maximum mean discrepancy (MMD)**：基于元素相似性，比较集合相似性：使用L2距离，对每个元素用EMD计算距离，然后用L2距离计算MMD。

    <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104224955119.png" alt="image-20230104224955119" style="zoom:80%;" />

  **对图生成结果的评估**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225042378.png" alt="image-20230104225042378" style="zoom:80%;" />

  计算举例：通过计算原图域生成图之前在clustering coefficient distribution上的区别，我们发现GraphRNN是表现最好的（即最相似的）。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225127016.png" alt="image-20230104225127016" style="zoom:80%;" />



## 19.4 深度图生成模型的应用

本节主要介绍深度图生成模型在药物发现领域的应用GCPN。



药物发现领域的问题是：我们如何学习一个模型，使其生成valid、真实的分子，且具有优化过的某一属性得分（如drug-likeness或可溶性等）？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225234326.png" alt="image-20230104225234326" style="zoom:80%;" />



**这种生成任务就是goal-directed graph generation**：

- 优化一个特定目标得分（high scores），如drug-likeness
- 遵从内蕴规则（valid），如chemical validity rules
- 从示例中学习（realistic），如模仿一个分子图数据集

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225400537.png" alt="image-20230104225400537" style="zoom:80%;" />



这一任务的难点在于需要在机器学习中引入黑盒：像drug-likeness这种受物理定律决定的目标是我们不可知的。

**我们的解决思路是使用强化学习的思想**：

强化学习是一个机器学习agent观察环境environment，采取行动action来与环境互动interact，收到正向或负面的反馈reward，根据反馈从这一回环之中进行学习。回环如图所示。

其核心思想在于agent是直接从环境这一对agent的黑盒中进行学习的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225614486.png" alt="image-20230104225614486" style="zoom:80%;" />



**我们的解决方法是GCPN：graph convolutional policy network**

结合了图表示学习和强化学习

**核心思想**：

1. GNN捕获图结构信息
2. 强化学习指导导向预期目标的图生成过程
3. 有监督训练模拟给定数据集的样例

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225715332.png" alt="image-20230104225715332" style="zoom:80%;" />



**GCPN vs GraphRNN**：

- **共同点**：
  - sequentially生成图
  - 模仿给定的图数据集
- **主要差异**：
  - **GCPN用GNN来预测图生成行为**。
    - 优势：GNN比RNN更具有表现力
    - 劣势：GNN比RNN更耗时（但是分子一般都是小图，所以我们负担得起这个时间代价）
  - GCPN使用RL来直接生成符合我们目标的图。RL使goal-directed graph generation成为可能。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225917172.png" alt="image-20230104225917172" style="zoom:80%;" />



**sequential graph generation**：

GraphRNN：基于RNN hidden states（捕获至此已生成图部分的信息）预测图生成行为。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104225954160.png" alt="image-20230104225954160" style="zoom:80%;" />

GCPN：基于GNN节点嵌入，用链接预测任务来预测图生成行为。

这种方式更具有表现力、更有鲁棒性，但更不scalable。

回忆链接预测任务的prediction head，concatenation+linear这种方式就是：
$$
Head_{edge}(\bold{h}_u^{(L)},\bold{h}_v^{(L)})=Linear(Concat(\bold{h}_u^{(L)},\bold{h}_v^{(L)}))
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230247672.png" alt="image-20230104230247672" style="zoom:80%;" />



**GCPN概览**：如图所示，首先插入节点5，然后用GNN预测节点5会与哪些节点相连，抽样边（action），检验其化学validity，计算reward。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230330957.png" alt="image-20230104230330957" style="zoom:80%;" />



**我们如何设置reward？**

**我们设置两种reward**：

- 一种是step reward，学习执行valid action：每一步对valid action分配小的正反馈。
- 一种是final reward，优化预期属性：在最后对高预期属性分配正反馈。

**reward=final reward + step reward**

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230439274.png" alt="image-20230104230439274" style="zoom:80%;" />



**训练过程分两部分**：

- 有监督训练：通过模仿给定被观测图的行为训练policy，用交叉熵梯度下降。（跟GraphRNN中的一样）
- 强化学习训练：训练policy以优化反馈，使用standard policy gradient algorithm。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230624215.png" alt="image-20230104230624215" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230656982.png" alt="image-20230104230656982" style="zoom:80%;" />



**GCPN实验结果**：在logP和QED这些医药上要优化的指标上都表现很好

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230756174.png" alt="image-20230104230756174" style="zoom:80%;" />

constrained optimization / complete任务：编辑给定分子，在几步之后就能达到高属性得分（如在以logP作为罚项的基础上，提升辛醇的可溶性）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230823640.png" alt="image-20230104230823640" style="zoom:80%;" />



## 19.5 总结

1. 复杂图可以用深度学习通过sequential generation成功生成。

2. 图生成决策的每一步都基于hidden state。

   hidden state可以是隐式的向量表示（因为RNN的中间过程都在hidden state里面，所以说是隐式的），由RNN解码；也可以是显式的中间生成图，由GCN解码。

3. 可以实现的任务包括模仿给定的图数据集和往给定目标优化图。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230104230941023.png" alt="image-20230104230941023" style="zoom:80%;" />





# 20 GNN高级主题

**本章主要内容**：

本章首先介绍了在此之前学习的message passing系GNN模型的限制，然后介绍了position-aware GNN和 identity-aware GNN (IDGNN)来解决相应的问题。

最后介绍了GNN模型的鲁棒性问题。



## 20.1 GNN高级主题

我们首先可以回忆一下图神经网络：输入图结构数据，经神经网络，输出节点或更大的网络结构（如子图或图）的嵌入。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105194344571.png" alt="image-20230105194344571" style="zoom:80%;" />

此外还可以回忆一下 general GNN framework 和 GNN training pipeline 的相关内容



## 20.2 图神经网络的局限性

**完美的GNN模型**：

在这里我们提出一个思想实验：完美的GNN应该做什么？

k层GNN基于K跳邻居结构嵌入一个节点，完美的GNN应该在邻居结构（无论多少跳）和节点嵌入之间建立单射函数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105194532371.png" alt="image-20230105194532371" style="zoom:80%;" />

因此，对于完美GNN，假设节点特征完全一样，所以节点嵌入的区别完全在于结构。在这种情况下

1. 有相同邻居结构的节点应该有相同的嵌入
2. 有不同邻居结构的节点应该有不同的嵌入。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105194630738.png" alt="image-20230105194630738" style="zoom:80%;" />



**现有GNN的不完美之处**：即使是符合上述条件的“完美GNN”也不完美。

- 即使有相同邻居结构的节点，我们也可能想对它们分配不同的嵌入，因为它们出现在图中的不同位置positions。这种任务就是position-aware tasks。

  比如即使是完美GNN，在图示任务上也会失效（左图比较显然，就是左下角点和右上角点虽然有完全一样的邻居结构，但是由于其位置不同，我们仍然希望它们有不同的嵌入。右图我没搞懂，它节点标哪儿了我都没看到）

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105194749939.png" alt="image-20230105194749939" style="zoom:80%;" />

- 在lecture 9中，我们讨论了message passing系GNN的表现能力上限为WL test。所以举例来说，图中的 $v_1$ 和 $v_2$ 在cycle length（就是节点所处环的节点数）上的结构差别就无法被捕获，因为虽然它们结构不同，但计算图是相同的：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105194908512.png" alt="image-20230105194908512" style="zoom:80%;" />



**本节lecture就要通过构建更有表现力的GNN模型来解决上述两个问题**：

- 对①问题：通过图中节点的位置来生成节点嵌入，方法举例：position-aware GNNs
- 对②问题：构建比WL test更有表现力的message passing系GNN，方法举例：identity-aware GNNs

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105195013505.png" alt="image-20230105195013505" style="zoom:80%;" />



我们希望不同的输入（节点、边或图）被标注为不同的标签。而嵌入是通过GNN计算图得到的，如图所示

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105204200827.png" alt="image-20230105204200827" style="zoom:80%;" />

**直接通过独热编码的方式来试图区分计算图是不行的，如图所示**：

给每个节点以一个独热编码表示的唯一ID的话，我们就能区分不同的节点/边/图，计算图显然是会不同的（就算个别层出现相同的情况，因为节点不一样，所以后面几层、最终的整条计算树总是会不同的），所以可以实现。

问题是这样不scalable（需要 $O(N)$ 维特征），也不inductive（显然无法泛化到新节点/新图上：这个很依赖于节点顺序，一换图就不行了）。

但是这种通过增加节点特征使得通过计算图可以更好地进行节点分类的思路是可行的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105204406734.png" alt="image-20230105204406734" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105204419988.png" alt="image-20230105204419988" style="zoom:80%;" />



## 20.3 位置感知图神经网络

如图所示，图上有两种任务：一种是structure-aware task（节点因在图上的结构不同而有不同的标签），一种是position-aware task（节点因在图上的位置不同而有不同的标签）。真实任务往往结合了structure-aware和position-aware，所以我们需要能够同时解决这两种任务的方法。
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105204530059.png" alt="image-20230105204530059" style="zoom:80%;" />



GNN往往对structure-aware tasks表现很好，如图所示，GNN可以通过不同的计算图来区分 $v_1$ 和 $v_2$ 这两个局部结构不同的节点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105204629800.png" alt="image-20230105204629800" style="zoom:80%;" />

但GNN对position-aware tasks表现较差，如图所示，因为结构对称性，$v_1$ 和 $v_2$ 会有相同的计算图，所以他们的嵌入也会相同。因此我们想要定义position-aware的深度学习方法来解决这一问题。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105204720763.png" alt="image-20230105204720763" style="zoom:80%;" />



**anchor**：解决方法就是使用anchor作为reference points

如图所示，随机选取节点 $s_1$ 作为anchor node，用对 $s_1$ 的相对距离来表示 $v_1$ 和 $v_2$ ，这样两个节点就不同了（$v_1$ 是1，$v_2$ 是0）。

anchor node相当于坐标轴，可以用于定位图中的节点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205018254.png" alt="image-20230105205018254" style="zoom:80%;" />



**anchors**：

如图所示随机选取 $s_1,s_2$ 作为anchor nodes，用相对它们的距离来表示节点。这样就可以更好地描述图中不同区域的节点位置。更多anchors相当于更多的坐标轴。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205116673.png" alt="image-20230105205116673" style="zoom:80%;" />



**anchor-sets**：

将anchor从一个节点泛化到一堆节点，定义某节点到一个anchor-set的距离是该节点距anchor-set中任一节点的最短距离（即到最短距离最短的节点的最短距离）。如图所示，$s_3$ 这个大小为2的anchor-set可以区分 $s_1$ 和 $s_2$ 这两个anchor区分不了的 $v_1$ 和 $v_3$ 节点。

有时大的anchor-set能提供更精确的位置估计。而且这种方法还能保持anchor总数较小，要不然计算代价太高。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205348444.png" alt="image-20230105205348444" style="zoom:80%;" />



**总结**：

节点的位置信息可以通过到随机选取anchor-sets的距离来编码，每一维对应一个anchor-set。

如图所示，每个节点都有了一个position encoding（图中的一行）。

具体在实践中，可以对anchor-set中的节点数指数级增长、同节点数的anchor-set数量指数级减小，如设置n个1个节点的anchor-set、n/2个2个节点的anchor-set……

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205548360.png" alt="image-20230105205548360" style="zoom:80%;" />



**如何使用position information（即上图中这个position encoding）**：

1. **简单方法**：直接把position encoding当作增强的节点特征来用。这样做的实践效果很好。

   这样做的问题在于，因为position encoding的每一维对应一个随机anchor，所以position encoding本身可以被随机打乱而不影响其实际意义，但在普通神经网络中如果打乱输入维度，输出肯定会发生变化。

   我对这个问题的理解就是：它是permutation invariant的，但是普通神经网络不是。这个问题本来就是GNN比欧式数据难的原因之一，结果GNN里面又出了这个问题真是微妙啊。

2. **严谨方法**：设计一个能保持position encoding的permutation invariant性质的特殊神经网络。

   由于打乱输入特征维度只会打乱输出维度，因此具体维的数据不用改变。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205836435.png" alt="image-20230105205836435" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205855852.png" alt="image-20230105205855852" style="zoom:80%;" />



## 20.4 身份感知图神经网络

GNN还有更多的失败案例：除了上述的position-aware tasks之外，GNN在structure-aware tasks上也是不完美的。以下展示节点、边、图三种层面上的GNN在structure-aware tasks上的失败案例，都是结构不同但计算图相同的情况：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105205950152.png" alt="image-20230105205950152" style="zoom:80%;" />

- **节点级别的失败案例：处在节点数不同的环上**

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105210051516.png" alt="image-20230105210051516" style="zoom:80%;" />

- **边级别的失败案例：如图中的边A和B，由于 $v_1$ 和 $v_2$ 计算图相同，因此A和B的嵌入也会相同**

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105210212791.png" alt="image-20230105210212791" style="zoom:80%;" />

- **图级别的失败案例：如图所示，每个图有8个节点，每个节点与另外的四个节点相连，左图与相邻节点和隔一个节点的节点相连，右图与相邻节点和隔两个节点的节点相连。这两个图是不同构的，但是它们的每个节点的嵌入都相同：WL test无法区分它们，因此GNN也无法区分它们**

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105210311749.png" alt="image-20230105210311749" style="zoom:80%;" />



**idea: inductive node coloring**

核心思想：对我们想要嵌入的节点分配一个颜色（作为augmented identity），如图所示

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105210429666.png" alt="image-20230105210429666" style="zoom:80%;" />

这个coloring是inductive的，与node ordering或identities无关。如图所示，打乱 $v_2$ 和 $v_3$ 的顺序之后，计算图不变

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105210601673.png" alt="image-20230105210601673" style="zoom:80%;" />



**inductive node coloring在各个级别上，都能帮助对应的图数据的计算图变得可区分**：

- **node level**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105210641620.png" alt="image-20230105210641620" style="zoom:80%;" />

- **graph level**：

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105211627690.png" alt="image-20230105211627690" style="zoom:80%;" />

- **edge level：需要嵌入两个节点，我们选择其中一个节点进行上色**

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105211716465.png" alt="image-20230105211716465" style="zoom:80%;" />



**那么我们应该如何在GNN中应用node coloring呢？**

这样就提出了ID-GNN (Identity-aware GNN)

**整体思路**：像异质图中使用的heterogenous message passing那样。传统GNN在每一层上对所有节点应用相同的message/aggregation操作，而heterogenous message passing是对不同的节点应用不同的message passing操作，ID-GNN就是在不同coloring的节点上应用不同的message/aggregation操作。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105211852955.png" alt="image-20230105211852955" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105211906324.png" alt="image-20230105211906324" style="zoom:80%;" />

heterogenous massage passing可以通过在嵌入计算过程中应用不同的神经网络，使这种计算图结构相同、node coloring不同的节点得到不同的嵌入。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105211943000.png" alt="image-20230105211943000" style="zoom:80%;" />



**GNN vs ID-GNN**：

ID-GNN可以计算一个节点所属环的节点数，而GNN不能。

如图所示，根据计算图中与根节点同色的节点所在层数，可以数出这个节点数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212045269.png" alt="image-20230105212045269" style="zoom:80%;" />



**simplified version: ID-GNN-Fast**

根据上一条的直觉，我们可以设计一个简化版本ID-GNN-Fast：将identity information作为augmented node feature（这样就不用进行heterogenous message passing操作了）

我们就用每一层的cycle counts作为augmented node feature，这样就可以应用于任意GNN了。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212143348.png" alt="image-20230105212143348" style="zoom:80%;" />



**Identity-aware GNN**：

ID-GNN是一个对GNN框架通用的强大扩展，可以应用在任何message passing GNNs上（如GCN，GraphSAGE，GIN等），在节点/边/图级别的任务上都给出了一致的效果提升。

ID-GNN比别的GNN表现力更好，是第一个比1-WL test更有表现力的message passing GNN。

我们可以通过流行的GNN工具包（如PyG，DGL等）来应用ID-GNN。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212258102.png" alt="image-20230105212258102" style="zoom:80%;" />



## 20.5 GNN 的鲁棒性

近年来，深度学习在各领域都体现出了令人印象深刻的表现效果，如在计算机视觉领域，深度卷积网络在ImageNet（图像分类任务）上达到了人类级别的效果。那么，这些模型可以应用于实践了吗？

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212343508.png" alt="image-20230105212343508" style="zoom:80%;" />



**对抗样本**：深度卷积网络对对抗攻击很脆弱，如图所示，只需要几乎肉眼无法察觉的噪音扰动，就会对预测结果产生巨大的改变。在自然语言处理和音频处理领域也报道过类似的对抗样本。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212422917.png" alt="image-20230105212422917" style="zoom:80%;" />



**对抗样本的启示**：

由于对抗样本的存在，深度学习模型部署到现实世界就不够可靠，对抗者可能会积极积极攻击深度学习模型，模型表现可能会比我们所期望的差很多。

深度学习往往不够鲁棒，事实上，使深度学习模型对对抗样本鲁棒仍然是个活跃的研究领域。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212529095.png" alt="image-20230105212529095" style="zoom:80%;" />



**GNNs的鲁棒性**：本节lecture将介绍GNN是否对对抗样本鲁棒。本节课介绍的基础是GNNs的广泛应用关乎于公开平台和经济利益，包括推荐系统、社交网络、搜索引擎等，对抗者有动机来操纵输入图和攻击GNN预测结果。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212649638.png" alt="image-20230105212649638" style="zoom:80%;" />



**研究GNN鲁棒性的基础设置**

任务：半监督学习节点分类任务

模型：GCN

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105212731475.png" alt="image-20230105212731475" style="zoom:80%;" />



**问题研究路径**：

1. 描述几种现实世界中的adversarial attack possibilities。
2. 我们再研究一下我们要攻击的GCN模型（了解对象）。
3. 我们将攻击问题用数学方法构建为优化问题。
4. 通过实验来检验GCN的预测结果对对抗攻击有多脆弱。



**attack possibilities**：

这个应该是指对对抗攻击任务类型的介绍。

- **target nodes** $t\in V$： 我们想要改变其标签预测结果。
- **attacker nodes** $S\sub V$：攻击者可以改变的节点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105213315623.png" alt="image-20230105213315623" style="zoom:80%;" />

- **direct attack**：attacker nodes就是traget node $S=\{t\}$ ，其分类如图所示

  1. 调整target node特征：如改变网站内容
  2. 对target node增加边（连接）：如买粉/买赞
  3. 对target node删除边（连接）：如对某些用户取消关注

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105213525852.png" alt="image-20230105213525852" style="zoom:80%;" />

- **indirect attack**：attacker nodes不是target node $t\notin S$ ，其分类如图所示

  1. 调整attacker node特征：如hijack（劫持、操纵） target nodes的好友。
  2. 对attackers增加边（连接）：如创建新链接，如link farm。
  3. 删除attackers的边（连接）：如删除不想要的链接。

  <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105220009996.png" alt="image-20230105220009996" style="zoom:80%;" />

  

**将对抗攻击构建为优化问题**：

attacker的目标：最大化target node标签预测结果的改变程度

subject to 图上的改变很小（如果对图的改变过大，将很容易被检测到。成功的攻击应该在对图的改变“小到无法察觉”时改变target的预测结果）

如图所示，在图上做很小的改变（改变两个节点的特征），学习GCN模型后，预测标签就得到了改变

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105220110022.png" alt="image-20230105220110022" style="zoom:80%;" />



**数学形式**：

原图：$A$ 邻接矩阵，$X$ 特征矩阵

操作后的图（已添加噪音）：$A'$ 邻接矩阵，$X'$ 特征矩阵

假设：$(A',X')\approx(A,X)$ （对图的操作足够小，以至于无法被察觉，如保留基本的图统计指标（如度数分布）和特征统计指标等）

对图的操作可以是direct（改变target nodes的特征或连接）或indirect的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105220324058.png" alt="image-20230105220324058" style="zoom:80%;" />

target node：$v\in V$

GCN学习原图：$\theta^*=\arg min_{\theta}\ L_{train}(\theta;A,X)$

GCN对target node的原始预测结果：$c_v^*=\arg max_{c}\ f_{\theta^*}(A,X)_{v,c}$ （节点 $v$ 预测概率最高的类）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105220643055.png" alt="image-20230105220643055" style="zoom:80%;" />

GCN学习被修改后的图：${\theta^*}'=\arg min_{\theta}\ L_{train}(\theta;A',X')$

注意这里的 $\theta$ 也可以不变，即指模型已经部署好了，参数不变

GCN对target node的预测结果：${c_v^*}'=\arg max_{c}\ f_{{\theta^*}'}(A',X')_{v,c}$

我们希望这个预测结果在图修改后产生变化：${c_v^*}'\ne c_v^*$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105220958375.png" alt="image-20230105220958375" style="zoom:80%;" />

target node $v$ 预测结果的改变量：$\Delta(v';A',X')=log\ f_{{\theta^*}'}(A',X')_{v,{c_v^*}'}-log\ f_{{\theta^*}'}(A',X')_{v,c_v^*}$ 

即新预测类被预测的概率取对数（我们想要提升的项）减原预测类被预测的概率取对数（我们想要减少的项）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105221606210.png" alt="image-20230105221606210" style="zoom:80%;" />

**最终的优化目标公式就是**：
$$
\arg max_{A',X'}\Delta(v';A',X')\\
subject\ to \ (A',X')\approx (A,X)
$$
**优化目标中存在的挑战**：

1. 邻接矩阵 $A'$ 是一个离散对象，无法使用基于梯度的优化方法。
2. 对每个经调整后的图 $A'$ 和 $X'$ ，GCN需要重新训练，这样做的计算代价很高。

**[Zügner et al. KDD2018]中提出了一些使优化过程tractable的近似技巧**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105221950098.png" alt="image-20230105221950098" style="zoom:80%;" />



**实验：对抗攻击**

在文献引用网络（有2800个节点，8000条边）上用GCN模型运行半监督节点分类任务。在原图上重复运行5次，对target node属于各类的预测概率如图所示

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105222308816.png" alt="image-20230105222308816" style="zoom:80%;" />

证明GCN模型可以很好地使原图上的证明GCN模型可以很好地使原图上的target node被分到正确的类中。target node被分到正确的类中。

在连接到target node上的5个边被修改（direct adversarial attack）后，GCN的预测结果如图所示：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105222342121.png" alt="image-20230105222342121" style="zoom:80%;" />

可以看出其结果被改变了。



**实验：attack comparison**

经实验发现：

- adversarial direct attack是最强的攻击，可以有效降低GCN的表现效果（与无攻击相比）。
- random attack比对抗攻击弱很多。
- indirect attack比direct attack更难。

实验结果如图所示，每个点代表一次攻击实验，classfication越低证明误分类越严重（即攻击效果越好）：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105222506907.png" alt="image-20230105222506907" style="zoom:80%;" />



**总结**：

1. 我们研究了应用于半监督节点分类任务的GCN模型的adversarial robustness。
2. 我们考虑了对图结构数据的不同attack possibilities。
3. 我们用数学方法将对抗攻击构建为优化问题。
4. 我们实证地证明了GCN的预测效果可能会因对抗攻击而被严重损害。
5. GCN对对抗攻击不鲁棒，但对indirect attacks和随机噪音还是有一些鲁棒性的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230105222617774.png" alt="image-20230105222617774" style="zoom:80%;" />



# 21 将GNN放大到大型图

**本章主要内容**：

本章首先介绍了GNN在实践中遇到的难以应用到大图上的问题，指出了scale up GNN这一课题的研究重要性。

接下来介绍了三种解决这一问题的方法：

- GraphSAGE模型的neighbor sampling，
- Cluster-GCN模型
- 简化GNN模型（SGC模型）。



## 21.1 介绍scale up GNN问题

图数据在当代ML研究中应用广泛，在很多领域中都出现了可供研究的大型图数据。

**推荐系统，推荐商品（链接预测），分类用户/物品（节点分类）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106163140289.png" alt="image-20230106163140289" style="zoom:80%;" />

**社交网络，好友推荐（边级别），用户属性预测（节点级别）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106163227121.png" alt="image-20230106163227121" style="zoom:67%;" />

**学术图，论文分类（节点分类），作者协作预测、文献引用预测（链接预测）**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106163357702.png" alt="image-20230106163357702" style="zoom:80%;" />

**知识图谱（KG），KG completion，推理**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106163427299.png" alt="image-20230106163427299" style="zoom:80%;" />



**这些图的共同点有二**：

1. 大规模。节点数从10M到10B，边数从100M到100B。
2. 任务都分为节点级别（用户/物品/论文分类）和边级别（推荐、completion）。

**本节课就将介绍如何scale up GNNs到大型图上**。



**scale up GNN 的难点在于**：

在传统大型数据集上的ML模型的训练方法

目标：最小化平均损失函数 
$$
l(\theta)=\frac1N\sum_{i=1}^{N-1}l_i(\theta)
$$
其中 $\theta$ 是模型参数，$l_i(\theta)$ 是第 $i$ 个数据点的损失函数。

**我们使用随机梯度下降stochastic gradient descent（SGD）方法**：随机抽样 $M(<<N)$ 个数据（mini-batches），计算 $M$ 个节点上的 $l_{sub}(\theta)$ ，用SGD更新模型：$\theta\leftarrow\theta-\nabla l_{sub}(\theta)$

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106164148480.png" alt="image-20230106164148480" style="zoom:80%;" />



**如果我们想直接将标准SGD应用于GNN的话，我们就会遇到问题**：

在mini-batch时，我们独立随机抽样 $M(<<N)$ 个节点，会出现如图所示的情况，抽样到的节点彼此孤立。由于GNN就是靠聚集邻居特征生成节点嵌入的，在mini-batch中无法获取邻居节点，因此这样的标准SGD无法有效训练GNNs。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106164730922.png" alt="image-20230106164730922" style="zoom:80%;" />

**因此我们会使用naïve full-batch**：

如图所示，对整张图上的所有节点同时生成嵌入：加载全部的图结构和特征，对每一层GNN用前一层嵌入计算所有节点的嵌入（message-passing），计算损失函数，应用梯度下降来更新参数。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106164848651.png" alt="image-20230106164848651" style="zoom:80%;" />

但full-batch应用不适用于大型图，因为我们想要使用GPU来加速训练，但GPU的memory严重受限（仅有10GB-20GB），整个图结构和特征信息无法加载到GPU上。

如图所示，CPU运算慢，memory大（1TB-10TB）；GPU运算快，memory受限（10GB-20GB）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106165111476.png" alt="image-20230106165111476" style="zoom:80%;" />



**本节课我们介绍三种scale up GNN的方法来解决这个问题，这三种方法分为两类**：

- 第一类是在每一mini-batch中在小的子图上运行message-passing，每次只需要把子图加载到GPU上：GraphSAGE的neighbor sampling和Cluster-GCN。
- 第二类是将GNN简化为特征预处理操作（可以在CPU上有效运行）：Simplified GCN。



## 21.2 GraphSAGE邻居采样：Scaling up GNNs

**计算图**：GNN通过聚合邻居生成节点嵌入，表现为计算图的形式（如右图所示）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106165425647.png" alt="image-20230106165425647" style="zoom:80%;" />



**可以观察到，一个2层GNN对节点0使用2跳邻居结构和特征生成嵌入**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106165534399.png" alt="image-20230106165534399" style="zoom:80%;" />

**可以泛化得到：K层GNN使用K跳邻居结构和特征生成嵌入**。



我们发现，为计算单一节点的嵌入，我们只需要其K跳邻居（以之定义计算图）。在一个mini-batch中给定M个不同的节点，我们可以用M个计算图来生成其嵌入（如图所示），这样就可以在GPU上计算了：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106165909393.png" alt="image-20230106165909393" style="zoom:80%;" />



**GNN随机训练**：我们现在就可以考虑用SGD策略训练K层GNN

1. 随机抽样 $M(<<N)$ 个节点。
2. 对每个被抽样的节点 $v$ 获取其K跳邻居，构建计算图，用以生成 $v$ 的嵌入。
3. 计算 $M$ 个节点上的平均损失函数 $l_{sub}(\theta)$ 。
4. 用SGD更新模型：$\theta\leftarrow\theta-\nabla l_{sub}(\theta)$ 

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106170626424.png" alt="image-20230106170626424" style="zoom:80%;" />



**随机训练的问题**:

对每个节点，我们都需要获取完整的K跳邻居并将其传入计算图中，我们需要聚合大量信息仅用于计算一个节点嵌入，这样计算开销过大。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106170716715.png" alt="image-20230106170716715" style="zoom:80%;" />

**更多细节**：

1. 计算图的大小会依层数K指数增长。如右图上面的图所示。
2. 在遇到hub node（度数很高的节点）时计算图会爆炸，但在现实世界的图中往往大多数节点度数较低，少量节点度数极高，就是会存在这种hub node。如右图下面的图所示。

**下一步，我们就要让计算图更小**。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106170837919.png" alt="image-20230106170837919" style="zoom:80%;" />



**neighbor sampling**：

核心思想：每一跳随机抽样H个邻居，构建计算图。

H=2时示例如图所示，从根节点到叶节点，每个节点抽样2个邻居作为子节点：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106171008473.png" alt="image-20230106171008473" style="zoom:80%;" />

我们可以用这个经剪枝的计算图来更有效地计算节点嵌入

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106171040807.png" alt="image-20230106171040807" style="zoom:80%;" />



**对K层GNN的neighbor sampling**：

对每个节点的计算图：对第k（k=1,2,…,K）层，对其k阶邻居随机抽样最多 $H_k$ 个节点，构成计算图。如图所示。

K层GNN一个计算图最多有 $\prod_{k=1}^KH_k$ 个叶节点（每一层都抽满）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106171347804.png" alt="image-20230106171347804" style="zoom:80%;" />



**neighbor sampling注意事项**：

1. 对sampling number $H$ 的权衡：小 $H$ 会使邻居聚合过程效率更高，但训练过程也会更不稳定（由于邻居聚合过程中variance会更大）。
2. 计算用时：即使有了neighbor sampling，计算图的尺寸还是跟GNN层数K呈指数增长。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106171504957.png" alt="image-20230106171504957" style="zoom:80%;" />

3. 如何抽样节点：

   - 随机抽样：快，但常不是最优的（可能会抽样到很多“不重要的”节点）

   - random walk with restarts：自然世界中的图是scale free的，即绝大多数节点度数很低，少量节点度数很高。对这种图随机抽样，会抽样出很多低度数的叶节点。

     抽样重要节点的策略：

     - 计算各节点自所求节点开始的random walk with restarts分值 $R_i$ 。
     - 每一层抽样 $H$ 个 $R_i$ 最高（与原节点相关性最高的节点）的邻居节点 $i$ 。

     在实践中，这种策略效果更好。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106171828711.png" alt="image-20230106171828711" style="zoom:80%;" />



**neighbor sampling总结**：

1. 计算图由每个mini-batch中的每个点构建。
2. 在neighbor sampling中，计算图为计算有效性而被剪枝/sub-sampled。此外也有增加模型鲁棒性的效果（因为提升了模型的随机性，有些类似于dropout）。
3. 我们用被剪枝的计算图来生成节点嵌入。
4. 尽管有了剪枝工作，计算图还是可能会很大，尤其在GNN的message-passing层增多时。这就需要batch size进一步减小（即剪枝更多），这使得结果variance更大、更不可靠。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106172020189.png" alt="image-20230106172020189" style="zoom:80%;" />



## 21.3 Cluster-GCN: Scaling up GNNs

**neighbor sampling的问题**：

1. 计算图的尺寸依GNN层数指数级增长。
2. 计算是冗余的，尤其在mini-batch中的节点有很多共享邻居时：如图所示，A节点和B节点具有同样的邻居C和D，在不考虑抽样的情况下这两个节点将会具有相同的计算图、即相同的节点，但在neighbor sampling的运算中需要分别各计算一次，就做了重复的事。

HAGs 这一篇是讲如何不用计算这些冗余嵌入的。列出来仅供参考。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106172245995.png" alt="image-20230106172245995" style="zoom:80%;" />



**在full-batch GNN中，所有节点嵌入都是根据前一层嵌入同时计算出的**：
$$
\forall v\in V,h_{v}^{(l)}=COMBINE(h_v^{(l-1)},AGGR(\{h_{u}^{(l)} \}_{u\in N(v)}))
$$
其中 $h_{u}^{(l)}$ 是message环节。

如图所示，对每一层来说，只需要计算2×边数次message（与邻居互相传递一次信息）；对K层GNN来说，只需要计算2K×边数次message。

GNN整体计算代价依边数和层数线性增长，这是很快的。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106172918884.png" alt="image-20230106172918884" style="zoom:80%;" />



从full-batch GNN中，我们可以发现，layer-wise的节点嵌入更新可以复用前一层的嵌入，这样就显著减少了neighbor sampling中产生的计算冗余问题。但是，由于GPU memory所限，这种layer-wise的更新方式不适用于大型图。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106173121093.png" alt="image-20230106173121093" style="zoom:80%;" />



**subgraph sampling**：

核心思想：从整个大型图中抽样出一个小子图，在子图上用GPU运行有效的layer-wise节点嵌入更新。如图所示：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106173253587.png" alt="image-20230106173253587" style="zoom:80%;" />

**核心问题：什么子图适用于训练GNNs**？

我们知道GNN通过边来传递信息，从而更新节点嵌入。因此，子图需要尽量多地保持原图中的边连接结构。通过这种方式，子图上的GNN就可以生成更接近于原图GNN的节点嵌入。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106173343940.png" alt="image-20230106173343940" style="zoom:80%;" />



**subgraph sampling: case study**

举例来说，下图中右边的两种子图，左边的子图更适用于训练GNN：保持了必要的社区结构、边连接模式，没有产生孤立点。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106173451978.png" alt="image-20230106173451978" style="zoom:80%;" />



**利用社区结构**：现实世界的图会呈现出社区结构，一个大型图可以被解构为多个小社区。将一个社区抽样为一个子图，每个子图就能保持必要的原图局部connectivity模式。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106173601755.png" alt="image-20230106173601755" style="zoom:80%;" />



**Cluster-GCN**：

1. **overview**：首先介绍 vanilla Cluster-GCN。

   Cluster-GCN分为两步：

   - pre-processing: 给定一个大型图，将其分割为多个node group（如子图）。
   - mini-batch training: 每次抽样一个node group，对其induced subgraph应用GNN的message passing。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106180219148.png" alt="image-20230106180219148" style="zoom:80%;" />

2. **pre-processing**：

   给定一个大型图 $G=(V,E)$ ，将其节点 $V$ 分割到 $C$ 个组中：$V_1,...,V_C$ 。

   我们可以使用任何scalable的社区发现方法，如Louvain或METIS方法。

   $V_1,...,V_C$  induces $C$ 个子图 $G_1,...,G_C$ 。

   （注意：group之间的边不被包含在这些子图中）

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106180532544.png" alt="image-20230106180532544" style="zoom:80%;" />

3. **mini-batching training**：

   对每个mini-batch，随机抽样一个node group $V_C$ ，构建induced subgraph $G_C=(V_C,E_C)$ 。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106180649945.png" alt="image-20230106180649945" style="zoom:80%;" />

   在 $G_C$ 上应用GNN的layer-wise节点更新，获取所有节点 $v\in V_C$ 的嵌入 $\bold{h}_v$ 。

   对每个节点求损失函数，对所有节点的损失函数求平均：
   $$
   l_{sub}(\theta)=\frac1{|V_C|}\cdot\sum_{v\in V_C}l_v(\theta)
   $$
   更新参数：$\theta\leftarrow\theta-\nabla l_{sub}(\theta)$ 

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181002241.png" alt="image-20230106181002241" style="zoom:80%;" />



**Cluster-GCN的问题**

如前文所述得到的induced subgraph移除了组间的链接，这使得其他组对该组的message会在message passing的过程中丢失，这会影响GNN的实现效果。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181058316.png" alt="image-20230106181058316" style="zoom:80%;" />

图社区检测算法会将相似的节点分到一类中，这样被抽样的node group就会倾向于仅包含整个数据的一个很集中的部分。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181143726.png" alt="image-20230106181143726" style="zoom:80%;" />

被抽样的节点不够多样化，不够用以表示整个图结构：这样经被抽样节点得到的梯度 $\frac1{|V_C|}\cdot\sum_{v\in V_C}\nabla l_v(\theta)$ 就会变得不可靠，会在不同node group上波动剧烈，即variance高。这会让SGD收敛变慢。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181324697.png" alt="image-20230106181324697" style="zoom:80%;" />



**advanced Cluster-GCN: overview**

对上述Clutser-GCN问题的解决方案：在每个mini-batch聚合多个node groups。

将图划分为相对来说较小的节点组，在每个mini-batch中：抽样并聚合多个node groups，构建induced subgraph，剩下工作就和vanilla Cluster-GCN相同（计算节点嵌入、计算损失函数、更新参数）。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181423097.png" alt="image-20230106181423097" style="zoom:80%;" />



**为什么这一策略有效**：

抽样多个node groups可以让被抽样节点更能代表全图节点，在梯度估算时variance更低。

聚合多个node groups的induced subgraph包含了组间的边，message可以在组间流动。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181504374.png" alt="image-20230106181504374" style="zoom:80%;" />



**advanced Cluster-GCN**

与vanilla Cluster-GCN相似，advanced Cluster-GCN分成两步：

1. **第一步：pre-processing**

   给定一个大型图 $G=(V,E)$ ，将其节点 $V$ 分割到 $C$ 个相对较小的组中：$V_1,...,V_C$ ，其中 $V_1,...,V_C$ 需要够小，才能使它们可以多个聚合起来，聚合后的组还是不会过大。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106181803443.png" alt="image-20230106181803443" style="zoom:80%;" />

2. **第二步：mini-batch training**

   对每个mini-batch，随机抽样一组 $q$ 个node groups：$\{V_{t_1},...,V_{t_q} \}\sub\{V_1,...,V_C \}$ 。

   聚合所有被抽样node groups中的节点：$V_{aggr}=V_{t_1}\cup...\cup V_{t_q}$ 。

   提取induced subgraph $G_{aggr}=(V_{aggr},E_{aggr})$ ，其中 $E_{aggr}=\{(u,v)|u,v\in V_{aggr} \}$ 。

   <img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106182205088.png" alt="image-20230106182205088" style="zoom:80%;" />



**时间复杂度对比**：

对于用K层GNN生成 $M(<<N)$ 个节点嵌入：

neighbor sampling（每层抽样 $H$ 个节点）：每个节点的K层计算图的尺寸是 $H^K$ ，M个节点的计算代价就是 $M\cdot H^K$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106182437596.png" alt="image-20230106182437596" style="zoom:80%;" />

Cluster-GCN: 对M个节点induced的subgraph运行message passing，子图共包含 $M\cdot D_{avg}$ 条边，对子图运行K层message passing的计算代价最多为 $K\cdot M\cdot D_{avg}$ 。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106182626740.png" alt="image-20230106182626740" style="zoom:80%;" />



总结：用K层GNN生成 $M(<<N)$ 个节点嵌入的计算代价为：

- neighbor-sampling（每层抽样 $H$ 个节点）：$M\cdot H^K$
- Cluster-GCN：$K\cdot M\cdot D_{avg}$ 

假设 $H=\frac{D_{avg}}{2}$ 即抽样一半邻居。这样的话，Cluster-GCN（计算代价 $2MHK$ ）就会远比neighbor sampling（计算代价 $M\cdot H^K$ ）更有效，因为Cluster-GCN依K线性增长而非指数增长。

一般我们会让 $H$ 比 $\frac{D_{avg}}{2}$ 大，可能2-3倍平均度数这样。现实中一般GNN都不深（即 $K$ 小），所以一般用neighbor sampling的很多。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106183225999.png" alt="image-20230106183225999" style="zoom:80%;" />



**Cluster-GCN总结**：

1. Cluster-GCN首先将整体节点分割到小node groups中。
2. 在每个mini-batch，抽样多个node groups然后聚合其节点。
3. GNN在这些节点的induced subgraph上进行layer-wise的节点嵌入更新。
4. Cluster-GCN比neighbor sampling计算效率更高，尤其当GNN层数大时。
5. 但Cluster-GCN会导致梯度估计出现系统偏差（由于缺少社区间的边。以及当GNN层数加深时，在原图中是真的可以加深的（增大感受野），但在子图中就不行，加深了会弹回来，是虚假的加深）

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106183402628.png" alt="image-20230106183402628" style="zoom:80%;" />



## 21.4 Scaling up by Simplifying GNNs

本节课讲SGC模型，这个模型是通过去除非线性激活函数简化了GCN模型。原论文证明经简化后在benchmark数据集上效果没有怎么变差。Simplified GCN证明是对模型设计很scalable的。





**mean-pool的GCN**：

给定图 $G=(V,E)$ ，输入节点特征 $X_v(v\in V)$ ，$E$ 包含自环（即对所有节点 $v$ ，都有 $(v,v)\in E$ ）

设置输入节点嵌入为 $h_v^{(0)}=X_v$ 

对 $k\in \{0,...,K-1 \}$ 层：对所有节点 $v$ ，以如下公式聚合邻居信息
$$
h_v^{(k+1)}=ReLU(W_k\cdot\frac1{|N(v)|}\sum_{u\in N(v)}h_v^{(k)} )
$$
最终得到节点嵌入：
$$
z_v=h_v^{(K)}
$$
<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106191006281.png" alt="image-20230106191006281" style="zoom:80%;" />



**GCN的矩阵格式**：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106191226376.png" alt="image-20230106191226376" style="zoom:80%;" />

以下给出了将GCN从向量形式（邻居聚合形式）转换为矩阵形式的公式：

注意GCN用的是re-normalized版本 $\tilde{A}=D^{-\frac12}AD^{-\frac12}$ ，这个版本在实证上比 $D^{-\frac12}A$  效果更好。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106191445006.png" alt="image-20230106191445006" style="zoom:80%;" />

上图中，第一个公式就是GCN以邻居聚合形式进行的定义，第二个公式就是GCN的矩阵形式，其中 $W$ 参数从左边移到右边可以考虑一下就因为 $H$ 是 $h^T$ 的堆叠，所以矩阵形式要乘 $W$ 的参数的话就要倒到右边并取转置。



**Simplifying GCN**：

移除掉GCN中的ReLU非线性激活函数，以简化GCN：$H^{(k+1)}=\tilde{A}H^{(k)}W_k^T$ 

经如下图所示（应该还挺直观的）的迭代推理，易知 $H^{(k)}$ 可以表示为 $\tilde{A}^KXW^T$ 的形式。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106191831516.png" alt="image-20230106191831516" style="zoom:80%;" />

其中 $\tilde{A}^KX$ 不含有任何需要训练得到的参数，因此可以被pre-compute。这可以通过一系列稀疏矩阵向量乘法来加速计算过程，即用 $\tilde{A}$ 左乘 $X$ K 次。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106191950828.png" alt="image-20230106191950828" style="zoom:80%;" />

设 $\tilde{X}=\tilde{A}^KX$ 为一个pre-computed matrix，则simplifie GCN的最终嵌入就是：$H^{(K)}=\tilde{X}W^T$ 。

这就是一个对pre-computed matrix的线性转换。

将矩阵形式转换回节点嵌入形式，即得到 $h_v^{(K)}=W\tilde{X}_v$ ，其中 $\tilde{X}_v$ 是pre-computed节点 $v$ 的特征向量），即节点 $v$ 的嵌入仅依赖于它自己的pre-processed特征。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106192213003.png" alt="image-20230106192213003" style="zoom:80%;" />

$\tilde{X} $ 计算完成后，$M$ 个节点的嵌入就可以以依 $M$ 线性增长的时间复杂度来生成：

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106192310344.png" alt="image-20230106192310344" style="zoom:80%;" />



**Simplified GCN: Summary**

Simplified GCN分成两步：

1. **pre-processing step**：

   计算 $\tilde{X}=\tilde{A}^KX$ （可以在CPU上做）。

2. **mini-batch training step**：

   对每个mini-batch，随机抽样M个节点 $\{ v_1,v_2,...,v_M\}$ 

   计算其嵌入 $h_{v_1}^{(K)}=W\tilde{X}_{v_1},h_{v_2}^{(K)}=W\tilde{X}_{v_2},...,h_{v_M}^{(K)}=W\tilde{X}_{v_M}$

   用该嵌入进行预测，计算M个数据点上的平均损失函数。

   应用SGD参数更新。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106192723537.png" alt="image-20230106192723537" style="zoom:80%;" />



**SGC与其他方法的比较**：

1. 与neighbor sampling相比：SGC生成节点嵌入的效率更高（不需要对每个节点构建大计算图）
2. 与Cluster-GCN相比：SGC的mini-batch节点可以从整体节点中完全随机地抽样，不需要从几个groups里面抽样。这样就可以减少训练过程中的SGD variance。
3. 这个模型的表现力也更低，因为生成节点嵌入的过程中没有非线性。

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106192915139.png" alt="image-20230106192915139" style="zoom:80%;" />

<img src="C:\Users\tu'tu\AppData\Roaming\Typora\typora-user-images\image-20230106192932706.png" alt="image-20230106192932706" style="zoom:80%;" />



但事实上，在半监督学习节点分类benchmark上，simplified GCN和原始GNNs的表现力几乎相当，这是由于graph homophily的存在：

很多节点分类任务的图数据都表现出homophily结构，即有边相连的节点对之间倾向于具有相同的标签。

举例来说，在文献引用网络中的文献分类任务，引用文献往往是同类；在社交网络中给用户推荐电影的任务，社交网络中是朋友关系的用户往往倾向于喜欢相同的电影。

在simplified GCN中，preprocessing阶段是用 $\tilde{A}$ 左乘 $X$ K次，即pre-processed特征是迭代求其邻居节点特征平均值而得到的（如下图三），因此有边相连的节点倾向于有相似的pre-processed特征，这样模型就倾向于将有边相连的节点预测为同一标签，从而很好地对齐了很多节点分类benchmark数据集中的graph homophily性质。


**simplified GCN: summary**

1. simplified GCN去除了GCN中的非线性，并将其简化为对节点特征进行的简单pre-processing。
2. 得到pre-processed特征后，就可以直接应用scalable mini-batch SGD直接优化参数。
3. simplified GCN在节点分类benchmark中表现很好，这是因为特征pre-processing很好地对齐了现实世界预测任务中的graph homophily现象。

