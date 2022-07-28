# 2-1. Node-level Prediction

2강에서는 전통적인 머신러닝 파이프라인에서 사용한 feature들을 알아보고자 한다.
<br>node-level 예측은 node classification 등과 같은 task로, 아래와 같은 feature를 사용한다.

### node degree
node가 갖고 있는 edge 개수로, 모든 이웃 node들을 동등하게 취급한다.

### node centrality
그래프 내 node의 중요도로, 기준에 따라 다양하게 정의할 수 있다.

- eigenvector centrality
: 중요한 node 사이에 둘러싸인 node일수록 중요
![image](https://user-images.githubusercontent.com/41580746/181077346-c1b30e52-f8bd-4a65-96a9-53395f825b3c.png)
- betweenness centrality
: 다른 node 사이 많은 최단 경로가 있는 node일수록 중요
![image](https://user-images.githubusercontent.com/41580746/181077478-8d19a5fe-87b1-4cbb-869d-40e14274bbbb.png)
- closeness centrality
: 다른 모든 node에 대한 경로 길이가 짧을수록 중요 (중심에 있을수록 다른 node로 가는 길이 짧다)
![image](https://user-images.githubusercontent.com/41580746/181077510-2b904bd7-8eaa-4e3e-830f-cbcf0b2dd3cc.png)

### clustering coefficient
node 주변 구조 (local structure) 를 고려하여, 이웃 node들끼리 얼마나 연결되어 있는지 측정한다.
![image](https://user-images.githubusercontent.com/41580746/181077557-223fa5d9-3953-47e1-9406-0e2245019b00.png)

### graphlets : rooted connected non-isomorphic subgraphs
node 개수에 따라 각 node가 취할 수 있는 위치를 나타낸다.
![image](https://user-images.githubusercontent.com/41580746/181077630-c6e6d77b-1cbd-4920-9f24-22b8955b85c3.png)

- GDV (Graphlet Degree Vector)
: 해당 node에서 rooted된 graphlet 개수 (Graphlet-based features for nodes)

```
- degree는 edge 개수를,
- clustering coefficients는 triangle 개수를,
- GDV는 graphlet 개수를 나타낸다.
```

# 2-2. Link-level Prediction

link-level 예측은 기존 link들에 기반하여 새로운 link들을 예측하는 task로,
<br>모든 node pair들에 대해 순위를 매기고 그 중 상위 K개 node pair를 link로 예측하는 식으로 진행된다.
<br>따라서 node pair들에 대한 feature를 구상하는 것이 핵심이다.

link prediction task는 세부적으로 두 가지로 나눌 수 있다.
- link missing at random
: 랜덤으로 지워진 link 예측
- links over time
: 시간이 지남에 따라 진화되는 네트워크가 있을 때, 시간 경과에 따라 link 예측
(미래에 나타날 link 순위 목록 생성)

사용될 수 있는 feature들은 다음과 같다.

### distance-based features
두 node 사이 최단 거리를 나타내는 feature로,
<br>간단하지만 이웃 node 개수 등을 담지 못하는 단점이 있다.

![image](https://user-images.githubusercontent.com/41580746/181077938-9a8ee036-fd79-4958-8b02-aa863649d183.png)

### local neighborhood overlap
공통으로 공유하는 이웃 node 개수와 관련된 feature로, 표현 방식에 따라 크게 3가지로 나눌 수 있다.

- Common neighbors : node pair 간 공통 node 개수
- Jaccard's coefficient : Common neighbors의 정규화된 버전
- Adamic-Adar index : 이웃이 많은 node보다 이웃이 적은 node가 주변에 있는 것이 중요함을 나타낸 지표
![image](https://user-images.githubusercontent.com/41580746/181078174-37692a0d-5cb0-4923-b4d0-af0b587a59b9.png)

하지만 node pair가 공통으로 가지는 이웃이 없다면 해당 metric은 항상 0인 단점이 있다.

### global neighborhood overlap
local neighborhood overlap의 단점을 보완하고자, 그래프 전체의 구조를 반영한 feature이다.
<br>대표적으로 node pair 사이의 모든 경로 수를 계산한 Katz index가 있다.

![image](https://user-images.githubusercontent.com/41580746/181078260-f8b532c7-a3a1-4529-96dc-db6ec8f6e1ab.png)

# 2-3. Graph-level Prediction

Graph-level 예측은 전체 그래프의 구조를 나타내는 feature를 기반으로 생성된다.
<br>해당 feature들을 알아보기 전에, 먼저 kernel 개념에 대해 알아보고자 한다.

### Kernel & Kernel matrix
- Kernel - K(G, G')
: 그래프 간 유사도나 datapoint 간 유사도를 측정하여 실수값을 반환한다.
- Kernel matrix
: 모든 datapoint 쌍이나 그래프 쌍 간의 유사도를 측정하는 행렬로, 항상 양수 값(positive eigenvalues)을 가지며 대칭 행렬이다.

### Graph Kernel
- 핵심 아이디어 : BoW(Bag-of-Words: 문서에 대한 단어 개수를 feature로 사용)
- 그래프에서는 단어를 node로 간주하여 그래프의 node 개수를 feature로 사용할 수 있다.
  ![image](https://user-images.githubusercontent.com/41580746/181078663-d9d8d173-945a-4295-b150-17f489416e0a.png)
- node 대신 node degree로 한다면, bag of nodes에서는 동일한 feature인 그래프가 서로 다른 feature를 나타낼 수 있다.
  ![image](https://user-images.githubusercontent.com/41580746/181078713-dfe6bc4f-7a21-4732-87f8-1e5777d19a2a.png)

### Graphlet Kernel
- 그래프 내 다른 graphlet 개수를 나타내는 feature
  ![image](https://user-images.githubusercontent.com/41580746/181078791-18488523-30fe-4c70-bffd-9546119aebdd.png)
- 예를 들어, k=3이라면 점 3개를 통해 가질 수 있는 graphlet이 해당 그래프에 존재하는 개수를 나타낸다.
  ![image](https://user-images.githubusercontent.com/41580746/181078839-1578d860-0fca-4fdb-a34c-daa5298e551d.png)
- kernel K(G, G')의 경우 두 feature에 대한 내적값으로 구할 수 있다.
- 하지만 graphlet 개수를 세는 비용이 많이 든다는 단점이 있다. (ex: size n인 그래프에서 size k의 graphlet를 세기 위해서는 n^k번의 계산이 필요하다.)

### Weisfeiler-Lehman Kernel
- 효율적인 그래프 설명 feature를 디자인 하는 것을 목표로, 이웃 구조(neighborhood structure)를 사용한다.
- bag of node degrees가 one-hop neighborhood만 보는 local한 정보를 담았다면, 해당 feature는 bag of node degrees의 generalized된 버전으로 볼 수 있다.
- 그래프 node에 대한 color refinement를 통해 진행되며, 계산 비용이 적게 든다. (각 과정에서의 color refinement 복잡도는 link 개수에 linear함)
- 과정
  1. 초기 color 할당
  2. 이웃 node에 대한 color 모으기
  3. hash table에 따라 color 재할당
  4. 2-3번 과정에 대해 k번 반복
  5. color별 node 개수 count
  6. 두 feature에 대한 내적으로 kernel 값 계산
