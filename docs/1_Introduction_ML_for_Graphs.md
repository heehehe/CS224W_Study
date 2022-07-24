# 1-1. Why Graphs

### Graph란?
- entity들을 관계(relations) 및 상호작용(interactions)과 함께 묘사/분석하기 위한 general language
  <br>(entity들을 isolated datapoints로 보기보다는, entity 사이의 networks나 relation의 측면으로 보는 것)
- ex) computer networks, 질병 감염 경로, 논문 인용 네트워크, knowledge graph, code graphs, 3D shape 등
- 2가지 종류
  - networks (natural graphs) : underlying domains가 자연스럽게 그래프로 표현
    - ex) social networks, communications, transactions (phone call, financial transactions)
  - graphs (as a representation) : relational 구조를 갖고 있는 것을 표현
    - ex) information, knowledge, software

### Graph : new frontier of Deep Learning
- 복잡한 도메인은 relational graph로 표현할 수 있는 풍부한 relational structure를 갖고 있다.
- 명시적으로 relationships을 모델링하여 더욱 정확히 예측할 수 있다.
- 최신 deep learning toolbox는 simple data types에 특화되어 있다.
  - ex1) sequence : linear structure 지님 - text, speech에서 사용
  - ex2) grid : 이미지를 resize해서 fixed size grids로 표현 가능
- 하지만 graph & networks는 더 복잡한 data이다.
  <br>: arbitrary size & complex topological structure
  - grid나 text 등이 갖고 있는 공간적 위치(spatial locality)가 없음
    <br>: text에서는 좌/우 알고 grid에서는 위/아래 알지만 graph에서는 그런 reference point가 없음
  - 고정된 node 순서가 없음
  - dynamic하고 multimodal한 features 가짐
- graph가 neural network를 광범위하게 적용시키기 위한 딥러닝의 개척지
  <br>: input으로 graph 넣어서 여러 예측을 end-to-end로 할 수 있는 neural network architecture 에 대해 알아볼 것
![image](https://user-images.githubusercontent.com/41580746/180640650-a4e6703e-9a2d-4dea-8814-05c07d156233.png)

### Representation Learning
- 이전 머신러닝 접근법은 feature engineering에 많은 노력이 들어간다.
- representation learning
  - graph에서 자동으로 feature들을 추출하고 학습
  - feature engineering 단계는 제거
  - 자동으로 graph에 대한 좋은 representation을 학습
    <br>→ 다른 머신러닝 알고리즘 task (downstream task) 에 사용 가능
![image](https://user-images.githubusercontent.com/41580746/180640705-e466c9b7-6304-4c4c-a46f-1e7f69263145.png)

- 목표 : node들을 d차원 실수 벡터로 표현해주는 함수 학습하기 → node들을 맵핑해서 d차원 임베딩 벡터 생성
  <br>→ 네트워크 내 유사한 node들이 임베딩 공간에서 가까이 임베딩될 수 있도록 함
![image](https://user-images.githubusercontent.com/41580746/180640724-c786e03b-80fe-4821-bd53-83f8bd930439.png)

# 1-2. Applications of Graph ML
1. node level task
    - node classification : node의 property 예측
      - ex) 온라인 사용자 분류
2. edge level task
    - link prediction : node 사이 link 비어있을 때, link가 존재할지 예측
      - ex) knowledge graph completion
3. graph level task
    - graph classification : 다른 graph 분류
      - ex) 분자 속성 예측
4. community level task (subgraph level task)
    - clustering : node들이 커뮤니티 형성하는지 판별
      - ex) social circle detection
5. generation level task
    - graph generation : 그래프 생성 예측
      - ex) drug discovery (새로운 분자 구조 생성 예측)
    - graph evolution : 그래프 진화 예측
      - ex) physical simulation (다양한 현상에 대해 정확한 시뮬레이션 돌려볼 때)

# 1-3. Choice of Graph Representation

### Graph 기본 구조
![image](https://user-images.githubusercontent.com/41580746/180640979-754f1313-77d8-446c-870e-4a8c4fb9192a.png)
- nodes, vertices : 정점
- links, edges : 간선 (node들을 연결해 주는 link)
- network, graph : node와 edge로 이루어진 system

### Graph 종류
![image](https://user-images.githubusercontent.com/41580746/180640992-a6943de1-2e4b-4464-821c-396d2e66743e.png)
- Directed Graph : link가 방향이 있는 그래프
- Undirected Graph : link에 방향이 없는 symmetric한 그래프

![image](https://user-images.githubusercontent.com/41580746/180641010-e19a08fb-9bb6-4d1a-9549-c24ced82671d.png)
- Bipartite Graph : node들이 두 set으로 나눠지고, 각 set에서의 node는 서로 연결되어 있지 않은 그래프
  - ex) authors-to-papers, actors-to-movies
  - 각 set에 대해서 projection 시킬 수 있음

![image](https://user-images.githubusercontent.com/41580746/180641024-435f1c5e-e52b-420f-8c59-1df0399f15e4.png)
- Weighted Graph : node나 edge에 가중치 부여 → 인접 행렬에서 0 / 1 대신 가중치로 표현
- Unweighted Graph : node나 edge에 가중치 부여 X

![image](https://user-images.githubusercontent.com/41580746/180641029-ca0b8e85-26c1-4389-842a-f7af6b07f40a.png)
- self-edges (self-loops) : 자신의 node로 향하는 edge가 있는 그래프 
- multigraph : 같은 node끼리 edge가 여러개 있는 그래프 → 인접 행렬에서 0 / 1 대신 edge 개수로 표현

### Graph 표현 방식
- 인접 행렬 (Adjacency Matrix)
  ![image](https://user-images.githubusercontent.com/41580746/180641042-f443dd46-61d3-456d-9d90-db8b66e0a98d.png)
  - matrix 내 (i, j) 원소값이 1이면, node i와 node j 사이 link가 있는 것
  - matrix 내 (i, j) 원소값이 0이면, node i와 node j 사이 link가 없는 것
  - undirected graph는 인접 행렬이 symmetric함
  - 실제 대부분 네트워크는 sparse하므로, 인접 행렬도 대부분 0으로 채워져 있음
- 인접 리스트 (Adjacency list)
  ![image](https://user-images.githubusercontent.com/41580746/180641055-3c58dc3b-c710-45e0-bc55-30feab78c0b7.png)
  - node별로 연결된 node 목록 나열
  - 그래프가 크거나 sparse할 때 유용
  - 빠르게 주어진 node에 대한 이웃을 찾을 수 있음
- node / edge attributes
  - weight : 가중치
  <br>ex) 커뮤니케이션 빈도 수, ...
  - ranking : 중요도 랭킹
  <br>ex) best friend, second best friend, ...
  - type : 형식
  <br>ex) friend, relative, co-worker, ...
  - sign : 표시
  <br>ex) 친구인지 적인지, 신뢰하는지 불신하는지, ...
  - 그래프 구조에 기반한 특성
  <br>ex) 공통 친구 수, ...
