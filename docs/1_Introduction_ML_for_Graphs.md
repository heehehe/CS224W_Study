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
