## 1. 프로젝트 개요

본 프로젝트의 목표는 대규모 ArXiv 논문 메타데이터를 이용해

1. 연구 토픽의 등장과 쇠퇴 추세
2. 핵심 용어의 의미 및 사용 맥락 변화

위 두 가지를 정량적으로 분석하고, 이를 SBERT 기반 임베딩과 클러스터링, 시맨틱 드리프트 분석, 시각화로 보여주는 것입니다.

전체 파이프라인은 크게 다음 네 단계로 구성됩니다.

1. 데이터 전처리와 SBERT 임베딩 생성
2. 슬롯별 토픽 클러스터링과 키워드 추출
3. 시간에 따른 토픽 볼륨 추세 및 토픽 계승 관계 계산
4. SBERT 기반 의미 변화 계산과 통합 시각화(figures)

각 단계는 독립된 스크립트로 구현되어 있으며, 동일한 중간 산출물(슬롯 파케이 파일, FAISS 메타데이터 등)을 공유함으로써 메모리 사용을 통제하고 재현성을 확보했습니다.

---

## 2. 데이터와 전처리

### 2.1 데이터 소스와 기본 필터링

* 입력 데이터는 `arxiv-metadata-oai-snapshot.json` 전체 스냅샷입니다.
* `data_preprocessing.py`의 `load_raw_data()`에서 JSON lines를 읽고, 캐시 파일(`arxiv_raw_cache.pkl`)로 저장해 반복 실행 시 I/O를 줄입니다.
* `extract_and_clean()` 단계에서
  * `created` 혹은 `versions`에서 연도를 정규식으로 추출해 `year` 컬럼을 생성
  * 제목과 초록을 합쳐 `text` 컬럼 생성
  * 최소 길이(`MIN_TEXT_LENGTH = 50`) 미만 문서는 제거
  * 소문자 변환, 특수문자 제거, 길이 3 미만 토큰 제거 등 최소 전처리 후 `text_processed` 컬럼 생성
* 이 결과는 `extracted.parquet`에 저장되어 이후 모든 단계의 공통 입력이 됩니다.

### 2.2 시간 슬롯 분할과 SBERT 임베딩

* `split_into_slots()`에서 3년 단위 슬롯(`SLOT_SIZE = 3`)로 데이터를 나눕니다.
  예: `1986_1988`, `1989_1991` 와 같은 형태의 `slot` 문자열 컬럼을 추가합니다.
* `compute_embeddings_batch_slotwise()`는 SentenceTransformers의 `all-mpnet-base-v2` 모델을 사용해 슬롯별로 배치 인코딩을 수행합니다.
  * `title`과 `abstract`를 각각 임베딩한 뒤 평균
  * L2 정규화 후 float32로 슬롯별 `.npy` 파일에 memmap 방식로 저장
* `save_embeddings_and_build_faiss_from_files()`에서
  * 슬롯별 임베딩 파일을 읽어 FAISS IndexFlatIP 인덱스를 생성
  * 슬롯별 메타데이터(`ids`, 차원, 벡터 수, 인덱스 경로)를 모아 `slot_metadata.pkl`로 저장
  * 동시에 슬롯별 메타데이터 parquet(`slot_{slot}.parquet`)를 `dataset/slots`에 저장
이로써, 이후 단계는 모두 슬롯별 parquet + FAISS 메타데이터를 기준으로 동작합니다.

---

## 3. 토픽 클러스터링과 키워드 추출

### 3.1 슬롯별 UMAP + HDBSCAN 클러스터링

* `clustering_runner.py`는 `dataset/slots/slot_*.parquet`와 `slot_metadata.pkl`을 읽습니다.
* `run_umap_hdbscan_per_slot()`에서 슬롯별로 다음을 수행합니다.
  * 슬롯별 임베딩을 `np.load(..., mmap_mode="r")`로 메모리 효율적으로 로드
  * UMAP으로 저차원(latent_dim)으로 축소
  * HDBSCAN으로 밀도 기반 클러스터링 수행
    * noise 포인트는 `cluster = -1`로 레이블링
* 결과 클러스터 라벨은 원래 슬롯 파케이의 `cluster` 컬럼으로 저장됩니다.
* 따라서 이후 스크립트는 모두 동일한 슬롯 파케이를 공유합니다.
* 코드 상에서는 Optuna 등 자동 하이퍼파라미터 탐색은 사용하지 않고, 고정된 설정으로 클러스터링을 수행하고 있습니다.
  * 데이터셋의 크기와 학습을 수행한 기기의 성능을 반영해 AutoML 서치는 생략했습니다.
* 예시 이미지 (2007 2009 슬롯에서 SBERT 임베딩을 UMAP으로 투영하고 HDBSCAN으로 클러스터링한 결과)
  ![image](/etc/umap_slot_2007_2009.png)

### 3.2 c TF IDF 기반 키워드 추출
* `topic_keywords_builder.py`는 슬롯별 파케이에서 `text_processed`, `cluster`를 사용합니다.
* `extract_keywords_for_slot()`에서
  * noise 클러스터(`cluster = -1`)를 제거
  * 클러스터별 문서를 하나의 긴 문서로 합친 뒤,
    `compute_c_tf_idf()`로 클래스 기반 TF IDF(c TF IDF)를 계산
  * 각 클러스터마다 TF IDF 값이 큰 상위 N개 토큰을 키워드로 반환
* 결과는 슬롯별로
  * `{slot}_keywords.json`
  * `{slot}_keywords.csv`
    두 형식으로 `results/topic_keywords`에 저장됩니다.
* 현재 코드 기준으로는 `CountVectorizer()`에 별도의 도메인 stopword 리스트를 넣지는 않고, 전처리 단계의 소문자화, 특수문자 제거, 길이 필터링만 적용되어 있습니다.
* 예시 이미지
  * 클러스터별 상위 키워드 집합 간 Jaccard 유사도를 히트맵으로 나타내면 아래와 같이 대각선 이외에는 전반적으로 낮은 중복도를 보인다
  ![image](/etc/keyword_overlap_2007_2009.png)

---

## 4. 토픽 볼륨 추세와 토픽 계승 관계
### 4.1 클러스터 볼륨 계산

* `topic_volume_trend_generator.py`
  * `cluster_volume.csv`와
  * 슬롯 간 클러스터 매칭 결과인 `cluster_matching.json`
    을 생성합니다.
* `compute_cluster_centroids()`에서 슬롯별로
  * `cluster` 그룹을 기준으로 SBERT 임베딩의 평균을 구해 클러스터 센트로이드를 계산합니다.
* `compute_volume()`는 각 클러스터별 문서 수를 세어 `volume`으로 저장합니다.
* 슬롯 이름에서 시작 연도와 끝 연도를 이용해, 연도 축에는 슬롯 중심값(예: `(1996+1998)/2`)을 사용합니다.

### 4.2 슬롯 간 클러스터 매칭과 topic chain

* `match_clusters_between_slots()`에서
  * 연속된 두 슬롯 A, B에 대해
  * 모든 클러스터 쌍의 코사인 유사도(센트로이드 벡터의 내적)를 계산하고
  * Greedy하게 가장 유사한 클러스터를 매칭하되, 일정 임계값 미만이면 매칭하지 않습니다.
* 이 결과가 `cluster_matching.json`에 저장됩니다.
* `composite_topic_figure.py` 내부의 `build_topic_chains()`는
  * `cluster_volume.csv`와 `cluster_matching.json`을 읽고
  * 역방향 매핑으로 “부모 클러스터가 없는” 클러스터들을 chain 시작점으로 잡은 뒤
  * `forward` 매핑을 따라가며 시간에 따라 이어지는 토픽 체인(topic chain)을 구성합니다.
* 체인 길이가 일정 이상인 경우만 사용하며, 각 체인의 총 volume 합을 기준으로 상위 토픽을 선택합니다.

---

## 5. SBERT 기반 의미 변화 분석

### 5.1 Anchor 단어 선택
* `semantic_shift_analysis.py`는 시맨틱 드리프트 계산을 위한 anchor 단어 집합을 구축합니다.
* 슬롯별 `text_processed`를 기반으로 `TfidfVectorizer`로 전체 코퍼스의 단어 분포를 분석하고,
  * 최소 등장 문서 수(`MIN_GLOBAL_FREQ`) 이상
  * 모든 사용 슬롯에서 등장하는 단어를 후보로 삼습니다.
* 여기에 단순 stopword 리스트를 적용해 의미 없는 일반 단어는 제외합니다.

### 5.2 슬롯별 의미 표현과 드리프트 지표
* SBERT는 시간에 따라 파라미터가 변하지 않는 정적 인코더이므로,
  diachronic Word2Vec처럼 공간 정렬을 다시 하지 않고, 슬롯별 국지적 분포 변화를 이용합니다.
* `compute_semantic_drift()`에서 각 anchor 단어에 대해 슬롯을 순차적으로 돌면서 다음을 계산합니다.
  1. 해당 단어를 포함하는 문서들의 SBERT 임베딩 평균을 단어의 슬롯별 centroid로 사용
  2. 해당 슬롯의 FAISS 인덱스에서 centroid 주변 이웃 문서들을 검색하고, 이웃의 `text_processed`에서 상위 키워드를 추출해 국지적 이웃 집합으로 사용
  3. 인접 슬롯 간
     * centroid 변위 기반 지표
       `cosine_change = 1 − cos(prev_centroid, cur_centroid)`
     * 이웃 집합의 Jaccard 유사도 기반 지표
       `jaccard_change = 1 − Jaccard(prev_neighbors, cur_neighbors)`
       를 누적해 평균값을 산출
* 최종적으로 의미 변화가 큰 단어 순으로 정렬하고, `results/sbert_semantic_shift_words.csv`에 저장합니다.
* 예시 이미지
  * 상위 단어의 분포와 cosine vs combined drift의 관계

---

## 6. 통합 시각화와 보고서용 그림

### 6.1 종합 figure 생성

* `composite_topic_figure.py`는 가장 대표적인 topic chain 하나를 선택해 세 가지 관점을 한 번에 보여주는 그림을 생성합니다.

  1. Slot별 요약(Summary 패널)
     * 해당 토픽에 속한 문서들의 `text_processed`를 합쳐
       로컬에 로드한 DistilBART 요약 모델로 한 번 더 영어 요약을 수행합니다.
       모델 로딩이 실패하거나 길이 제한 등에 걸릴 경우,
       TF IDF 기반 대표 문장 선택으로 fallback 합니다.
  2. Keyword Timeline 패널
     * `results/topic_keywords/{slot}_keywords.json`에서
       슬롯별 상위 키워드를 가져와 y축 키워드, x축 슬롯 위치에 점을 찍어
       키워드 등장 시점을 시각화합니다.
  3. Word Cloud Evolution 패널
     * 슬롯별 키워드 집합을 WordCloud로 그려
       한 토픽이 시간에 따라 어떤 단어들로 설명되는지 직관적으로 보여줍니다.
* 하나의 대형 종합 그림(`topic_{id}_composite.png`)과 별도 파일로 저장합니다다
  * 요약만 따로(`topic_{id}_summary.png`)
  * 타임라인만 따로(`topic_{id}_timeline.png`)
  * 워드클라우드만 따로(`topic_{id}_wordclouds.png`)

* 예시 이미지
  * 실제 한 토픽이 SCADA > smart grid > renewable energy market으로 이동하는 모습을 보여줍니다
  ![image](/etc/topic_926_summary.png)
  ![image](/etc/topic_926_timeline.png)
  ![image](/etc/topic_926_wordclouds.png)
    

### 6.2 시맨틱 드리프트 상위 단어 및 토픽 트렌드

* 별도의 시각화 스크립트에서
  * `sbert_semantic_shift_words.csv`의 상위 단어를 바 차트로
  * `cluster_volume.csv`와 topic chain을 이용한 상위 토픽의 volume 추세를 스무딩된 라인 차트로
    출력해, “어떤 토픽이 언제 부상 혹은 쇠퇴했는지, 그 과정에서 어떤 핵심 용어들이 의미 변화를 보였는지”를 함께 해석할 수 있도록 구성했습니다.

---

## 7. 이 프로젝트의 해석과 한계
### 7.1 해석 가능성

* SBERT와 FAISS, HDBSCAN을 결합해 아래 세 가지를 하나의 일관된 파이프라인에서 계산합니다.
  * 문서 수준 임베딩
  * 토픽 수준 센트로이드와 볼륨 변화
  * 단어 수준 의미 변화 지표

### 7.2 한계

* 다음과 같은 한계가 존재합니다.

* 클러스터링 하이퍼파라미터는 고정값이며, 자동 탐색이나 정량 평가 지표(ARI, NMI 등)는 구현돼 있지 않습니다.
* 슬롯의 폭이 3년으로 고정되어 있어, 연 단위의 미세한 변화까지는 포착하기 어렵습니다.
* SBERT는 정적 인코더이므로, 진짜 diachronic embedding(연도별 Word2Vec 학습 후 정렬)과는 다르게 문서 분포의 변화와 코퍼스 맥락 변화를 기반으로 의미 변화를 추정합니다. 코드 상에서 Procrustes 정렬이나 연도별 별도 임베딩 학습은 수행하지 않습니다.
* topic_keywords_builder는 일반적인 전처리를 제외하면 도메인 특화 stopword 제거는 별도로 하지 않습니다.

* 기타 문제 
  * 데이터 분석 및 처리는 완료하였는데 이를 설명할 시각화 코드 준비가 미비합니다. 해당 부분 추가가 필요합니다.
  * 스파게티 코드입니다. 파서 및 config 정리가 필요합니다.
---

## 8. 코드 내 구현 정리


1. `data_preprocessing.py`

   * 3년 단위 슬롯 분할 및 SBERT 임베딩
     * `Config.SLOT_SIZE = 3`
     * `compute_embeddings_batch_slotwise()`에서 `SentenceTransformer('all-mpnet-base-v2')` 사용
     * 임베딩을 슬롯별 `embeddings_{slot}.npy`로 저장
   * `save_embeddings_and_build_faiss_from_files()`에서
     * `faiss.IndexFlatIP(dim)` 생성
     * 슬롯별 인덱스와 메타데이터를 `slot_metadata.pkl`로 저장
     * 슬롯별 메타 데이터를 `slot_{slot}.parquet`로 저장

3. `clustering_runner.py`

   * UMAP + HDBSCAN 클러스터링
     * `run_umap_hdbscan_per_slot()`에서 UMAP으로 축소 후 HDBSCAN 수행
     * noise는 `cluster = -1`

4. `topic_keywords_builder.py`

   * c TF IDF 기반 키워드 추출
     * `compute_c_tf_idf()`에서 `CountVectorizer`와 `TfidfTransformer` 사용
     * 클러스터별 문서를 하나로 합친 뒤 c TF IDF 계산
     * 상위 TF IDF 토큰을 키워드로 선택

5. `topic_volume_trend_generator.py`

   * 클러스터 볼륨과 슬롯 간 매칭 
     * `compute_cluster_centroids()`에서 SBERT 임베딩 평균으로 센트로이드 계산
     * `compute_volume()`에서 클러스터별 문서 수 계산
     * `match_clusters_between_slots()`에서 코사인 유사도 기반 greedy 매칭 구현

6. `composite_topic_figure.py`

   * topic chain 구성
     * `cluster_volume.csv`와 `cluster_matching.json`을 기반으로

7. `semantic_shift_analysis.py`

   * SBERT 기반 semantic drift 계산
     * anchor 선정, FAISS 인덱스 통한 neighbor 검색
     * 슬롯 간 centroid 코사인 변화와 neighbor Jaccard 변화 계산
     * `combined_score = 0.6 * neighbor + 0.4 * cosine`

8. `composite_topic_figure.py`

   * 통합 figure 및 개별 figure 저장

     * `generate_composite()`에서 `topic_{id}_composite.png` 생성

     * 상단에서 DistilBART 요약 모델을 transformers로 로드
     * 실패 시 TF IDF 기반 대표 문장 선택으로 fallback

