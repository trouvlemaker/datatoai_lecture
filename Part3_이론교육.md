# Part 3: 데이터 분석 자동화 — RAG Chatbot 이론 교육 핸드아웃

---

## 1. RAG (Retrieval-Augmented Generation)

### 1.1 LLM의 한계와 RAG의 필요성

LLM(Large Language Model)은 학습 데이터에 포함된 일반 지식은 잘 답변하지만, 다음과 같은 한계가 있다:
- **최신 정보 부재**: 학습 이후의 데이터를 모름
- **사내 데이터 부재**: 회사 내부 매출 데이터, 분석 보고서를 모름
- **환각(Hallucination)**: 모르는 내용을 그럴듯하게 지어냄

RAG는 이 한계를 해결한다: **질문이 들어오면 관련 문서를 먼저 검색하고, 검색 결과를 LLM에게 함께 전달**하여 답변을 생성한다.

```
질문 → 검색(Retrieval) → 관련 문서 + 질문을 LLM에 전달 → 답변 생성(Generation)
```

### 1.2 RAG의 두 가지 검색 경로

이 프로젝트에서는 질문 유형에 따라 두 가지 검색 경로를 사용한다:

| 경로 | 데이터 소스 | 검색 방법 | 질문 유형 |
|------|-----------|----------|----------|
| **CSV 경로** | Part 1에서 생성한 CSV 파일 | DuckDB (SQL) | "연도별 매출 합계는?", "Top 5 매장은?" |
| **MD 경로** | Part 1에서 생성한 MD 요약 파일 | ChromaDB (벡터 검색) | "프로모션 전략을 추천해줘", "성장 둔화 원인은?" |

> **더 알아보기**
> - [RAG 개념 설명 — AWS](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
> - [LangChain RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/)

---

## 2. 벡터 데이터베이스와 임베딩

### 2.1 임베딩(Embedding)이란

텍스트를 **고차원 숫자 벡터**로 변환하는 것이다. 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치한다:

```
"매출이 증가했다" → [0.12, -0.34, 0.56, ...]  ─┐ 가까움
"판매가 늘었다"   → [0.11, -0.33, 0.55, ...]  ─┘

"날씨가 좋다"     → [-0.45, 0.78, -0.12, ...] ← 멀리 떨어짐
```

이 프로젝트에서는 `multilingual-e5-small` 모델을 사용한다. 다국어(한국어 포함)를 지원하면서 경량이라 로컬에서 빠르게 실행된다.

### 2.2 ChromaDB

ChromaDB는 **벡터 데이터베이스**로, 임베딩된 문서를 저장하고 유사도 검색을 수행한다:

1. **저장**: MD 문서를 청크로 나누고 → 각 청크를 임베딩 → ChromaDB에 저장
2. **검색**: 질문을 임베딩 → 저장된 벡터와 유사도 비교 → 가장 유사한 문서 반환

### 2.3 DuckDB와 Text-to-SQL

DuckDB는 **인메모리 관계형 데이터베이스**로, CSV 파일을 직접 SQL로 쿼리할 수 있다:

```sql
SELECT year, SUM(sales) FROM train GROUP BY year ORDER BY year
```

Text-to-SQL은 자연어 질문을 SQL로 변환하는 것이다:
- 사용자: "연도별 매출 합계는?"
- LLM이 SQL 생성: `SELECT year, SUM(sales) FROM ...`
- DuckDB가 실행 → 결과 반환

> **더 알아보기**
> - [ChromaDB 공식 문서](https://docs.trychroma.com/)
> - [DuckDB 공식 문서](https://duckdb.org/docs/)
> - [HuggingFace Embeddings 가이드](https://huggingface.co/blog/getting-started-with-embeddings)

---

## 3. LLM 라우팅 (Routing)

### 3.1 왜 라우팅이 필요한가

두 가지 검색 경로(CSV/MD)가 있으므로, 질문이 들어왔을 때 **어느 경로로 보낼지 결정**해야 한다. 이것이 라우팅이다.

### 3.2 라우팅 방식의 진화

| 방식 | 동작 | 장단점 |
|------|------|--------|
| **규칙 기반** | 키워드 매칭 ("합계", "비교" → CSV) | 단순하지만 유연성 부족 |
| **LLM 프롬프트** | LLM에게 "CSV인지 MD인지 판단해줘" | 유연하지만 출력 형식 불안정 |
| **Tool Calling** | LLM이 도구를 직접 선택 | 안정적 + 유연, API 수준 지원 |

### 3.3 Tool Calling이란

LLM이 **사용 가능한 도구(함수) 목록**을 받고, 질문에 적합한 도구를 **구조화된 형식(JSON)**으로 선택하는 기능이다:

```
입력: "연도별 매출 합계는?"
도구 목록: [tool_search_csv, tool_search_md]

LLM 출력: { "name": "tool_search_csv", "args": {"question": "연도별 매출 합계"} }
```

Tool Calling의 핵심 장점:
- LLM이 자유 텍스트가 아닌 **구조화된 JSON**으로 응답 → 파싱 오류 없음
- 도구의 docstring을 읽고 **의미 기반으로 판단** → 키워드 매칭보다 정확
- OpenAI, Anthropic 등 주요 LLM API가 기본 지원

> **더 알아보기**
> - [OpenAI Function Calling 가이드](https://platform.openai.com/docs/guides/function-calling)
> - [LangChain Tool Calling 문서](https://python.langchain.com/docs/concepts/tool_calling/)

---

## 4. LangGraph와 에이전트 아키텍처

### 4.1 LangGraph란

LangGraph는 LangChain 팀이 만든 **LLM 에이전트 워크플로우 프레임워크**다. 노드(Node)와 엣지(Edge)로 구성된 **그래프**로 에이전트의 동작 흐름을 정의한다:

- **노드(Node)**: 각 처리 단계 (라우팅, 검색, 답변 생성 등)
- **엣지(Edge)**: 노드 간 연결 (다음에 어디로 갈지)
- **조건부 엣지**: 상태에 따라 다른 노드로 분기
- **상태(State)**: 그래프 전체에서 공유하는 데이터

### 4.2 왜 LangGraph를 사용하는가

단순한 "질문 → 검색 → 답변" 파이프라인은 LCEL(LangChain Expression Language)로 충분하다. 하지만 다음과 같은 복잡한 흐름이 필요하면 LangGraph가 적합하다:
- 검색 결과가 부족하면 **다시 검색** (반복)
- 질문이 모호하면 **재질문** (분기)
- 여러 도구를 **순차적으로 호출** (다단계)

> **더 알아보기**
> - [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
> - [LangGraph 튜토리얼 (LangChain Blog)](https://blog.langchain.dev/langgraph/)

---

## 5. ReAct 패턴

### 5.1 ReAct란

ReAct(Reasoning + Acting)는 LLM 에이전트의 대표적인 패턴이다. 핵심은 **생각하고(Reason) → 행동하고(Act) → 관찰하고(Observe) → 다시 생각하는** 반복 루프다:

```
Plan(생각) → Act(도구 실행) → Observe(결과 확인) → Plan(다시 생각) → ...
```

사람이 문제를 푸는 방식과 동일하다:
1. "매출 데이터가 필요하니까 SQL로 조회하자" (Plan)
2. SQL 실행 (Act)
3. "결과가 나왔는데, 추가로 인사이트 문서도 필요하겠다" (Observe → Plan)
4. 문서 검색 (Act)
5. "충분한 정보가 모였으니 답변을 작성하자" (Plan → 최종 답변)

### 5.2 이 프로젝트에서의 ReAct 구현

```
질문 → Plan(구체성 판단 → 도구 선택) → Act(도구 실행) → Plan → ... → 최종 답변
         ↓ (모호하면)
       재질문 → 사용자 응답 → 질문 재작성 → Plan
```

- **Plan 노드**: Tool Calling으로 다음 행동 결정 (query_sql / search_md / final / clarify)
- **Act 노드**: 선택된 도구 실행
- **Generate 노드**: 수집된 정보로 최종 답변 생성
- **Clarify → Rewrite**: 모호한 질문 처리

> **더 알아보기**
> - [ReAct 논문 (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
> - [ReAct 패턴 설명 (Prompt Engineering Guide)](https://www.promptingguide.ai/techniques/react)

---

## 6. Streamlit 챗봇 UI

### 6.1 Streamlit이란

Streamlit은 Python으로 **웹 앱을 빠르게 만드는 프레임워크**다. HTML/CSS/JavaScript 없이 Python 코드만으로 인터랙티브한 UI를 구현할 수 있다.

### 6.2 챗봇 UI 구성

이 프로젝트의 Streamlit 챗봇은:
- `st.chat_input()`: 사용자 입력
- `st.chat_message()`: 대화 메시지 표시
- `st.session_state`: 대화 이력 유지
- `st.sidebar`: 디버그 정보 (라우팅 결과, 검색 결과 등)

> **더 알아보기**
> - [Streamlit 공식 문서](https://docs.streamlit.io/)
> - [Streamlit Chat 튜토리얼](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-a-basic-llm-chat-app)

---

## 7. 전체 아키텍처 요약

```
사용자 질문
    ↓
[Tool Calling 라우팅] → CSV 경로: DuckDB (SQL)
                      → MD 경로: ChromaDB (벡터 검색)
    ↓
[LangGraph ReAct 에이전트]
  Plan → Act → Observe → Plan → ... → Generate
    ↓
최종 답변 (Streamlit UI)
```

Part 1에서 생성한 데이터(CSV/MD)가 Part 3 챗봇의 지식 베이스가 되고, Part 2의 예측 모델 결과도 시나리오 분석 질문에 활용할 수 있다. 세 파트가 하나의 **데이터 → AI 파이프라인**으로 연결된다.
