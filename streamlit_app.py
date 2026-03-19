import streamlit as st
import os
import glob
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import duckdb
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- 설정 ---
RESULTS_PATH = './results'
CHROMA_PATH = './chroma_db_part3'
DUCKDB_PATH = './data/sales_analysis_v2.db'

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# --- DB 로드 ---
@st.cache_resource
def load_duckdb():
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    return conn

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

conn = load_duckdb()
vectorstore = load_vectorstore()

# --- 검색 함수 (노트북과 동일) ---

def get_table_schemas():
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    schema_parts = []
    for (table_name,) in tables:
        cols = conn.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{table_name}'"
        ).fetchall()
        col_str = ', '.join([f'{c[0]}({c[1]})' for c in cols])
        schema_parts.append(f'테이블: {table_name}\n  컬럼: {col_str}')
    return '\n\n'.join(schema_parts)

def search_md(question, k=3):
    results = vectorstore.similarity_search_with_score(question, k=k)
    docs = [{'source': doc.metadata.get('source', ''), 'score': round(float(score), 4),
             'snippet': doc.page_content[:200]} for doc, score in results]
    context = '\n---\n'.join([doc.page_content for doc, _ in results])
    return {'type': 'md', 'docs': docs, 'context': context}

def search_csv(question):
    schema = get_table_schemas()
    sql_prompt = ChatPromptTemplate.from_messages([
        ('system', '당신은 DuckDB SQL 전문가입니다. 스키마를 참고하여 SQL을 작성하세요.\n\n{schema}\n\n규칙: SELECT 문만, SQL만 출력, 테이블명은 큰따옴표로'),
        ('human', '{question}')
    ])
    try:
        sql = llm.invoke(sql_prompt.format_messages(schema=schema, question=question)).content.strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()
        result_df = conn.execute(sql).fetchdf()
        return {'type': 'csv', 'sql': sql, 'rows': result_df.head(20).to_dict('records'), 'summary': result_df.to_string(index=False)}
    except Exception as e:
        return {'type': 'csv', 'sql': '', 'rows': [], 'summary': f'SQL 실행 실패: {str(e)}'}

def route_question(question):
    q = question.lower()
    csv_kw = ['얼마', '몇', 'top', '평균', '합계', '최대', '최소', '순위', '건수', '개수', '비율']
    md_kw = ['왜', '전략', '어떻게', '이유', '설명', '분석', '인사이트', '권장', '추천', '의미']
    csv_s = sum(1 for kw in csv_kw if kw in q)
    md_s = sum(1 for kw in md_kw if kw in q)
    if csv_s > 0 and md_s == 0:
        return {'route': 'csv', 'confidence': 0.9, 'reason': f'룰: CSV 키워드 {csv_s}개'}
    if md_s > 0 and csv_s == 0:
        return {'route': 'md', 'confidence': 0.9, 'reason': f'룰: MD 키워드 {md_s}개'}
    router_prompt = ChatPromptTemplate.from_messages([
        ('system', '질문을 csv, md, clarify 중 하나로 분류하세요. 하나만 출력.'),
        ('human', '{question}')
    ])
    result = llm.invoke(router_prompt.format_messages(question=question)).content.strip().lower()
    route = result if result in ['csv', 'md', 'clarify'] else 'md'
    return {'route': route, 'confidence': 0.7, 'reason': f'LLM: {result}'}

def generate_final(question, search_results):
    context_parts = []
    citations = []
    for r in search_results:
        if r['type'] == 'md':
            context_parts.append(f'[문서 검색]\n{r["context"]}')
            citations.extend([d['source'] for d in r.get('docs', [])])
        elif r['type'] == 'csv' and r['rows']:
            context_parts.append(f'[데이터 조회]\nSQL: {r["sql"]}\n{r["summary"]}')
    context = '\n\n'.join(context_parts) if context_parts else '검색 결과 없음'
    answer_prompt = ChatPromptTemplate.from_messages([
        ('system', '매출 데이터 분석 전문가로서 검색 결과를 바탕으로 한국어로 답변하세요.\n\n{context}'),
        ('human', '{question}')
    ])
    answer = llm.invoke(answer_prompt.format_messages(context=context, question=question)).content
    return {'answer': answer, 'citations': list(set(citations))}

# --- 3가지 방식 ---

def ask_langchain(question):
    md_r = search_md(question)
    csv_r = search_csv(question)
    return generate_final(question, [md_r, csv_r])

def ask_langgraph(question):
    routing = route_question(question)
    if routing['route'] == 'csv':
        result = search_csv(question)
    else:
        result = search_md(question)
    out = generate_final(question, [result])
    out['route'] = routing['route']
    out['reason'] = routing['reason']
    return out

def ask_react(question):
    routing = route_question(question)
    if routing['route'] == 'clarify':
        return {'needs_clarify': True, 'clarify_question': '추가로 어떤 정보가 필요한지 알려주세요.', 'route': 'clarify'}
    if routing['route'] == 'csv':
        result = search_csv(question)
    else:
        result = search_md(question)
    out = generate_final(question, [result])
    out['route'] = routing['route']
    out['needs_clarify'] = False
    return out

def ask_react_with_clarify(original_q, user_response):
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ('system', '원래 질문과 추가 정보를 결합하여 명확한 질문으로 재작성하세요. 질문만 출력.'),
        ('human', '원래 질문: {original}\n추가 정보: {response}')
    ])
    new_q = llm.invoke(rewrite_prompt.format_messages(original=original_q, response=user_response)).content
    return ask_react(new_q), new_q

# --- Streamlit UI ---

st.set_page_config(page_title='매출 분석 RAG Chatbot', page_icon='📊', layout='wide')
st.title('📊 매출 분석 RAG Chatbot')

method = st.sidebar.radio(
    '방식 선택',
    ['LangChain (둘 다 실행)', 'LangGraph (라우터 선택)', 'ReAct (재질문)'],
    index=1
)

st.sidebar.markdown('---')
st.sidebar.markdown('### 방식 설명')
if 'LangChain' in method:
    st.sidebar.info('MD + CSV 둘 다 검색 후 결과를 통합합니다.')
elif 'LangGraph' in method:
    st.sidebar.info('질문 유형을 판단하여 MD 또는 CSV 중 하나만 검색합니다.')
else:
    st.sidebar.info('질문이 모호하면 추가 질문 후 재실행합니다.')

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'waiting_clarify' not in st.session_state:
    st.session_state.waiting_clarify = False
if 'original_question' not in st.session_state:
    st.session_state.original_question = ''

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if st.session_state.waiting_clarify:
    clarify_response = st.chat_input('추가 정보를 입력하세요...')
    if clarify_response:
        st.session_state.messages.append({'role': 'user', 'content': clarify_response})
        with st.chat_message('user'):
            st.markdown(clarify_response)

        with st.chat_message('assistant'):
            with st.spinner('질문 재작성 후 검색 중...'):
                result, new_q = ask_react_with_clarify(st.session_state.original_question, clarify_response)
                response = f'**[재작성된 질문]**: {new_q}\n\n{result["answer"]}'
                st.markdown(response)

        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.session_state.waiting_clarify = False
else:
    user_input = st.chat_input('매출 데이터에 대해 질문하세요...')
    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            with st.spinner('검색 중...'):
                start = time.time()
                if 'LangChain' in method:
                    result = ask_langchain(user_input)
                elif 'LangGraph' in method:
                    result = ask_langgraph(user_input)
                else:
                    result = ask_react(user_input)
                elapsed = round(time.time() - start, 2)

            if result.get('needs_clarify'):
                response = f'🤔 **추가 정보가 필요합니다**\n\n{result["clarify_question"]}'
                st.markdown(response)
                st.session_state.waiting_clarify = True
                st.session_state.original_question = user_input
            else:
                route_info = f' (경로: {result.get("route", "all")})' if result.get('route') else ''
                response = f'{result["answer"]}\n\n---\n⏱ {elapsed}초{route_info}'
                st.markdown(response)

        st.session_state.messages.append({'role': 'assistant', 'content': response})
