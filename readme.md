# Django RAG LLM Search System

## 프로젝트 소개
이 프로젝트는 Django 기반의 RAG(Retrieval-Augmented Generation) LLM 검색 시스템입니다. 사용자의 질문에 대해 인터넷 검색을 수행하고, 검색 결과를 기반으로 LLM이 답변을 생성하는 웹 애플리케이션입니다.

## 주요 기능
- 사용자 질문 입력 및 처리
- DuckDuckGo를 통한 실시간 웹 검색
- LLaMA 모델을 활용한 답변 생성
- 검색 결과 출처 표시 (URL 포함)
- Markdown 형식의 응답 렌더링

## 기술 스택
- Backend: Django
- LLM: Bllossom/llama-3.2-Korean-Bllossom-3B
- Frontend: HTML, JavaScript, Tailwind CSS
- Search: DuckDuckGo Search
- Additional Libraries: 
  - transformers
  - torch
  - beautifulsoup4
  - marked.js (Markdown 렌더링)

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd rag_project
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. settings.py 설정
```python
ALLOWED_HOSTS = ['your_ip_address', 'localhost', '127.0.0.1']
```

## 프로젝트 구조
```
rag_project/
│
├── rag_project/          
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── rag_app/             
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── rag_llm.py      # LLM 관련 코드
│   └── templates/
│       └── rag_app/
│           └── index.html
│
├── requirements.txt     # 의존성 파일
├── README.md           # 이 파일
└── manage.py
```

## 실행 방법
1. 데이터베이스 마이그레이션
```bash
python manage.py makemigrations
python manage.py migrate
```

2. 서버 실행
```bash
# 로컬에서만 접속
python manage.py runserver

# 외부 접속 허용
python manage.py runserver 0.0.0.0:8000
```

## 시스템 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- 안정적인 인터넷 연결

## 주의사항
- CUDA/GPU 지원이 필요할 수 있습니다
- 충분한 시스템 메모리가 필요합니다
- 안정적인 인터넷 연결이 필요합니다
- 첫 실행시 모델 다운로드에 시간이 소요될 수 있습니다

## 문제 해결
1. CUDA 관련 에러
   - NVIDIA 드라이버 설치 확인
   - CUDA Toolkit 설치 확인
   - PyTorch CUDA 버전 재설치

2. 메모리 부족 에러
   - 다른 프로그램 종료
   - 시스템 메모리 확인

3. 모델 로딩 실패
   - 인터넷 연결 확인
   - 모델 캐시 삭제 후 재시도

## 라이센스
이 프로젝트는 MIT 라이센스를 따릅니다.

## 기여 방법
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 문의사항
프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주시기 바랍니다.
