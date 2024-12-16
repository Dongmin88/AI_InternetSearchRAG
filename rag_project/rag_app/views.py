from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .rag_llm import RAGSearchLLM
from .models import Query
import traceback  # 추가
import logging  # 추가

logger = logging.getLogger(__name__)  # 추가

def index(request):
    return render(request, 'rag_app/index.html')

@csrf_exempt
def query_llm(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt')
            
            # RAG LLM 초기화
            model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
            rag_llm = RAGSearchLLM(model_id)
            
            # 검색 결과와 응답 가져오기
            search_results = rag_llm.search_internet(prompt)
            response, results = rag_llm.generate_response(prompt, search_results)
            
            # 데이터베이스에 저장
            query = Query.objects.create(
                prompt=prompt,
                response=response,
                sources=results
            )
            
            return JsonResponse({
                'response': response,
                'sources': results  # URL과 제목이 포함된 전체 검색 결과 반환
            })
        except Exception as e:
            return JsonResponse({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}, status=500)
    
    return JsonResponse({'error': '잘못된 요청 방식입니다.'}, status=400)