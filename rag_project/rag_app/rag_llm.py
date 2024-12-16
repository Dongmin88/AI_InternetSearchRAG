import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSearchLLM:
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.ddgs = DDGS()
        # CUDA 사용 가능 여부
        print("CUDA available:", torch.cuda.is_available())

        # CUDA 버전
        print("CUDA version:", torch.version.cuda)

        # 사용 가능한 GPU 개수
        print("GPU count:", torch.cuda.device_count())

        # 현재 GPU 이름
        if torch.cuda.is_available():
            print("Current GPU:", torch.cuda.get_device_name(0))
        
    def clean_text(self, text: str) -> str:
        """텍스트 정리를 위한 유틸리티 함수"""
        # 불필요한 공백 제거
        text = ' '.join(text.split())
        # 특수문자 처리 등 추가 가능
        return text
        
    def search_internet(self, query: str, max_results: int = 5) -> List[dict]:  # 반환 타입을 List[dict]로 변경
        """인터넷 검색을 수행하고 결과를 반환합니다."""
        logger.info(f"검색 쿼리: {query}")
        
        try:
            # DuckDuckGo 검색 수행
            results = []
            for r in self.ddgs.text(query, max_results=max_results):
                logger.info(f"검색 결과: {r}")
                results.append(r)
            
            logger.info(f"검색 결과 수: {len(results)}")
            contents = []
            
            for result in results:
                try:
                    # 검색 결과에서 URL과 제목 추출
                    url = result.get('href', '')
                    title = result.get('title', '')
                    
                    content_dict = {
                        'url': url,
                        'title': title,
                        'content': ''
                    }

                    if 'body' in result:
                        content_dict['content'] = f"내용: {result['body']}"
                    
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                        }
                        
                        if url:
                            logger.info(f"URL 접근 시도: {url}")
                            response = requests.get(
                                url,
                                headers=headers,
                                timeout=10,
                                verify=False
                            )
                            
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, 'html.parser')
                                
                                text_parts = []
                                for element in soup.find_all(['article', 'main', 'div', 'p']):
                                    text = element.get_text(strip=True)
                                    if len(text) > 100:
                                        text_parts.append(text)
                                
                                if text_parts:
                                    combined_text = ' '.join(text_parts)
                                    content_dict['content'] = combined_text[:1500]
                                    logger.info(f"웹페이지에서 컨텐츠 추출 성공 (길이: {len(content_dict['content'])})")
                    except:
                        logger.info("웹페이지 접근 실패, 검색 결과 텍스트만 사용")
                    
                    if content_dict['content']:
                        contents.append(content_dict)
                    
                except Exception as e:
                    logger.error(f"결과 처리 중 오류 발생: {str(e)}")
                    continue
            
            logger.info(f"최종 추출된 콘텐츠 수: {len(contents)}")
            return contents if contents else [{"url": "", "title": "검색 실패", "content": "검색 결과를 찾을 수 없습니다."}]
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return [{"url": "", "title": "오류", "content": "검색 결과를 가져오는 중 오류가 발생했습니다."}]


    def generate_response(self, instruction: str, search_results: List[dict]) -> str:
        """검색 결과를 포함하여 응답을 생성합니다."""
        # 컨텍스트 생성 시 URL과 제목 포함
        contexts = []
        for result in search_results:
            context = f"제목: {result['title']}\n"
            context += f"URL: {result['url']}\n"
            context += f"내용: {result['content']}\n"
            contexts.append(context)
        
        context = "\n---\n".join(contexts)
        prompt = f"""다음은 검색된 관련 정보입니다:
    ---
    {context}
    ---

    질문: {instruction}
    답변:"""
        
        logger.info("생성된 프롬프트:")
        logger.info(prompt)
        
        messages = [{"role": "user", "content": prompt}]

    # 입력 생성
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # attention_mask 생성 및 device로 이동
        attention_mask = torch.ones_like(inputs)
        inputs = inputs.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        
        terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response, search_results  # 검색 결과도 함께 반환

    def query(self, instruction: str) -> str:
        """검색과 응답 생성을 통합하여 실행합니다."""
        search_results = self.search_internet(instruction)
        return self.generate_response(instruction, search_results)

# 사용 예시
if __name__ == "__main__":
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    rag_llm = RAGSearchLLM(model_id)
    
    # 다양한 주제에 대한 테스트
    queries = [
        "세종대왕 시대 떄 나온 책 다 알려줘."
    ]
    
    for query in queries:
        print(f"\n질문: {query}")
        response = rag_llm.query(query)
        print("답변:", response)