import os
import io
import json
import uuid
import time
import requests
import pandas as pd
import pdfplumber
import docx
from pdf2image import convert_from_path

# ==========================================
# 🔑 네이버 CLOVA OCR API 키 설정 (여기를 수정하세요!)
# ==========================================
API_URL = os.environ.get("CLOVA_API_URL")
SECRET_KEY = os.environ.get("CLOVA_SECRET_KEY")
# ==========================================


def call_clova_ocr(image_bytes):
    """CLOVA OCR API에 이미지를 전송하고 텍스트를 반환받는 함수"""
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json)}
    files = [('file', ('image.jpg', image_bytes, 'image/jpeg'))]
    headers = {'X-OCR-SECRET': SECRET_KEY}

    try:
        response = requests.post(
            API_URL, headers=headers, data=payload, files=files)
        res_json = response.json()

        # API 응답에서 인식된 텍스트(inferText)만 순서대로 추출하여 이어붙임
        extracted_text = ""
        if 'images' in res_json and res_json['images']:
            for field in res_json['images'][0].get('fields', []):
                extracted_text += field['inferText'] + " "
        return extracted_text.strip()

    except Exception as e:
        print(f"CLOVA OCR API 호출 에러: {e}")
        return ""


def extract_text_from_pdf(file_path):
    """PDF 텍스트 추출 (일반 텍스트 + CLOVA OCR 하이브리드)"""
    text_data = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()

                # 1. 정상적인 텍스트 문서인 경우 (15자 이상)
                if text and len(text.strip()) > 15:
                    clean_text = " ".join(text.split())
                    text_data.append(
                        {"page": page_num + 1, "content": clean_text})

                # 2. 이미지 문서 또는 노이즈만 있는 경우 -> CLOVA OCR 가동
                else:
                    print(
                        f"이미지 감지됨. CLOVA OCR 진행 중: {os.path.basename(file_path)} (Page {page_num + 1})")
                    # 해당 페이지만 이미지로 변환
                    images = convert_from_path(
                        file_path, first_page=page_num+1, last_page=page_num+1, dpi=300)

                    if images:
                        # 하드디스크에 사진을 저장하지 않고 메모리(Bytes)에서 바로 API로 전송 (속도 최적화)
                        img_byte_arr = io.BytesIO()
                        images[0].save(img_byte_arr, format='JPEG')
                        img_bytes = img_byte_arr.getvalue()

                        ocr_text = call_clova_ocr(img_bytes)
                        clean_ocr_text = " ".join(ocr_text.split())
                        text_data.append(
                            {"page": page_num + 1, "content": clean_ocr_text})

    except Exception as e:
        print(f"PDF 읽기 에러 ({file_path}): {e}")

    return text_data


def extract_text_from_docx(file_path):
    """DOCX 텍스트 추출"""
    text_data = []
    try:
        doc = docx.Document(file_path)
        full_text = [para.text.strip()
                     for para in doc.paragraphs if para.text.strip()]
        clean_text = " ".join(full_text)
        text_data.append({"page": 1, "content": clean_text})
    except Exception as e:
        print(f"DOCX 읽기 에러 ({file_path}): {e}")
    return text_data


# --- 메인 실행 로직 ---
test_folder = "./ncp_data_test"
all_documents = []

if not os.path.exists(test_folder):
    print(f"'{test_folder}' 폴더가 없습니다.")
else:
    print("문서 추출 파이프라인 시작...")
    for file_name in os.listdir(test_folder):
        file_path = os.path.join(test_folder, file_name)
        extracted_pages = []

        if file_name.lower().endswith(".pdf"):
            extracted_pages = extract_text_from_pdf(file_path)
        elif file_name.lower().endswith(".docx"):
            extracted_pages = extract_text_from_docx(file_path)
        else:
            continue

        for page_data in extracted_pages:
            all_documents.append({
                "file_name": file_name,
                "page_number": page_data["page"],
                "text_chunk": page_data["content"]
            })

    if all_documents:
        df = pd.DataFrame(all_documents)
        df.to_csv("extracted_dataset1.csv", index=False, encoding="utf-8-sig")
        print("\n✅ 데이터 추출 완료! 'extracted_dataset1.csv'를 확인하세요.")
    else:
        print("\n⚠️ 추출할 텍스트가 없습니다.")
