import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def get_advice(disease_name, confidence):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Gemini önerileri için GEMINI_API_KEY ortam değişkenini veya yerel .env dosyasını ayarlayın."

    prompt = f"""
Domates yaprağı hastalık tespit modeli şu sonucu verdi:

Hastalık: {disease_name}
Güven oranı: %{confidence:.2f}

Türkçe, sade ve anlaşılır cevap ver.
Şunları açıkla:
1. Bu hastalık nedir?
2. Bitkide ne yapılmalı?
3. Korunmak için neler yapılmalı?

İlaç markası ve kesin kimyasal doz önerme.
Maksimum 180 kelime kullan.
"""

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

    return response.text
