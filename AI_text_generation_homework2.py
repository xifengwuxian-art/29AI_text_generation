import base64, requests
from io import BytesIO
from PIL import Image
from config import HF_API_KEY

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
INFER_URL = "https://router.huggingface.co/hf-inference/models"
CHAT_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

# ✅ Text (chat) fallbacks
TEXT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3:together",
    "Qwen/Qwen2.5-7B-Instruct:together",
    "Qwen/Qwen2.5-14B-Instruct:together",
]

# ✅ Vision caption (chat+vision) fallbacks
VISION_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct:together",
    "Qwen/Qwen3-VL-8B-Instruct:together",
]

# ⚠️ Image generation models on HF Router often require paid credits (402) or aren't served (404).
# Keep a short list; script will clearly report if none work.
IMAGE_MODELS = [
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

def _extract_err(r: requests.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict):
            e = j.get("error")
            if isinstance(e, dict):
                return e.get("message") or str(j)
            return str(e or j)
        return str(j)
    except Exception:
        return (r.text or "").strip() or r.reason or "Request failed."

def _data_url(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

def _chat(models, messages, max_tokens=180, temperature=0.7) -> str:
    last = None
    for model in models:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        try:
            r = requests.post(CHAT_URL, headers=CHAT_HEADERS, json=payload, timeout=120)
        except requests.RequestException as e:
            last = f"Request failed: {e}"
            continue
        if r.status_code != 200:
            last = f"{r.status_code}: {_extract_err(r)}"
            continue
        try:
            d = r.json()
        except Exception:
            last = "Non-JSON response received from the API."
            continue
        out = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if out:
            return out
        last = "Empty response from model."
    raise Exception(last or "All chat models failed.")

def generate_text(prompt: str) -> str:
    return _chat(TEXT_MODELS, [{"role": "user", "content": prompt}], max_tokens=180, temperature=0.7)

def generate_image(prompt: str) -> bytes:
    last = None
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "image/png", "Content-Type": "application/json"}
    for model in IMAGE_MODELS:
        url = f"{INFER_URL}/{model}"
        # try common payload shapes
        for payload in ({"inputs": prompt}, {"inputs": {"prompt": prompt}}, {"prompt": prompt}):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=180)
            except requests.RequestException as e:
                last = f"{model}: Request failed: {e}"
                continue
            ct = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and ct.startswith("image/"):
                return r.content
            last = f"{model}: {r.status_code}: {_extract_err(r)}"
    raise Exception(last or "Image generation failed (likely 402 credits required or 404 model not available).")

def caption_image(image_bytes: bytes) -> str:
    img_url = _data_url(image_bytes)
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Write a short caption for this image."},
            {"type": "image_url", "image_url": {"url": img_url}},
        ],
    }]
    return _chat(VISION_MODELS, msgs, max_tokens=90, temperature=0.2)

def main():
    user_prompt = input("Enter a short prompt or story idea: ")
    if not user_prompt.strip():
        print("No prompt entered. Exiting.")
        return

    print("\n=== Generating Text ===")
    text_prompt = generate_text(user_prompt)
    print("Generated Text Prompt:\n", text_prompt)

    print("\n=== Generating Image ===")
    image_data = generate_image(text_prompt)
    with open("generated_image.png", "wb") as f:
        f.write(image_data)
    Image.open(BytesIO(image_data)).show()

    print("\n=== Captioning the Generated Image ===")
    final_caption = caption_image(image_data)
    print("Final AI-Generated Caption:\n", final_caption)

if __name__ == "__main__":
    main()
