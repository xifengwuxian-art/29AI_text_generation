from config import HF_API_KEY
import requests, io, base64
from PIL import Image
from colorama import Fore, Style, init

init(autoreset=True)

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

# Vision caption models (together)
VISION_MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct:together",
    "Qwen/Qwen3-VL-32B-Instruct:together",
    "Qwen/Qwen2.5-VL-32B-Instruct:together",
    "Qwen/Qwen2-VL-7B-Instruct:together",
]

# Text expansion models (together)
TEXT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct:together",
    "Qwen/Qwen2.5-14B-Instruct:together",
    "mistralai/Mistral-7B-Instruct-v0.3:together",
]

def data_url_from_pil(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_err(r: requests.Response) -> str:
    try:
        j = r.json()
        return j.get("error", {}).get("message") or str(j)
    except Exception:
        return (r.text or "").strip() or r.reason or "Request failed."

def chat_with_models(models, messages, max_tokens=120, temperature=0.3):
    last = None
    for model in models:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        try:
            r = requests.post(ROUTER_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.RequestException as e:
            last = f"Request failed: {e}"
            continue
        if r.status_code != 200:
            last = f"Status {r.status_code}: {extract_err(r)}"
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
    raise Exception(last or "All models failed.")

def words(text: str):
    return (text or "").strip().split()

def get_caption(image: Image.Image) -> str:
    img_url = data_url_from_pil(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Write one clear, complete sentence caption for this image."},
            {"type": "image_url", "image_url": {"url": img_url}},
        ],
    }]
    return chat_with_models(VISION_MODELS, messages, max_tokens=80, temperature=0.2)

def expand_caption(caption: str) -> str:
    prompt = (
        "Expand the following into EXACTLY 30 words. Single paragraph. One complete sentence. "
        "End with a period. No title.\n\n"
        f"Caption: {caption}"
    )
    txt = chat_with_models(TEXT_MODELS, [{"role": "user", "content": prompt}], max_tokens=120, temperature=0.4)
    w = words(txt)
    out = " ".join(w[:30])
    if out and out[-1] not in ".!?":
        out += "."
    return out

def main():
    path = input("Enter image path: ").strip()
    try:
        image = Image.open(path)

        print(Fore.YELLOW + "Generating caption...")
        caption = get_caption(image)
        print(Fore.GREEN + "Caption: " + Style.BRIGHT + caption)

        more = input("Expand to 30 words? (y/n): ").strip().lower()
        if more == "y":
            print(Fore.YELLOW + "Expanding...")
            description = expand_caption(caption)
            print(Fore.GREEN + "Expanded: " + Style.BRIGHT + description)

    except Exception as e:
        print(Fore.RED + f"Error: {e}")

if __name__ == "__main__":
    main()
