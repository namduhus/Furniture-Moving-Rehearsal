"""Meta model calls behind one interface: Muse Spark chat, SAM 3.1 segmentation and Muse Image generation and editing.

Providers share the OpenAI-compatible request shapes: the Meta Model API (Muse Spark, SAM 3.1) and partner inference
for the open-weight Muse Spark Open. Keys come only from the environment or the git-ignored `.env` at the repository
root and are never printed or logged. Responses are cached on disk by request hash so repeated runs do not repeat paid
calls, and every call (or cache hit) is appended to a call log with model, latency and token usage.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = {"meta": {"base": "https://api.meta.ai/v1", "key": "META_API_KEY"},
             "baseten": {"base": "https://inference.baseten.co/v1", "key": "BASETEN_API_KEY"}}
CACHE = ROOT / "outputs/meta/cache"
LOG = ROOT / "outputs/meta/calls.jsonl"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def api_key(name, env_file=ROOT / ".env"):
    """The key from the environment, else from the `.env` file; the error names the variable, never a value."""
    if os.environ.get(name):
        return os.environ[name]
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == name and value.strip().strip("\"'"):
                return value.strip().strip("\"'")
    raise RuntimeError(f"Set {name} in the environment or in the git-ignored .env file")


def image_data_url(path):
    path = Path(path)
    if path.suffix.lower() not in MIME:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    return f"data:{MIME[path.suffix.lower()]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def http_transport(url, headers, body, stream, timeout=180):
    """POST JSON (or GET when body is None); returns the decoded JSON, or the list of SSE event objects when streaming."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not stream:
                return json.loads(response.read())
            events = []
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
            return events
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"{error.code} from {url}: {detail}") from None


# A record is `id<|box;...|>` plus an optional `<|mask;...|>`; the base85 mask payload may itself contain `|>` or
# commas, so a mask ends only where the next record or the line ends.
SAM_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|>(?:<\|mask;(.*?)\|>)?(?=,\d+<\|box;|$)")


def parse_sam(text):
    """Records of SAM's special-token output: one line per frame, `<Nf>` followed by comma-separated records."""
    records = []
    for line in text.splitlines():
        match = re.match(r"\s*<(\d+)f>(.*)$", line)
        if not match:
            continue
        for object_id, box_text, mask_text in SAM_RECORD.findall(match[2].strip()):
            box = dict(item.split("=", 1) for item in box_text.split(";"))
            record = {"frame": int(match[1]), "id": int(object_id),
                      "box": {k: float(v) if k == "c" else int(v) for k, v in box.items()}}
            if mask_text:
                fields, _, data = mask_text.partition("data=")
                meta = dict(item.split("=", 1) for item in fields.strip(";").split(";") if item)
                height, width, payload = data.split(",", 2)
                record["mask"] = {"height": int(height), "width": int(width), "encoding": payload[0], "payload": payload[1:],
                                  **({"c": float(meta["c"])} if "c" in meta else {})}
            records.append(record)
    return records


def mask_raster(record):
    """A record's box-local mask as an (H, W) array of 0/1, decoded with Meta's SAM parser (`meta-sam-parser`).

    The decoder allocates H*W before validating the payload, so the declared size is checked against the box first. Its
    payload includes the encoding marker (checked against the published 5 x 5 test vector).
    """
    import numpy as np
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    mask, box = record["mask"], record["box"]
    if not 0 < mask["height"] <= box["y2"] - box["y1"] + 1 or not 0 < mask["width"] <= box["x2"] - box["x1"] + 1:
        raise ValueError("SAM mask is larger than its box")
    raster = decode_mask_to_raster(SegmentationMask(encoding={"!": "one_bit", "~": "lossless"}[mask["encoding"]],
                                                    payload=mask["encoding"] + mask["payload"], width=mask["width"],
                                                    height=mask["height"]))
    return np.frombuffer(raster, dtype=np.uint8).reshape(mask["height"], mask["width"])


class MetaModels:
    """One entry point for every Meta model call; `transport` is injectable for tests."""

    def __init__(self, provider="meta", cache=CACHE, log=LOG, transport=http_transport):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider, self.cache, self.log, self.transport = provider, Path(cache), Path(log), transport

    def _call(self, path, body, stream=False):
        key = hashlib.sha256(json.dumps({"provider": self.provider, "path": path, "body": body}, sort_keys=True).encode()).hexdigest()
        cached = self.cache / f"{key}.json"
        tic = time.perf_counter()
        if cached.is_file():
            response, hit = json.loads(cached.read_text()), True
        else:
            settings = PROVIDERS[self.provider]
            headers = {"Authorization": f"Bearer {api_key(settings['key'])}", "Content-Type": "application/json"}
            response, hit = self.transport(settings["base"] + path, headers, body, stream), False
            self.cache.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(response))
        usage = response.get("usage") if isinstance(response, dict) else None
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as file:
            file.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": self.provider, "path": path,
                                   "model": (body or {}).get("model"), "cache_hit": hit, "request": key,
                                   "seconds": time.perf_counter() - tic, "usage": usage}) + "\n")
        return response, hit

    def models(self):
        response, _ = self._call("/models", None)
        return [m["id"] for m in response.get("data", [])]

    def chat(self, model, messages, schema=None, reasoning_effort=None, max_tokens=None):
        """Chat completion; with a JSON schema the reply is parsed into a dict."""
        body = {"model": model, "messages": messages}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": schema["title"], "schema": schema}}
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        response, hit = self._call("/chat/completions", body)
        content = response["choices"][0]["message"]["content"]
        return {"content": json.loads(content) if schema is not None else content, "usage": response.get("usage"),
                "cached": hit}

    def generate_image(self, prompt, size="1536x1024", output_format="png", model="muse-image-1.0"):
        """One Muse Image picture; image/web search and shell tools are disabled so the picture is generated, not
        retrieved. `size` sets the aspect ratio only."""
        body = {"model": model, "prompt": prompt, "n": 1, "size": size, "output_format": output_format,
                "response_format": "b64_json",
                "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}
        response, hit = self._call("/images/generations", body)
        return {"image": base64.b64decode(response["data"][0]["b64_json"]),
                "format": response.get("output_format", output_format), "cached": hit}

    def edit_image(self, image, prompt, output_format="png", model="muse-image-1.0"):
        """One Muse Image edit of a picture by an instruction (the API takes no mask); image/web search and shell tools are
        disabled."""
        body = {"model": model, "prompt": prompt, "n": 1, "images": [{"image_url": image_data_url(image)}],
                "output_format": output_format, "response_format": "b64_json",
                "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}
        response, hit = self._call("/images/edits", body)
        return {"image": base64.b64decode(response["data"][0]["b64_json"]),
                "format": response.get("output_format", output_format), "cached": hit}

    def segment(self, image, phrase, model="sam-3.1", encoding="one_bit", confidence=True):
        """SAM 3.1 boxes and masks for one short noun phrase in one image (streamed special-token text)."""
        body = {"model": model, "stream": True,
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": phrase}, {"type": "input_image", "image_url": image_data_url(image)}]}],
                "metadata": {"mask_encoding": encoding, "include_confidence": "true" if confidence else "false"}}
        events, hit = self._call("/responses", body, stream=True)
        text = "".join(e.get("delta", "") for e in events if e.get("type") == "response.output_text.delta")
        return {"records": parse_sam(text), "text": text, "cached": hit}


SMOKE_SCHEMA = {"title": "VisibleObjects", "type": "object", "required": ["objects"], "additionalProperties": False,
                "properties": {"objects": {"type": "array", "items": {
                    "type": "object", "required": ["name", "box_0_1000"], "additionalProperties": False,
                    "properties": {"name": {"type": "string"},
                                   "box_0_1000": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}}}}


def smoke(image, chat_model, phrase, models):
    """Three calls: list models, one Muse Spark image description as JSON, one SAM 3.1 segmentation."""
    listed = models.models()
    described = models.chat(chat_model, [{"role": "user", "content": [
        {"type": "text", "text": "List the distinct physical objects in this simulator image with a bounding box "
                                 "[x1, y1, x2, y2] on a 0-1000 grid."},
        {"type": "image_url", "image_url": {"url": image_data_url(image)}}]}], schema=SMOKE_SCHEMA)
    segmented = models.segment(image, phrase)
    masks = [{"id": r["id"], "height": r["mask"]["height"], "width": r["mask"]["width"],
              "area_fraction": float(mask_raster(r).mean())} for r in segmented["records"] if "mask" in r]
    return {"models": listed, "chat": described,
            "sam": {"phrase": phrase, "records": [{k: v for k, v in r.items() if k != "mask"} for r in segmented["records"]],
                    "masks": masks, "cached": segmented["cached"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    s = sub.add_parser("smoke", help="list models, one Muse Spark image call and one SAM 3.1 call")
    s.add_argument("--image", type=Path, required=True)
    s.add_argument("--chat-model", default="muse-spark-1.3")
    s.add_argument("--phrase", default="cabinet")
    s.add_argument("--provider", default="meta", choices=sorted(PROVIDERS))
    s.add_argument("--dry-run", action="store_true")
    g = sub.add_parser("generate", help="generate the images listed in a prompt file with Muse Image")
    g.add_argument("--config", type=Path, default=ROOT / "configs/home_images.json")
    g.add_argument("--out", type=Path, default=ROOT / "outputs/meta/images")
    g.add_argument("--dry-run", action="store_true")
    e = sub.add_parser("edit", help="edit generated images with Muse Image as listed under the prompt file's edits")
    e.add_argument("--config", type=Path, default=ROOT / "configs/home_images.json")
    e.add_argument("--out", type=Path, default=ROOT / "outputs/meta/images")
    e.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "edit":
        edits = json.loads(args.config.read_text())["edits"]
        if args.dry_run:
            print(json.dumps({"status": "local_plan_only", "edits": sorted(edits), "estimated_usd": 0.01*len(edits)}, indent=2))
            return 0
        models = MetaModels()
        for name, edit in edits.items():
            result = models.edit_image(args.out / f"{edit['source']}.png", edit["prompt"])
            path = args.out / f"{name}.{result['format']}"
            path.write_bytes(result["image"])
            print(json.dumps({"image": str(path), "source": edit["source"], "bytes": len(result["image"]), "cached": result["cached"]}))
        return 0
    if args.action == "generate":
        config = json.loads(args.config.read_text())
        if args.dry_run:
            print(json.dumps({"status": "local_plan_only", "images": sorted(config["images"]),
                              "estimated_usd": 0.01*len(config["images"])}, indent=2))
            return 0
        models = MetaModels()
        args.out.mkdir(parents=True, exist_ok=True)
        for name, prompt in config["images"].items():
            result = models.generate_image(f"{config['style']} {prompt}", config["size"])
            path = args.out / f"{name}.{result['format']}"
            path.write_bytes(result["image"])
            print(json.dumps({"image": str(path), "bytes": len(result["image"]), "cached": result["cached"]}))
        return 0
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "calls": ["GET /models", f"POST /chat/completions ({args.chat_model}, 1 image)",
                                                                 f"POST /responses (sam-3.1, '{args.phrase}')"],
                          "estimated_usd": "< 0.02 (Muse Spark $1.25/M input, $4.25/M output; SAM 3.1 $0.0025/image)",
                          "image": str(args.image)}, indent=2))
        return 0
    print(json.dumps(smoke(args.image, args.chat_model, args.phrase, MetaModels(args.provider)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
