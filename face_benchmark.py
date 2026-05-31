import argparse
import csv
import importlib.util
import json
import os
import platform
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL_PATH = os.getenv("FACE_BENCHMARK_MODEL_PATH", "./models/Qwen2.5-VL-7B-Instruct")
DEFAULT_IMAGE_ROOT = "./test_images/face"
DEFAULT_OUTPUT_DIR = "./outputs/face_benchmark"
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dirname: str
    display_name: str
    instruction: str


TASKS: Dict[str, TaskSpec] = {
    "face_style": TaskSpec(
        key="face_style",
        dirname="face_style",
        display_name="脸型",
        instruction="请只判断图片中人物的脸型轮廓，不要判断发型、妆容或表情。",
    ),
    "eye": TaskSpec(
        key="eye",
        dirname="eye",
        display_name="眼型",
        instruction="请只判断图片中人物的眼型，不要输出眼妆、眼距或其他面部信息。",
    ),
    "eyebrow_style": TaskSpec(
        key="eyebrow_style",
        dirname="eyebrow_style",
        display_name="眉型",
        instruction="请只判断图片中人物的眉型，不要输出脸型、眼型或妆容信息。",
    ),
}


FACE_FILENAME_TO_LABEL = {
    "oval": "椭圆",
    "diamond-shaped": "菱形",
    "diamond": "菱形",
    "inverted_triangle": "倒三角",
    "round": "圆脸",
    "square": "方脸",
    "triangle": "正三角",
}

TASK_LABEL_ORDER = {
    "face_style": ["椭圆", "圆脸", "方脸", "倒三角", "菱形", "正三角"],
    "eye": ["上斜眼", "下垂眼", "圆眼", "细长眼"],
    "eyebrow_style": ["一字眉", "上扬眉", "小欧眉", "平直眉", "标准眉", "标平眉", "落尾眉"],
}

LABEL_DEFINITIONS = {
    "face_style": {
        "椭圆": "脸长略大于脸宽，额头和下颌线条圆润流畅，下巴不过尖。",
        "圆脸": "脸长和脸宽接近，脸颊饱满，整体轮廓圆润。",
        "方脸": "下颌角明显，脸部轮廓偏方，额头、颧骨、下颌宽度接近。",
        "倒三角": "额头或颧骨较宽，下巴明显变窄或偏尖，包含心形脸/瓜子脸。",
        "菱形": "颧骨最宽，额头和下颌较窄，下巴偏尖。",
        "正三角": "下颌区域较宽，额头较窄，包含梨形脸。",
    },
    "eye": {
        "上斜眼": "眼尾明显高于眼头，眼裂走势向上。",
        "下垂眼": "眼尾低于眼头或整体眼尾下垂。",
        "圆眼": "眼睛纵向高度较大，整体更圆，眼白暴露较明显。",
        "细长眼": "眼裂横向长、纵向高度小，整体狭长。",
    },
    "eyebrow_style": {
        "一字眉": "眉头到眉尾整体接近平直，眉峰不明显。",
        "上扬眉": "眉尾明显上扬，整体走势向上。",
        "小欧眉": "眉峰较高，有欧式弧度，但整体不夸张。",
        "平直眉": "整体较平直，眉峰轻微，比一字眉更自然。",
        "标准眉": "眉头、眉峰、眉尾比例均衡，有自然弧度。",
        "标平眉": "介于标准眉和平直眉之间，弧度较低。",
        "落尾眉": "眉尾下落或走势向下，尾部低于眉峰。",
    },
}


ALIASES = {
    "face_style": {
        "椭圆脸": "椭圆",
        "椭圆": "椭圆",
        "鹅蛋脸": "椭圆",
        "鹅蛋": "椭圆",
        "长圆脸": "椭圆",
        "方脸": "方脸",
        "方形脸": "方脸",
        "方形": "方脸",
        "长方脸": "方脸",
        "五角脸": "方脸",
        "圆脸": "圆脸",
        "倒三角脸": "倒三角",
        "倒三角": "倒三角",
        "心形脸": "倒三角",
        "心形": "倒三角",
        "瓜子脸": "倒三角",
        "瓜子": "倒三角",
        "菱形脸": "菱形",
        "菱形": "菱形",
        "正三角脸": "正三角",
        "正三角": "正三角",
        "梨形脸": "正三角",
        "梨形": "正三角",
    },
    "eye": {
        "上斜眼": "上斜眼",
        "吊眼": "上斜眼",
        "丹凤眼": "上斜眼",
        "瑞凤眼": "上斜眼",
        "狐狸眼": "上斜眼",
        "下垂眼": "下垂眼",
        "圆眼": "圆眼",
        "杏眼": "圆眼",
        "细长眼": "细长眼",
        "细长": "细长眼",
    },
    "eyebrow_style": {
        "一字眉": "一字眉",
        "上扬眉": "上扬眉",
        "小欧眉": "小欧眉",
        "欧式眉": "小欧眉",
        "平直眉": "平直眉",
        "标准眉": "标准眉",
        "标平眉": "标平眉",
        "落尾眉": "落尾眉",
        "落笔眉": "落尾眉",
    },
}


def parse_gt_label(task_key: str, image_path: Path) -> str:
    stem = image_path.stem.strip()
    if task_key == "face_style":
        key = stem.lower()
        if "_face" in key:
            key = key.split("_face", 1)[0]
        else:
            key = re.sub(r"[\s_-]*\d+$", "", key)
        return FACE_FILENAME_TO_LABEL.get(key, key)
    return re.sub(r"[\s_-]*\d+$", "", stem).strip()


def collect_samples(image_root: Path, task_keys: Sequence[str], limit_per_task: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    for task_key in task_keys:
        spec = TASKS[task_key]
        task_dir = image_root / spec.dirname
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task image directory not found: {task_dir}")
        paths = sorted(p for p in task_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if limit_per_task > 0 and len(paths) > limit_per_task:
            paths = sorted(rng.sample(paths, limit_per_task))
        for path in paths:
            samples.append(
                {
                    "task": task_key,
                    "task_name": spec.display_name,
                    "image_path": path,
                    "image_name": path.name,
                    "ground_truth": parse_gt_label(task_key, path),
                    "ground_truth_raw": path.stem,
                }
            )
    return samples


def labels_for_task(samples: Sequence[Dict[str, Any]], task_key: str) -> List[str]:
    labels = {str(s["ground_truth"]) for s in samples if s["task"] == task_key}
    ordered = [label for label in TASK_LABEL_ORDER.get(task_key, []) if label in labels]
    extras = sorted(labels.difference(ordered))
    return ordered + extras


def build_prompt(spec: TaskSpec, labels: Sequence[str]) -> str:
    label_text = "、".join(labels)
    definition_lines = []
    for label in labels:
        definition = LABEL_DEFINITIONS.get(spec.key, {}).get(label)
        if definition:
            definition_lines.append(f"- {label}：{definition}")
    definitions = "\n".join(definition_lines)
    return (
        f"你是一个严格的{spec.display_name}分类评测模型。{spec.instruction}\n"
        f"候选标签只有这些：{label_text}\n"
        f"判别标准：\n{definitions}\n"
        "如果图片存在参考文字、文件名、目录名或水印，请忽略它们，只看人物面部区域。\n"
        "请必须且只能从候选标签中选择一个最接近的标签。\n"
        "只输出合法 JSON，不要输出 Markdown，不要解释，不要输出候选标签之外的内容。\n"
        "输出格式固定为：{\"label\":\"候选标签之一\"}"
    )


def extract_prediction(raw_text: str) -> Tuple[str, bool]:
    text = (raw_text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", text, flags=re.I)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and data.get("label") is not None:
            return str(data.get("label")).strip(), True
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and data.get("label") is not None:
                return str(data.get("label")).strip(), True
        except Exception:
            pass

    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    for sep in ("：", ":", "、", "，", "。", ",", ";", "；"):
        if sep in first_line:
            first_line = first_line.split(sep)[-1].strip()
    return first_line, False


def normalize_prediction(task_key: str, prediction: str, allowed_labels: Sequence[str]) -> str:
    text = (prediction or "").strip()
    if not text:
        return "其他"

    allowed = set(allowed_labels)
    if text in allowed:
        return text

    # Prefer explicit allowed labels that appear in the answer.
    for label in sorted(allowed, key=len, reverse=True):
        if label and label in text:
            return label

    for alias, standard in ALIASES.get(task_key, {}).items():
        if alias in text and standard in allowed:
            return standard

    return "其他"


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def latency_stats(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Optional[float]]:
    values = [float(r[field]) for r in rows if r.get("ok") and r.get(field) is not None]
    if not values:
        return {"mean": None, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "p50": round(percentile(values, 0.50), 4),
        "p90": round(percentile(values, 0.90), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize_subset(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(subset)
        ok = sum(1 for r in subset if r.get("ok"))
        correct = sum(1 for r in subset if r.get("correct"))
        json_ok = sum(1 for r in subset if r.get("json_ok"))
        token_speeds = [float(r["tokens_per_sec"]) for r in subset if r.get("tokens_per_sec") is not None]
        return {
            "total_runs": total,
            "ok_runs": ok,
            "correct_runs": correct,
            "accuracy_all_runs": round(correct / total, 4) if total else None,
            "accuracy_ok_runs": round(correct / ok, 4) if ok else None,
            "json_success_rate": round(json_ok / total, 4) if total else None,
            "end_to_end_latency_sec": latency_stats(subset, "end_to_end_sec"),
            "generate_latency_sec": latency_stats(subset, "generate_sec"),
            "tokens_per_sec_mean": round(statistics.mean(token_speeds), 4) if token_speeds else None,
        }

    def confusion_matrix(subset: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {}
        for row in subset:
            gt = str(row.get("ground_truth") or "")
            pred = str(row.get("prediction") or "ERROR")
            matrix.setdefault(gt, {})
            matrix[gt][pred] = matrix[gt].get(pred, 0) + 1
        return matrix

    by_task = {}
    for task_key in TASKS:
        subset = [r for r in rows if r.get("task") == task_key]
        if subset:
            task_summary = summarize_subset(subset)
            task_summary["confusion_matrix"] = confusion_matrix(subset)
            by_task[task_key] = task_summary

    return {
        "overall": summarize_subset(rows),
        "by_task": by_task,
    }


class Qwen25VLRunner:
    def __init__(
        self,
        model_path: str,
        dtype: str,
        device_map: str,
        max_new_tokens: int,
        attn_implementation: str,
        min_pixels: int,
        max_pixels: int,
        allow_tf32: bool,
    ):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.allow_tf32 = allow_tf32
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This benchmark is configured for an NVIDIA GPU; "
                "verify the PyTorch CUDA installation and CUDA_VISIBLE_DEVICES."
            )
        if attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "flash_attention_2 was requested but package 'flash_attn' is not installed. "
                "Install flash-attn or use --attn-implementation sdpa."
            )
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch_dtype = self._resolve_dtype(dtype)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
        self.model.eval()

    def _resolve_dtype(self, dtype: str):
        if dtype == "auto":
            return "auto"
        if dtype == "float16":
            return self.torch.float16
        if dtype == "float32":
            return self.torch.float32
        return self.torch.bfloat16

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def environment_info(self) -> Dict[str, Any]:
        device = self.torch.cuda.current_device()
        props = self.torch.cuda.get_device_properties(device)
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": self.torch.__version__,
            "cuda_available": self.torch.cuda.is_available(),
            "torch_cuda_version": self.torch.version.cuda,
            "cudnn_version": self.torch.backends.cudnn.version(),
            "gpu_index": device,
            "gpu_name": props.name,
            "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
            "gpu_capability": f"{props.major}.{props.minor}",
            "dtype": str(next(self.model.parameters()).dtype),
            "attn_implementation": self.attn_implementation,
            "allow_tf32": self.allow_tf32,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

    def predict(self, image_path: Path, prompt: str) -> Tuple[str, Dict[str, Any]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

        # CUDA kernels are asynchronous; synchronize the measured boundaries.
        self.synchronize()
        total_start = time.perf_counter()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        self.synchronize()
        generate_start = time.perf_counter()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        self.synchronize()
        generate_sec = time.perf_counter() - generate_start

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.synchronize()
        end_to_end_sec = time.perf_counter() - total_start

        generated_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else None
        max_gpu_mem_gb = None
        if self.torch.cuda.is_available():
            max_gpu_mem_gb = self.torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

        return output_text, {
            "end_to_end_sec": end_to_end_sec,
            "generate_sec": generate_sec,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": generated_tokens / generate_sec if generated_tokens and generate_sec > 0 else None,
            "max_gpu_mem_gb": max_gpu_mem_gb,
        }


def run_one(
    runner: Qwen25VLRunner,
    sample: Dict[str, Any],
    labels_by_task: Dict[str, List[str]],
    prompts_by_task: Dict[str, str],
    round_idx: int,
) -> Dict[str, Any]:
    task_key = sample["task"]
    row: Dict[str, Any] = {
        "round": round_idx,
        "task": task_key,
        "task_name": sample["task_name"],
        "image_name": sample["image_name"],
        "image_path": str(sample["image_path"]),
        "ground_truth": sample["ground_truth"],
        "ground_truth_raw": sample["ground_truth_raw"],
        "prediction": None,
        "prediction_raw": None,
        "correct": False,
        "ok": False,
        "json_ok": False,
        "end_to_end_sec": None,
        "generate_sec": None,
        "generated_tokens": None,
        "tokens_per_sec": None,
        "max_gpu_mem_gb": None,
        "raw_output": "",
        "error": "",
    }

    try:
        raw_output, metrics = runner.predict(Path(sample["image_path"]), prompts_by_task[task_key])
        pred_raw, json_ok = extract_prediction(raw_output)
        prediction = normalize_prediction(task_key, pred_raw, labels_by_task[task_key])
        row.update(
            {
                "prediction": prediction,
                "prediction_raw": pred_raw,
                "correct": prediction == sample["ground_truth"],
                "ok": True,
                "json_ok": json_ok,
                "end_to_end_sec": round(metrics["end_to_end_sec"], 4),
                "generate_sec": round(metrics["generate_sec"], 4),
                "generated_tokens": metrics.get("generated_tokens"),
                "tokens_per_sec": round(metrics["tokens_per_sec"], 4) if metrics.get("tokens_per_sec") else None,
                "max_gpu_mem_gb": round(metrics["max_gpu_mem_gb"], 4) if metrics.get("max_gpu_mem_gb") is not None else None,
                "raw_output": raw_output,
            }
        )
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "round",
        "task",
        "image_name",
        "ground_truth",
        "prediction",
        "prediction_raw",
        "correct",
        "ok",
        "json_ok",
        "end_to_end_sec",
        "generate_sec",
        "generated_tokens",
        "tokens_per_sec",
        "max_gpu_mem_gb",
        "error",
        "raw_output",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_task_keys(task_arg: str) -> List[str]:
    if task_arg == "all":
        return list(TASKS.keys())
    task_keys = [x.strip() for x in task_arg.split(",") if x.strip()]
    bad = [x for x in task_keys if x not in TASKS]
    if bad:
        raise ValueError(f"Unknown tasks: {bad}. Valid tasks: all,{','.join(TASKS)}")
    return task_keys


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path
    legacy_path = Path(str(path) + "~")
    if legacy_path.exists():
        print(f"Model path not found; using legacy path: {legacy_path}")
        return legacy_path
    raise FileNotFoundError(
        f"Model path not found: {path}. Pass the correct directory with --model-path "
        "or set FACE_BENCHMARK_MODEL_PATH."
    )


def print_file_plan() -> None:
    print(
        "\nFile structure used by this benchmark:\n"
        "  face_benchmark.py                         # current runner\n"
        "  test_images/face/face_style/*             # face shape samples, filename => GT\n"
        "  test_images/face/eye/*                    # eye shape samples, filename => GT\n"
        "  test_images/face/eyebrow_style/*          # eyebrow shape samples, filename => GT\n"
        "  outputs/face_benchmark/face_benchmark_*.csv\n"
        "  outputs/face_benchmark/face_benchmark_*.json\n"
    )


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    task_keys = parse_task_keys(args.tasks)
    model_path = resolve_model_path(args.model_path)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(image_root, task_keys, args.limit_per_task, args.seed)
    if not samples:
        raise FileNotFoundError(f"No face benchmark images found under {image_root}")

    labels_by_task = {task_key: labels_for_task(samples, task_key) for task_key in task_keys}
    prompts_by_task = {
        task_key: build_prompt(TASKS[task_key], labels_by_task[task_key])
        for task_key in task_keys
    }

    print_file_plan()
    print(f"Model path: {model_path}")
    print(f"Image root: {image_root}")
    print(f"Tasks: {', '.join(task_keys)}")
    for task_key in task_keys:
        count = sum(1 for s in samples if s["task"] == task_key)
        print(f"  {task_key}: {count} images, labels={labels_by_task[task_key]}")
    print(f"Repeat: {args.repeat}, warmup: {args.warmup}, max_new_tokens: {args.max_new_tokens}")
    print(
        f"dtype: {args.dtype}, attention: {args.attn_implementation}, "
        f"pixels: {args.min_pixels}..{args.max_pixels}, TF32: {not args.disable_tf32}"
    )

    load_start = time.perf_counter()
    runner = Qwen25VLRunner(
        model_path=str(model_path),
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        allow_tf32=not args.disable_tf32,
    )
    load_sec = time.perf_counter() - load_start
    environment = runner.environment_info()
    print(f"Model loaded in {load_sec:.2f}s")
    print(
        f"Runtime: torch={environment['torch']} cuda={environment['torch_cuda_version']} "
        f"gpu={environment['gpu_name']} ({environment['gpu_memory_gb']} GB)"
    )

    warmup_samples = samples[: max(0, args.warmup)]
    for idx, sample in enumerate(warmup_samples, 1):
        print(f"[warmup {idx}/{len(warmup_samples)}] {sample['task']} {sample['image_name']}")
        _ = run_one(runner, sample, labels_by_task, prompts_by_task, round_idx=0)

    rows: List[Dict[str, Any]] = []
    for round_idx in range(1, args.repeat + 1):
        for idx, sample in enumerate(samples, 1):
            print(f"[{round_idx}/{args.repeat} {idx}/{len(samples)}] {sample['task']} {sample['image_name']}")
            row = run_one(runner, sample, labels_by_task, prompts_by_task, round_idx)
            rows.append(row)
            if row["ok"]:
                print(
                    "  gt={gt} pred={pred} correct={correct} e2e={e2e}s gen={gen}s".format(
                        gt=row["ground_truth"],
                        pred=row["prediction"],
                        correct=row["correct"],
                        e2e=row["end_to_end_sec"],
                        gen=row["generate_sec"],
                    )
                )
            else:
                print(f"  ERROR: {row['error']}")

    summary = summarize_rows(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"face_benchmark_{timestamp}.csv"
    json_path = output_dir / f"face_benchmark_{timestamp}.json"

    payload = {
        "config": {
            "model_path": str(model_path),
            "image_root": str(image_root),
            "tasks": task_keys,
            "labels_by_task": labels_by_task,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "limit_per_task": args.limit_per_task,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "allow_tf32": not args.disable_tf32,
            "load_sec": round(load_sec, 4),
        },
        "environment": environment,
        "prompts_by_task": prompts_by_task,
        "summary": summary,
        "rows": rows,
    }

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-VL face classification accuracy and latency.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen2.5-VL model path.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, help="Root directory containing face_style/eye/eyebrow_style.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/JSON benchmark outputs.")
    parser.add_argument("--tasks", default="all", help="all or comma-separated tasks: face_style,eye,eyebrow_style")
    parser.add_argument("--limit-per-task", type=int, default=0, help="Randomly sample at most N images per task. 0 means all.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured repeats per image.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup images before measured runs.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Generation token cap. Keep low for classification latency.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention backend. sdpa works with PyTorch 2.5; flash_attention_2 needs flash-attn.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help="Minimum pixels passed to the Qwen2.5-VL processor.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum pixels passed to the Qwen2.5-VL processor; keep fixed when comparing runs.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 acceleration on Ampere GPUs such as A100.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_pixels <= 0 or args.max_pixels <= 0 or args.min_pixels > args.max_pixels:
        parser.error("--min-pixels and --max-pixels must be positive and min <= max.")
    return args


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
