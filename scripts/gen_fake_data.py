"""生成伪数据 (JSONL) 用于本地测试 pipeline.
- 二分类情感风格: 正样本包含正向词, 负样本包含负向词
"""
from __future__ import annotations
import json, random, argparse, os
from pathlib import Path

POS_WORDS = ["good", "excellent", "amazing", "nice", "wonderful"]
NEG_WORDS = ["bad", "terrible", "awful", "poor", "worse"]
NEUTRAL = ["movie", "product", "service", "experience", "design", "quality", "time"]

def sample_sentence(label: int, length: int = 6):
    words = []
    pool = POS_WORDS if label == 1 else NEG_WORDS
    words.append(random.choice(pool))
    for _ in range(length - 1):
        if random.random() < 0.3:
            words.append(random.choice(pool))
        else:
            words.append(random.choice(NEUTRAL))
    random.shuffle(words)
    return " ".join(words)

def gen(n: int, pos_ratio: float = 0.5):
    data = []
    for i in range(n):
        label = 1 if random.random() < pos_ratio else 0
        text = sample_sentence(label)
        data.append({"id": i, "text": text, "gold": label})
    return data

def write_jsonl(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--train', type=int, default=50)
    ap.add_argument('--valid', type=int, default=20)
    ap.add_argument('--test', type=int, default=20)
    args = ap.parse_args()

    random.seed(42)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / 'train.jsonl', gen(args.train))
    write_jsonl(out_dir / 'valid.jsonl', gen(args.valid))
    write_jsonl(out_dir / 'test.jsonl', gen(args.test))
    print(f"Fake data generated under {out_dir}")
