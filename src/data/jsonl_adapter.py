"""Simple JSONL dataset adapter (占位实现)."""
from __future__ import annotations
from typing import List, Iterable
import json
from pathlib import Path

from ..interfaces import DataSample

class JSONLAdapter:
    def __init__(self, data_dir: str, train_file: str, valid_file: str, test_file: str):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / train_file
        self.valid_path = self.data_dir / valid_file
        self.test_path = self.data_dir / test_file
        self._cache = {}

    def _load_file(self, path: Path) -> List[DataSample]:
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            # 占位：返回空数据
            data: List[DataSample] = []
        else:
            with path.open('r', encoding='utf-8') as f:
                data = [DataSample(json.loads(line)) for line in f if line.strip()]
        self._cache[path] = data
        return data

    def load(self) -> Iterable[DataSample]:
        return self.train_split()

    def train_split(self) -> List[DataSample]:
        return self._load_file(self.train_path)

    def valid_split(self) -> List[DataSample]:
        return self._load_file(self.valid_path)

    def test_split(self) -> List[DataSample]:
        return self._load_file(self.test_path)
