from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class KnowledgeCase:
    workflow_id: str
    task_id: str
    scheduler_type: str
    chosen_node: str
    task_execution_time: float
    workflow_makespan: float
    task_features: List[float] | None = None
    workflow_embedding: List[float] | None = None
    quality_score: float | None = None  # optional quality metric (e.g., inverse makespan)

class JSONKnowledgeBase:
    def __init__(self):
        self.cases: List[KnowledgeCase] = []
        self.case_index: Dict[str, List[int]] = {}
        self.embedding_dim: int | None = None

    def add_case(self, case: KnowledgeCase):
        idx = len(self.cases)
        self.cases.append(case)
        key = f"{case.workflow_id}:{case.task_id}"
        self.case_index.setdefault(key, []).append(idx)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "total_cases": len(self.cases),
            "embedding_dim": self.embedding_dim,
            "cases": [asdict(c) for c in self.cases]
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "JSONKnowledgeBase":
        kb = cls()
        kb.embedding_dim = data.get("embedding_dim")
        
        # ==================================================================
        # [FIX] 核心修正：健壮地处理数据不匹配问题
        # 1. 获取KnowledgeCase所有合法的字段名
        # 2. 在创建对象前，过滤掉JSON中所有多余的字段
        # ==================================================================
        valid_field_names = {f.name for f in fields(KnowledgeCase)}

        for raw in data.get("cases", []):
            # 兼容旧版'makespan'键名
            if 'makespan' in raw and 'workflow_makespan' not in raw:
                raw['workflow_makespan'] = raw.pop('makespan')

            # 过滤掉所有不在KnowledgeCase定义中的键
            filtered_raw = {k: v for k, v in raw.items() if k in valid_field_names}

            # 检查是否所有必需的字段都存在
            # (此步骤可选，但可以增加调试信息)
            missing_keys = valid_field_names - filtered_raw.keys()
            # 过滤掉可选字段
            required_keys = {f.name for f in fields(KnowledgeCase) if f.default is None and f.default_factory is None}
            missing_required = required_keys - filtered_raw.keys()

            if not missing_required:
                try:
                    case = KnowledgeCase(**filtered_raw)
                    kb.add_case(case)
                except TypeError as e:
                    print(f"[警告] 创建KnowledgeCase失败，跳过该记录。错误: {e}")
                    print(f"[调试] 过滤后的数据: {filtered_raw}")
            else:
                print(f"[警告] 记录缺少必要字段 {missing_required}，跳过。")
                print(f"[调试] 原始数据: {raw}")
                
        return kb

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "JSONKnowledgeBase":
        path = Path(path)
        if not path.exists():
            print(f"[错误] 知识库文件不存在: {path}")
            return cls() # 返回一个空的知识库
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    # ... (其余方法保持不变) ...
    def quality_filter(self, top_p: float = 0.5):
        scored = [c for c in self.cases if c.quality_score is not None]
        if not scored:
            return
        scored.sort(key=lambda c: c.quality_score, reverse=True)
        cutoff = int(len(scored) * top_p)
        keep_set = set(scored[:cutoff])
        new_cases = [c for c in self.cases if c in keep_set or c.quality_score is None]
        self.cases = []
        self.case_index.clear()
        for c in new_cases:
            self.add_case(c)

    def compute_quality_scores(self, baseline_makespan: float | None = None):
        makespans = [c.workflow_makespan for c in self.cases if c.workflow_makespan]
        if not makespans:
            return
        if baseline_makespan is None:
            sorted_m = sorted(makespans)
            idx = int(0.9 * (len(sorted_m)-1))
            baseline_makespan = sorted_m[idx]
        for c in self.cases:
            if c.workflow_makespan and c.workflow_makespan > 0:
                c.quality_score = float(baseline_makespan / c.workflow_makespan)
            else:
                c.quality_score = None

    def top_k(self, k: int) -> list[KnowledgeCase]:
        if k <= 0:
            return []
        scored = [c for c in self.cases if c.quality_score is not None]
        if not scored:
            return self.cases[:k]
        scored.sort(key=lambda c: c.quality_score, reverse=True)
        return scored[:k]


__all__ = ["KnowledgeCase", "JSONKnowledgeBase"]