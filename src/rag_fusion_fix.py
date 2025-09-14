import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class RAGFusionFix:
    """修复后的RAG融合机制"""
    
    def __init__(self):
        self.fusion_history = []
        self.rag_confidence_history = []
        
    def calculate_rag_confidence(self, rag_suggestions):
        """
        计算RAG建议的置信度
        
        Args:
            rag_suggestions: RAG建议列表，每个元素包含调度建议和置信度
        
        Returns:
            置信度分数 [0, 1]
        """
        if not rag_suggestions:
            return 0.0
        
        # 1. 计算建议的方差（方差越大，置信度越低）
        scores = [s.get('score', 0.0) for s in rag_suggestions]
        variance = np.var(scores)
        
        # 2. 计算最大值（最大值越大，置信度越高）
        max_score = max(scores) if scores else 0.0
        
        # 3. 使用tanh函数将置信度限制在[0, 1]范围内
        confidence = np.tanh(max_score * 10.0 - variance * 5.0)
        
        # 4. 确保置信度在[0, 1]范围内
        confidence = max(0.0, min(1.0, confidence))
        
        # 5. 记录历史
        self.rag_confidence_history.append(confidence)
        
        return confidence
    
    def enhance_rag_suggestions(self, rag_suggestions, available_nodes, current_loads):
        """
        增强RAG建议，处理全零向量问题
        
        Args:
            rag_suggestions: 原始RAG建议
            available_nodes: 可用节点列表
            current_loads: 当前负载情况
        
        Returns:
            增强后的RAG建议
        """
        if not rag_suggestions:
            # 如果没有RAG建议，生成启发式建议
            return self._generate_heuristic_suggestions(available_nodes, current_loads)
        
        # 检查是否为全零向量
        scores = [s.get('score', 0.0) for s in rag_suggestions]
        if all(score == 0.0 for score in scores):
            # 如果是全零向量，使用启发式建议
            logger.warning("RAG suggestions are all zeros, using heuristic suggestions")
            return self._generate_heuristic_suggestions(available_nodes, current_loads)
        
        # 否则，对原始建议进行增强
        enhanced_suggestions = []
        for suggestion in rag_suggestions:
            # 增加基于负载的调整
            node_name = suggestion.get('node')
            if node_name in current_loads:
                load = current_loads[node_name]
                # 负载越高，分数越低
                load_penalty = 1.0 - min(1.0, load)
                enhanced_score = suggestion.get('score', 0.0) * (0.7 + 0.3 * load_penalty)
            else:
                enhanced_score = suggestion.get('score', 0.0)
            
            enhanced_suggestions.append({
                'node': node_name,
                'score': enhanced_score,
                'original_score': suggestion.get('score', 0.0)
            })
        
        return enhanced_suggestions
    
    def _generate_heuristic_suggestions(self, available_nodes, current_loads):
        """
        生成启发式建议
        
        Args:
            available_nodes: 可用节点列表
            current_loads: 当前负载情况
        
        Returns:
            启发式建议列表
        """
        suggestions = []
        
        # 基于负载的启发式
        for node in available_nodes:
            load = current_loads.get(node, 0.0)
            # 负载越低，分数越高
            score = 1.0 - min(1.0, load)
            suggestions.append({
                'node': node,
                'score': score,
                'heuristic': True
            })
        
        # 按分数排序
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return suggestions
    
    def dynamic_fusion(self, q_values, rag_suggestions, current_loads, training_progress):
        """
        动态融合DRL Q值和RAG建议
        
        Args:
            q_values: DRL Q值列表
            rag_suggestions: RAG建议列表
            current_loads: 当前负载情况
            training_progress: 训练进度 [0, 1]
        
        Returns:
            融合后的决策结果
        """
        # 1. 增强RAG建议
        enhanced_rag = self.enhance_rag_suggestions(rag_suggestions, list(current_loads.keys()), current_loads)
        
        # 2. 计算RAG置信度
        rag_confidence = self.calculate_rag_confidence(enhanced_rag)
        
        # 3. 动态调整融合权重
        # 训练初期更依赖RAG，后期更依赖DRL
        base_alpha = 0.3 + 0.4 * training_progress  # DRL权重
        base_beta = 0.7 - 0.4 * training_progress    # RAG权重
        
        # 根据RAG置信度调整权重
        if rag_confidence < 0.3:
            # RAG置信度低，减少RAG权重
            beta = base_beta * 0.5
            alpha = 1.0 - beta
        elif rag_confidence > 0.7:
            # RAG置信度高，增加RAG权重
            beta = min(0.8, base_beta * 1.2)
            alpha = 1.0 - beta
        else:
            # 中等情况，使用基础权重
            alpha, beta = base_alpha, base_beta
        
        # 4. 归一化Q值
        q_norm = self._normalize_values(q_values)
        
        # 5. 归一化RAG分数
        rag_scores = [s.get('score', 0.0) for s in enhanced_rag]
        rag_norm = self._normalize_values(rag_scores)
        
        # 6. 融合决策
        fused_scores = []
        for i, (q, r) in enumerate(zip(q_norm, rag_norm)):
            fused_score = alpha * q + beta * r
            fused_scores.append(fused_score)
        
        # 7. 选择最佳决策
        best_idx = np.argmax(fused_scores)
        
        # 8. 记录融合历史
        self.fusion_history.append({
            'alpha': alpha,
            'beta': beta,
            'rag_confidence': rag_confidence,
            'best_idx': best_idx
        })
        
        return {
            'index': best_idx,
            'alpha': alpha,
            'beta': beta,
            'rag_confidence': rag_confidence,
            'fused_scores': fused_scores,
            'q_norm': q_norm,
            'rag_norm': rag_norm
        }
    
    def _normalize_values(self, values):
        """归一化值列表"""
        if not values:
            return []
        
        values = np.array(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == min_val:
            # 所有值相同，返回均匀分布
            return [1.0 / len(values)] * len(values)
        
        # 归一化到[0, 1]
        normalized = (values - min_val) / (max_val - min_val)
        return normalized.tolist()
    
    def debug_fusion_info(self, task_id, fusion_result):
        """记录融合调试信息"""
        logger.info(f"Fusion Debug - Task={task_id}, "
                   f"Alpha={fusion_result['alpha']:.3f}, "
                   f"Beta={fusion_result['beta']:.3f}, "
                   f"RAGConfidence={fusion_result['rag_confidence']:.3f}, "
                   f"BestIndex={fusion_result['best_idx']}")