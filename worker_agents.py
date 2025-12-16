"""
Worker Agents Module
Implements Research, Analysis, and supporting agents.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import json

class WorkerAgent(ABC):
    """Base class for all worker agents."""
    
    def __init__(self, name: str, memory_agent=None):
        self.name = name
        self.memory_agent = memory_agent
        self.execution_trace = []
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's task."""
        pass
    
    def log_execution(self, action: str, input_data: Any, output_data: Any, 
                     confidence: float = 1.0):
        """Log execution for tracing."""
        trace_entry = {
            "agent": self.name,
            "action": action,
            "input": input_data,
            "output": output_data,
            "confidence": confidence
        }
        self.execution_trace.append(trace_entry)
        
        if self.memory_agent:
            self.memory_agent.record_agent_action(
                self.name, action, input_data, output_data, confidence
            )
    
    def clear_trace(self):
        """Clear execution trace for new task."""
        self.execution_trace = []


class ResearchAgent(WorkerAgent):
    """Retrieves information from a simulated knowledge base."""
    
    # Mock knowledge base for simulation
    KNOWLEDGE_BASE = {
        "machine learning optimization": {
            "gradient descent": {
                "description": "Basic optimization algorithm that iteratively moves in direction of negative gradient",
                "pros": ["Simple", "Widely used", "Works well for convex problems"],
                "cons": ["Can be slow", "May get stuck in local minima", "Needs learning rate tuning"],
                "use_cases": ["Linear regression", "Neural networks"]
            },
            "adam": {
                "description": "Adaptive learning rate optimization combining momentum and RMSprop",
                "pros": ["Fast convergence", "Adaptive learning rates", "Robust to noisy data"],
                "cons": ["More memory intensive", "Can diverge in some cases", "Complex hyperparameters"],
                "use_cases": ["Deep learning", "Computer vision", "NLP"]
            },
            "sgd": {
                "description": "Stochastic Gradient Descent updates weights using random samples",
                "pros": ["Fast", "Memory efficient", "Good generalization"],
                "cons": ["Noisy updates", "Requires learning rate scheduling", "Less stable"],
                "use_cases": ["Large scale problems", "Online learning"]
            },
            "rmsprop": {
                "description": "Root Mean Square Propagation - adapts learning rate based on magnitude of gradients",
                "pros": ["Works well with RNNs", "Adaptive learning rates", "Stable"],
                "cons": ["Sensitive to learning rate", "Less intuitive", "Not always fastest"],
                "use_cases": ["RNNs", "Time series", "Speech recognition"]
            },
            "momentum": {
                "description": "Accumulates gradient information over time with momentum term",
                "pros": ["Faster convergence", "Reduces oscillations", "Better generalization"],
                "cons": ["Extra hyperparameter", "May overshoot", "Requires tuning"],
                "use_cases": ["Deep neural networks", "Computer vision"]
            }
        },
        "deep learning frameworks": {
            "tensorflow": "Open-source ML platform by Google, production-ready, supports distributed training",
            "pytorch": "Deep learning framework emphasizing dynamic graphs and research flexibility",
            "keras": "High-level API running on TensorFlow, intuitive and beginner-friendly",
            "jax": "Composable transformations of NumPy programs, for research",
            "mxnet": "Scalable deep learning framework supporting multiple programming languages"
        },
        "neural network architectures": {
            "cnn": {
                "description": "Convolutional Neural Networks - specialized for image processing with convolutional layers",
                "pros": ["Excellent for image tasks", "Parameter sharing reduces complexity", "Highly efficient"],
                "cons": ["Requires large datasets", "Computationally intensive training", "Less suitable for sequential data"],
                "use_cases": ["Image classification", "Object detection", "Computer vision"]
            },
            "rnn": {
                "description": "Recurrent Neural Networks - specialized for sequential data with feedback connections",
                "pros": ["Handles variable length sequences", "Captures temporal dependencies", "Memory of past inputs"],
                "cons": ["Slow training", "Vanishing gradient problem", "Limited long-term memory"],
                "use_cases": ["Time series", "Language modeling", "Speech recognition"]
            },
            "transformer": {
                "description": "Self-attention based architecture - foundation of modern NLP using multi-head attention mechanisms",
                "pros": ["Parallelizable training", "Captures long-range dependencies", "State-of-the-art performance"],
                "cons": ["High computational cost", "Large memory requirements", "Complex architecture"],
                "use_cases": ["NLP tasks", "Machine translation", "Question answering"]
            },
            "gan": {
                "description": "Generative Adversarial Networks - for generating synthetic data through adversarial training",
                "pros": ["Generates realistic data", "Unsupervised learning", "Creative applications"],
                "cons": ["Unstable training", "Mode collapse issues", "Difficult to evaluate"],
                "use_cases": ["Image generation", "Data augmentation", "Style transfer"]
            },
            "lstm": {
                "description": "Long Short-Term Memory - improved RNN for long-term dependencies with gating mechanisms",
                "pros": ["Solves vanishing gradient problem", "Long-term memory", "Better than basic RNN"],
                "cons": ["Slower than CNN", "Still limited very long sequences", "More parameters than RNN"],
                "use_cases": ["Sequence prediction", "Machine translation", "Time series forecasting"]
            }
        },
        "reinforcement learning": {
            "q-learning": {
                "description": "Off-policy algorithm learning action-value function using Bellman equation",
                "pros": ["Model-free approach", "Guaranteed convergence", "Off-policy learning"],
                "cons": ["Slow convergence", "Large memory for Q-tables", "Not suitable for continuous spaces"],
                "use_cases": ["Game playing", "Robot control", "Resource allocation"]
            },
            "policy gradient": {
                "description": "On-policy algorithm directly optimizing policy using gradient ascent",
                "pros": ["Works with continuous actions", "Direct policy optimization", "Sample efficient variants exist"],
                "cons": ["High variance", "Requires trajectory collection", "Local optima risk"],
                "use_cases": ["Continuous control", "Robotic learning", "Game AI"]
            },
            "deep reinforcement learning": {
                "description": "Combines deep neural networks with reinforcement learning for complex tasks",
                "pros": ["Handles high-dimensional inputs", "End-to-end learning", "Solves complex problems"],
                "cons": ["Requires massive data", "Unstable training", "Difficult to debug"],
                "use_cases": ["Autonomous driving", "Game mastery", "Complex robotics"]
            },
            "actor-critic": {
                "description": "Combines policy gradient and value function methods using separate actor and critic networks",
                "pros": ["Lower variance than policy gradient", "More stable than pure RL", "Good convergence"],
                "cons": ["Two networks to train", "Increased complexity", "Hyperparameter tuning needed"],
                "use_cases": ["Control tasks", "Game playing", "Robotics"]
            },
            "multi-armed bandit": {
                "description": "Balances exploration and exploitation in sequential decision making with limited feedback",
                "pros": ["Simple to implement", "Low computational cost", "Effective exploration strategies"],
                "cons": ["Limited to simple problems", "No state dependency", "Poor for complex tasks"],
                "use_cases": ["Online advertising", "A/B testing", "Recommendation systems"]
            }
        },
        "machine learning approaches": {
            "supervised learning": {
                "description": "Learning from labeled data to map inputs to outputs with known ground truth",
                "pros": ["High accuracy possible", "Clear objective function", "Well-studied methods"],
                "cons": ["Requires labeled data", "Expensive labeling", "May overfit with limited data"],
                "use_cases": ["Classification", "Regression", "Prediction tasks"]
            },
            "unsupervised learning": {
                "description": "Learning patterns from unlabeled data without predefined outputs",
                "pros": ["No labeling required", "Discovers hidden patterns", "Cost-effective"],
                "cons": ["Harder to evaluate", "May find spurious patterns", "Less control over output"],
                "use_cases": ["Clustering", "Dimensionality reduction", "Feature discovery"]
            },
            "semi-supervised learning": {
                "description": "Combines labeled and unlabeled data to improve learning with limited supervision",
                "pros": ["Uses all available data", "More accurate than unsupervised", "Less labeling needed"],
                "cons": ["Assumptions about unlabeled data", "Complex algorithms", "Moderate effectiveness"],
                "use_cases": ["Text classification", "Image labeling", "Medical diagnosis"]
            },
            "transfer learning": {
                "description": "Leverages knowledge from source task to improve learning on target task",
                "pros": ["Faster training", "Better generalization", "Works with small datasets"],
                "cons": ["Domain mismatch issues", "Fine-tuning required", "Knowledge relevance matters"],
                "use_cases": ["Computer vision", "NLP", "Domain adaptation"]
            },
            "federated learning": {
                "description": "Distributed learning where models train on decentralized data without centralization",
                "pros": ["Privacy-preserving", "Decentralized control", "Works with edge devices"],
                "cons": ["Communication overhead", "Slower convergence", "Debugging challenges"],
                "use_cases": ["Mobile devices", "Healthcare", "Privacy-critical applications"]
            }
        },
        "computational efficiency": {
            "model compression": {
                "description": "Reducing model size and computational requirements while maintaining performance",
                "pros": ["Faster inference", "Lower memory use", "Better mobile deployment"],
                "cons": ["Performance degradation", "Complex process", "Requires retraining"],
                "use_cases": ["Mobile inference", "Edge deployment", "Real-time systems"]
            },
            "quantization": {
                "description": "Reducing precision of model weights and activations from float32 to lower bit-widths",
                "pros": ["Significant speedup", "Reduced memory", "Hardware acceleration support"],
                "cons": ["Accuracy loss", "Implementation complexity", "Hardware dependent"],
                "use_cases": ["Edge devices", "Mobile phones", "Embedded systems"]
            },
            "pruning": {
                "description": "Removing unnecessary weights and neurons to simplify model architecture",
                "pros": ["Reduced parameters", "Faster training", "Lower inference latency"],
                "cons": ["Manual threshold tuning", "Structured pruning complexity", "Potential accuracy drop"],
                "use_cases": ["Model optimization", "Energy-efficient computing", "Resource-constrained devices"]
            },
            "knowledge distillation": {
                "description": "Training smaller student model to mimic large teacher model behavior",
                "pros": ["Compact models", "Preserved performance", "Training acceleration"],
                "cons": ["Teacher dependency", "Complex training procedure", "Hyperparameter sensitivity"],
                "use_cases": ["Model compression", "Transfer learning", "Ensemble learning"]
            },
            "batch normalization": {
                "description": "Normalizing layer inputs to accelerate training and improve stability",
                "pros": ["Faster convergence", "Higher learning rates possible", "Regularization effect"],
                "cons": ["Batch size dependent", "Inference different from training", "Computational overhead"],
                "use_cases": ["Deep neural networks", "Computer vision", "Natural language processing"]
            }
        }
    }
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research task."""
        query = task.get("query", "")
        search_area = task.get("search_area", "machine learning optimization")
        
        self.clear_trace()
        
        findings = self._search_knowledge_base(query, search_area)
        
        output = {
            "success": len(findings) > 0,
            "findings": findings,
            "query": query,
            "search_area": search_area,
            "result_count": len(findings)
        }
        
        self.log_execution(
            action="knowledge_base_search",
            input_data=task,
            output_data=output,
            confidence=0.9 if findings else 0.5
        )
        
        return output
    
    def _search_knowledge_base(self, query: str, search_area: str) -> List[Dict[str, Any]]:
        """Search mock knowledge base."""
        findings = []
        query_lower = query.lower()
        
        area_data = self.KNOWLEDGE_BASE.get(search_area.lower(), {})
        
        print(f"[ResearchAgent] Searching in area: {search_area}")
        print(f"[ResearchAgent] Query: '{query_lower}'")
        print(f"[ResearchAgent] Topics in area: {list(area_data.keys())}")
        
        # If query contains keywords like "all", "information", "techniques", "types", "main", search all topics
        generic_keywords = ["information", "techniques", "all", "find", "tell me", "types", "main", "what are"]
        is_generic_search = any(kw in query_lower for kw in generic_keywords)
        
        print(f"[ResearchAgent] Is generic search: {is_generic_search}")
        
        for topic, details in area_data.items():
            topic_lower = topic.lower()
            
            # Match if:
            # 1. Topic is in query
            # 2. Query is in topic
            # 3. Generic search (find all, tell me about, what are, etc.)
            if topic_lower in query_lower or query_lower in topic_lower or is_generic_search:
                print(f"[ResearchAgent] ✓ MATCH FOUND: {topic}")
                if isinstance(details, dict) and "description" in details:
                    # Detailed topic
                    findings.append({
                        "topic": topic,
                        "content": details,
                        "type": "detailed"
                    })
                else:
                    # Simple entry
                    findings.append({
                        "topic": topic,
                        "content": details,
                        "type": "simple"
                    })
        
        print(f"[ResearchAgent] Total findings: {len(findings)}")
        return findings


class AnalysisAgent(WorkerAgent):
    """Performs comparisons, reasoning, and calculations on data."""
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis task."""
        analysis_type = task.get("analysis_type", "comparison")
        data = task.get("data", [])
        criteria = task.get("criteria", [])
        
        self.clear_trace()
        
        if analysis_type == "comparison":
            result = self._perform_comparison(data, criteria)
        elif analysis_type == "ranking":
            result = self._rank_items(data, criteria)
        elif analysis_type == "summary":
            result = self._summarize_data(data)
        else:
            result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        self.log_execution(
            action=f"analysis_{analysis_type}",
            input_data=task,
            output_data=result,
            confidence=0.85
        )
        
        return result
    
    def _perform_comparison(self, items: List[Dict], criteria: List[str]) -> Dict[str, Any]:
        """Compare items across criteria."""
        if not items:
            return {"error": "No items to compare"}
        
        comparison = {
            "type": "detailed_comparison",
            "items": [],
            "summary": ""
        }
        
        for item in items:
            if isinstance(item, dict):
                item_analysis = {
                    "name": item.get("topic", "Unknown"),
                    "strengths": item.get("content", {}).get("pros", []) if isinstance(item.get("content"), dict) else [],
                    "weaknesses": item.get("content", {}).get("cons", []) if isinstance(item.get("content"), dict) else [],
                    "use_cases": item.get("content", {}).get("use_cases", []) if isinstance(item.get("content"), dict) else []
                }
                comparison["items"].append(item_analysis)
        
        if comparison["items"]:
            strengths_count = [len(i["strengths"]) for i in comparison["items"]]
            best_idx = strengths_count.index(max(strengths_count)) if strengths_count else 0
            best_name = comparison["items"][best_idx]["name"] if comparison["items"] else "Unknown"
            comparison["summary"] = f"Based on criteria analysis, {best_name} shows strongest features."
        
        return comparison
    
    def _rank_items(self, items: List[Dict], criteria: List[str]) -> Dict[str, Any]:
        """Rank items based on criteria."""
        if not items:
            return {"error": "No items to rank"}
        
        # Simple scoring: count pros and divide by cons
        scores = []
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("content"), dict):
                pros = len(item.get("content", {}).get("pros", []))
                cons = len(item.get("content", {}).get("cons", []))
                score = pros / (cons + 1)  # Avoid division by zero
                scores.append({
                    "name": item.get("topic", "Unknown"),
                    "score": round(score, 2),
                    "pros_count": pros,
                    "cons_count": cons
                })
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "type": "ranking",
            "ranked_items": scores,
            "top_choice": scores[0]["name"] if scores else "No ranking available"
        }
    
    def _summarize_data(self, items: List[Dict]) -> Dict[str, Any]:
        """Summarize a collection of data."""
        summary = {
            "type": "summary",
            "total_items": len(items),
            "items_summary": []
        }
        
        for item in items:
            if isinstance(item, dict):
                summary["items_summary"].append({
                    "name": item.get("topic", "Unknown"),
                    "description": str(item.get("content", ""))[:100] + "..."
                })
        
        return summary


class DataAggregatorAgent(WorkerAgent):
    """Aggregates and synthesizes information from multiple sources."""
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation task."""
        data_sources = task.get("data_sources", [])
        aggregation_type = task.get("aggregation_type", "merge")
        
        self.clear_trace()
        
        if aggregation_type == "merge":
            result = self._merge_data(data_sources)
        elif aggregation_type == "synthesize":
            result = self._synthesize_data(data_sources)
        else:
            result = {"error": f"Unknown aggregation type: {aggregation_type}"}
        
        self.log_execution(
            action=f"aggregation_{aggregation_type}",
            input_data=task,
            output_data=result,
            confidence=0.8
        )
        
        return result
    
    def _merge_data(self, sources: List[Dict]) -> Dict[str, Any]:
        """Merge data from multiple sources."""
        merged = {
            "type": "merged_data",
            "source_count": len(sources),
            "combined_findings": []
        }
        
        for source in sources:
            if isinstance(source, dict):
                merged["combined_findings"].append(source)
        
        return merged
    
    def _synthesize_data(self, sources: List[Dict]) -> Dict[str, Any]:
        """Synthesize data into coherent narrative."""
        synthesis = {
            "type": "synthesized_analysis",
            "key_insights": [],
            "integrated_view": ""
        }
        
        # Extract key points
        for source in sources:
            if isinstance(source, dict):
                if "findings" in source:
                    for finding in source["findings"]:
                        if isinstance(finding, dict):
                            synthesis["key_insights"].append(
                                finding.get("topic", "Unknown insight")
                            )
        
        synthesis["integrated_view"] = f"Analysis synthesized from {len(sources)} sources with {len(synthesis['key_insights'])} key insights."
        
        return synthesis