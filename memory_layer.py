"""
Memory Layer Module
Handles conversation history, knowledge base, and agent state with vector similarity search.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

class VectorStore:
    """Simple in-memory vector store using cosine similarity."""
    
    def __init__(self, embedding_dim: int = 128):
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.embedding_dim = embedding_dim
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Generate simple deterministic embedding from text."""
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(self.embedding_dim)
    
    def add(self, key: str, text: str, metadata: Dict[str, Any] = None):
        """Add vector and metadata to store."""
        vector = self._simple_embedding(text)
        vector = vector / (np.linalg.norm(vector) + 1e-8)  # Normalize
        self.vectors[key] = vector
        self.metadata[key] = metadata or {}
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search using cosine similarity."""
        query_vec = self._simple_embedding(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        results = []
        for key, vector in self.vectors.items():
            similarity = np.dot(query_vec, vector)
            results.append((key, similarity, self.metadata[key]))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class ConversationMemory:
    """Manages conversation history with timestamps."""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.turn_count = 0
    
    def add_turn(self, user_input: str, response: str, agent_trace: List[Dict] = None):
        """Record a conversation turn."""
        turn = {
            "turn_id": self.turn_count,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "agent_trace": agent_trace or [],
            "unix_time": time.time()
        }
        self.history.append(turn)
        self.turn_count += 1
    
    def get_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """Retrieve last n turns for context."""
        return self.history[-num_turns:]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve full conversation history."""
        return self.history


class KnowledgeBase:
    """Persistent store of facts and findings with provenance."""
    
    def __init__(self):
        self.facts: Dict[str, Dict[str, Any]] = {}
        self.vector_store = VectorStore()
        self.fact_counter = 0
    
    def add_fact(self, content: str, topics: List[str], source_agent: str, 
                 confidence: float = 1.0) -> str:
        """Store a fact with metadata."""
        fact_id = f"fact_{self.fact_counter}"
        self.fact_counter += 1
        
        fact = {
            "fact_id": fact_id,
            "content": content,
            "topics": topics,
            "source_agent": source_agent,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time()
        }
        self.facts[fact_id] = fact
        
        # Add to vector store for similarity search
        self.vector_store.add(fact_id, content, fact)
        return fact_id
    
    def search_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Find facts by topic."""
        results = []
        for fact in self.facts.values():
            if topic.lower() in [t.lower() for t in fact["topics"]]:
                results.append(fact)
        return results
    
    def search_by_similarity(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find facts by vector similarity."""
        results = self.vector_store.search(query, top_k)
        return [metadata for _, _, metadata in results if metadata]
    
    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Retrieve all facts."""
        return list(self.facts.values())


class AgentStateMemory:
    """Track what each agent learned and accomplished."""
    
    def __init__(self):
        self.agent_states: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def record_action(self, agent_name: str, action: str, input_data: Any, 
                     output_data: Any, confidence: float = 1.0):
        """Record an agent's action."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "input": input_data,
            "output": output_data,
            "confidence": confidence,
            "unix_time": time.time()
        }
        self.agent_states[agent_name].append(record)
    
    def get_agent_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get history for a specific agent."""
        return self.agent_states.get(agent_name, [])
    
    def get_all_states(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all agent states."""
        return dict(self.agent_states)


class MemoryAgent:
    """Coordinator for all memory operations."""
    
    def __init__(self):
        self.conversation_memory = ConversationMemory()
        self.knowledge_base = KnowledgeBase()
        self.agent_state_memory = AgentStateMemory()
    
    def store_finding(self, content: str, topics: List[str], source_agent: str,
                     confidence: float = 1.0) -> str:
        """Store a finding in knowledge base."""
        return self.knowledge_base.add_fact(content, topics, source_agent, confidence)
    
    def search_knowledge(self, query: str, search_type: str = "similarity") -> List[Dict]:
        """Search knowledge base by similarity or topic."""
        if search_type == "topic":
            return self.knowledge_base.search_by_topic(query)
        else:
            return self.knowledge_base.search_by_similarity(query)
    
    def record_agent_action(self, agent_name: str, action: str, input_data: Any,
                           output_data: Any, confidence: float = 1.0):
        """Record an agent's action."""
        self.agent_state_memory.record_action(agent_name, action, input_data, output_data, confidence)
    
    def add_conversation_turn(self, user_input: str, response: str, agent_trace: List[Dict] = None):
        """Add a conversation turn."""
        self.conversation_memory.add_turn(user_input, response, agent_trace)
    
    def get_recent_context(self, num_turns: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        return self.conversation_memory.get_context(num_turns)
    
    def get_full_history(self) -> Dict[str, Any]:
        """Get all memories."""
        return {
            "conversation_history": self.conversation_memory.get_all(),
            "knowledge_base": self.knowledge_base.get_all_facts(),
            "agent_states": self.agent_state_memory.get_all_states()
        }
    
    def has_prior_knowledge(self, query: str) -> bool:
        """Check if relevant knowledge exists."""
        results = self.knowledge_base.search_by_similarity(query, top_k=1)
        return len(results) > 0