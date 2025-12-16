"""
Coordinator Agent Module
Orchestrates worker agents and manages system state.
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
from worker_agents import ResearchAgent, AnalysisAgent, DataAggregatorAgent
from memory_layer import MemoryAgent

class TaskComplexity(Enum):
    """Enumeration for task complexity levels."""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3

class Coordinator:
    """Central orchestrator managing all agents and task decomposition."""
    
    def __init__(self):
        self.memory_agent = MemoryAgent()
        self.research_agent = ResearchAgent(name="ResearchAgent", memory_agent=self.memory_agent)
        self.analysis_agent = AnalysisAgent(name="AnalysisAgent", memory_agent=self.memory_agent)
        self.aggregator_agent = DataAggregatorAgent(name="AggregatorAgent", memory_agent=self.memory_agent)
        
        self.execution_trace = []
        self.task_queue = []
        self.current_task_id = 0
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point: accept user query, analyze, decompose, and route to agents.
        """
        print(f"\n{'='*80}")
        print(f"USER QUERY: {user_query}")
        print(f"{'='*80}\n")
        
        # Check for prior knowledge ONLY if query is very similar to previous
        # Don't use memory for comparison/analysis queries - those need fresh analysis
        is_comparison_query = any(kw in user_query.lower() for kw in 
                                  ["compare", "difference", "versus", "better", "which", "recommendation"])
        
        has_prior = False
        if not is_comparison_query:
            has_prior = self.memory_agent.has_prior_knowledge(user_query)
            print(f"[COORDINATOR] Prior knowledge check: {'FOUND' if has_prior else 'NOT FOUND'}")
        else:
            print(f"[COORDINATOR] Comparison query detected - skipping memory cache for fresh analysis")
        
        if has_prior and not is_comparison_query:
            prior_findings = self.memory_agent.search_knowledge(user_query, search_type="similarity")
            print(f"[COORDINATOR] Retrieved {len(prior_findings)} prior findings")
            if prior_findings:
                return self._format_response_from_memory(user_query, prior_findings)
        
        # Analyze task complexity
        complexity = self._analyze_complexity(user_query)
        print(f"[COORDINATOR] Task complexity: {complexity.name}")
        
        # Decompose into subtasks
        subtasks = self._decompose_task(user_query, complexity)
        print(f"[COORDINATOR] Decomposed into {len(subtasks)} subtasks\n")
        
        # Execute subtasks
        task_results = self._execute_subtasks(subtasks)
        
        # Synthesize final answer
        final_response = self._synthesize_response(user_query, task_results)
        
        # Store in memory
        self._store_findings(final_response, user_query)
        
        # Add to conversation history
        self.memory_agent.add_conversation_turn(
            user_input=user_query,
            response=final_response["answer"],
            agent_trace=self.execution_trace
        )
        
        self.execution_trace = []  # Clear trace for next query
        
        return final_response
    
    def _analyze_complexity(self, query: str) -> TaskComplexity:
        """Analyze query complexity to determine agent orchestration strategy."""
        keywords_simple = ["what is", "define", "list", "who is", "tell me about"]
        keywords_moderate = ["compare", "analyze", "how", "why", "explain", "difference", 
                            "versus", "versus", "better", "best", "effectiveness", "which"]
        keywords_complex = ["optimize", "combine", "integrate", "comprehensive", "find all",
                           "all", "and analyze"]
        
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in keywords_complex):
            return TaskComplexity.COMPLEX
        elif any(kw in query_lower for kw in keywords_moderate):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _decompose_task(self, query: str, complexity: TaskComplexity) -> List[Dict[str, Any]]:
        """Decompose user query into agent-executable subtasks."""
        subtasks = []
        
        if complexity == TaskComplexity.SIMPLE:
            # Single research task
            subtasks.append({
                "task_id": self.current_task_id,
                "agent": "research",
                "action": "search",
                "query": query,
                "search_area": self._infer_search_area(query),
                "priority": 1
            })
            self.current_task_id += 1
        
        elif complexity == TaskComplexity.MODERATE:
            # Research followed by analysis
            subtasks.append({
                "task_id": self.current_task_id,
                "agent": "research",
                "action": "search",
                "query": query,
                "search_area": self._infer_search_area(query),
                "priority": 1
            })
            self.current_task_id += 1
            
            subtasks.append({
                "task_id": self.current_task_id,
                "agent": "analysis",
                "action": "analyze",
                "analysis_type": self._infer_analysis_type(query),
                "depends_on": subtasks[-1]["task_id"],
                "priority": 2
            })
            self.current_task_id += 1
        
        else:  # COMPLEX
            # Multi-stage: research, analysis, aggregation
            subtasks.append({
                "task_id": self.current_task_id,
                "agent": "research",
                "action": "search",
                "query": query,
                "search_area": self._infer_search_area(query),
                "priority": 1
            })
            research_task_id = self.current_task_id
            self.current_task_id += 1
            
            subtasks.append({
                "task_id": self.current_task_id,
                "agent": "analysis",
                "action": "analyze",
                "analysis_type": self._infer_analysis_type(query),
                "depends_on": research_task_id,
                "priority": 2
            })
            analysis_task_id = self.current_task_id
            self.current_task_id += 1
            
            subtasks.append({
                "task_id": self.current_task_id,
                "agent": "aggregator",
                "action": "synthesize",
                "aggregation_type": "synthesize",
                "depends_on": [research_task_id, analysis_task_id],
                "priority": 3
            })
            self.current_task_id += 1
        
        return subtasks
    
    def _infer_search_area(self, query: str) -> str:
        """Infer the search domain from query."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["reinforcement", "q-learning", "policy", "actor-critic", "bandit"]):
            return "reinforcement learning"
        elif any(kw in query_lower for kw in ["supervised", "unsupervised", "semi-supervised", "transfer", "federated"]):
            return "machine learning approaches"
        elif any(kw in query_lower for kw in ["compression", "quantization", "pruning", "distillation", "batch norm", "efficiency"]):
            return "computational efficiency"
        elif any(kw in query_lower for kw in ["cnn", "rnn", "transformer", "lstm", "gan", "neural network", "architecture"]):
            return "neural network architectures"
        elif any(kw in query_lower for kw in ["framework", "tensorflow", "pytorch", "keras"]):
            return "deep learning frameworks"
        elif any(kw in query_lower for kw in ["optim", "gradient", "adam", "sgd"]):
            return "machine learning optimization"
        else:
            return "machine learning optimization"
    
    def _infer_analysis_type(self, query: str) -> str:
        """Infer analysis type from query."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["compare", "versus", "difference", "which"]):
            return "comparison"
        elif any(kw in query_lower for kw in ["rank", "best", "most effective"]):
            return "ranking"
        elif any(kw in query_lower for kw in ["summarize", "overview", "what", "all"]):
            return "summary"
        else:
            return "comparison"
    
    def _execute_subtasks(self, subtasks: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Execute subtasks respecting dependencies."""
        results = {}
        completed = set()
        
        # Sort by priority
        subtasks.sort(key=lambda x: x.get("priority", 999))
        
        for subtask in subtasks:
            task_id = subtask["task_id"]
            
            # Check dependencies
            dependencies = subtask.get("depends_on", [])
            if isinstance(dependencies, int):
                dependencies = [dependencies]
            
            if not all(dep in completed for dep in dependencies):
                print(f"[COORDINATOR] Skipping task {task_id}: dependencies not met")
                continue
            
            print(f"[COORDINATOR] Executing subtask {task_id} ({subtask['agent']})")
            
            try:
                if subtask["agent"] == "research":
                    result = self._execute_research_task(subtask)
                elif subtask["agent"] == "analysis":
                    result = self._execute_analysis_task(subtask, results)
                elif subtask["agent"] == "aggregator":
                    result = self._execute_aggregation_task(subtask, results)
                else:
                    result = {"error": f"Unknown agent: {subtask['agent']}"}
                
                results[task_id] = result
                completed.add(task_id)
                print(f"[COORDINATOR] Task {task_id} completed successfully\n")
            
            except Exception as e:
                print(f"[COORDINATOR] Task {task_id} failed: {str(e)}")
                results[task_id] = {"error": str(e)}
                # Attempt fallback
                results[task_id] = self._fallback_strategy(subtask, results)
        
        return results
    
    def _execute_research_task(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research agent task."""
        task_input = {
            "query": subtask.get("query", ""),
            "search_area": subtask.get("search_area", "machine learning optimization")
        }
        result = self.research_agent.execute(task_input)
        
        # Debug: Print what was found
        print(f"[COORDINATOR] Research found {result.get('result_count', 0)} results")
        
        # Log to execution trace
        self.execution_trace.append({
            "agent": "ResearchAgent",
            "task_id": subtask["task_id"],
            "input": task_input,
            "output": result
        })
        
        return result
    
    def _execute_analysis_task(self, subtask: Dict[str, Any], 
                              prior_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Execute analysis agent task."""
        # Get data from prior research
        depends_on = subtask.get("depends_on")
        if depends_on in prior_results:
            research_result = prior_results[depends_on]
            findings = research_result.get("findings", [])
        else:
            findings = []
        
        task_input = {
            "analysis_type": subtask.get("analysis_type", "comparison"),
            "data": findings,
            "criteria": []
        }
        result = self.analysis_agent.execute(task_input)
        
        # Log to execution trace
        self.execution_trace.append({
            "agent": "AnalysisAgent",
            "task_id": subtask["task_id"],
            "input": task_input,
            "output": result
        })
        
        return result
    
    def _execute_aggregation_task(self, subtask: Dict[str, Any],
                                 prior_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Execute aggregation agent task."""
        # Collect all prior results
        depends_on = subtask.get("depends_on", [])
        if isinstance(depends_on, int):
            depends_on = [depends_on]
        
        data_sources = [prior_results.get(dep, {}) for dep in depends_on]
        
        task_input = {
            "aggregation_type": subtask.get("aggregation_type", "synthesize"),
            "data_sources": data_sources
        }
        result = self.aggregator_agent.execute(task_input)
        
        # Log to execution trace
        self.execution_trace.append({
            "agent": "AggregatorAgent",
            "task_id": subtask["task_id"],
            "input": task_input,
            "output": result
        })
        
        return result
    
    def _fallback_strategy(self, subtask: Dict[str, Any], 
                          prior_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Fallback strategy when a task fails."""
        print(f"[COORDINATOR] Attempting fallback for task {subtask['task_id']}")
        
        if subtask["agent"] == "analysis":
            # Fallback to summary analysis
            return self.analysis_agent.execute({
                "analysis_type": "summary",
                "data": prior_results.get(subtask.get("depends_on"), {}).get("findings", []),
                "criteria": []
            })
        
        return {"error": "Fallback failed"}
    
    def _synthesize_response(self, query: str, task_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Synthesize final response from task results."""
        print("[COORDINATOR] Synthesizing final response...")
        
        final_response = {
            "query": query,
            "answer": "",
            "sources": [],
            "confidence": 0.8,
            "agent_contributions": []
        }
        
        # Extract and format results
        for task_id, result in task_results.items():
            if "error" not in result:
                if "findings" in result and result["findings"]:
                    final_response["sources"].extend(result["findings"])
                    final_response["agent_contributions"].append("Research completed")
                
                if "items" in result and result["items"]:
                    final_response["agent_contributions"].append("Comparison/Analysis completed")
                
                if "combined_findings" in result or "integrated_view" in result:
                    final_response["agent_contributions"].append("Synthesis completed")
        
        # Build answer string with detailed content
        if final_response["sources"]:
            answer_parts = []
            for source in final_response["sources"]:
                if isinstance(source, dict):
                    topic = source.get("topic", "Unknown")
                    content = source.get("content", {})
                    
                    # Format based on content type
                    if isinstance(content, dict):
                        desc = content.get("description", "")
                        if desc:
                            answer_parts.append(f"\n{topic}:")
                            answer_parts.append(f"  {desc}")
                            
                            pros = content.get("pros", [])
                            if pros:
                                answer_parts.append(f"  Pros: {', '.join(pros)}")
                            
                            cons = content.get("cons", [])
                            if cons:
                                answer_parts.append(f"  Cons: {', '.join(cons)}")
                    else:
                        answer_parts.append(f"\n{topic}:")
                        answer_parts.append(f"  {str(content)}")
            
            final_response["answer"] = "Found information:" + "\n".join(answer_parts)
        else:
            final_response["answer"] = "No information found for this query."
        
        print(f"[COORDINATOR] Response synthesis complete\n")
        return final_response
    
    def _format_response_from_memory(self, query: str, findings: List[Dict]) -> Dict[str, Any]:
        """Format response using prior memory findings."""
        print("[COORDINATOR] Using cached knowledge...\n")
        
        answer_parts = []
        
        for finding in findings:
            content = finding.get("content", "")
            # Content now contains full details stored as string
            answer_parts.append(content)
        
        answer_text = "From previous discussion:\n\n" + "\n\n".join(answer_parts) if answer_parts else "No prior knowledge found."
        
        return {
            "query": query,
            "answer": answer_text,
            "from_memory": True,
            "confidence": 0.95,
            "agent_contributions": ["Memory retrieval"]
        }
    
    def _store_findings(self, response: Dict[str, Any], query: str):
        """Store important findings in knowledge base."""
        if response.get("sources"):
            for source in response["sources"]:
                if isinstance(source, dict):
                    topic = source.get("topic", "Finding")
                    content = source.get("content", {})
                    
                    # Store full content as string for readability in memory
                    if isinstance(content, dict):
                        desc = content.get("description", "")
                        pros = content.get("pros", [])
                        cons = content.get("cons", [])
                        use_cases = content.get("use_cases", [])
                        
                        full_content = f"{topic}: {desc}\nPros: {', '.join(pros)}\nCons: {', '.join(cons)}\nUse Cases: {', '.join(use_cases)}"
                    else:
                        full_content = f"{topic}: {str(content)}"
                    
                    self.memory_agent.store_finding(
                        content=full_content,
                        topics=self._extract_topics(query),
                        source_agent="Coordinator",
                        confidence=response.get("confidence", 0.8)
                    )
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics from query."""
        keywords = ["optimization", "learning", "neural network", "framework", "algorithm"]
        found_topics = [kw for kw in keywords if kw in query.lower()]
        return found_topics if found_topics else ["general"]
    
    def display_system_state(self):
        """Display current system state and memory."""
        print(f"\n{'='*80}")
        print("SYSTEM STATE & MEMORY")
        print(f"{'='*80}\n")
        
        memory = self.memory_agent.get_full_history()
        
        print("CONVERSATION HISTORY:")
        for turn in memory["conversation_history"][-3:]:
            print(f"  Turn {turn['turn_id']}: {turn['user_input'][:50]}...")
        
        print(f"\nKNOWLEDGE BASE ({len(memory['knowledge_base'])} facts):")
        for fact in memory['knowledge_base'][-3:]:
            print(f"  - {fact['content'][:50]}... (Topics: {', '.join(fact['topics'])})")
        
        print(f"\nAGENT ACTIVITIES:")
        for agent, states in memory['agent_states'].items():
            print(f"  {agent}: {len(states)} actions completed")
        
        print(f"{'='*80}\n")