"""
Test Suite for Multi-Agent System
Validates all components and demonstrates functionality.
"""

import sys
import json
from coordinator import Coordinator
from memory_layer import MemoryAgent, VectorStore
from worker_agents import ResearchAgent, AnalysisAgent, DataAggregatorAgent
from utils import Logger, ConfigManager, ValidationHelper, PerformanceMonitor

class SystemValidator:
    """Validates system components and functionality."""
    
    def __init__(self):
        self.coordinator = Coordinator()
        self.logger = Logger()
        self.validator = ValidationHelper()
        self.performance = PerformanceMonitor()
        self.test_results = []
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "="*80)
        print("MULTI-AGENT SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*80 + "\n")
        
        tests = [
            ("Memory Layer Tests", self.test_memory_layer),
            ("Vector Store Tests", self.test_vector_store),
            ("Research Agent Tests", self.test_research_agent),
            ("Analysis Agent Tests", self.test_analysis_agent),
            ("Aggregator Agent Tests", self.test_aggregator_agent),
            ("Coordinator Tests", self.test_coordinator),
            ("Task Decomposition Tests", self.test_task_decomposition),
            ("Memory-Aware Workflow Tests", self.test_memory_aware_workflow),
            ("Error Handling Tests", self.test_error_handling),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'─'*80}")
            print(f"Running: {test_name}")
            print(f"{'─'*80}")
            try:
                test_func()
                self.test_results.append((test_name, "✓ PASSED"))
                print(f"✓ {test_name} PASSED")
            except AssertionError as e:
                self.test_results.append((test_name, f"✗ FAILED: {str(e)}"))
                print(f"✗ {test_name} FAILED: {str(e)}")
            except Exception as e:
                self.test_results.append((test_name, f"✗ ERROR: {str(e)}"))
                print(f"✗ {test_name} ERROR: {str(e)}")
        
        self.print_summary()
    
    def test_memory_layer(self):
        """Test memory layer components."""
        memory_agent = MemoryAgent()
        
        # Test storing finding
        fact_id = memory_agent.store_finding(
            content="Adam optimizer converges faster",
            topics=["optimization", "adam"],
            source_agent="TestAgent",
            confidence=0.95
        )
        assert fact_id is not None, "Fact ID should not be None"
        assert "fact_" in fact_id, "Fact ID should contain 'fact_'"
        
        # Test knowledge search
        results = memory_agent.search_knowledge("adam", search_type="similarity")
        assert len(results) > 0, "Should find stored fact"
        
        # Test topic search
        topic_results = memory_agent.search_knowledge("adam", search_type="topic")
        assert len(topic_results) > 0, "Should find by topic"
        
        # Test conversation memory
        memory_agent.add_conversation_turn(
            user_input="Test query",
            response="Test response",
            agent_trace=[]
        )
        context = memory_agent.get_recent_context(num_turns=1)
        assert len(context) == 1, "Should have one conversation turn"
        assert context[0]["user_input"] == "Test query"
        
        # Test agent state memory
        memory_agent.record_agent_action(
            agent_name="TestAgent",
            action="test_action",
            input_data={"test": "input"},
            output_data={"test": "output"},
            confidence=0.9
        )
        agent_history = memory_agent.get_full_history()
        assert "agent_states" in agent_history, "Should have agent states"
        
        print("  ✓ Conversation memory working")
        print("  ✓ Knowledge base storage working")
        print("  ✓ Agent state tracking working")
    
    def test_vector_store(self):
        """Test vector store and similarity search."""
        vector_store = VectorStore(embedding_dim=128)
        
        # Add vectors
        vector_store.add("doc1", "gradient descent optimization", {"type": "algorithm"})
        vector_store.add("doc2", "adam optimizer adaptive learning", {"type": "algorithm"})
        vector_store.add("doc3", "neural network architecture", {"type": "architecture"})
        
        # Test exact match
        results = vector_store.search("gradient descent")
        assert len(results) > 0, "Should find gradient descent"
        assert results[0][0] == "doc1", "Should rank correct document first"
        
        # Test semantic similarity
        results = vector_store.search("optimization algorithm")
        assert len(results) > 0, "Should find optimization results"
        
        # Test top_k
        results = vector_store.search("optimization", top_k=2)
        assert len(results) <= 2, "Should respect top_k limit"
        
        # Verify normalization
        for key, vector in vector_store.vectors.items():
            norm = (vector ** 2).sum() ** 0.5
            assert abs(norm - 1.0) < 0.01, f"Vector {key} not normalized"
        
        print("  ✓ Vector storage working")
        print("  ✓ Similarity search working")
        print("  ✓ Vector normalization correct")
    
    def test_research_agent(self):
        """Test research agent functionality."""
        research_agent = ResearchAgent(name="TestResearch")
        
        # Test basic search
        task = {
            "query": "gradient descent",
            "search_area": "machine learning optimization"
        }
        result = research_agent.execute(task)
        
        assert result["success"], "Should have successful search"
        assert result["result_count"] > 0, "Should find results"
        assert len(result["findings"]) > 0, "Should have findings"
        
        # Test all query
        task = {
            "query": "all",
            "search_area": "machine learning optimization"
        }
        result = research_agent.execute(task)
        assert result["result_count"] >= 5, "Should find all optimizers"
        
        # Test execution trace
        assert len(research_agent.execution_trace) > 0, "Should have execution trace"
        
        print("  ✓ Research agent search working")
        print("  ✓ Knowledge base retrieval working")
        print("  ✓ Execution tracing working")
    
    def test_analysis_agent(self):
        """Test analysis agent functionality."""
        analysis_agent = AnalysisAgent(name="TestAnalysis")
        
        # Create test data
        test_data = [
            {
                "topic": "gradient descent",
                "content": {
                    "pros": ["Simple", "Widely used"],
                    "cons": ["Slow", "Local minima"],
                    "use_cases": ["Linear regression"]
                }
            },
            {
                "topic": "adam",
                "content": {
                    "pros": ["Fast", "Adaptive"],
                    "cons": ["Complex", "Memory intensive"],
                    "use_cases": ["Deep learning"]
                }
            }
        ]
        
        # Test comparison
        task = {
            "analysis_type": "comparison",
            "data": test_data,
            "criteria": []
        }
        result = analysis_agent.execute(task)
        
        assert result["type"] == "detailed_comparison", "Should return comparison type"
        assert len(result["items"]) == 2, "Should analyze both items"
        
        # Test ranking
        task = {
            "analysis_type": "ranking",
            "data": test_data,
            "criteria": []
        }
        result = analysis_agent.execute(task)
        
        assert result["type"] == "ranking", "Should return ranking type"
        assert len(result["ranked_items"]) == 2, "Should rank both items"
        assert result["top_choice"] is not None, "Should have top choice"
        
        # Test summary
        task = {
            "analysis_type": "summary",
            "data": test_data,
            "criteria": []
        }
        result = analysis_agent.execute(task)
        
        assert result["type"] == "summary", "Should return summary type"
        assert result["total_items"] == 2, "Should summarize both items"
        
        print("  ✓ Comparison analysis working")
        print("  ✓ Ranking analysis working")
        print("  ✓ Summary analysis working")
    
    def test_aggregator_agent(self):
        """Test aggregator agent functionality."""
        aggregator_agent = DataAggregatorAgent(name="TestAggregator")
        
        # Test merge
        data_sources = [
            {"findings": [{"topic": "adam"}]},
            {"findings": [{"topic": "sgd"}]}
        ]
        
        task = {
            "aggregation_type": "merge",
            "data_sources": data_sources
        }
        result = aggregator_agent.execute(task)
        
        assert result["type"] == "merged_data", "Should return merged type"
        assert result["source_count"] == 2, "Should merge both sources"
        
        # Test synthesize
        task = {
            "aggregation_type": "synthesize",
            "data_sources": data_sources
        }
        result = aggregator_agent.execute(task)
        
        assert result["type"] == "synthesized_analysis", "Should return synthesis type"
        assert "integrated_view" in result, "Should have integrated view"
        
        print("  ✓ Data merge working")
        print("  ✓ Data synthesis working")
    
    def test_coordinator(self):
        """Test coordinator functionality."""
        # Test complexity analysis
        simple_query = "What is gradient descent?"
        complexity = self.coordinator._analyze_complexity(simple_query)
        assert str(complexity) == "TaskComplexity.SIMPLE", "Should detect simple query"
        
        moderate_query = "Compare gradient descent and Adam"
        complexity = self.coordinator._analyze_complexity(moderate_query)
        assert str(complexity) == "TaskComplexity.MODERATE", "Should detect moderate query"
        
        complex_query = "Find all optimizers and compare effectiveness"
        complexity = self.coordinator._analyze_complexity(complex_query)
        assert str(complexity) == "TaskComplexity.COMPLEX", "Should detect complex query"
        
        # Test search area inference
        area = self.coordinator._infer_search_area("gradient descent optimizer")
        assert area == "machine learning optimization", "Should infer correct area"
        
        # Test analysis type inference
        analysis_type = self.coordinator._infer_analysis_type("Compare Adam and SGD")
        assert analysis_type == "comparison", "Should infer comparison"
        
        print("  ✓ Query complexity analysis working")
        print("  ✓ Search area inference working")
        print("  ✓ Analysis type inference working")
    
    def test_task_decomposition(self):
        """Test task decomposition logic."""
        simple_query = "What is gradient descent?"
        subtasks = self.coordinator._decompose_task(
            simple_query,
            self.coordinator._analyze_complexity(simple_query)
        )
        assert len(subtasks) == 1, "Simple should have 1 subtask"
        assert subtasks[0]["agent"] == "research", "Should use research agent"
        
        moderate_query = "Compare algorithms"
        subtasks = self.coordinator._decompose_task(
            moderate_query,
            self.coordinator._analyze_complexity(moderate_query)
        )
        assert len(subtasks) == 2, "Moderate should have 2 subtasks"
        assert subtasks[0]["agent"] == "research", "First should be research"
        assert subtasks[1]["agent"] == "analysis", "Second should be analysis"
        
        complex_query = "Find and analyze all optimizers"
        subtasks = self.coordinator._decompose_task(
            complex_query,
            self.coordinator._analyze_complexity(complex_query)
        )
        assert len(subtasks) == 3, "Complex should have 3 subtasks"
        assert subtasks[2]["agent"] == "aggregator", "Third should be aggregator"
        
        print("  ✓ Simple task decomposition working")
        print("  ✓ Moderate task decomposition working")
        print("  ✓ Complex task decomposition working")
    
    def test_memory_aware_workflow(self):
        """Test memory-aware task execution."""
        query1 = "What is gradient descent?"
        response1 = self.coordinator.process_query(query1)
        
        assert response1["answer"] != "", "Should have answer"
        assert response1["confidence"] > 0, "Should have confidence"
        
        # Now check if memory was updated
        has_prior = self.coordinator.memory_agent.has_prior_knowledge(query1)
        assert has_prior, "Should have prior knowledge after query"
        
        # Similar query should use memory
        query2 = "Tell me about gradient descent again"
        response2 = self.coordinator.process_query(query2)
        
        assert response2.get("from_memory"), "Should use memory for similar query"
        
        print("  ✓ Memory storage working")
        print("  ✓ Memory retrieval working")
        print("  ✓ Memory-aware execution working")
    
    def test_error_handling(self):
        """Test error handling and fallback strategies."""
        # Test with invalid agent
        coordinator = Coordinator()
        
        # Task with missing dependencies
        subtasks = [
            {
                "task_id": 0,
                "agent": "research",
                "priority": 1
            },
            {
                "task_id": 1,
                "agent": "analysis",
                "depends_on": 999,  # Non-existent dependency
                "priority": 2
            }
        ]
        
        results = coordinator._execute_subtasks(subtasks)
        
        # First task should complete
        assert 0 in results, "Research task should execute"
        
        # Second should be skipped due to missing dependency
        assert 1 not in results or "error" in str(results.get(1, {})), "Should handle missing dependency"
        
        print("  ✓ Dependency validation working")
        print("  ✓ Error handling working")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80 + "\n")
        
        passed = sum(1 for _, result in self.test_results if "PASSED" in result)
        failed = len(self.test_results) - passed
        
        for test_name, result in self.test_results:
            status = "✓" if "PASSED" in result else "✗"
            print(f"{status} {test_name}: {result}")
        
        print(f"\n{'─'*80}")
        print(f"Total: {len(self.test_results)} | Passed: {passed} | Failed: {failed}")
        print(f"Success Rate: {(passed/len(self.test_results)*100):.1f}%")
        print("="*80 + "\n")
        
        return failed == 0


def main():
    """Run test suite."""
    validator = SystemValidator()
    
    success = validator.run_all_tests()
    
    if success:
        print("\n✓ ALL TESTS PASSED!\n")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    main()