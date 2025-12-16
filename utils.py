"""
Utilities Module
Provides logging, configuration, and helper functions.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class Logger:
    """Logging utility for system events."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.events = []
    
    def log(self, level: str, component: str, message: str, data: Dict = None):
        """Log an event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "component": component,
            "message": message,
            "data": data or {}
        }
        self.events.append(event)
        
        # Also write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def info(self, component: str, message: str, data: Dict = None):
        """Log info level."""
        self.log("INFO", component, message, data)
    
    def error(self, component: str, message: str, data: Dict = None):
        """Log error level."""
        self.log("ERROR", component, message, data)
    
    def debug(self, component: str, message: str, data: Dict = None):
        """Log debug level."""
        self.log("DEBUG", component, message, data)
    
    def get_logs(self, component: Optional[str] = None, 
                 level: Optional[str] = None) -> List[Dict]:
        """Retrieve logs with optional filtering."""
        results = self.events
        
        if component:
            results = [e for e in results if e["component"] == component]
        
        if level:
            results = [e for e in results if e["level"] == level]
        
        return results


class ConfigManager:
    """Configuration management for system parameters."""
    
    DEFAULT_CONFIG = {
        "system": {
            "max_agents": 10,
            "timeout_seconds": 30,
            "enable_logging": True,
            "enable_tracing": True
        },
        "memory": {
            "max_conversation_history": 1000,
            "max_knowledge_base_size": 10000,
            "vector_embedding_dim": 128,
            "enable_vector_search": True
        },
        "agents": {
            "research_agent": {
                "enabled": True,
                "timeout": 10,
                "retry_count": 2
            },
            "analysis_agent": {
                "enabled": True,
                "timeout": 10,
                "retry_count": 2
            },
            "aggregator_agent": {
                "enabled": True,
                "timeout": 10,
                "retry_count": 2
            }
        },
        "coordinator": {
            "enable_complexity_analysis": True,
            "enable_fallback_strategy": True,
            "enable_memory_check": True
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except:
                return self.DEFAULT_CONFIG.copy()
        else:
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key_path: str, default=None) -> Any:
        """Get configuration value by dot-notation path."""
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-notation path."""
        keys = key_path.split(".")
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration."""
        return self.config.copy()


class TaskTracer:
    """Traces task execution for debugging."""
    
    def __init__(self):
        self.traces = []
        self.current_task = None
    
    def start_task(self, task_id: str, task_name: str):
        """Start tracing a task."""
        self.current_task = {
            "task_id": task_id,
            "task_name": task_name,
            "start_time": datetime.now(),
            "steps": []
        }
    
    def add_step(self, step_name: str, status: str, details: Dict = None):
        """Add a step to current task trace."""
        if self.current_task:
            step = {
                "step_name": step_name,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details or {}
            }
            self.current_task["steps"].append(step)
    
    def end_task(self, status: str = "completed"):
        """End current task tracing."""
        if self.current_task:
            self.current_task["end_time"] = datetime.now()
            self.current_task["status"] = status
            self.current_task["duration_seconds"] = (
                self.current_task["end_time"] - self.current_task["start_time"]
            ).total_seconds()
            self.traces.append(self.current_task)
            self.current_task = None
    
    def get_traces(self) -> List[Dict]:
        """Get all task traces."""
        return self.traces
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        
        total_tasks = len(self.traces)
        completed = sum(1 for t in self.traces if t["status"] == "completed")
        total_duration = sum(t.get("duration_seconds", 0) for t in self.traces)
        
        print(f"Total Tasks: {total_tasks}")
        print(f"Completed: {completed}")
        print(f"Failed: {total_tasks - completed}")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if self.traces:
            print("\nDetailed Breakdown:")
            for trace in self.traces:
                print(f"\n  Task: {trace['task_name']} (ID: {trace['task_id']})")
                print(f"    Status: {trace['status']}")
                print(f"    Duration: {trace.get('duration_seconds', 0):.2f}s")
                print(f"    Steps: {len(trace['steps'])}")
        
        print("\n" + "="*80 + "\n")


class ValidationHelper:
    """Helpers for validating agent outputs."""
    
    @staticmethod
    def validate_research_output(output: Dict) -> bool:
        """Validate research agent output."""
        required_keys = ["success", "findings", "query", "result_count"]
        return all(key in output for key in required_keys)
    
    @staticmethod
    def validate_analysis_output(output: Dict) -> bool:
        """Validate analysis agent output."""
        required_keys = ["type"]
        return all(key in output for key in required_keys)
    
    @staticmethod
    def validate_aggregation_output(output: Dict) -> bool:
        """Validate aggregation agent output."""
        required_keys = ["type"]
        return all(key in output for key in required_keys)
    
    @staticmethod
    def calculate_confidence(outputs: List[Dict]) -> float:
        """Calculate overall confidence from multiple outputs."""
        if not outputs:
            return 0.0
        
        confidences = [
            o.get("confidence", 0.5) for o in outputs 
            if isinstance(o, dict)
        ]
        
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.5


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "queries_processed": 0,
            "total_agent_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_response_time": 0.0,
            "memory_events": 0
        }
        self.response_times = []
    
    def record_query(self):
        """Record a processed query."""
        self.metrics["queries_processed"] += 1
    
    def record_agent_call(self, success: bool, response_time: float = 0.0):
        """Record an agent call."""
        self.metrics["total_agent_calls"] += 1
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
        
        if response_time > 0:
            self.response_times.append(response_time)
            self.metrics["avg_response_time"] = sum(self.response_times) / len(self.response_times)
    
    def record_memory_event(self):
        """Record a memory operation."""
        self.metrics["memory_events"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def print_report(self):
        """Print performance report."""
        print("\n" + "="*80)
        print("PERFORMANCE REPORT")
        print("="*80)
        print(f"Queries Processed: {self.metrics['queries_processed']}")
        print(f"Total Agent Calls: {self.metrics['total_agent_calls']}")
        print(f"Successful: {self.metrics['successful_calls']}")
        print(f"Failed: {self.metrics['failed_calls']}")
        if self.metrics['total_agent_calls'] > 0:
            success_rate = (self.metrics['successful_calls'] / self.metrics['total_agent_calls']) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        print(f"Avg Response Time: {self.metrics['avg_response_time']:.3f}s")
        print(f"Memory Events: {self.metrics['memory_events']}")
        print("="*80 + "\n")