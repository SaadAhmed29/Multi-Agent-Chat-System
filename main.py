"""
Main Entry Point - Multi-Agent Coordination System
Runs the interactive console interface.
"""

import json
from coordinator import Coordinator
from utils import Logger, ConfigManager

class MultiAgentSystem:
    """Main system controller."""
    
    def __init__(self):
        self.coordinator = Coordinator()
        self.logger = Logger()
        self.config = ConfigManager()
        self.running = True
    
    def display_welcome(self):
        """Display welcome message."""
        print("\n" + "="*80)
        print("MULTI-AGENT COORDINATION SYSTEM")
        print("="*80)
        print("\nCapabilities:")
        print("  • Query Processing with Natural Language Understanding")
        print("  • Multi-Agent Task Decomposition & Execution")
        print("  • Structured Memory with Vector Search")
        print("  • Agent State Tracking & Tracing")
        print("  • Intelligent Fallback & Error Handling")
        print("\nCommands:")
        print("  - Type your question to query the system")
        print("  - 'memory' to display system memory")
        print("  - 'help' to show example queries")
        print("  - 'trace' to show last execution trace")
        print("  - 'exit' to quit")
        print("="*80 + "\n")
    
    def display_help(self):
        """Display help and example queries."""
        examples = [
            "Find information about machine learning optimization techniques",
            "Compare gradient descent and Adam optimizer",
            "What are the best deep learning frameworks?",
            "Analyze the effectiveness of different optimization algorithms",
            "Find information about neural network architectures",
            "What did we learn about optimizers earlier?"
        ]
        
        print("\n" + "="*80)
        print("EXAMPLE QUERIES")
        print("="*80)
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("="*80 + "\n")
    
    def display_trace(self):
        """Display last execution trace."""
        if not self.coordinator.execution_trace:
            print("\nNo recent execution trace.\n")
            return
        
        print("\n" + "="*80)
        print("EXECUTION TRACE")
        print("="*80 + "\n")
        
        for i, trace_entry in enumerate(self.coordinator.execution_trace, 1):
            print(f"Step {i}: {trace_entry['agent']}")
            print(f"  Task ID: {trace_entry.get('task_id', 'N/A')}")
            print(f"  Input: {json.dumps(trace_entry['input'], indent=2)[:100]}...")
            print(f"  Output Success: {'success' in trace_entry['output'] and trace_entry['output']['success']}")
            print()
        
        print("="*80 + "\n")
    
    def process_user_input(self, user_input: str) -> bool:
        """Process user input and dispatch to appropriate handler."""
        user_input = user_input.strip()
        
        if not user_input:
            return True
        
        if user_input.lower() == "exit":
            print("\nShutting down multi-agent system...\n")
            return False
        
        elif user_input.lower() == "help":
            self.display_help()
            return True
        
        elif user_input.lower() == "memory":
            self.coordinator.display_system_state()
            return True
        
        elif user_input.lower() == "trace":
            self.display_trace()
            return True
        
        else:
            # Process as query
            response = self.coordinator.process_query(user_input)
            self._display_response(response)
            return True
    
    def _display_response(self, response: dict):
        """Display formatted response."""
        print("="*80)
        print("COORDINATOR RESPONSE")
        print("="*80)
        print(f"\nAnswer:\n{response.get('answer', 'No answer generated')}")
        print(f"\nConfidence: {response.get('confidence', 0.0):.2%}")
        print(f"Agent Contributions: {', '.join(response.get('agent_contributions', []))}")
        
        if response.get("from_memory"):
            print("\n[Note: Response generated from memory]")
        
        print("\n" + "="*80 + "\n")
    
    def run_interactive(self):
        """Run the system in interactive mode."""
        self.display_welcome()
        
        while self.running:
            try:
                user_input = input("You: ").strip()
                if not self.process_user_input(user_input):
                    self.running = False
            
            except KeyboardInterrupt:
                print("\n\nSystem interrupted. Type 'exit' to quit.\n")
            except Exception as e:
                print(f"\nError processing input: {str(e)}\n")
    
    def run_demo(self):
        """Run a demonstration with predefined queries."""
        self.display_welcome()
        
        demo_queries = [
            "Find information about machine learning optimization techniques",
            "Compare gradient descent and Adam optimizer",
            "What are the best deep learning frameworks?",
            "What did we learn about optimizers earlier?"
        ]
        
        print("[DEMO MODE] Running predefined queries...\n")
        
        for query in demo_queries:
            print(f"\n{'*'*80}")
            print(f"DEMO QUERY: {query}")
            print(f"{'*'*80}\n")
            
            response = self.coordinator.process_query(query)
            self._display_response(response)
            
            input("Press Enter to continue...")
        
        print("\n[DEMO COMPLETE]")
        self.coordinator.display_system_state()


def main():
    """Main entry point."""
    import sys
    
    system = MultiAgentSystem()
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        system.run_demo()
    else:
        system.run_interactive()


if __name__ == "__main__":
    main()