# Multi-Agent Coordination System

A sophisticated Python-based multi-agent system demonstrating distributed task orchestration, intelligent memory management, and agent collaboration.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  COORDINATOR (Manager)                   │
│  • Task decomposition                                    │
│  • Agent orchestration                                   │
│  • Result synthesis                                      │
└─────────────────────────────────────────────────────────┘
           ↓              ↓              ↓
    ┌──────────────┬──────────────┬──────────────┐
    │   Research   │   Analysis   │  Aggregator  │
    │    Agent     │    Agent     │    Agent     │
    └──────────────┴──────────────┴──────────────┘
           ↓              ↓              ↓
    └─────────────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────────────┐
    │         MEMORY LAYER                        │
    │  • Conversation History                     │
    │  • Knowledge Base (with vector search)      │
    │  • Agent State Tracking                     │
    └─────────────────────────────────────────────┘
```

## Components

### 1. **Coordinator Agent (coordinator.py)**
- **Responsibility**: Orchestrates all system operations
- **Key Features**:
  - Analyzes query complexity (SIMPLE, MODERATE, COMPLEX)
  - Decomposes tasks into subtasks
  - Routes subtasks to appropriate agents
  - Manages dependencies between tasks
  - Implements fallback strategies on failures
  - Synthesizes final responses
  - Stores important findings in memory

### 2. **Worker Agents (worker_agents.py)**

#### Research Agent
- Simulates information retrieval
- Searches pre-loaded knowledge base (mock)
- Supports multiple domains:
  - Machine Learning Optimization
  - Deep Learning Frameworks
  - Neural Network Architectures
- Returns structured findings with confidence scores

#### Analysis Agent
- Performs comparative analysis
- Ranking and scoring
- Data summarization
- Processes data from research phase
- Provides detailed comparisons with pros/cons

#### Data Aggregator Agent
- Synthesizes information from multiple sources
- Merges findings
- Creates integrated views
- Combines research and analysis results

### 3. **Memory Layer (memory_layer.py)**

#### ConversationMemory
- Stores full conversation history
- Maintains timestamps and turn counts
- Provides context window retrieval
- Supports historical queries

#### KnowledgeBase
- Persistent storage of discovered facts
- Metadata: timestamp, source agent, confidence, topics
- Vector similarity search (cosine similarity)
- Topic-based search
- Prevents redundant work by checking prior knowledge

#### AgentStateMemory
- Tracks what each agent accomplished
- Records actions with inputs/outputs
- Maintains per-agent execution history
- Supports performance analysis

#### MemoryAgent (Coordinator)
- Unified interface to all memory subsystems
- Handles storage and retrieval
- Checks for prior knowledge before new research
- Enables adaptive decision-making

### 4. **Utilities (utils.py)**

#### Logger
- Event logging with levels (INFO, DEBUG, ERROR)
- File persistence with timestamps
- Component-based filtering

#### ConfigManager
- System configuration management
- Agent-specific settings
- Timeout and retry configurations
- Persistent configuration file support

#### TaskTracer
- Execution tracing for debugging
- Step-by-step task monitoring
- Duration tracking
- Summary reports

#### ValidationHelper
- Output validation for each agent type
- Confidence calculation
- Result verification

#### PerformanceMonitor
- Query and agent call tracking
- Success rate calculation
- Response time monitoring
- Metrics reporting

## Setup Instructions

### Prerequisites
```bash
Python 3.8+
NumPy (for vector operations)
```

### Installation

1. **Create project directory**:
```bash
mkdir multi-agent-system
cd multi-agent-system
```

2. **Create Python virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install numpy
```

4. **Create module files** in your project directory:
   - `memory_layer.py` - Memory management system
   - `worker_agents.py` - Research, Analysis, Aggregator agents
   - `coordinator.py` - Coordinator orchestrator
   - `utils.py` - Utility functions and helpers
   - `main.py` - Main entry point

5. **Verify structure**:
```
multi-agent-system/
├── memory_layer.py
├── worker_agents.py
├── coordinator.py
├── utils.py
├── main.py
├── config.json (auto-generated)
└── logs/ (auto-generated)
```

## Usage

### Interactive Mode
```bash
python main.py
```

Then enter your queries:
```
You: Find information about machine learning optimization techniques
You: Compare gradient descent and Adam optimizer
You: What did we learn about optimizers earlier?
```

### Demo Mode
```bash
python main.py --demo
```

Runs predefined queries with step-by-step walkthrough.

### Available Commands
- Type a question to query the system
- `memory` - Display system memory and knowledge base
- `trace` - Show last execution trace
- `help` - Display example queries
- `exit` - Quit the system

## Example Interactions

### Example 1: Simple Query
```
Query: "What are machine learning optimization algorithms?"

[COORDINATOR] Task complexity: SIMPLE
[COORDINATOR] Decomposed into 1 subtasks
[COORDINATOR] Executing subtask 0 (research)
[ResearchAgent] Found 5 optimization techniques

Response: Information retrieved on gradient descent, Adam, SGD, RMSprop, Momentum
```

### Example 2: Moderate Query with Analysis
```
Query: "Compare gradient descent and Adam optimizer"

[COORDINATOR] Task complexity: MODERATE
[COORDINATOR] Decomposed into 2 subtasks
[COORDINATOR] Executing subtask 0 (research)
[COORDINATOR] Executing subtask 1 (analysis)
[AnalysisAgent] Performed comparison analysis

Response: Detailed comparison with pros/cons for each optimizer
```

### Example 3: Complex Query with Full Pipeline
```
Query: "Find and analyze all optimization techniques"

[COORDINATOR] Task complexity: COMPLEX
[COORDINATOR] Decomposed into 3 subtasks
[COORDINATOR] Executing subtask 0 (research)
[COORDINATOR] Executing subtask 1 (analysis)
[COORDINATOR] Executing subtask 2 (aggregation)

Response: Comprehensive synthesis with integrated insights
```

### Example 4: Memory-Aware Query
```
Query: "What did we learn about optimizers earlier?"

[COORDINATOR] Prior knowledge check: FOUND
[COORDINATOR] Retrieved 3 prior findings
Response: From memory...
```

## Task Decomposition Examples

### Complexity Analysis
The system analyzes query keywords to determine complexity:

**SIMPLE** tasks:
- Keywords: "what is", "define", "list", "who is"
- Execution: Single research task
- Result: Direct answer from knowledge base

**MODERATE** tasks:
- Keywords: "compare", "analyze", "how", "why", "explain"
- Execution: Research → Analysis
- Result: Processed insights with reasoning

**COMPLEX** tasks:
- Keywords: "optimize", "combine", "integrate", "comprehensive"
- Execution: Research → Analysis → Aggregation
- Result: Synthesized comprehensive answer with integration

## Memory System Features

### Vector Similarity Search
- Deterministic embeddings based on text hash
- Cosine similarity computation
- Top-K retrieval (default K=5)
- Prevents redundant research

### Knowledge Persistence
Each fact stored includes:
```json
{
  "fact_id": "fact_0",
  "content": "Adam is a fast optimizer...",
  "topics": ["optimization", "learning"],
  "source_agent": "ResearchAgent",
  "confidence": 0.95,
  "timestamp": "2024-01-15T10:30:00",
  "unix_time": 1705318200.123
}
```

### Agent State Tracking
Records what each agent did:
- Action performed
- Input data processed
- Output generated
- Confidence level
- Execution timestamp

## Fallback & Error Handling

### Strategies Implemented
1. **Dependency Failure**: Skip dependent tasks gracefully
2. **Agent Failure**: Attempt degraded analysis (e.g., summary vs comparison)
3. **Missing Data**: Use cached knowledge when available
4. **Timeout**: Return partial results with confidence adjustment

## Performance Monitoring

### Metrics Tracked
- Queries processed
- Agent call success rate
- Average response time
- Memory operations count
- Agent-specific performance

### Generate Report
Use `PerformanceMonitor` class to track and display metrics.

## Configuration

### Default Configuration (config.json)
```json
{
  "system": {
    "max_agents": 10,
    "timeout_seconds": 30,
    "enable_logging": true
  },
  "memory": {
    "max_conversation_history": 1000,
    "vector_embedding_dim": 128
  },
  "agents": {
    "research_agent": {
      "enabled": true,
      "timeout": 10,
      "retry_count": 2
    }
  }
}
```

Customize by editing `config.json` before running.

## Logging & Tracing

### Log Files
Generated in `logs/` directory with format: `system_YYYYMMDD_HHMMSS.log`

### Trace Information Includes
- Agent name and task ID
- Input and output data
- Execution duration
- Success/failure status
- Dependency information

## Code Structure and Design Patterns

### Class Hierarchy
```
WorkerAgent (ABC)
├── ResearchAgent
├── AnalysisAgent
└── DataAggregatorAgent

MemoryAgent
├── ConversationMemory
├── KnowledgeBase
├── AgentStateMemory
└── VectorStore

Coordinator
└── [manages all agents and memory]

MultiAgentSystem
└── [main controller with CLI]
```

### Key Design Principles
1. **Separation of Concerns**: Each agent has clear responsibility
2. **Memory-Driven**: Prior knowledge influences decisions
3. **Traceable Execution**: Full visibility into agent actions
4. **Fault Tolerant**: Graceful degradation on failures
5. **Extensible**: Easy to add new agents or memory types

## Advanced Features

### Vector Search
- Deterministic embeddings using text hash
- L2 normalization for stability
- Configurable embedding dimension
- Efficient similarity computation

### Adaptive Behavior
- Checks prior memory before research
- Skips redundant work based on similarity
- Adjusts confidence based on information sources
- Learns from previous interactions

### Multi-Stage Pipelines
- Task dependencies tracked and managed
- Automatic sequencing respecting order
- Parallel-ready architecture (single-threaded in current implementation)
- Clear error boundaries

## Testing Example Queries

```bash
python main.py

# Query 1: Simple
You: What is gradient descent?

# Query 2: Moderate  
You: Compare Adam and SGD optimizers

# Query 3: Complex
You: Find and analyze all optimization techniques, which are most effective?

# Query 4: Memory-aware
You: What did we learn about Adam?

# View system state
You: memory

# View execution details
You: trace

# Help
You: help

# Exit
You: exit
```

## Extending the System

### Adding a New Worker Agent
1. Subclass `WorkerAgent` in `worker_agents.py`
2. Implement `execute()` method
3. Call `self.log_execution()` for tracing
4. Register in `Coordinator.__init__()`
5. Add routing logic in `_execute_subtasks()`

### Adding New Memory Types
1. Create new class in `memory_layer.py`
2. Implement storage interface
3. Add search methods
4. Integrate with `MemoryAgent`

### Adding New Analysis Types
Extend `AnalysisAgent._perform_comparison()` with custom logic.

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'numpy'"
- **Solution**: `pip install numpy`

**Issue**: Queries returning empty results
- **Solution**: Check knowledge base in Research Agent's `KNOWLEDGE_BASE` dictionary

**Issue**: Memory not persisting between runs
- **Solution**: Memory is in-memory by design. For persistence, extend with database backend.

**Issue**: Agent timeouts
- **Solution**: Increase timeout in `config.json` or optimize agent logic

## Performance Characteristics

- **Memory Growth**: O(n) where n = number of facts stored
- **Search Time**: O(n) with vector similarity (linear scan)
- **Agent Call Overhead**: ~10-50ms per call
- **Conversation Limit**: Configured via `max_conversation_history`

For large-scale deployments, consider:
- FAISS for faster vector search
- Database backend for persistence
- Caching layer for frequent queries
- Distributed agent execution

## Future Enhancements

- [ ] Persistent database backend (SQLite, PostgreSQL)
- [ ] FAISS integration for faster vector search
- [ ] Multi-threaded agent execution
- [ ] Natural language understanding improvements
- [ ] Confidence-based filtering
- [ ] Learning from user feedback
- [ ] Agent-to-agent communication protocols
- [ ] Dynamic agent spawning
- [ ] REST API interface
- [ ] Web UI dashboard

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Enable debug logging in `config.json`
3. Review execution traces with `trace` command
4. Check example queries in `help` command