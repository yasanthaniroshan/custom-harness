"""The SDLC pipeline on LangChain / LangGraph.

    stages.py         what each stage sees and is asked to do
    artifacts.py      typed artifacts (Pydantic), their rendering and validation
    graph.py          the SDLC graph with the test-suite gate and best-of-N
    implementer.py    the tool-using implement subgraph
    runner.py         concurrent, resumable batch execution -> runs JSONL
    service.py        generate / evaluate / run one task (what the CLI calls)
"""
