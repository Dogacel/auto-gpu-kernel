# kbench

Kbench separates benchmark adapters (what to run) from execution backends (where to
run it). FlashInfer is the built-in adapter and arbitrary repositories use an adapter
backed by the agent-generated `validate.py` and `benchmark.py` scripts.

```mermaid
classDiagram
    direction LR

    class BenchmarkAdapter {
        <<abstract>>
        +run(candidates, request)
        +bench(request)
        +ab(baseline, request)
    }
    class FlashInferAdapter
    class GeneratedTaskAdapter
    class Job
    class ValidateScript {
        <<agent-generated>>
    }
    class BenchmarkScript {
        <<agent-generated>>
    }

    FlashInferAdapter --|> BenchmarkAdapter
    GeneratedTaskAdapter --|> BenchmarkAdapter
    FlashInferAdapter --> Job : sends through local / Modal / FAL
    GeneratedTaskAdapter --> ValidateScript
    GeneratedTaskAdapter --> BenchmarkScript
```
