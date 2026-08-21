# kopt

Kopt scaffolds optimization projects and supervises repeated OMP turns. Its classes
use composition rather than inheritance: a loop owns its iteration history, agent
session, and append-only recorder.

```mermaid
classDiagram
    direction LR

    class Loop
    class Iteration
    class Recorder
    class RpcClient {
        <<external>>
    }

    Loop "1" *-- "0..*" Iteration : history
    Loop --> Recorder : writes events
    Loop --> RpcClient : owns session
```
