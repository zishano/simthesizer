# Event Engine Architecture

## Purpose

The engine owns virtual time and execution order. It knows four node kinds
and nothing about requests, batches, or models.

## Graph

The event graph is a DAG of nodes and dependency edges. A node becomes ready
when every predecessor has completed, and starts at the later of its own
release time and its last predecessor's finish.

- A **compute node** holds one layer descriptor, one batch of per-request
  `(q_len, kv_len, lm_head_len)` states, and optional runner context. Its
  compute simulator returns its latency.
- A **network node** holds one point-to-point transfer `(src, dst, bytes)`.
  Its network simulator is polled until the transfer completes.
- A **logical node** holds an opaque payload for one owning handler. The
  handler returns a latency and may mutate the graph.
- A **resource node** holds concurrent service demands, exclusive stream slots,
  and memory reservations/releases. Physical resource IDs are shared across
  such nodes. The resource domain owns rates and capacity; the engine advances
  it at every ready event or service-component boundary.

Payloads are private to their owner. The engine never inspects them, so new
behavior needs no engine change.

## Execution

Ready nodes run in `(start time, node ID)` order. Node IDs increase with
creation, so ties resolve by creation order and runs are deterministic.
Release times and latencies must be finite and nonnegative; pending network
polls must advance representable time.

A completed node releases its children and is removed. The run ends when no
node is ready; any node left in the graph means a cycle or an orphan, and the
run fails.

A network poll answers either complete or pending with a delay. A pending node
requeues itself after that delay, which lets concurrent transfers change each
other's rates. Polls reach a simulator in nondecreasing time, so it may advance
its own clock lazily.

## Graph Mutation

Only a running logical handler mutates the graph, in two ways:

- add nodes beneath the running node;
- detach the running node's outgoing edges and reattach those children beneath
  newly added work.

Detach and reattach splice dynamic work into a pre-seeded DAG: a static
dependency survives, but its target now also waits for work that did not exist
when it was seeded.

Initially ready nodes are collected once, before execution. A node created at
run time with no path from the running node never runs and fails the run as a
leftover.

## Ownership and Safety

Nodes reference their handler or simulator by raw pointer. This is sound
because:

- every handler and simulator lives in the boxed config tree, which is neither
  moved nor restructured after initialization, or is the graph-owned boxed KV
  lifecycle handler, whose allocation is stable for the run;
- execution is single-threaded, so handler and simulator calls never overlap.

The graph retains the shared resource domain and KV handler for the whole run.
KV semantics remain in that handler. Resource nodes complete at their actual
finish time, unlike fixed-cost logical/compute nodes whose future finish can
be resolved at dispatch. Rate changes leave no stale completion predictions.

## Evolution Rules

Read-only observation callbacks run after each timestamp has drained, including
zero-duration descendants. The engine does not interpret the component state or
choose a storage format; see [observability](observability.md).

- Express new semantics as logical nodes owned by a component, not as new node
  kinds.
- Return a latency or add nodes; never advance time from a component.
- Parallel execution first requires an ownership model that replaces the raw
  pointers.
