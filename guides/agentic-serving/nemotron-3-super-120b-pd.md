# Agentic Code Generation — NVIDIA-Nemotron-3-Super-120B, P/D-disaggregated

This is one of the accelerator-specific deployments of the agentic code-generation workload; see the
[agentic-serving README](README.md#deployments) for the workload framing and the alternatives:
[NVIDIA-Nemotron-3-Ultra-550B on H200](nemotron-3-ultra-550b-h200.md),
[GLM-5.2-FP8 on H200](glm-5-2-h200.md),
[Qwen3-Coder-480B on TPU v7](qwen3-coder-480b-tpu.md).

## Overview

This guide deploys [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8),
**prefill/decode disaggregated** into 3 prefill and 1 decode replica, layering the same agentic
optimizations onto disaggregated serving as
[Nemotron-3-Ultra-550B on H200](nemotron-3-ultra-550b-h200.md), scaled down for this smaller model:

- **P/D disaggregation** so heavy prefill never stalls decode, stabilizing ITL.
- **Disagg-aware, prefix-cache routing** that scores both the on-device (GPU) and CPU-offload
  prefix caches when picking a prefill/decode endpoint.
- **KV cache offloading** to CPU DRAM — `100 GiB` per model server — to extend the cacheable
  working set beyond HBM for long, resumable sessions.
- **FP8 weights + FP8 KV cache**, served with Expert Parallelism.
- **`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`** to raise multi-turn prefix-cache hit rates, since
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` is a Mamba-hybrid model (`NemotronHForCausalLM`);
  vLLM rejects the flag on full-attention models, so it is set per-deployment rather than in the
  baseline manifests.

See [Nemotron-3-Super-120B, decode-only](nemotron-3-super-120b-decode-only.md) for the
non-disaggregated alternative on the same model.

## Default Configuration

| Parameter          | Value                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------- |
| Model              | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) |
| Serving topology   | P/D disaggregated — 3 prefill replicas, 1 decode replica                                      |
| TP size / EP size  | TP=4, EP enabled                                                                               |
| KV cache           | FP8-quantized, with `100 GiB` CPU offload per replica                                          |

### Supported Hardware Backends

| Backend           | Directory                                 | Notes                                              |
| ----------------- | ------------------------------------------ | --------------------------------------------------- |
| NVIDIA GPU (vLLM) | `modelserver/gpu/vllm/nemotron-3-super-pd/` | P/D disaggregated (3 prefill / 1 decode), TP=4     |

## Prerequisites

- Installed proper client tools (kubectl, helm).
- Set the following environment variables:

  ```bash
  export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
  source ${REPO_ROOT}/guides/env.sh
  export GUIDE_NAME="nemotron-3-super-pd"
  export NAMESPACE=<your namespace>
  ```

- An `llm-d-hf-token` secret with a valid `HF_TOKEN` key must already exist in the target
  namespace (see [Create the `llm-d-hf-token` secret](../../helpers/hf-token.md)) — not managed by
  this guide.

## Installation Instructions

### 1. Deploy the llm-d Router

This deployment uses the disaggregation-aware router values
([`router/nemotron-3-super-pd.values.yaml`](router/nemotron-3-super-pd.values.yaml)), which run
separate `prefill` and `decode` scheduling profiles:

```bash
helm install ${GUIDE_NAME} \
    ${ROUTER_STANDALONE_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/agentic-serving/router/nemotron-3-super-pd.values.yaml \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

### 2. Deploy the Model Server (GPUs)

Apply the Kustomize overlay for the P/D-disaggregated deployment:

```bash
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/agentic-serving/modelserver/gpu/vllm/nemotron-3-super-pd/
```

This deploys 3 prefill and 1 decode replica. Wait for them to become ready (model load is large;
the startup probe allows up to ~40 minutes):

```bash
kubectl rollout status deployment/nemotron-3-super-pd-prefill -n ${NAMESPACE} --timeout=45m
kubectl rollout status deployment/nemotron-3-super-pd-decode -n ${NAMESPACE} --timeout=45m
```

> [!NOTE]
> The Deployments' `progressDeadlineSeconds` (2400s) is shorter than `kubectl rollout status`'s own
> `--timeout`, and is what `rollout status` actually respects. If you see `exceeded its progress
> deadline` shortly after issuing the command, check `kubectl get pods -n ${NAMESPACE}` — the pods
> may still be healthy and legitimately still downloading/loading the model weights.

## Verification

### 1. Get the IP of the Proxy

```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```

### 2. Send Test Requests

Open a temporary interactive shell inside the cluster:

```bash
kubectl run curl-debug --rm -it \
    --image=cfmanteiga/alpine-bash-curl-jq \
    --env="IP=$IP" \
    --env="NAMESPACE=$NAMESPACE" \
    -- /bin/bash
```

Send a completion request:

```bash
curl -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "prompt": "Explain how a simple agent loop works in 3 sentences."
    }' | jq
```

## Cleanup

To clean up resources:

```bash
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/agentic-serving/modelserver/gpu/vllm/nemotron-3-super-pd/
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
```