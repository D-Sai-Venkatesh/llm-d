# Agentic Code Generation — NVIDIA-Nemotron-3-Super-120B, decode-only

This is one of the accelerator-specific deployments of the agentic code-generation workload; see the
[agentic-serving README](README.md#deployments) for the workload framing and the alternatives:
[NVIDIA-Nemotron-3-Ultra-550B on H200](nemotron-3-ultra-550b-h200.md),
[GLM-5.2-FP8 on H200](glm-5-2-h200.md),
[Qwen3-Coder-480B on TPU v7](qwen3-coder-480b-tpu.md).

## Overview

This guide deploys [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
as a single, non-disaggregated decode pool — no prefill/decode split, no KV-offloading. It's the
smallest-footprint deployment in this guide family, suited to benchmarking the routing/prefix-cache
baseline without the added complexity of P/D disaggregation. See
[Nemotron-3-Super-120B, P/D-disaggregated](nemotron-3-super-120b-pd.md) for the disaggregated
alternative on the same model.

## Default Configuration

| Parameter          | Value                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------- |
| Model              | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) |
| Serving topology   | Non-disaggregated — 3 decode replicas                                                         |
| TP size            | TP=4                                                                                           |
| KV cache           | FP8-quantized, on-device only (no CPU offload)                                                |

### Supported Hardware Backends

| Backend           | Directory                                            | Notes            |
| ----------------- | ----------------------------------------------------- | ---------------- |
| NVIDIA GPU (vLLM) | `modelserver/gpu/vllm/nemotron-3-super-decode-only/` | 3 decode replicas, TP=4 |

## Prerequisites

- Installed proper client tools (kubectl, helm).
- Set the following environment variables:

  ```bash
  export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
  source ${REPO_ROOT}/guides/env.sh
  export GUIDE_NAME="nemotron-3-super-decode-only"
  export NAMESPACE=<your namespace>
  ```

- An `llm-d-hf-token` secret with a valid `HF_TOKEN` key must already exist in the target
  namespace (see [Create the `llm-d-hf-token` secret](../../helpers/hf-token.md)) — not managed by
  this guide.

## Installation Instructions

### 1. Deploy the llm-d Router

This deployment uses the non-disaggregated router values
([`router/nemotron-3-super-decode-only.values.yaml`](router/nemotron-3-super-decode-only.values.yaml)),
which run a single `default` scheduling profile:

```bash
helm install ${GUIDE_NAME} \
    ${ROUTER_STANDALONE_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/agentic-serving/router/nemotron-3-super-decode-only.values.yaml \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

### 2. Deploy the Model Server (GPUs)

Apply the Kustomize overlay for the decode-only deployment:

```bash
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/agentic-serving/modelserver/gpu/vllm/nemotron-3-super-decode-only/
```

This deploys 3 decode replicas. Wait for them to become ready (model load is large; the startup
probe allows up to ~40 minutes):

```bash
kubectl rollout status deployment/nemotron-3-super-decode-only-decode -n ${NAMESPACE} --timeout=45m
```

> [!NOTE]
> The Deployment's `progressDeadlineSeconds` (2400s) is shorter than `kubectl rollout status`'s own
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
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/agentic-serving/modelserver/gpu/vllm/nemotron-3-super-decode-only/
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
```