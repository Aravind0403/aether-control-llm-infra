# GCP Cloud Micro-Run Guide (Proof-of-Concept for < $5–$10)

This guide details how to execute a cost-optimized **Google Cloud Engine (GKE)** deployment of **AetherControl** for under **$5–$10 total cloud cost**, generating real cloud proof logs, DCGM hardware metrics, and Grafana screenshots.

---

## 🗺️ Step-by-Step Execution Plan

### Step 1: GCP Quota Verification
Before provisioning hardware, verify your GPU quota in GCP Console:
1. Go to **GCP Console $\rightarrow$ IAM & Admin $\rightarrow$ Quotas**.
2. Search for **NVIDIA L4 GPUs** (or NVIDIA T4 GPUs).
3. Confirm a regional quota of at least **1 GPU** (e.g., region `us-central1`).

---

### Step 2: Cost-Optimized Terraform Provisioning (`make cloud-up`)
We use **Preemptible Spot Instances** (`g2-standard-4` with 1x NVIDIA L4 GPU) to reduce compute costs by ~70%:

```bash
# Provision GKE Spot GPU Cluster, GCS Bucket, and Artifact Registry
make cloud-up
```

#### Provisioned Infrastructure:
* **GKE Cluster:** `aethercontrol-gke` (1 Node, Spot NVIDIA L4 24GB VRAM)
* **GCS Storage Bucket:** `gs://aethercontrol-dev-aethercontrol-datasets/`
* **Artifact Registry:** `us-central1-docker.pkg.dev/aethercontrol-dev/aethercontrol-repo`

---

### Step 3: Artifact Upload & Container Deployment

```bash
# 1. Fetch Cluster Credentials
gcloud container clusters get-credentials aethercontrol-gke --region us-central1

# 2. Upload GSM8K Dataset to GCS
gsutil cp trainsight/sample_data/sample_gsm8k.jsonl gs://aethercontrol-dev-aethercontrol-datasets/gsm8k_subset.jsonl

# 3. Apply Kubernetes Secrets & Manifests
kubectl apply -f k8s-infra/manifests/secrets.yaml
kubectl apply -f k8s-infra/manifests/vllm-deployment.yaml
kubectl apply -f k8s-infra/manifests/dcgm-exporter.yaml
kubectl apply -f k8s-infra/manifests/keda-autoscaler.yaml
kubectl apply -f k8s-infra/manifests/prometheus-grafana.yaml
```

---

### Step 4: Verification & Portfolio Proof Collection

1. **TrainSight InitContainer Guard:**
   * Run `kubectl get pods` to verify `trainsight-data-validator` InitContainer scans dataset and exits with `0` before vLLM starts.
2. **vLLM Engine Loading:**
   * Run `kubectl logs -l app=vllm-server -c vllm-container` to verify model weights loading onto NVIDIA L4 GPU.
3. **Observability & DCGM Metrics:**
   * Forward Grafana port: `kubectl port-forward svc/grafana-service 3000:3000`.
   * Run SLA benchmark: `vllm-bench benchmark --host http://<EXTERNAL_IP>:8000 --num-requests 50 --concurrency 4`.
   * Capture Grafana screenshot showing **DCGM GPU Utilization Spikes** and **TTFT/TPOT Latency Histograms**.
4. **DeepSpeed GRPO Verification:**
   * Run 10 steps of PyTorch GRPO training using [grpo_trl_trainer.py](rlhf-pipeline/rlhf_pipeline/grpo_trl_trainer.py) and [deepspeed_config.json](rlhf-pipeline/deepspeed_config.json).

---

### Step 5: Teardown (Zero Idle Cost Guarantee)
Once screenshots and logs are captured, destroy all cloud resources immediately:

```bash
make cloud-down
```

> **README Statement:**  
> *"Infrastructure is fully ephemeral. The entire GKE cluster, GCS bucket, and Artifact Registry are provisioned and destroyed via `make cloud-up` and `make cloud-down`, guaranteeing zero idle GPU costs."*
