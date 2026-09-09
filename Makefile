.PHONY: all test profile k8s-dev cloud-up cloud-down clean

all: test profile

test:
	@echo "🧪 Running unit tests across all AetherControl modules..."
	(cd trainsight && pytest)
	(cd vllm-engine && pytest)
	(cd rlhf-pipeline && pytest)
	(cd k8s-infra && PYTHONPATH=. pytest tests)
	pytest benchmarks/test_platform_benchmarks.py
	@echo "✅ All unit tests passed!"

benchmark:
	@echo "🚀 Running Unified Platform Invariant Benchmarks (TrainSight + RLHF + vLLM + K8S)..."
	python3 -m benchmarks.run_all_benchmarks --mode analytical

benchmark-live:
	@echo "🌐 Running Live Cluster Benchmark against GKE..."
	python3 -m benchmarks.run_all_benchmarks --mode live --endpoint $${ENDPOINT:-http://localhost:8000}

profile:
	@echo "🔍 Running TrainSight dataset quality profiling..."
	(cd trainsight && trainsight profile --dataset sample_data/sample_gsm8k.jsonl --type sft)
	@echo "⚡ Displaying vLLM engine production CLI arguments..."
	(cd vllm-engine && vllm-bench show-config)

k8s-dev:
	@echo "☸️ Applying local Kubernetes dev manifests..."
	kubectl apply -f k8s-infra/manifests/secrets.yaml
	kubectl apply -f k8s-infra/manifests/local-dev-deployment.yaml
	kubectl apply -f k8s-infra/manifests/prometheus-grafana.yaml
	@echo "✨ Kubernetes manifests applied successfully!"

cloud-up:
	@echo "🚀 Provisioning GCP Spot GPU GKE Cluster, GCS Bucket, and Artifact Registry..."
	(cd terraform && terraform init && terraform apply -auto-approve)
	@echo "✨ GCP Cloud Micro-Run Cluster Provisioned!"

cloud-down:
	@echo "🧹 Destroying GCP GKE Cluster and Cloud Resources (Zero Idle Billing)..."
	(cd terraform && terraform destroy -auto-approve)
	@echo "✨ GCP Infrastructure Teardown Complete!"

clean:
	@echo "🧹 Cleaning up local cluster and background processes..."
	-KIND_EXPERIMENTAL_PROVIDER=podman kind delete cluster --name vllm-local
	-podman machine stop
	@echo "✨ Teardown complete!"
