# pyrefly: ignore [missing-import]
import gradio as gr
import matplotlib.pyplot as plt

def validate_dataset(file):
    """Simulates TrainSight Pre-Flight Data Quality Inspection."""
    return {
        "Dataset Name": "real_gsm8k_1000.jsonl",
        "Total Samples": 1000,
        "Avg Sequence Length": "129.6 tokens",
        "P95 Sequence Length": "224.0 tokens",
        "OOM Risk Count (>2048)": 0,
        "Empty Completions": 0,
        "Duplicate Prompts": 0,
        "Validation Status": "✅ PASS (Exit Code 0)"
    }

def chat_inference(message, history):
    """Simulates vLLM SLA Inference with TTFT & TPOT metrics."""
    response = (
        "To solve 2x + 4 = 10:\n"
        "<think>\n"
        "1. Subtract 4 from both sides: 2x = 6\n"
        "2. Divide by 2: x = 3\n"
        "</think>\n"
        "<answer>3</answer>"
    )
    metrics = "\n\n📊 [vLLM Metrics] TTFT: 851.9ms | TPOT: 24.7ms/token | Status: HTTP 200 OK"
    return response + metrics

def plot_training_telemetry():
    """Generates GRPO Training Progression Curves."""
    steps = [1, 5, 10, 15, 20]
    mean_rewards = [0.42, 0.50, 0.64, 0.72, 0.85]
    kl_divergence = [0.0376, 0.0581, 0.0595, 0.0536, 0.0586]
    grpo_losses = [1.1934, 0.9610, 0.7322, 0.4367, 0.1730]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.plot(steps, mean_rewards, marker='o', color='green', linewidth=2)
    ax1.set_title("Mean Reward (r_mean)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward Scale [0-1]")

    ax2.plot(steps, kl_divergence, marker='s', color='orange', linewidth=2)
    ax2.set_title("KL Divergence (D_KL)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("KL Policy Shift")

    ax3.plot(steps, grpo_losses, marker='^', color='red', linewidth=2)
    ax3.set_title("GRPO Loss (L_grpo)")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Policy Loss")

    plt.tight_layout()
    return fig

# ─── Assemble Tabbed Interface ────────────────────────────────────
demo = gr.TabbedInterface(
    [
        gr.Interface(validate_dataset, gr.File(label="Upload Dataset (.jsonl)"), gr.JSON(label="TrainSight Quality Report"), title="📊 Data Validation"),
        gr.Interface(chat_inference, gr.Textbox(label="User Reasoning Prompt"), gr.Textbox(label="Model Response & SLA Telemetry"), title="💬 Inference Playground"),
        gr.Interface(plot_training_telemetry, None, gr.Plot(), title="📈 GRPO Telemetry"),
    ],
    tab_names=["📊 Data Validation", "💬 Inference Playground", "📈 GRPO Telemetry"],
    title="AetherControl: LLM Serving & Post-Training Control Plane Demo"
)

if __name__ == "__main__":
    demo.launch()
