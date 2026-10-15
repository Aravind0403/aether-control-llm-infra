#!/usr/bin/env python3
"""CLI Entrypoint for the 4-Mode Serving Pilot Experiment."""
import asyncio
from vllm_engine.benchmarks.pilot_four_modes import run_pilot

if __name__ == "__main__":
    asyncio.run(run_pilot())
