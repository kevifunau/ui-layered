# -*- coding: utf-8 -*-
"""Pipeline step base class and the pipeline orchestrator (with per-stage timing)."""
import time


class PipelineStep:
    """Base class for a pipeline step.  Override run(ctx) in subclasses."""

    name = "unnamed"

    def run(self, ctx):
        raise NotImplementedError


class Pipeline:
    """Runs a list of steps in order, sharing a PipelineContext.

    The article publishes a per-stage timing chart, so every step is timed here and
    the result ends up in audit.json -> "timings".
    """

    def __init__(self, steps=None):
        self.steps = steps or []

    def run(self, ctx):
        for step in self.steps:
            ctx._step_name = step.name
            ctx.log(f"=== {step.name} ===")
            t0 = time.time()
            step.run(ctx)
            dt = time.time() - t0
            ctx.timings[step.name] = round(dt, 2)
            ctx.log(f"--- {step.name}: {dt:.1f}s (total {ctx.elapsed():.1f}s)")
        ctx.timings["total"] = round(ctx.elapsed(), 2)
        return ctx