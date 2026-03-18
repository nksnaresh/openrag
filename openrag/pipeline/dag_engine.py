"""DAG-based pipeline engine for the OpenRAG ingestion pipeline.

Stages are executed in topological order. Independent stages run
concurrently via anyio task groups.

Usage::

    engine = DAGPipelineEngine(stages=[
        PipelineStage("enrich", context_enricher_stage, depends_on=[]),
        PipelineStage("text",   text_processor,  depends_on=["enrich"],
                      block_types=[BlockType.TEXT]),
        PipelineStage("image",  image_processor, depends_on=["enrich"],
                      block_types=[BlockType.IMAGE]),
    ], config=config)

    processed = await engine.execute(payload, context)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from openrag.config import OpenRAGConfig
from openrag.models.content import BlockType, ContentBlock, ContentPayload
from openrag.models.processing import ProcessedBlock, ProcessingContext
from openrag.processors.base import BaseModalityProcessor


class CyclicDependencyError(ValueError):
    """Raised when the pipeline DAG contains a cycle."""


@dataclass
class PipelineStage:
    """A single stage in the processing DAG.

    Attributes:
        name:           Unique stage identifier used for dependency lookup.
        processor:      Modality processor that processes blocks for this stage.
        depends_on:     Names of stages that must complete before this one.
        timeout_seconds: Per-stage execution timeout (not enforced in Phase 2,
                         wired for future use).
        block_types:    If set, only blocks of these types are processed by
                        this stage. None means all block types are processed.
    """

    name: str
    processor: BaseModalityProcessor
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: float = 120.0
    block_types: list[BlockType] | None = None


class DAGPipelineEngine:
    """Executes a set of PipelineStages in topological order.

    Independent stages at the same depth are dispatched concurrently via
    ``asyncio.gather``.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        config: OpenRAGConfig,
    ) -> None:
        self._stages = {s.name: s for s in stages}
        self._config = config
        self._order = self._topological_sort(stages)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def execute(
        self,
        payload: ContentPayload,
        context: ProcessingContext,
    ) -> list[ProcessedBlock]:
        """Execute all stages and return the collected ProcessedBlocks.

        Stages with no mutual dependencies are run in parallel within each
        topological level.
        """
        all_results: list[ProcessedBlock] = []
        # completed_outputs: stage_name → list[ProcessedBlock]
        completed: dict[str, list[ProcessedBlock]] = {}

        for level in self._order:
            level_tasks = [
                self._run_stage(
                    self._stages[name], payload, context, completed
                )
                for name in level
            ]
            level_results = await asyncio.gather(*level_tasks)
            for name, blocks in zip(level, level_results, strict=False):
                completed[name] = blocks
                all_results.extend(blocks)

        return all_results

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _run_stage(
        self,
        stage: PipelineStage,
        payload: ContentPayload,
        context: ProcessingContext,
        _completed: dict[str, list[ProcessedBlock]],
    ) -> list[ProcessedBlock]:
        """Run a single stage: filter blocks, call processor, collect results."""
        blocks_to_process: list[ContentBlock] = [
            b for b in payload.blocks
            if stage.block_types is None or b.block_type in stage.block_types
        ]
        if not blocks_to_process:
            return []

        tasks = [stage.processor.process(block, context) for block in blocks_to_process]
        gathered = await asyncio.gather(*tasks)
        results: list[ProcessedBlock] = list(gathered)
        return list(results)

    @staticmethod
    def _topological_sort(stages: list[PipelineStage]) -> list[list[str]]:
        """Kahn's algorithm → levels of independent stages.

        Returns a list of levels, where each level is a list of stage names
        that can run concurrently.

        Raises:
            CyclicDependencyError: If the dependency graph contains a cycle.
        """
        names = {s.name for s in stages}
        in_degree: dict[str, int] = {s.name: 0 for s in stages}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for stage in stages:
            for dep in stage.depends_on:
                if dep not in names:
                    raise ValueError(
                        f"Stage '{stage.name}' depends on unknown stage '{dep}'."
                    )
                adjacency[dep].append(stage.name)
                in_degree[stage.name] += 1

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        levels: list[list[str]] = []
        visited = 0

        while queue:
            # Drain all nodes at the current level
            level: list[str] = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node)
                visited += 1
                for neighbour in adjacency[node]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        queue.append(neighbour)
            levels.append(level)

        if visited != len(stages):
            raise CyclicDependencyError(
                "Pipeline DAG contains a cycle. Check 'depends_on' relationships."
            )

        return levels


def build_default_pipeline(
    config: OpenRAGConfig,
    processors: dict[str, Any],
) -> DAGPipelineEngine:
    """Build the default pipeline from a config and a dict of processor instances.

    Args:
        config:     OpenRAGConfig with processor enable flags.
        processors: Mapping of processor name to BaseModalityProcessor instance.
                    Expected keys: "text", "image", "table", "equation", "code".

    Returns:
        Configured DAGPipelineEngine ready to execute.
    """
    stages: list[PipelineStage] = []

    mapping: list[tuple[str, list[BlockType]]] = [
        ("text",     [BlockType.TEXT]),
        ("image",    [BlockType.IMAGE]),
        ("table",    [BlockType.TABLE]),
        ("equation", [BlockType.EQUATION]),
        ("code",     [BlockType.CODE]),
    ]

    for name, block_types in mapping:
        if name in processors:
            stages.append(
                PipelineStage(
                    name=name,
                    processor=processors[name],
                    depends_on=[],
                    block_types=block_types,
                )
            )

    return DAGPipelineEngine(stages, config)
