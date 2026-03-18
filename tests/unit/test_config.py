"""Unit tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openrag.config import OpenRAGConfig


class TestOpenRAGConfig:
    def test_defaults(self) -> None:
        config = OpenRAGConfig()
        assert config.namespace == "default"
        assert config.vector_db.adapter == "in_memory"
        assert config.graph_db.adapter == "networkx"
        assert config.llm.model == "gpt-4o-mini"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENRAG_NAMESPACE", "prod-ns")
        monkeypatch.setenv("OPENRAG_LLM__MODEL", "gpt-4o")
        config = OpenRAGConfig()
        assert config.namespace == "prod-ns"
        assert config.llm.model == "gpt-4o"

    def test_from_yaml(self, tmp_path: Path) -> None:
        pipeline = {
            "namespace": "yaml-ns",
            "vector_db": {"adapter": "qdrant", "url": "http://localhost:6333"},
            "llm": {"model": "gpt-4o-mini", "temperature": 0.2},
        }
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(yaml.dump(pipeline))

        config = OpenRAGConfig.from_yaml(config_file)
        assert config.namespace == "yaml-ns"
        assert config.vector_db.adapter == "qdrant"
        assert config.llm.temperature == 0.2

    def test_yaml_env_interpolation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-test-key")
        pipeline_yaml = "llm:\n  api_key: ${TEST_API_KEY}\n"
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(pipeline_yaml)

        config = OpenRAGConfig.from_yaml(config_file)
        assert config.llm.api_key == "sk-test-key"

    def test_ensure_working_dir(self, tmp_path: Path) -> None:
        config = OpenRAGConfig(working_dir=str(tmp_path / "openrag_storage"))
        path = config.ensure_working_dir()
        assert path.exists()
        assert path.is_dir()

    def test_processors_defaults(self) -> None:
        config = OpenRAGConfig()
        assert config.processors.text.enabled is True
        assert config.processors.image.enabled is True
        assert config.processors.audio_video.enabled is False
