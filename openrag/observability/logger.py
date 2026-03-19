"""Structured logging for OpenRAG."""

import logging
import json
import time
from typing import Any, Dict
from opentelemetry import trace

class StructuredLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            self._logger.addHandler(handler)

    def _format(self, msg: str, level: str, **kwargs: Any) -> str:
        span = trace.get_current_span()
        span_context = span.get_span_context()
        
        log_entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "level": level,
            "message": msg,
            "trace_id": format(span_context.trace_id, '032x') if span_context.is_valid else None,
            "span_id": format(span_context.span_id, '016x') if span_context.is_valid else None,
            **kwargs
        }
        return json.dumps(log_entry)

    def info(self, msg: str, **kwargs: Any):
        self._logger.info(self._format(msg, "INFO", **kwargs))

    def error(self, msg: str, **kwargs: Any):
        self._logger.error(self._format(msg, "ERROR", **kwargs))

    def warning(self, msg: str, **kwargs: Any):
        self._logger.warning(self._format(msg, "WARNING", **kwargs))

def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
