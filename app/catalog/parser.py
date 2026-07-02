"""Parse raw SHL scraper output into validated catalog records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import ValidationError

from app.models.assessment import Assessment
from app.utils.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


class CatalogParser:
    """Load, normalize, validate, and persist SHL assessment catalog records."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def parse_file(self, input_path: Path | None = None) -> list[Assessment]:
        """Parse raw scraper JSON from disk."""

        source = input_path or self.settings.catalog_json_path
        raw_records = self.load_raw_records(source)
        return self.parse_records(raw_records)

    def load_raw_records(self, path: Path) -> list[dict[str, Any]]:
        """Load raw JSON records from the scraper."""

        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Catalog JSON must contain a list of assessment records")

        records: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                records.append(item)
            else:
                logger.warning("catalog_record_not_object", record=repr(item))
        return records

    def parse_records(self, raw_records: list[dict[str, Any]]) -> list[Assessment]:
        """Normalize and validate raw scraped records."""

        assessments: list[Assessment] = []
        seen_urls: set[str] = set()

        for raw_record in raw_records:
            normalized = self._normalize_record(raw_record)
            try:
                assessment = Assessment.model_validate(normalized)
            except ValidationError as exc:
                logger.warning(
                    "catalog_record_rejected",
                    name=normalized.get("name"),
                    url=normalized.get("url"),
                    errors=exc.errors(),
                )
                continue

            if assessment.url in seen_urls:
                continue
            seen_urls.add(assessment.url)
            assessments.append(assessment)

        return sorted(assessments, key=lambda item: item.name.casefold())

    def save_parsed_catalog(
        self,
        assessments: list[Assessment],
        output_path: Path | None = None,
    ) -> Path:
        """Persist validated catalog records as JSON."""

        destination = output_path or self.settings.catalog_json_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [assessment.model_dump() for assessment in assessments]
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("catalog_parsed", count=len(payload), output_path=str(destination))
        return destination

    def parse_and_save(
        self,
        input_path: Path | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Parse raw catalog JSON and save validated records."""

        assessments = self.parse_file(input_path)
        return self.save_parsed_catalog(assessments, output_path)

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Map raw scraper keys into the validated assessment schema."""

        name = self._clean_string(record.get("name"))
        url = self._clean_string(record.get("url"))
        description = self._clean_string(record.get("description"))
        duration = self._clean_optional_string(record.get("duration"))
        test_type = self._clean_string(record.get("test_type") or record.get("assessment_type"))

        return {
            "name": name,
            "url": url,
            "description": description,
            "duration": duration,
            "duration_minutes": record.get("duration_minutes"),
            "skills_measured": record.get("skills_measured") or record.get("skills") or [],
            "languages": record.get("languages") or [],
            "remote_testing": self._normalize_bool(record.get("remote_testing")),
            "adaptive_support": self._normalize_bool(record.get("adaptive_support")),
            "test_type": test_type,
        }

    def _clean_string(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    def _clean_optional_string(self, value: Any) -> str | None:
        cleaned = self._clean_string(value)
        return cleaned or None

    def _normalize_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"yes", "true", "available", "supported", "y"}:
                return True
            if normalized in {"no", "false", "not available", "not supported", "n"}:
                return False
        return None


def parse_catalog(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Parse SHL catalog JSON using default settings."""

    return CatalogParser().parse_and_save(input_path=input_path, output_path=output_path)
