"""Async scraper for SHL Individual Test Solutions catalog pages."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
import structlog
from bs4 import BeautifulSoup

from app.utils.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


CATALOG_PATH_HINTS = (
    "/products/product-catalog/",
    "/solutions/products/product-catalog/",
    "/product-catalog/",
)

INDIVIDUAL_SOLUTION_HINTS = (
    "individual test solution",
    "individual-test-solution",
    "individual assessment",
)


@dataclass(frozen=True)
class ScrapedAssessment:
    """Raw assessment record scraped from SHL catalog content."""

    name: str
    url: str
    description: str = ""
    duration: str | None = None
    skills_measured: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    remote_testing: bool | None = None
    adaptive_support: bool | None = None
    test_type: str = ""


class SHLCatalogScraper:
    """Crawl and extract SHL Individual Test Solutions from catalog pages."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.shl_catalog_base_url.rstrip("/")
        self.start_url = self.settings.shl_catalog_start_url
        self.timeout = httpx.Timeout(self.settings.scraper_timeout_seconds)
        self._base_netloc = urlparse(self.base_url).netloc.lower()

    async def scrape(self) -> list[ScrapedAssessment]:
        """Scrape catalog assessments and return de-duplicated records."""

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": f"{self.settings.app_name}/1.0"},
        ) as client:
            listing_html = await self._fetch_text(client, self.start_url)
            candidate_urls = self._extract_candidate_urls(listing_html, self.start_url)

            if not candidate_urls:
                logger.warning("no_candidate_urls_found", start_url=self.start_url)
                return []

            semaphore = asyncio.Semaphore(self.settings.scraper_max_concurrency)

            async def fetch_and_parse(url: str) -> ScrapedAssessment | None:
                async with semaphore:
                    try:
                        html = await self._fetch_text(client, url)
                    except httpx.HTTPError as exc:
                        logger.warning("assessment_fetch_failed", url=url, error=str(exc))
                        return None
                    return self._parse_assessment_page(html, url)

            records = await asyncio.gather(*(fetch_and_parse(url) for url in candidate_urls))
            assessments = [record for record in records if record is not None]
            return self._deduplicate(assessments)

    async def scrape_to_json(self, output_path: Path | None = None) -> Path:
        """Scrape catalog pages and write raw assessment records to JSON."""

        destination = output_path or self.settings.catalog_json_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        assessments = await self.scrape()
        payload = [asdict(assessment) for assessment in assessments]
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("catalog_scraped", count=len(payload), output_path=str(destination))
        return destination

    async def _fetch_text(self, client: httpx.AsyncClient, url: str) -> str:
        response = await client.get(url)
        response.raise_for_status()
        return response.text

    def _extract_candidate_urls(self, html: str, page_url: str) -> list[str]:
        """Extract likely assessment detail URLs from a catalog listing page."""

        soup = BeautifulSoup(html, "lxml")
        urls: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            absolute_url = self._normalize_url(anchor["href"], page_url)
            if absolute_url and self._is_catalog_url(absolute_url):
                text = self._clean_text(anchor.get_text(" ", strip=True)).lower()
                if self._looks_like_assessment_link(absolute_url, text):
                    urls.add(absolute_url)

        return sorted(urls)

    def _parse_assessment_page(self, html: str, url: str) -> ScrapedAssessment | None:
        """Parse a detail page into a raw assessment record."""

        soup = BeautifulSoup(html, "lxml")
        page_text = self._clean_text(soup.get_text(" ", strip=True))
        lower_text = page_text.lower()
        if not any(hint in lower_text for hint in INDIVIDUAL_SOLUTION_HINTS):
            logger.debug("skipping_non_individual_solution", url=url)
            return None

        name = self._extract_title(soup)
        if not name:
            logger.warning("assessment_missing_name", url=url)
            return None

        return ScrapedAssessment(
            name=name,
            url=url,
            description=self._extract_description(soup),
            duration=self._extract_labeled_value(page_text, ("duration", "completion time", "time")),
            skills_measured=self._extract_list_value(page_text, ("skills measured", "measures", "knowledge")),
            languages=self._extract_list_value(page_text, ("languages", "available languages")),
            remote_testing=self._extract_boolean_value(page_text, ("remote testing", "remote proctoring")),
            adaptive_support=self._extract_boolean_value(page_text, ("adaptive", "adaptive testing")),
            test_type=self._extract_labeled_value(page_text, ("test type", "assessment type")) or "",
        )

    def _normalize_url(self, href: str, page_url: str) -> str | None:
        if href.startswith(("mailto:", "tel:", "#")):
            return None
        absolute_url = urljoin(page_url, href).split("#", 1)[0]
        parsed = urlparse(absolute_url)
        if parsed.scheme not in {"http", "https"}:
            return None
        if parsed.netloc.lower() != self._base_netloc:
            return None
        return absolute_url.rstrip("/")

    def _is_catalog_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return any(hint in path for hint in CATALOG_PATH_HINTS)

    def _looks_like_assessment_link(self, url: str, anchor_text: str) -> bool:
        path = urlparse(url).path.lower()
        if path.rstrip("/").endswith(tuple(hint.rstrip("/") for hint in CATALOG_PATH_HINTS)):
            return False
        if "learn more" in anchor_text or "view details" in anchor_text:
            return True
        return len(path.rstrip("/").split("/")) > 3

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for selector in ("h1", "meta[property='og:title']", "title"):
            element = soup.select_one(selector)
            if not element:
                continue
            text = element.get("content") if element.name == "meta" else element.get_text(" ", strip=True)
            cleaned = self._clean_text(text or "").replace("| SHL", "").strip(" -")
            if cleaned:
                return cleaned
        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        meta = soup.select_one("meta[name='description'], meta[property='og:description']")
        if meta and meta.get("content"):
            return self._clean_text(meta["content"])

        for selector in ("main p", "article p", ".product-detail p", "p"):
            paragraph = soup.select_one(selector)
            if paragraph:
                text = self._clean_text(paragraph.get_text(" ", strip=True))
                if len(text) >= 40:
                    return text
        return ""

    def _extract_labeled_value(self, text: str, labels: Iterable[str]) -> str | None:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[:\-]?\s*([^.;|]{{1,120}})"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        return None

    def _extract_list_value(self, text: str, labels: Iterable[str]) -> list[str]:
        value = self._extract_labeled_value(text, labels)
        if not value:
            return []
        parts = re.split(r",|;|\band\b|\|", value)
        return [self._clean_text(part) for part in parts if self._clean_text(part)]

    def _extract_boolean_value(self, text: str, labels: Iterable[str]) -> bool | None:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[:\-]?\s*(yes|no|true|false|available|supported|not available|not supported)"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).lower()
            return value in {"yes", "true", "available", "supported"}
        return None

    def _deduplicate(self, assessments: Iterable[ScrapedAssessment]) -> list[ScrapedAssessment]:
        by_url: dict[str, ScrapedAssessment] = {}
        for assessment in assessments:
            by_url[assessment.url] = assessment
        return sorted(by_url.values(), key=lambda item: item.name.lower())

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


async def scrape_catalog(output_path: Path | None = None) -> Path:
    """Run the SHL catalog scraper with default settings."""

    return await SHLCatalogScraper().scrape_to_json(output_path)


if __name__ == "__main__":
    asyncio.run(scrape_catalog())
