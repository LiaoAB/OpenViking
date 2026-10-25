# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Reference adapter from OpenViking's Understanding API to LlamaParse v2.

Files API -> LlamaParse upload; Responses API -> LlamaParse parse and status;
LlamaParse Markdown and images -> the ZIP returned through ``result.zip_url``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import os
import re
import time
import zipfile
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, AsyncIterator, BinaryIO, Mapping, Optional
from urllib.parse import quote, urlencode, urlsplit

import httpx
from fastapi import Depends, FastAPI, Form, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

REGION_URLS = {
    "na": "https://api.cloud.llamaindex.ai",
    "eu": "https://api.cloud.eu.llamaindex.ai",
}
SUPPORTED_TIERS = {"cost_effective", "agentic", "agentic_plus"}
MAX_CACHED_ARTIFACTS = 32
MAX_CONCURRENT_ARTIFACT_BUILDS = 2
MAX_CONCURRENT_IMAGE_DOWNLOADS = 8
IMAGE_PROVENANCE_FILENAME = ".image_provenance.json"

_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^\n]+?)\)")
_HTML_IMAGE = re.compile(r"(<img\s[^>]*?src=[\"'])([^\"']+)([\"'][^>]*>)", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^\s*(?P<kind>screenshot|image)\s*:\s*(?P<label>.+?)\s*$", re.IGNORECASE)


class BridgeError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class LlamaParseError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        value = int(env.get(name, "").strip() or default)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _parse_options(raw: str) -> dict[str, Any]:
    try:
        options = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError("LLAMAPARSE_PARSE_OPTIONS_JSON must be valid JSON") from exc
    if not isinstance(options, dict):
        raise ValueError("LLAMAPARSE_PARSE_OPTIONS_JSON must contain a JSON object")
    conflicts = sorted({"file_id", "source_url", "tier", "version"} & options.keys())
    if conflicts:
        raise ValueError(
            "LLAMAPARSE_PARSE_OPTIONS_JSON cannot set bridge-managed fields: "
            + ", ".join(conflicts)
        )
    for name in ("output_options", "processing_options"):
        if name in options and not isinstance(options[name], dict):
            raise ValueError(f"LLAMAPARSE_PARSE_OPTIONS_JSON.{name} must be an object")
    output = options.get("output_options")
    if isinstance(output, dict) and not isinstance(output.get("images_to_save", []), list):
        raise ValueError(
            "LLAMAPARSE_PARSE_OPTIONS_JSON.output_options.images_to_save must be an array"
        )
    return options


@dataclass(frozen=True)
class Settings:
    llama_api_key: str
    bridge_api_key: str
    llama_base_url: str = REGION_URLS["na"]
    tier: str = "agentic"
    version: str = "latest"
    cost_optimizer: bool = True
    public_url: str = "http://127.0.0.1:8080"
    bind_host: str = "127.0.0.1"
    bind_port: int = 8080
    timeout_seconds: int = 120
    artifact_ttl_seconds: int = 300
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    parse_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.llama_api_key.strip():
            raise ValueError("LLAMA_CLOUD_API_KEY is required")
        if len(self.bridge_api_key.strip()) < 32:
            raise ValueError("PARSER_BRIDGE_API_KEY must contain at least 32 characters")
        if self.tier not in SUPPORTED_TIERS:
            raise ValueError(f"unsupported LlamaParse tier: {self.tier}")
        if self.cost_optimizer and self.tier == "cost_effective":
            raise ValueError("LLAMAPARSE_COST_OPTIMIZER requires an agentic tier")
        for value, name in (
            (self.llama_base_url, "LLAMAPARSE_BASE_URL"),
            (self.public_url, "BRIDGE_PUBLIC_URL"),
        ):
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} must be an HTTP(S) URL")
            if parsed.query or parsed.fragment:
                raise ValueError(f"{name} must not contain a query or fragment")
        if not 1 <= self.bind_port <= 65535:
            raise ValueError("BRIDGE_PORT must be between 1 and 65535")
        if self.timeout_seconds <= 0 or self.artifact_ttl_seconds <= 0:
            raise ValueError("bridge timeouts must be greater than zero")
        _parse_options(json.dumps(self.parse_options))


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    values = env if env is not None else os.environ
    region = values.get("LLAMAPARSE_REGION", "na").strip().lower() or "na"
    base_url = values.get("LLAMAPARSE_BASE_URL", "").strip()
    if not base_url:
        try:
            base_url = REGION_URLS[region]
        except KeyError as exc:
            raise ValueError("LLAMAPARSE_REGION must be na or eu") from exc
    return Settings(
        llama_api_key=values.get("LLAMA_CLOUD_API_KEY", "").strip(),
        bridge_api_key=values.get("PARSER_BRIDGE_API_KEY", "").strip(),
        llama_base_url=base_url.rstrip("/"),
        tier=values.get("LLAMAPARSE_TIER", "agentic").strip().lower(),
        version=values.get("LLAMAPARSE_VERSION", "latest").strip() or "latest",
        cost_optimizer=_boolean(values, "LLAMAPARSE_COST_OPTIMIZER", True),
        public_url=values.get("BRIDGE_PUBLIC_URL", "http://127.0.0.1:8080").strip().rstrip("/"),
        bind_host=values.get("BRIDGE_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1",
        bind_port=_positive_int(values, "BRIDGE_PORT", 8080),
        timeout_seconds=_positive_int(values, "BRIDGE_HTTP_TIMEOUT_SECONDS", 120),
        artifact_ttl_seconds=_positive_int(values, "BRIDGE_ARTIFACT_TTL_SECONDS", 300),
        organization_id=values.get("LLAMAPARSE_ORGANIZATION_ID", "").strip() or None,
        project_id=values.get("LLAMAPARSE_PROJECT_ID", "").strip() or None,
        parse_options=_parse_options(values.get("LLAMAPARSE_PARSE_OPTIONS_JSON", "")),
    )


class LlamaParseClient:
    def __init__(
        self,
        settings: Settings,
        *,
        api_client: Optional[httpx.AsyncClient] = None,
        download_client: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings
        timeout = httpx.Timeout(settings.timeout_seconds)
        self._owns_api = api_client is None
        self._owns_downloads = download_client is None
        self.api = api_client or httpx.AsyncClient(
            base_url=settings.llama_base_url,
            headers={"Authorization": f"Bearer {settings.llama_api_key}"},
            timeout=timeout,
        )
        # This client has no LlamaCloud authorization header.
        self.downloads = download_client or httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        )

    async def aclose(self) -> None:
        if self._owns_api:
            await self.api.aclose()
        if self._owns_downloads:
            await self.downloads.aclose()

    def _scope(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "organization_id": self.settings.organization_id,
                "project_id": self.settings.project_id,
            }.items()
            if value
        }

    @staticmethod
    async def _request(
        client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise LlamaParseError(504, "LlamaParse request timed out") from exc
        except httpx.RequestError as exc:
            raise LlamaParseError(502, "LlamaParse request failed") from exc
        if response.is_error:
            try:
                payload = response.json()
                detail = payload.get("detail") or payload.get("error") or payload.get("message")
                if isinstance(detail, dict):
                    detail = detail.get("message")
            except (ValueError, AttributeError):
                detail = response.text.strip()
            raise LlamaParseError(
                response.status_code, str(detail or f"HTTP {response.status_code}")
            )
        return response

    @classmethod
    async def _json_request(
        cls, client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
    ) -> dict[str, Any]:
        response = await cls._request(client, method, url, **kwargs)
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaParseError(502, "LlamaParse returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise LlamaParseError(502, "LlamaParse returned an invalid response object")
        return payload

    async def upload_file(
        self, filename: str, file: BinaryIO, content_type: Optional[str]
    ) -> dict[str, Any]:
        return await self._json_request(
            self.api,
            "POST",
            "/api/v1/beta/files",
            params=self._scope(),
            data={"purpose": "parse"},
            files={"file": (filename, file, content_type or "application/octet-stream")},
        )

    async def create_job(
        self, *, file_id: Optional[str] = None, source_url: Optional[str] = None
    ) -> dict[str, Any]:
        if bool(file_id) == bool(source_url):
            raise ValueError("exactly one of file_id or source_url is required")
        config = deepcopy(self.settings.parse_options)
        config.update(tier=self.settings.tier, version=self.settings.version)
        config.setdefault("client_name", "openviking-understanding-bridge")
        images = config.setdefault("output_options", {}).setdefault("images_to_save", [])
        for category in ("embedded", "layout"):
            if category not in images:
                images.append(category)
        processing = config.setdefault("processing_options", {})
        if self.settings.cost_optimizer:
            processing["cost_optimizer"] = {"enable": True}
        else:
            processing.pop("cost_optimizer", None)
        if not processing:
            config.pop("processing_options")
        config["file_id" if file_id else "source_url"] = file_id or source_url
        return await self._json_request(
            self.api, "POST", "/api/v2/parse", params=self._scope(), json=config
        )

    async def get_job(self, job_id: str, *, include_result: bool = False) -> dict[str, Any]:
        params = self._scope()
        if include_result:
            params["expand"] = "markdown_full,markdown,images_content_metadata,items,metadata"
        return await self._json_request(
            self.api, "GET", f"/api/v2/parse/{quote(job_id, safe='')}", params=params
        )

    async def download_asset(self, url: str) -> bytes:
        return (await self._request(self.downloads, "GET", url)).content


class ArtifactSigner:
    def __init__(self, secret: str, ttl_seconds: int):
        self.secret = secret.encode()
        self.ttl_seconds = ttl_seconds

    def _signature(self, job_id: str, expires: int) -> str:
        return hmac.new(self.secret, f"{job_id}:{expires}".encode(), hashlib.sha256).hexdigest()

    def create_url(self, public_url: str, job_id: str) -> str:
        expires = int(time.time()) + self.ttl_seconds
        query = urlencode({"expires": expires, "signature": self._signature(job_id, expires)})
        return f"{public_url}/artifacts/{quote(job_id, safe='')}.zip?{query}"

    def verify(self, job_id: str, expires: str, signature: str) -> bool:
        try:
            expiry = int(expires)
        except ValueError:
            return False
        return int(time.time()) <= expiry and hmac.compare_digest(
            self._signature(job_id, expiry), signature
        )


def _markdown(result: dict[str, Any]) -> str:
    markdown = result.get("markdown")
    pages = markdown.get("pages") if isinstance(markdown, dict) else []
    if not isinstance(pages, list):
        pages = []
    failed = [
        str(page.get("page_number", index))
        for index, page in enumerate(pages, start=1)
        if isinstance(page, dict) and page.get("success") is False
    ]
    full = result.get("markdown_full")
    if isinstance(full, str) and full.strip():
        text = full.strip()
    else:
        parts = [
            page.get("markdown")
            for page in pages
            if isinstance(page, dict) and page.get("success") is not False
        ]
        text = "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    if not text:
        raise LlamaParseError(502, "completed LlamaParse job has no usable Markdown")
    if failed:
        label = "Page" if len(failed) == 1 else "Pages"
        text += f"\n\n> {label} {', '.join(failed)} failed to parse."
    return text


def _safe_image_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LlamaParseError(502, "LlamaParse image has no filename")
    name = value.strip()
    if any(ord(character) < 32 for character in name) or ":" in name:
        raise LlamaParseError(502, "LlamaParse image has an unsafe filename")
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise LlamaParseError(502, "LlamaParse image has an unsafe filename")
    return path.name


def _path_basename(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().strip("<>")
    if not value:
        return ""
    path = urlsplit(value).path
    return PurePosixPath(path.replace("\\", "/")).name


def _bbox_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    keys = ("x", "y", "w", "h")
    if not all(key in left and key in right for key in keys):
        return False
    try:
        return all(abs(float(left[key]) - float(right[key])) <= 1e-3 for key in keys)
    except (TypeError, ValueError):
        return False


def _item_bbox(item: Mapping[str, Any]) -> Any:
    bbox = item.get("bbox")
    if isinstance(bbox, list):
        return next((candidate for candidate in bbox if isinstance(candidate, dict)), None)
    return bbox


def _image_items(result: Mapping[str, Any]) -> list[dict[str, Any]]:  # noqa: C901
    """Return image items with their page and block position.

    The provider has added fields to image items over time. Keep this reader
    intentionally tolerant: the stable parts of the contract are ``type``,
    ``md``, ``bbox`` and the page number.
    """
    value = result.get("items")
    pages = value.get("pages") if isinstance(value, dict) else []
    if not isinstance(pages, list):
        return []
    metadata_value = result.get("metadata")
    metadata_pages = metadata_value.get("pages") if isinstance(metadata_value, dict) else []
    metadata_by_page = {
        page.get("page_number"): page
        for page in metadata_pages
        if isinstance(page, dict) and page.get("page_number") is not None
    }

    found: list[dict[str, Any]] = []

    def visit(item: Any, page_number: Any, block_index: int) -> None:
        if not isinstance(item, dict):
            return
        if str(item.get("type", "")).lower() == "image":
            found.append(
                {
                    "page_number": page_number,
                    "block_index": block_index,
                    "bbox": _item_bbox(item),
                    "page_metadata": metadata_by_page.get(page_number),
                    "md": item.get("md") if isinstance(item.get("md"), str) else "",
                    "references": [
                        item.get(key)
                        for key in ("url", "image_url", "src", "filename", "name")
                        if isinstance(item.get(key), str)
                    ],
                }
            )
        nested = item.get("items")
        if isinstance(nested, list):
            for index, child in enumerate(nested):
                visit(child, page_number, index)

    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        items = page.get("items")
        if isinstance(items, list):
            for index, item in enumerate(items):
                visit(item, page_number, index)
    return found


def _asset_position(asset: Mapping[str, Any], item: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Copy provider provenance without copying temporary download URLs."""
    position: dict[str, Any] = {}
    for key in (
        "page_number",
        "slide_number",
        "block",
        "block_index",
        "occurrence",
        "bbox",
    ):
        if key in asset and asset[key] is not None:
            position[key] = asset[key]
    if item is not None:
        for key in ("page_number", "block_index", "bbox"):
            if key not in position and item.get(key) is not None:
                position[key] = item[key]
        if isinstance(item.get("page_metadata"), dict):
            position["page_metadata"] = item["page_metadata"]
    return position


def _match_image_items(
    image: Mapping[str, Any], image_items: list[dict[str, Any]], image_count: int
) -> list[dict[str, Any]]:
    name = str(image.get("filename", ""))
    exact = [
        item
        for item in image_items
        if any(_path_basename(reference).casefold() == name.casefold() for reference in item["references"])
    ]
    if exact:
        return exact

    page_number = image.get("page_number")
    bbox = image.get("bbox")
    by_position = [
        item
        for item in image_items
        if (page_number is None or item.get("page_number") == page_number)
        and _bbox_equal(bbox, item.get("bbox"))
    ]
    if by_position:
        return by_position

    # The documented image index is extraction order. Only use it when the
    # complete image list has the same cardinality; otherwise an index can
    # silently point at a screenshot or a layout crop.
    index = image.get("index")
    if image_count == len(image_items) and isinstance(index, int) and 0 <= index < len(image_items):
        return [image_items[index]]
    return []


def _fenced_ranges(markdown: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    fence_start: Optional[int] = None
    fence_char = ""
    fence_length = 0
    for line in markdown.splitlines(keepends=True):
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence_start is None:
            if match:
                fence_start = offset
                fence_char = match.group(1)[0]
                fence_length = len(match.group(1))
        elif match and match.group(1)[0] == fence_char and len(match.group(1)) >= fence_length:
            ranges.append((fence_start, offset + len(line)))
            fence_start = None
        offset += len(line)
    if fence_start is not None:
        ranges.append((fence_start, len(markdown)))
    return ranges


def _in_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _rewrite_provider_image_refs(markdown: str, image_paths: Mapping[str, str]) -> str:
    """Turn provider/temporary image links into local artifact links."""
    protected = _fenced_ranges(markdown)

    def markdown_replacer(match: re.Match[str]) -> str:
        if _in_ranges(match.start(), protected):
            return match.group(0)
        reference = match.group(2).strip()
        target = image_paths.get(_path_basename(reference).casefold())
        if target is None:
            return match.group(0)
        return f"![{match.group(1)}]({target})"

    def html_replacer(match: re.Match[str]) -> str:
        if _in_ranges(match.start(), protected):
            return match.group(0)
        target = image_paths.get(_path_basename(match.group(2)).casefold())
        if target is None:
            return match.group(0)
        return f"{match.group(1)}{target}{match.group(3)}"

    markdown = _MARKDOWN_IMAGE.sub(markdown_replacer, markdown)
    return _HTML_IMAGE.sub(html_replacer, markdown)


def _markdown_occurrences(markdown: str, image_path: str) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    protected = _fenced_ranges(markdown)
    for match in _MARKDOWN_IMAGE.finditer(markdown):
        if _in_ranges(match.start(), protected):
            continue
        if _path_basename(match.group(2).strip()).casefold() != image_path.casefold():
            continue
        occurrences.append({"line": markdown.count("\n", 0, match.start()) + 1})
    for match in _HTML_IMAGE.finditer(markdown):
        if _in_ranges(match.start(), protected):
            continue
        if _path_basename(match.group(2).strip()).casefold() != image_path.casefold():
            continue
        occurrences.append({"line": markdown.count("\n", 0, match.start()) + 1})
    return occurrences


def _insertion_line(markdown: str, item: Mapping[str, Any]) -> Optional[int]:
    protected = _fenced_ranges(markdown)
    anchor = item.get("md")
    if isinstance(anchor, str) and anchor.strip():
        position = markdown.find(anchor)
        if position >= 0 and not _in_ranges(position, protected):
            return markdown.count("\n", 0, position)

    offset = 0
    for line_number, line in enumerate(markdown.splitlines(keepends=True), start=0):
        if not _in_ranges(offset, protected) and _PLACEHOLDER.match(line.rstrip("\r\n")):
            return line_number
        offset += len(line)
    return None


def _insert_missing_occurrences(  # noqa: C901
    markdown: str,
    assets: list[dict[str, Any]],
    source_url: Optional[str],
) -> tuple[str, list[str]]:
    """Insert only occurrences with an explicit, unique text anchor."""
    lines = markdown.splitlines(keepends=True)
    protected = _fenced_ranges(markdown)
    placeholders = [
        index
        for index, line in enumerate(lines)
        if not _in_ranges(sum(len(previous) for previous in lines[:index]), protected)
        and _PLACEHOLDER.match(line.rstrip("\r\n"))
    ]
    used_lines: set[int] = set()
    warnings: list[str] = []
    pending: list[tuple[dict[str, Any], int]] = []

    for asset in assets:
        if not asset.get("artifact_path"):
            continue
        if _markdown_occurrences(markdown, asset["artifact_path"]):
            continue
        items = asset["items"]
        for item in items:
            line = _insertion_line(markdown, item)
            if line is not None and line not in used_lines:
                pending.append((asset, line))
                used_lines.add(line)

    # A placeholder is a valid, explicit anchor even when the provider did not
    # return the structured items tree. One-to-one matching prevents a loose
    # "append every image" fallback.
    unresolved = [
        asset
        for asset in assets
        if asset.get("artifact_path")
        and not _markdown_occurrences(markdown, asset["artifact_path"])
    ]
    remaining = [asset for asset in unresolved if not any(owner is asset for owner, _ in pending)]
    free_placeholders = [line for line in placeholders if line not in used_lines]
    if len(remaining) == len(free_placeholders):
        for asset, line in zip(remaining, free_placeholders, strict=True):
            pending.append((asset, line))
            used_lines.add(line)

    for asset in assets:
        if not asset.get("artifact_path"):
            continue
        if not _markdown_occurrences(markdown, asset["artifact_path"]) and not any(
            owner is asset for owner, _ in pending
        ):
            warnings.append(f"image {asset['original_filename']} has no reliable Markdown occurrence")

    for asset, line in sorted(pending, key=lambda pair: pair[1], reverse=True):
        label = asset.get("label") or asset["original_filename"]
        reference = f"![{label}]({asset['artifact_path']})"
        if source_url:
            reference = f"[View source document]({source_url})\n\n{reference}"
        suffix = "" if lines[line].endswith("\n") else "\n"
        lines[line] = lines[line] + suffix + "\n" + reference + "\n"
        asset["inserted"] = True
    return "".join(lines), warnings


def _make_zip(
    markdown: str,
    images: list[tuple[str, bytes]],
    provenance: Optional[dict[str, Any]] = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("content.md", markdown.encode())
        for name, content in images:
            archive.writestr(name, content)
        if provenance is not None:
            archive.writestr(
                IMAGE_PROVENANCE_FILENAME,
                json.dumps(provenance, ensure_ascii=False, indent=2).encode(),
            )
    return output.getvalue()


async def build_artifact(  # noqa: C901
    client: LlamaParseClient,
    job_id: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> bytes:
    result = await client.get_job(job_id, include_result=True)
    job = result.get("job")
    if not isinstance(job, dict) or str(job.get("status", "")).upper() != "COMPLETED":
        raise LlamaParseError(409, "LlamaParse job is not complete")
    metadata = result.get("images_content_metadata")
    images_value = metadata.get("images") if isinstance(metadata, dict) else None
    raw_images = images_value if isinstance(images_value, list) else []
    assets: list[dict[str, Any]] = []
    names = {"content.md"}
    for image in raw_images:
        if not isinstance(image, dict):
            raise LlamaParseError(502, "LlamaParse returned an invalid image metadata entry")
        name = _safe_image_name(image.get("filename") if isinstance(image, dict) else None)
        url = image.get("presigned_url") if isinstance(image, dict) else None
        if name.casefold() in names:
            raise LlamaParseError(502, f"LlamaParse returned duplicate image: {name}")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise LlamaParseError(502, f"LlamaParse image has no secure download URL: {name}")
        assets.append(
            {
                "original_filename": name,
                "artifact_path": name,
                "category": image.get("category"),
                "content_type": image.get("content_type"),
                "bbox": image.get("bbox"),
                "metadata": image,
                "items": [],
            }
        )
        names.add(name.casefold())
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_DOWNLOADS)

    async def download(url: str) -> bytes:
        async with semaphore:
            return await client.download_asset(url)

    tasks = [
        asyncio.create_task(download(image["metadata"]["presigned_url"])) for image in assets
    ]
    contents = await asyncio.gather(*tasks, return_exceptions=True)
    image_bytes: list[tuple[dict[str, Any], bytes]] = []
    download_warnings: list[str] = []
    for asset, content in zip(assets, contents, strict=True):
        if isinstance(content, BaseException):
            asset["download_error"] = f"{type(content).__name__}: {content}"
            asset["artifact_path"] = None
            download_warnings.append(
                f"image {asset['original_filename']} could not be downloaded; requires_review"
            )
            continue
        image_bytes.append((asset, content))

    # Identical bytes share one artifact file, while every metadata entry and
    # every body occurrence remains represented in the provenance manifest.
    canonical_by_hash: dict[str, str] = {}
    images: list[tuple[str, bytes]] = []
    for asset, content in image_bytes:
        digest = hashlib.sha256(content).hexdigest()
        asset["sha256"] = digest
        canonical = canonical_by_hash.get(digest)
        if canonical is None:
            canonical = asset["artifact_path"]
            canonical_by_hash[digest] = canonical
            images.append((canonical, content))
        asset["artifact_path"] = canonical

    original_markdown = _markdown(result)
    for asset in assets:
        asset["original_occurrence_count"] = len(
            _markdown_occurrences(original_markdown, asset["original_filename"])
        )
    image_paths = {
        asset["original_filename"].casefold(): asset["artifact_path"] for asset in assets
        if asset.get("artifact_path")
    }
    markdown = _rewrite_provider_image_refs(original_markdown, image_paths)
    image_items = _image_items(result)
    for asset in assets:
        asset["items"] = _match_image_items(asset["metadata"], image_items, len(assets))
        asset["label"] = str(asset.get("category") or "image")

    source_info: dict[str, Any] = {key: value for key, value in (source or {}).items() if value}
    job_value = result.get("job")
    if isinstance(job_value, dict):
        for key in ("source_url", "source_uri", "file_id"):
            if key not in source_info and job_value.get(key):
                source_info[key] = job_value[key]
    source_url = source_info.get("source_url") or source_info.get("source_uri")
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        source_url = None
    markdown_pages = result.get("markdown")
    successful_pages = [
        page.get("page_number")
        for page in (markdown_pages.get("pages", []) if isinstance(markdown_pages, dict) else [])
        if isinstance(page, dict)
        and page.get("success") is not False
        and page.get("page_number") is not None
    ]
    if len(successful_pages) == 1:
        # A single successful page is a safe document-level inference. For a
        # multi-page document we never guess a page from array position.
        for asset in assets:
            if asset["metadata"].get("page_number") is None:
                asset["metadata"]["page_number"] = successful_pages[0]
    markdown, warnings = _insert_missing_occurrences(markdown, assets, source_url)
    warnings.extend(download_warnings)

    provenance_assets: list[dict[str, Any]] = []
    occurrence_cursor_by_path: dict[str, int] = {}
    all_occurrences_by_path = {
        path: _markdown_occurrences(markdown, path)
        for path in {asset["artifact_path"] for asset in assets if asset.get("artifact_path")}
    }
    for asset in assets:
        artifact_path = asset.get("artifact_path")
        all_occurrences = all_occurrences_by_path.get(artifact_path, [])
        occurrence_cursor = occurrence_cursor_by_path.get(artifact_path, 0)
        occurrence_count = int(asset.get("original_occurrence_count", 0))
        if asset.get("inserted") and occurrence_count == 0:
            occurrence_count = 1
        occurrences = all_occurrences[occurrence_cursor : occurrence_cursor + occurrence_count]
        if artifact_path:
            occurrence_cursor_by_path[artifact_path] = occurrence_cursor + occurrence_count
        item_occurrences = [
            {
                "status": "inserted" if asset.get("inserted") else "linked",
                "markdown_path": "content.md",
                "markdown_occurrence": occurrence,
                "source_position": _asset_position(
                    asset["metadata"], asset["items"][min(index, len(asset["items"]) - 1)]
                ),
            }
            for index, occurrence in enumerate(occurrences)
            if asset["items"]
        ]
        if not item_occurrences and occurrences:
            item_occurrences = [
                {
                    "status": "inserted" if asset.get("inserted") else "linked",
                    "markdown_path": "content.md",
                    "markdown_occurrence": occurrence,
                    "source_position": _asset_position(asset["metadata"], None),
                }
                for occurrence in occurrences
            ]
        if not item_occurrences and asset["items"]:
            item_occurrences = [
                {
                    "status": "requires_review",
                    "markdown_path": "content.md",
                    "source_position": _asset_position(asset["metadata"], item),
                }
                for item in asset["items"]
            ]
        if not item_occurrences:
            item_occurrences = [
                {
                    "status": "requires_review",
                    "markdown_path": "content.md",
                    "source_position": _asset_position(asset["metadata"], None),
                }
            ]
        provenance_assets.append(
            {
                "original_filename": asset["original_filename"],
                "artifact_path": asset["artifact_path"],
                "sha256": asset.get("sha256"),
                "category": asset.get("category"),
                "content_type": asset.get("content_type"),
                "bbox": asset["metadata"].get("bbox"),
                "download_error": asset.get("download_error"),
                "source_occurrences": item_occurrences,
                # Filled by post-commit image URI rewriting. It is deliberately
                # not a presigned provider URL.
                "final_ov_image_uri": None,
            }
        )

    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "source": source_info,
        "assets": provenance_assets,
        "warnings": warnings,
        "requires_review": bool(warnings),
    }
    return await asyncio.to_thread(_make_zip, markdown, images, manifest)


def _source(payload: dict[str, Any]) -> tuple[str, str]:
    inputs = payload.get("input")
    content = (
        inputs[0].get("content")
        if isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], dict)
        else None
    )
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise BridgeError(400, "invalid_request", "input must contain one content item")
    item = content[0]
    if "lark_file" in item:
        raise BridgeError(400, "unsupported_input", "credential-gated URLs are not supported")
    if item.get("type") == "file":
        file = item.get("file")
        file_id = file.get("file_id") if isinstance(file, dict) else None
        if isinstance(file_id, str) and file_id.strip():
            return "file_id", file_id.strip()
        raise BridgeError(400, "invalid_request", "file.file_id is required")
    content_type = item.get("type")
    url_fields = {
        "input_file": "file_url",
        "input_image": "image_url",
        "input_audio": "audio_url",
    }
    url_field = url_fields.get(content_type) if isinstance(content_type, str) else None
    if url_field:
        url = item.get(url_field)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return "source_url", url.strip()
        raise BridgeError(400, "invalid_request", f"{url_field} must be an HTTP URL")
    raise BridgeError(400, "unsupported_input", f"unsupported input type: {content_type}")


class Bridge:
    def __init__(self, settings: Settings, client: LlamaParseClient, owns_client: bool):
        self.settings = settings
        self.client = client
        self.owns_client = owns_client
        self.signer = ArtifactSigner(settings.bridge_api_key, settings.artifact_ttl_seconds)
        self.artifacts: dict[str, tuple[float, bytes]] = {}
        self.builds: dict[str, asyncio.Task[None]] = {}
        self.file_sources: dict[str, dict[str, Any]] = {}
        self.job_sources: dict[str, dict[str, Any]] = {}
        self.build_slots = asyncio.Semaphore(MAX_CONCURRENT_ARTIFACT_BUILDS)

    def _store_artifact(self, job_id: str, content: bytes) -> None:
        if len(self.artifacts) >= MAX_CACHED_ARTIFACTS:
            self.artifacts.pop(min(self.artifacts, key=lambda key: self.artifacts[key][0]))
        self.artifacts[job_id] = (time.monotonic() + self.settings.artifact_ttl_seconds, content)

    async def _build_artifact(self, job_id: str) -> None:
        async with self.build_slots:
            content = await build_artifact(self.client, job_id, source=self.job_sources.get(job_id))
        self._store_artifact(job_id, content)

    def _build_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is None and self.builds.get(job_id) is task:
            self.builds.pop(job_id, None)

    def _prune(self) -> None:
        now = time.monotonic()
        for job_id in [key for key, (expires, _) in self.artifacts.items() if expires <= now]:
            self.artifacts.pop(job_id)

    async def _artifact_ready(self, job_id: str) -> bool:
        self._prune()
        if job_id in self.artifacts:
            _, content = self.artifacts[job_id]
            self.artifacts[job_id] = (
                time.monotonic() + self.settings.artifact_ttl_seconds,
                content,
            )
            return True
        task = self.builds.get(job_id)
        if task is None:
            task = asyncio.create_task(self._build_artifact(job_id))
            task.add_done_callback(lambda done: self._build_done(job_id, done))
            self.builds[job_id] = task
            return False
        if not task.done():
            return False
        self.builds.pop(job_id, None)
        await task
        return job_id in self.artifacts

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for task in self.builds.values():
                task.cancel()
            await asyncio.gather(*self.builds.values(), return_exceptions=True)
            if self.owns_client:
                await self.client.aclose()

    async def require_auth(self, authorization: Optional[str] = Header(default=None)) -> None:
        expected = f"Bearer {self.settings.bridge_api_key}".encode()
        if not hmac.compare_digest((authorization or "").encode(), expected):
            raise BridgeError(401, "unauthorized", "invalid bridge API key")

    async def create_file(
        self, file: UploadFile, purpose: str = Form(default="user_data")
    ) -> dict[str, Any]:
        del purpose
        filename = PurePosixPath((file.filename or "upload").replace("\\", "/")).name
        if filename in {"", ".", ".."}:
            filename = "upload"
        result = await self.client.upload_file(filename, file.file, file.content_type)
        if not isinstance(result.get("id"), str) or not result["id"]:
            raise LlamaParseError(502, "LlamaParse file response has no id")
        self.file_sources[result["id"]] = {"file_id": result["id"], "filename": filename}
        return result

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_name, value = _source(payload)
        result = await self.client.create_job(
            file_id=value if source_name == "file_id" else None,
            source_url=value if source_name == "source_url" else None,
        )
        response_id = result.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise LlamaParseError(502, "LlamaParse create response has no id")
        self.job_sources[response_id] = (
            {"source_url": value} if source_name == "source_url" else self.file_sources.get(value, {"file_id": value})
        )
        return {"id": response_id, "object": "response", "status": "in_progress"}

    async def get_response(self, response_id: str) -> dict[str, Any]:
        result = await self.client.get_job(response_id)
        job = result.get("job")
        if not isinstance(job, dict):
            raise LlamaParseError(502, "LlamaParse response has no job object")
        status = str(job.get("status", "")).upper()
        response: dict[str, Any] = {"id": response_id, "object": "response"}
        if status == "COMPLETED":
            if await self._artifact_ready(response_id):
                response.update(
                    status="completed",
                    result={
                        "zip_url": self.signer.create_url(self.settings.public_url, response_id)
                    },
                )
            else:
                response["status"] = "in_progress"
        elif status in {"FAILED", "CANCELLED"}:
            message = str(job.get("error_message") or f"LlamaParse job ended as {status}")
            response.update(
                status="failed",
                output=[{"content": [{"type": "output_text", "text": message}]}],
            )
        elif status in {"PENDING", "RUNNING"}:
            response["status"] = "in_progress"
        else:
            raise LlamaParseError(502, f"LlamaParse returned unknown job status: {status}")
        return response

    async def get_artifact(
        self, job_id: str, expires: str = Query(...), signature: str = Query(...)
    ) -> Response:
        if not self.signer.verify(job_id, expires, signature):
            raise BridgeError(
                403, "invalid_artifact_signature", "artifact URL is invalid or expired"
            )
        self._prune()
        if job_id not in self.artifacts:
            raise BridgeError(404, "artifact_not_found", "artifact is not ready or has expired")
        return Response(content=self.artifacts[job_id][1], media_type="application/zip")


async def _bridge_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, BridgeError)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": str(error)}},
    )


async def _llama_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, LlamaParseError)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": f"llamaparse_http_{error.status_code}", "message": str(error)}},
    )


def create_app(
    settings: Optional[Settings] = None, client: Optional[LlamaParseClient] = None
) -> FastAPI:
    resolved = settings or load_settings()
    bridge = Bridge(resolved, client or LlamaParseClient(resolved), client is None)
    app = FastAPI(title="OpenViking LlamaParse bridge", lifespan=bridge.lifespan)
    app.add_exception_handler(BridgeError, _bridge_error)
    app.add_exception_handler(LlamaParseError, _llama_error)
    auth = [Depends(bridge.require_auth)]
    app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    app.add_api_route("/api/v3/files", bridge.create_file, methods=["POST"], dependencies=auth)
    app.add_api_route(
        "/api/v3/responses", bridge.create_response, methods=["POST"], dependencies=auth
    )
    app.add_api_route(
        "/api/v3/responses/{response_id}",
        bridge.get_response,
        methods=["GET"],
        dependencies=auth,
    )
    app.add_api_route("/artifacts/{job_id}.zip", bridge.get_artifact, methods=["GET"])
    return app
