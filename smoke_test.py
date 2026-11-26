#!/usr/bin/env python3
"""Read-only smoke tests for a deployed Advanced RAG Pipeline instance.

Usage:
  python smoke_test.py --base-url https://your-app.up.railway.app

Optional HTTP Basic Auth:
  APP_USERNAME=... APP_PASSWORD=... python smoke_test.py --base-url ...

The script checks health, recovered document downloads, the ZIP archive,
SSE completion, source metadata, and a small set of corpus-grounded questions.
It never uploads, deletes, or re-ingests documents.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
import zipfile
from io import BytesIO


class SmokeTestError(RuntimeError):
    pass


class SmokeTester:
    def __init__(self, base_url: str, timeout: int = 90):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.passed = 0
        self.failed = 0
        self.auth_header = None
        username = os.getenv("APP_USERNAME")
        password = os.getenv("APP_PASSWORD")
        if username and password:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self.auth_header = f"Basic {token}"

    def request(self, path: str, method: str = "GET", body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "*/*"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        req = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SmokeTestError(f"{method} {path}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SmokeTestError(f"{method} {path}: {exc}") from exc

    def check(self, label: str, callback):
        try:
            detail = callback()
            self.passed += 1
            print(f"PASS  {label}" + (f" — {detail}" if detail else ""))
            return True
        except Exception as exc:
            self.failed += 1
            print(f"FAIL  {label} — {exc}")
            return False

    def query(self, question: str, conversation_id: str | None = None):
        status, _, raw = self.request(
            "/api/query",
            method="POST",
            body={"question": question, "conversation_id": conversation_id},
        )
        if status != 200:
            raise SmokeTestError(f"query returned HTTP {status}")
        answer_parts = []
        corrected_answer = None
        events = []
        sources = []
        meta = {}
        current_line = ""
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            events.append(event_type)
            if event_type == "token":
                answer_parts.append(event.get("token", ""))
            elif event_type == "correction":
                corrected_answer = event.get("corrected_answer", "")
            elif event_type == "sources":
                sources = event.get("sources", [])
            elif event_type == "meta":
                meta = event
            elif event_type == "error":
                raise SmokeTestError(
                    f"pipeline error at {event.get('stage')}: {event.get('message')}"
                )
        answer = corrected_answer if corrected_answer is not None else "".join(answer_parts)
        answer = answer.strip()
        if "done" not in events:
            raise SmokeTestError(f"stream did not complete; events={events}")
        if not answer:
            raise SmokeTestError("completed stream contained no answer")
        return {
            "answer": answer,
            "sources": sources,
            "meta": meta,
            "events": events,
        }

    def run(self):
        print(f"Testing {self.base_url}\n")

        status_data = {}

        def health():
            nonlocal status_data
            _, _, raw = self.request("/api/status")
            status_data = json.loads(raw)
            if not status_data.get("ready"):
                raise SmokeTestError(f"app is not ready: {status_data}")
            if not status_data.get("chunks"):
                raise SmokeTestError("ready app reports zero indexed chunks")
            return f"{status_data.get('chunks')} chunks; {status_data.get('documents')} documents"

        self.check("Application health and loaded index", health)

        diagnostics = {}

        def snapshot():
            nonlocal diagnostics
            _, _, raw = self.request("/api/documents/diagnostics")
            diagnostics = json.loads(raw)
            if not diagnostics.get("snapshot_exists"):
                raise SmokeTestError("nodes.pkl snapshot is missing")
            if diagnostics.get("documents_with_recoverable_text", 0) < 1:
                raise SmokeTestError("no documents have recoverable text")
            if diagnostics.get("chunks_with_text", 0) < 1:
                raise SmokeTestError("no text chunks were recovered")
            return (
                f"{diagnostics.get('documents_with_recoverable_text')} recoverable documents; "
                f"{diagnostics.get('chunks_with_text')} text chunks"
            )

        self.check("Persistent snapshot and recoverable text", snapshot)

        documents = []

        def library_and_downloads():
            nonlocal documents
            _, _, raw = self.request("/api/documents")
            documents = json.loads(raw)
            if not documents:
                raise SmokeTestError("document library is empty")
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    raise SmokeTestError("document record has no ID")
                _, headers, content = self.request(
                    f"/api/documents/{urllib.parse.quote(doc_id, safe='')}/download"
                )
                if not content.strip():
                    raise SmokeTestError(f"empty download for {doc.get('name')}")
                if "attachment" not in headers.get("Content-Disposition", "").lower():
                    raise SmokeTestError(f"download is not marked as an attachment: {doc.get('name')}")
            return f"{len(documents)} individual document downloads readable"

        self.check("Document library and individual downloads", library_and_downloads)

        def archive():
            _, headers, raw = self.request("/api/documents/archive/download-all")
            if "application/zip" not in headers.get("Content-Type", ""):
                raise SmokeTestError(f"unexpected archive content type: {headers.get('Content-Type')}")
            with zipfile.ZipFile(BytesIO(raw)) as bundle:
                names = bundle.namelist()
                if len(names) != len(documents):
                    raise SmokeTestError(f"ZIP contains {len(names)} files; library lists {len(documents)}")
                for name in names:
                    if not bundle.read(name).strip():
                        raise SmokeTestError(f"empty file in archive: {name}")
            return f"{len(names)} non-empty files in ZIP"

        if documents:
            self.check("Bulk ZIP archive integrity", archive)

        convo_id = str(uuid.uuid4())

        def grounded_ingestion():
            result = self.query("What does the ingestion process create?", convo_id)
            answer = result["answer"].lower()
            if "nodes.pkl" not in answer or "qdrant" not in answer:
                raise SmokeTestError(f"expected nodes.pkl and Qdrant in answer; got: {result['answer']}")
            if not result["sources"]:
                raise SmokeTestError("answer has no source attribution")
            return f"grounded answer; sources={len(result['sources'])}"

        self.check("Grounded answer: ingestion artifacts", grounded_ingestion)

        def grounded_credentials():
            result = self.query("Which credentials are used for reranking and web fallback?", convo_id)
            answer = result["answer"].lower()
            if "cohere" not in answer or "tavily" not in answer:
                raise SmokeTestError(f"expected Cohere and Tavily; got: {result['answer']}")
            return f"answer references Cohere and Tavily; sources={len(result['sources'])}"

        self.check("Grounded answer: provider credentials", grounded_credentials)

        def multi_user_boundary():
            result = self.query("Is the hosted demo multi-user ready?", convo_id)
            answer = result["answer"].lower()
            if not any(term in answer for term in ("no", "not multi-user", "shared login", "individual accounts")):
                raise SmokeTestError(f"answer did not explain the shared-login limitation: {result['answer']}")
            return "shared-login limitation surfaced"

        self.check("Grounded answer: demo access limitation", multi_user_boundary)

        def unsupported_question():
            result = self.query("What is the capital of France?", convo_id)
            answer = result["answer"].lower()
            refusal_terms = (
                "do not contain", "does not contain", "not provided",
                "not available in", "not enough information", "cannot be found",
            )
            if not any(term in answer for term in refusal_terms):
                raise SmokeTestError(f"expected an explicit unsupported-answer refusal; got: {result['answer']}")
            return "unsupported question refused"

        self.check("Out-of-corpus refusal", unsupported_question)

        print(f"\nResult: {self.passed} passed, {self.failed} failed")
        return 0 if self.failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Read-only hosted RAG smoke tests")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. https://your-app.up.railway.app")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    tester = SmokeTester(args.base_url, timeout=args.timeout)
    sys.exit(tester.run())


if __name__ == "__main__":
    main()
