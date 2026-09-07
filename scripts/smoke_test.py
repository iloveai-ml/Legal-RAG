"""
End to end smoke test.

Exercises every route against a running server, including the paths that only
show up in production: an upload, a question answered against that upload, the
no key retrieval only fallback, and the house punctuation rule.

Usage:
    python scripts/smoke_test.py [--base http://127.0.0.1:8077]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import uuid

BANNED = {
    "—": "em dash",
    "–": "en dash",
    "�": "replacement character",
    " ": "narrow no break space",
    "‑": "non breaking hyphen",
}

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def request(base: str, path: str, *, method: str = "GET", payload=None, headers=None):
    url = f"{base}{path}"
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body[:200]}


def upload(base: str, filename: str, content: bytes):
    boundary = f"----nyaya{uuid.uuid4().hex}"
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    buf.write(b"Content-Type: text/plain\r\n\r\n")
    buf.write(content)
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{base}/api/upload",
        data=buf.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode()[:200]}


def scan_punctuation(label: str, blob) -> None:
    text = json.dumps(blob, ensure_ascii=False)
    hits = [name for ch, name in BANNED.items() if ch in text]
    check(f"{label} uses house punctuation", not hits, ", ".join(hits) or "clean")


SAMPLE_AGREEMENT = """CONSULTANCY AGREEMENT

1. Term and Termination
1.1 This Agreement commences on 1 April 2026 and continues for twelve months
unless terminated earlier in accordance with this clause.
1.2 Either party may terminate this Agreement for convenience on ninety days
written notice to the other party.
1.3 The Company may terminate this Agreement with immediate effect if the
Consultant commits a material breach that is not remedied within fifteen days
of written notice.

2. Fees and Payment
2.1 The Company shall pay the Consultant a fee of Rupees Four Lakh per month.
2.2 Invoices are payable within forty five days of receipt. Interest accrues on
overdue amounts at eighteen percent per annum.

3. Confidentiality
3.1 The Consultant shall not disclose Confidential Information to any third
party during the Term or for a period of five years after termination.

4. Governing Law and Dispute Resolution
4.1 This Agreement is governed by the laws of India.
4.2 Any dispute shall be referred to arbitration seated at Mumbai under the
Arbitration and Conciliation Act, 1996. The tribunal shall consist of a sole
arbitrator appointed jointly by the parties.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8077")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"\nNyaya smoke test against {base}\n")

    # ---------------------------------------------------------- health
    print("Health and status")
    status_code, health = request(base, "/api/health")
    check("health responds 200", status_code == 200, str(status_code))
    check("index is loaded", health.get("index_ready") is True)
    check("corpus is populated", health.get("chunks", 0) > 1000, f"{health.get('chunks')} chunks")

    status_code, status = request(base, "/api/status")
    check("status responds 200", status_code == 200)
    # Assert the providers by name rather than by count, so adding a fifth
    # does not fail a test that has nothing to do with the change.
    expected_providers = {"groq", "gemini", "cloudflare", "openai", "huggingface"}
    listed = {p.get("id") for p in status.get("providers", [])}
    check(
        "providers are enumerated",
        expected_providers <= listed,
        ", ".join(sorted(listed)) or "none",
    )
    configured = [p["id"] for p in status.get("providers", []) if p["configured"]]
    print(f"        configured providers: {configured or 'none'}")

    # ---------------------------------------------------------- search
    print("\nRetrieval, no model involved")
    status_code, search = request(
        base, "/api/search", method="POST",
        payload={"query": "right to privacy fundamental right", "top_k": 6},
    )
    check("search responds 200", status_code == 200)
    sources = search.get("sources", [])
    check("search returns passages", len(sources) > 0, f"{len(sources)} passages")
    check("search is fast", search.get("elapsed_ms", 9999) < 2000, f"{search.get('elapsed_ms')}ms")
    check(
        "privacy query surfaces Puttaswamy",
        any("puttaswamy" in s["doc_title"].lower() for s in sources),
        sources[0]["doc_title"][:48] if sources else "no results",
    )
    scan_punctuation("corpus text", search)

    # a filter the user set explicitly must actually restrict results
    status_code, filtered = request(
        base, "/api/search", method="POST",
        payload={"query": "fundamental rights", "doc_types": ["constitution"], "top_k": 6},
    )
    kinds = {s["doc_type"] for s in filtered.get("sources", [])}
    check("explicit doc type filter is honoured", kinds <= {"constitution"}, str(kinds))

    # ---------------------------------------------------------- ask
    print("\nQuestion answering")
    status_code, answer = request(
        base, "/api/ask", method="POST",
        payload={"question": "Is the right to privacy a fundamental right under the Indian Constitution?"},
    )
    check("ask responds 200", status_code == 200)
    check("an answer came back", len(answer.get("answer", "")) > 40)
    check("sources are attached", len(answer.get("sources", [])) > 0)

    if answer.get("mode") == "synthesised":
        inline = answer["answer"].count("[S")
        check("answer carries inline citation tags", inline > 0, f"{inline} tags")
        check(
            "citations verified against retrieved passages",
            answer.get("unverified_count", 0) == 0,
            f"{answer.get('verified_count')} verified, {answer.get('unverified_count')} not",
        )
        print(f"        provider: {answer.get('provider')} / {answer.get('model')}, {answer.get('elapsed_ms')}ms")
    else:
        print(f"        retrieval only mode, no provider reachable")
    scan_punctuation("answer", answer)

    # out of scope questions should not retrieve
    status_code, off = request(
        base, "/api/ask", method="POST",
        payload={"question": "What is the best pizza topping?"},
    )
    check(
        "an out of scope question is refused rather than answered",
        off.get("intent") == "out_of_scope" or not off.get("sources"),
        off.get("intent", "?"),
    )

    # ---------------------------------------------------------- upload
    print("\nDocument upload")
    status_code, doc = upload(base, "consultancy_agreement.txt", SAMPLE_AGREEMENT.encode())
    check("upload accepted", status_code == 200, doc.get("detail", "")[:80])

    if status_code == 200:
        doc_id = doc["doc_id"]
        check("document was chunked", doc.get("n_chunks", 0) > 0, f"{doc['n_chunks']} passages")

        status_code, grounded = request(
            base, "/api/ask", method="POST",
            payload={
                "question": "What notice period applies if a party terminates this agreement for convenience?",
                "upload_ids": [doc_id],
            },
        )
        check("question against the upload responds 200", status_code == 200)
        from_upload = [s for s in grounded.get("sources", []) if s["origin"] == "upload"]
        check("the uploaded document is retrieved", len(from_upload) > 0, f"{len(from_upload)} passages")
        if grounded.get("mode") == "synthesised":
            check(
                "the answer finds the ninety day notice period",
                "ninety" in grounded["answer"].lower() or "90" in grounded["answer"],
                grounded["answer"][:90].replace("\n", " "),
            )
        scan_punctuation("upload answer", grounded)

        status_code, removed = request(base, f"/api/upload/{doc_id}", method="DELETE")
        check("upload can be removed", removed.get("removed") is True)

    status_code, bad = upload(base, "notes.exe", b"binary junk")
    check("unsupported file type is rejected", status_code == 400, str(status_code))

    # ---------------------------------------------------------- outcomes
    print("\nOutcome analysis")
    status_code, outcome = request(
        base, "/api/outcome", method="POST",
        payload={
            "case_description": (
                "The accused is charged under Section 302 of the Indian Penal Code for a "
                "murder arising out of a sudden quarrel. There were no eyewitnesses and the "
                "prosecution relies on circumstantial evidence and a recovery made at the "
                "instance of the accused."
            ),
            "task": "cjpe",
            "max_cases": 8,
        },
    )
    check("outcome responds 200", status_code == 200)
    total = outcome.get("favorable_count", 0) + outcome.get("unfavorable_count", 0)
    check("comparable matters were found", total > 0, f"{total} cases")
    check("every case carries a recorded outcome", all(
        c["outcome"] in (0, 1) for c in outcome.get("similar_cases", [])
    ))
    check(
        "the split is reported honestly",
        outcome.get("favorable_pct", -1) >= 0
        and abs(outcome["favorable_pct"] - (100 * outcome["favorable_count"] / total if total else 0)) < 1,
    )
    check("caveats are always present", len(outcome.get("caveats", [])) > 0)
    scan_punctuation("outcome", outcome)

    # ---------------------------------------------------------- validation
    print("\nInput validation")
    status_code, _ = request(base, "/api/ask", method="POST", payload={"question": ""})
    check("an empty question is rejected", status_code == 422, str(status_code))

    status_code, _ = request(
        base, "/api/outcome", method="POST", payload={"case_description": "too short"}
    )
    check("a too short case description is rejected", status_code == 422, str(status_code))

    status_code, _ = request(
        base, "/api/ask", method="POST",
        payload={"question": "test", "top_k": 500},
    )
    check("an out of range top_k is rejected", status_code == 422, str(status_code))

    # ---------------------------------------------------------- static
    print("\nStatic assets")
    for path in ("/", "/static/app.css", "/static/app.js", "/favicon.svg"):
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=20) as resp:
                check(f"{path} serves", resp.status == 200)
        except Exception as exc:  # noqa: BLE001
            check(f"{path} serves", False, str(exc)[:60])

    print(f"\n{passed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
