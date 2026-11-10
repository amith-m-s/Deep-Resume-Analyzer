import json


def process_message(raw, analyze):
    """Validate one newline-delimited worker request and return its response."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {"id": None, "ok": False, "error": "Invalid JSON message"}

    if not isinstance(payload, dict):
        return {"id": None, "ok": False, "error": "Message must be a JSON object"}

    request_id = payload.get("id")
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
        return {
            "id": None,
            "ok": False,
            "error": "Message id must be a non-empty string of at most 128 characters",
        }

    resume_text = payload.get("resume_text", "")
    job_description = payload.get("job_description", "")
    if not isinstance(resume_text, str) or not isinstance(job_description, str):
        return {
            "id": request_id,
            "ok": False,
            "error": "resume_text and job_description must be strings",
        }

    try:
        result = analyze(resume_text, job_description)
    except Exception as exc:
        return {"id": request_id, "ok": False, "error": str(exc)}

    return {"id": request_id, "ok": True, "result": result}
