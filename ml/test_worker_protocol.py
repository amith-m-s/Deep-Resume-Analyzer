import json
import unittest

from worker_protocol import process_message


class WorkerProtocolTests(unittest.TestCase):
    def test_valid_request_returns_analysis_with_matching_id(self):
        calls = []

        def analyze(resume, job):
            calls.append((resume, job))
            return {"semanticScore": 81.5}

        response = process_message(
            json.dumps(
                {
                    "id": "request-1",
                    "resume_text": "Python",
                    "job_description": "Backend",
                }
            ),
            analyze,
        )

        self.assertEqual(calls, [("Python", "Backend")])
        self.assertEqual(
            response,
            {
                "id": "request-1",
                "ok": True,
                "result": {"semanticScore": 81.5},
            },
        )

    def test_invalid_json_returns_error_without_calling_analyzer(self):
        response = process_message(
            "{not-json", lambda *_: self.fail("analyzer must not run")
        )

        self.assertIsNone(response["id"])
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "Invalid JSON message")

    def test_non_object_json_values_are_rejected_without_crashing(self):
        for payload in (None, [], ["x"], 42, True, "text"):
            with self.subTest(payload=payload):
                response = process_message(
                    json.dumps(payload),
                    lambda *_: self.fail("analyzer must not run"),
                )

                self.assertIsNone(response["id"])
                self.assertFalse(response["ok"])
                self.assertEqual(
                    response["error"], "Message must be a JSON object"
                )

    def test_invalid_message_does_not_reuse_a_previous_request_id(self):
        analyze = lambda *_: {"ok": True}
        first = process_message(
            json.dumps(
                {
                    "id": "previous",
                    "resume_text": "",
                    "job_description": "",
                }
            ),
            analyze,
        )
        second = process_message("{broken", analyze)

        self.assertEqual(first["id"], "previous")
        self.assertIsNone(second["id"])
        self.assertEqual(second["error"], "Invalid JSON message")

    def test_missing_or_invalid_request_ids_are_rejected(self):
        for request_id in (None, "", "   ", 123, "x" * 129):
            with self.subTest(request_id=request_id):
                response = process_message(
                    json.dumps({"id": request_id}),
                    lambda *_: self.fail("analyzer must not run"),
                )

                self.assertIsNone(response["id"])
                self.assertFalse(response["ok"])
                self.assertIn("Message id must be", response["error"])

    def test_non_string_text_fields_are_rejected(self):
        response = process_message(
            json.dumps(
                {
                    "id": "request-2",
                    "resume_text": ["Python"],
                    "job_description": "Backend",
                }
            ),
            lambda *_: self.fail("analyzer must not run"),
        )

        self.assertEqual(response["id"], "request-2")
        self.assertFalse(response["ok"])
        self.assertIn("must be strings", response["error"])

    def test_missing_text_fields_default_to_empty_strings(self):
        calls = []
        response = process_message(
            json.dumps({"id": "request-3"}),
            lambda *args: calls.append(args) or {},
        )

        self.assertEqual(calls, [("", "")])
        self.assertTrue(response["ok"])

    def test_analyzer_exceptions_become_correlated_error_responses(self):
        def fail(*_):
            raise RuntimeError("model unavailable")

        response = process_message(
            json.dumps(
                {
                    "id": "request-4",
                    "resume_text": "resume",
                    "job_description": "job",
                }
            ),
            fail,
        )

        self.assertEqual(
            response,
            {"id": "request-4", "ok": False, "error": "model unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
