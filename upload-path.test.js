const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { createUploadPath } = require("./upload-path");

const uploadDir = path.resolve("test-uploads");
const UUID_PDF = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\\.pdf$/i;

test("creates a PDF path inside the configured upload directory", () => {
  const uploadPath = createUploadPath(uploadDir);

  assert.equal(path.dirname(uploadPath), uploadDir);
  assert.match(path.basename(uploadPath), UUID_PDF);
});

test("generates a distinct server-side path for each upload", () => {
  const first = createUploadPath(uploadDir);
  const second = createUploadPath(uploadDir);

  assert.notEqual(first, second);
});

test("rejects a relative upload directory", () => {
  assert.throws(() => createUploadPath("uploads"), {
    name: "TypeError",
    message: "Upload directory must be an absolute path"
  });
});

test("rejects missing or non-string upload directories", () => {
  assert.throws(() => createUploadPath(undefined), TypeError);
  assert.throws(() => createUploadPath(null), TypeError);
});
