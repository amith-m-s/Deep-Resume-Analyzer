const test = require("node:test");
const assert = require("node:assert/strict");
const { hasPdfSignature } = require("./upload-validation");

test("accepts a PDF signature at the beginning of the file", () => {
  assert.equal(hasPdfSignature(Buffer.from("%PDF-1.7\nbody")), true);
});

test("accepts a PDF signature within the first 1024 bytes", () => {
  const prefix = Buffer.from("%PDF-1.4\n");
  const preamble = Buffer.alloc(80, 0x20);
  assert.equal(hasPdfSignature(Buffer.concat([preamble, prefix])), true);
});

test("rejects a file with a non-PDF signature even if its MIME type claims PDF", () => {
  assert.equal(hasPdfSignature(Buffer.from("<html><body>not a PDF</body></html>")), false);
});

test("rejects a PDF-looking header after the first 1024 bytes", () => {
  const preamble = Buffer.alloc(1024, 0x20);
  assert.equal(hasPdfSignature(Buffer.concat([preamble, Buffer.from("%PDF-1.7")])), false);
});

test("rejects missing, short, or non-buffer inputs", () => {
  assert.equal(hasPdfSignature(null), false);
  assert.equal(hasPdfSignature("%PDF-1.7"), false);
  assert.equal(hasPdfSignature(Buffer.from("%PDF")), false);
});
