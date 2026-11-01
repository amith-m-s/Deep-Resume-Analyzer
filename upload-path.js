const path = require("node:path");
const { randomUUID } = require("node:crypto");

function createUploadPath(uploadDir) {
  if (typeof uploadDir !== "string" || !path.isAbsolute(uploadDir)) {
    throw new TypeError("Upload directory must be an absolute path");
  }

  return path.join(uploadDir, `${randomUUID()}.pdf`);
}

module.exports = { createUploadPath };
