const PDF_HEADER = Buffer.from("%PDF-");
const PDF_HEADER_SEARCH_BYTES = 1024;

/**
 * Check the PDF magic header before handing an uploaded file to the parser.
 * PDF readers commonly tolerate the header appearing within the first 1024 bytes.
 */
function hasPdfSignature(value) {
  if (!Buffer.isBuffer(value)) return false;
  return value.subarray(0, PDF_HEADER_SEARCH_BYTES).indexOf(PDF_HEADER) !== -1;
}

module.exports = { hasPdfSignature };
