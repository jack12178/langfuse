//! Remove large inline media before ingestion parses and normalizes the JSON.
//!
//! Each extracted attachment becomes a temporary reference in the JSON. Its removed
//! source text is retained for upload or restoration later.
//! For example (reference abbreviated):
//!
//! ```text
//! {"image":"data:image/png;base64,..."} -> {"image":"<media-ref>"} + retained media
//! ```
//!
//! Supported forms, in the payload locations selected by `rules`:
//! - Data URIs in string values, including URIs surrounded by ordinary text.
//!   The URI supplies its own MIME type; no provider object is needed.
//! - Recognized provider objects containing raw base64 or Python bytes literals.
//!   Sibling fields identify the body and MIME type (see `provider_shapes`).
//!
//! Both forms can also occur in JSON stored inside strings, within `scanner`'s
//! nesting limits. Beyond those limits, only unambiguous Data URI text is extracted.
//! Arbitrary base64-looking strings outside recognized provider objects stay inline,
//! as do candidates smaller than `rules::MIN_EARLY_MEDIA_BYTES`.
//!
//! The two entry points separate validation from extraction so masking can run in
//! between. Changed masking output must be validated again before extraction.
//!
//! ```text
//! validate(bytes)
//!   `-- json::validate_json          check syntax with jiter; retain owned bytes
//!        -> ValidatedPayload        checked JSON, not a parsed object tree
//!
//! ValidatedPayload::compact()
//!   |-- scanner::discover
//!   |    |-- structural_walk::scan   find strings that could contain media
//!   |    |    -> Candidate           byte range of one quoted JSON string
//!   |    `-- inspect those strings  decode with jiter; recognize and hash media
//!   |         -> MediaManifest      edit plan: ranges to replace + media metadata
//!   `-- payload::apply_edit_plan     replace those ranges with temporary references
//!        -> EarlyMediaResult        smaller JSON + extracted media records
//! ```
//!
//! Validation and discovery each use a fresh jiter cursor; neither builds a JSON
//! tree. Discovery keeps the field context needed to select media while jiter
//! advances through tokens. Borrowed serde spans handle tokens jiter cannot read.
//! `validate` takes ownership of the input buffer. Cursors and discovery borrow it;
//! `compact` consumes it and returns owned compact JSON plus retained media. Media
//! ranges keep the source alive through `Arc`, or own a detached copy (see `payload`).
//! `otel_input` owns that result behind N-API handles; consumers perform uploads later.

use std::collections::HashSet;

mod data_uri;
mod encoding;
mod json;
mod json_cursor;
mod legacy;
mod normalized;
mod payload;
mod provider_shapes;
mod rules;
mod scanner;
mod source_text;
mod structural_walk;

/// Test counters for traversal scaling, not production telemetry. Outside tests,
/// there are no fields and `add_bytes` does nothing. Tests use these counts to detect
/// repeated traversal and growing index allocations without timing assertions.
#[derive(Default)]
pub(super) struct WalkStats {
    // Instrumented cursor advances and marker scans, not every read inside libraries.
    #[cfg(test)]
    pub(super) bytes_walked: usize,
    // Stack/candidate-vector capacities, not total heap use or RSS. Retained
    // media has separate accounting.
    #[cfg(test)]
    pub(super) peak_index_bytes: usize,
}

impl WalkStats {
    #[inline(always)]
    pub(super) fn add_bytes(&mut self, bytes: usize) {
        #[cfg(test)]
        {
            self.bytes_walked = self.bytes_walked.saturating_add(bytes);
        }
        #[cfg(not(test))]
        {
            let _ = bytes;
        }
    }

    #[cfg(test)]
    pub(super) fn update_peak(&mut self, bytes: usize) {
        self.peak_index_bytes = self.peak_index_bytes.max(bytes);
    }
}

pub use legacy::MediaReferenceCollector;
pub use normalized::extract_normalized_media;
pub use payload::EarlyMediaError;
pub use payload::{validate, EarlyMediaResult, ExtractedMedia, ValidatedPayload};
pub(crate) use rules::may_contain_media_candidate;
pub use source_text::rewrite_known_references;

use scanner::MAX_EMBEDDED_JSON_DEPTH;

const MAX_LEGACY_MEDIA_DEPTH: usize = 10;

/// Run extraction for a JSON value nested in an already parsed string. The
/// caller supplies the inherited string depth so the embedded-document limit
/// cannot be reset by normalization.
pub(crate) fn extract_media_at_depth(
    input: &[u8],
    embedded_depth: usize,
    min_candidate_bytes: usize,
) -> Result<EarlyMediaResult, EarlyMediaError> {
    let text = std::str::from_utf8(input).map_err(|error| EarlyMediaError::InvalidJson {
        offset: error.valid_up_to(),
        message: "invalid UTF-8",
    })?;
    json::validate_json(text)?;
    let mut manifest = scanner::discover_at_depth(text, embedded_depth, min_candidate_bytes)?;
    // The inherited depth limits discovery, but compaction removes one escape
    // layer from this source. Retained spelling and upload decoding therefore
    // count layers relative to this document, not its former outer wrappers.
    for entry in &mut manifest.entries {
        entry.original_json_depth -= embedded_depth as u8;
        entry.json_decode_layers = entry
            .json_decode_layers
            .saturating_sub(embedded_depth as u8);
    }
    payload::compact_manifest(text, manifest)
}

/// Collect temporary references from values without treating object keys as payloads.
pub fn collect_media_references(value: &str, references: &mut HashSet<String>) {
    source_text::for_each_media_reference(value, |reference| {
        references.insert(reference.to_owned());
    });
}

/// Ensure every extracted occurrence has its own temporary reference. Metadata
/// may be shared by content hash, but restoration must remain occurrence-specific.
pub fn validate_media_registry(
    existing: &[ExtractedMedia],
    additions: &[ExtractedMedia],
) -> Result<(), EarlyMediaError> {
    let mut references = HashSet::new();
    for media in existing.iter().chain(additions) {
        let reference = media.reference();
        if !references.insert(reference.clone()) {
            return Err(EarlyMediaError::UnsupportedMediaReferenceAmbiguity { reference });
        }
    }
    Ok(())
}

#[cfg(test)]
pub fn extract_media(input: &[u8]) -> Result<EarlyMediaResult, payload::EarlyMediaError> {
    validate(input.to_vec())?.compact()
}

#[cfg(test)]
mod tests;
