//! Residual media extraction from the normalized Rust value.
//!
//! Most media is removed before normalization. This adapter handles provider shapes that only
//! become recognizable after OTLP attributes are projected into ordinary objects, plus embedded
//! JSON strings that survive that projection. It uses the same occurrence-specific references
//! and media encodings as the byte scanner.

use std::mem;

use crate::otel::value::{JsString, Object, Value};

use super::encoding::{is_python_bytes_literal, is_valid_base64_syntax};
use super::legacy::{legacy_structured_target, LegacyStructuredTarget};
use super::payload::{
    EarlyMediaError, ExtractedMedia, MediaEncoding, MediaPayloadKind, MediaSource,
};
use super::rules::{
    is_media_reference, is_supported_content_type, may_be_serialized_json,
    may_contain_serialized_media, DATA_URI_PREFIX,
};
use super::scanner::{find_data_uri_candidates, parse_data_uri};
use super::{extract_media_at_depth, MAX_EMBEDDED_JSON_DEPTH, MAX_LEGACY_MEDIA_DEPTH};

pub fn extract_normalized_media(
    value: &mut Value,
    registry: &mut Vec<ExtractedMedia>,
) -> Result<(), EarlyMediaError> {
    let mut extractor = NormalizedExtractor { registry };
    extractor.visit(value, 0, 0)
}

struct NormalizedExtractor<'a> {
    registry: &'a mut Vec<ExtractedMedia>,
}

impl NormalizedExtractor<'_> {
    fn visit(
        &mut self,
        value: &mut Value,
        structural_depth: usize,
        embedded_depth: usize,
    ) -> Result<(), EarlyMediaError> {
        if structural_depth >= MAX_LEGACY_MEDIA_DEPTH {
            return Ok(());
        }
        match value {
            Value::String(text) => self.visit_string(text, embedded_depth),
            Value::Array(values) => {
                for value in values {
                    self.visit(value, structural_depth + 1, embedded_depth)?;
                }
                Ok(())
            }
            Value::Object(object) => self.visit_object(object, structural_depth, embedded_depth),
            _ => Ok(()),
        }
    }

    fn visit_object(
        &mut self,
        object: &mut Object,
        structural_depth: usize,
        embedded_depth: usize,
    ) -> Result<(), EarlyMediaError> {
        // Recognized provider objects are leaves in the legacy processor. Reusing its classifier
        // prevents a data URI in an unrelated sibling from becoming an extra upload.
        if let Some(target) = legacy_structured_target(object) {
            match target {
                LegacyStructuredTarget::Direct {
                    property,
                    content_type,
                    kind,
                } => {
                    let content_type = content_type.to_owned();
                    if let Some(Value::String(text)) = object.get_mut(property) {
                        self.extract_provider_text(text, &content_type, kind)?;
                    }
                }
                LegacyStructuredTarget::Nested {
                    container,
                    property,
                    content_type,
                    kind,
                } => {
                    let content_type = content_type.to_owned();
                    if let Some(Value::Object(nested)) = object.get_mut(container) {
                        if let Some(Value::String(text)) = nested.get_mut(property) {
                            self.extract_provider_text(text, &content_type, kind)?;
                        }
                    }
                }
            }
            return Ok(());
        }

        for value in object.values_mut() {
            self.visit(value, structural_depth + 1, embedded_depth)?;
        }
        Ok(())
    }

    fn visit_string(
        &mut self,
        text: &mut JsString,
        embedded_depth: usize,
    ) -> Result<(), EarlyMediaError> {
        if is_media_reference(text.as_str()) {
            return Ok(());
        }

        if may_be_serialized_json(text.as_str())
            && text.as_utf8().is_some()
            && embedded_depth < MAX_EMBEDDED_JSON_DEPTH
            && (text.as_str().contains(DATA_URI_PREFIX)
                || text.as_str().contains("\\u")
                || may_contain_serialized_media(text.as_str()))
        {
            match extract_media_at_depth(text.as_bytes(), embedded_depth + 1, 0) {
                Ok(extracted) => {
                    if !extracted.media.is_empty() {
                        *text = extracted.compact_json.into();
                        self.registry.extend(extracted.media);
                        return Ok(());
                    }
                }
                Err(error @ EarlyMediaError::UnsupportedMediaReferenceAmbiguity { .. }) => {
                    return Err(error);
                }
                Err(_) => {
                    // Invalid or unsupported embedded JSON is ordinary text. The direct Data URI
                    // path below still covers candidates inside such strings.
                }
            }
        }

        self.extract_data_uris(text)
    }

    fn extract_provider_text(
        &mut self,
        text: &mut JsString,
        content_type: &str,
        kind: MediaPayloadKind,
    ) -> Result<(), EarlyMediaError> {
        if text.as_utf8().is_none() {
            return Ok(());
        }

        let (content_type, kind, source, encoding) = if text.as_str().starts_with(DATA_URI_PREFIX) {
            let Some(candidate) = parse_data_uri(text.as_str(), 0) else {
                return Ok(());
            };
            let Some(uri_content_type) = candidate
                .valid
                .filter(|_| candidate.start == 0 && candidate.end == text.as_str().len())
            else {
                return Ok(());
            };
            (
                uri_content_type.to_owned(),
                MediaPayloadKind::DataUri,
                MediaSource::Base64DataUri,
                MediaEncoding::Base64DataUri,
            )
        } else {
            if !is_supported_content_type(content_type) {
                return Ok(());
            }
            let (encoding, valid) = provider_encoding(text.as_str());
            if !valid {
                return Ok(());
            }
            (content_type.to_owned(), kind, MediaSource::Bytes, encoding)
        };

        let original = mem::take(text);
        let Some(media) = ExtractedMedia::from_owned(
            original.as_bytes().to_vec(),
            &content_type,
            kind,
            source,
            encoding,
        )?
        else {
            *text = original;
            return Ok(());
        };
        *text = media.reference().into();
        self.registry.push(media);
        Ok(())
    }

    fn extract_data_uris(&mut self, text: &mut JsString) -> Result<(), EarlyMediaError> {
        let candidates = find_data_uri_candidates(text.as_str());
        if candidates.is_empty() {
            return Ok(());
        }

        let mut replacements = Vec::with_capacity(candidates.len());
        for (range, content_type) in candidates {
            // An unpaired surrogate in the URI itself cannot be retained by the UTF-8
            // media registry. Keep that occurrence inline; surrounding surrogates are safe.
            let Ok(encoded) = std::str::from_utf8(&text.as_bytes()[range.clone()]) else {
                continue;
            };
            let Some(media) = ExtractedMedia::from_owned(
                encoded.as_bytes().to_vec(),
                content_type,
                MediaPayloadKind::DataUri,
                MediaSource::Base64DataUri,
                MediaEncoding::Base64DataUri,
            )?
            else {
                continue;
            };
            let reference = media.reference();
            self.registry.push(media);
            replacements.push((range, reference));
        }

        if replacements.is_empty() {
            return Ok(());
        }
        let mut output = JsString::with_capacity(text.len());
        let mut end = 0;
        for (range, replacement) in replacements {
            output.push_slice(text, end..range.start);
            output.push_str(&replacement);
            end = range.end;
        }
        output.push_slice(text, end..text.len());
        *text = output;
        Ok(())
    }
}

fn provider_encoding(value: &str) -> (MediaEncoding, bool) {
    if is_python_bytes_literal(value) {
        return (MediaEncoding::PythonBytesLiteral, true);
    }
    (
        MediaEncoding::Base64,
        is_valid_base64_syntax(value.as_bytes()),
    )
}

#[cfg(test)]
mod tests {
    use super::super::encoding::BASE64;
    use super::super::scanner::MIN_EARLY_MEDIA_BYTES;
    use super::*;
    use crate::otel::value::json;
    use base64::Engine;

    #[test]
    fn extracts_provider_media_without_serializing_the_containing_value() {
        let mut value = json!({
            "type": "base64",
            "media_type": "image/png",
            "data": BASE64.encode(b"provider"),
            "sibling": data_uri(b"sibling"),
        });
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("extract provider media");

        assert_eq!(registry.len(), 1);
        assert_eq!(registry[0].kind, MediaPayloadKind::Anthropic);
        assert_eq!(registry[0].decode().unwrap(), b"provider");
        assert!(value["data"]
            .as_str()
            .unwrap()
            .starts_with(super::super::rules::MEDIA_REFERENCE_PREFIX));
        assert_eq!(value["sibling"], Value::from(data_uri(b"sibling")));
    }

    #[test]
    fn extracts_data_uri_and_embedded_provider_media_from_owned_values() {
        let direct_body = vec![b'd'; MIN_EARLY_MEDIA_BYTES];
        let nested_body = vec![b'n'; MIN_EARLY_MEDIA_BYTES];
        let nested = format!(r#"{{"content":"{}","number":1.0}}"#, data_uri(&nested_body));
        let mut value = json!({
            "input": data_uri(&direct_body),
            "serialized": nested,
        });
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("extract normalized media");

        assert_eq!(registry.len(), 2);
        assert!(value["input"]
            .as_str()
            .unwrap()
            .starts_with(super::super::rules::MEDIA_REFERENCE_PREFIX));
        assert!(value["serialized"]
            .as_str()
            .unwrap()
            .contains("@@@langfuseMedia:type=image/png|id="));
        assert_eq!(registry[0].decode().unwrap(), direct_body);
        assert_eq!(registry[1].decode().unwrap(), nested_body);
    }

    #[test]
    fn late_normalization_extracts_small_media_from_stringified_documents() {
        let uri = data_uri(b"tiny");
        let provider_data = BASE64.encode(b"small-provider");
        let mut value = Value::String(format!(
            r#"{{"uri":"{uri}","provider":{{"type":"file","mediaType":"image/png","data":"{provider_data}"}}}}"#
        ).into());
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry)
            .expect("extract small media from stringified JSON");

        assert_eq!(registry.len(), 2);
        let compacted = value.as_str().expect("stringified document stays a string");
        assert_eq!(compacted.matches("@@@langfuseMedia:").count(), 2);
        assert!(registry
            .iter()
            .any(|media| media.decode().unwrap() == b"tiny"));
        assert!(registry
            .iter()
            .any(|media| media.decode().unwrap() == b"small-provider"));
    }

    #[test]
    fn late_escaped_provider_media_decodes_and_restores_each_retained_layer() {
        let mut original = json!({
            "type": "file", "mediaType": "image/png", "data": r"b'\x80\x00\xFF'"
        })
        .to_string();
        for layers in 1..=2 {
            let mut value = Value::String(original.clone().into());
            let mut registry = Vec::new();
            extract_normalized_media(&mut value, &mut registry).unwrap();
            assert_eq!(registry.len(), 1);
            let media = &registry[0];
            assert_eq!(media.decode().unwrap(), [0x80, 0x00, 0xff]);
            let restored = value.as_str().unwrap().replace(
                &media.reference(),
                &media.original_value_for_layers(Some(layers)).unwrap(),
            );
            assert_eq!(restored, original);
            original = serde_json::to_string(&original).unwrap();
        }
    }

    #[test]
    fn surrogate_text_survives_media_replacement() {
        let mut text = Value::parse(r#""\ud800 data:image/png;base64,aGk= \udfff""#).unwrap();
        let mut registry = Vec::new();
        extract_normalized_media(&mut text, &mut registry).unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(registry[0].decode().unwrap(), b"hi");
        assert_eq!(
            text.to_json_string(),
            format!(r#""\ud800 {} \udfff""#, registry[0].reference())
        );

        let source = r#""data:image/png;name=\ud800;base64,aGk=""#;
        let mut text = Value::parse(source).unwrap();
        registry.clear();
        extract_normalized_media(&mut text, &mut registry).unwrap();
        assert!(registry.is_empty());
        assert_eq!(text.to_json_string(), source);
    }

    #[test]
    fn leaves_stringified_json_unchanged_when_no_media_is_extracted() {
        let original = r#"{"number":1.0,"value":"data:image/png;base64,not#base64"}"#;
        let mut value = Value::String(original.to_owned().into());
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("scan normalized media");

        assert!(registry.is_empty());
        assert_eq!(value, Value::String(original.to_owned().into()));
    }

    #[test]
    fn continues_direct_scan_when_embedded_json_has_an_undecodable_sibling() {
        let original = r#"{"value":"\ud800 data:image/png;base64,aGk="}"#;
        let mut value = Value::String(original.to_owned().into());
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("scan normalized media");

        assert_eq!(registry.len(), 1);
        assert!(value
            .as_str()
            .unwrap()
            .contains("@@@langfuseMedia:type=image/png|id="));
    }

    #[test]
    fn extracts_unicode_escaped_data_uri_markers_in_embedded_json() {
        let original = r#"{"image":"\u0064ata:image/png;base64,aGk="}"#;
        let mut value = Value::String(original.to_owned().into());
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("scan normalized media");

        assert_eq!(registry.len(), 1);
        assert!(value
            .as_str()
            .unwrap()
            .contains("@@@langfuseMedia:type=image/png|id="));
    }

    #[test]
    fn uses_data_uri_metadata_inside_provider_values() {
        let data_uri = data_uri(b"provider-uri");
        let mut value = json!({
            "type": "file",
            "mediaType": "image/jpeg",
            "data": data_uri,
        });
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("extract provider URI");

        assert_eq!(registry.len(), 1);
        assert_eq!(registry[0].kind, MediaPayloadKind::DataUri);
        assert_eq!(registry[0].metadata.content_type, "image/png");
        assert_eq!(registry[0].metadata.source, MediaSource::Base64DataUri);
    }

    #[test]
    fn falls_back_to_camel_case_gemini_mime_when_snake_case_is_null() {
        let mut value = json!({
            "inline_data": {
                "mime_type": null,
                "mimeType": "image/png",
                "data": BASE64.encode(b"gemini"),
            },
        });
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("extract Gemini media");

        assert_eq!(registry.len(), 1);
        assert_eq!(registry[0].kind, MediaPayloadKind::Gemini);
        assert_eq!(registry[0].decode().unwrap(), b"gemini");
    }

    #[test]
    fn scans_data_uris_in_malformed_json_looking_text() {
        let original = r#"{"broken":data:image/png;base64,aGk=}"#;
        let mut value = Value::String(original.to_owned().into());
        let mut registry = Vec::new();

        extract_normalized_media(&mut value, &mut registry).expect("scan normalized media");

        assert_eq!(registry.len(), 1);
        assert!(value
            .as_str()
            .unwrap()
            .contains("@@@langfuseMedia:type=image/png|id="));
    }

    fn data_uri(value: &[u8]) -> String {
        format!("data:image/png;base64,{}", BASE64.encode(value))
    }
}
