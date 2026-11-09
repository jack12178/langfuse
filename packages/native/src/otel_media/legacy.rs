//! Collect native media references that survive normalization.
//!
//! Native discovery has already decided which references are valid media. This adapter walks the
//! normalized values that still contain those references, preserving source spelling for failed
//! uploads and retaining one request index per temporary occurrence.

use std::collections::HashMap;

use crate::otel::value::{Object, Value};

use super::payload::{ExtractedMedia, MediaPayloadKind};
use super::source_text::for_each_media_reference;

/// Find native references that survive normalization.
/// Registry indices remain stable while the scratch result is reused across fields.
pub struct MediaReferenceCollector {
    registry: HashMap<String, usize>,
    eligible: Vec<usize>,
}

impl MediaReferenceCollector {
    pub fn new<'a, I>(media: I) -> Self
    where
        I: IntoIterator<Item = &'a ExtractedMedia>,
    {
        let registry = media
            .into_iter()
            .enumerate()
            .map(|(index, entry)| (entry.reference(), index))
            .collect();
        Self {
            registry,
            eligible: Vec::new(),
        }
    }

    /// Collect references from values in traversal order. The scratch result is
    /// reused across fields, so callers must consume it before collecting again.
    pub fn collect<'a, I>(&mut self, values: I) -> &[usize]
    where
        I: IntoIterator<Item = &'a Value>,
    {
        self.eligible.clear();
        for value in values {
            Self::visit(&self.registry, value, &mut self.eligible);
        }
        &self.eligible
    }

    fn visit(registry: &HashMap<String, usize>, value: &Value, eligible: &mut Vec<usize>) {
        match value {
            Value::String(text) => Self::collect_text(registry, text.as_str(), eligible),
            Value::Array(values) => {
                for value in values {
                    Self::visit(registry, value, eligible);
                }
            }
            Value::Object(object) => {
                for value in object.values() {
                    Self::visit(registry, value, eligible);
                }
            }
            _ => {}
        }
    }

    fn collect_text(registry: &HashMap<String, usize>, text: &str, eligible: &mut Vec<usize>) {
        for_each_media_reference(text, |reference| {
            if let Some(index) = registry.get(reference) {
                eligible.push(*index);
            }
        });
    }
}

pub(crate) enum LegacyStructuredTarget<'a> {
    Direct {
        property: &'static str,
        content_type: &'a str,
        kind: MediaPayloadKind,
    },
    Nested {
        container: &'static str,
        property: &'static str,
        content_type: &'a str,
        kind: MediaPayloadKind,
    },
}

pub(crate) fn legacy_structured_target(object: &Object) -> Option<LegacyStructuredTarget<'_>> {
    if let Some(type_name) = object.get("type").and_then(Value::as_str) {
        let (mime_field, body_field, fallback_field, kind) = match type_name {
            "base64" => ("media_type", "data", None, MediaPayloadKind::Anthropic),
            "media" => ("mime_type", "data", None, MediaPayloadKind::Vertex),
            "blob" => ("mime_type", "content", None, MediaPayloadKind::AiSdkV7),
            "file" => (
                "mediaType",
                "data",
                Some("image"),
                MediaPayloadKind::AiSdkV6,
            ),
            _ => return structured_gemini_target(object),
        };
        let property = [Some(body_field), fallback_field]
            .into_iter()
            .flatten()
            .find(|field| object.get(*field).is_some_and(Value::is_string));
        if let (Some(property), Some(content_type)) =
            (property, object.get(mime_field).and_then(Value::as_str))
        {
            return Some(LegacyStructuredTarget::Direct {
                property,
                content_type,
                kind,
            });
        }
    }

    structured_gemini_target(object)
}

fn structured_gemini_target(object: &Object) -> Option<LegacyStructuredTarget<'_>> {
    for inline_name in ["inline_data", "inlineData"] {
        let Some(inline) = object.get(inline_name).and_then(Value::as_object) else {
            continue;
        };
        let Some(content_type) = inline
            .get("mime_type")
            .filter(|value| !value.is_null())
            .or_else(|| inline.get("mimeType"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        if inline.get("data").is_some_and(Value::is_string) {
            return Some(LegacyStructuredTarget::Nested {
                container: inline_name,
                property: "data",
                content_type,
                kind: MediaPayloadKind::Gemini,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::otel::value::Value;
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;

    use super::MediaReferenceCollector;
    use crate::otel_media::extract_media;
    use crate::otel_media::payload::MediaPayloadKind;
    use crate::otel_media::scanner::MIN_EARLY_MEDIA_BYTES;

    #[test]
    fn root_string_keeps_json_spelling_and_collects_all_references() {
        let payload = vec![b'x'; MIN_EARLY_MEDIA_BYTES];
        let provider_data = STANDARD.encode(&payload);
        let data_body = vec![b'y'; MIN_EARLY_MEDIA_BYTES];
        let data_uri = format!("data:image/png;base64,{}", STANDARD.encode(data_body));
        let provider_source = format!(
            "{{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"{provider_data}\"}}"
        );
        let provider_media = extract_media(provider_source.as_bytes())
            .expect("valid provider JSON")
            .media
            .into_iter()
            .next()
            .expect("provider media");
        assert_eq!(provider_media.kind, MediaPayloadKind::Anthropic);
        let data_uri_source = format!(r#"{{"sibling":"{data_uri}"}}"#);
        let data_uri_media = extract_media(data_uri_source.as_bytes())
            .expect("valid data URI JSON")
            .media
            .into_iter()
            .next()
            .expect("sibling data URI");
        assert_eq!(data_uri_media.kind, MediaPayloadKind::DataUri);
        let provider_reference = provider_media.reference();
        let data_uri_reference = data_uri_media.reference();
        let compacted_text = format!(
            "{{\n  \"type\" : \"base64\",\n  \"media_type\" : \"image/png\",\n  \"data\" : \"{provider_reference}\",\n  \"sibling\" : \"{data_uri_reference}\",\n  \"large\" : 9007199254740993.25,\n  \"ratio\" : 1.2300\n}}"
        );
        let root = Value::String(compacted_text.clone().into());
        let media = [provider_media, data_uri_media];
        let mut collector = MediaReferenceCollector::new(media.iter());

        assert_eq!(collector.collect(std::iter::once(&root)), &[0, 1]);
        assert_eq!(root.as_str(), Some(compacted_text.as_str()));
        assert!(root.as_str().unwrap().contains(&provider_reference));
        assert!(root.as_str().unwrap().contains("9007199254740993.25"));
        assert!(root.as_str().unwrap().contains("1.2300"));

        let mut too_deep = Value::String(data_uri_reference.clone().into());
        for _ in 0..32 {
            too_deep = Value::Array(vec![too_deep]);
        }
        assert_eq!(collector.collect(std::iter::once(&too_deep)), &[1]);
        let mut leaf = &too_deep;
        while let Value::Array(values) = leaf {
            leaf = &values[0];
        }
        assert_eq!(leaf.as_str(), Some(data_uri_reference.as_str()));

        let malformed_prefix =
            Value::String(format!("@@@langfuseMedia:unterminated {data_uri_reference}").into());
        assert_eq!(collector.collect(std::iter::once(&malformed_prefix)), &[1]);
    }
}
