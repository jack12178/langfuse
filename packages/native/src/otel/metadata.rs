//! Metadata and experiment/evaluation field extraction.
//!
//! OTEL metadata is carried both as JSON blobs and as dotted attributes. Keeping this logic in a
//! pure module makes the precedence rules reusable by trace and observation construction and by
//! the pricing/enrichment request builder.

use std::sync::LazyLock;

use super::attributes::{is_js_truthy, JsonParseCache};
use super::types::{flatten_object_owned, JsonObject};
use super::value::Value;
use chrono::{DateTime, NaiveDate, NaiveDateTime, SecondsFormat, Utc};

pub const TRACE_METADATA: &str = "langfuse.trace.metadata";
pub const OBSERVATION_METADATA: &str = "langfuse.observation.metadata";
pub const GENERIC_METADATA: &str = "langfuse.metadata";
pub const AI_METADATA: &str = "ai.telemetry.metadata";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataDomain {
    Trace,
    Observation,
}

impl MetadataDomain {
    fn prefix(self) -> &'static str {
        match self {
            Self::Trace => TRACE_METADATA,
            Self::Observation => OBSERVATION_METADATA,
        }
    }
}

/// Extract the effective metadata object while retaining insertion order from the input map.
pub fn extract_metadata(
    attributes: &JsonObject,
    domain: MetadataDomain,
    parse_cache: &mut JsonParseCache,
    decode_values: bool,
) -> JsonObject {
    let prefix = domain.prefix();
    let top_level = attributes
        .get(prefix)
        .filter(|value| is_js_truthy(value))
        .or_else(|| attributes.get(GENERIC_METADATA));

    extract_prefixed_metadata(
        attributes,
        top_level,
        [
            (prefix, decode_values),
            (GENERIC_METADATA, false),
            (AI_METADATA, false),
        ],
        parse_cache,
    )
}

fn extract_prefixed_metadata<const N: usize>(
    attributes: &JsonObject,
    base: Option<&Value>,
    prefixes: [(&str, bool); N],
    parse_cache: &mut JsonParseCache,
) -> JsonObject {
    let mut result = parse_metadata_value(base, parse_cache);
    let markers = prefixes.map(|(prefix, decode)| (format!("{prefix}."), decode));
    for (key, value) in attributes {
        let key_text = key.as_str();
        if matches!(
            key_text,
            "ai.telemetry.metadata.userId"
                | "ai.telemetry.metadata.sessionId"
                | "ai.telemetry.metadata.tags"
                | "ai.telemetry.metadata.langfusePrompt"
        ) {
            continue;
        }
        for (marker, decode) in &markers {
            if let Some(metadata_key) = key.strip_prefix(marker) {
                // The TypeScript metadata accumulator ignores this reserved dotted key. A
                // `__proto__` key inside the parsed JSON blob is an own data property and is
                // intentionally retained by parse_metadata_value.
                if !metadata_key.is_empty() && metadata_key.as_utf8() != Some("__proto__") {
                    result.insert(
                        metadata_key,
                        if *decode {
                            decode_metadata_value(value, parse_cache)
                        } else {
                            value.clone()
                        },
                    );
                }
            }
        }
    }
    result
}

/// Langfuse SDK scopes supply their own version; other scopes use the exporting SDK's
/// request version. The request SDK name takes precedence over the resource language.
pub fn should_decode_structured_metadata(
    scope_name: &str,
    scope_version: Option<&str>,
    header_sdk_name: &str,
    header_sdk_version: &str,
    telemetry_sdk_language: Option<&str>,
) -> bool {
    let canonical_sdk =
        normalize_ingestion_sdk_name(header_sdk_name).or(match telemetry_sdk_language {
            Some("python") => Some("python"),
            Some("nodejs" | "webjs") => Some("javascript"),
            _ => None,
        });
    let Some(canonical_sdk) = canonical_sdk else {
        return false;
    };
    let version = if scope_name.starts_with("langfuse-sdk") {
        scope_version
    } else {
        Some(header_sdk_version)
    };
    let Some(version) = version.and_then(parse_sdk_major) else {
        return false;
    };

    version >= if canonical_sdk == "python" { 5 } else { 6 }
}

fn normalize_ingestion_sdk_name(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "python" | "langfuse-python" => Some("python"),
        "javascript"
        | "js"
        | "typescript"
        | "ts"
        | "langfuse-js"
        | "langfuse-ts"
        | "@langfuse/client"
        | "@langfuse/browser"
        | "@langfuse/core"
        | "@langfuse/langchain"
        | "@langfuse/otel"
        | "@langfuse/openai"
        | "@langfuse/tracing"
        | "@langfuse/vercel-ai-sdk" => Some("javascript"),
        _ => None,
    }
}

fn parse_sdk_major(value: &str) -> Option<u64> {
    static SDK_VERSION: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
        fancy_regex::Regex::new(r"^v?(\d+)\.(\d+)\.(\d+)(?:[.+-].+|(?i:(?:a|b|rc)\d+))?$")
            .expect("static SDK version regex is valid")
    });
    let captures = SDK_VERSION.captures(value.trim()).ok()??;
    let major = parse_safe_version_number(captures.get(1)?.as_str())?;
    parse_safe_version_number(captures.get(2)?.as_str())?;
    parse_safe_version_number(captures.get(3)?.as_str())?;
    Some(major)
}

fn parse_safe_version_number(value: &str) -> Option<u64> {
    let number = value.parse::<u64>().ok()?;
    (number <= 9_007_199_254_740_991).then_some(number)
}

fn decode_metadata_value(value: &Value, parse_cache: &mut JsonParseCache) -> Value {
    let Value::String(encoded) = value else {
        return value.clone();
    };
    if let Some(decoded) = parse_cache.parse(Some(value)).cloned() {
        return decoded;
    }
    metrics::counter!(
        "langfuse.ingestion.metadata_undecoded",
        "kind" => classify_decode_failure(encoded.as_str())
    )
    .increment(1);
    value.clone()
}

fn classify_decode_failure(value: &str) -> &'static str {
    let head = value.chars().take(64).collect::<String>();
    let head = head.trim_start();
    if head.is_empty() {
        return "empty";
    }
    let python_container = head
        .strip_prefix('{')
        .or_else(|| head.strip_prefix('['))
        .is_some_and(|rest| rest.trim_start().starts_with('\''));
    let python_scalar = ["True", "False", "None"].iter().any(|candidate| {
        head.strip_prefix(candidate).is_some_and(|rest| {
            rest.chars()
                .next()
                .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
        })
    });
    if python_container || python_scalar {
        return "python_repr";
    }
    if let Some(opening) = head
        .chars()
        .next()
        .filter(|character| *character == '{' || *character == '[')
    {
        let expected_close = if opening == '{' { '}' } else { ']' };
        let tail = value.chars().rev().take(64).collect::<String>();
        let last = tail.trim_start().chars().next();
        return if last == Some(expected_close) {
            "loose_json"
        } else {
            "truncated_json"
        };
    }
    "unquoted_string"
}

pub fn extract_metadata_from_prefix(
    attributes: &JsonObject,
    prefix: &str,
    parse_cache: &mut JsonParseCache,
) -> JsonObject {
    extract_prefixed_metadata(
        attributes,
        attributes.get(prefix),
        [(prefix, false)],
        parse_cache,
    )
}

pub fn apply_context_fields(
    row: &mut crate::native_schema::OtelStorageRow,
    attributes: &JsonObject,
    parse_cache: &mut JsonParseCache,
) {
    (
        row.experiment_metadata_names,
        row.experiment_metadata_values,
    ) = flatten_object_owned(extract_metadata_from_prefix(
        attributes,
        "langfuse.experiment.metadata",
        parse_cache,
    ));
    (
        row.experiment_item_metadata_names,
        row.experiment_item_metadata_values,
    ) = flatten_object_owned(extract_metadata_from_prefix(
        attributes,
        "langfuse.experiment.item.metadata",
        parse_cache,
    ));
    row.experiment_item_version = parse_experiment_item_version(
        attributes.get("langfuse.experiment.item.version"),
        parse_cache,
    );
    row.evaluator_execution_is_test = match attributes.get("langfuse.evaluator.execution.is_test") {
        Some(value) if !value.is_null() => is_evaluator_test_value(value),
        _ => false,
    };
}

/// Apply the ClickHouse evaluator-test default after the final metadata map exists.
///
/// The TypeScript path only derives this default when all explicit evaluation attributes are
/// nullish. Keeping this separate from context extraction avoids parsing and cloning metadata
/// once per ordinary span.
pub fn apply_evaluator_test_metadata_default(
    row: &mut crate::native_schema::OtelStorageRow,
    attributes: &JsonObject,
    metadata: &JsonObject,
) {
    let explicit = [
        "langfuse.evaluator.id",
        "langfuse.evaluation.rule.id",
        "langfuse.evaluator.execution.is_test",
    ]
    .into_iter()
    .any(|key| attributes.get(key).is_some_and(|value| !value.is_null()));
    if !explicit {
        row.evaluator_execution_is_test = metadata
            .get("evaluator_test")
            .is_some_and(is_evaluator_test_value);
    }
}

fn parse_metadata_value(value: Option<&Value>, parse_cache: &mut JsonParseCache) -> JsonObject {
    let Some(value) = value else {
        return JsonObject::new();
    };
    match value {
        Value::Object(value) => value.clone(),
        Value::Array(value) => array_to_object(value),
        Value::String(encoded) => {
            let Some(parsed) = parse_cache.parse(Some(value)) else {
                record_metadata_drop(classify_decode_failure(encoded.as_str()));
                return JsonObject::new();
            };
            match parsed {
                Value::Object(value) => value.clone(),
                Value::Array(value) => array_to_object(value),
                _ => {
                    record_metadata_drop("non_object_top_level");
                    JsonObject::new()
                }
            }
        }
        _ => {
            record_metadata_drop("primitive");
            JsonObject::new()
        }
    }
}

fn array_to_object(values: &[Value]) -> JsonObject {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| (index.to_string().into(), value.clone()))
        .collect()
}

fn record_metadata_drop(kind: &'static str) {
    metrics::counter!("langfuse.ingestion.metadata_dropped", "kind" => kind).increment(1);
}

fn is_evaluator_test_value(value: &Value) -> bool {
    matches!(value, Value::Bool(true)) || value.as_str() == Some("true")
}

fn value_to_string(value: Option<&Value>) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if value.is_empty() => None,
        Some(Value::String(value)) => Some(value.as_str().to_owned()),
        Some(value) => Some(super::attributes::javascript_json_stringify(value)),
    }
}

fn parse_experiment_item_version(
    value: Option<&Value>,
    parse_cache: &mut JsonParseCache,
) -> Option<String> {
    let value = value_to_string(value)?;
    let value = if is_valid_date_string(&value) {
        value
    } else {
        let encoded = Value::String(value.into());
        let parsed = parse_cache.parse(Some(&encoded))?;
        let Value::String(parsed) = parsed else {
            return None;
        };
        let parsed = parsed.as_str();
        is_valid_date_string(parsed).then(|| parsed.to_owned())?
    };
    Some(to_clickhouse_datetime(&value))
}

/// `experiment_item_version` is a dataset item's `valid_from` timestamp, not a free-form label.
/// Match the TS `new Date(...).toISOString()`/ClickHouse conversion and reject values such as
/// `v1`, rather than treating a hyphen anywhere in the string as evidence of a date.
fn is_valid_date_string(value: &str) -> bool {
    DateTime::parse_from_rfc3339(value).is_ok()
        || NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").is_ok()
        || NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M").is_ok()
        || NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").is_ok()
        || NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M").is_ok()
        || NaiveDate::parse_from_str(value, "%Y-%m-%d").is_ok()
}

fn to_clickhouse_datetime(value: &str) -> String {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return parsed
            .with_timezone(&Utc)
            .to_rfc3339_opts(SecondsFormat::Millis, true)
            .replace('T', " ")
            .trim_end_matches('Z')
            .to_owned();
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f") {
        return parsed.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M") {
        return parsed.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f") {
        return parsed.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M") {
        return parsed.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    }
    if let Ok(parsed) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return parsed
            .and_hms_opt(0, 0, 0)
            .expect("midnight is valid")
            .format("%Y-%m-%d %H:%M:%S%.3f")
            .to_string();
    }
    value.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::value::json;

    fn context_fields(
        attributes: &JsonObject,
        cache: &mut JsonParseCache,
    ) -> crate::native_schema::OtelStorageRow {
        let empty = JsonObject::new();
        let mut row = crate::native_schema::OtelStorageRow::from_attributes(
            &mut super::super::attributes::EventAttributes {
                span: attributes,
                resource: &empty,
                scope: &empty,
                cache,
                decode_metadata: false,
            },
        );
        apply_context_fields(&mut row, attributes, cache);
        row
    }

    #[test]
    fn metadata_version_comes_from_the_exporter_for_custom_scopes() {
        for (scope, scope_version, sdk, header_version, expected) in [
            ("custom", Some("1.0.0"), "python", "5.0.0", true),
            ("custom", None, "javascript", "6.0.0", true),
            ("custom", Some("6.0.0"), "javascript", "5.0.0", false),
            ("langfuse-sdk", Some("4.0.0"), "python", "5.0.0", false),
            ("langfuse-sdk", Some("5.0.0"), "python", "4.0.0", true),
        ] {
            assert_eq!(
                should_decode_structured_metadata(scope, scope_version, sdk, header_version, None),
                expected,
                "{scope}: {scope_version:?} / {sdk} {header_version}",
            );
        }
    }

    #[test]
    fn prefixed_metadata_overlays_blob_metadata() {
        let attributes = json!({
            "langfuse.trace.metadata": "{\"blob\":1,\"same\":\"blob\",\"__proto__\":\"blob-proto\"}",
            "langfuse.trace.metadata.same": "dotted",
            "langfuse.trace.metadata.__proto__": "dotted-proto",
            "langfuse.trace.metadata.extra": true,
            "ai.telemetry.metadata.userId": "user",
        });
        let metadata = extract_metadata(
            attributes.as_object().unwrap(),
            MetadataDomain::Trace,
            &mut JsonParseCache::default(),
            false,
        );
        assert_eq!(metadata.get("same"), Some(&json!("dotted")));
        assert_eq!(metadata.get("blob"), Some(&json!(1)));
        assert_eq!(metadata.get("__proto__"), Some(&json!("blob-proto")));
        assert!(!metadata.contains_key("userId"));
    }

    #[test]
    fn metadata_arrays_use_javascript_object_spread_keys() {
        let attributes = json!({
            "langfuse.trace.metadata": ["first", {"nested": true}],
        });
        let metadata = extract_metadata(
            attributes.as_object().unwrap(),
            MetadataDomain::Trace,
            &mut JsonParseCache::default(),
            false,
        );
        assert_eq!(metadata.get("0"), Some(&json!("first")));
        assert_eq!(metadata.get("1"), Some(&json!({"nested": true})));
    }

    #[test]
    fn metadata_evaluator_test_fills_the_storage_default() {
        let attributes = json!({
            "langfuse.observation.metadata": r#"{"evaluator_test":true}"#,
        });
        let mut fields = context_fields(
            attributes.as_object().unwrap(),
            &mut JsonParseCache::default(),
        );
        let metadata = extract_metadata(
            attributes.as_object().unwrap(),
            MetadataDomain::Observation,
            &mut JsonParseCache::default(),
            false,
        );
        apply_evaluator_test_metadata_default(
            &mut fields,
            attributes.as_object().unwrap(),
            &metadata,
        );
        assert!(fields.evaluator_execution_is_test);
    }

    #[test]
    fn preserves_wtf8_suffixes_in_dotted_metadata_keys() {
        let mut attributes = json!({}).as_object().unwrap().clone();
        let mut key = TRACE_METADATA.as_bytes().to_vec();
        key.extend_from_slice(b".label-");
        key.extend_from_slice(&[0xed, 0xa0, 0x80]);
        let key = super::super::value::JsString::from_wtf8(key).unwrap();
        attributes.insert(key.clone(), json!("value"));

        let metadata = extract_metadata(
            &attributes,
            MetadataDomain::Trace,
            &mut JsonParseCache::default(),
            false,
        );
        let suffix = metadata
            .keys()
            .find(|candidate| candidate.as_bytes().ends_with(b"label-\xed\xa0\x80"))
            .expect("WTF-8 metadata suffix should be retained");
        assert_eq!(suffix.as_bytes(), b"label-\xed\xa0\x80");
    }

    #[test]
    fn classifies_undecodable_metadata_without_recording_its_value() {
        assert_eq!(classify_decode_failure("  "), "empty");
        assert_eq!(classify_decode_failure("{'key': 1}"), "python_repr");
        assert_eq!(classify_decode_failure("{not json}"), "loose_json");
        assert_eq!(classify_decode_failure(r#"{"key":"cut"#), "truncated_json");
        assert_eq!(classify_decode_failure("plain text"), "unquoted_string");
    }

    #[test]
    fn decodes_metadata_strings_even_when_their_contents_are_json() {
        let document = r#"{"image":"data:image/png;base64,AAAA"}"#;
        let encoded = serde_json::to_string(document).unwrap();
        let attributes = json!({
            "langfuse.observation.metadata.json": encoded.clone(),
            "langfuse.observation.metadata.label": serde_json::to_string("quoted").unwrap(),
            "langfuse.observation.metadata.numeric": serde_json::to_string("42").unwrap(),
            "langfuse.metadata.generic": r#"{"nested":1}"#,
            "ai.telemetry.metadata.provider": "null",
        });

        let metadata = extract_metadata(
            attributes.as_object().unwrap(),
            MetadataDomain::Observation,
            &mut JsonParseCache::default(),
            true,
        );

        assert_eq!(metadata.get("json"), Some(&json!(document)));
        assert_eq!(metadata.get("label"), Some(&json!("quoted")));
        assert_eq!(metadata.get("numeric"), Some(&json!("42")));
        assert_eq!(metadata.get("generic"), Some(&json!(r#"{"nested":1}"#)));
        assert_eq!(metadata.get("provider"), Some(&json!("null")));
    }

    #[test]
    fn experiment_metadata_is_flattened() {
        let attributes = json!({
            "langfuse.experiment.metadata.foo": {"bar": 1},
            "langfuse.experiment.item.metadata.dataset": "x",
            "langfuse.experiment.id": "exp",
        });
        let fields = context_fields(
            attributes.as_object().unwrap(),
            &mut JsonParseCache::default(),
        );
        assert_eq!(fields.experiment_id.as_deref(), Some("exp"));
        assert_eq!(fields.experiment_metadata_names, ["foo.bar"]);
        assert_eq!(fields.experiment_metadata_values, ["1"]);
        assert_eq!(fields.experiment_item_metadata_names, ["dataset"]);
    }

    #[test]
    fn experiment_item_version_requires_a_real_date_and_uses_clickhouse_shape() {
        let attributes = json!({
            "langfuse.experiment.item.version": "2026-04-24T15:22:36.622Z",
        });
        assert_eq!(
            context_fields(
                attributes.as_object().unwrap(),
                &mut JsonParseCache::default(),
            )
            .experiment_item_version
            .as_deref(),
            Some("2026-04-24 15:22:36.622")
        );

        let attributes = json!({
            "langfuse.experiment.item.version": "2026-04-24T15:22",
        });
        assert_eq!(
            context_fields(
                attributes.as_object().unwrap(),
                &mut JsonParseCache::default(),
            )
            .experiment_item_version
            .as_deref(),
            Some("2026-04-24 15:22:00.000")
        );

        let attributes = json!({"langfuse.experiment.item.version": "v1"});
        assert_eq!(
            context_fields(
                attributes.as_object().unwrap(),
                &mut JsonParseCache::default(),
            )
            .experiment_item_version,
            None
        );
    }
}
