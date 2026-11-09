//! Construction of owned direct-v4 event inputs from OTLP resource spans.
//!
//! This is the Rust equivalent of the synchronous `processToEvent` phase. It intentionally stops
//! before model/prompt lookups, media upload, evaluation scheduling, and ClickHouse encoding. Those
//! stages receive `OwnedEvent`s and can enrich or finalize them without converting the whole graph
//! back to JavaScript.

use super::event_skills::extract_skills;
use super::value::{Object, Value};
use serde::de::{Error as DeError, SeqAccess, Visitor};
use serde::Deserializer;

use super::attributes::{
    attributes_to_map, is_js_truthy, parse_id, resolve_timestamps, value_as_string,
    AttributeSource, EventAttributes, JsonParseCache,
};
use super::input::{ResourceSpan, Span};
use super::metadata::{
    apply_context_fields, apply_evaluator_test_metadata_default, extract_metadata,
    should_decode_structured_metadata, MetadataDomain,
};
use super::providers::{
    claude_managed_agent_session_id, extract_completion_start_time, extract_cost_details,
    extract_input_output, extract_is_app_root, extract_model_name, extract_model_parameters,
    extract_name, extract_prompt_version, extract_prompt_version_lookup, extract_public,
    extract_tags, extract_usage_details, observation_type, should_skip_model_usage_and_cost,
};
use super::tools::normalize_tools_for_observation;
use super::types::{JsonObject, OwnedEvent};

const INTERNAL_LLM_JUDGE_ENVIRONMENT: &str = "langfuse-llm-as-a-judge";
const AI_GATEWAY_SERVICE_NAME: &str = "langfuse-ai-gateway";
const AI_GATEWAY_INGESTION_API_KEY_PREFIX: &str = "gateway:";
const AI_GATEWAY_KEY_ID_METADATA_KEY: &str = "langfuse.gateway.api_key.id";

#[derive(Clone, Debug)]
pub struct EventBuildConfig {
    pub project_id: String,
    pub ingestion_api_key: String,
    pub ingestion_sdk_name: String,
    pub ingestion_sdk_version: String,
    pub source: String,
    pub fallback_time: String,
    pub is_langfuse_internal: bool,
}

impl Default for EventBuildConfig {
    fn default() -> Self {
        Self {
            project_id: String::new(),
            ingestion_api_key: String::new(),
            ingestion_sdk_name: String::new(),
            ingestion_sdk_version: String::new(),
            source: "otel".to_owned(),
            fallback_time: "1970-01-01T00:00:00.000Z".to_owned(),
            is_langfuse_internal: false,
        }
    }
}

/// Process a compact OTLP array without materializing the array or all of its resources.
///
/// Each resource is decoded into typed envelope fields, converted into owned events, and then
/// dropped before the next resource is read. Ignored fields never become values. The pipeline
/// retains normalized events for media resolution and enrichment.
pub fn process_to_events(
    bytes: &[u8],
    config: &EventBuildConfig,
) -> Result<Vec<OwnedEvent>, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let mut events = Vec::new();
    match deserializer.deserialize_seq(ResourceSpansVisitor {
        config,
        events: &mut events,
    }) {
        Ok(()) => deserializer.end().map_err(|error| error.to_string())?,
        Err(error) => {
            let message = error.to_string();
            if ![
                "lone leading surrogate",
                "unexpected end of hex escape",
                "number out of range",
                "recursion limit exceeded",
            ]
            .iter()
            .any(|prefix| message.starts_with(prefix))
            {
                return Err(message);
            }
            // Serde's ordinary decoder rejects some valid JavaScript strings and numbers.
            // Retry with borrowed spans; no events escape before the whole batch succeeds.
            events.clear();
            super::input::CompatibleContainerDeserializer::from_bytes(bytes)
                .map_err(|error| error.to_string())?
                .deserialize_seq(ResourceSpansVisitor {
                    config,
                    events: &mut events,
                })
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(events)
}

struct ResourceSpansVisitor<'a> {
    config: &'a EventBuildConfig,
    events: &'a mut Vec<OwnedEvent>,
}

impl<'de> Visitor<'de> for ResourceSpansVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an array of OTLP resource spans")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while let Some(resource_span) = sequence.next_element::<ResourceSpan>()? {
            process_resource_span(resource_span, self.config, self.events)
                .map_err(A::Error::custom)?;
        }
        Ok(())
    }
}

fn process_resource_span(
    resource_span: ResourceSpan,
    config: &EventBuildConfig,
    events: &mut Vec<OwnedEvent>,
) -> Result<(), String> {
    let resource_attributes = resource_span
        .resource
        .and_then(|resource| resource.attributes)
        .map(|attributes| attributes_to_map(attributes.into_value()))
        .unwrap_or_default();
    let scopes = resource_span
        .scope_spans
        .and_then(|value| value.0)
        .or_else(|| resource_span.scope_spans_alias.and_then(|value| value.0))
        .unwrap_or_default();
    for scope_span in scopes {
        let mut scope = match scope_span.scope.map(|scope| scope.into_value()) {
            Some(Value::Object(scope)) => scope,
            _ => Object::new(),
        };
        // Normalize once before copying the scope into each event's metadata.
        // Replacing in place preserves the original field order.
        let scope_attributes = scope.entry("attributes".into()).or_default();
        *scope_attributes = Value::Object(attributes_to_map(std::mem::take(scope_attributes)));
        let scope_name = scope
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let scope_version = scope.get("version").and_then(value_as_string);

        for span in scope_span
            .spans
            .and_then(|value| value.0)
            .unwrap_or_default()
        {
            if let Some(event) = build_event(
                span,
                &resource_attributes,
                &scope,
                scope_name,
                scope_version.as_deref(),
                config,
            )? {
                events.push(event);
            }
        }
    }
    Ok(())
}

fn build_event(
    mut span: Span,
    resource_attributes: &JsonObject,
    scope: &Object,
    scope_name: &str,
    scope_version: Option<&str>,
    config: &EventBuildConfig,
) -> Result<Option<OwnedEvent>, String> {
    let trace_id = span
        .trace_id
        .as_deref()
        .or(span.trace_id_alias.as_deref())
        .map(parse_id)
        .unwrap_or_default();
    let span_id = span
        .span_id
        .as_deref()
        .or(span.span_id_alias.as_deref())
        .map(parse_id)
        .unwrap_or_default();
    if trace_id.is_empty() || span_id.is_empty() {
        return Ok(None);
    }
    let parent_span_id = span
        .parent_span_id
        .as_deref()
        .or(span.parent_span_id_alias.as_deref())
        .filter(|value| !value.is_null())
        .map(parse_id)
        .filter(|value| !value.is_empty());

    let attributes = object_attributes(span.attributes.take().map(|value| value.into_value()));
    let span_name = span
        .name
        .as_deref()
        .and_then(Value::as_str)
        .unwrap_or_default();
    let event_type = observation_type(
        &attributes,
        resource_attributes,
        scope_name,
        scope_version,
        span_name,
    );
    let name = extract_name(span_name, &attributes, &event_type, scope_name);
    let (start_time_iso, end_time_iso) = resolve_timestamps(
        span.start_time
            .as_deref()
            .or(span.start_time_alias.as_deref()),
        span.end_time.as_deref().or(span.end_time_alias.as_deref()),
        &config.fallback_time,
    );
    let events = span
        .events
        .take()
        .and_then(|events| events.0)
        .unwrap_or_default();
    let mut parse_cache = JsonParseCache::default();
    let decode_metadata_values = should_decode_structured_metadata(
        scope_name,
        scope_version,
        &config.ingestion_sdk_name,
        &config.ingestion_sdk_version,
        resource_attributes
            .get("telemetry.sdk.language")
            .and_then(Value::as_str),
    );
    let mut event_attributes = EventAttributes {
        span: &attributes,
        resource: resource_attributes,
        scope,
        cache: &mut parse_cache,
        decode_metadata: decode_metadata_values,
    };
    let mut row = crate::native_schema::OtelStorageRow::from_attributes(&mut event_attributes);
    if claude_managed_agent_session_id(&attributes, scope_name).is_some() {
        row.session_id = event_attributes.identity(&[
            AttributeSource::Span("langfuse.session.id"),
            AttributeSource::Span("session.id"),
            AttributeSource::Span("anthropic.session.id"),
        ]);
    }
    apply_context_fields(&mut row, &attributes, &mut parse_cache);
    let provider_io = extract_input_output(
        scope_name,
        &attributes,
        events,
        parent_span_id.as_deref().map(|_| "observation"),
        &mut parse_cache,
    );

    // Match processToEvent's object-spread order. The filtered span attributes and OTLP
    // resource/scope envelopes are inserted first; observation and trace metadata are spread on
    // top afterward. This order is visible in metadata_names because storage flattening preserves
    // insertion order.
    let mut metadata = JsonObject::new();
    // The legacy processor stores filtered span attributes below `attributes`; keeping them at
    // the metadata root changes both persisted shape and pricing/metadata lookup behavior. SDK
    // spans are the one exception retained by the v3 contract.
    if !scope_name.starts_with("langfuse-sdk") {
        metadata.insert(
            "attributes".into(),
            Value::Object(provider_io.filtered_attributes),
        );
    }
    metadata.insert(
        "resourceAttributes".into(),
        Value::Object(resource_attributes.clone()),
    );
    metadata.insert("scope".into(), Value::Object(scope.clone()));
    for (key, value) in extract_metadata(
        &attributes,
        MetadataDomain::Observation,
        &mut parse_cache,
        decode_metadata_values,
    ) {
        metadata.insert(key, value);
    }
    for (key, value) in extract_metadata(
        &attributes,
        MetadataDomain::Trace,
        &mut parse_cache,
        decode_metadata_values,
    ) {
        metadata.insert(key, value);
    }

    // Keep this after provider extraction and metadata construction. The production processor
    // moves metadata-carried definitions into the input before extracting definitions/calls;
    // doing it earlier would either leave stale metadata behind or miss provider-specific shapes.
    let normalized_tools = normalize_tools_for_observation(
        provider_io.input,
        provider_io.output,
        metadata,
        &mut parse_cache,
    );
    let tools = normalized_tools.tools;
    let metadata = normalized_tools.metadata;
    let input = normalized_tools.input;
    let output = normalized_tools.output;
    apply_evaluator_test_metadata_default(&mut row, &attributes, &metadata);
    let can_link_prompt = event_type == "GENERATION";
    let skip_model_usage_and_cost =
        should_skip_model_usage_and_cost(&attributes, scope_name, span_name, &event_type);
    let status_error = span
        .status
        .as_ref()
        .map(|status| {
            // OTLP's status code is a numeric enum. The legacy processor uses strict
            // `span.status?.code === 2`; accepting a string or a free-form status label would
            // turn unrelated exporter metadata into an error level.
            matches!(
                status.code.as_deref(),
                Some(Value::Number(value)) if value.as_i64() == Some(2)
            )
        })
        .unwrap_or(false);
    let level = parse_observation_level(attributes.get("langfuse.observation.level"))
        .or_else(|| status_error.then(|| "ERROR".to_owned()))
        .or_else(|| {
            (scope_name == "livekit-agents" && livekit_debug_span_name(span_name))
                .then(|| "DEBUG".to_owned())
        })
        .unwrap_or_else(|| "DEFAULT".to_owned());
    let (skills_available, skills_resource_loaded) = extract_skills(
        &attributes,
        &metadata,
        &event_type,
        &name,
        input.as_ref(),
        output.as_ref(),
        level == "ERROR",
    );
    row.status_message = row.status_message.or_else(|| {
        span.status
            .as_ref()
            .and_then(|status| status.message.as_deref())
            .and_then(value_as_string)
    });

    let environment = extract_environment(
        &attributes,
        resource_attributes,
        config.is_langfuse_internal,
        &mut parse_cache,
    );
    let prompt_name = if can_link_prompt {
        first_non_null_string(
            &attributes,
            &["langfuse.observation.prompt.name", "langfuse.prompt.name"],
        )
        .or_else(|| {
            parse_prompt(&attributes, &mut parse_cache)
                .and_then(|value| value.get("name").and_then(value_as_string))
        })
    } else {
        None
    };
    let prompt_version_lookup = can_link_prompt
        .then(|| extract_prompt_version_lookup(&attributes, &mut parse_cache))
        .flatten();
    let prompt_version = can_link_prompt
        .then(|| extract_prompt_version(&attributes, &mut parse_cache))
        .flatten();
    let provided_usage_details = if skip_model_usage_and_cost {
        Default::default()
    } else {
        extract_usage_details(&attributes, scope_name, &mut parse_cache)
    };
    let provided_cost_details = if skip_model_usage_and_cost {
        Default::default()
    } else {
        extract_cost_details(&attributes, &mut parse_cache)
    };
    let Some((ingestion_api_key, service_name, service_version)) = resolve_gateway_attribution(
        &config.ingestion_api_key,
        &metadata,
        row.service_name.take(),
        row.service_version.take(),
        scope_version,
    ) else {
        // The TypeScript adapter rejects only this span when a trusted gateway
        // batch omits the caller key; the other spans remain processable.
        tracing::warn!(
            project_id = %config.project_id,
            observation_id = %span_id,
            "Skipping AI gateway span without a caller API key"
        );
        return Ok(None);
    };
    let event = OwnedEvent {
        row: crate::native_schema::OtelStorageRow {
            project_id: config.project_id.clone(),
            trace_id,
            span_id,
            parent_span_id,
            name,
            event_type,
            environment,
            level,
            start_time: start_time_iso.clone(),
            end_time: end_time_iso,
            completion_start_time: extract_completion_start_time(
                &attributes,
                &start_time_iso,
                &mut parse_cache,
            ),
            prompt_name,
            prompt_version,
            provided_model_name: if skip_model_usage_and_cost {
                None
            } else {
                extract_model_name(&attributes)
            },
            tags: extract_tags(&attributes, &mut parse_cache, decode_metadata_values),
            bookmarked: false,
            public: extract_public(&attributes),
            is_app_root: extract_is_app_root(&attributes),
            tool_definitions: tools.tool_definitions,
            tool_calls: tools.tool_calls,
            tool_call_names: tools.tool_call_names,
            service_name,
            service_version,
            source: config.source.clone(),
            ingestion_api_key,
            ingestion_sdk_name: config.ingestion_sdk_name.clone(),
            ingestion_sdk_version: config.ingestion_sdk_version.clone(),
            ..row
        },
        prompt_version_lookup,
        model_parameters: extract_model_parameters(&attributes, scope_name, &mut parse_cache),
        provided_usage_details,
        provided_cost_details,
        input,
        output,
        metadata,
        skills_available,
        skills_resource_loaded,
    };
    Ok(Some(event))
}

/// Resolve the persisted key and service fields for one event.
///
/// A gateway grant authenticates the batch, while each span carries the caller
/// key id in metadata. A missing caller id drops that span; it does not reject
/// its siblings in the same OTLP request.
fn resolve_gateway_attribution(
    ingestion_api_key: &str,
    metadata: &JsonObject,
    service_name: Option<String>,
    service_version: Option<String>,
    scope_version: Option<&str>,
) -> Option<(String, Option<String>, Option<String>)> {
    if !ingestion_api_key.starts_with(AI_GATEWAY_INGESTION_API_KEY_PREFIX) {
        if service_name.as_deref() == Some(AI_GATEWAY_SERVICE_NAME) {
            return Some((ingestion_api_key.to_owned(), None, None));
        }
        return Some((ingestion_api_key.to_owned(), service_name, service_version));
    }

    let key_id = metadata
        .get(AI_GATEWAY_KEY_ID_METADATA_KEY)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    Some((
        key_id.to_owned(),
        Some(AI_GATEWAY_SERVICE_NAME.to_owned()),
        scope_version.map(str::to_owned),
    ))
}

fn object_attributes(value: Option<Value>) -> JsonObject {
    value
        .map(|mut value| {
            value
                .as_object_mut()
                .and_then(|object| object.shift_remove("attributes"))
                .unwrap_or(value)
        })
        .map(attributes_to_map)
        .unwrap_or_default()
}

fn first_non_null_string(attributes: &JsonObject, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        attributes
            .get(*key)
            // Prompt names use `??` in the TypeScript processor. In particular, an explicit
            // empty string suppresses the alias instead of being treated as a missing value.
            .filter(|value| !value.is_null())
            .and_then(value_as_string)
    })
}

fn parse_prompt(attributes: &JsonObject, parse_cache: &mut JsonParseCache) -> Option<Value> {
    match attributes.get("ai.telemetry.metadata.langfusePrompt")? {
        value @ Value::String(_) => parse_cache.parse(Some(value)).cloned(),
        value => Some(value.clone()),
    }
}

fn extract_environment(
    attributes: &JsonObject,
    resource: &JsonObject,
    internal: bool,
    parse_cache: &mut JsonParseCache,
) -> String {
    for key in [
        "langfuse.environment",
        "deployment.environment.name",
        "deployment.environment",
    ] {
        // The TypeScript processor checks the span value with `if (value)` and only then falls
        // back to the resource value. A present but falsy span value (for example `""`) must
        // therefore not suppress a truthy resource value. Once a truthy value is selected,
        // normalization owns invalid-value handling and we must not fall through to another key.
        let value = attributes
            .get(key)
            .filter(|value| is_js_truthy(value))
            .or_else(|| resource.get(key).filter(|value| is_js_truthy(value)));
        if let Some(value) = value {
            return value
                .as_str()
                .map(|value| normalize_environment(value, internal))
                .unwrap_or_else(|| "default".to_owned());
        }
    }

    // Failed OpenRouter broadcasts can echo the reserved evaluator environment inside the raw
    // request even when the custom trace metadata was omitted. Match the legacy recovery rule so
    // such spans remain internal and are not scheduled as fresh evaluations.
    if resource.get("service.name").and_then(Value::as_str) == Some("openrouter")
        && attributes.get("openrouter.source").and_then(Value::as_str) == Some("openrouter")
    {
        let completion = attributes
            .get("gen_ai.completion")
            .and_then(|value| parse_json_value(value, parse_cache));
        if completion
            .as_ref()
            .and_then(|value| value.get("completion"))
            .is_some_and(Value::is_null)
            && completion
                .as_ref()
                .and_then(|value| value.path(["rawRequest", "trace", "environment"]))
                .and_then(Value::as_str)
                == Some(INTERNAL_LLM_JUDGE_ENVIRONMENT)
        {
            // The legacy fallback calls normalizeEnvironment without the internal option, even
            // when the queue itself is internal. Preserve that public normalization contract.
            return normalize_environment(INTERNAL_LLM_JUDGE_ENVIRONMENT, false);
        }
    }

    "default".to_owned()
}

fn parse_observation_level(value: Option<&Value>) -> Option<String> {
    let value = value?.as_str()?.trim().to_ascii_uppercase();
    let normalized = match value.as_str() {
        "DEBUG" | "TRACE" | "VERBOSE" => "DEBUG",
        "DEFAULT" | "INFO" | "LOG" | "NOTICE" | "OK" | "SUCCESS" => "DEFAULT",
        "WARNING" | "WARN" => "WARNING",
        "ERROR" | "FATAL" | "CRITICAL" => "ERROR",
        _ => return None,
    };
    Some(normalized.to_owned())
}

fn livekit_debug_span_name(name: &str) -> bool {
    matches!(
        name,
        "user_speaking"
            | "eou_detection"
            | "llm_request_run"
            | "tts_node"
            | "tts_request"
            | "tts_request_run"
            | "tts_stream_adapter"
            | "agent_speaking"
            | "drain_agent_activity"
            | "on_exit"
            | "on_enter"
            | "llm_fallback_adapter"
            | "tts_fallback_adapter"
            | "start_agent_activity"
    )
}

fn normalize_environment(value: &str, internal: bool) -> String {
    let mut value = value.to_owned();
    if !internal {
        value = value.to_ascii_lowercase();
        while let Some(rest) = value.strip_prefix("langfuse") {
            value = rest
                .strip_prefix('-')
                .or_else(|| rest.strip_prefix('_'))
                .unwrap_or(rest)
                .to_owned();
        }
    }
    let value = value.chars().take(40).collect::<String>();
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
    {
        "default".to_owned()
    } else {
        value
    }
}

fn parse_json_value(value: &Value, parse_cache: &mut JsonParseCache) -> Option<Value> {
    match value {
        Value::String(_) => parse_cache.parse(Some(value)).cloned(),
        value => Some(value.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::value::json;
    use super::*;

    #[test]
    fn skips_ignored_envelope_values_without_materializing_them() {
        for ignored in [
            format!("{}null{}", "[".repeat(200), "]".repeat(200)),
            r#""\ud800""#.to_owned(),
            "1e309".to_owned(),
        ] {
            let input = format!(
                r#"[{{"ignored":{ignored},"resource":{{"ignored":{ignored}}},
                  "scopeSpans":[{{"ignored":{ignored},"spans":[{{
                    "traceId":"trace","spanId":"span","name":"kept",
                    "ignored":{ignored},"status":{{"ignored":{ignored}}},
                    "events":[{{"name":"gen_ai.user.message","ignored":{ignored},
                      "attributes":[{{"key":"content","value":{{"stringValue":"hello"}}}}]}}]
                }}]}}]}}]"#,
            );
            let events = process_to_events(input.as_bytes(), &EventBuildConfig::default()).unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].name, "kept");
            assert_eq!(events[0].level, "DEFAULT");
            assert_eq!(
                events[0].input,
                Some(json!([{"role":"user", "content":"hello"}]))
            );
            assert_eq!(events[0].metadata["resourceAttributes"], json!({}));
        }
        assert!(process_to_events(br#"[{"ignored":[1,]}]"#, &EventBuildConfig::default()).is_err());

        // A materialized field also must not reject its resource or the next one.
        let nested = "[".repeat(200) + "null" + &"]".repeat(200);
        let input = format!(
            r#"[{{"scopeSpans":[{{"spans":[{{"traceId":"t","spanId":"deep","attributes":{nested}}}]}}]}},{{"scopeSpans":[{{"spans":[{{"traceId":"t","spanId":"kept"}}]}}]}}]"#
        );
        let events = process_to_events(input.as_bytes(), &EventBuildConfig::default()).unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| event.span_id.as_str())
                .collect::<Vec<_>>(),
            ["deep", "kept"]
        );
    }

    #[test]
    fn ignores_unrepresentable_scalars_only_at_container_boundaries() {
        for scalar in [r#""\ud800""#, r#""\udfff""#, "1e309"] {
            let input = format!(
                r#"[{{"scopeSpans":[{{"spans":[{{"traceId":"t","spanId":"first"}}]}}]}},
                {scalar},{{"resource":{scalar},"scopeSpans":[{scalar},{{"spans":[{scalar},
                {{"traceId":"t","spanId":"second","status":{scalar},"events":{scalar},
                  "attributes":[{{"key":"langfuse.observation.input","value":{{"stringValue":"\ud800"}}}}]}}
            ]}}]}}]"#
            );
            let events = process_to_events(input.as_bytes(), &EventBuildConfig::default()).unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[0].span_id, "first");
            assert_eq!(events[1].span_id, "second");
            assert_eq!(
                events[1].input.as_ref().unwrap().to_json_string(),
                r#""\ud800""#
            );
        }
    }

    #[test]
    fn keeps_duplicate_fields_and_alias_precedence() {
        // Ignored entries must not retain a full typed Span each; explicit null fields remain.
        let spans: crate::otel::input::Array<Span> =
            serde_json::from_str(r#"[null,false,0,[],{},{"ignored":0},{"spanId":null}]"#).unwrap();
        assert_eq!(spans.0.unwrap().len(), 1);

        let span = r#"{"traceId":"trace","spanId":"old","spanId":"kept","span_id":"alias"}"#;
        let scopes = format!(r#"[null,{{"spans":[false,[],{span}]}}]"#);
        for (fields, count) in [
            (format!(r#""scopeSpans":[],"scopeSpans":{scopes}"#), 1),
            (format!(r#""scopeSpans":null,"scope_spans":{scopes}"#), 1),
            (format!(r#""scopeSpans":[],"scope_spans":{scopes}"#), 0),
            (format!(r#""scopeSpans":{scopes},"scopeSpans":false"#), 0),
            (
                format!(r#""scopeSpans":[{{"spans":[{span}]}}]"#)
                    .replace(r#""spanId":"kept""#, r#""spanId":null"#),
                0,
            ),
        ] {
            let input = format!("[null,1,[],{{{fields}}}]");
            let events = process_to_events(input.as_bytes(), &EventBuildConfig::default()).unwrap();
            assert_eq!(events.len(), count, "{input}");
            if let Some(event) = events.first() {
                assert_eq!(event.span_id, "kept");
            }
        }
    }

    fn process_json(value: &Value, config: &EventBuildConfig) -> Result<Vec<OwnedEvent>, String> {
        process_to_events(&serde_json::to_vec(value).unwrap(), config)
    }

    fn string_attribute(key: &str, value: &str) -> Value {
        json!({"key": key, "value": {"stringValue": value}})
    }

    fn metadata_event(
        attributes: Vec<Value>,
        sdk_name: &str,
        scope_name: &str,
        scope_version: &str,
        telemetry_sdk_language: Option<&str>,
    ) -> OwnedEvent {
        let resource_attributes = telemetry_sdk_language
            .map(|language| vec![string_attribute("telemetry.sdk.language", language)])
            .unwrap_or_default();
        let input = json!([{
            "resource": {"attributes": resource_attributes},
            "scopeSpans": [{
                "scope": {"name": scope_name, "version": scope_version},
                "spans": [{
                    "traceId": "trace",
                    "spanId": "span",
                    "name": "chat",
                    "startTimeUnixNano": "1000000000",
                    "endTimeUnixNano": "2000000000",
                    "attributes": attributes
                }]
            }]
        }]);
        process_json(
            &input,
            &EventBuildConfig {
                ingestion_sdk_name: sdk_name.to_owned(),
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
    }

    #[test]
    fn claude_managed_agent_lifecycle_respects_provider_priority() {
        for (span_name, operation, explicit_type, expected_type, expected_name, skip_usage) in [
            (
                "anthropic.sessions.create",
                None,
                None,
                "SPAN",
                "anthropic.sessions.create",
                true,
            ),
            (
                "anthropic.messages.create",
                Some("chat"),
                None,
                "GENERATION",
                "anthropic.messages.create",
                false,
            ),
            (
                "anthropic.sessions.create",
                None,
                Some("generation"),
                "GENERATION",
                "anthropic.sessions.create",
                false,
            ),
            (
                "anthropic.session.turn",
                Some("invoke_agent"),
                None,
                "AGENT",
                "Researcher",
                true,
            ),
        ] {
            let mut attributes = vec![
                string_attribute("anthropic.session.id", "session"),
                string_attribute("gen_ai.conversation.id", "thread"),
                string_attribute("gen_ai.agent.name", "Researcher"),
                string_attribute("gen_ai.request.model", "claude"),
                json!({"key": "gen_ai.usage.input_tokens", "value": {"intValue": 4}}),
                json!({"key": "gen_ai.usage.cost", "value": {"doubleValue": 0.5}}),
            ];
            for (key, value) in [
                ("gen_ai.operation.name", operation),
                ("langfuse.observation.type", explicit_type),
            ] {
                if let Some(value) = value {
                    attributes.push(string_attribute(key, value));
                }
            }
            let input = json!([{"scopeSpans": [{
                "scope": {"name": "com.anthropic.claude-platform"},
                "spans": [{"traceId": "trace", "spanId": "span", "name": span_name, "attributes": attributes}]
            }]}]);
            let event = process_json(&input, &EventBuildConfig::default())
                .unwrap()
                .remove(0);
            assert_eq!(
                event.event_type, expected_type,
                "{span_name}, {explicit_type:?}"
            );
            assert_eq!(event.name, expected_name);
            assert_eq!(event.session_id.as_deref(), Some("session"));
            assert_eq!(
                event.provided_model_name.as_deref(),
                (!skip_usage).then_some("claude")
            );
            assert_eq!(
                event.provided_usage_details.get("input"),
                (!skip_usage).then_some(&json!(4))
            );
            assert_eq!(
                event.provided_cost_details.get("total"),
                (!skip_usage).then_some(&json!(0.5))
            );
            if skip_usage {
                assert!(event.provided_usage_details.is_empty());
                assert!(event.provided_cost_details.is_empty());
            }
        }
    }

    #[test]
    fn python_sdk_override_precedes_explicit_type_only_for_legacy_span() {
        for (version, language, explicit_type, expected) in [
            ("3.3.99", "python", "span", "GENERATION"),
            ("3.4.0", "python", "span", "SPAN"),
            ("3.3.0", "javascript", "span", "SPAN"),
            ("3.3.0", "python", "tool", "TOOL"),
        ] {
            let event = metadata_event(
                vec![
                    string_attribute("langfuse.observation.type", explicit_type),
                    string_attribute("langfuse.observation.model.name", ""),
                ],
                language,
                "langfuse-sdk",
                version,
                Some(language),
            );
            assert_eq!(
                event.event_type, expected,
                "{version}, {language}, {explicit_type}"
            );
        }
    }

    #[test]
    fn builds_owned_events_without_reencoding_the_batch() {
        let input = json!([{
            "resource": {"attributes": [{"key":"service.name","value":{"stringValue":"svc"}}]},
            "scopeSpans": [{"scope":{"name":"ai","version":"1"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[
                    {"key":"gen_ai.operation.name","value":{"stringValue":"chat"}},
                    {"key":"ai.prompt.messages","value":{"arrayValue":{"values":[]}}}
                ]
            }]}]
        }]);
        let events = process_json(
            &input,
            &EventBuildConfig {
                project_id: "p".into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].project_id, "p");
        assert_eq!(events[0].service_name.as_deref(), Some("svc"));
    }

    #[test]
    fn trusted_gateway_attribution_uses_caller_key_and_scope_version() {
        let input = json!([{
            "resource": {"attributes": [
                {"key":"service.name","value":{"stringValue":"provider"}},
                {"key":"service.version","value":{"stringValue":"provider-v1"}}
            ]},
            "scopeSpans": [{"scope":{"name":"gateway","version":"gateway-v2"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[{"key":"langfuse.observation.metadata.langfuse.gateway.api_key.id",
                    "value":{"stringValue":"caller-key"}}]
            }]}]
        }]);
        let events = process_json(
            &input,
            &EventBuildConfig {
                ingestion_api_key: "gateway:grant".into(),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].ingestion_api_key, "caller-key");
        assert_eq!(
            events[0].service_name.as_deref(),
            Some(AI_GATEWAY_SERVICE_NAME)
        );
        assert_eq!(events[0].service_version.as_deref(), Some("gateway-v2"));
    }

    #[test]
    fn missing_gateway_caller_key_skips_only_that_span() {
        let input = json!([{
            "resource": {"attributes": []},
            "scopeSpans": [{"scope":{"name":"gateway","version":"1"},"spans":[
                {"traceId":"trace","spanId":"missing","name":"chat",
                 "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                 "attributes":[]},
                {"traceId":"trace","spanId":"kept","name":"chat",
                 "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                 "attributes":[{"key":"langfuse.observation.metadata.langfuse.gateway.api_key.id",
                     "value":{"stringValue":"caller-key"}}]}
            ]}]
        }]);
        let events = process_json(
            &input,
            &EventBuildConfig {
                ingestion_api_key: "gateway:grant".into(),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].span_id, "kept");
    }

    #[test]
    fn ordinary_auth_clears_a_spoofed_gateway_service() {
        let input = json!([{
            "resource": {"attributes": [
                {"key":"service.name","value":{"stringValue":AI_GATEWAY_SERVICE_NAME}},
                {"key":"service.version","value":{"stringValue":"spoofed"}}
            ]},
            "scopeSpans": [{"scope":{"name":"ordinary","version":"1"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[]
            }]}]
        }]);
        let events = process_json(
            &input,
            &EventBuildConfig {
                ingestion_api_key: "project-key".into(),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].service_name, None);
        assert_eq!(events[0].service_version, None);
    }

    #[test]
    fn preserves_legacy_metadata_nesting_and_environment_normalization() {
        let input = json!([{
            "resource": {"attributes": [{"key":"deployment.environment","value":{"stringValue":"Langfuse-Prod"}}]},
            "scopeSpans": [{"scope":{"name":"otel.scope","version":"1","schemaUrl":"schema"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[{"key":"custom","value":{"stringValue":"value"}}]
            }]}]
        }]);
        let events = process_json(
            &input,
            &EventBuildConfig {
                project_id: "p".into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(events[0].environment, "prod");
        assert_eq!(events[0].metadata["attributes"]["custom"], json!("value"));
        assert_eq!(events[0].metadata.get("custom"), None);
        assert_eq!(events[0].metadata["scope"]["schemaUrl"], json!("schema"));
        assert_eq!(
            events[0].metadata["scope"]["attributes"]["custom"],
            Value::Null
        );
    }

    #[test]
    fn non_string_environment_uses_default_and_ingestion_cannot_bookmark() {
        let input = json!([{
            "resource": {"attributes": [{"key":"deployment.environment","value":{"stringValue":"prod"}}]},
            "scopeSpans": [{"scope":{"name":"otel.scope"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[
                    {"key":"langfuse.environment","value":{"intValue":"42"}},
                    {"key":"langfuse.trace.bookmarked","value":{"boolValue":true}}
                ]
            }]}]
        }]);
        let events = process_json(&input, &EventBuildConfig::default()).unwrap();
        assert_eq!(events[0].environment, "default");
        assert!(!events[0].bookmarked);
    }

    #[test]
    fn nullish_prompt_name_does_not_fall_through_an_explicit_empty_alias() {
        let input = json!([{
            "resource": {"attributes": []},
            "scopeSpans": [{"scope":{"name":"otel.scope"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[
                    {"key":"langfuse.observation.type","value":{"stringValue":"generation"}},
                    {"key":"langfuse.observation.prompt.name","value":{"stringValue":""}},
                    {"key":"langfuse.prompt.name","value":{"stringValue":"alias"}}
                ]
            }]}]
        }]);
        let events = process_json(&input, &EventBuildConfig::default()).unwrap();
        assert_eq!(events[0].prompt_name.as_deref(), Some(""));
    }

    #[test]
    fn recovers_internal_environment_from_failed_openrouter_request() {
        let input = json!([{
            "resource": {"attributes": [
                {"key":"service.name","value":{"stringValue":"openrouter"}}
            ]},
            "scopeSpans": [{"scope":{"name":"otel.scope"},"spans":[{
                "traceId":"trace","spanId":"span","name":"chat",
                "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                "attributes":[
                    {"key":"openrouter.source","value":{"stringValue":"openrouter"}},
                    {"key":"gen_ai.completion","value":{"stringValue":"{\"completion\":null,\"rawRequest\":{\"trace\":{\"environment\":\"langfuse-llm-as-a-judge\"}}}"}}
                ]
        }]}]
        }]);
        let events = process_json(&input, &EventBuildConfig::default()).unwrap();
        assert_eq!(events[0].environment, "llm-as-a-judge");
    }

    #[test]
    fn only_numeric_otlp_status_code_two_sets_error_level() {
        for (code, expected) in [
            (json!(2), "ERROR"),
            (json!("2"), "DEFAULT"),
            (json!("ERROR"), "DEFAULT"),
        ] {
            let input = json!([{
                "resource": {"attributes": []},
                "scopeSpans": [{"scope":{"name":"otel.scope"},"spans":[{
                    "traceId":"trace","spanId":"span","name":"chat",
                    "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                    "status":{"code": code}, "attributes": []
                }]}]
            }]);
            let events = process_json(&input, &EventBuildConfig::default()).unwrap();
            assert_eq!(events[0].level, expected);
        }
    }

    #[test]
    fn resolves_user_identity_from_resource_and_span_values() {
        let identity_event = |span_attributes: Vec<Value>, resource_attributes: Vec<Value>| {
            let input = json!([{
                "resource": {"attributes": resource_attributes},
                "scopeSpans": [{"scope":{"name":"otel.scope"},"spans":[{
                    "traceId":"trace","spanId":"span","name":"chat",
                    "startTimeUnixNano":"1000000000","endTimeUnixNano":"2000000000",
                    "attributes": span_attributes
                }]}]
            }]);
            process_json(&input, &EventBuildConfig::default())
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
        };
        for (span, resource, expected) in [
            (
                vec![string_attribute("user.id", "span-id")],
                vec![string_attribute("user.id", "resource-id")],
                Some("span-id"),
            ),
            (
                vec![string_attribute("user.name", "span-name")],
                vec![string_attribute("langfuse.user.id", "resource-id")],
                Some("resource-id"),
            ),
            (
                vec![string_attribute("user.name", "span-name")],
                vec![string_attribute("user.name", "resource-name")],
                Some("span-name"),
            ),
            (
                Vec::new(),
                vec![string_attribute("user.name", "resource-name")],
                Some("resource-name"),
            ),
        ] {
            assert_eq!(identity_event(span, resource).user_id.as_deref(), expected);
        }
        let encoded = metadata_event(
            vec![string_attribute(
                "langfuse.observation.metadata.langfuse_user_id",
                "\"metadata-id\"",
            )],
            "python",
            "langfuse-sdk",
            "5.0.0",
            None,
        );
        assert_eq!(encoded.user_id.as_deref(), Some("metadata-id"));
    }

    #[test]
    fn streaming_preserves_order_across_resources() {
        let input = json!([
            {
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": "otel.scope"},
                    "spans": [{
                        "traceId": "trace",
                        "spanId": "root",
                        "name": "root",
                        "startTimeUnixNano": "1000000000",
                        "endTimeUnixNano": "2000000000",
                        "attributes": []
                    }]
                }]
            },
            {
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": "otel.scope"},
                    "spans": [{
                        "traceId": "trace",
                        "spanId": "child",
                        "parentSpanId": "root",
                        "name": "child",
                        "startTimeUnixNano": "3000000000",
                        "endTimeUnixNano": "4000000000",
                        "attributes": []
                    }]
                }]
            }
        ]);
        let actual = process_json(&input, &EventBuildConfig::default()).unwrap();

        assert_eq!(actual[0].span_id, "root");
        assert_eq!(actual[1].span_id, "child");
    }

    #[test]
    fn decodes_per_key_metadata_only_for_structured_langfuse_sdk_majors() {
        let cases = [
            ("Python 4", "python", "langfuse-sdk", "4.99.0", None, false),
            (
                "Python 5",
                "  LANGFUSE-PYTHON ",
                "langfuse-sdk",
                "5.0.0",
                None,
                true,
            ),
            (
                "Python PEP 440 release candidate",
                "python",
                "langfuse-sdk",
                "5.0.0rc1",
                None,
                true,
            ),
            (
                "JavaScript 5",
                "@langfuse/otel",
                "langfuse-sdk",
                "5.99.1",
                None,
                false,
            ),
            (
                "JavaScript 6",
                "typescript",
                "langfuse-sdk-v2",
                "6.0.0",
                None,
                true,
            ),
            (
                "uppercase v version is rejected",
                "python",
                "langfuse-sdk",
                "V5.0.0",
                None,
                false,
            ),
            (
                "header wins over resource language",
                "python",
                "langfuse-sdk",
                "5.0.0",
                Some("nodejs"),
                true,
            ),
            (
                "language fallback",
                "unknown",
                "langfuse-sdk",
                "6.0.0-beta.1",
                Some("nodejs"),
                true,
            ),
            (
                "non-Langfuse scope",
                "python",
                "openinference",
                "5.0.0",
                None,
                false,
            ),
        ];

        for (name, sdk_name, scope_name, scope_version, language, should_decode) in cases {
            let event = metadata_event(
                vec![string_attribute(
                    "langfuse.observation.metadata.environment",
                    "\"prod\"",
                )],
                sdk_name,
                scope_name,
                scope_version,
                language,
            );
            let expected = if should_decode {
                json!("prod")
            } else {
                json!("\"prod\"")
            };
            assert_eq!(event.metadata["environment"], expected, "{name}");
        }
    }

    #[test]
    fn per_key_metadata_scalars_objects_and_arrays_match_a_blob() {
        let metadata = json!({
            "environment": "prod",
            "count": 42,
            "ratio": 1.5,
            "large": 9007199254740993u64,
            "enabled": true,
            "missing": null,
            "database": {"host": "localhost", "port": 5432},
            "steps": [{"name": "fetch", "args": {"url": "https://example.com"}}],
        });

        for domain in ["observation", "trace"] {
            let blob = metadata_event(
                vec![string_attribute(
                    &format!("langfuse.{domain}.metadata"),
                    &metadata.to_string(),
                )],
                "python",
                "langfuse-sdk",
                "5.0.0",
                None,
            );
            let per_key = metadata
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| {
                    string_attribute(
                        &format!("langfuse.{domain}.metadata.{}", key.as_str()),
                        &serde_json::to_string(value).unwrap(),
                    )
                })
                .collect();
            let decoded = metadata_event(per_key, "python", "langfuse-sdk", "5.0.0", None);

            assert_eq!(decoded.metadata, blob.metadata, "{domain}");
            assert_eq!(
                crate::otel::types::flatten_object_owned(decoded.metadata.clone()),
                crate::otel::types::flatten_object_owned(blob.metadata.clone()),
                "{domain} flattened metadata",
            );
        }
    }

    #[test]
    fn preserves_truncated_per_key_values_while_decoding_valid_siblings() {
        let event = metadata_event(
            vec![
                string_attribute(
                    "langfuse.observation.metadata.note",
                    r#"{"message":"truncated"#,
                ),
                string_attribute("langfuse.observation.metadata.environment", "\"prod\""),
            ],
            "python",
            "langfuse-sdk",
            "5.0.0",
            None,
        );

        assert_eq!(event.metadata["note"], json!(r#"{"message":"truncated"#));
        assert_eq!(event.metadata["environment"], json!("prod"));
    }
}
