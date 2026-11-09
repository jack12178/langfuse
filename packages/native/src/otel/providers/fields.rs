//! Provider identity, observation classification, and scalar field precedence.

use super::{is_meaningful, parse_javascript_number, select_first, trim_javascript};
use crate::otel::attributes::{
    is_js_truthy, javascript_string, value_as_f64, value_as_string, JsonParseCache,
};
use crate::otel::types::JsonObject;
use crate::otel::value::Value;

const MODEL_NAME_KEYS: &[&str] = &[
    "langfuse.observation.model.name",
    "litellm.provider.model",
    "gen_ai.response.model",
    "ai.model.id",
    "gen_ai.request.model",
    "llm.response.model",
    "llm.model_name",
    "model",
];
pub fn extract_name(
    span_name: &str,
    attributes: &JsonObject,
    observation_type: &str,
    scope_name: &str,
) -> String {
    for key in ["gen_ai.tool.name", "logfire.msg"] {
        if let Some(value) = attributes.get(key).filter(|value| is_js_truthy(value)) {
            return value_as_string(value).unwrap_or_default();
        }
    }
    for key in ["genkit:name", "ai.toolCall.name"] {
        if let Some(value) = attributes.get(key) {
            return value_as_string(value).unwrap_or_default();
        }
    }
    if let Some(operation) = attributes.get("ai.operationId") {
        let prefix = attributes
            .get("ai.telemetry.functionId")
            .filter(|value| is_js_truthy(value))
            .and_then(value_as_string)
            .map(|value| format!("{value}:"))
            .unwrap_or_default();
        return format!("{prefix}{}", value_as_string(operation).unwrap_or_default());
    }
    if span_name == "anthropic.session.turn"
        && observation_type == "AGENT"
        && scope_name == "com.anthropic.claude-platform"
        && attributes
            .get("gen_ai.operation.name")
            .and_then(Value::as_str)
            == Some("invoke_agent")
        && claude_managed_agent_session_id(attributes, scope_name).is_some()
    {
        if let Some(name) = attributes
            .get("gen_ai.agent.name")
            .and_then(Value::as_str)
            .filter(|name| !trim_javascript(name).is_empty())
        {
            return name.to_owned();
        }
    }
    span_name.to_owned()
}

pub fn extract_model_name(attributes: &JsonObject) -> Option<String> {
    if let Some(value) = select_first(attributes, MODEL_NAME_KEYS, is_js_truthy) {
        return value_as_string(value);
    }
    if attributes.get("genkit:metadata:subtype") == Some(&Value::String("model".into())) {
        return attributes
            .get("genkit:name")
            .filter(|value| is_js_truthy(value))
            .and_then(value_as_string);
    }
    None
}

pub fn extract_model_parameters(
    attributes: &JsonObject,
    scope_name: &str,
    parse_cache: &mut JsonParseCache,
) -> Option<Value> {
    let explicit = attributes
        .get("langfuse.observation.model.parameters")
        .and_then(|value| parse_cache.parse(Some(value)))
        .map(sanitize_model_params);
    if explicit.is_some() && scope_name != "ai" {
        return explicit;
    }
    if scope_name == "genkit-tracer" {
        let config = parse_cache
            .parse(attributes.get("genkit:input"))
            .filter(|value| !value.is_null())
            .or_else(|| attributes.get("genkit:input"))
            .and_then(|value| value.get("config").filter(|value| !value.is_null()))
            .cloned()
            .or_else(|| {
                parse_cache
                    .parse(attributes.get("genkit:output"))
                    .filter(|value| !value.is_null())
                    .or_else(|| attributes.get("genkit:output"))
                    .and_then(|value| value.get("request"))
                    .and_then(|value| value.get("config").filter(|value| !value.is_null()))
                    .cloned()
            });
        if let Some(config) = config {
            return Some(sanitize_model_params(&config));
        }
    }
    if scope_name == "ai" {
        let response_service_tier = parse_cache
            .parse(attributes.get("ai.response.providerMetadata"))
            .and_then(|value| value.get("openai"))
            .and_then(|value| value.get("serviceTier"));
        let mut params = JsonObject::new();
        for rule in AI_PARAM_RULES {
            insert_ai_param(&mut params, rule, attributes);
        }
        if let Some(value @ Value::String(_)) = response_service_tier {
            params.insert("service_tier".into(), value.clone());
        }
        if let Some(Value::Object(explicit)) = explicit {
            params.extend(explicit);
        }
        return Some(Value::Object(params));
    }

    for key in ["llm.invocation_parameters", "model_config"] {
        if let Some(value) = parse_cache.parse(attributes.get(key)) {
            return Some(sanitize_model_params(value));
        }
    }
    let mut params = JsonObject::new();
    for prefix in ["gen_ai.request.", "llm.invocation_parameters."] {
        for (key, value) in attributes {
            if let Some(parameter) = key.strip_prefix(prefix) {
                if parameter.as_utf8() != Some("model") {
                    params.insert(parameter, value.clone());
                }
            }
        }
    }
    Some(sanitize_model_params(&Value::Object(params)))
}

pub fn extract_tags(
    attributes: &JsonObject,
    parse_cache: &mut JsonParseCache,
    decode_metadata_values: bool,
) -> Vec<String> {
    let tag_keys = [
        "langfuse.trace.tags",
        "langfuse.tags",
        "langfuse.observation.metadata.langfuse_tags",
        "langfuse.trace.metadata.langfuse_tags",
        "ai.telemetry.metadata.tags",
        "tag.tags",
    ];
    let decoded_metadata_key = |key: &str| {
        decode_metadata_values
            && matches!(
                key,
                "langfuse.observation.metadata.langfuse_tags"
                    | "langfuse.trace.metadata.langfuse_tags"
            )
    };
    let value = tag_keys.iter().find_map(|key| {
        let raw = attributes.get(*key);
        let decoded = decoded_metadata_key(key)
            .then(|| parse_cache.parse(raw))
            .flatten()
            .or(raw);
        decoded.filter(|value| is_js_truthy(value)).cloned()
    });
    let Some(tag_value) = value else {
        return Vec::new();
    };
    match &tag_value {
        Value::Array(values) => values.iter().map(javascript_string).collect(),
        Value::String(text) if text.as_str().trim_start().starts_with('[') => parse_cache
            .parse(Some(&tag_value))
            .and_then(Value::as_array)
            .map(|values| values.iter().map(javascript_string).collect())
            .unwrap_or_else(|| vec![text.as_str().to_owned()]),
        Value::String(text) if text.as_str().contains(',') => text
            .as_str()
            .split(',')
            .map(|value| value.trim().to_owned())
            .collect(),
        Value::String(text) => vec![text.as_str().to_owned()],
        // The legacy method returns [] for a non-string, non-array top-level value. JS String
        // coercion applies to array members, not to the tag source itself.
        _ => Vec::new(),
    }
}

pub fn extract_completion_start_time(
    attributes: &JsonObject,
    start_time: &str,
    parse_cache: &mut JsonParseCache,
) -> Option<String> {
    if let Some(value) = attributes.get("langfuse.observation.completion_start_time") {
        // The legacy adapter accepts a valid date string and also the double-stringified form
        // emitted by older SDKs. Merely checking for `T` or `-` would persist arbitrary text.
        if let Some(value) = valid_date_string_with_cache(value, parse_cache) {
            return Some(value);
        }
    }
    let primary_offset = attributes
        .get("ai.response.msToFirstChunk")
        .or_else(|| attributes.get("ai.stream.msToFirstChunk"));
    let offset = match primary_offset {
        Some(value) if is_js_truthy(value) => value_as_f64(Some(value)),
        Some(_) => None,
        None => attributes
            .get("gen_ai.client.operation.time_to_first_chunk")
            .filter(|value| is_js_truthy(value))
            .and_then(|value| value_as_f64(Some(value)))
            .map(|seconds| seconds * 1000.0)
            .filter(|milliseconds| milliseconds.is_finite()),
    }?;
    let start = chrono::DateTime::parse_from_rfc3339(start_time).ok()?;
    let milliseconds = offset.ceil();
    if !milliseconds.is_finite() || milliseconds < i64::MIN as f64 || milliseconds > i64::MAX as f64
    {
        return None;
    }
    let duration = chrono::TimeDelta::try_milliseconds(milliseconds as i64)?;
    let timestamp = start.checked_add_signed(duration)?;
    Some(timestamp.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

/// Read the prompt version used by the event processor. The event shape accepts an integer JSON
/// number or a string containing only decimal digits; values such as `"12foo"` and `1.5` are
/// deliberately rejected even though the later IngestionService prompt lookup uses `parseInt`
/// once this field has been accepted.
pub fn extract_prompt_version_lookup(
    attributes: &JsonObject,
    parse_cache: &mut JsonParseCache,
) -> Option<f64> {
    let prompt = parse_cache.parse(attributes.get("ai.telemetry.metadata.langfusePrompt"));
    let value = select_first(
        attributes,
        &[
            "langfuse.observation.prompt.version",
            "langfuse.prompt.version",
        ],
        |value| !value.is_null(),
    )
    .or_else(|| {
        prompt
            .and_then(|value| value.get("version"))
            .filter(|value| !value.is_null())
    })?;
    match value {
        Value::Number(value) => value
            .as_f64()
            .filter(|value| value.is_finite() && value.fract() == 0.0),
        Value::String(value) => {
            let value = value.as_str();
            let value = value.trim();
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            value.parse::<f64>().ok()
        }
        _ => None,
    }
}

/// Convert the accepted prompt version to the UInt16 storage field. Lookup keeps the wider
/// number separately because prompt resolution happens before ClickHouse's UInt16 constraint.
pub fn extract_prompt_version(
    attributes: &JsonObject,
    parse_cache: &mut JsonParseCache,
) -> Option<u16> {
    extract_prompt_version_lookup(attributes, parse_cache).and_then(|value| {
        (value >= 0.0 && value <= u16::MAX as f64 && value.fract() == 0.0).then_some(value as u16)
    })
}

fn is_true(value: Option<&Value>) -> bool {
    matches!(value, Some(Value::Bool(true))) || value.and_then(Value::as_str) == Some("true")
}

pub fn extract_public(attributes: &JsonObject) -> bool {
    is_true(select_first(
        attributes,
        &["langfuse.trace.public", "langfuse.public"],
        |_| true,
    ))
}

pub fn extract_is_app_root(attributes: &JsonObject) -> bool {
    is_true(attributes.get("langfuse.internal.is_app_root"))
}

/// A session ID distinguishes managed-agent traces from ordinary Anthropic API calls.
pub fn claude_managed_agent_session_id<'a>(
    attributes: &'a JsonObject,
    scope_name: &str,
) -> Option<&'a str> {
    if scope_name != "com.anthropic.claude-platform"
        && !scope_name.starts_with("com.anthropic.sdk.")
    {
        return None;
    }
    attributes
        .get("anthropic.session.id")
        .and_then(Value::as_str)
        .filter(|id| !trim_javascript(id).is_empty())
}

fn is_claude_session_creation(attributes: &JsonObject, scope_name: &str, span_name: &str) -> bool {
    span_name == "anthropic.sessions.create"
        && !attributes
            .get("gen_ai.operation.name")
            .is_some_and(is_meaningful)
        && claude_managed_agent_session_id(attributes, scope_name).is_some()
}

pub fn should_skip_model_usage_and_cost(
    attributes: &JsonObject,
    scope_name: &str,
    span_name: &str,
    observation_type: &str,
) -> bool {
    matches!(
        attributes
            .get("gen_ai.operation.name")
            .and_then(Value::as_str),
        Some("invoke_agent" | "agent_step")
    ) || (observation_type == "SPAN"
        && is_claude_session_creation(attributes, scope_name, span_name))
}

const DIRECT_OBSERVATION_TYPES: &[(&str, &str)] = &[
    ("span", "SPAN"),
    ("generation", "GENERATION"),
    ("event", "EVENT"),
    ("embedding", "EMBEDDING"),
    ("agent", "AGENT"),
    ("tool", "TOOL"),
    ("chain", "CHAIN"),
    ("retriever", "RETRIEVER"),
    ("guardrail", "GUARDRAIL"),
    ("evaluator", "EVALUATOR"),
];

// Ordered provider declarations. Aliases within each source have equal priority.
const DECLARED_OBSERVATION_TYPES: &[(&str, &[(&str, &str)])] = &[
    (
        "openinference.span.kind",
        &[
            ("CHAIN", "CHAIN"),
            ("RETRIEVER", "RETRIEVER"),
            ("LLM", "GENERATION"),
            ("EMBEDDING", "EMBEDDING"),
            ("AGENT", "AGENT"),
            ("TOOL", "TOOL"),
            ("GUARDRAIL", "GUARDRAIL"),
            ("EVALUATOR", "EVALUATOR"),
        ],
    ),
    (
        "gen_ai.operation.name",
        &[
            ("chat", "GENERATION"),
            ("completion", "GENERATION"),
            ("text_completion", "GENERATION"),
            ("generate_content", "GENERATION"),
            ("generate", "GENERATION"),
            ("embeddings", "EMBEDDING"),
            ("invoke_agent", "AGENT"),
            ("create_agent", "AGENT"),
            ("execute_tool", "TOOL"),
        ],
    ),
    (
        "genkit:metadata:subtype",
        &[
            ("background-model", "GENERATION"),
            ("model", "GENERATION"),
            ("embedder", "EMBEDDING"),
            ("tool", "TOOL"),
            ("tool.v2", "TOOL"),
            ("retriever", "RETRIEVER"),
            ("evaluator", "EVALUATOR"),
        ],
    ),
];

fn declared_observation_type(attributes: &JsonObject) -> Option<&'static str> {
    if let Some(explicit) = attributes
        .get("langfuse.observation.type")
        .and_then(Value::as_str)
    {
        if let Some((_, kind)) = DIRECT_OBSERVATION_TYPES
            .iter()
            .find(|(alias, _)| *alias == explicit)
        {
            return Some(kind);
        }
    }
    DECLARED_OBSERVATION_TYPES
        .iter()
        .find_map(|(key, aliases)| {
            let value = attributes.get(*key)?.as_str()?;
            aliases
                .iter()
                .find_map(|(alias, kind)| (*alias == value).then_some(*kind))
        })
}

fn legacy_python_generation(
    attributes: &JsonObject,
    resource: &JsonObject,
    scope_name: &str,
    scope_version: Option<&str>,
) -> bool {
    if attributes
        .get("langfuse.observation.type")
        .and_then(Value::as_str)
        != Some("span")
        || scope_name != "langfuse-sdk"
        || resource
            .get("telemetry.sdk.language")
            .and_then(Value::as_str)
            != Some("python")
    {
        return false;
    }
    // The compatibility rule compares major/minor only; 3.3 patch releases remain affected.
    let mut version = scope_version.unwrap_or_default().split('.');
    let major = version.next().and_then(parse_javascript_number);
    let minor = version.next().and_then(parse_javascript_number);
    if major.is_some_and(|major| {
        major > 3.0 || (major == 3.0 && minor.is_some_and(|minor| minor > 3.0))
    }) {
        return false;
    }
    [
        "langfuse.observation.model.name",
        "langfuse.observation.cost_details",
        "langfuse.observation.usage_details",
        "langfuse.observation.completion_start_time",
        "langfuse.observation.model.parameters",
        "langfuse.observation.prompt.name",
        "langfuse.observation.prompt.version",
    ]
    .iter()
    .any(|key| attributes.contains_key(*key))
}

fn has_meaningful_attribute(attributes: &JsonObject, keys: &[&str]) -> bool {
    select_first(attributes, keys, is_meaningful).is_some()
}

fn vercel_observation_type(attributes: &JsonObject) -> Option<&'static str> {
    let matches = |operations: &[&str]| {
        attributes
            .get("operation.name")
            .and_then(Value::as_str)
            .is_some_and(|name| {
                operations
                    .iter()
                    .any(|operation| name.starts_with(operation))
            })
            || attributes
                .get("ai.operationId")
                .and_then(Value::as_str)
                .is_some_and(|id| operations.contains(&id))
    };
    for (operations, kind) in [
        (
            &[
                "ai.generateText.doGenerate",
                "ai.streamText.doStream",
                "ai.generateObject.doGenerate",
                "ai.streamObject.doStream",
            ][..],
            "GENERATION",
        ),
        (
            &["ai.embedMany.doEmbed", "ai.embed.doEmbed"][..],
            "EMBEDDING",
        ),
    ] {
        if matches(operations) {
            return has_meaningful_attribute(
                attributes,
                &[
                    "langfuse.observation.model.name",
                    "ai.model.id",
                    "gen_ai.request.model",
                    "gen_ai.response.model",
                ],
            )
            .then_some(kind);
        }
    }
    matches(&["ai.toolCall"]).then_some("TOOL")
}

pub fn observation_type(
    attributes: &JsonObject,
    resource: &JsonObject,
    scope_name: &str,
    scope_version: Option<&str>,
    span_name: &str,
) -> String {
    legacy_python_generation(attributes, resource, scope_name, scope_version)
        .then_some("GENERATION")
        .or_else(|| declared_observation_type(attributes))
        .or_else(|| vercel_observation_type(attributes))
        .or_else(|| {
            [
                (
                    &[
                        "gen_ai.tool.name",
                        "gen_ai.tool.call.id",
                        "flue.tool.name",
                        "flue.tool.call_id",
                    ][..],
                    "TOOL",
                ),
                (&["flue.task.id", "flue.task.agent"][..], "AGENT"),
            ]
            .into_iter()
            .find_map(|(keys, kind)| has_meaningful_attribute(attributes, keys).then_some(kind))
        })
        .or(match (scope_name, span_name) {
            ("livekit-agents", "agent_turn" | "start_agent_activity") => Some("AGENT"),
            ("livekit-agents", "function_tool") => Some("TOOL"),
            _ => None,
        })
        .or_else(|| is_claude_session_creation(attributes, scope_name, span_name).then_some("SPAN"))
        // ai.model.id implies a generation only together with a Vercel operation.
        .or_else(|| {
            has_meaningful_attribute(
                attributes,
                &[
                    "langfuse.observation.model.name",
                    "gen_ai.request.model",
                    "gen_ai.response.model",
                    "llm.model_name",
                    "model",
                ],
            )
            .then_some("GENERATION")
        })
        .unwrap_or("SPAN")
        .to_owned()
}
fn valid_date_string_with_cache(value: &Value, parse_cache: &mut JsonParseCache) -> Option<String> {
    let candidate = match value {
        Value::String(value) => Some(value.as_str().to_owned()),
        _ => None,
    }?;
    if is_valid_date_string(&candidate) {
        return Some(candidate);
    }
    let parsed = parse_cache.parse(Some(value))?.as_str()?.to_owned();
    is_valid_date_string(&parsed).then_some(parsed)
}

fn is_valid_date_string(value: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(value).is_ok()
        || chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").is_ok()
        || chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").is_ok()
}
const AI_PARAM_RULES: &[(&str, &[&str])] = &[
    ("maxSteps", &["ai.settings.maxSteps"]),
    ("toolChoice", &["ai.prompt.toolChoice"]),
    ("maxTokens", &["gen_ai.request.max_tokens"]),
    (
        "finishReason",
        &["gen_ai.response.finish_reasons", "gen_ai.finishReason"],
    ),
    ("system", &["gen_ai.system", "ai.model.provider"]),
    ("maxRetries", &["ai.settings.maxRetries"]),
    ("mode", &["ai.settings.mode"]),
    ("temperature", &["gen_ai.request.temperature"]),
    ("service_tier", &["gen_ai.request.service_tier"]),
];

fn insert_ai_param(
    object: &mut JsonObject,
    (output, inputs): &(&str, &[&str]),
    attributes: &JsonObject,
) {
    let value = inputs.iter().find_map(|key| attributes.get(*key));
    object.insert(
        (*output).to_owned().into(),
        value.map_or(Value::Null, |value| match value {
            Value::Null | Value::String(_) => value.clone(),
            _ => Value::String(javascript_string(value).into()),
        }),
    );
}

/// Apply the same storage-safe model-parameter conversion as the TypeScript processor. Strings
/// and numbers remain typed; booleans, arrays, objects, and null become their JSON string form.
fn sanitize_model_params(value: &Value) -> Value {
    let Some(object) = value.as_object() else {
        return value.clone();
    };
    Value::Object(
        object
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Value::String(_) | Value::Number(_) => value.clone(),
                    _ => Value::String(
                        crate::otel::attributes::javascript_json_stringify(value).into(),
                    ),
                };
                (key.clone(), value)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::value::json;

    #[test]
    fn completion_start_time_accepts_gen_ai_time_to_first_chunk_seconds() {
        let attributes = json!({
            "gen_ai.client.operation.time_to_first_chunk": 1.18922124899996
        });

        assert_eq!(
            extract_completion_start_time(
                attributes.as_object().unwrap(),
                "2024-01-01T00:00:00.000Z",
                &mut JsonParseCache::default(),
            ),
            Some("2024-01-01T00:00:01.190Z".to_owned())
        );
    }

    #[test]
    fn completion_start_time_ignores_zero_time_to_first_chunk() {
        let attributes = json!({
            "ai.response.msToFirstChunk": 0,
        });

        assert_eq!(
            extract_completion_start_time(
                attributes.as_object().unwrap(),
                "2024-01-01T00:00:00.000Z",
                &mut JsonParseCache::default(),
            ),
            None
        );
    }

    #[test]
    fn model_name_prefers_litellm_dispatched_model() {
        let attrs = json!({
            "gen_ai.request.model": "requested-model",
            "litellm.provider.model": "served-model",
        });
        assert_eq!(
            extract_model_name(attrs.as_object().unwrap()).as_deref(),
            Some("served-model")
        );
    }

    #[test]
    fn model_parameters_keep_js_coercion_and_prefix_precedence() {
        let attributes = json!({
            "gen_ai.request.temperature": [1, 2],
            "llm.invocation_parameters.temperature": "llm",
            "ai.settings.maxRetries": null,
        });
        let params = extract_model_parameters(
            attributes.as_object().unwrap(),
            "other",
            &mut JsonParseCache::default(),
        )
        .unwrap();
        assert_eq!(params["temperature"], json!("llm"));
        assert_eq!(params["maxRetries"], Value::Null);
    }

    #[test]
    fn prompt_version_accepts_only_the_event_shape_and_preserves_lookup_range() {
        for (primary, fallback, lookup, stored) in [
            (None, json!("70000"), Some(70000.0), None),
            (None, json!(12.5), None, None),
            (None, json!("12foo"), None, None),
            (None, json!(12), Some(12.0), Some(12)),
            (Some(json!(null)), json!("12"), Some(12.0), Some(12)),
            (Some(json!("")), json!("12"), None, None),
        ] {
            let mut attributes =
                JsonObject::from_iter([("langfuse.prompt.version".into(), fallback)]);
            if let Some(primary) = primary {
                attributes.insert("langfuse.observation.prompt.version".into(), primary);
            }
            let mut cache = JsonParseCache::default();
            assert_eq!(
                extract_prompt_version_lookup(&attributes, &mut cache),
                lookup,
                "{attributes:?}"
            );
            assert_eq!(
                extract_prompt_version(&attributes, &mut cache),
                stored,
                "{attributes:?}"
            );
        }
    }

    #[test]
    fn a_model_attribute_is_a_generation_fallback() {
        let attributes = json!({"langfuse.observation.model.name": "claude"});
        assert_eq!(
            observation_type(
                attributes.as_object().unwrap(),
                &JsonObject::new(),
                "langfuse-sdk",
                None,
                "run"
            ),
            "GENERATION"
        );
        for (operation, operation_id, expected) in [
            (Some("unrelated"), "ai.embed.doEmbed", "EMBEDDING"),
            (None, "ai.embed.doEmbed function-id", "GENERATION"),
        ] {
            let attributes = json!({
                "gen_ai.request.model": "embedding-model",
                "operation.name": operation,
                "ai.operationId": operation_id,
            });
            assert_eq!(
                observation_type(
                    attributes.as_object().unwrap(),
                    &JsonObject::new(),
                    "ai",
                    None,
                    "run"
                ),
                expected,
                "{operation:?}, {operation_id:?}",
            );
        }
    }

    #[test]
    fn tags_use_javascript_fallback_and_string_coercion() {
        let attributes = json!({
            "langfuse.trace.tags": 0,
            "langfuse.tags": [null, false, 1, {"kind": "object"}],
        });
        assert_eq!(
            extract_tags(
                attributes.as_object().unwrap(),
                &mut JsonParseCache::default(),
                false,
            ),
            ["null", "false", "1", "[object Object]"]
        );

        let attributes = json!({"langfuse.trace.tags": 42});
        assert!(extract_tags(
            attributes.as_object().unwrap(),
            &mut JsonParseCache::default(),
            false,
        )
        .is_empty());
    }

    #[test]
    fn tags_keep_plain_text_and_decode_structured_metadata() {
        for (attributes, decode, expected) in [
            (
                json!({"langfuse.trace.tags": "production"}),
                false,
                vec!["production".to_owned()],
            ),
            (
                json!({"langfuse.observation.metadata.langfuse_tags": "\"prod,blue\""}),
                true,
                vec!["prod".to_owned(), "blue".to_owned()],
            ),
            (
                json!({"ai.telemetry.metadata.tags": "null"}),
                true,
                vec!["null".to_owned()],
            ),
            (
                json!({"ai.telemetry.metadata.tags": "false"}),
                true,
                vec!["false".to_owned()],
            ),
        ] {
            assert_eq!(
                extract_tags(
                    attributes.as_object().unwrap(),
                    &mut JsonParseCache::default(),
                    decode,
                ),
                expected,
            );
        }
    }
}
