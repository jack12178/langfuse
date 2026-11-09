//! Provider input/output selection and reconstruction from flattened attributes.

use super::select_first;
use crate::otel::attributes::{
    is_js_truthy, javascript_json_stringify, javascript_string, value_as_string, JsonParseCache,
};
use crate::otel::input::SpanEvent;
use crate::otel::types::JsonObject;
use crate::otel::value::{JsString, Value};
use std::collections::HashSet;

#[derive(Clone, Debug, Default)]
pub struct ProviderIO {
    pub input: Option<Value>,
    pub output: Option<Value>,
    pub filtered_attributes: JsonObject,
}

type Payload = [Option<Value>; 2];

const SDK_IO: [[&str; 2]; 2] = [
    ["langfuse.trace.input", "langfuse.trace.output"],
    ["langfuse.observation.input", "langfuse.observation.output"],
];
const GENKIT_IO: [&str; 2] = ["genkit:input", "genkit:output"];
const AI_INPUT_KEYS: &[&str] = &["ai.prompt.messages", "ai.prompt", "ai.toolCall.args"];
const AI_OUTPUT_KEYS: &[&str] = &[
    "ai.response.text",
    "ai.result.text",
    "ai.toolCall.result",
    "ai.response.object",
    "ai.result.object",
    "ai.response.toolCalls",
    "ai.result.toolCalls",
];
const FLUE_IO: [&[&str]; 2] = [
    &[
        "flue.turn.input",
        "flue.tool.arguments",
        "flue.task.prompt",
        "flue.workflow.payload",
    ],
    &[
        "flue.turn.output",
        "flue.tool.result",
        "flue.task.result",
        "flue.operation.result",
        "flue.workflow.result",
    ],
];
const VERTEX_IO: [[&str; 2]; 2] = [
    [
        "gcp.vertex.agent.llm_request",
        "gcp.vertex.agent.llm_response",
    ],
    [
        "gcp.vertex.agent.tool_call_args",
        "gcp.vertex.agent.tool_response",
    ],
];
const LOGFIRE_IO: [&str; 2] = ["prompt", "all_messages_events"];
const LIVEKIT_INPUT_KEYS: &[&str] = &[
    "lk.input_text",
    "lk.pii.input_text",
    "lk.user_transcript",
    "lk.pii.user_transcript",
    "lk.chat_ctx",
    "lk.pii.chat_ctx",
];
const LIVEKIT_OUTPUT_KEYS: &[&str] = &[
    "lk.function_tool.output",
    "lk.pii.function_tool.output",
    "lk.response.text",
    "lk.pii.response.text",
];
const DIRECT_IO: [[&str; 2]; 4] = [
    ["mlflow.spanInputs", "mlflow.spanOutputs"],
    ["traceloop.entity.input", "traceloop.entity.output"],
    ["input.value", "output.value"],
    ["input", "output"],
];
const PYDANTIC_IO: [&str; 2] = ["pydantic_ai.all_messages", "final_result"];
const TOOL_IO: [&str; 2] = ["tool_arguments", "tool_response"];
const ANTHROPIC_TOOL_RESULT: &str = "anthropic.tool_result.content";
const GEN_AI_IO: [[&str; 2]; 2] = [
    ["gen_ai.input.messages", "gen_ai.output.messages"],
    ["gen_ai.tool.call.arguments", "gen_ai.tool.call.result"],
];
const FLATTENED_IO: [[&str; 2]; 2] = [
    ["gen_ai.prompt", "gen_ai.completion"],
    ["llm.input_messages", "llm.output_messages"],
];
fn io_values(attributes: &JsonObject, keys: [&str; 2]) -> Payload {
    keys.map(|key| attributes.get(key).cloned())
}

fn has_io(payload: &Payload, predicate: impl Fn(&Value) -> bool) -> bool {
    payload.iter().flatten().any(predicate)
}

/// Extract provider payloads and the legacy diagnostic attribute remainder.
pub fn extract_input_output(
    scope_name: &str,
    attributes: &JsonObject,
    events: Vec<SpanEvent>,
    domain: Option<&str>,
    parse_cache: &mut JsonParseCache,
) -> ProviderIO {
    let [input, output] = provider_payload(scope_name, attributes, events, domain, parse_cache);
    ProviderIO {
        input,
        output,
        filtered_attributes: filter_attributes(attributes, scope_name),
    }
}

fn provider_payload(
    scope_name: &str,
    attributes: &JsonObject,
    mut events: Vec<SpanEvent>,
    domain: Option<&str>,
    parse_cache: &mut JsonParseCache,
) -> Payload {
    // Trace attributes have truthy precedence; a null-only SDK pair falls through.
    let sdk = [0, 1].map(|side| {
        (domain == Some("trace"))
            .then(|| {
                attributes
                    .get(SDK_IO[0][side])
                    .filter(|value| is_js_truthy(value))
            })
            .flatten()
            .or_else(|| attributes.get(SDK_IO[1][side]))
            .cloned()
    });
    if has_io(&sdk, |value| !value.is_null()) {
        return sdk;
    }
    match scope_name {
        "genkit-tracer" => {
            return [0, 1].map(|side| {
                let raw = attributes.get(GENKIT_IO[side]);
                let value = parse_cache
                    .parse(raw)
                    .filter(|value| !value.is_null())
                    .or(raw)?;
                Some(
                    value
                        .get(["messages", "message"][side])
                        .filter(|value| !value.is_null())
                        .unwrap_or(value)
                        .clone(),
                )
            })
        }
        "ai" => {
            let mut payload = [
                select_first(attributes, AI_INPUT_KEYS, |_| true).cloned(),
                ai_output(attributes),
            ];
            if let Some(fallback) = gen_ai_io(attributes, parse_cache) {
                for (value, fallback) in payload.iter_mut().zip(fallback) {
                    if value.as_ref().is_none_or(Value::is_null) {
                        *value = fallback;
                    }
                }
            }
            return payload;
        }
        "@flue/opentelemetry" => {
            let payload = FLUE_IO.map(|keys| {
                let raw = select_first(attributes, keys, |value| !value.is_null());
                parse_cache
                    .parse(raw)
                    .filter(|value| !value.is_null())
                    .or(raw)
                    .cloned()
            });
            if has_io(&payload, |_| true) {
                return payload;
            }
        }
        _ => {}
    }

    // New GenAI events carry the same message attributes as spans.
    for event in &mut events {
        if event.name() == Some("gen_ai.client.inference.operation.details") {
            if let Some(payload) = gen_ai_io(&event_to_map(event), parse_cache) {
                return payload;
            }
        }
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for event in &mut events {
        match event.name() {
            Some(
                "gen_ai.system.message"
                | "gen_ai.user.message"
                | "gen_ai.assistant.message"
                | "gen_ai.tool.message",
            ) => inputs.push(event_to_message(event)),
            Some("gen_ai.choice") => outputs.push(Value::Object(event_to_map(event))),
            _ => {}
        }
    }
    if !inputs.is_empty() || !outputs.is_empty() {
        return [
            nonempty_array(inputs),
            match outputs.len() {
                1 => outputs.pop(),
                _ => nonempty_array(outputs),
            },
        ];
    }
    // Semantic Kernel wraps provider attributes in prompt/completion events.
    let legacy = ["gen_ai.content.prompt", "gen_ai.content.completion"]
        .map(|name| events.iter().position(|event| event.name() == Some(name)));
    if legacy.iter().any(Option::is_some) {
        return [0, 1].map(|side| {
            legacy[side].map(|index| {
                let attributes = event_to_map(&mut events[index]);
                provider_payload(scope_name, &attributes, Vec::new(), None, parse_cache)[side]
                    .take()
                    .filter(is_js_truthy)
                    .unwrap_or(Value::Object(attributes))
            })
        });
    }

    let vertex = [0, 1].map(|side| {
        attributes
            .get(VERTEX_IO[0][side])
            .filter(|value| is_js_truthy(value) && value.as_str() != Some("{}"))
            .or_else(|| attributes.get(VERTEX_IO[1][side]))
            .cloned()
    });
    accepted_payload(vertex, is_js_truthy)
        .or_else(|| accepted_payload(io_values(attributes, LOGFIRE_IO), is_js_truthy))
        .or_else(|| accepted_payload(livekit_io(attributes), is_js_truthy))
        .or_else(|| logfire_events(attributes, parse_cache))
        .or_else(|| {
            DIRECT_IO
                .iter()
                .find_map(|&keys| accepted_payload(io_values(attributes, keys), is_js_truthy))
        })
        .or_else(|| {
            if scope_name != "pydantic-ai" {
                return None;
            }
            let [input, output] = io_values(attributes, PYDANTIC_IO);
            let input = input.map(|value| {
                prepend_system_instructions(
                    value,
                    attributes.get("gen_ai.system_instructions"),
                    parse_cache,
                )
            });
            accepted_payload([input, output], is_js_truthy)
        })
        .or_else(|| accepted_payload(io_values(attributes, TOOL_IO), is_js_truthy))
        .or_else(|| {
            FLATTENED_IO.iter().find_map(|keys| {
                accepted_payload(
                    [
                        flattened_payload(attributes, keys[0]),
                        flattened_payload(attributes, keys[1]),
                    ],
                    |_| true,
                )
            })
        })
        .or_else(|| gen_ai_io(attributes, parse_cache))
        .unwrap_or_default()
}

fn accepted_payload(payload: Payload, predicate: impl Fn(&Value) -> bool) -> Option<Payload> {
    has_io(&payload, predicate).then_some(payload)
}

fn logfire_events(attributes: &JsonObject, parse_cache: &mut JsonParseCache) -> Option<Payload> {
    let events = parse_cache.parse(attributes.get("events"))?.as_array()?;
    let is_choice =
        |event: &&Value| event.get("event.name").and_then(Value::as_str) == Some("gen_ai.choice");
    let output = events.iter().find(is_choice).cloned();
    let input = nonempty_array(
        events
            .iter()
            .filter(|event| !is_choice(event))
            .cloned()
            .collect(),
    );
    accepted_payload([input, output], |_| true)
}

fn nonempty_array(values: Vec<Value>) -> Option<Value> {
    (!values.is_empty()).then_some(Value::Array(values))
}

fn livekit_io(attributes: &JsonObject) -> Payload {
    let user_input = select_first(
        attributes,
        &["lk.user_input", "lk.pii.user_input"],
        is_js_truthy,
    );
    let input = select_first(attributes, LIVEKIT_INPUT_KEYS, |value| !value.is_null())
        .cloned()
        .or_else(|| {
            attributes
                .get("lk.pii.instructions")
                .filter(|value| is_js_truthy(value))
                .map(|instructions| {
                    let mut messages = vec![json_message("system", instructions.clone())];
                    if let Some(user) = user_input {
                        messages.push(json_message("user", user.clone()));
                    }
                    Value::Array(messages)
                })
        })
        .or_else(|| user_input.cloned())
        .or_else(|| attributes.get("lk.pii.function_tool.arguments").cloned());
    let output = select_first(attributes, LIVEKIT_OUTPUT_KEYS, is_js_truthy)
        .or_else(|| {
            attributes
                .get("lk.pii.response.function_calls")
                .filter(|value| is_js_truthy(value) && value.as_str() != Some("[]"))
        })
        .cloned();
    [input, output]
}

/// Remove payload and metadata carriers before putting the remainder under `metadata.attributes`.
pub fn filter_attributes(attributes: &JsonObject, scope_name: &str) -> JsonObject {
    const GATEWAY_ATTRIBUTES: &[&str] = &[
        "langfuse.trace.name",
        "user.id",
        "session.id",
        "langfuse.trace.tags",
        "langfuse.trace.public",
        "langfuse.trace.metadata",
        "langfuse.trace.input",
        "langfuse.trace.output",
        "langfuse.observation.type",
        "langfuse.observation.metadata",
        "langfuse.observation.level",
        "langfuse.observation.status_message",
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.observation.skills_available",
        "langfuse.observation.completion_start_time",
        "langfuse.observation.model.name",
        "langfuse.observation.model.parameters",
        "langfuse.observation.usage_details",
        "langfuse.observation.cost_details",
        "langfuse.observation.prompt.name",
        "langfuse.observation.prompt.version",
        "langfuse.environment",
        "langfuse.release",
        "langfuse.version",
        "langfuse.internal.as_root",
        "langfuse.internal.is_app_root",
        "langfuse.user.id",
        "langfuse.session.id",
        "langfuse.experiment.id",
        "langfuse.experiment.name",
        "langfuse.experiment.metadata",
        "langfuse.experiment.description",
        "langfuse.experiment.dataset.id",
        "langfuse.experiment.item.id",
        "langfuse.experiment.item.version",
        "langfuse.experiment.item.metadata",
        "langfuse.experiment.item.root_observation_id",
        "langfuse.experiment.item.expected_output",
        "langfuse.evaluator.id",
        "langfuse.evaluation.rule.id",
        "langfuse.evaluator.execution.is_test",
    ];
    const PAYLOAD_GROUPS: &[&[&str]] = &[
        &SDK_IO[0],
        &SDK_IO[1],
        &GENKIT_IO,
        AI_INPUT_KEYS,
        AI_OUTPUT_KEYS,
        FLUE_IO[0],
        FLUE_IO[1],
        &VERTEX_IO[0],
        &VERTEX_IO[1],
        &LOGFIRE_IO,
        LIVEKIT_INPUT_KEYS,
        LIVEKIT_OUTPUT_KEYS,
        &DIRECT_IO[0],
        &DIRECT_IO[1],
        &DIRECT_IO[2],
        &DIRECT_IO[3],
        &PYDANTIC_IO,
        &GEN_AI_IO[0],
        &GEN_AI_IO[1],
        &[
            ANTHROPIC_TOOL_RESULT,
            "events",
            "lk.user_input",
            "lk.pii.user_input",
            "lk.pii.instructions",
            "lk.pii.function_tool.arguments",
            "lk.pii.response.function_calls",
            "gen_ai.system_instructions",
            "langfuse.observation.skills_available",
        ],
    ];
    const PREFIXES: &[&str] = &[
        "gen_ai.prompt",
        "gen_ai.completion",
        "llm.input_messages",
        "llm.output_messages",
        "langfuse.trace.metadata",
        "langfuse.observation.metadata",
        "ai.telemetry.metadata",
    ];
    attributes
        .iter()
        .filter(|(key, _)| {
            let key = key.as_str();
            !PAYLOAD_GROUPS.iter().any(|keys| keys.contains(&key))
                && !PREFIXES.iter().any(|prefix| key.starts_with(prefix))
                && !(scope_name == "langfuse-ai-gateway" && GATEWAY_ATTRIBUTES.contains(&key))
        })
        .map(|(key, value)| {
            let value = match value {
                Value::String(_) => value.clone(),
                _ => Value::String(javascript_json_stringify(value).into()),
            };
            (key.clone(), value)
        })
        .collect()
}
fn ai_output(attributes: &JsonObject) -> Option<Value> {
    if let (Some(text), Some(calls)) = (
        attributes.get("ai.response.text"),
        attributes.get("ai.response.toolCalls"),
    ) {
        return Some(Value::String(
            crate::otel::attributes::javascript_json_stringify(&Value::Object(
                JsonObject::from_iter([
                    ("role".into(), "assistant".into()),
                    ("content".into(), text.clone()),
                    ("tool_calls".into(), calls.clone()),
                ]),
            ))
            .into(),
        ));
    }
    attributes
        .get(AI_OUTPUT_KEYS[0])
        .filter(|value| is_js_truthy(value))
        .or_else(|| select_first(attributes, &AI_OUTPUT_KEYS[1..], |_| true))
        .cloned()
}

fn gen_ai_io(attributes: &JsonObject, parse_cache: &mut JsonParseCache) -> Option<Payload> {
    for (index, keys) in GEN_AI_IO.into_iter().enumerate() {
        let [input, mut output] = keys.map(|key| {
            attributes
                .get(key)
                .filter(|value| is_js_truthy(value))
                .cloned()
        });
        if index == 1 && attributes.get(keys[1]).is_none_or(Value::is_null) {
            output = attributes
                .get(ANTHROPIC_TOOL_RESULT)
                .filter(|value| is_js_truthy(value))
                .cloned();
        }
        if input.is_some() || output.is_some() {
            return Some([
                input.map(|value| {
                    if index == 0 {
                        prepend_system_instructions(
                            value,
                            attributes.get("gen_ai.system_instructions"),
                            parse_cache,
                        )
                    } else {
                        value
                    }
                }),
                output,
            ]);
        }
    }
    None
}

fn event_to_message(event: &mut SpanEvent) -> Value {
    let name = event.name().unwrap_or_default();
    let role = name
        .strip_prefix("gen_ai.")
        .unwrap_or(name)
        .strip_suffix(".message")
        .unwrap_or(name);
    let mut message = JsonObject::new();
    message.insert("role".into(), Value::String(role.to_owned().into()));
    message.extend(event_to_map(event));
    Value::Object(message)
}

fn event_to_map(event: &mut SpanEvent) -> JsonObject {
    event
        .attributes
        .take()
        .map(|attributes| crate::otel::attributes::attributes_to_map(attributes.into_value()))
        .unwrap_or_default()
}

fn json_message(role: &str, content: Value) -> Value {
    let content = match content {
        Value::String(content) => Value::String(content),
        content => Value::String(value_as_string(&content).unwrap_or_default().into()),
    };
    Value::Object(JsonObject::from_iter([
        ("role".into(), Value::String(role.into())),
        ("content".into(), content),
    ]))
}

const MAX_ARRAY_SLOTS: usize = 10_001;
const MAX_PATH_DEPTH: usize = 64;

fn flattened_payload(attributes: &JsonObject, prefix: &str) -> Option<Value> {
    // Falsy exact values fall through to dotted reconstruction.
    if let Some(value) = attributes.get(prefix).filter(|value| is_js_truthy(value)) {
        return Some(value.clone());
    }
    let marker = format!("{prefix}.");
    let paths = attributes
        .iter()
        .filter_map(|(key, value)| {
            let parts = split_wtf8_path(&key.strip_prefix(&marker)?);
            let valid = parts.len() <= MAX_PATH_DEPTH
                && parts.iter().all(|part| {
                    !part.is_empty()
                        && !matches!(
                            part.as_utf8(),
                            Some("__proto__" | "constructor" | "prototype")
                        )
                        && canonical_array_index_js(part)
                            .is_none_or(|index| index < MAX_ARRAY_SLOTS)
                });
            valid.then_some((parts, value))
        })
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return None;
    }
    let mut root = if paths
        .iter()
        .any(|(parts, _)| parts.len() > 1 && canonical_array_index_js(&parts[0]).is_some())
    {
        Value::Array(Vec::new())
    } else {
        Value::Object(JsonObject::new())
    };
    let mut created = HashSet::new();
    let mut array_slots = 0usize;
    for (parts, value) in paths {
        insert_path(
            &mut root,
            &parts,
            value.clone(),
            &mut created,
            &[],
            &mut array_slots,
        );
    }
    Some(root)
}

fn split_wtf8_path(value: &JsString) -> Vec<JsString> {
    value
        .as_bytes()
        .split(|byte| *byte == b'.')
        .map(|part| JsString::from_wtf8(part.to_vec()).expect("WTF-8 path segment"))
        .collect()
}

fn canonical_array_index(value: &str) -> Option<usize> {
    if value == "0"
        || (value
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_digit() && *byte != b'0')
            && value.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return value
            .parse::<usize>()
            .ok()
            .filter(|index| *index < usize::MAX);
    }
    None
}

fn canonical_array_index_js(value: &JsString) -> Option<usize> {
    value.as_utf8().and_then(canonical_array_index)
}

fn insert_path(
    current: &mut Value,
    parts: &[JsString],
    value: Value,
    created: &mut HashSet<Vec<JsString>>,
    current_path: &[JsString],
    array_slots: &mut usize,
) {
    if current.is_null() {
        if !created.contains(&current_path.to_vec()) {
            return;
        }
        *current = if parts.first().and_then(canonical_array_index_js).is_some() {
            Value::Array(Vec::new())
        } else {
            Value::Object(JsonObject::new())
        };
    }
    let Some(key) = parts.first() else {
        return;
    };
    let (slot, key, missing) = match current {
        Value::Object(object) => {
            let missing = !object.contains_key(key);
            (
                object.entry(key.clone()).or_insert(Value::Null),
                key.clone(),
                missing,
            )
        }
        Value::Array(array) => {
            let Some(index) =
                canonical_array_index_js(key).filter(|index| *index < MAX_ARRAY_SLOTS)
            else {
                return;
            };
            let missing = index >= array.len();
            if missing {
                let old_len = array.len();
                let additional = index + 1 - old_len;
                // Bound the slots this reconstruction allocates across all arrays,
                // rather than permitting one sparse index to dominate memory.
                let Some(total) = array_slots.checked_add(additional) else {
                    return;
                };
                if total > MAX_ARRAY_SLOTS {
                    return;
                }
                *array_slots = total;
                array.resize_with(index + 1, || Value::Null);
                for hole in old_len..index {
                    let mut hole_path = current_path.to_vec();
                    hole_path.push(hole.to_string().into());
                    created.insert(hole_path);
                }
            }
            (&mut array[index], index.to_string().into(), missing)
        }
        _ => return,
    };
    if parts.len() == 1 {
        if slot.is_object() || slot.is_array() {
            return;
        }
        *slot = value;
        let mut slot_path = current_path.to_vec();
        slot_path.push(key);
        created.remove(&slot_path);
        return;
    }
    let mut child_path = current_path.to_vec();
    child_path.push(key.clone());
    if missing {
        *slot = if canonical_array_index_js(&parts[1]).is_some() {
            Value::Array(Vec::new())
        } else {
            Value::Object(JsonObject::new())
        };
        created.insert(child_path.clone());
    } else if !created.contains(&child_path) {
        return;
    }
    insert_path(slot, &parts[1..], value, created, &child_path, array_slots)
}

fn prepend_system_instructions(
    input: Value,
    instructions: Option<&Value>,
    parse_cache: &mut JsonParseCache,
) -> Value {
    let Some(instructions) = instructions.filter(|value| is_js_truthy(value)) else {
        return input;
    };
    let Some(mut messages) = parse_cache
        .parse(Some(&input))
        .and_then(Value::as_array)
        .cloned()
    else {
        return input;
    };
    if messages
        .iter()
        .any(|message| message.get("role") == Some(&Value::String("system".into())))
    {
        return input;
    }
    let content = match parse_cache
        .parse_or_self(Some(instructions))
        .expect("instructions present")
    {
        Value::Array(parts) => {
            parts
                .iter()
                .enumerate()
                .fold(JsString::default(), |mut content, (index, part)| {
                    if index > 0 {
                        content.push_str("\n");
                    }
                    let part = part
                        .get("content")
                        .filter(|value| is_js_truthy(value))
                        .map(|value| {
                            js_string_value(value)
                                .unwrap_or_else(|| javascript_string(value).into())
                        })
                        .unwrap_or_else(|| {
                            js_string_value(part).unwrap_or_else(|| javascript_string(part).into())
                        });
                    content.push_slice(&part, 0..part.len());
                    content
                })
        }
        value => js_string_value(value)
            .unwrap_or_else(|| value_as_string(value).unwrap_or_default().into()),
    };
    messages.insert(0, json_message("system", Value::String(content)));
    if input.is_string() {
        Value::String(
            crate::otel::attributes::javascript_json_stringify(&Value::Array(messages)).into(),
        )
    } else {
        Value::Array(messages)
    }
}

fn js_string_value(value: &Value) -> Option<JsString> {
    match value {
        Value::String(value) => Some(value.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::providers::extract_model_parameters;
    use crate::otel::value::json;

    #[test]
    fn provider_fallbacks_match_typescript() {
        let cases = [
            (
                "SDK null falls through",
                "ai",
                json!({
                    "langfuse.observation.input": null, "ai.prompt": "fallback",
                }),
                Some(json!("fallback")),
                None,
            ),
            (
                "Flue skips null aliases",
                "@flue/opentelemetry",
                json!({
                    "flue.turn.input": null, "flue.tool.arguments": "{\"x\":1}",
                }),
                Some(json!({"x": 1})),
                None,
            ),
            (
                "LiveKit accepts another scope",
                "other",
                json!({
                    "lk.user_input": "hello",
                }),
                Some(json!("hello")),
                None,
            ),
            (
                "LiveKit keeps bare user input",
                "livekit-agents",
                json!({
                    "lk.user_input": "hello",
                }),
                Some(json!("hello")),
                None,
            ),
            (
                "Anthropic output is consumed",
                "other",
                json!({
                    "gen_ai.tool.call.arguments": "{\"x\":1}", "anthropic.tool_result.content": "result",
                }),
                Some(json!("{\"x\":1}")),
                Some(json!("result")),
            ),
            (
                "AI prompt and combined tool calls",
                "ai",
                json!({
                    "ai.prompt.messages": [{"role":"user","content":"hi"}],
                    "ai.response.text": "done",
                    "ai.response.toolCalls": [{"name":"weather"}],
                }),
                Some(json!([{"role":"user","content":"hi"}])),
                Some(json!(
                    r#"{"role":"assistant","content":"done","tool_calls":[{"name":"weather"}]}"#
                )),
            ),
        ];
        let failures = cases
            .into_iter()
            .filter_map(|(name, scope, attributes, input, output)| {
                let io = extract_input_output(
                    scope,
                    attributes.as_object().unwrap(),
                    Vec::new(),
                    None,
                    &mut JsonParseCache::default(),
                );
                let actual = (io.input, io.output, io.filtered_attributes);
                let expected = (input, output, JsonObject::new());
                (actual != expected)
                    .then(|| format!("{name}: expected {expected:?}, got {actual:?}"))
            })
            .collect::<Vec<_>>();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn provider_wrappers_keep_non_null_fallbacks_and_diagnostic_values() {
        let attributes = json!({
            "genkit:input": {"messages": null, "config": {"temperature": 1}},
            "genkit:output": "null",
            "tool_arguments": "args",
            "custom": {"nested": true},
        });
        let io = extract_input_output(
            "genkit-tracer",
            attributes.as_object().unwrap(),
            Vec::new(),
            None,
            &mut JsonParseCache::default(),
        );
        assert_eq!(
            io.input,
            Some(json!({"messages": null, "config": {"temperature": 1}}))
        );
        assert_eq!(io.output, Some(json!("null")));
        assert_eq!(io.filtered_attributes["tool_arguments"], json!("args"));
        assert_eq!(
            io.filtered_attributes["custom"],
            json!(r#"{"nested":true}"#)
        );

        let gateway = json!({
            "langfuse.trace.name": "trace",
            "langfuse.observation.custom": "keep",
        });
        let filtered = filter_attributes(gateway.as_object().unwrap(), "langfuse-ai-gateway");
        assert!(!filtered.contains_key("langfuse.trace.name"));
        assert_eq!(filtered["langfuse.observation.custom"], json!("keep"));
    }

    #[test]
    fn reconstructs_flattened_message_arrays_and_prefers_truthy_exact_values() {
        for (attributes, expected) in [
            (
                json!({
                    "gen_ai.prompt.0.role": "user",
                    "gen_ai.prompt.0.content": "hello",
                    "gen_ai.prompt.1.role": "assistant",
                    "gen_ai.prompt.1.content": "world",
                }),
                json!([
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"}
                ]),
            ),
            (
                json!({
                    "gen_ai.prompt": [{"role": "system", "content": "exact"}],
                    "gen_ai.prompt.0.role": "ignored",
                }),
                json!([{"role": "system", "content": "exact"}]),
            ),
            (
                json!({
                    "gen_ai.prompt.__proto__.polluted": true,
                    "gen_ai.prompt.0.content": "kept",
                }),
                json!([{"content": "kept"}]),
            ),
        ] {
            assert_eq!(
                flattened_payload(attributes.as_object().unwrap(), "gen_ai.prompt"),
                Some(expected),
                "{attributes:?}",
            );
        }
    }

    #[test]
    fn flattened_paths_do_not_cross_containers_or_mutate_explicit_nulls() {
        let attributes = json!({
            "gen_ai.prompt.a.x.z": 1,
            "gen_ai.prompt.b.x": {"old": 2},
            "gen_ai.prompt.b.x.new": 3,
        });
        let payload = flattened_payload(attributes.as_object().unwrap(), "gen_ai.prompt")
            .expect("flattened payload should be reconstructed");

        assert_eq!(
            payload["a"],
            json!({"x": {"z": 1}}),
            "a later branch must not write through b's generated container"
        );
        assert_eq!(payload["b"]["x"], json!({"old": 2}));

        let attributes = json!({
            "gen_ai.prompt.0.0": null,
            "gen_ai.prompt.0.0.new": 1,
        });
        let payload = flattened_payload(attributes.as_object().unwrap(), "gen_ai.prompt")
            .expect("flattened payload should be reconstructed");
        assert_eq!(payload[0][0], Value::Null);
    }

    #[test]
    fn flattened_paths_bound_total_dense_array_slots() {
        let attributes = json!({
            "gen_ai.prompt.0.5000": "first",
            "gen_ai.prompt.1.5000": "second",
        });
        let payload = flattened_payload(attributes.as_object().unwrap(), "gen_ai.prompt")
            .expect("flattened payload should be reconstructed");

        fn array_slots(value: &Value) -> usize {
            match value {
                Value::Array(values) => {
                    values.len() + values.iter().map(array_slots).sum::<usize>()
                }
                Value::Object(object) => object.values().map(array_slots).sum(),
                _ => 0,
            }
        }
        assert!(array_slots(&payload) <= MAX_ARRAY_SLOTS);
    }

    #[test]
    fn flattened_array_holes_can_be_filled_later() {
        for (attributes, expected) in [
            (
                json!({
                    "gen_ai.prompt.1.role": "assistant",
                    "gen_ai.prompt.0.role": "user",
                }),
                json!([{"role": "user"}, {"role": "assistant"}]),
            ),
            (
                json!({
                    "gen_ai.prompt.1.role": "assistant",
                    "gen_ai.prompt.0": null,
                    "gen_ai.prompt.0.role": "ignored",
                }),
                json!([null, {"role": "assistant"}]),
            ),
            (
                json!({
                    "gen_ai.prompt.1.role": "assistant",
                    "gen_ai.prompt.0": {"explicit": true},
                    "gen_ai.prompt.0.role": "ignored",
                }),
                json!([{"explicit": true}, {"role": "assistant"}]),
            ),
        ] {
            assert_eq!(
                flattened_payload(attributes.as_object().unwrap(), "gen_ai.prompt"),
                Some(expected)
            );
        }
    }

    #[test]
    fn noncanonical_array_segments_remain_object_keys() {
        let attributes = json!({"gen_ai.prompt.+1.role": "user"});
        assert_eq!(
            flattened_payload(attributes.as_object().unwrap(), "gen_ai.prompt"),
            Some(json!({"+1": {"role": "user"}}))
        );
    }

    #[test]
    fn preserves_wtf8_segments_in_flattened_payload_paths() {
        let mut attributes = JsonObject::new();
        let mut key = b"gen_ai.prompt.0.".to_vec();
        key.extend_from_slice(&[b'l', 0xed, 0xa0, 0x80]);
        attributes.insert(
            JsString::from_wtf8(key).unwrap(),
            Value::String("value".into()),
        );

        let payload = flattened_payload(&attributes, "gen_ai.prompt")
            .expect("flattened payload should be reconstructed");
        let field = payload[0]
            .as_object()
            .unwrap()
            .iter()
            .next()
            .expect("field should exist");
        assert_eq!(field.0.as_bytes(), b"l\xed\xa0\x80");
        assert_eq!(field.1.as_str(), Some("value"));
    }

    #[test]
    fn preserves_wtf8_json_message_content() {
        let content = Value::String(JsString::from_wtf8(vec![b'x', 0xed, 0xa0, 0x80]).unwrap());
        let message = json_message("user", content);
        assert_eq!(message["content"].as_str(), Some("x\u{fffd}"));
        let Value::String(content) = &message["content"] else {
            panic!("message content should remain a string");
        };
        assert_eq!(content.as_bytes(), b"x\xed\xa0\x80");

        let attributes = Value::parse(r#"{"gen_ai.system":"\ud800","ai.response.providerMetadata":{"openai":{"serviceTier":"\ud801"}}}"#).unwrap();
        let parameters = extract_model_parameters(
            attributes.as_object().unwrap(),
            "ai",
            &mut JsonParseCache::default(),
        )
        .unwrap();
        assert_eq!(parameters["system"].to_json_string(), r#""\ud800""#);
        assert_eq!(parameters["service_tier"].to_json_string(), r#""\ud801""#);
    }

    #[test]
    fn prepending_system_instructions_preserves_wtf8_content() {
        let input = json!([{"role": "user", "content": "hello"}]);
        let instructions = Value::parse(r#"[{"content":"x\ud800"},"part",null]"#).unwrap();

        let result =
            prepend_system_instructions(input, Some(&instructions), &mut JsonParseCache::default());
        let Value::Array(messages) = result else {
            panic!("expected an array of messages");
        };
        let Value::String(content) = &messages[0]["content"] else {
            panic!("expected system content to remain a string");
        };
        assert_eq!(content.as_bytes(), b"x\xed\xa0\x80\npart\nnull");
    }

    #[test]
    fn falsy_system_instructions_are_not_prepended() {
        let input = json!([{"role": "user", "content": "hello"}]);
        for instructions in [json!(null), json!(false), json!("")] {
            assert_eq!(
                prepend_system_instructions(
                    input.clone(),
                    Some(&instructions),
                    &mut JsonParseCache::default(),
                ),
                input
            );
        }
    }

    #[test]
    fn filtering_attributes_keeps_the_javascript_object_order() {
        let attributes = json!({
            "input": "removed",
            "first": "kept",
            "gen_ai.prompt.messages": "removed",
            "middle": "kept",
            "langfuse.observation.input": "removed",
            "last": "kept",
        });
        let filtered = filter_attributes(attributes.as_object().unwrap(), "unexpected scope");
        assert_eq!(
            filtered
                .keys()
                .map(|key| key.as_str().to_owned())
                .collect::<Vec<_>>(),
            ["first".to_owned(), "middle".to_owned(), "last".to_owned()]
        );
    }
}
