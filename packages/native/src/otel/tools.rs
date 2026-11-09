//! Tool extraction for prepared OTEL input and output.
//!
//! The ClickHouse representation keeps tool definitions, calls, and names in separate columns.
//! This module performs that lossless-enough projection while the payload is still structured;
//! it does not stringify the event or mutate the input/output values.

use std::collections::{BTreeMap, HashSet};

use super::attributes::{is_js_truthy, javascript_json_stringify, JsonParseCache, ParsedJson};
use super::types::JsonObject;
use super::value::{JsString, Value};

#[derive(Clone, Debug, Default)]
pub struct NormalizedTools {
    pub tool_definitions: BTreeMap<String, String>,
    pub tool_calls: Vec<String>,
    pub tool_call_names: Vec<String>,
}

/// The result of the same post-provider normalization that the TypeScript processor applies
/// before constructing an EventInput. Tool definitions can be carried in metadata and are moved
/// into the input projection so downstream extraction sees one canonical request shape.
#[derive(Clone, Debug, Default)]
pub struct NormalizedObservation {
    pub input: Option<Value>,
    pub output: Option<Value>,
    pub metadata: JsonObject,
    pub tools: NormalizedTools,
}

/// Normalize tool metadata, augment the input when necessary, then extract the storage columns.
///
/// The order matters: definitions removed from metadata are appended to input first, and only
/// that normalized input is used for definition extraction. Output extraction remains independent
/// so tool results and definitions cannot accidentally become tool calls.
pub fn normalize_tools_for_observation(
    mut input: Option<Value>,
    output: Option<Value>,
    mut metadata: JsonObject,
    parse_cache: &mut JsonParseCache,
) -> NormalizedObservation {
    let (tools, cleaned_metadata) = collect_metadata_tools(&metadata, parse_cache);
    if !tools.is_empty() && append_tools_to_input(&mut input, tools, parse_cache) {
        metadata = cleaned_metadata;
    }
    let mut columns = ToolColumns {
        result: NormalizedTools::default(),
        names: HashSet::new(),
        calls: HashSet::new(),
        cache: parse_cache,
    };
    if let Some(value) = &input {
        columns.input(value);
    }
    if let Some(value) = &output {
        columns.output(value);
    }
    NormalizedObservation {
        input,
        output,
        metadata,
        tools: columns.result,
    }
}

fn parse_if_string<'a>(value: &'a Value, parse_cache: &mut JsonParseCache) -> ParsedJson<'a> {
    parse_cache.shared_or_self(value)
}

fn definition_data(value: &Value) -> Option<&JsonObject> {
    let object = value.as_object()?;
    Some(
        object
            .get("function")
            .and_then(Value::as_object)
            .unwrap_or(object),
    )
}

fn is_definition_like(
    value: &Value,
    allow_provider_without_name: bool,
    parse_cache: &mut JsonParseCache,
) -> bool {
    let parsed = parse_if_string(value, parse_cache);
    definition_data(&parsed)
        .and_then(|data| data.get("name"))
        .and_then(Value::as_str)
        .is_some_and(|name| !name.is_empty())
        || (allow_provider_without_name
            && parsed
                .as_object()
                .and_then(|object| object.get("type"))
                .is_some_and(Value::is_string))
}

fn parse_definition_array(
    value: Option<&Value>,
    require_every_item: bool,
    allow_provider_without_name: bool,
    parse_cache: &mut JsonParseCache,
) -> Option<Vec<Value>> {
    let parsed = parse_if_string(value?, parse_cache);
    let values = parsed.as_array()?;
    let mut valid = Vec::new();
    for value in values {
        if is_definition_like(value, allow_provider_without_name, parse_cache) {
            valid.push(parse_if_string(value, parse_cache).into_owned());
        }
    }
    if require_every_item && valid.len() != values.len() {
        return None;
    }
    (values.is_empty() || !valid.is_empty()).then_some(valid)
}

fn dedupe_definitions(
    values: impl IntoIterator<Item = Value>,
    parse_cache: &mut JsonParseCache,
) -> Vec<Value> {
    let mut names = std::collections::HashSet::<JsString>::new();
    values
        .into_iter()
        .filter(|value| {
            let parsed = parse_if_string(value, parse_cache);
            definition_data(&parsed)
                .and_then(|data| data.get("name"))
                .and_then(string_value_js)
                .map(|name| names.insert(name))
                .unwrap_or(true)
        })
        .collect()
}

fn collect_metadata_tools(
    metadata: &JsonObject,
    parse_cache: &mut JsonParseCache,
) -> (Vec<Value>, JsonObject) {
    let mut tools = Vec::new();
    let root_tools = parse_definition_array(metadata.get("tools"), true, false, parse_cache);
    let remove_root_tools = root_tools.as_ref().is_some_and(|tools| !tools.is_empty());
    tools.extend(root_tools.into_iter().flatten());

    let attributes_value = metadata
        .get("attributes")
        .map(|value| parse_if_string(value, parse_cache));
    let Some(attributes) = attributes_value.as_deref().and_then(Value::as_object) else {
        let mut cleaned = JsonObject::new();
        if !tools.is_empty() {
            cleaned = metadata.clone();
            if remove_root_tools {
                cleaned.shift_remove("tools");
            }
        }
        return (dedupe_definitions(tools, parse_cache), cleaned);
    };
    // Most spans carry no tools. Delay cloning diagnostic metadata until a definition
    // actually has to be removed; the caller keeps the original map otherwise.
    let mut cleaned_attributes = None;
    for (key, allow_provider_without_name) in [
        ("ai.prompt.tools", true),
        ("gen_ai.tool.definitions", false),
    ] {
        if let Some(values) = parse_definition_array(
            attributes.get(key),
            true,
            allow_provider_without_name,
            parse_cache,
        ) {
            tools.extend(values);
            cleaned_attributes
                .get_or_insert_with(|| attributes.clone())
                .shift_remove(key);
        }
    }

    if let Some(model_parameters) = attributes
        .get("model_request_parameters")
        .map(|value| parse_if_string(value, parse_cache))
        .and_then(|value| value.as_object().cloned())
    {
        let mut cleaned_parameters = model_parameters;
        if let Some(values) = parse_definition_array(
            cleaned_parameters.get("function_tools"),
            true,
            false,
            parse_cache,
        ) {
            tools.extend(values);
            cleaned_parameters.shift_remove("function_tools");
        }
        let cleaned = cleaned_attributes.get_or_insert_with(|| attributes.clone());
        if cleaned_parameters.is_empty() {
            cleaned.shift_remove("model_request_parameters");
        } else if let Some(original) = attributes.get("model_request_parameters") {
            cleaned.insert(
                "model_request_parameters".into(),
                if original.is_string() {
                    Value::String(
                        javascript_json_stringify(&Value::Object(cleaned_parameters)).into(),
                    )
                } else {
                    Value::Object(cleaned_parameters)
                },
            );
        }
    }

    let mut indexed = attributes
        .keys()
        .filter_map(|key| {
            let key_text = key.as_str();
            let suffix = key_text
                .strip_prefix("llm.tools.")?
                .strip_suffix(".tool.json_schema")?;
            Some((suffix.parse::<usize>().ok()?, key.clone()))
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    for (_, key) in indexed {
        if let Some(value) = attributes.get(&key) {
            let parsed = parse_if_string(value, parse_cache);
            if is_definition_like(&parsed, false, parse_cache) {
                tools.push(parsed.into_owned());
                cleaned_attributes
                    .get_or_insert_with(|| attributes.clone())
                    .shift_remove(&key);
            }
        }
    }
    if tools.is_empty() {
        return (tools, JsonObject::new());
    }
    let mut cleaned = metadata.clone();
    if remove_root_tools {
        cleaned.shift_remove("tools");
    }
    if let Some(cleaned_attributes) = cleaned_attributes {
        if cleaned_attributes.is_empty() {
            cleaned.shift_remove("attributes");
        } else {
            cleaned.insert("attributes".into(), Value::Object(cleaned_attributes));
        }
    }
    (dedupe_definitions(tools, parse_cache), cleaned)
}

/// Validate the merge before taking the input; failed merges keep both original fields.
fn append_tools_to_input(
    input: &mut Option<Value>,
    tools: Vec<Value>,
    cache: &mut JsonParseCache,
) -> bool {
    let parsed = input.as_ref().map(|value| cache.shared_or_self(value));
    let existing = match parsed.as_deref() {
        Some(Value::Bool(_) | Value::Number(_)) => return false,
        Some(Value::Object(object))
            if object.get("tools").is_some_and(|value| !value.is_null()) =>
        {
            let Some(existing) = parse_definition_array(object.get("tools"), false, true, cache)
            else {
                return false;
            };
            Some(existing)
        }
        _ => None,
    };
    let value = match parsed {
        Some(ParsedJson::Shared(value)) => std::rc::Rc::unwrap_or_clone(value),
        _ => input.take().unwrap_or(Value::Null),
    };
    let mut object = match value {
        Value::Object(object) => object,
        Value::Array(messages) => {
            JsonObject::from_iter([("messages".into(), Value::Array(messages))])
        }
        Value::String(prompt) => JsonObject::from_iter([("prompt".into(), Value::String(prompt))]),
        Value::Null => JsonObject::new(),
        _ => unreachable!("merge shape checked before taking input"),
    };
    let tools = match existing {
        Some(existing) => dedupe_definitions(existing.into_iter().chain(tools), cache),
        None => tools,
    };
    object.insert("tools".into(), Value::Array(tools));
    *input = Some(Value::Object(object));
    true
}

/// The extractor emits storage columns directly. Borrowed provider values never become
/// a second definition/call graph; only normalized strings and dedupe identities are retained.
struct ToolColumns<'a> {
    result: NormalizedTools,
    names: HashSet<JsString>,
    calls: HashSet<CallIdentity>,
    cache: &'a mut JsonParseCache,
}

impl ToolColumns<'_> {
    fn input(&mut self, value: &Value) {
        let parsed = self.cache.shared_or_self(value);
        if let Some(tools) = parsed.get("tools") {
            self.definitions(tools);
        }
        let messages = parsed
            .as_array()
            .or_else(|| parsed.get("messages").and_then(Value::as_array));
        for message in messages.into_iter().flatten() {
            if let Some(tools) = message.get("tools") {
                self.definitions(tools);
            }
            if message.get("role").and_then(Value::as_str) == Some("tool") {
                if let Some(content) = message.get("content").filter(|content| {
                    content.get("type").and_then(Value::as_str) == Some("function")
                        && content.get("function").is_some()
                }) {
                    self.definition(content);
                }
            }
        }
    }

    fn definitions(&mut self, value: &Value) {
        if let Some(values) = value.as_array() {
            for value in values {
                self.definition(value);
            }
        }
    }

    fn definition(&mut self, value: &Value) {
        let parsed = self.cache.shared_or_self(value);
        let Some(data) = definition_data(&parsed) else {
            return;
        };
        let name = data.get("name").and_then(string_value_js);
        // Index columns use UTF-8 names. Keep lone-surrogate names only in the payload,
        // rather than merging distinct names into the same replacement character.
        let Some(name) = name.filter(|name| {
            !name.is_empty() && name.as_utf8().is_some() && self.names.insert(name.clone())
        }) else {
            return;
        };
        let description = first(data, &["description", "desc"])
            .and_then(string_value_js)
            .unwrap_or_default();
        let parameters = first(
            data,
            &["parameters", "parameters_json_schema", "inputSchema"],
        )
        .filter(|value| is_js_truthy(value))
        .map(javascript_json_stringify)
        .unwrap_or_default();
        let normalized = Value::Object(JsonObject::from_iter([
            ("description".into(), Value::String(description)),
            ("parameters".into(), parameters.into()),
        ]));
        self.result.tool_definitions.insert(
            name.into_utf8_lossy(),
            javascript_json_stringify(&normalized),
        );
    }

    fn output(&mut self, value: &Value) {
        let parsed = self.cache.shared_or_self(value);
        if let Some(values) = parsed.as_array() {
            for value in values {
                self.output_item(value, false);
            }
        } else {
            self.output_item(&parsed, true);
        }
    }

    fn output_item(&mut self, value: &Value, top_level: bool) {
        let Some(object) = value.as_object() else {
            return;
        };
        if is_tool_call_like(object)
            && !["role", "content", "tool_calls", "additional_kwargs"]
                .iter()
                .any(|key| object.contains_key(*key))
        {
            self.call(value, false);
            return;
        }
        self.call_array(first(object, &["tool_calls", "toolCalls"]));
        if top_level {
            for choice in object
                .get("choices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(message) = choice.get("message").and_then(Value::as_object) {
                    self.call_array(first(message, &["tool_calls", "toolCalls"]));
                }
            }
        }
        for part in object
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match part.get("type").and_then(Value::as_str) {
                Some("tool_use") => self.call(part, true),
                Some("tool-call") => self.call(part, false),
                _ => {}
            }
        }
        if top_level {
            self.call_array(
                object
                    .get("additional_kwargs")
                    .and_then(|value| value.get("tool_calls")),
            );
        }
    }

    fn call_array(&mut self, value: Option<&Value>) {
        let Some(value) = value else { return };
        let parsed = self.cache.shared_or_self(value);
        for value in parsed.as_array().into_iter().flatten() {
            self.call(value, false);
        }
    }

    fn call(&mut self, value: &Value, content_tool_use: bool) {
        let Some(object) = value.as_object() else {
            return;
        };
        let function = object.get("function").and_then(Value::as_object);
        let Some(name) = function
            .and_then(|function| function.get("name"))
            .filter(|value| !value.is_null())
            .or_else(|| first(object, &["name", "toolName"]))
            .and_then(string_value_js)
        else {
            return;
        };
        let function_arguments = function
            .and_then(|function| function.get("arguments"))
            .filter(|value| !value.is_null());
        let arguments = if content_tool_use && function_arguments.is_none() {
            javascript_json_stringify(
                object
                    .get("input")
                    .filter(|value| !value.is_null())
                    .unwrap_or(&Value::Object(JsonObject::new())),
            )
            .into()
        } else {
            match function_arguments.or_else(|| first(object, &["arguments", "args", "input"])) {
                Some(Value::String(text)) => text.clone(),
                Some(value) => javascript_json_stringify(value).into(),
                None => "{}".into(),
            }
        };
        let id = first(object, &["call_id", "id", "toolCallId"])
            .and_then(string_value_js)
            .unwrap_or_default();
        let identity = if id.is_empty() {
            CallIdentity::NameAndArguments(name.clone(), arguments.clone())
        } else {
            CallIdentity::Id(id.clone())
        };
        if name.is_empty() || name.as_utf8().is_none() || !self.calls.insert(identity) {
            return;
        }
        let kind = object
            .get("type")
            .and_then(string_value_js)
            .unwrap_or_else(|| function.map_or_else(JsString::default, |_| "function".into()));
        let index = if content_tool_use {
            None
        } else {
            object
                .get("index")
                .filter(|value| !value.is_null())
                .cloned()
        };
        let call = Value::Object(JsonObject::from_iter([
            ("id".into(), Value::String(id)),
            ("arguments".into(), Value::String(arguments)),
            ("type".into(), Value::String(kind)),
            ("index".into(), index.unwrap_or_else(|| Value::from(0))),
        ]));
        self.result.tool_call_names.push(name.into_utf8_lossy());
        self.result
            .tool_calls
            .push(javascript_json_stringify(&call));
    }
}

fn first<'a>(object: &'a JsonObject, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|key| object.get(*key).filter(|value| !value.is_null()))
}

fn is_tool_call_like(object: &JsonObject) -> bool {
    let kind = object.get("type").and_then(Value::as_str);
    if kind == Some("tool-result") {
        return false;
    }
    let has = |key| object.contains_key(key);
    object
        .get("function")
        .and_then(Value::as_object)
        .is_some_and(|function| function.contains_key("name") && function.contains_key("arguments"))
        || (has("toolName") && (kind == Some("tool-call") || has("input") || has("args")))
        || (has("call_id") && has("name") && has("arguments"))
        || (kind == Some("tool_use") && has("name") && has("input"))
        || (has("name")
            && has("arguments")
            && (has("id")
                || has("index")
                || matches!(
                    kind,
                    Some("function" | "function_call" | "tool-call" | "tool_use")
                )))
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum CallIdentity {
    Id(JsString),
    NameAndArguments(JsString, JsString),
}

fn string_value_js(value: &Value) -> Option<JsString> {
    match value {
        Value::String(value) => Some(value.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::value::json;

    #[test]
    fn extracts_openai_definitions_and_calls_in_order() {
        let tools = normalize_tools_for_observation(
            Some(json!({
                "tools": [{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object"}}}]
            })),
            Some(json!({
                "tool_calls": [{"id":"call-1","type":"function","function":{"name":"weather","arguments":{"city":"Amsterdam"}}}]
            })),
            JsonObject::new(),
            &mut JsonParseCache::default(),
        )
        .tools;
        assert_eq!(tools.tool_call_names, ["weather"]);
        assert!(tools.tool_calls[0].contains("Amsterdam"));
        assert!(tools.tool_definitions["weather"].contains("forecast"));

        let output = Value::parse(
            r#"{"tool_calls":[{"id":"call-surrogate","type":"function","function":{"name":"echo","arguments":"x\ud800"}}]}"#,
        )
        .unwrap();
        let tools = normalize_tools_for_observation(
            None,
            Some(output),
            JsonObject::new(),
            &mut JsonParseCache::default(),
        )
        .tools;
        assert!(tools.tool_calls[0].contains(r#"x\ud800"#));
    }

    #[test]
    fn moves_metadata_tools_into_input_before_extraction() {
        let normalized = normalize_tools_for_observation(
            Some(json!({"messages": [{"role": "user", "content": "weather"}]})),
            Some(json!({
                "content": [{
                    "type": "tool-call",
                    "toolName": "weather",
                    "input": {"city": "Amsterdam"},
                    "toolCallId": "call-1"
                }]
            })),
            json!({
                "attributes": {
                    "ai.prompt.tools": [{
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "forecast",
                            "parameters": {"type": "object"}
                        }
                    }]
                }
            })
            .as_object()
            .unwrap()
            .clone(),
            &mut JsonParseCache::default(),
        );

        assert!(normalized.metadata.get("attributes").is_none());
        assert_eq!(
            normalized.input.as_ref().unwrap()["tools"][0]["function"]["name"],
            json!("weather")
        );
        assert_eq!(normalized.tools.tool_call_names, ["weather"]);
        assert!(normalized.tools.tool_calls[0].contains("Amsterdam"));
        assert!(normalized.tools.tool_definitions["weather"].contains("forecast"));
    }

    #[test]
    fn merges_empty_tool_placeholders_without_decoding_unrelated_metadata() {
        for placeholder in [json!([]), Value::Null] {
            let metadata = json!({"tools": [{"name": "weather"}], "attributes": "{\"custom\":1}"});
            let normalized = normalize_tools_for_observation(
                Some(json!({"tools": placeholder})),
                None,
                metadata.as_object().unwrap().clone(),
                &mut JsonParseCache::default(),
            );
            assert_eq!(normalized.tools.tool_definitions.len(), 1);
            assert!(normalized.metadata.get("tools").is_none());
            assert_eq!(normalized.metadata["attributes"], metadata["attributes"]);
        }
    }

    #[test]
    fn normalizes_nullish_tool_fields_before_deduplication() {
        let output = json!({"tool_calls": [
            {"call_id": null, "id": "a", "function": {"name": null, "arguments": null},
             "name": "weather", "arguments": null, "args": {"city": "Oslo"}, "index": null},
            {"call_id": null, "id": "b", "name": "weather", "arguments": {"city": "Oslo"}},
            {"id": "c", "name": "", "arguments": {}},
            {"id": "c", "name": "weather", "arguments": {}}
        ], "content": [{"type": "tool_use", "id": "d", "name": "weather", "input": null, "index": 3}]});
        let normalized = normalize_tools_for_observation(
            None,
            Some(output),
            JsonObject::new(),
            &mut JsonParseCache::default(),
        );
        let calls = normalized
            .tools
            .tool_calls
            .iter()
            .map(|call| Value::parse(call).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            calls
                .iter()
                .map(|call| call["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        assert_eq!(calls[0]["arguments"], json!("{\"city\":\"Oslo\"}"));
        assert_eq!(calls[0]["index"], json!(0));
        assert_eq!(calls[3]["arguments"], json!("{}"));
        assert_eq!(calls[3]["index"], json!(0));
    }

    #[test]
    fn extracts_only_definition_arrays_with_nonempty_names_and_truthy_parameters() {
        for tools in [json!("[{\"name\":\"weather\"}]"), json!({"weather": {}})] {
            let normalized = normalize_tools_for_observation(
                Some(json!({"tools": tools})),
                None,
                JsonObject::new(),
                &mut JsonParseCache::default(),
            );
            assert!(normalized.tools.tool_definitions.is_empty());
        }
        for (parameters, expected) in [(Value::Null, "{\"type\":\"object\"}"), (json!(false), "")] {
            let normalized = normalize_tools_for_observation(
                Some(
                    json!({"tools": [{"name":""}, {"name":"weather", "parameters": parameters, "inputSchema": {"type":"object"}}]}),
                ),
                None,
                JsonObject::new(),
                &mut JsonParseCache::default(),
            );
            assert_eq!(normalized.tools.tool_definitions.len(), 1);
            assert_eq!(
                Value::parse(&normalized.tools.tool_definitions["weather"]).unwrap()["parameters"],
                json!(expected)
            );
        }
        let input = Value::parse(r#"{"tools":[{"name":"\ud800"},{"name":"\udc00"}]}"#).unwrap();
        let output = Value::parse(
            r#"{"tool_calls":[{"name":"\ud800","arguments":{}},{"name":"\udc00","arguments":{}}]}"#,
        )
        .unwrap();
        let normalized = normalize_tools_for_observation(
            Some(input.clone()),
            Some(output.clone()),
            JsonObject::new(),
            &mut JsonParseCache::default(),
        );
        assert!(normalized.tools.tool_definitions.is_empty());
        assert!(normalized.tools.tool_call_names.is_empty());
        assert_eq!(normalized.input, Some(input));
        assert_eq!(normalized.output, Some(output));
    }

    #[test]
    fn preserves_input_when_metadata_tools_cannot_be_merged() {
        let original = json!(42);
        let normalized = normalize_tools_for_observation(
            Some(original.clone()),
            None,
            json!({"tools": [{"name": "weather"}]})
                .as_object()
                .unwrap()
                .clone(),
            &mut JsonParseCache::default(),
        );
        assert_eq!(normalized.input, Some(original));
        assert_eq!(normalized.metadata["tools"][0]["name"], json!("weather"));
        assert!(normalized.tools.tool_definitions.is_empty());
    }
}
