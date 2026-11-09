//! Provider-specific normalization rules over owned OTEL values.

use crate::otel::types::JsonObject;
use crate::otel::value::Value;

fn select_first<'a>(
    attributes: &'a JsonObject,
    keys: &[&str],
    predicate: impl Fn(&Value) -> bool,
) -> Option<&'a Value> {
    keys.iter()
        .find_map(|key| attributes.get(*key).filter(|value| predicate(value)))
}
fn trim_javascript(value: &str) -> &str {
    value.trim_matches(|character: char| {
        (character.is_whitespace() && character != '\u{0085}') || character == '\u{feff}'
    })
}

fn parse_javascript_number(value: &str) -> Option<f64> {
    let value = trim_javascript(value);
    if value.is_empty() {
        return None;
    }

    let radix = match value.as_bytes() {
        [b'0', b'x' | b'X', ..] => Some(16),
        [b'0', b'b' | b'B', ..] => Some(2),
        [b'0', b'o' | b'O', ..] => Some(8),
        _ => None,
    };

    let number = match radix {
        Some(radix) if value.len() > 2 => {
            let (mut prefix, mut dropped_bits) = (0u128, 0i32);
            for digit in value[2..].chars() {
                let digit = u128::from(digit.to_digit(radix)?);
                if dropped_bits == 0 {
                    if let Some(next) = prefix
                        .checked_mul(u128::from(radix))
                        .and_then(|prefix| prefix.checked_add(digit))
                    {
                        prefix = next;
                        continue;
                    }
                }
                // Retain enough exact bits for one f64 rounding, including whether
                // any discarded digit was nonzero (the sticky bit).
                dropped_bits = dropped_bits.saturating_add(radix.trailing_zeros() as i32);
                prefix |= u128::from(digit != 0);
            }
            prefix as f64 * 2f64.powi(dropped_bits)
        }
        Some(_) => return None,
        None if matches!(value, "Infinity" | "+Infinity") => f64::INFINITY,
        None if value == "-Infinity" => f64::NEG_INFINITY,
        None if value
            .bytes()
            .any(|byte| byte.is_ascii_alphabetic() && !matches!(byte, b'e' | b'E')) =>
        {
            return None
        }
        None => value.parse::<f64>().ok()?,
    };
    Some(number)
}
fn is_meaningful(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Bool(_) | Value::Number(_) => true,
    }
}
fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(value) => value.is_empty(),
        _ => false,
    }
}

mod fields;
pub use fields::{
    claude_managed_agent_session_id, extract_completion_start_time, extract_is_app_root,
    extract_model_name, extract_model_parameters, extract_name, extract_prompt_version,
    extract_prompt_version_lookup, extract_public, extract_tags, observation_type,
    should_skip_model_usage_and_cost,
};
mod payload;
pub use payload::{extract_input_output, filter_attributes, ProviderIO};
