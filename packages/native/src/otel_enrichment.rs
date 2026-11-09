//! Usage, pricing-tier selection, and cost computation for Rust-owned OTEL events.
//!
//! The lookup result is deliberately a small JSON object supplied by the worker: it contains the
//! resolved model, prompt identity, and pricing tiers. The event payload never crosses back to
//! JavaScript for this stage. Keeping the rules here equivalent to
//! `IngestionService.getUsageUnits`, `matchPricingTier`, and `calculateUsageCosts` lets the final
//! storage adapter receive one already-enriched row.

use std::collections::BTreeMap;
use std::str::FromStr;

use crate::otel::attributes::{javascript_number_to_string, javascript_string};
use crate::otel::types::{EnrichmentResult, OwnedEvent};
use crate::otel::value::{Number, Value};
use fancy_regex::Regex;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;

/// Apply submitted usage, pricing-tier matching, and cost calculation to one owned event.
///
/// This follows the ordering of the TypeScript ingestion path:
/// provided usage is normalized first, a tier is selected from the resulting usage when it is
/// non-empty, and finally provided costs either win or calculated costs are produced from that
/// tier.
pub fn enrich(event: &OwnedEvent, lookup: &Value) -> EnrichmentResult {
    let model = lookup.get("model").filter(|value| value.is_object());
    let usage = add_usage_total(normalize_usage(&event.provided_usage_details));
    let has_provided_costs = has_provided_costs(event);

    // The TypeScript path leaves tier fields unset when there is no usage vector. In particular,
    // an ERROR generation or a gateway request with no provider usage must not match the default
    // tier against an invented all-zero vector.
    let tier = if usage.is_empty() {
        None
    } else {
        select_tier(lookup.get("pricingTiers"), &usage, event)
    };

    let cost_details = if has_provided_costs {
        provided_costs(&event.provided_cost_details)
    } else {
        compute_costs(&usage, tier)
    };

    EnrichmentResult {
        prompt_id: lookup
            .get("promptId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        model_id: model
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        usage_details: usage,
        cost_details,
        usage_pricing_tier_id: tier
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        usage_pricing_tier_name: tier
            .and_then(|value| value.get("name"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        skills_available: lookup
            .get("skillsAvailable")
            .and_then(Value::as_array)
            .map(|skills| skills.iter().filter_map(parse_skill_reference).collect())
            .unwrap_or_default(),
        skills_resource_loaded: lookup
            .get("skillsResourceLoaded")
            .and_then(Value::as_array)
            .map(|skills| {
                skills
                    .iter()
                    .filter_map(parse_skill_resource_loaded)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn parse_skill_reference(value: &Value) -> Option<crate::native_codec::SkillReference> {
    Some(crate::native_codec::SkillReference {
        langfuse_skill_id: value
            .get("langfuseSkillId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        skill_version: skill_version(value.get("skillVersion"))?,
        skill_name: value.get("skillName")?.as_str()?.to_owned(),
    })
}

fn parse_skill_resource_loaded(value: &Value) -> Option<crate::native_codec::SkillResourceLoaded> {
    Some(crate::native_codec::SkillResourceLoaded {
        langfuse_skill_id: value
            .get("langfuseSkillId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        skill_version: skill_version(value.get("skillVersion"))?,
        skill_name: value.get("skillName")?.as_str()?.to_owned(),
        file_path: value.get("filePath")?.as_str()?.to_owned(),
    })
}

fn skill_version(value: Option<&Value>) -> Option<Option<u32>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(Value::Number(number)) => Some(Some(u32::try_from(number.as_u64()?).ok()?)),
        _ => None,
    }
}

fn has_provided_costs(event: &OwnedEvent) -> bool {
    event
        .provided_cost_details
        .values()
        .any(|value| !value.is_null())
}

/// Match `normalizeProvidedUsageDetails`: numeric strings from ClickHouse and JSON numbers are
/// accepted, while null, malformed, negative, and non-finite values are ignored.
fn normalize_usage(values: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    values
        .iter()
        .filter_map(|(key, value)| {
            let number = number(value).filter(|number| number.is_finite() && *number >= 0.0)?;
            Some((key.clone(), number_value(number)))
        })
        .collect()
}

fn add_usage_total(mut usage: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    if !usage.is_empty() && !usage.contains_key("total") {
        let total = usage.values().filter_map(number).sum::<f64>();
        usage.insert("total".to_owned(), number_value(total));
    }
    usage
}

/// Provided cost points are authoritative. Preserve the original map, including null values,
/// and synthesize `total` only for the same input/output-only case as the legacy writer.
fn provided_costs(values: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let provided_keys = values
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    let final_total = values
        .get("total")
        .filter(|value| !value.is_null())
        .cloned()
        .or_else(|| {
            provided_keys
                .iter()
                .all(|key| matches!(*key, "input" | "output"))
                .then(|| javascript_add(values.get("input"), values.get("output")))
        });

    let mut costs = values.clone();
    if !costs.contains_key("total") {
        if let Some(total) = final_total {
            costs.insert("total".to_owned(), total);
        }
    }
    costs
}

/// Reproduce the `input ?? 0 + output ?? 0` expression used by the TypeScript ingestion path.
/// In particular, JavaScript concatenates when either operand is a string; parsing numeric
/// strings here would change the persisted Decimal64 value because the final writer treats that
/// synthesized string as a non-number and stores zero.
fn javascript_add(left: Option<&Value>, right: Option<&Value>) -> Value {
    let zero = Value::from(0);
    let left = left.filter(|value| !value.is_null()).unwrap_or(&zero);
    let right = right.filter(|value| !value.is_null()).unwrap_or(&zero);
    if matches!(left, Value::String(_) | Value::Array(_) | Value::Object(_))
        || matches!(right, Value::String(_) | Value::Array(_) | Value::Object(_))
    {
        return Value::String(
            format!("{}{}", javascript_string(left), javascript_string(right)).into(),
        );
    }

    let total = javascript_number(left) + javascript_number(right);
    number_value(total)
}

fn javascript_number(value: &Value) -> f64 {
    match value {
        Value::Bool(value) => u8::from(*value) as f64,
        Value::Number(value) => value.as_f64().unwrap_or(f64::NAN),
        Value::Null => 0.0,
        _ => f64::NAN,
    }
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) if value.as_str().trim().is_empty() => Some(0.0),
        Value::String(value) => value.as_str().parse().ok(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn number_value(value: f64) -> Value {
    if value.is_finite() && value.fract() == 0.0 && value >= 0.0 {
        if let Ok(integer) = u64::try_from(value as u128) {
            return Value::Number(Number::from(integer));
        }
    }
    Number::from_f64(value).map_or(Value::Null, Value::Number)
}

fn select_tier<'a>(
    tiers: Option<&'a Value>,
    usage: &BTreeMap<String, Value>,
    event: &OwnedEvent,
) -> Option<&'a Value> {
    let tiers = tiers?.as_array()?;
    let mut conditioned = tiers
        .iter()
        .filter(|tier| tier.get("isDefault") != Some(&Value::Bool(true)))
        .collect::<Vec<_>>();
    conditioned.sort_by(|left, right| {
        let left = left.get("priority").and_then(Value::as_f64).unwrap_or(0.0);
        let right = right.get("priority").and_then(Value::as_f64).unwrap_or(0.0);
        left.partial_cmp(&right)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    conditioned
        .into_iter()
        .find(|tier| {
            tier.get("conditions")
                .and_then(Value::as_array)
                .is_some_and(|conditions| {
                    !conditions.is_empty()
                        && conditions
                            .iter()
                            .all(|condition| matches_condition(condition, usage, event))
                })
        })
        .or_else(|| {
            tiers
                .iter()
                .find(|tier| tier.get("isDefault") == Some(&Value::Bool(true)))
        })
}

fn matches_condition(
    condition: &Value,
    usage: &BTreeMap<String, Value>,
    event: &OwnedEvent,
) -> bool {
    match condition.get("source").and_then(Value::as_str) {
        Some(source) => matches_attribute_condition(condition, source, event),
        None => matches_usage_condition(condition, usage),
    }
}

fn matches_attribute_condition(condition: &Value, source: &str, event: &OwnedEvent) -> bool {
    let Some(key) = condition.get("key").and_then(Value::as_str) else {
        return false;
    };
    let attributes = if source == "model_parameters" {
        pricing_attributes(event.model_parameters.as_ref())
    } else {
        pricing_object_attributes(&event.metadata)
    };
    let Some(attribute_value) = attributes.get(key) else {
        return false;
    };
    condition
        .get("values")
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .any(|value| value.as_str() == Some(attribute_value))
        })
}

fn matches_usage_condition(condition: &Value, usage: &BTreeMap<String, Value>) -> bool {
    let Some(pattern) = condition.get("usageDetailPattern").and_then(Value::as_str) else {
        return false;
    };
    let pattern = if condition.get("caseSensitive") == Some(&Value::Bool(true)) {
        pattern.to_owned()
    } else {
        format!("(?i){pattern}")
    };
    let Ok(regex) = Regex::new(&pattern) else {
        return false;
    };
    let total = usage
        .iter()
        .filter(|(key, _)| regex.is_match(key).unwrap_or(false))
        .filter_map(|(_, value)| number(value))
        .sum::<f64>();
    let Some(expected) = condition.get("value").and_then(Value::as_f64) else {
        return false;
    };
    match condition.get("operator").and_then(Value::as_str) {
        Some("gt") => total > expected,
        Some("gte") => total >= expected,
        Some("lt") => total < expected,
        Some("lte") => total <= expected,
        Some("eq") => total == expected,
        Some("neq") => total != expected,
        _ => false,
    }
}

/// Keep only the top-level primitive attributes used by pricing conditions. This mirrors
/// `toPricingAttributeRecord`, including parsing model parameters that arrived as a JSON string.
fn pricing_attributes(value: Option<&Value>) -> BTreeMap<String, String> {
    let Some(value) = value else {
        return BTreeMap::new();
    };
    let parsed;
    let value = if let Value::String(value) = value {
        let Ok(value) = crate::otel::value::parse_js_string(value) else {
            return BTreeMap::new();
        };
        parsed = value;
        &parsed
    } else {
        value
    };
    let Some(object) = value.as_object() else {
        return BTreeMap::new();
    };
    pricing_object_attributes(object)
}

fn pricing_object_attributes(object: &crate::otel::value::Object) -> BTreeMap<String, String> {
    object
        .iter()
        .filter_map(|(key, value)| match value {
            Value::String(value) => Some((key.as_str().to_owned(), value.as_str().to_owned())),
            Value::Number(value) => {
                Some((key.as_str().to_owned(), javascript_number_to_string(value)))
            }
            Value::Bool(value) => Some((key.as_str().to_owned(), value.to_string())),
            _ => None,
        })
        .collect()
}

fn compute_costs(usage: &BTreeMap<String, Value>, tier: Option<&Value>) -> BTreeMap<String, Value> {
    let mut costs = BTreeMap::new();
    let Some(prices) = tier
        .and_then(|tier| tier.get("prices"))
        .and_then(Value::as_array)
    else {
        return costs;
    };
    let price_for = |key: &str| {
        prices
            .iter()
            .find(|price| price.get("usageType").and_then(Value::as_str) == Some(key))
    };

    for (key, value) in usage {
        let price = price_for(key).or_else(|| {
            if matches!(
                key.as_str(),
                "input_cache_creation_5m" | "input_cache_creation_1h"
            ) {
                price_for("cache_creation_input_tokens")
                    .or_else(|| price_for("input_cache_creation"))
            } else {
                None
            }
        });
        let Some(price) = price
            .and_then(|price| price.get("price"))
            .and_then(|price| price.as_str())
        else {
            continue;
        };
        let Some(units) = number(value) else { continue };
        let (Ok(price), Ok(units)) = (
            Decimal::from_str(price),
            Decimal::from_str(&units.to_string()),
        ) else {
            continue;
        };
        let Some(cost) = price.checked_mul(units).and_then(|cost| cost.to_f64()) else {
            continue;
        };
        if cost.is_finite() {
            costs.insert(key.clone(), number_value(cost));
        }
    }

    if !costs.is_empty() && !costs.contains_key("total") {
        let total = costs.values().filter_map(number).sum::<f64>();
        costs.insert("total".to_owned(), number_value(total));
    }
    costs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::types::OwnedEvent;
    use crate::otel::value::json;

    #[test]
    fn normalizes_usage_and_adds_total() {
        let event = OwnedEvent {
            provided_usage_details: BTreeMap::from([
                ("input".into(), json!("2")),
                ("output".into(), json!(3)),
            ]),
            ..OwnedEvent::default()
        };
        let result = enrich(&event, &json!({"model": null, "pricingTiers": []}));
        assert_eq!(result.usage_details["input"], json!(2));
        assert_eq!(result.usage_details["output"], json!(3));
        assert_eq!(result.usage_details["total"], json!(5));
    }

    #[test]
    fn empty_usage_does_not_match_default_tier() {
        let event = OwnedEvent {
            row: crate::native_schema::OtelStorageRow {
                level: "ERROR".into(),
                ..Default::default()
            },
            ..OwnedEvent::default()
        };
        let lookup = json!({
            "model": {"id": "model"},
            "pricingTiers": [{"id": "default", "name": "Default", "isDefault": true, "prices": []}]
        });
        let result = enrich(&event, &lookup);
        assert!(result.usage_details.is_empty());
        assert!(result.usage_pricing_tier_id.is_none());
    }

    #[test]
    fn provided_costs_are_authoritative_and_get_an_input_output_total() {
        let event = OwnedEvent {
            provided_cost_details: BTreeMap::from([
                ("input".into(), json!(1.25)),
                ("output".into(), json!(0.75)),
            ]),
            ..OwnedEvent::default()
        };
        let result = enrich(&event, &json!({"model": {"id": "model"}}));
        assert!(result.usage_details.is_empty());
        assert_eq!(result.cost_details["total"], json!(2));
    }

    #[test]
    fn provided_cost_total_keeps_javascript_string_concatenation() {
        let event = OwnedEvent {
            provided_cost_details: BTreeMap::from([
                ("input".into(), json!(1.25)),
                ("output".into(), json!("0.75")),
            ]),
            ..OwnedEvent::default()
        };
        let result = enrich(&event, &json!({"model": {"id": "model"}}));
        assert_eq!(result.cost_details["total"], json!("1.250.75"));
    }

    #[test]
    fn pricing_attributes_match_top_level_model_parameter() {
        let event = OwnedEvent {
            model_parameters: Some(json!(r#"{"region":"eu","temperature":0.2}"#)),
            provided_usage_details: BTreeMap::from([("input".into(), json!(1))]),
            ..OwnedEvent::default()
        };
        let lookup = json!({
            "model": {"id": "model"},
            "pricingTiers": [
                {"id": "eu", "name": "EU", "isDefault": false, "priority": 1,
                 "conditions": [{"source": "model_parameters", "key": "region", "values": ["eu"]}], "prices": []},
                {"id": "default", "name": "Default", "isDefault": true, "prices": []}
            ]
        });
        let result = enrich(&event, &lookup);
        assert_eq!(result.usage_pricing_tier_id.as_deref(), Some("eu"));
    }

    #[test]
    fn cache_ttl_usage_uses_the_aggregate_price_fallback() {
        let event = OwnedEvent {
            provided_usage_details: BTreeMap::from([("input_cache_creation_5m".into(), json!(2))]),
            ..OwnedEvent::default()
        };
        let lookup = json!({
            "model": {"id": "model"},
            "pricingTiers": [{"id": "default", "name": "Default", "isDefault": true,
                "prices": [{"usageType": "input_cache_creation", "price": "0.5"}]}]
        });
        let result = enrich(&event, &lookup);
        assert_eq!(result.cost_details["input_cache_creation_5m"], json!(1));
        assert_eq!(result.cost_details["total"], json!(1));
    }
}
