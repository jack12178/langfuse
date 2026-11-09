//! Conversions from the logical OTEL row to ClickHouse-owned field types.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::native_codec::{parse_datetime_text, parse_decimal, DateTime64Micros, Decimal64};
use crate::native_schema::DefaultPolicy;
use crate::otel::value::Number;

pub(crate) trait IntoPrepared<T> {
    fn into_prepared(self) -> Result<T, String>;
}

pub(crate) fn into_prepared<T, S: IntoPrepared<T>>(value: S, name: &str) -> Result<T, String> {
    value
        .into_prepared()
        .map_err(|error| format!("{name}: {error}"))
}

impl<T> IntoPrepared<T> for T {
    fn into_prepared(self) -> Result<T, String> {
        Ok(self)
    }
}

impl IntoPrepared<String> for Option<String> {
    fn into_prepared(self) -> Result<String, String> {
        Ok(self.unwrap_or_default())
    }
}

pub(crate) fn default_string(
    value: Option<String>,
    policy: DefaultPolicy,
    metadata_names: &[String],
    metadata_values: &[String],
) -> String {
    value.unwrap_or_else(|| match policy {
        DefaultPolicy::None | DefaultPolicy::MetadataEquals(_, _) => String::new(),
        DefaultPolicy::Literal(value) => value.to_owned(),
        DefaultPolicy::Metadata(key) => metadata(metadata_names, metadata_values, key)
            .unwrap_or_default()
            .to_owned(),
        DefaultPolicy::MetadataFallback(key, fallback) => {
            metadata(metadata_names, metadata_values, key)
                .filter(|value| !value.is_empty())
                .or_else(|| metadata(metadata_names, metadata_values, fallback))
                .unwrap_or_default()
                .to_owned()
        }
    })
}

impl IntoPrepared<DateTime64Micros> for String {
    fn into_prepared(self) -> Result<DateTime64Micros, String> {
        parse_datetime_text(&self, "datetime").map(DateTime64Micros)
    }
}

impl IntoPrepared<Option<DateTime64Micros>> for String {
    fn into_prepared(self) -> Result<Option<DateTime64Micros>, String> {
        parse_datetime_text(&self, "datetime")
            .map(DateTime64Micros)
            .map(Some)
    }
}

impl IntoPrepared<Option<DateTime64Micros>> for Option<String> {
    fn into_prepared(self) -> Result<Option<DateTime64Micros>, String> {
        self.map(|value| parse_datetime_text(&value, "datetime").map(DateTime64Micros))
            .transpose()
    }
}

impl IntoPrepared<BTreeMap<String, u64>> for BTreeMap<String, Value> {
    fn into_prepared(self) -> Result<BTreeMap<String, u64>, String> {
        self.into_iter()
            .map(|(key, value)| integral_from_value(value, &key).map(|value| (key, value)))
            .collect()
    }
}

impl IntoPrepared<BTreeMap<String, Decimal64>> for BTreeMap<String, Value> {
    fn into_prepared(self) -> Result<BTreeMap<String, Decimal64>, String> {
        Ok(self
            .into_iter()
            .map(|(key, value)| {
                let decimal = match value {
                    Value::Number(number) => parse_decimal(&number.to_string(), &key).unwrap_or(0),
                    _ => 0,
                };
                (key, Decimal64(decimal))
            })
            .collect())
    }
}

fn metadata<'a>(names: &'a [String], values: &'a [String], key: &str) -> Option<&'a str> {
    names
        .iter()
        .zip(values)
        .find_map(|(name, value)| (name == key).then_some(value.as_str()))
}

fn integral_from_value(value: Value, name: &str) -> Result<u64, String> {
    let Value::Number(number) = value else {
        return Err(format!("{name}: expected a non-negative integral number"));
    };

    // Usage maps follow JS number spelling even when Rust retained an integer:
    // 2^63 is written to JSONEachRow as 9223372036854776000.
    let number = number
        .as_f64()
        .filter(|number| number.is_finite() && *number >= 0.0 && number.fract() == 0.0)
        .ok_or_else(|| format!("{name}: expected a non-negative integral number"))?;
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if number <= MAX_SAFE_INTEGER {
        return Ok(number as u64);
    }

    crate::otel::attributes::javascript_number_to_string(
        &Number::from_f64(number).ok_or_else(|| format!("{name}: expected a finite integer"))?,
    )
    .parse::<u64>()
    .map_err(|_| format!("{name}: expected a non-negative integer below 2^64"))
}

pub(crate) trait RestoreMediaField {
    fn restore_media(
        &mut self,
        restore: &mut impl FnMut(&mut String) -> Result<(), String>,
    ) -> Result<(), String>;
}

pub(crate) fn restore_field<T: RestoreMediaField>(
    field: &mut T,
    restore: &mut impl FnMut(&mut String) -> Result<(), String>,
) -> Result<(), String> {
    field.restore_media(restore)
}

impl<T: RestoreMediaField> RestoreMediaField for Option<T> {
    fn restore_media(
        &mut self,
        restore: &mut impl FnMut(&mut String) -> Result<(), String>,
    ) -> Result<(), String> {
        if let Some(value) = self {
            value.restore_media(restore)?;
        }
        Ok(())
    }
}

impl<T: RestoreMediaField> RestoreMediaField for Vec<T> {
    fn restore_media(
        &mut self,
        restore: &mut impl FnMut(&mut String) -> Result<(), String>,
    ) -> Result<(), String> {
        for value in self {
            value.restore_media(restore)?;
        }
        Ok(())
    }
}

impl<K: Ord, T: RestoreMediaField> RestoreMediaField for BTreeMap<K, T> {
    fn restore_media(
        &mut self,
        restore: &mut impl FnMut(&mut String) -> Result<(), String>,
    ) -> Result<(), String> {
        for value in self.values_mut() {
            value.restore_media(restore)?;
        }
        Ok(())
    }
}

impl RestoreMediaField for String {
    fn restore_media(
        &mut self,
        restore: &mut impl FnMut(&mut String) -> Result<(), String>,
    ) -> Result<(), String> {
        restore(self)
    }
}

impl RestoreMediaField for Value {
    fn restore_media(
        &mut self,
        restore: &mut impl FnMut(&mut String) -> Result<(), String>,
    ) -> Result<(), String> {
        match self {
            Value::String(value) => restore(value),
            Value::Array(values) => {
                for value in values {
                    value.restore_media(restore)?;
                }
                Ok(())
            }
            Value::Object(values) => {
                for value in values.values_mut() {
                    value.restore_media(restore)?;
                }
                Ok(())
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => Ok(()),
        }
    }
}

macro_rules! no_media_field {
    ($($ty:ty),* $(,)?) => {$ (
        impl RestoreMediaField for $ty {
            fn restore_media(
                &mut self,
                _restore: &mut impl FnMut(&mut String) -> Result<(), String>,
            ) -> Result<(), String> {
                Ok(())
            }
        }
    )*};
}
no_media_field!(
    bool,
    u8,
    u16,
    u64,
    DateTime64Micros,
    Decimal64,
    crate::native_codec::SkillReference,
    crate::native_codec::SkillResourceLoaded,
);
