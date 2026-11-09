//! Decode the OTLP envelope directly into the fields event construction reads.
//!
//! Serde skips unknown fields with `IgnoredAny`, without building a JSON tree.
//! Attributes and scope metadata remain dynamic because their contents are persisted.
//! Wrong-shaped containers are empty; repeated fields keep their last value.

use std::marker::PhantomData;
use std::ops::Deref;

use super::value::{JsString, Value};
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

const JSON_FIELD_TOKEN: &str = "langfuse::otel::JsonField";

/// A serde adapter for the typed OTLP envelope fallback.
///
/// The normal path lets `serde_json` drive `deserialize_any` directly. That is
/// the cheapest path, but its string decoder rejects a lone surrogate before a
/// container visitor can classify a scalar as the wrong shape. On the fallback
/// path we first borrow each value as `RawValue`, then hand object and array
/// members back through this adapter. Scalar members become `unit`, which is
/// exactly the existing wrong-shaped-container behavior.
///
/// This adapter is intentionally limited to typed container fields. Dynamic
/// fields still use `JsonField`/`Value`; the raw-value parser preserves their
/// contents and drops only a field that exceeds the materialization depth.
pub struct CompatibleContainerDeserializer<'de, E = serde_json::Error> {
    raw: &'de serde_json::value::RawValue,
    error: PhantomData<fn() -> E>,
}

impl<'de> CompatibleContainerDeserializer<'de> {
    pub fn from_bytes(bytes: &'de [u8]) -> Result<Self, serde_json::Error> {
        let raw = serde_json::from_slice::<&'de serde_json::value::RawValue>(bytes)?;
        Ok(Self {
            raw,
            error: PhantomData,
        })
    }
}

impl<'de, E> CompatibleContainerDeserializer<'de, E> {
    fn from_raw(raw: &'de serde_json::value::RawValue) -> Self {
        Self {
            raw,
            error: PhantomData,
        }
    }
}

struct RawContainerVisitor<V>(V);

impl<'de, V> Visitor<'de> for RawContainerVisitor<V>
where
    V: Visitor<'de>,
{
    type Value = V::Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.expecting(formatter)
    }

    fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        self.0.visit_map(RawContainerMapAccess { map })
    }

    fn visit_seq<A>(self, sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        self.0.visit_seq(RawContainerSeqAccess { sequence })
    }
}

struct RawContainerMapAccess<A> {
    map: A,
}

impl<'de, A> MapAccess<'de> for RawContainerMapAccess<A>
where
    A: MapAccess<'de>,
{
    type Error = A::Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: serde::de::DeserializeSeed<'de>,
    {
        self.map.next_key_seed(seed)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::DeserializeSeed<'de>,
    {
        let raw = self.map.next_value::<&'de serde_json::value::RawValue>()?;
        seed.deserialize(CompatibleContainerDeserializer::<A::Error>::from_raw(raw))
    }
}

struct RawContainerSeqAccess<A> {
    sequence: A,
}

impl<'de, A> SeqAccess<'de> for RawContainerSeqAccess<A>
where
    A: SeqAccess<'de>,
{
    type Error = A::Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: serde::de::DeserializeSeed<'de>,
    {
        let Some(raw) = self
            .sequence
            .next_element::<&'de serde_json::value::RawValue>()?
        else {
            return Ok(None);
        };
        seed.deserialize(CompatibleContainerDeserializer::<A::Error>::from_raw(raw))
            .map(Some)
    }
}

impl<'de, E> Deserializer<'de> for CompatibleContainerDeserializer<'de, E>
where
    E: serde::de::Error,
{
    type Error = E;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.raw.get().as_bytes().first() {
            Some(b'{') | Some(b'[') => {
                let mut deserializer = serde_json::Deserializer::from_str(self.raw.get());
                deserializer
                    .deserialize_any(RawContainerVisitor(visitor))
                    .map_err(E::custom)
            }
            _ => visitor.visit_unit(),
        }
    }

    // JsonField normally parses directly. On this fallback path, give its visitor
    // the source span so it can preserve surrogate strings and overflowing numbers.
    fn deserialize_newtype_struct<V>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if name == JSON_FIELD_TOKEN {
            return visitor.visit_borrowed_str(self.raw.get());
        }
        let mut deserializer = serde_json::Deserializer::from_str(self.raw.get());
        deserializer
            .deserialize_newtype_struct(name, visitor)
            .map_err(E::custom)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let mut deserializer = serde_json::Deserializer::from_str(self.raw.get());
        deserializer.deserialize_bytes(visitor).map_err(E::custom)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_unit()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        byte_buf option unit unit_struct seq tuple tuple_struct map struct enum identifier
    }
}

/// Only consumed dynamic fields need JavaScript number normalization.
/// Boxed so absent fields occupy one pointer in sparse spans.
pub struct JsonField(Box<Value>);

impl Deref for JsonField {
    type Target = Value;

    fn deref(&self) -> &Value {
        &self.0
    }
}

impl<'de> Deserialize<'de> for JsonField {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonFieldVisitor;
        impl<'de> Visitor<'de> for JsonFieldVisitor {
            type Value = JsonField;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an OTLP attribute value")
            }

            fn visit_newtype_struct<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                super::value::deserialize_fast(deserializer).map(JsonField::new)
            }

            fn visit_borrowed_str<E: serde::de::Error>(
                self,
                source: &'de str,
            ) -> Result<Self::Value, E> {
                match Value::parse(source) {
                    Ok(value) => Ok(JsonField::new(value)),
                    Err(error) if is_recursion_limit(&error) => Ok(JsonField::new(Value::Null)),
                    Err(error) => Err(E::custom(error)),
                }
            }
        }
        deserializer.deserialize_newtype_struct(JSON_FIELD_TOKEN, JsonFieldVisitor)
    }
}

fn is_recursion_limit(error: &impl std::fmt::Display) -> bool {
    error.to_string().contains("recursion limit exceeded")
}

impl JsonField {
    fn new(value: Value) -> Self {
        Self(Box::new(value))
    }

    pub fn into_value(self) -> Value {
        *self.0
    }
}

/// Distinguish an empty array from a wrong-shaped value when selecting aliases.
pub struct Array<T>(pub Option<Vec<T>>);

impl<T> Default for Array<T> {
    fn default() -> Self {
        Self(None)
    }
}

trait Container<'de>: Default {
    fn has_fields(&self) -> bool {
        false
    }

    fn read_map<A: MapAccess<'de>>(mut map: A) -> Result<Self, A::Error> {
        while map.next_key::<JsString>()?.is_some() {
            map.next_value::<IgnoredAny>()?;
        }
        Ok(Self::default())
    }

    fn read_seq<A: SeqAccess<'de>>(mut sequence: A) -> Result<Self, A::Error> {
        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(Self::default())
    }
}

struct ContainerVisitor<T>(PhantomData<T>);

impl<'de, T: Container<'de>> Visitor<'de> for ContainerVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an OTLP container")
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<T, A::Error> {
        T::read_map(map)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, sequence: A) -> Result<T, A::Error> {
        T::read_seq(sequence)
    }

    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<T, E> {
        Ok(T::default())
    }

    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<T, E> {
        Ok(T::default())
    }

    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<T, E> {
        Ok(T::default())
    }

    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<T, E> {
        Ok(T::default())
    }

    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<T, E> {
        Ok(T::default())
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<T, E> {
        Ok(T::default())
    }
}

impl<'de, T: Deserialize<'de> + Container<'de>> Container<'de> for Array<T> {
    fn read_seq<A: SeqAccess<'de>>(mut sequence: A) -> Result<Self, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<T>()? {
            if value.has_fields() {
                values.push(value);
            }
        }
        Ok(Self(Some(values)))
    }
}

impl<'de, T: Deserialize<'de> + Container<'de>> Deserialize<'de> for Array<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ContainerVisitor(PhantomData))
    }
}

// Derived Deserialize rejects duplicate fields and wrong-shaped containers. The
// OTLP boundary instead follows JSON.parse's last-key-wins and ignores non-objects.
macro_rules! otlp_object {
    ($name:ident { $($field:ident: $ty:ty => $key:literal),* $(,)? }) => {
        #[derive(Default)]
        pub struct $name {
            $(pub $field: Option<$ty>,)*
        }

        impl<'de> Container<'de> for $name {
            fn has_fields(&self) -> bool {
                false $(|| self.$field.is_some())*
            }

            fn read_map<A: MapAccess<'de>>(mut map: A) -> Result<Self, A::Error> {
                let mut value = Self::default();
                while let Some(key) = map.next_key::<JsString>()? {
                    match key.as_str() {
                        $($key => value.$field = Some(map.next_value()?),)*
                        _ => { map.next_value::<IgnoredAny>()?; }
                    }
                }
                Ok(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                deserializer.deserialize_any(ContainerVisitor(PhantomData))
            }
        }
    };
}

otlp_object!(ResourceSpan {
    resource: Resource => "resource",
    scope_spans: Array<ScopeSpan> => "scopeSpans",
    scope_spans_alias: Array<ScopeSpan> => "scope_spans",
});

otlp_object!(Resource {
    attributes: JsonField => "attributes",
});

otlp_object!(ScopeSpan {
    scope: JsonField => "scope",
    spans: Array<Span> => "spans",
});

otlp_object!(Span {
    trace_id: JsonField => "traceId",
    trace_id_alias: JsonField => "trace_id",
    span_id: JsonField => "spanId",
    span_id_alias: JsonField => "span_id",
    parent_span_id: JsonField => "parentSpanId",
    parent_span_id_alias: JsonField => "parent_span_id",
    start_time: JsonField => "startTimeUnixNano",
    start_time_alias: JsonField => "start_time_unix_nano",
    end_time: JsonField => "endTimeUnixNano",
    end_time_alias: JsonField => "end_time_unix_nano",
    name: JsonField => "name",
    attributes: JsonField => "attributes",
    events: Array<SpanEvent> => "events",
    status: Status => "status",
});

otlp_object!(SpanEvent {
    name: JsonField => "name",
    attributes: JsonField => "attributes",
});

impl SpanEvent {
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().and_then(Value::as_str)
    }
}

otlp_object!(Status {
    code: JsonField => "code",
    message: JsonField => "message",
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_decoding_keeps_last_fields_and_skips_wrong_shapes() {
        for ignored in ["null", r#""\ud800""#, "1e309"] {
            let input = format!(
                r#"{{"scopeSpans":[{ignored},{{"spans":[false,{{"name":"old","name":"current","attributes":{{"\ud800":1,"\t":2}},"status":{ignored}}}]}}]}}"#
            );
            let compatible = ResourceSpan::deserialize(
                CompatibleContainerDeserializer::from_bytes(input.as_bytes()).unwrap(),
            )
            .unwrap();
            let fast = serde_json::from_str::<ResourceSpan>(&input);
            if ignored == "null" {
                assert!(fast.is_ok(), "attribute keys must not force a batch retry");
            }
            let fast = fast.ok();
            for resource in Some(compatible).into_iter().chain(fast) {
                let scopes = resource.scope_spans.unwrap().0.unwrap();
                assert_eq!(scopes.len(), 1);
                let spans = scopes.into_iter().next().unwrap().spans.unwrap().0.unwrap();
                assert_eq!(spans.len(), 1);
                let span = spans.into_iter().next().unwrap();
                assert_eq!(
                    span.name.as_deref().and_then(Value::as_str),
                    Some("current")
                );
                assert!(span.status.unwrap().code.is_none());
            }
        }
        let input = r#"{"scopeSpans":{"\ud800":1}}"#;
        let resource = ResourceSpan::deserialize(
            CompatibleContainerDeserializer::from_bytes(input.as_bytes()).unwrap(),
        )
        .unwrap();
        assert!(resource.scope_spans.unwrap().0.is_none());
    }

    #[test]
    fn deep_dynamic_field_is_dropped_without_rejecting_sibling_span() {
        let nested = "[".repeat(128) + "0" + &"]".repeat(128);
        let input = format!(
            r#"{{"scopeSpans":[{{"spans":[
                {{"traceId":"trace-1","spanId":"span-1","name":"deep","attributes":{nested}}},
                {{"traceId":"trace-2","spanId":"span-2","name":"kept"}}
            ]}}]}}"#
        );
        let resource = ResourceSpan::deserialize(
            CompatibleContainerDeserializer::from_bytes(input.as_bytes()).unwrap(),
        )
        .expect("one deep field must not reject its resource span");
        let scopes = resource.scope_spans.unwrap().0.unwrap();
        let spans = scopes[0].spans.as_ref().unwrap().0.as_ref().unwrap();
        assert!(spans[0].attributes.as_ref().unwrap().is_null());
        assert_eq!(
            spans[1].span_id.as_deref().and_then(Value::as_str),
            Some("span-2")
        );
    }
}
