//! One schema for the mutable ingestion row and finalized Native row.

use napi::bindgen_prelude::Object;
use serde::Serialize;
use serde_json::Value;

use crate::native_codec::{DateTime64Micros, Decimal64, SkillReference, SkillResourceLoaded};
use crate::otel::attributes::AttributeSource::{Resource, Scope, Span};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DefaultPolicy {
    None,
    Literal(&'static str),
    Metadata(&'static str),
    MetadataFallback(&'static str, &'static str),
    MetadataEquals(&'static str, &'static str),
}

macro_rules! column_name {
    ($field:ident) => {
        stringify!($field)
    };
    ($field:ident as $name:literal) => {
        $name
    };
}
macro_rules! logical_type {
    ($native:ty, $logical:ty) => {
        $logical
    };
    ($native:ty) => {
        $native
    };
}
macro_rules! read_owned {
    ($row:ident, $field:ident, $ty:ty, $name:expr, $policy:expr, metadata) => {
        crate::native_owned::default_string(
            $row.$field,
            $policy,
            &$row.metadata_names,
            &$row.metadata_values,
        )
    };
    ($row:ident, $field:ident, $ty:ty, $name:expr, $policy:expr) => {
        crate::native_owned::into_prepared::<$ty, _>($row.$field, $name)?
    };
}
macro_rules! restore_field {
    ($row:ident, $field:ident, $restore:ident, $payload:literal) => {};
    ($row:ident, $field:ident, $restore:ident) => {
        crate::native_owned::restore_field(&mut $row.$field, $restore)?;
    };
}

macro_rules! apply_attribute {
    ($row:ident, $field:ident, $attributes:ident, $method:ident [$($key:expr),*]) => {
        $row.$field = $attributes.$method(&[$($key),*]);
    };
    ($row:ident, $field:ident, $attributes:ident) => {};
}

macro_rules! define_row_types {
    ($($field:ident $(as $name:literal)? $(logical($logical_ty:ty))? : $ty:ty),* $(,)?) => {
        #[derive(Debug)]
        pub struct PreparedEventRow { $(pub(crate) $field: $ty,)* }

        #[derive(Clone, Debug, Default, Serialize)]
        pub struct OtelStorageRow {
            $(
                $(#[serde(rename = $name)])?
                pub(crate) $field: logical_type!($ty $(, $logical_ty)?),
            )*
        }
    };
}

macro_rules! impl_prepared_row {
    ($($field:ident $(as $name:literal)? : $ty:ty => $policy:expr),* $(,)?) => {
        impl PreparedEventRow {
            pub(crate) fn columns() -> Vec<(String, String, bool)> {
                vec![$((column_name!($field $(as $name)?).into(),
                    crate::native_columns::column_type::<$ty>().to_string(),
                    $policy != DefaultPolicy::None),)*]
            }

            pub(crate) fn schema() -> ch_core_rs::schema::Schema {
                ch_core_rs::schema::Schema::new(vec![$(
                    ch_core_rs::schema::Field {
                        name: column_name!($field $(as $name)?).into(),
                        ch_type: crate::native_columns::column_type::<$ty>(),
                    },
                )*])
            }

            pub(crate) fn build_columns(rows: &[std::sync::Arc<Self>])
                -> Result<Vec<ch_core_rs::column::Column>, crate::native_columns::NativeColumnError>
            {
                Ok(vec![$(crate::native_columns::build_column::<$ty, _>(rows.iter().map(|row| &row.$field))?,)*])
            }
        }
    };
}

macro_rules! impl_otel_row {
    ($(
        $field:ident $(as $name:literal)?
        $(otel($method:ident [$($key:expr),*]))?
        $(media_payload($payload:literal))? $(default($default:ident))?
        : $ty:ty => $policy:expr
    ),* $(,)?) => {
        impl OtelStorageRow {
            pub fn from_attributes(attributes: &mut crate::otel::attributes::EventAttributes<'_>) -> Self {
                let mut row = Self::default();
                $(apply_attribute!(row, $field, attributes $(, $method [$($key),*])?);)*
                row
            }

            pub fn restore_media(&mut self, restore: &mut impl FnMut(&mut String) -> Result<(), String>) -> Result<(), String> {
                $(restore_field!(self, $field, restore $(, $payload)?);)*
                Ok(())
            }

            pub fn into_prepared(self) -> Result<PreparedEventRow, String> {
                Ok(PreparedEventRow { $(
                    $field: read_owned!(self, $field, $ty, column_name!($field $(as $name)?), $policy $(, $default)?),
                )* })
            }
        }
    };
}

macro_rules! compat_read {
    ($object:ident, $metadata:ident, metadata_names $($rest:tt)*) => {
        $metadata.take_names()
    };
    ($object:ident, $metadata:ident, metadata_values $($rest:tt)*) => {
        $metadata.take_values()
    };
    (
        $object:ident,
        $metadata:ident,
        $field:ident $(as $name:literal)? $(logical($logical_ty:ty))?
        $(otel($method:ident [$($key:expr),*]))?
        $(media_payload($payload:literal))? $(default($default:ident))?
        : $ty:ty => $policy:expr
    ) => {
        crate::native_js::from_js_field::<$ty>(
            $object,
            column_name!($field $(as $name)?),
            $policy,
            &$metadata,
        )?
    };
}

macro_rules! impl_js_compat {
    ($(
        $field:ident $(as $name:literal)? $(logical($logical_ty:ty))?
        $(otel($method:ident [$($key:expr),*]))?
        $(media_payload($payload:literal))? $(default($default:ident))?
        : $ty:ty => $policy:expr
    ),* $(,)?) => {
        impl PreparedEventRow {
            pub(crate) fn from_js_fields(
                object: Object<'_>,
                mut metadata: crate::native_js::MetadataValues,
            ) -> Result<Self, String> {
                $(let $field = compat_read!(
                    object,
                    metadata,
                    $field $(as $name)? $(logical($logical_ty))?
                        $(otel($method [$($key),*]))?
                        $(media_payload($payload))? $(default($default))?
                        : $ty => $policy
                );)*
                Ok(Self { $($field,)* })
            }
        }
    };
}

/// Columns with metadata defaults precede the metadata arrays, so conversion can
/// resolve their defaults before taking ownership of those arrays.
macro_rules! clickhouse_table_schema {
    ($(
        $field:ident $(as $name:literal)? $(logical($logical_ty:ty))?
        $(otel($method:ident [$($key:expr),*]))?
        $(media_payload($payload:literal))? $(default($default:ident))?
        : $ty:ty => $policy:expr
    ),* $(,)?) => {
        define_row_types! { $($field $(as $name)? $(logical($logical_ty))?: $ty,)* }
        impl_prepared_row! { $($field $(as $name)?: $ty => $policy,)* }
        impl_js_compat! {
            $($field $(as $name)? $(logical($logical_ty))?
                $(otel($method [$($key),*]))?
                $(media_payload($payload))? $(default($default))?: $ty => $policy,)*
        }
        impl_otel_row! {
            $($field $(as $name)? $(otel($method [$($key),*]))?
                $(media_payload($payload))? $(default($default))?: $ty => $policy,)*
        }
    };
}

impl OtelStorageRow {
    /// Estimate the serialized row for trace-queue capacity, only when recording a trace.
    /// Persisted `event_bytes` is computed separately by ClickHouse.
    pub fn serialized_byte_length(&self) -> Result<u64, serde_json::Error> {
        #[derive(Serialize)]
        struct Event<'a> {
            id: &'a str,
            #[serde(flatten)]
            row: &'a OtelStorageRow,
        }
        let mut counter = ByteCounter(0);
        serde_json::to_writer(
            &mut counter,
            &Event {
                id: &self.span_id,
                row: self,
            },
        )?;
        Ok(counter.0)
    }

    pub fn evaluation_value(&self) -> Result<Value, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        value
            .as_object_mut()
            .expect("row is an object")
            .insert("id".into(), Value::String(self.span_id.clone()));
        Ok(value)
    }
}

struct ByteCounter(u64);
impl std::io::Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// Keep this list in the same order as the insertable events_full contract. Materialized and alias
// columns remain server-owned and therefore are intentionally absent.
clickhouse_table_schema! {
    project_id: String => DefaultPolicy::None,
    trace_id: String => DefaultPolicy::None,
    span_id: String => DefaultPolicy::None,
    parent_span_id logical(Option<String>): String => DefaultPolicy::None,
    name: String => DefaultPolicy::None,
    environment: String => DefaultPolicy::Literal("default"),
    version logical(Option<String>) otel(text [Span("langfuse.version"), Resource("service.version")]): String => DefaultPolicy::None,
    release logical(Option<String>) otel(text [Span("langfuse.release"), Resource("langfuse.release")]): String => DefaultPolicy::None,
    trace_name logical(Option<String>) otel(text [Span("langfuse.trace.name")]): String => DefaultPolicy::None,
    user_id logical(Option<String>) otel(identity [Span("langfuse.user.id"), Span("user.id"), Span("langfuse.observation.metadata.langfuse_user_id"), Span("langfuse.trace.metadata.langfuse_user_id"), Span("ai.telemetry.metadata.userId"), Resource("langfuse.user.id"), Resource("user.id"), Span("user.name"), Resource("user.name")]): String => DefaultPolicy::None,
    session_id logical(Option<String>) otel(identity [Span("langfuse.session.id"), Span("session.id"), Span("gen_ai.conversation.id"), Span("langfuse.observation.metadata.langfuse_session_id"), Span("langfuse.trace.metadata.langfuse_session_id"), Span("ai.telemetry.metadata.sessionId")]): String => DefaultPolicy::None,
    level: String => DefaultPolicy::None,
    status_message logical(Option<String>) otel(text [Span("langfuse.observation.status_message")]): String => DefaultPolicy::None,
    prompt_id logical(Option<String>): String => DefaultPolicy::None,
    prompt_name logical(Option<String>): String => DefaultPolicy::None,
    model_id logical(Option<String>): String => DefaultPolicy::None,
    provided_model_name logical(Option<String>): String => DefaultPolicy::None,
    model_parameters logical(Option<String>): String => DefaultPolicy::None,
    input logical(Option<String>) media_payload(true): String => DefaultPolicy::None,
    output logical(Option<String>) media_payload(true): String => DefaultPolicy::None,
    evaluator_id logical(Option<String>) otel(json_text [Span("langfuse.evaluator.id")]) default(metadata): String => DefaultPolicy::Metadata("evaluator_id"),
    evaluation_rule_id logical(Option<String>) otel(json_text [Span("langfuse.evaluation.rule.id")]) default(metadata): String => {
        DefaultPolicy::MetadataFallback("evaluation_rule_id", "job_configuration_id")
    },
    experiment_id logical(Option<String>) otel(json_text [Span("langfuse.experiment.id")]): String => DefaultPolicy::None,
    experiment_name logical(Option<String>) otel(json_text [Span("langfuse.experiment.name")]): String => DefaultPolicy::None,
    experiment_description logical(Option<String>) otel(json_text [Span("langfuse.experiment.description")]): String => DefaultPolicy::None,
    experiment_dataset_id logical(Option<String>) otel(json_text [Span("langfuse.experiment.dataset.id")]): String => DefaultPolicy::None,
    experiment_item_id logical(Option<String>) otel(json_text [Span("langfuse.experiment.item.id")]): String => DefaultPolicy::None,
    experiment_item_expected_output logical(Option<String>) otel(json_text [Span("langfuse.experiment.item.expected_output")]): String => DefaultPolicy::None,
    experiment_item_root_span_id logical(Option<String>) otel(json_text [Span("langfuse.experiment.item.root_observation_id")]): String => DefaultPolicy::None,
    source: String => DefaultPolicy::None,
    service_name logical(Option<String>) otel(text [Resource("service.name")]): String => DefaultPolicy::None,
    service_version logical(Option<String>) otel(text [Resource("service.version")]): String => DefaultPolicy::None,
    scope_name logical(Option<String>) otel(scope_name [Scope("name")]): String => DefaultPolicy::None,
    scope_version logical(Option<String>) otel(text [Scope("version")]): String => DefaultPolicy::None,
    telemetry_sdk_language logical(Option<String>) otel(text [Resource("telemetry.sdk.language")]): String => DefaultPolicy::None,
    telemetry_sdk_name logical(Option<String>) otel(text [Resource("telemetry.sdk.name")]): String => DefaultPolicy::None,
    telemetry_sdk_version logical(Option<String>) otel(text [Resource("telemetry.sdk.version")]): String => DefaultPolicy::None,
    blob_storage_file_path: String => DefaultPolicy::None,
    ingestion_api_key: String => DefaultPolicy::None,
    ingestion_sdk_name: String => DefaultPolicy::Literal("unknown"),
    ingestion_sdk_version: String => DefaultPolicy::Literal("unknown"),
    event_type as "type": String => DefaultPolicy::None,
    start_time logical(String): DateTime64Micros => DefaultPolicy::None,
    created_at logical(String): DateTime64Micros => DefaultPolicy::None,
    updated_at logical(String): DateTime64Micros => DefaultPolicy::None,
    event_ts logical(String): DateTime64Micros => DefaultPolicy::None,
    end_time logical(String): Option<DateTime64Micros> => DefaultPolicy::None,
    completion_start_time logical(Option<String>): Option<DateTime64Micros> => DefaultPolicy::None,
    experiment_item_version logical(Option<String>): Option<DateTime64Micros> => DefaultPolicy::None,
    prompt_version: Option<u16> => DefaultPolicy::None,
    is_app_root: bool => DefaultPolicy::None,
    bookmarked: bool => DefaultPolicy::None,
    public: bool => DefaultPolicy::None,
    evaluator_execution_is_test: bool => {
        DefaultPolicy::MetadataEquals("evaluator_test", "true")
    },
    is_deleted: u8 => DefaultPolicy::None,
    tags: Vec<String> => DefaultPolicy::None,
    tool_calls: Vec<String> => DefaultPolicy::None,
    tool_call_names: Vec<String> => DefaultPolicy::None,
    skills_available: Vec<SkillReference> => DefaultPolicy::None,
    skills_resource_loaded: Vec<SkillResourceLoaded> => DefaultPolicy::None,
    metadata_names: Vec<String> => DefaultPolicy::None,
    metadata_values media_payload(true): Vec<String> => DefaultPolicy::None,
    experiment_metadata_names: Vec<String> => DefaultPolicy::None,
    experiment_metadata_values: Vec<String> => DefaultPolicy::None,
    experiment_item_metadata_names: Vec<String> => DefaultPolicy::None,
    experiment_item_metadata_values: Vec<String> => DefaultPolicy::None,
    provided_usage_details logical(std::collections::BTreeMap<String, serde_json::Value>): std::collections::BTreeMap<String, u64> => DefaultPolicy::None,
    usage_details logical(std::collections::BTreeMap<String, serde_json::Value>): std::collections::BTreeMap<String, u64> => DefaultPolicy::None,
    provided_cost_details logical(std::collections::BTreeMap<String, serde_json::Value>): std::collections::BTreeMap<String, Decimal64> => DefaultPolicy::None,
    cost_details logical(std::collections::BTreeMap<String, serde_json::Value>): std::collections::BTreeMap<String, Decimal64> => DefaultPolicy::None,
    usage_pricing_tier_id logical(Option<String>): Option<String> => DefaultPolicy::None,
    usage_pricing_tier_name logical(Option<String>): Option<String> => DefaultPolicy::None,
    tool_definitions: std::collections::BTreeMap<String, String> => DefaultPolicy::None,
}
