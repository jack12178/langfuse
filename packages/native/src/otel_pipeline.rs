//! Owned direct-v4 state between narrow asynchronous service calls.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::otel::value::{JsString, Value};
use base64::Engine;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use sha2::{Digest, Sha256};

use crate::native_schema::OtelStorageRow;
use crate::native_task::{to_js_error, NativeResult, OwnedTask};
use crate::otel::{process_to_events, EventBuildConfig, OwnedEvent, StorageContext};
use crate::otel_input::{EarlyOtelBatch, ExtractedBatch};
use crate::otel_media::{
    collect_media_references, extract_normalized_media, rewrite_known_references,
    validate_media_registry, EarlyMediaError, ExtractedMedia, MediaReferenceCollector,
};
use crate::PreparedEvent;

#[napi(object)]
pub struct OtelProcessingContext {
    pub project_id: String,
    pub ingestion_api_key: String,
    pub ingestion_sdk_name: String,
    pub ingestion_sdk_version: String,
    pub file_key: String,
    pub now: String,
    pub is_langfuse_internal: bool,
}
struct Pipeline {
    source: Arc<ExtractedBatch>,
    rows: BatchRows,
    context: StorageContext,
    media_requests: Vec<u32>,
    late_media: Vec<ExtractedMedia>,
    overflow_requests: Vec<OtelOverflowRequest>,
}

enum BatchRows {
    Prepared(Vec<OwnedEvent>),
    Ready(Vec<Option<EnrichedStorageRow>>),
    Finalized,
}

impl BatchRows {
    fn ready(&self) -> NativeResult<&[Option<EnrichedStorageRow>]> {
        match self {
            Self::Ready(rows) => Ok(rows),
            _ => Err(failure("enrich before finalization")),
        }
    }
}

struct EnrichedStorageRow {
    row: OtelStorageRow,
    start_time_iso: String,
}

#[napi]
pub struct RustOtelBatch {
    state: Arc<Mutex<Option<Pipeline>>>,
}

#[napi(object)]
pub struct OtelMediaRequest {
    pub index: u32,
    pub media_index: u32,
    pub trace_id: String,
    pub observation_id: String,
    pub field: String,
    pub reference: String,
    pub content_type: String,
    pub sha256_hash: String,
    pub kind: String,
    pub original_byte_length: f64,
}

#[napi(object)]
pub struct OtelMediaResolution {
    pub index: u32,
    /// Absent after an upload/link failure; restore this occurrence's original inline value.
    pub reference: Option<String>,
}

#[napi(object)]
pub struct OtelSkillAvailable {
    pub name: String,
    pub version: Option<u32>,
}

#[napi(object)]
pub struct OtelSkillResourceLoaded {
    pub skill_name: String,
    pub file_path: String,
    pub skill_version: Option<u32>,
}

#[napi(object)]
pub struct OtelLookupRequest {
    pub span_id: String,
    pub event_type: String,
    pub model_name: Option<String>,
    pub prompt_name: Option<String>,
    pub prompt_version: Option<f64>,
    pub skills_available: Vec<OtelSkillAvailable>,
    pub skills_resource_loaded: Vec<OtelSkillResourceLoaded>,
}

#[napi(object)]
#[derive(Clone)]
pub struct OtelOverflowRequest {
    pub index: u32,
    pub event_index: u32,
    pub trace_id: String,
    pub observation_id: String,
    pub field: String,
    pub metadata_index: Option<u32>,
    pub byte_length: f64,
    pub sha256_hash: String,
}

#[napi(object, object_from_js = false)]
pub struct FinalizedOtelEvent {
    pub event: PreparedEvent,
    pub trace_id: String,
    pub start_time_iso: String,
    pub environment: String,
    /// Post-overflow JSON estimate for trace-queue capacity; absent when not requested.
    pub serialized_event_bytes: Option<f64>,
}

#[napi(object, object_from_js = false)]
pub struct FinalizedOtelBatch {
    pub events: Vec<FinalizedOtelEvent>,
    /// Evaluation snapshots retain full fields, before overflow replaces them with references.
    #[napi(ts_type = "object[]")]
    pub evaluation_rows: Vec<serde_json::Value>,
}

#[napi(object, object_from_js = false)]
pub struct PreparedOtelBatch {
    pub batch: RustOtelBatch,
    pub media_requests: Vec<OtelMediaRequest>,
    pub lookup_requests: Vec<OtelLookupRequest>,
    pub trace_ids: Vec<String>,
}

#[napi(object, object_from_js = false)]
pub struct OtelCompletion {
    pub finalized: Option<FinalizedOtelBatch>,
    pub overflow_requests: Vec<OtelOverflowRequest>,
}

#[napi]
impl EarlyOtelBatch {
    /// Consume the compact document and collect the external work for one batch.
    #[napi(ts_return_type = "Promise<PreparedOtelBatch>")]
    pub fn process(
        &mut self,
        env: Env,
        context: OtelProcessingContext,
    ) -> napi::Result<AsyncTask<OwnedTask<PreparedOtelBatch>>> {
        let source = self.take_data().map_err(|error| to_js_error(env, error))?;
        Ok(AsyncTask::new(OwnedTask::new(move || {
            // Decode one resource at a time so a compact batch does not also need a full
            // resource-span array in memory. Each resource is normalized with JavaScript number
            // semantics before its owned events are appended to the batch.
            let mut events = source.consume_json(|json| {
                process_to_events(
                    json.as_bytes(),
                    &EventBuildConfig {
                        project_id: context.project_id,
                        ingestion_api_key: context.ingestion_api_key,
                        ingestion_sdk_name: context.ingestion_sdk_name,
                        ingestion_sdk_version: context.ingestion_sdk_version,
                        fallback_time: context.now.clone(),
                        is_langfuse_internal: context.is_langfuse_internal,
                        ..EventBuildConfig::default()
                    },
                )
                .map_err(failure)
            })?;
            // Lookup keys and model parameters are not media payloads. The same source
            // attribute may also feed input/metadata, where the compact reference stays.
            if !source.media.is_empty() {
                for event in &mut events {
                    for text in [&mut event.row.trace_id, &mut event.row.span_id] {
                        restore_text(text, &source.media)?;
                    }
                    for text in [
                        &mut event.row.provided_model_name,
                        &mut event.row.prompt_name,
                        &mut event.row.parent_span_id,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        restore_text(text, &source.media)?;
                    }
                    if let Some(value) = &mut event.model_parameters {
                        restore_value(value, &source.media)?;
                    }
                }
            }
            let trace_ids = events
                .iter()
                .map(|event| event.trace_id.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let mut pipeline = Pipeline {
                source,
                rows: BatchRows::Prepared(events),
                context: StorageContext {
                    blob_storage_file_path: context.file_key,
                    created_at: context.now.clone(),
                    updated_at: context.now.clone(),
                    event_ts: context.now,
                },
                media_requests: Vec::new(),
                late_media: Vec::new(),
                overflow_requests: Vec::new(),
            };
            let media_requests = collect_media_requests(&mut pipeline)?;
            let lookup_requests = collect_lookup_requests(&pipeline);
            let batch = RustOtelBatch {
                state: Arc::new(Mutex::new(Some(pipeline))),
            };
            Ok(PreparedOtelBatch {
                batch,
                media_requests,
                lookup_requests,
                trace_ids,
            })
        })))
    }
}

impl RustOtelBatch {
    fn run<T: Send + ToNapiValue + TypeName + 'static>(
        &self,
        work: impl FnOnce(&mut Pipeline) -> NativeResult<T> + Send + 'static,
    ) -> AsyncTask<OwnedTask<T>> {
        let state = Arc::clone(&self.state);
        AsyncTask::new(OwnedTask::new(move || {
            let mut guard = state
                .lock()
                .map_err(|_| failure("OTEL processing lock poisoned"))?;
            let result = {
                let pipeline = guard
                    .as_mut()
                    .ok_or_else(|| failure("OTEL batch already disposed"))?;
                work(pipeline)
            };
            if guard
                .as_ref()
                .is_some_and(|pipeline| matches!(pipeline.rows, BatchRows::Finalized))
            {
                guard.take();
            }
            result
        }))
    }
}

#[napi]
impl RustOtelBatch {
    /// Free the graph and media ownership even when a service call failed before finalization.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn dispose<'env>(
        &self,
        env: &'env Env,
    ) -> napi::Result<Either<AsyncTask<OwnedTask<()>>, PromiseRaw<'env, ()>>> {
        if self.state.try_lock().is_ok_and(|state| state.is_none()) {
            return PromiseRaw::resolve(env, ()).map(Either::B);
        }
        let state = Arc::clone(&self.state);
        Ok(Either::A(AsyncTask::new(OwnedTask::new(move || {
            state
                .lock()
                .map_err(|_| failure("OTEL processing lock poisoned"))?
                .take();
            Ok(())
        }))))
    }

    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn media_body(&self, index: u32) -> AsyncTask<OwnedTask<Buffer>> {
        self.run(move |state| {
            media_entry(state, index)?
                .decode()
                .map(Buffer::from)
                .map_err(failure)
        })
    }

    /// Resolve uploads, apply lookups and either finalize or return overflow work for one batch.
    #[napi(
        ts_args_type = "mediaResults: OtelMediaResolution[], lookups: object[], overflowLimitBytes: number | undefined, includeEvaluationRows: boolean, traceActivityIds?: string[]",
        ts_return_type = "Promise<OtelCompletion>"
    )]
    pub fn complete(
        &self,
        media_results: Vec<OtelMediaResolution>,
        lookups: Vec<serde_json::Value>,
        overflow_limit_bytes: Option<f64>,
        include_evaluation_rows: bool,
        trace_activity_ids: Option<Vec<String>>,
    ) -> AsyncTask<OwnedTask<OtelCompletion>> {
        self.run(move |state| {
            let BatchRows::Prepared(events) = &state.rows else {
                return Err(failure("OTEL batch already enriched"));
            };
            validate_results(
                &media_results,
                state.media_requests.len(),
                "incomplete OTEL media results",
            )?;
            if lookups.len() != events.iter().filter(|event| needs_lookup(event)).count() {
                return Err(failure("incomplete OTEL lookup results"));
            }
            if overflow_limit_bytes
                .is_some_and(|limit| !limit.is_finite() || limit <= 0.0 || limit.fract() != 0.0)
            {
                return Err(failure("overflow limit must be a positive integer"));
            }
            resolve_media_inner(state, media_results)?;
            enrich_inner(state, lookups)?;
            let overflow_requests = overflow_limit_bytes
                .map(|limit| overflow_requests_inner(state, limit))
                .transpose()?
                .unwrap_or_default();
            if overflow_requests.is_empty() {
                let finalized = finalize_inner(
                    state,
                    Vec::new(),
                    include_evaluation_rows,
                    trace_activity_ids,
                )?;
                Ok(OtelCompletion {
                    finalized: Some(finalized),
                    overflow_requests,
                })
            } else {
                Ok(OtelCompletion {
                    finalized: None,
                    overflow_requests,
                })
            }
        })
    }

    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn overflow_body(&self, index: u32) -> AsyncTask<OwnedTask<Buffer>> {
        self.run(move |state| {
            let request = state
                .overflow_requests
                .get(index as usize)
                .ok_or_else(|| failure("unknown overflow request"))?;
            let rows = state.rows.ready()?;
            let value = overflow_text(request, rows);
            Ok(Buffer::from(value.as_bytes().to_vec()))
        })
    }

    /// Finalize once, after all mutable processing; the writer receives immutable typed handles.
    #[napi(ts_return_type = "Promise<FinalizedOtelBatch>")]
    pub fn finalize(
        &self,
        overflow: Vec<OtelMediaResolution>,
        include_evaluation_rows: bool,
        trace_activity_ids: Option<Vec<String>>,
    ) -> AsyncTask<OwnedTask<FinalizedOtelBatch>> {
        self.run(move |state| {
            finalize_inner(state, overflow, include_evaluation_rows, trace_activity_ids)
        })
    }
}

/// Discover late media and build every upload descriptor while the normalized events are
/// already owned by the worker task. This is intentionally paired with lookup collection in
/// `EarlyOtelBatch::process`; neither operation needs a separate native handover.
fn collect_media_requests(state: &mut Pipeline) -> NativeResult<Vec<OtelMediaRequest>> {
    let BatchRows::Prepared(events) = &mut state.rows else {
        unreachable!("media discovery precedes completion")
    };
    // Normalization can copy a pre-existing public reference into a different field before it
    // also exposes a newly extractable candidate. Detect that collision before any upload
    // request is returned; otherwise one content-derived reference could identify two values.
    let existing_references = events
        .iter()
        .flat_map(|event| {
            event
                .input
                .iter()
                .chain(event.output.iter())
                .chain(event.metadata.values())
        })
        .fold(HashSet::new(), |mut references, value| {
            collect_value_media_references(value, &mut references);
            references
        });
    for event in events.iter_mut() {
        if let Some(value) = &mut event.input {
            extract_remaining_media(value, &mut state.late_media)?;
        }
        if let Some(value) = &mut event.output {
            extract_remaining_media(value, &mut state.late_media)?;
        }
        let mut metadata = Value::Object(std::mem::take(&mut event.metadata));
        extract_remaining_media(&mut metadata, &mut state.late_media)?;
        event.metadata = metadata
            .as_object_mut()
            .map(std::mem::take)
            .expect("metadata stays an object");
    }
    validate_media_registry(&state.source.media, &state.late_media).map_err(failure)?;
    if let Some(media) = state
        .late_media
        .iter()
        .find(|media| existing_references.contains(&media.reference()))
    {
        return Err(failure(
            EarlyMediaError::UnsupportedMediaReferenceAmbiguity {
                reference: media.reference(),
            },
        ));
    }
    let registry = state
        .source
        .media
        .iter()
        .chain(&state.late_media)
        .collect::<Vec<_>>();
    if registry.is_empty() {
        return Ok(Vec::new());
    }
    let mut collector = MediaReferenceCollector::new(registry.iter().copied());
    let mut requests = Vec::new();
    for event in events.iter() {
        for field in ["input", "output", "metadata"] {
            let media_indices = match field {
                "input" => collector.collect(event.input.iter()),
                "output" => collector.collect(event.output.iter()),
                _ => collector.collect(event.metadata.values()),
            };
            for &media_index in media_indices {
                let media = registry[media_index];
                state.media_requests.push(media_index as u32);
                requests.push(OtelMediaRequest {
                    index: requests.len() as u32,
                    media_index: media_index as u32,
                    trace_id: event.trace_id.clone(),
                    observation_id: event.span_id.clone(),
                    field: field.to_owned(),
                    reference: media.reference(),
                    content_type: media.metadata.content_type.clone(),
                    sha256_hash: media.metadata.sha256_hash.clone(),
                    kind: media.kind.as_str().to_owned(),
                    original_byte_length: media.original_byte_length() as f64,
                });
            }
        }
    }
    Ok(requests)
}

fn collect_lookup_requests(state: &Pipeline) -> Vec<OtelLookupRequest> {
    let BatchRows::Prepared(events) = &state.rows else {
        unreachable!("lookups precede completion")
    };
    events
        .iter()
        .filter(|event| needs_lookup(event))
        .map(|event| OtelLookupRequest {
            span_id: event.span_id.clone(),
            event_type: event.event_type.clone(),
            model_name: event.provided_model_name.clone(),
            prompt_name: event.prompt_name.clone(),
            prompt_version: event.prompt_version_lookup,
            skills_available: event
                .skills_available
                .iter()
                .map(|skill| OtelSkillAvailable {
                    name: skill.name.clone(),
                    version: skill.version,
                })
                .collect(),
            skills_resource_loaded: event
                .skills_resource_loaded
                .iter()
                .map(|skill| OtelSkillResourceLoaded {
                    skill_name: skill.skill_name.clone(),
                    file_path: skill.file_path.clone(),
                    skill_version: skill.skill_version,
                })
                .collect(),
        })
        .collect()
}

fn validate_results(
    results: &[OtelMediaResolution],
    count: usize,
    error: &'static str,
) -> NativeResult<()> {
    if results.len() != count
        || results
            .iter()
            .enumerate()
            .any(|(index, result)| result.index as usize != index)
    {
        return Err(failure(error));
    }
    Ok(())
}

fn resolve_media_inner(
    state: &mut Pipeline,
    results: Vec<OtelMediaResolution>,
) -> NativeResult<()> {
    if state.media_requests.is_empty() {
        return Ok(());
    }
    let registry = state
        .source
        .media
        .iter()
        .chain(&state.late_media)
        .enumerate()
        .map(|(index, media)| (media.reference(), (index as u32, media)))
        .collect::<HashMap<_, _>>();
    let depth_references = results
        .iter()
        .filter(|result| result.reference.is_none())
        .map(|result| {
            media_entry(state, state.media_requests[result.index as usize])
                .map(ExtractedMedia::reference)
        })
        .collect::<NativeResult<HashSet<_>>>()?;
    // Collection and replacement traverse input, output, then metadata in the same order.
    // Consume each outcome once, including when normalization duplicates a reference.
    let mut pending = results.into_iter().zip(&state.media_requests);
    let mut resolve_text = |text: &mut JsString| {
        rewrite_known_references(
            text,
            |reference| registry.contains_key(reference),
            |reference| depth_references.contains(reference),
            |reference, json_layers| {
                let (outcome, expected_index) = pending
                    .next()
                    .ok_or_else(|| failure("incomplete OTEL media traversal"))?;
                let (index, media) = registry[reference];
                if index != *expected_index {
                    return Err(failure("OTEL media traversal order changed"));
                }
                match outcome.reference {
                    Some(replacement) => Ok(replacement),
                    None => media
                        .original_value_for_layers(Some(json_layers))
                        .map_err(failure),
                }
            },
        )
        .map(|_| ())
    };
    let BatchRows::Prepared(events) = &mut state.rows else {
        unreachable!("media resolution precedes enrichment")
    };
    for event in events {
        for value in event
            .input
            .iter_mut()
            .chain(event.output.iter_mut())
            .chain(event.metadata.values_mut())
        {
            visit_strings_mut(value, &mut resolve_text)?;
        }
    }
    if pending.next().is_some() {
        return Err(failure("incomplete OTEL media traversal"));
    }
    Ok(())
}

fn enrich_inner(state: &mut Pipeline, lookups: Vec<serde_json::Value>) -> NativeResult<()> {
    let BatchRows::Prepared(events) = std::mem::replace(&mut state.rows, BatchRows::Finalized)
    else {
        return Err(failure("OTEL batch already enriched"));
    };
    let mut lookups = lookups.into_iter().map(Value::from);
    let mut rows = Vec::with_capacity(events.len());
    for event in events {
        let lookup = if needs_lookup(&event) {
            lookups.next().expect("validated lookup count")
        } else {
            Value::Null
        };
        if lookup.get("error").is_some() {
            rows.push(None);
            continue;
        }
        let enrichment = crate::otel_enrichment::enrich(&event, &lookup);
        let start_time_iso = event.start_time.clone();
        let mut row = event.into_storage(enrichment, &state.context);
        if !state.source.media.is_empty() {
            row.restore_media(&mut |text| {
                restore_text(text, &state.source.media).map_err(|error| error.to_string())
            })
            .map_err(failure)?;
        }
        rows.push(Some(EnrichedStorageRow {
            row,
            start_time_iso,
        }));
    }
    state.rows = BatchRows::Ready(rows);
    Ok(())
}

fn overflow_requests_inner(
    state: &mut Pipeline,
    limit_bytes: f64,
) -> NativeResult<Vec<OtelOverflowRequest>> {
    let rows = state.rows.ready()?;
    let mut requests = Vec::new();
    for (event_index, row) in rows.iter().enumerate() {
        let Some(row) = row else {
            continue;
        };
        let mut add = |field: &str, metadata_index: Option<u32>, value: Option<&String>| {
            if let Some(text) = value.filter(|text| text.len() as f64 > limit_bytes) {
                requests.push(OtelOverflowRequest {
                    index: requests.len() as u32,
                    event_index: event_index as u32,
                    trace_id: row.row.trace_id.clone(),
                    observation_id: row.row.span_id.clone(),
                    field: field.to_owned(),
                    metadata_index,
                    byte_length: text.len() as f64,
                    sha256_hash: base64::engine::general_purpose::STANDARD
                        .encode(Sha256::digest(text.as_bytes())),
                });
            }
        };
        add("input", None, row.row.input.as_ref());
        add("output", None, row.row.output.as_ref());
        for (index, value) in row.row.metadata_values.iter().enumerate() {
            add("metadata", Some(index as u32), Some(value));
        }
    }
    state.overflow_requests = requests.clone();
    Ok(requests)
}

/// Requests are generated from these rows and their indices never cross a mutable JS boundary.
fn overflow_text<'a>(
    request: &OtelOverflowRequest,
    rows: &'a [Option<EnrichedStorageRow>],
) -> &'a String {
    let row = &rows[request.event_index as usize]
        .as_ref()
        .expect("overflow row exists")
        .row;
    match request.metadata_index {
        Some(index) => &row.metadata_values[index as usize],
        None if request.field == "input" => row.input.as_ref().expect("overflow input exists"),
        None => row.output.as_ref().expect("overflow output exists"),
    }
}

fn finalize_inner(
    state: &mut Pipeline,
    overflow: Vec<OtelMediaResolution>,
    include_evaluation_rows: bool,
    trace_activity_ids: Option<Vec<String>>,
) -> NativeResult<FinalizedOtelBatch> {
    validate_results(
        &overflow,
        state.overflow_requests.len(),
        "incomplete overflow results",
    )?;
    let rows = state.rows.ready()?;
    let evaluation_rows = if include_evaluation_rows {
        evaluation_rows(rows)?
    } else {
        Vec::new()
    };
    let BatchRows::Ready(mut rows) = std::mem::replace(&mut state.rows, BatchRows::Finalized)
    else {
        unreachable!("ready rows checked")
    };
    for (result, request) in overflow.into_iter().zip(&state.overflow_requests) {
        let Some(reference) = result.reference else {
            continue;
        };
        let row = &mut rows[request.event_index as usize]
            .as_mut()
            .expect("overflow row exists")
            .row;
        match request.metadata_index {
            Some(index) => row.metadata_values[index as usize] = reference,
            None if request.field == "input" => row.input = Some(reference),
            None => row.output = Some(reference),
        }
    }
    let trace_activity_ids = trace_activity_ids
        .unwrap_or_default()
        .into_iter()
        .collect::<HashSet<_>>();
    let events = rows
        .into_iter()
        .flatten()
        .map(|row| {
            let serialized_event_bytes = trace_activity_ids
                .contains(&row.row.trace_id)
                .then(|| row.row.serialized_byte_length())
                .transpose()
                .map_err(failure)?
                .map(|bytes| bytes as f64);
            let environment = row.row.environment.clone();
            let trace_id = row.row.trace_id.clone();
            let span_id = row.row.span_id.clone();
            let prepared = row.row.into_prepared().map_err(|error| {
                failure(format!("cannot finalize OTEL event {span_id}: {error}"))
            })?;
            Ok(FinalizedOtelEvent {
                event: PreparedEvent {
                    row: Arc::new(prepared),
                },
                trace_id,
                start_time_iso: row.start_time_iso,
                environment,
                serialized_event_bytes,
            })
        })
        .collect::<NativeResult<_>>()?;
    Ok(FinalizedOtelBatch {
        events,
        evaluation_rows,
    })
}

fn evaluation_rows(rows: &[Option<EnrichedStorageRow>]) -> NativeResult<Vec<serde_json::Value>> {
    rows.iter()
        .flatten()
        .map(|row| {
            let mut value = row.row.evaluation_value().map_err(failure)?;
            // Evaluation consumers expect parsed model parameters rather than their storage string.
            if let Some(text) = value["model_parameters"].as_str() {
                if let Ok(parsed) = serde_json::from_str(text) {
                    value["model_parameters"] = parsed;
                }
            }
            Ok(value)
        })
        .collect()
}

fn failure(error: impl std::fmt::Display) -> Error<&'static str> {
    Error::new("ERR_OTEL_PROCESSING", error.to_string())
}

/// Visit string values in source order. Object keys are structural and are never rewritten.
fn visit_strings_mut(
    value: &mut Value,
    visit: &mut impl FnMut(&mut JsString) -> NativeResult<()>,
) -> NativeResult<()> {
    match value {
        Value::String(text) => visit(text),
        Value::Array(values) => {
            for value in values {
                visit_strings_mut(value, visit)?;
            }
            Ok(())
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                visit_strings_mut(value, visit)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn media_entry(state: &Pipeline, index: u32) -> NativeResult<&ExtractedMedia> {
    media_entry_parts(&state.source, &state.late_media, index)
}

fn media_entry_parts<'a>(
    source: &'a ExtractedBatch,
    late_media: &'a [ExtractedMedia],
    index: u32,
) -> NativeResult<&'a ExtractedMedia> {
    let index = index as usize;
    source
        .media
        .get(index)
        .or_else(|| late_media.get(index.saturating_sub(source.media.len())))
        .ok_or_else(|| failure("unknown media index"))
}

fn needs_lookup(event: &OwnedEvent) -> bool {
    event
        .provided_model_name
        .as_ref()
        .is_some_and(|name| !name.is_empty())
        || (event
            .prompt_name
            .as_ref()
            .is_some_and(|name| !name.is_empty())
            && event
                .prompt_version_lookup
                .is_some_and(|version| version != 0.0 && !version.is_nan()))
        || !event.skills_available.is_empty()
        || !event.skills_resource_loaded.is_empty()
}

fn collect_value_media_references(value: &Value, references: &mut HashSet<String>) {
    match value {
        Value::String(text) => collect_media_references(text.as_str(), references),
        Value::Array(values) => {
            for value in values {
                collect_value_media_references(value, references);
            }
        }
        Value::Object(values) => {
            // Object keys are structural and are deliberately excluded from
            // both discovery and replacement.
            for value in values.values() {
                collect_value_media_references(value, references);
            }
        }
        _ => {}
    }
}

// Some providers construct media objects from OTLP key/value arrays during normalization.
// Those candidates cannot be recognized as structured media in the raw document.
fn extract_remaining_media(
    value: &mut Value,
    registry: &mut Vec<ExtractedMedia>,
) -> NativeResult<()> {
    extract_normalized_media(value, registry).map_err(failure)
}

fn restore_value(value: &mut Value, registry: &[ExtractedMedia]) -> NativeResult<()> {
    visit_strings_mut(value, &mut |text| restore_js_text(text, registry))
}

fn restore_text(text: &mut String, registry: &[ExtractedMedia]) -> NativeResult<()> {
    if !text.contains("@@@langfuseMedia:") {
        return Ok(());
    }
    let mut owned = JsString::from(std::mem::take(text));
    let result = restore_js_text(&mut owned, registry);
    *text = owned.into_utf8_lossy();
    result
}

fn restore_js_text(text: &mut JsString, registry: &[ExtractedMedia]) -> NativeResult<()> {
    if !text.as_str().contains("@@@langfuseMedia:") {
        return Ok(());
    }
    rewrite_known_references(
        text,
        |reference| registry.iter().any(|media| media.reference() == reference),
        |_| true,
        |reference, json_layers| {
            registry
                .iter()
                .find(|media| media.reference() == reference)
                .ok_or_else(|| failure("unknown media reference"))?
                .original_value_for_layers(Some(json_layers))
                .map_err(failure)
        },
    )
    .map(|_| ())
}
