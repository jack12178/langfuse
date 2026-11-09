import { describe, expect, it } from "vitest";
import {
  PreparedEvent,
  encodeClickhouseEvents,
  validateOtelJson,
} from "@langfuse/native";
import { codecSpan, prepareCodecRows } from "./helpers/nativeCodecFixtures";

describe("Native codec NAPI boundary", () => {
  it("associates sparse lookup results with their events and retains events needing no I/O", async () => {
    const selectedTraceId = "0123456789abcdef0123456789abcdef";
    const unselectedTraceId = "fedcba9876543210fedcba9876543210";
    const spans = [0, 1, 2].map((index) => ({
      traceId: index === 2 ? unselectedTraceId : selectedTraceId,
      spanId: `0123456789abcde${index}`,
      name: "sparse-lookups".repeat(1000),
      startTimeUnixNano: "1750000000000000000",
      endTimeUnixNano: "1750000001000000000",
      attributes:
        index === 1
          ? [
              {
                key: "gen_ai.request.model",
                value: { stringValue: "test-model" },
              },
            ]
          : [],
    }));
    const validated = await validateOtelJson(
      Buffer.from(JSON.stringify([{ resource: {}, scopeSpans: [{ spans }] }])),
    );
    const batch = await validated.extract(true);
    const {
      batch: processing,
      mediaRequests,
      lookupRequests,
    } = await batch.process({
      projectId: "native-boundary",
      fileKey: "sparse-lookups.json",
      now: "2026-09-24T00:00:00.000Z",
      ingestionApiKey: "",
      ingestionSdkName: "test",
      ingestionSdkVersion: "test",
      isLangfuseInternal: false,
    });
    try {
      await expect(batch.copyJsonBuffer()).rejects.toMatchObject({
        code: "ERR_OTEL_CLOSED",
      });
      expect(mediaRequests).toEqual([]);
      expect(lookupRequests).toMatchObject([
        { spanId: "0123456789abcde1", modelName: "test-model" },
      ]);
      await expect(
        processing.complete([], [], undefined, true),
      ).rejects.toThrow("incomplete OTEL lookup results");
      const { finalized } = await processing.complete(
        [],
        [{ model: { id: "matched-model" } }],
        undefined,
        true,
        [selectedTraceId],
      );
      expect(finalized).toBeDefined();
      expect(finalized!.evaluationRows).toMatchObject([
        { span_id: "0123456789abcde0", model_id: null },
        { span_id: "0123456789abcde1", model_id: "matched-model" },
        { span_id: "0123456789abcde2", model_id: null },
      ]);
      expect(finalized!.events).toHaveLength(3);
      expect(finalized!.events[0]?.serializedEventBytes).toBeGreaterThan(
        Buffer.byteLength(spans[0].name),
      );
      expect(finalized!.events[2]?.serializedEventBytes).toBeUndefined();
      expect(
        PreparedEvent.columns().some(({ name }) => name === "event_bytes"),
      ).toBe(false);
    } finally {
      await processing.dispose();
      await batch.dispose();
      await validated.dispose();
    }
  });

  it("rejects the entire Rust batch when an event cannot fit the storage schema", async () => {
    // Valid JSON and ingestion usage can still exceed ClickHouse's UInt64 range. Returning
    // just the first event would let the queue acknowledge a partially lost batch.
    const spans = [1, 1e30].map((usage, index) => ({
      traceId: "0123456789abcdef0123456789abcdef",
      spanId: `0123456789abcde${index}`,
      name: "storage-boundary",
      startTimeUnixNano: "1750000000000000000",
      endTimeUnixNano: "1750000001000000000",
      attributes: [
        {
          key: "langfuse.observation.usage_details",
          value: { stringValue: JSON.stringify({ input: usage }) },
        },
      ],
    }));
    const validated = await validateOtelJson(
      Buffer.from(JSON.stringify([{ resource: {}, scopeSpans: [{ spans }] }])),
    );
    const batch = await validated.extract(true);
    const {
      batch: processing,
      mediaRequests,
      lookupRequests,
    } = await batch.process({
      projectId: "native-boundary",
      fileKey: "storage-boundary.json",
      now: "2026-09-24T00:00:00.000Z",
      ingestionApiKey: "",
      ingestionSdkName: "test",
      ingestionSdkVersion: "test",
      isLangfuseInternal: false,
    });
    try {
      expect(mediaRequests).toEqual([]);
      expect(lookupRequests).toEqual([]);
      await expect(
        processing.complete([], [], undefined, true),
      ).rejects.toThrow("usage_details");
    } finally {
      await processing.dispose();
      await batch.dispose();
      await validated.dispose();
    }
  });

  it("retains Rust-owned rows while handles and evaluation snapshots change during encoding", async () => {
    const { events, evaluationRows } = await prepareCodecRows([
      codecSpan(1, {
        "langfuse.observation.input": "🔥".repeat(1024),
        "langfuse.observation.usage_details": '{"input":42}',
        "langfuse.observation.cost_details": '{"input":0.25}',
      }),
    ]);
    const prepared = events[0].event;
    const expected = await encodeClickhouseEvents([prepared], 1);
    const handles = Object.assign([prepared], {
      toJSON: () => {
        throw new Error("the NAPI boundary must not serialize the batch");
      },
    });
    const pending = encodeClickhouseEvents(handles, 1);
    Object.assign(evaluationRows[0], {
      input: "changed",
      usage_details: { input: 99 },
      cost_details: { input: 99 },
    });
    handles.length = 0;

    const blocks = await pending;
    expect(blocks).toEqual(expected);
    expect(prepared.ids).toEqual({
      project_id: "native-boundary",
      trace_id: "0123456789abcdef0123456789abcdef",
      id: "0000000000000001",
    });
    prepared.ids.id = "changed";
    expect(prepared.ids.id).toBe("0000000000000001");
    expect(await encodeClickhouseEvents([prepared], 1)).toEqual(expected);
    expect(Buffer.isBuffer(blocks[0].bytes)).toBe(true);
    expect(blocks[0].bytes.length).toBeGreaterThan(0);
  });

  it("merges handles into one wire block while retaining the input handles", async () => {
    const { events } = await prepareCodecRows([
      codecSpan(1),
      codecSpan(2),
      codecSpan(3),
    ]);
    const handles = events.map(({ event }) => event);
    const [combined] = await encodeClickhouseEvents(handles, 3);
    const separate = await encodeClickhouseEvents(handles, 2);
    expect(combined.rowCount).toBe(3);
    expect(separate.map((block) => block.rowCount)).toEqual([2, 1]);
    // Batching writes the schema header once, instead of concatenating two complete blocks.
    expect(combined.bytes.length).toBeLessThan(
      separate.reduce((size, block) => size + block.bytes.length, 0),
    );
    expect(await encodeClickhouseEvents(handles, 3)).toEqual([combined]);
    expect(await encodeClickhouseEvents([], 1)).toEqual([]);
  });

  it.each([0, -1, 1.5, NaN, Infinity])(
    "rejects invalid JS block row limit %s",
    async (maxRows) => {
      const { events } = await prepareCodecRows();
      await expect(async () =>
        encodeClickhouseEvents([events[0].event], maxRows),
      ).rejects.toThrow();
    },
  );
});
