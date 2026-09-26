# Phase 10.6A Dataset Forensics / Validation

This report intentionally performs provenance and raw-data validation only. It does not modify the canonical ICT 2022 strategy logic, sniper gate, execution logic, or live-trading configuration.

## Summary

- Generated at UTC: 2026-09-26T00:35:08.598615+00:00
- Dataset count: 2
- Status: forensic_review_only

## PRIMARY — dukascopy

- Source: Dukascopy free-data export
- Source timezone: UTC
- Internal timezone: UTC
- Timeframe: tick
- Rows: 23531
- First timestamp (UTC): 2022-03-01T12:00:00+00:00
- Last timestamp (UTC): 2022-03-01T12:59:59+00:00
- Duplicate rows after first: 20117
- Duplicate rows in duplicate groups: 23322
- Duplicate timestamp values: 3205
- Non-monotonic rows: 0
- Invalid OHLC rows: 0
- Negative volume rows: 0
- Volume semantics: UNKNOWN

### Gap distribution

- 0 days 00:00:01: 3263
- 0 days 00:00:02: 125
- 0 days 00:00:03: 17
- 0 days 00:00:04: 6
- 0 days 00:00:05: 1
- 0 days 00:00:06: 1

### Largest gaps

- 2022-03-01T12:33:46+00:00 -> 2022-03-01T12:33:52+00:00 (0 days 00:00:06)
- 2022-03-01T12:20:12+00:00 -> 2022-03-01T12:20:17+00:00 (0 days 00:00:05)
- 2022-03-01T12:44:18+00:00 -> 2022-03-01T12:44:22+00:00 (0 days 00:00:04)
- 2022-03-01T12:44:03+00:00 -> 2022-03-01T12:44:07+00:00 (0 days 00:00:04)
- 2022-03-01T12:18:19+00:00 -> 2022-03-01T12:18:23+00:00 (0 days 00:00:04)

### Notes

- Raw observations are preserved without forward-filling or synthetic candle generation.
- Duplicate timestamps are kept for Dukascopy raw tick observations and are not silently dropped.
- Octa MT4 values are normalized from local UTC+3 provenance to internal UTC before use.
- The 15-minute Octa dataset is treated as a 15min source only and not coerced to a finer timeframe.
- No missing timestamps are filled or synthesized; timestamps are validated as-is from the originating raw file.
- Octa 15M history is never resampled downward into 5M, 3M, or 1M series for research use.

## SECONDARY — octa_mt4

- Source: Octa MT4 History Center
- Source timezone: UTC+3
- Internal timezone: UTC
- Timeframe: 15min
- Rows: 494235
- First timestamp (UTC): 2004-06-11T04:15:00+00:00
- Last timestamp (UTC): 2026-01-30T20:45:00+00:00
- Duplicate rows after first: 0
- Duplicate rows in duplicate groups: 0
- Duplicate timestamp values: 0
- Non-monotonic rows: 0
- Invalid OHLC rows: 0
- Negative volume rows: 0
- Volume semantics: UNKNOWN

### Gap distribution

- 0 days 00:15:00: 485714
- 0 days 00:30:00: 1814
- 0 days 00:45:00: 322
- 0 days 01:00:00: 729
- 0 days 01:15:00: 3406
- 0 days 01:30:00: 300
- 0 days 01:45:00: 216
- 0 days 02:00:00: 86
- 0 days 02:15:00: 313
- 0 days 02:30:00: 10
- 0 days 02:45:00: 14
- 0 days 03:00:00: 8
- 0 days 03:15:00: 4
- 0 days 03:30:00: 6
- 0 days 03:45:00: 28
- 0 days 04:00:00: 29
- 0 days 04:15:00: 2
- 0 days 04:30:00: 3
- 0 days 04:45:00: 7
- 0 days 05:00:00: 6
- 0 days 05:15:00: 35
- 0 days 05:30:00: 2
- 0 days 05:45:00: 3
- 0 days 06:00:00: 1
- 0 days 06:15:00: 2
- 0 days 07:00:00: 4
- 0 days 07:15:00: 2
- 0 days 07:30:00: 1
- 0 days 08:30:00: 2
- 0 days 09:00:00: 1
- 0 days 10:15:00: 3
- 0 days 10:30:00: 1
- 0 days 11:15:00: 1
- 0 days 12:00:00: 1
- 0 days 19:00:00: 1
- 0 days 23:15:00: 1
- 1 days 01:15:00: 9
- 1 days 01:30:00: 1
- 1 days 02:15:00: 2
- 1 days 04:30:00: 2
- 1 days 05:15:00: 2
- 1 days 06:15:00: 1
- 1 days 09:15:00: 2
- 1 days 09:45:00: 2
- 1 days 13:15:00: 1
- 1 days 13:45:00: 1
- 1 days 14:15:00: 4
- 1 days 16:15:00: 1
- 1 days 16:30:00: 1
- 1 days 17:15:00: 2
- 2 days 00:15:00: 11
- 2 days 01:15:00: 645
- 2 days 01:30:00: 6
- 2 days 01:45:00: 1
- 2 days 02:00:00: 3
- 2 days 02:15:00: 268
- 2 days 02:30:00: 16
- 2 days 02:45:00: 23
- 2 days 03:00:00: 15
- 2 days 03:15:00: 19
- 2 days 03:30:00: 4
- 2 days 03:45:00: 3
- 2 days 04:00:00: 3
- 2 days 04:15:00: 5
- 2 days 04:30:00: 11
- 2 days 04:45:00: 1
- 2 days 05:00:00: 4
- 2 days 05:15:00: 9
- 2 days 05:30:00: 7
- 2 days 05:45:00: 6
- 2 days 06:15:00: 1
- 2 days 06:30:00: 1
- 2 days 07:00:00: 1
- 2 days 07:15:00: 1
- 2 days 07:45:00: 1
- 2 days 20:00:00: 1
- 2 days 21:15:00: 1
- 3 days 01:15:00: 19
- 3 days 01:30:00: 4
- 3 days 02:00:00: 1
- 3 days 02:15:00: 3
- 3 days 04:15:00: 1
- 3 days 04:30:00: 1
- 3 days 06:00:00: 1
- 3 days 06:15:00: 4
- 3 days 06:45:00: 1
- 3 days 08:00:00: 1
- 3 days 09:15:00: 1
- 3 days 09:30:00: 1
- 3 days 10:15:00: 1
- 3 days 10:30:00: 1
- 3 days 13:15:00: 2
- 3 days 13:30:00: 4
- 3 days 13:45:00: 5
- 3 days 14:00:00: 2
- 3 days 14:15:00: 1
- 4 days 06:00:00: 1
- 4 days 13:15:00: 1
- 9 days 04:30:00: 1
- 32 days 08:00:00: 1

### Largest gaps

- 2025-09-12T20:45:00+00:00 -> 2025-10-15T04:45:00+00:00 (32 days 08:00:00)
- 2026-01-13T11:15:00+00:00 -> 2026-01-22T15:45:00+00:00 (9 days 04:30:00)
- 2005-11-23T16:15:00+00:00 -> 2005-11-28T05:30:00+00:00 (4 days 13:15:00)
- 2006-11-22T16:00:00+00:00 -> 2006-11-26T22:00:00+00:00 (4 days 06:00:00)
- 2006-02-17T15:15:00+00:00 -> 2006-02-21T05:30:00+00:00 (3 days 14:15:00)

### Notes

- Raw observations are preserved without forward-filling or synthetic candle generation.
- Duplicate timestamps are kept for Dukascopy raw tick observations and are not silently dropped.
- Octa MT4 values are normalized from local UTC+3 provenance to internal UTC before use.
- The 15-minute Octa dataset is treated as a 15min source only and not coerced to a finer timeframe.
- No missing timestamps are filled or synthesized; timestamps are validated as-is from the originating raw file.
- Octa 15M history is never resampled downward into 5M, 3M, or 1M series for research use.
