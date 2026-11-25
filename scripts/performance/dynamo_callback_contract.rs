// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::{Value, json};

fn config(speculation: Value, nextn: u32) -> Value {
    json!({
        "model": "Qwen/Qwen3-8B", "system": "h200_sxm", "backend": "vllm",
        "backend_version": "current", "worker_type": "aggregated",
        "estimation_mode": "op_level", "database_mode": "SOL",
        "speculation": speculation, "nextn": nextn,
    })
}

fn assert_decode(config: Value, expected_depth: u32) {
    let callback = build_model(&config).unwrap();
    assert_eq!(callback.nextn, expected_depth);
    // Independent packing: two requests, context 129, width depth + 1.
    // Do not call the adapter's decode_metrics helper to form the oracle.
    let packed = 2 * (expected_depth + 1);
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: packed,
            sum_decode_kv_tokens: packed * 129,
            ..Default::default()
        },
        ..Default::default()
    };
    let expected = callback
        .model
        .estimate_forward_pass_time_ms(&[metrics])
        .unwrap()
        .unwrap();
    let actual = callback.predict_decode(2, 128, 2).unwrap();
    assert!(actual.is_finite() && actual > 0.0);
    assert!((actual - expected).abs() <= expected.abs() * 1e-12);
}

#[test]
fn autoregressive_and_legacy_depth_reach_actual_decode() {
    assert_decode(config(Value::Null, 0), 0);
    for depth in [1, 5] {
        assert_decode(config(Value::Null, depth), depth);
    }
}

#[test]
fn ngram_depth_reaches_actual_decode() {
    for depth in [1, 5] {
        assert_decode(
            config(
                json!({"kind": "ngram", "params": {"num_speculative_tokens": depth}}),
                0,
            ),
            depth,
        );
    }
}

#[test]
fn schemes_other_than_ngram_are_rejected() {
    for spec in [
        json!({"kind": "none"}),
        json!({"kind": "none", "params": {}}),
        json!({"kind": "mtp", "params": {"depth": 1}}),
    ] {
        let error = build_model(&config(spec.clone(), 0))
            .err()
            .unwrap_or_else(|| panic!("{spec} must not enable replay"));
        assert!(
            error
                .to_string()
                .contains("only ngram speculation or legacy nextn"),
            "{error}"
        );
    }
}

#[test]
fn invalid_ngram_depth_is_rejected() {
    for depth in [json!(-1), json!(1.5), json!(true), json!("1"), json!(u64::from(u32::MAX) + 1)] {
        let error = build_model(&config(
            json!({"kind": "ngram", "params": {"num_speculative_tokens": depth.clone()}}),
            0,
        ))
        .err()
        .unwrap_or_else(|| panic!("ngram depth {depth} must be rejected"));
        assert!(
            error
                .to_string()
                .contains("num_speculative_tokens must be a nonnegative u32"),
            "{error}"
        );
    }
}
