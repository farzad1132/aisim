// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM async scheduling in offline replay.

use std::collections::VecDeque;

use aisim_core::engine::{
    EngineConfig, G2Connector, NativeHostOffloadConfig, PassLaunchConfig, TimingModelConfig,
};
use aisim_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayArtifactKvEventVisibility,
    ReplayEngineConfig, ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec,
    ReplayTopology, Replayer, WorkerPoolSpec,
};
use uuid::Uuid;

/// One vLLM worker with `num_gpu_blocks` 4-token blocks. A pass with prefill
/// takes 100 ms and a decode pass 10 ms.
fn engine(num_gpu_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_gpu_blocks,
        max_num_batched_tokens: 64,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 100.0,
            decode_ms: 10.0,
        },
        ..EngineConfig::default()
    }
}

/// `engine` with `launch`: launching a pass takes `ms` without prefill, with up
/// to 16 scheduled tokens and above.
fn launch(engine: EngineConfig, ms: [f64; 3], eager_queue_fraction: f64) -> EngineConfig {
    let [decode_ms, piecewise_ms, eager_ms] = ms;
    EngineConfig {
        launch: Some(PassLaunchConfig {
            decode_ms,
            piecewise_ms,
            eager_ms,
            max_cudagraph_capture_size: 16,
            eager_queue_fraction,
        }),
        ..engine
    }
}

fn spec(engine: ReplayEngineConfig) -> ReplaySpec {
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 1,
                ..WorkerPoolSpec::default()
            },
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

/// A request with `(id, prompt tokens, outputs, arrival)`.
fn request(&(id, prompt, outputs, arrival_ms): &(u128, u32, usize, f64)) -> DirectRequest {
    DirectRequest {
        tokens: (0..prompt).map(|token| token + id as u32 * 1000).collect(),
        max_output_tokens: outputs,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn replay_requests(engine: ReplayEngineConfig, requests: VecDeque<DirectRequest>) -> Replayer {
    Replayer::new(spec(engine), ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(requests))
}

/// A replayer of `(id, prompt tokens, outputs, arrival)` requests.
fn replayer(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> Replayer {
    replay_requests(
        ReplayEngineConfig {
            rank: engine,
            ..Default::default()
        },
        requests.iter().map(request).collect(),
    )
}

fn run(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> ReplayReport {
    replayer(engine, requests).run().unwrap()
}

fn record(report: &ReplayReport, id: u128) -> &PerRequestRecord {
    let uuid = Uuid::from_u128(id).to_string();
    report
        .per_request
        .iter()
        .find(|record| record.uuid == uuid)
        .unwrap()
}

#[test]
fn arrivals_after_the_last_running_request_get_its_blocks_at_once() {
    // R (24 tokens) prefills until 100 and finishes in its decode pass at 110.
    // Requests 2 and 3 arrive during that pass, so vLLM had already decided
    // the pass at 110, which has nothing to run. That empty step frees R's
    // blocks, and vLLM then admits both at 110, though request 3 (16 tokens)
    // fits only in R's blocks.
    let report = run(
        engine(10),
        &[(1, 24, 2, 0.0), (2, 4, 2, 105.0), (3, 16, 2, 106.0)],
    );
    assert_eq!(
        [2, 3].map(|id| record(&report, id).first_admit_ms),
        [Some(110.0); 2]
    );
}

#[test]
fn every_attention_dp_rank_frees_finished_blocks_after_the_next_decision() {
    // On rank 0, X and R prefill until 100, and W (16 tokens), arriving at 50,
    // fits only in R's blocks. R finishes in its decode pass, which ends at
    // 110, but rank 1's prefill of Y holds the group pass until 200. Rank 0
    // decides its pass at 200 before it frees R's blocks, as a single rank
    // would, so that pass decodes only X and W is admitted at 210.
    let requests = [
        ((1, 8, 4, 0.0), 0),
        ((2, 24, 2, 0.0), 0),
        ((3, 16, 2, 50.0), 0),
        ((4, 8, 1, 50.0), 1),
    ]
    .iter()
    .map(|(fields, rank)| DirectRequest {
        preferred_dp_rank: Some(*rank),
        ..request(fields)
    })
    .collect();
    let two_ranks = ReplayEngineConfig {
        dp_size: 2,
        rank: engine(10),
        ..Default::default()
    };
    let report = replay_requests(two_ranks, requests).run().unwrap();
    assert_eq!(record(&report, 3).first_admit_ms, Some(210.0));
}

/// A prefills from 0 to 100 and decodes until 110 and 120; B (32 tokens)
/// arrives while the first decode runs, so the pass from 120 is eager.
const EAGER_AFTER_DECODE: [(u128, u32, usize, f64); 2] = [(1, 8, 3, 0.0), (2, 32, 1, 105.0)];

#[test]
fn a_pass_outputs_leave_once_the_next_pass_is_launched() {
    // vLLM decided the eager pass at 110, once the decode before it had
    // launched and the pass before that had ended, so it launches it until
    // 110 + 50. A's last token and its completion leave then, instead of at
    // 120. If the launch waits until the GPU has run 0.8 of that 100 ms pass,
    // they leave at 200. Passes do not move: B runs from 120 to 220.
    let plain = run(engine(64), &EAGER_AFTER_DECODE);
    let a = record(&plain, 1);
    assert_eq!(
        (a.first_token_ms, a.last_token_ms),
        (Some(100.0), Some(120.0))
    );
    for (fraction, returned_ms) in [(0.0, 160.0), (0.8, 200.0)] {
        let report = run(
            launch(engine(64), [2.0, 20.0, 50.0], fraction),
            &EAGER_AFTER_DECODE,
        );
        let a = record(&report, 1);
        assert_eq!(a.first_token_ms, Some(100.0));
        assert_eq!(a.last_token_ms, Some(returned_ms));
        assert_eq!(a.terminal_time_ms, returned_ms);
        let b = record(&report, 2);
        assert_eq!(
            (b.first_admit_ms, b.first_token_ms),
            (Some(120.0), Some(220.0))
        );
    }
}

#[test]
fn held_outputs_leave_before_those_of_the_pass_that_releases_them() {
    // A decodes until 120, and its last token comes from the pass that also
    // prefills B, from 120 to 230. vLLM decided that eager pass at 110, and
    // its 150 ms launch lasts until the pass ends, so A's third token, held
    // since 120, and its last token both leave at 230, in that order.
    let report = run(
        launch(engine(64), [2.0, 20.0, 150.0], 0.0),
        &[(1, 8, 4, 0.0), (2, 32, 1, 105.0)],
    );
    assert_eq!(record(&report, 1).last_token_ms, Some(230.0));
}

#[test]
fn each_worker_returns_its_held_outputs_at_its_own_launch() {
    // Round-robin placement gives worker 0 the requests of EAGER_AFTER_DECODE,
    // whose A leaves at 160, and worker 1 the same pair 3 ms later with an
    // 8-token B. Worker 1 decides that pass at 113 and launches it piecewise
    // for 20 ms, so its A leaves at 133: after worker 0 starts to hold its
    // A, at 120, and before worker 0 returns it.
    let requests = [
        (1, 8, 3, 0.0),
        (3, 8, 3, 3.0),
        (2, 32, 1, 105.0),
        (4, 8, 1, 108.0),
    ];
    let two_workers = ReplaySpec {
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 2,
                ..WorkerPoolSpec::default()
            },
        },
        ..spec(ReplayEngineConfig {
            rank: launch(engine(64), [2.0, 20.0, 50.0], 0.0),
            ..Default::default()
        })
    };
    let report = Replayer::new(two_workers, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(
            requests.iter().map(request).collect(),
        ))
        .run()
        .unwrap();
    assert_eq!(
        [1, 3].map(|id| record(&report, id).last_token_ms),
        [Some(160.0), Some(133.0)]
    );
}

#[test]
fn a_decode_pass_above_the_graph_limit_launches_eager() {
    // A, B and C prefill until 100, and A's last token comes from the decode
    // pass that ends at 110. With a one-token graph limit the two-token decode
    // pass after it runs eager: vLLM decided it at 100 and launches it for
    // 50 ms, so A's last token leaves when that pass ends at 120, not at 110.
    let mut config = launch(engine(64), [2.0, 20.0, 50.0], 0.0);
    config.launch.as_mut().unwrap().max_cudagraph_capture_size = 1;
    let report = run(config, &[(1, 3, 2, 0.0), (2, 3, 3, 0.0), (3, 3, 3, 0.0)]);
    assert_eq!(record(&report, 1).last_token_ms, Some(120.0));
}

#[test]
fn detokenization_starts_when_vllm_returns_the_outputs() {
    // A 1 ms detokenizer takes A's last token when vLLM returns it at 160.
    let launched = replayer(
        launch(engine(64), [2.0, 20.0, 50.0], 0.0),
        &EAGER_AFTER_DECODE,
    )
    .with_detokenization(1, 1.0, 0.0)
    .unwrap()
    .run()
    .unwrap();
    assert_eq!(record(&launched, 1).last_token_ms, Some(161.0));
}

#[test]
fn an_arrival_joins_the_first_pass_decided_after_it() {
    // A's prefill runs from 0 to 100 and vLLM decides the next pass once it
    // has launched it, at 20. B arrives before that and joins the next pass,
    // which prefills B and decodes A until 210; C arrives after it and waits
    // for the pass after. Counting passes, both would skip the pass vLLM had
    // already queued and join the one from 110.
    let requests = [(1, 8, 2, 0.0), (2, 8, 1, 10.0), (3, 8, 1, 30.0)];
    let admitted =
        |report: &ReplayReport| [2, 3].map(|id| record(report, id).first_admit_ms.unwrap());
    assert_eq!(admitted(&run(engine(64), &requests)), [110.0, 110.0]);
    assert_eq!(
        admitted(&run(launch(engine(64), [2.0, 20.0, 50.0], 0.0), &requests)),
        [100.0, 210.0]
    );
    // With no launch cost vLLM decides each pass when the one before it
    // starts, so the rule counts passes as before: B, arriving mid-pass at
    // 105, skips one pass; C, arriving as a pass ends at 110, skips two.
    let requests = [(1, 8, 10, 0.0), (2, 8, 1, 105.0), (3, 8, 1, 110.0)];
    assert_eq!(
        admitted(&run(launch(engine(64), [0.0; 3], 0.0), &requests)),
        admitted(&run(engine(64), &requests))
    );
}

#[test]
fn launch_needs_a_one_gpu_engine() {
    // With more GPUs vLLM's multiprocess executor returns a pass's outputs
    // independently of the next launch.
    let two_gpus = ReplayEngineConfig {
        tensor_parallel_size: 2,
        rank: launch(engine(64), [2.0, 20.0, 50.0], 0.0),
        ..Default::default()
    };
    let error = Replayer::new(spec(two_gpus), ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap_err();
    assert!(error.to_string().contains("uniprocess executor"), "{error}");
}

/// `launch` with prefix caching and a rank-local `offloading_v0_32` CPU tier
/// of 64 blocks: stores copy at once, and loads copy 16 B per token at
/// `h2d_gbps`.
fn polled_cpu_tier(num_gpu_blocks: usize, h2d_gbps: f64) -> EngineConfig {
    let mut config = launch(engine(num_gpu_blocks), [2.0, 20.0, 50.0], 0.0);
    config.enable_prefix_caching = true;
    config.kv_cache_bytes_per_token = Some(16);
    config.native_host_offload = Some(
        NativeHostOffloadConfig::new(64)
            .with_bandwidths(0.0, h2d_gbps)
            .with_connector(G2Connector::OffloadingV032),
    );
    config
}

/// A request with `(id, tokens, outputs, arrival)`.
fn tokens_request(id: u128, tokens: Vec<u32>, outputs: usize, arrival_ms: f64) -> DirectRequest {
    DirectRequest {
        tokens,
        max_output_tokens: outputs,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn run_tokens(engine: EngineConfig, requests: Vec<DirectRequest>) -> ReplayReport {
    replay_requests(
        ReplayEngineConfig {
            rank: engine,
            ..Default::default()
        },
        requests.into(),
    )
    .run()
    .unwrap()
}

#[test]
fn a_simple_cpu_copy_finishing_before_its_issuing_pass_forms_reuses_host_tokens() {
    // Request 1 stores five prompt blocks in G2; request 2 evicts them from
    // G1. R (3) prefills from 600 to 700. N (4) arrives after R's launch, so
    // it joins the pass decided at 700, prefilling beside R's decode from
    // 710 to 820. A (5) arrives at 701 and issues its load in the next pass,
    // m, decided at 710. SimpleCPU starts copying at m's launch end, 715,
    // and finishes 40 ms later, at 755, before replay forms m at 820.
    // m+1 polls at 825 and sees the copy; A is eligible two decisions later,
    // in m+3 at 850. Starting the copy at pass formation would end it at 860
    // and miss this admission. Its first admission must retain the G2 reuse
    // in the replay report, even though the copy ended before m formed.
    let five_blocks_in_40_ms = 320.0 / (40.0 * 1e6);
    let mut config = launch(engine(20), [5.0; 3], 0.0);
    config.enable_prefix_caching = true;
    config.kv_cache_bytes_per_token = Some(16);
    config.native_host_offload = Some(
        NativeHostOffloadConfig::new(64)
            .with_bandwidths(0.0, five_blocks_in_40_ms)
            .with_connector(G2Connector::SimpleCpuV032),
    );
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..22).collect(), 1, 0.0),
            tokens_request(2, (500..576).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 20, 600.0),
            tokens_request(4, (700..708).collect(), 4, 695.0),
            tokens_request(5, (0..30).collect(), 2, 701.0),
        ],
    );
    assert_eq!(record(&report, 4).first_admit_ms, Some(710.0));
    let restored = record(&report, 5);
    assert_eq!(restored.first_admit_ms, Some(850.0));
    assert_eq!(restored.first_admission_host_reused_input_tokens, Some(20));
    assert_eq!(report.per_request.len(), 5);
    assert!(
        report
            .per_request
            .iter()
            .all(|request| request.output_length == request.requested_output_length)
    );
}

#[test]
fn a_load_awaiting_vllms_poll_does_not_stall_an_engine_with_only_a_blocked_waiter() {
    // Request 1 stores its five prompt blocks in the CPU tier, and request 2
    // fills G1 and evicts them. R (request 3) prefills from 600 and decodes
    // until 750. A (request 4) arrives at 605, joins R's first decode pass,
    // which vLLM decides at 620 and replay forms at 700, and loads request 1's
    // prefix: the copy waits for that pass's GPU work, until 710, and takes
    // 300 ms, so it ends at 1010. B (request 5, 24 tokens) does not fit beside
    // A's reservation in the 10-block G1. Once R finishes nothing runs, and
    // vLLM takes empty steps that poll the connector: it sees A's copy at 1010
    // and admits A then. A finishes at 1120, and B is admitted.
    let five_blocks_in_300_ms = 320.0 / (300.0 * 1e6);
    let report = run_tokens(
        polled_cpu_tier(10, five_blocks_in_300_ms),
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..536).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 6, 600.0),
            tokens_request(4, (0..24).collect(), 2, 605.0),
            tokens_request(5, (2000..2024).collect(), 2, 606.0),
        ],
    );
    let a = record(&report, 4);
    assert_eq!(
        (a.first_admit_ms, a.first_admission_host_reused_input_tokens),
        (Some(1010.0), Some(20))
    );
    assert_eq!(record(&report, 5).first_admit_ms, Some(1120.0));
}

#[test]
fn a_request_sharing_a_load_waits_for_it_until_vllm_has_seen_the_copy() {
    // As above with a 20-block G1, and five blocks copying in 50 ms: A
    // (request 4) loads request 1's prefix from the pass formed at 700, and
    // the copy runs from 710 to 760. C (request 5) shares that prefix and waits
    // for A's load. R's decode passes start every 10 ms, and each polls 8 ms
    // before it starts: it is decided at the previous pass's start and
    // launched in 2 ms. The pass starting at 770 polls at 762, the first poll
    // after the copy, so A is schedulable two decisions later, at 790. C waits
    // until then too and reuses A's blocks from G1 instead of copying the
    // prefix again.
    let five_blocks_in_50_ms = 320.0 / (50.0 * 1e6);
    let report = run_tokens(
        polled_cpu_tier(20, five_blocks_in_50_ms),
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..576).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 12, 600.0),
            tokens_request(4, (0..24).collect(), 2, 605.0),
            tokens_request(5, (0..20).chain(3000..3008).collect(), 2, 606.0),
        ],
    );
    let [a, c] = [4, 5].map(|id| record(&report, id));
    assert_eq!(
        (a.first_admit_ms, a.first_admission_host_reused_input_tokens),
        (Some(790.0), Some(20))
    );
    assert_eq!(
        (
            c.first_admit_ms,
            c.first_admission_reused_input_tokens,
            c.first_admission_host_reused_input_tokens
        ),
        (Some(790.0), 20, Some(0))
    );
}

#[test]
fn work_that_starts_before_an_idle_engine_sees_a_load_runs_before_it() {
    // The sharing run's G1 with five blocks copying in 8 ms, and R (request
    // 3) decoding only three tokens: its last decode runs from 710 to 720. A
    // (request 4) loads request 1's prefix from the pass formed at 700, and
    // the copy runs from 710 to 718. While R's last decode runs, vLLM takes
    // an empty step that polls at 710, before the copy ends, then waits for
    // R. With nothing else to run, it then steps empty and sees the copy:
    // replay admits A at 720. If N (request 6) arrives at 715, vLLM schedules
    // N at 720 instead; N's prefill launch polls at 740 and sees the copy, so
    // A is schedulable two decisions later, at 820, and its pass starts at
    // 830.
    let five_blocks_in_8_ms = 320.0 / (8.0 * 1e6);
    let run = |new_work: bool| {
        let mut requests = vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..576).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 3, 600.0),
            tokens_request(4, (0..24).collect(), 2, 605.0),
        ];
        if new_work {
            requests.push(tokens_request(6, (3000..3024).collect(), 2, 715.0));
        }
        let report = run_tokens(polled_cpu_tier(20, five_blocks_in_8_ms), requests);
        let admitted = |id| record(&report, id).first_admit_ms;
        (admitted(4), new_work.then(|| admitted(6)).flatten())
    };
    assert_eq!(run(false), (Some(720.0), None));
    assert_eq!(run(true), (Some(830.0), Some(720.0)));
}

#[test]
fn an_arrival_during_a_loads_idle_to_busy_launch_joins_the_next_pass() {
    // As above, A's copy ends at 718, after the empty step's poll at 710.
    // The engine sees it only after R finishes at 720. A's prefill must be
    // decided then, so its 20 ms launch ends at 740 and the next decision
    // includes N, which arrives at 735. A prefills from 720 to 820, and N
    // joins its next pass at 820. Inheriting the old 710 decision instead
    // closes the arrival window at 730 and delays N until 830.
    let five_blocks_in_8_ms = 320.0 / (8.0 * 1e6);
    let report = run_tokens(
        polled_cpu_tier(20, five_blocks_in_8_ms),
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..576).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 3, 600.0),
            tokens_request(4, (0..24).collect(), 4, 605.0),
            tokens_request(6, (3000..3012).collect(), 2, 735.0),
        ],
    );
    let restored = record(&report, 4);
    assert_eq!(restored.first_admit_ms, Some(720.0));
    assert_eq!(restored.first_admission_host_reused_input_tokens, Some(20));
    assert_eq!(record(&report, 6).first_admit_ms, Some(820.0));
    assert_eq!(report.per_request.len(), 5);
    assert!(
        report
            .per_request
            .iter()
            .all(|request| request.output_length == request.requested_output_length)
    );
}

#[test]
fn a_restore_blocked_by_the_prefill_budget_keeps_other_loads_on_the_poll_path() {
    // Requests 1 and 2 store 5 and 20 blocks; request 3 evicts them from G1.
    // R (4) prefills at 1600 and issues A's (5) and B's (6) loads with its
    // final decode at 1700. The serial copies start after the decode ends at
    // 1710: A copies until 1760 and B until 1960.
    // P (7) prefills in three 64-token chunks from 1710 to 2010. Its first
    // launch polls at 1760 and sees A, making A runnable two decisions later,
    // at 1910. P consumes the whole token budget then, so A stays in
    // connector_waiting. That pass polls at 1860, before B's copy ends.
    // When P finishes, A keeps the engine busy: B must still wait for A's
    // prefill poll at 2030 and two more decisions, until 2120. Ignoring A in
    // the idle check would let B join A's prefill at 2010.
    let five_blocks_in_50_ms = 320.0 / (50.0 * 1e6);
    let mut config = polled_cpu_tier(128, five_blocks_in_50_ms);
    config.native_host_offload.as_mut().unwrap().num_host_blocks = 512;
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (100..180).collect(), 1, 300.0),
            tokens_request(3, (500..1008).collect(), 1, 600.0),
            tokens_request(4, (2000..2004).collect(), 2, 1600.0),
            tokens_request(5, (0..24).collect(), 2, 1605.0),
            tokens_request(6, (100..184).collect(), 2, 1606.0),
            tokens_request(7, (3000..3192).collect(), 1, 1705.0),
        ],
    );
    assert_eq!(record(&report, 7).first_admit_ms, Some(1710.0));
    assert_eq!(record(&report, 5).first_admit_ms, Some(2010.0));
    assert_eq!(record(&report, 6).first_admit_ms, Some(2120.0));
    assert_eq!(
        record(&report, 5).first_admission_host_reused_input_tokens,
        Some(20)
    );
    assert_eq!(
        record(&report, 6).first_admission_host_reused_input_tokens,
        Some(80)
    );
    assert_eq!(report.per_request.len(), 7);
    assert!(
        report
            .per_request
            .iter()
            .all(|request| request.output_length == request.requested_output_length)
    );
}

#[test]
fn a_queued_empty_decision_does_not_admit_a_restore_before_its_poll() {
    // Requests 1 and 2 store distinct prefixes and request 3 evicts them.
    // R (9) prefills at 1000 and decodes until 1160. A (4) and B (5) issue
    // their loads with the decode passes starting at 1130 and 1140. Their
    // copies end at 1143 and 1153. The decode poll at 1145 sees A, making it
    // eligible two decisions later. The queued empty decision polls at 1150
    // and misses B; A then prefills from 1160 to 1240. A's poll at 1210 sees
    // B, which must wait two decisions until 1250. Settling loads as idle
    // before the queued decision instead admits both at 1160.
    let five_blocks_in_3_ms = 320.0 / (3.0 * 1e6);
    let mut config = polled_cpu_tier(16, five_blocks_in_3_ms);
    config.launch = Some(PassLaunchConfig {
        decode_ms: 5.0,
        piecewise_ms: 50.0,
        eager_ms: 150.0,
        max_cudagraph_capture_size: 4096,
        eager_queue_fraction: 0.4,
    });
    config.timing_model = TimingModelConfig::Fixed {
        prefill_ms: 80.0,
        decode_ms: 10.0,
    };
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..22).collect(), 1, 0.0),
            tokens_request(2, (100..122).collect(), 1, 300.0),
            tokens_request(3, (500..560).collect(), 1, 600.0),
            tokens_request(9, (900..906).collect(), 9, 1000.0),
            tokens_request(4, (0..22).collect(), 2, 1116.0),
            tokens_request(5, (100..122).collect(), 2, 1126.0),
        ],
    );
    assert_eq!(record(&report, 4).first_admit_ms, Some(1160.0));
    assert_eq!(record(&report, 5).first_admit_ms, Some(1250.0));
    for id in [4, 5] {
        assert_eq!(
            record(&report, id).first_admission_host_reused_input_tokens,
            Some(20)
        );
    }
    assert_eq!(report.per_request.len(), 6);
    assert!(
        report
            .per_request
            .iter()
            .all(|request| request.output_length == request.requested_output_length)
    );
}

#[test]
fn simple_cpu_copies_from_the_launch_end_and_offloading_from_the_pass_end() {
    // The sharing run's G1 with five blocks copying in 50 ms; R (request 3)
    // decodes twelve tokens. A (request 4) arrives with R, so the pass formed
    // at 600 prefills R and issues A's load: its launch ends at 620 and its
    // GPU work at 700. SimpleCPU's copy thread starts at the launch end, so
    // the copy runs from 620 to 670, inside that pass's GPU work and before
    // replay forms the next pass at 700. That pass polls at 622 and misses
    // it; the next, decided at 700, polls at 702 and sees it, so A is
    // schedulable two decisions later, in the pass decided at 720 that
    // starts at 730. The offloading connector waits for the GPU work: its
    // copy runs from 700 to 750, the pass decided at 750 polls at 752 and
    // sees it, and A's pass starts at 780. Both reuse A's 20 host-tier
    // tokens.
    let five_blocks_in_50_ms = 320.0 / (50.0 * 1e6);
    // A queued load retains its queue time when pass pricing resolves the
    // copy end. Offloading prepares R's store before reporting the queue;
    // SimpleCPU prepares R's store when the prefill has finished.
    for (connector, admission_ms, expected_events) in [
        (
            G2Connector::SimpleCpuV032,
            730.0,
            [(600.0, Some(670.0)), (700.0, None)],
        ),
        (
            G2Connector::OffloadingV032,
            780.0,
            [(600.0, None), (600.0, Some(750.0))],
        ),
    ] {
        let mut config = polled_cpu_tier(20, five_blocks_in_50_ms);
        config.native_host_offload = config
            .native_host_offload
            .map(|host| host.with_connector(connector));
        let (report, artifacts) = replay_requests(
            ReplayEngineConfig {
                rank: config,
                ..Default::default()
            },
            [
                tokens_request(1, (0..20).collect(), 1, 0.0),
                tokens_request(2, (500..576).collect(), 1, 300.0),
                tokens_request(3, (900..904).collect(), 12, 600.0),
                tokens_request(4, (0..24).collect(), 2, 600.0),
            ]
            .into(),
        )
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
        let a = record(&report, 4);
        assert_eq!(a.first_admit_ms, Some(admission_ms), "{connector:?}");
        assert_eq!(a.first_admission_host_reused_input_tokens, Some(20));
        let artifacts = serde_json::to_value(artifacts).unwrap();
        let restoring = Uuid::from_u128(4).to_string();
        let running = Uuid::from_u128(3).to_string();
        let events: Vec<_> = artifacts["host_offload_events"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|record| {
                let at = record["observed_at_ms"].as_f64().unwrap();
                let event = &record["event"];
                match event["event"].as_str().unwrap() {
                    "load_queued" if record["request_id"] == restoring => {
                        Some((at, Some(event["completes_at_ms"].as_f64().unwrap())))
                    }
                    "store_prepared" if record["request_id"] == running && at <= 700.0 => {
                        Some((at, None))
                    }
                    _ => None,
                }
            })
            .collect();
        assert_eq!(events, expected_events, "{connector:?}");
    }
}

#[test]
fn a_simple_cpu_preemption_polls_after_the_transfers_it_synchronizes() {
    // A 10-block G1 with SimpleCPU copying five blocks in 80 ms. Requests 3
    // and 5 decode long enough to preempt each other. A (request 4) loads
    // request 1's prefix from the pass decided at 700, whose launch ends at
    // 702: SimpleCPU copies from 702 to 782. The pass decided at 740 preempts
    // a request, so SimpleCPU synchronizes every transfer in flight before it
    // launches that pass: its forward waits until 782, and its poll, after
    // the synchronization, sees A's copy. A is schedulable two decisions
    // later, in the pass formed at 802. Polling at that pass's launch end
    // without the synchronization, 742, misses the copy until the poll at
    // 812, and A would wait until 822.
    let mut config = polled_cpu_tier(10, 320.0 / (80.0 * 1e6));
    config.native_host_offload = config
        .native_host_offload
        .map(|host| host.with_connector(G2Connector::SimpleCpuV032));
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..536).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 24, 600.0),
            tokens_request(5, (1900..1904).collect(), 24, 600.0),
            tokens_request(4, (0..24).collect(), 2, 625.0),
        ],
    );
    let restored = record(&report, 4);
    assert_eq!(
        (
            restored.first_admit_ms,
            restored.first_admission_host_reused_input_tokens
        ),
        (Some(802.0), Some(20))
    );
    assert!(
        report
            .per_request
            .iter()
            .map(|request| request.readmission_count)
            .sum::<usize>()
            > 0,
        "the run preempts"
    );
    assert!(
        report
            .per_request
            .iter()
            .all(|request| request.output_length == request.requested_output_length)
    );
}

#[test]
fn simple_cpu_preemption_polls_a_restore_completed_between_decision_and_formation() {
    // Request 4 loads five blocks in 44 ms, from 702 to 746. The preempting
    // pass is decided at 740 but forms at 750. Its pre-launch sync waits for
    // the copy, so its 2 ms launch polls at 748 and sees the restore. Two
    // decisions later request 4 runs at 770. Using only the copies still in
    // flight at 750 gives a poll at 742 and delays admission to 780.
    let mut config = polled_cpu_tier(10, 320.0 / (44.0 * 1e6));
    config.native_host_offload = config
        .native_host_offload
        .map(|host| host.with_connector(G2Connector::SimpleCpuV032));
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..536).collect(), 1, 300.0),
            tokens_request(3, (900..901).collect(), 24, 600.0),
            tokens_request(5, (1900..1908).collect(), 24, 600.0),
            tokens_request(4, (0..24).collect(), 2, 625.0),
        ],
    );
    let restored = record(&report, 4);
    assert_eq!(restored.first_admission_host_reused_input_tokens, Some(20));
    assert_eq!(restored.first_admit_ms, Some(770.0));
}

#[test]
fn simple_cpu_preemption_delays_previous_outputs_and_the_next_pass_restore() {
    // Each four-token block is 64 B. D2H takes 7 ms per block and H2D 42 ms.
    // The decode pass decided at 880 preempts and waits for request 3's D2H
    // from 890 to 897. Its 2 ms launch ends at 899, when request 5's final
    // token can be returned and the following pass can be decided.
    // That pass launches until 901 and issues request 6's two-block restore.
    // The copy ends at 985. The poll at 979 misses it, the poll at 989 sees
    // it, and the pass two decisions later admits request 6 at 1017.
    let h2d_gbps = 64.0 / (42.0 * 1e6);
    let mut config = polled_cpu_tier(14, h2d_gbps);
    config.native_host_offload = Some(
        NativeHostOffloadConfig::new(64)
            .with_bandwidths(64.0 / (7.0 * 1e6), h2d_gbps)
            .with_connector(G2Connector::SimpleCpuV032),
    );
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..552).collect(), 1, 300.0),
            tokens_request(3, (900..912).collect(), 26, 600.0),
            tokens_request(5, (1900..1911).collect(), 10, 600.0),
            tokens_request(4, (0..12).collect(), 2, 615.0),
            tokens_request(6, (500..511).collect(), 1, 865.0),
        ],
    );
    let restored = record(&report, 6);
    assert_eq!(restored.first_admission_host_reused_input_tokens, Some(8));
    assert_eq!(
        (record(&report, 5).last_token_ms, restored.first_admit_ms),
        (Some(899.0), Some(1017.0))
    );
}

#[test]
fn offloading_store_flush_precedes_the_connector_poll() {
    let gbps = |ms_per_block: f64| 64.0 / (ms_per_block * 1e6);
    let run = |arrival, h2d_ms| {
        let mut config = polled_cpu_tier(10, gbps(h2d_ms));
        config.native_host_offload = Some(
            NativeHostOffloadConfig::new(64)
                .with_bandwidths(gbps(34.0), gbps(h2d_ms))
                .with_connector(G2Connector::OffloadingV032),
        );
        let report = run_tokens(
            config,
            vec![
                tokens_request(1, (0..20).collect(), 1, 0.0),
                tokens_request(2, (500..536).collect(), 1, 300.0),
                tokens_request(3, (900..901).collect(), 24, 600.0),
                tokens_request(5, (1900..1908).collect(), 24, 600.0),
                tokens_request(4, (0..24).collect(), 2, arrival),
            ],
        );
        let restored = record(&report, 4);
        (
            restored.first_admission_host_reused_input_tokens,
            restored.first_admit_ms,
            restored.last_token_ms,
        )
    };
    // The arrival at 625 is after the 620 nominal launch but before the
    // required D2H fence at 706. It joins the next pass and reuses 20
    // tokens, with its 15 ms H2D ending at 831.
    assert_eq!(run(625.0, 3.0), (Some(20), Some(884.0), Some(1004.0)));
    // At 615 it joins that pass with either launch window. The restore ends
    // at 866. Pass 8 waits for a D2H until 874, then polls. Its two-pass
    // pipeline therefore admits the request in pass 10 at 894.
    assert_eq!(run(615.0, 10.0), (Some(20), Some(894.0), Some(1014.0)));
}

#[test]
fn offloading_preemption_waits_for_its_store_completed_after_the_decision() {
    let gbps = |ms_per_block: f64| 64.0 / (ms_per_block * 1e6);
    let mut config = polled_cpu_tier(9, gbps(6.0));
    config.launch.as_mut().unwrap().decode_ms = 10.0;
    config.native_host_offload = Some(
        NativeHostOffloadConfig::new(64)
            .with_bandwidths(gbps(7.0), gbps(6.0))
            .with_connector(G2Connector::OffloadingV032),
    );
    let report = run_tokens(
        config,
        vec![
            tokens_request(1, (0..1).collect(), 10, 0.0),
            tokens_request(2, (100..115).collect(), 19, 0.0),
            tokens_request(3, (300..305).collect(), 3, 121.0),
        ],
    );
    // Pass 7 is decided at 250, while the preempted request's store is in
    // flight. The store ends at 257, before replay forms the pass at 260.
    // Its 10 ms launch ends at 267 and shifts subsequent decisions by 7 ms.
    assert_eq!(record(&report, 1).last_token_ms, Some(297.0));
    let second = record(&report, 2);
    assert_eq!(
        (second.readmission_count, second.last_token_ms),
        (0, Some(490.0))
    );
}

#[test]
fn simple_cpu_restore_waits_for_the_store_that_held_its_gpu_destination() {
    let gbps = |ms_per_block: f64| 64.0 / (ms_per_block * 1e6);
    let run = |d2h_ms, h2d_ms, arrival| {
        let mut config = polled_cpu_tier(8, gbps(h2d_ms));
        config.native_host_offload = Some(
            NativeHostOffloadConfig::new(128)
                .with_bandwidths(gbps(d2h_ms), gbps(h2d_ms))
                .with_connector(G2Connector::SimpleCpuV032),
        );
        let report = run_tokens(
            config,
            vec![
                tokens_request(1, (0..8).collect(), 1, 0.0),
                tokens_request(2, (500..528).collect(), 1, 300.0),
                tokens_request(3, (1000..1004).collect(), 1, 600.0),
                tokens_request(4, (1700..1704).collect(), 24, 600.0),
                tokens_request(5, (0..10).collect(), 2, arrival),
            ],
        );
        let restored = record(&report, 5);
        assert_eq!(restored.first_admission_host_reused_input_tokens, Some(8));
        (restored.first_admit_ms, restored.first_token_ms)
    };
    // Request 2's seven-block store runs from 400 to 680. Pass 5, decided
    // at 620 and formed at 700, restores request 5 into two of those blocks.
    // Its launch ends at 640, but the copy cannot start until 680 and ends
    // at 704. The poll at 702 misses it. The following poll at 802 sees it,
    // making the request schedulable two decisions later at 830.
    assert_eq!(run(40.0, 12.0, 605.0), (Some(830.0), Some(940.0)));
    // Here request 2's store ended at 519. An unrelated store ends at 727,
    // but the selected blocks are ready when the launch ends at 722. Their
    // copy ends at 728 and the poll at 732 sees it.
    assert_eq!(run(17.0, 3.0, 710.0), (Some(760.0), Some(870.0)));
}
