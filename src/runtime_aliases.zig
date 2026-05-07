//! Shared re-exports of the `runtime.zig` and `runtime_hybrid.zig`
//! symbols used by the per-command and per-smoke modules. Replaces the
//! per-file `const X = runtime.X;` blocks that used to live at the top
//! of every consumer.
//!
//! Naming convention: dense-family aliases keep their runtime name
//! (`ChatKernels`, `recordForwardStep`, `MatmulPush`, ...). Hybrid-family
//! aliases get the `Hybrid`/`hybrid` prefix to disambiguate from the
//! dense versions, since `runtime_hybrid.zig` re-uses the same internal
//! names (`ChatKernels`, `recordForwardStep`, ...).

const runtime = @import("runtime.zig");
const runtime_hybrid = @import("runtime_hybrid.zig");

// ── runtime.zig ─────────────────────────────────────────────────────

pub const ChatKernels = runtime.ChatKernels;
pub const Tq4VHooks = runtime.Tq4VHooks;
pub const Tq4PackPush = runtime.Tq4PackPush;
pub const ForwardPushes = runtime.ForwardPushes;
pub const EmbedLookupPush = runtime.EmbedLookupPush;
pub const AddInPlacePush = runtime.AddInPlacePush;
pub const AttnScoresPush = runtime.AttnScoresPush;
pub const AttnOutputPush = runtime.AttnOutputPush;
pub const KvWritePush = runtime.KvWritePush;
pub const RmsnormPush = runtime.RmsnormPush;
pub const MatmulPush = runtime.MatmulPush;
pub const RopePush = runtime.RopePush;
pub const RopePartialPush = runtime.RopePartialPush;
pub const SoftmaxPush = runtime.SoftmaxPush;
pub const GegluPush = runtime.GegluPush;
pub const ReluPush = runtime.ReluPush;
pub const ReluBackwardPush = runtime.ReluBackwardPush;
pub const SgdStepPush = runtime.SgdStepPush;
pub const OuterProductPush = runtime.OuterProductPush;
pub const LinearBackwardDxPush = runtime.LinearBackwardDxPush;
pub const MseLossGradPush = runtime.MseLossGradPush;

pub const computeForwardPushes = runtime.computeForwardPushes;
pub const recordOneLayer = runtime.recordOneLayer;
pub const recordForwardStep = runtime.recordForwardStep;
pub const recDispatch1D = runtime.recDispatch1D;
pub const recDispatchPerRow = runtime.recDispatchPerRow;
pub const recDispatchMatmul = runtime.recDispatchMatmul;
pub const recDispatchRope = runtime.recDispatchRope;

// ── runtime_hybrid.zig (prefixed `Hybrid`/`hybrid` to disambiguate) ──

pub const HybridChatKernels = runtime_hybrid.ChatKernels;
pub const HybridChatScratch = runtime_hybrid.Scratch;
pub const HybridChatState = runtime_hybrid.State;
pub const HybridTq4VHooks = runtime_hybrid.Tq4VHooks;
pub const HybridForwardPushes = runtime_hybrid.ForwardPushes;

pub const ScalePush = runtime_hybrid.ScalePush;
pub const SliceCopyPush = runtime_hybrid.SliceCopyPush;
pub const SigmoidMulPush = runtime_hybrid.SigmoidMulPush;
pub const SplitQGatePush = runtime_hybrid.SplitQGatePush;
pub const L2normPush = runtime_hybrid.L2normPush;
pub const Conv1dUpdatePush = runtime_hybrid.Conv1dUpdatePush;
pub const RmsnormGatedPush = runtime_hybrid.RmsnormGatedPush;
pub const GatedDeltaStepPush = runtime_hybrid.GatedDeltaStepPush;

pub const computeHybridForwardPushes = runtime_hybrid.computeForwardPushes;
pub const recordOneHybridLayer = runtime_hybrid.recordOneLayer;
pub const recordHybridForwardStep = runtime_hybrid.recordForwardStep;
