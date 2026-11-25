import type { RecipePhase, RecipePrecision, OpCategory, RecipeShapeSupport } from "@specs-feup/onnx-flow";

export interface HardwareProfile {
    readonly name: string;
    readonly description: string;
    /** Operations natively supported by the target backend (need no further decomposition) */
    readonly nativePrimitives: ReadonlySet<string>;
}

export const BUILTIN_PROFILES: Record<string, HardwareProfile> = {
    "scalar-cgra": {
        name: "scalar-cgra",
        description: "Strict Scalar/AGU CGRA (Basic ALU, Logic, Loop control, Gather/Scatter memory)",
        nativePrimitives: new Set([
            // Arithmetic & Logic
            "Add", "Sub", "Mul", "And", "Or", "Xor", "Not",
            "Equal", "Greater", "Less", "Where", "Cast",
            // Bitwise Ops
            "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor",
            // Control Flow & Memory Access
            "Loop", "If", "Gather", "ScatterElements",
            // Zero-cost / Metadata View Ops
            "Reshape", "Squeeze", "Unsqueeze", "Identity", "Constant", "ConstantOfShape", "Shape", "Size",
        ]),
    },
    "vector-dsp": {
        name: "vector-dsp",
        description: "Vector DSP supporting native reductions, slicing, bitwise ALU, and basic math",
        nativePrimitives: new Set([
            "Add", "Sub", "Mul", "Div", "Max", "Min", "Abs", "Neg", "Exp", "Sqrt",
            "And", "Or", "Xor", "Not", "Equal", "Greater", "Less", "Where", "Cast",
            "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor",
            "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
            "Loop", "If", "Gather", "ScatterElements", "Slice", "Concat", "Pad",
            "Reshape", "Squeeze", "Unsqueeze", "Identity", "Constant", "ConstantOfShape", "Shape", "Size",
        ]),
    },
};

/** Dimensionality stratification for ONNX operators */
export type TensorRankBucket = "0D" | "1D" | "2D" | "ND" | "unknown";

/** Separates neural-network compute from loop indexing, control flow, memory, and zero-cost views */
export type OpRole = "compute" | "indexing" | "control" | "memory" | "view";

export interface OpQuantityEntry {
    readonly opType: string;
    readonly rank: TensorRankBucket;
    readonly role: OpRole;
    readonly count: number;
    /** True when emitted inside a Loop / If region (loopDepth > 0) */
    readonly inLoop: boolean;
    /** True when count comes from static declaredProducedOps (lower-bound estimate of 1) */
    readonly estimated: boolean;
}

export interface DynamicProbeResult {
    readonly testedOp: string;
    readonly probeInputSummary: string;
    readonly succeeded: boolean;
    readonly actualProducedOps: string[];
    readonly emittedEntries: OpQuantityEntry[];
    readonly undeclaredOps: string[]; // Emitted in graph, missing from recipe.producedOps
    readonly unobservedOps: string[]; // Listed in recipe.producedOps, not emitted in static test
    readonly error?: string;
}

export interface AnalyzedRecipe {
    readonly name: string;
    readonly rawTarget: string | OpCategory;
    /** Concrete ONNX operator names this recipe can match */
    readonly targetOps: string[];
    readonly phase: RecipePhase;
    readonly precision: RecipePrecision;
    readonly priority: number;
    readonly shapeSupport: RecipeShapeSupport;
    readonly tags: readonly string[];
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly declaredProducedOps: string[];
    /** Backwards-compatible primary probe result (targetOps[0]) */
    dynamicAudit?: DynamicProbeResult;
    /** Per-operator probe results keyed by opType (essential for category-wide recipes) */
    dynamicAuditsByOp?: Record<string, DynamicProbeResult>;
}

export type OpCoverageStatus =
    | "native-primitive"       // Directly supported by target hardware
    | "fully-lowerable-exact"  // Can be transitively lowered purely to native primitives (exact)
    | "fully-lowerable-approx" // Can be transitively lowered to native primitives (uses >=1 approximation)
    | "partial-or-blocked"     // Has a recipe, but its transitive outputs hit a dead-end gap op
    | "unhandled-gap";         // Not native and has no recipe at all

export interface DirectDecompositionDetail {
    readonly recipeName: string;
    readonly phase: RecipePhase;
    readonly precision: RecipePrecision;
    readonly priority: number;
    readonly shapeSupport: RecipeShapeSupport;
    readonly tags: readonly string[];
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly provenance: "dynamic-probe" | "static-declared";
    readonly probeInputSummary?: string;
    readonly emittedNodes: OpQuantityEntry[];
    readonly totalNodes: number;
    readonly scalarComputeCount: number;
    readonly nonScalarComputeCount: number;
}

export interface RankBreakdown {
    readonly "0D": number;
    readonly "1D": number;
    readonly "2D": number;
    readonly "ND": number;
    readonly unknown: number;
}

export interface ReachableTerminalSet {
    readonly id: string;
    readonly recipeChain: string[];
    readonly precision: RecipePrecision;
    readonly ops: OpQuantityEntry[];
    readonly totalOps: number;
    readonly computeByRank: RankBreakdown;
    readonly scalarComputeCount: number;
    readonly nonScalarComputeCount: number;
    readonly indexingScalarCount: number;
    readonly controlMemoryViewCount: number;
    /** Ratio of 0D compute ops to total compute ops (1.0 if no compute ops remain) */
    readonly scalarComputeRatio: number;
    /** True if all remaining compute ops are 0D and no non-native blocking gaps remain */
    readonly reachesFullScalar: boolean;
    readonly blockingGaps: string[];
    readonly residualNonScalarCompute: OpQuantityEntry[];
    readonly hasEstimatedCounts: boolean;
}

export type ScalarReadinessStatus =
    | "full-scalar-exact"
    | "full-scalar-approx"
    | "scalar-blocked-non-native"
    | "partial-scalar-native"
    | "native-vector-ready"
    | "partial-scalar"
    | "non-scalar-or-blocked";

export interface PerOperatorAtlasEntry {
    readonly opType: string;
    readonly category?: string;
    readonly directDecompositions: DirectDecompositionDetail[];
    readonly terminalSets: ReachableTerminalSet[];
    readonly truncatedTerminalSets: boolean;
    readonly scalarReadiness: ScalarReadinessStatus;
    readonly bestScalarComputeRatio: number;
    readonly bestTerminalSetId?: string;
    readonly bestBlockingGaps: string[];
    readonly bestResidualNonScalar: OpQuantityEntry[];
}

export interface OperatorAnalysis {
    readonly opType: string;
    readonly category?: string;
    readonly isNativePrimitive: boolean;
    readonly directRecipes: AnalyzedRecipe[];
    /** Highest shape generality across direct recipes ("dynamic" > "dynamic-batch" > "static") */
    readonly bestShapeSupport?: RecipeShapeSupport;
    readonly producedByRecipes: string[];
    readonly transitiveDependentCount: number;
    readonly transitiveLeafOps: string[];
    readonly blockingLeafGaps: string[];
    readonly requiresApproximation: boolean;
    readonly maxLoweringDepth: number;
    readonly status: OpCoverageStatus;
    readonly atlasDetail?: PerOperatorAtlasEntry;
    readonly hierarchyMetrics?: OperatorHierarchyMetrics;
}

export interface UnimplementedSummary {
    readonly totalStandardOps: number;
    readonly unimplementedCount: number;
    readonly byCategory: Record<string, string[]>;
}

export interface FusedLoweringMetrics {
    readonly opType: string;
    readonly succeeded: boolean;
    readonly loopCount: number;
    readonly scalarComputeCount: number;
    readonly nonScalarComputeCount: number;
    readonly indexingScalarCount: number;
    readonly controlMemoryViewCount: number;
    readonly totalOps: number;
    readonly error?: string;
}

export interface LibraryAnalysisReport {
    readonly profile: HardwareProfile;
    readonly recipes: AnalyzedRecipe[];
    readonly operators: Map<string, OperatorAnalysis>;
    readonly cycles: string[][];
    /** Non-native ops with 0 recipes, ranked by how many other recipes they block */
    readonly prioritizedGaps: OperatorAnalysis[];
    /** Standard ONNX operators from OpRegistry with no recipes and not native */
    readonly unimplementedSummary: UnimplementedSummary;
    readonly hierarchy: DynamicHierarchyReport;
    readonly fusedBenchmarks?: FusedLoweringMetrics[];
}

export interface OperatorHierarchyMetrics {
    /** Emergent stratum: 0 = native primitive, 1..H = lowering levels, -1 = unhandled gap */
    readonly level: number;
    /** Shortest lowering height across all applicable recipes */
    readonly minLevel: number;
    /** Longest lowering height across all applicable recipes */
    readonly maxLevel: number;
    /** Highest-priority recipe used to determine the primary lowering path */
    readonly primaryRecipe?: string;
    readonly sccId: number;
    /** Number of ops that directly emit this op on their primary lowering path */
    readonly directFanIn: number;
    /** Number of higher-level ops that transitively lower through this op */
    readonly transitiveFanIn: number;
    /** Number of distinct lower ops emitted by this op's primary recipe */
    readonly directFanOut: number;
    /** True if the primary recipe crosses into loop-lowering or exposes control flow */
    readonly crossesIntoLoop: boolean;
}

export interface AbstractionLevelSummary {
    readonly level: number;
    readonly operators: string[];
    readonly dominantCategories: { category: string; count: number; ratio: number }[];
    readonly canonicalizationCount: number;
    readonly loopLoweringCount: number;
    readonly controlFlowBoundaryCount: number;
    readonly approximationCount: number;
    readonly avgFanIn: number;
    /** Auto-derived descriptor summarizing the stratum's emergent role */
    readonly signature: string;
}

export interface DialectHubEntry {
    readonly opType: string;
    readonly level: number;
    readonly category?: string;
    readonly directFanIn: number;
    readonly transitiveFanIn: number;
    readonly directFanOut: number;
    readonly primaryPhase?: RecipePhase;
}

export interface LevelTransitionEdge {
    readonly fromLevel: number;
    readonly toLevel: number;
    readonly count: number;
    /** True when the edge steps exactly one stratum down (fromLevel - toLevel === 1) */
    readonly isProgressive: boolean;
    readonly sampleOps: string[];
}

export interface DynamicHierarchyReport {
    readonly maxLevel: number;
    /** Emergent strata ordered top-down from maxLevel to 0 */
    readonly levels: AbstractionLevelSummary[];
    /** Non-native ops with 0 recipes referenced by the library */
    readonly unresolvedGaps: string[];
    /** Intermediate ops (0 < level < maxLevel) ranked by cross-level fan-in */
    readonly convergenceHubs: DialectHubEntry[];
    readonly transitions: LevelTransitionEdge[];
    /** Fraction of inter-level edges that step strictly k -> k-1 */
    readonly progressiveRatio: number;
    /** Levels where ops cross from tensor-level rewrites into Loop/control-flow lowering */
    readonly loopBoundaryLevels: number[];
    /** Non-trivial strongly connected components (size > 1), if any */
    readonly sccCycles: string[][];
}