import type { RecipePhase, RecipePrecision, OpCategory } from "@specs-feup/onnx-flow";

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
            // Control Flow & Memory Access
            "Loop", "If", "Gather", "ScatterElements",
            // Zero-cost / Metadata View Ops
            "Reshape", "Squeeze", "Unsqueeze", "Identity", "Constant", "ConstantOfShape",
        ]),
    },
    "vector-dsp": {
        name: "vector-dsp",
        description: "Vector DSP supporting native reductions, slicing, and basic math",
        nativePrimitives: new Set([
            "Add", "Sub", "Mul", "Div", "Max", "Min", "Abs", "Neg", "Exp", "Sqrt",
            "And", "Or", "Not", "Equal", "Greater", "Less", "Where", "Cast",
            "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
            "Loop", "Gather", "ScatterElements", "Slice", "Concat", "Pad",
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
    /** Operators that directly produce this op during their decomposition */
    readonly producedByRecipes: string[];
    /** How many recipes (directly or transitively) depend on this operator */
    readonly transitiveDependentCount: number;
    /** All leaf operators reached when decomposing this operator as far as possible */
    readonly transitiveLeafOps: string[];
    /** Subset of transitiveLeafOps that are NOT in HardwareProfile.nativePrimitives */
    readonly blockingLeafGaps: string[];
    /** Whether any path in the lowering tree uses an approximation recipe */
    readonly requiresApproximation: boolean;
    /** Maximum lowering depth to reach leaf operations */
    readonly maxLoweringDepth: number;
    readonly status: OpCoverageStatus;
    /** Populated for all operators that have at least one direct decomposition recipe */
    readonly atlasDetail?: PerOperatorAtlasEntry;
}

export interface UnimplementedSummary {
    readonly totalStandardOps: number;
    readonly unimplementedCount: number;
    readonly byCategory: Record<string, string[]>;
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
}