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
    /** Populated if dynamic verification is enabled */
    dynamicAudit?: {
        testedOp: string;
        succeeded: boolean;
        actualProducedOps: string[];
        undeclaredOps: string[]; // Emitted in graph, missing from recipe.producedOps
        unobservedOps: string[]; // Listed in recipe.producedOps, not emitted in static test
        error?: string;
    };
}

export type OpCoverageStatus =
    | "native-primitive"       // Directly supported by target hardware
    | "fully-lowerable-exact"  // Can be transitively lowered purely to native primitives (exact)
    | "fully-lowerable-approx" // Can be transitively lowered to native primitives (uses >=1 approximation)
    | "partial-or-blocked"     // Has a recipe, but its transitive outputs hit a dead-end gap op
    | "unhandled-gap";         // Not native and has no recipe at all

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