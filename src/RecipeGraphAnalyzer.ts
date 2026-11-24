import {
    initializeSchemaRegistry,
    OpRegistry,
    OpCategory,
    Recipes,
    getRecipeMetadata,
    type BaseRecipe,
    type OpSchema,
    RecipeShapeSupport,
} from "@specs-feup/onnx-flow";
import type {
    AnalyzedRecipe,
    DirectDecompositionDetail,
    HardwareProfile,
    LibraryAnalysisReport,
    OpCoverageStatus,
    OperatorAnalysis,
    OpQuantityEntry,
    OpRole,
    PerOperatorAtlasEntry,
    RankBreakdown,
    ReachableTerminalSet,
    ScalarReadinessStatus,
    TensorRankBucket,
    UnimplementedSummary,
} from "./types.js";

const OPSET_VERSION = 19;
const MAX_TERMINAL_SETS = 12;

const CONTROL_OPS = new Set(["Loop", "If", "Scan"]);

const MEMORY_OPS = new Set([
    "Gather", "ScatterElements", "ScatterND", "GatherND", "GatherElements",
]);

const VIEW_OPS = new Set([
    "Reshape", "Squeeze", "Unsqueeze", "Identity",
    "Constant", "ConstantOfShape", "Shape", "Size",
]);

/** Ops in OpCategory.Reduction that are explicitly excluded by LowerReduction / LowerDynamicReduction */
const NON_LOOP_REDUCTION_OPS = new Set(["ArgMax", "ArgMin", "TopK"]);

const SHAPE_SUPPORT_RANK: Record<RecipeShapeSupport, number> = {
    static: 1,
    "dynamic-batch": 2,
    dynamic: 3,
};

export class RecipeGraphAnalyzer {
    private categoryToOps = new Map<OpCategory, string[]>();
    private allSchemaOps = new Map<string, OpSchema | undefined>();

    constructor() {
        initializeSchemaRegistry();
        this.indexSchemas();
    }

    /**
     * Reads all registered ONNX operator names from OpRegistry without needing `as unknown`.
     */
    private getRegisteredSchemaNames(): string[] {
        const rawSchemas = Reflect.get(OpRegistry.getInstance(), "schemas");
        if (rawSchemas instanceof Map) {
            return Array.from(rawSchemas.keys()).filter(
                (k): k is string => typeof k === "string",
            );
        }
        return [];
    }

    /**
     * Extracts all registered ONNX operators from OpRegistry at opset 19.
     */
    private indexSchemas(): void {
        const registry = OpRegistry.getInstance();
        const schemaKeys = this.getRegisteredSchemaNames();

        // Only fall back to a static list if OpRegistry is somehow empty
        const opNames =
            schemaKeys.length > 0
                ? schemaKeys
                : [
                      "Add", "Sub", "Mul", "Div", "Mod", "Pow", "Abs", "Neg", "Sign", "Exp", "Log", "Sqrt",
                      "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh",
                      "Erf", "Ceil", "Floor", "Round", "Clip", "Relu", "LeakyRelu", "Elu", "Celu", "Selu",
                      "Sigmoid", "HardSigmoid", "Softplus", "Softsign", "Mish", "Softmax",
                      "And", "Or", "Xor", "Not", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
                      "Max", "Min", "Where", "IsNaN", "IsInf",
                      "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
                      "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
                      "MatMul", "Gemm", "Conv", "AveragePool", "MaxPool",
                      "BatchNormalization", "LayerNormalization", "InstanceNormalization", "GroupNormalization",
                      "QuantizeLinear", "DequantizeLinear",
                      "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Concat", "Split", "Slice", "Pad", "Expand",
                      "Gather", "ScatterElements", "Range", "Shape", "Size", "Cast", "Identity",
                      "Constant", "ConstantOfShape", "Loop", "If", "Scan",
                  ];

        for (const opName of opNames) {
            const schema = registry.get(opName, OPSET_VERSION);
            this.allSchemaOps.set(opName, schema);
            if (schema) {
                const list = this.categoryToOps.get(schema.category) ?? [];
                list.push(opName);
                this.categoryToOps.set(schema.category, list);
            }
        }

        for (const list of this.categoryToOps.values()) {
            list.sort();
        }
    }

    /**
     * Instantiates and normalizes all recipes from ONNX-Flow's catalog.
     */
    public loadRecipes(): AnalyzedRecipe[] {
        const instances: { recipe: BaseRecipe; explicitPhase: "canonicalization" | "loop-lowering" }[] = [];

        const canonRegistry: Record<string, new () => BaseRecipe> = Recipes.canonicalization;
        const loopRegistry: Record<string, new () => BaseRecipe> = Recipes.loopLowering;

        for (const RecipeClass of Object.values(canonRegistry)) {
            instances.push({ recipe: new RecipeClass(), explicitPhase: "canonicalization" });
        }
        for (const RecipeClass of Object.values(loopRegistry)) {
            instances.push({ recipe: new RecipeClass(), explicitPhase: "loop-lowering" });
        }

        const allCategories = new Set<string>(Object.values(OpCategory));

        return instances.map(({ recipe, explicitPhase }) => {
            const meta = getRecipeMetadata(recipe);
            const rawTarget = meta.targetOp;

            let targetOps = allCategories.has(String(rawTarget))
                ? (this.categoryToOps.get(rawTarget as OpCategory) ?? [String(rawTarget)])
                : [String(rawTarget)];

            if (rawTarget === OpCategory.Reduction) {
                targetOps = targetOps.filter((op) => !NON_LOOP_REDUCTION_OPS.has(op));
            }

            return {
                name: meta.name,
                rawTarget,
                targetOps,
                phase: meta.phase ?? explicitPhase,
                precision: meta.precision,
                priority: meta.priority,
                shapeSupport: meta.shapeSupport,
                tags: meta.tags,
                exposesControlFlow: meta.exposesControlFlow,
                exposesDataAccess: meta.exposesDataAccess,
                declaredProducedOps: meta.producedOps,
            };
        });
    }

    /**
     * Returns all standard ONNX operator names known to OpRegistry that currently
     * have zero direct decomposition recipes.
     */
    public getUnimplementedSummary(
        profile: HardwareProfile,
        recipes: AnalyzedRecipe[],
    ): UnimplementedSummary {
        const coveredOps = new Set<string>();
        for (const r of recipes) {
            r.targetOps.forEach((op) => coveredOps.add(op));
        }

        const registry = OpRegistry.getInstance();
        const schemaKeys = this.getRegisteredSchemaNames();
        const byCategory: Record<string, string[]> = {};
        let unimplementedCount = 0;

        for (const opName of schemaKeys) {
            if (!coveredOps.has(opName) && !profile.nativePrimitives.has(opName)) {
                unimplementedCount++;
                const schema = registry.get(opName, OPSET_VERSION);
                const catName = schema?.category ?? "Uncategorized";
                if (!byCategory[catName]) {
                    byCategory[catName] = [];
                }
                byCategory[catName].push(opName);
            }
        }

        for (const cat of Object.keys(byCategory)) {
            byCategory[cat].sort();
        }

        return {
            totalStandardOps: schemaKeys.length,
            unimplementedCount,
            byCategory,
        };
    }

    /**
     * Runs the full static, transitive, cycle, and gap analysis for a given hardware profile.
     */
    public analyze(profile: HardwareProfile, customRecipes?: AnalyzedRecipe[]): LibraryAnalysisReport {
        const recipes = customRecipes ?? this.loadRecipes();

        // 1. Collect the universe of operators
        const allOps = new Set<string>(this.allSchemaOps.keys());
        const directRecipesByOp = new Map<string, AnalyzedRecipe[]>();
        const producedByMap = new Map<string, Set<string>>();

        for (const r of recipes) {
            for (const targetOp of r.targetOps) {
                allOps.add(targetOp);
                const list = directRecipesByOp.get(targetOp) ?? [];
                list.push(r);
                directRecipesByOp.set(targetOp, list);
            }
            for (const prodOp of r.declaredProducedOps) {
                allOps.add(prodOp);
                const producers = producedByMap.get(prodOp) ?? new Set<string>();
                producers.add(r.name);
                producedByMap.set(prodOp, producers);
            }
        }
        for (const list of directRecipesByOp.values()) {
            list.sort((a, b) => {
                if (a.phase !== b.phase) {
                    return a.phase === "canonicalization" ? -1 : 1;
                }
                if (b.priority !== a.priority) return b.priority - a.priority;
                return a.name.localeCompare(b.name);
            });
        }
        for (const prim of profile.nativePrimitives) {
            allOps.add(prim);
        }

        // 2. Detect Cycles in Rewrite Rules (ignoring self-loops like Loop -> Loop or Add -> Add in LoopBody)
        const cycles = this.detectRewriteCycles(allOps, directRecipesByOp);

        // 3. Compute Transitive Lowering Closure per Operator
        const operators = new Map<string, OperatorAnalysis>();
        const dependentsCount = new Map<string, Set<string>>();

        for (const op of allOps) {
            dependentsCount.set(op, new Set<string>());
        }

        for (const op of Array.from(allOps).sort()) {
            const isNative = profile.nativePrimitives.has(op);
            const direct = directRecipesByOp.get(op) ?? [];

            const visitedOps = new Set<string>();
            const leafOps = new Set<string>();
            let requiresApproximation = false;
            let maxDepth = 0;

            const dfs = (currentOp: string, depth: number, path: Set<string>) => {
                maxDepth = Math.max(maxDepth, depth);

                // If the op is already a native hardware primitive, it doesn't NEED further decomposition
                if (profile.nativePrimitives.has(currentOp)) {
                    leafOps.add(currentOp);
                    return;
                }

                const opRecipes = directRecipesByOp.get(currentOp) ?? [];
                // Prefer canonicalization recipes first if an op has both, otherwise take all available
                const canonicalRecipes = opRecipes.filter((r) => r.phase === "canonicalization");
                const activeRecipes = canonicalRecipes.length > 0 ? canonicalRecipes : opRecipes;

                if (activeRecipes.length === 0) {
                    // Dead end / Leaf gap!
                    leafOps.add(currentOp);
                    return;
                }

                for (const rec of activeRecipes) {
                    if (rec.precision === "approximation") {
                        requiresApproximation = true;
                    }
                    for (const nextOp of rec.declaredProducedOps) {
                        // Track that `op` transitively depends on `nextOp`
                        if (nextOp !== op) {
                            dependentsCount.get(nextOp)?.add(op);
                        }

                        if (nextOp === currentOp && rec.name === "LowerElementWise") {
                            leafOps.add(nextOp);
                            continue;
                        }

                        if (path.has(nextOp)) {
                            // Cycle guard
                            continue;
                        }

                        if (!visitedOps.has(`${rec.name}->${nextOp}`)) {
                            visitedOps.add(`${rec.name}->${nextOp}`);
                            const nextPath = new Set(path);
                            nextPath.add(nextOp);
                            dfs(nextOp, depth + 1, nextPath);
                        }
                    }
                }
            };

            dfs(op, 0, new Set([op]));

            const transitiveLeafOps = Array.from(leafOps).sort();
            const blockingLeafGaps = transitiveLeafOps.filter(
                (leaf) => !profile.nativePrimitives.has(leaf),
            );

            let status: OpCoverageStatus;
            if (isNative) {
                status = "native-primitive";
            } else if (direct.length === 0) {
                status = "unhandled-gap";
            } else if (blockingLeafGaps.length === 0) {
                status = requiresApproximation ? "fully-lowerable-approx" : "fully-lowerable-exact";
            } else {
                status = "partial-or-blocked";
            }

            operators.set(op, {
                opType: op,
                category: this.allSchemaOps.get(op)?.category,
                isNativePrimitive: isNative,
                directRecipes: direct,
                producedByRecipes: Array.from(producedByMap.get(op) ?? []).sort(),
                transitiveDependentCount: 0, // Populated in step 4
                transitiveLeafOps,
                blockingLeafGaps,
                requiresApproximation,
                maxLoweringDepth: maxDepth,
                status,
            });
        }

        // 4. Backfill transitiveDependentCount and prioritize gaps
        for (const [op, analysis] of operators.entries()) {
            const depCount = dependentsCount.get(op)?.size ?? 0;
            operators.set(op, { ...analysis, transitiveDependentCount: depCount });
        }

        // 5. Build Per-Operator Atlas Details and reconcile best-path coverage status
        for (const [op, analysis] of operators.entries()) {
            if (analysis.directRecipes.length > 0) {
                const atlasDetail = this.buildPerOperatorAtlas(
                    op,
                    analysis.category,
                    profile,
                    directRecipesByOp,
                );

                const bestShapeSupport = analysis.directRecipes.reduce<RecipeShapeSupport>(
                    (best, rec) =>
                        SHAPE_SUPPORT_RANK[rec.shapeSupport] > SHAPE_SUPPORT_RANK[best]
                            ? rec.shapeSupport
                            : best,
                    "static",
                );

                let status = analysis.status;
                let blockingLeafGaps = analysis.blockingLeafGaps;
                let requiresApproximation = analysis.requiresApproximation;

                if (!analysis.isNativePrimitive) {
                    if (atlasDetail.scalarReadiness === "full-scalar-exact") {
                        status = "fully-lowerable-exact";
                        blockingLeafGaps = [];
                        requiresApproximation = false;
                    } else if (atlasDetail.scalarReadiness === "full-scalar-approx") {
                        status = "fully-lowerable-approx";
                        blockingLeafGaps = [];
                        requiresApproximation = true;
                    } else if (atlasDetail.bestBlockingGaps.length > 0) {
                        status = "partial-or-blocked";
                        blockingLeafGaps = atlasDetail.bestBlockingGaps;
                    }
                }

                operators.set(op, {
                    ...analysis,
                    bestShapeSupport,
                    status,
                    blockingLeafGaps,
                    requiresApproximation,
                    atlasDetail,
                });
            }
        }

        const prioritizedGaps = Array.from(operators.values())
            .filter((o) => o.status === "unhandled-gap")
            .sort((a, b) => {
                if (b.transitiveDependentCount !== a.transitiveDependentCount) {
                    return b.transitiveDependentCount - a.transitiveDependentCount;
                }
                return b.producedByRecipes.length - a.producedByRecipes.length;
            });

        const unimplementedSummary = this.getUnimplementedSummary(profile, recipes);

        return {
            profile,
            recipes,
            operators,
            cycles,
            prioritizedGaps,
            unimplementedSummary,
        };
    }

    private detectRewriteCycles(
        allOps: Set<string>,
        directRecipesByOp: Map<string, AnalyzedRecipe[]>,
    ): string[][] {
        const cycles: string[][] = [];
        const visited = new Set<string>();
        const onStack = new Set<string>();
        const path: string[] = [];

        const visit = (op: string) => {
            visited.add(op);
            onStack.add(op);
            path.push(op);

            // Only check canonicalization rewrites for algebraic cycles
            // (Loop-lowering naturally places the scalar version of an op inside a Loop body)
            const recipes = (directRecipesByOp.get(op) ?? []).filter(
                (r) => r.phase === "canonicalization",
            );

            for (const rec of recipes) {
                for (const prodOp of rec.declaredProducedOps) {
                    if (prodOp === op) continue;
                    if (!visited.has(prodOp)) {
                        visit(prodOp);
                    } else if (onStack.has(prodOp)) {
                        const cycleStart = path.indexOf(prodOp);
                        if (cycleStart !== -1) {
                            cycles.push([...path.slice(cycleStart), prodOp]);
                        }
                    }
                }
            }

            path.pop();
            onStack.delete(op);
        };

        for (const op of allOps) {
            if (!visited.has(op)) visit(op);
        }
        return cycles;
    }

    private formatCategory(cat?: OpCategory | string): string | undefined {
        if (cat === undefined) return undefined;
        if (typeof cat === "string") return cat;
        const map: Record<string, string> = {
            [OpCategory.ElementWise]: "ElementWise",
            [OpCategory.Reduction]: "Reduction",
            [OpCategory.ControlFlow]: "ControlFlow",
            [OpCategory.DataMovement]: "DataMovement",
        };
        return map[String(cat)] ?? String(cat);
    }

    private buildPerOperatorAtlas(
        rootOp: string,
        category: string | undefined,
        profile: HardwareProfile,
        directRecipesByOp: Map<string, AnalyzedRecipe[]>,
    ): PerOperatorAtlasEntry {
        const directRecipes = directRecipesByOp.get(rootOp) ?? [];
        const directDecompositions = directRecipes.map((rec) =>
            this.buildDirectDecomposition(rootOp, rec),
        );

        const { sets: terminalSets, truncated } = this.computeReachableTerminalSets(
            rootOp,
            profile,
            directRecipesByOp,
        );

        // Pick best terminal set by: (1) full scalar exact, (2) full scalar approx, (3) highest scalar ratio, (4) fewest blocking gaps
        const sortedSets = [...terminalSets].sort((a, b) => {
            if (a.reachesFullScalar !== b.reachesFullScalar) {
                return a.reachesFullScalar ? -1 : 1;
            }
            if (a.reachesFullScalar && b.reachesFullScalar && a.precision !== b.precision) {
                return a.precision === "exact" ? -1 : 1;
            }
            if (a.blockingGaps.length !== b.blockingGaps.length) {
                return a.blockingGaps.length - b.blockingGaps.length;
            }
            if (b.scalarComputeRatio !== a.scalarComputeRatio) {
                return b.scalarComputeRatio - a.scalarComputeRatio;
            }
            return a.totalOps - b.totalOps;
        });

        const best = sortedSets[0];
        let scalarReadiness: ScalarReadinessStatus = "non-scalar-or-blocked";
        if (best) {
            if (best.reachesFullScalar) {
                scalarReadiness =
                    best.precision === "approximation"
                        ? "full-scalar-approx"
                        : "full-scalar-exact";
            } else if (best.scalarComputeRatio === 1 && best.blockingGaps.length > 0) {
                scalarReadiness = "scalar-blocked-non-native";
            } else if (best.scalarComputeRatio > 0) {
                scalarReadiness = "partial-scalar";
            }
        }

        return {
            opType: rootOp,
            category,
            directDecompositions,
            terminalSets: sortedSets.map((s, idx) => ({ ...s, id: `Set ${idx + 1}` })),
            truncatedTerminalSets: truncated,
            scalarReadiness,
            bestScalarComputeRatio: best?.scalarComputeRatio ?? 0,
            bestTerminalSetId: best ? "Set 1" : undefined,
            bestBlockingGaps: best?.blockingGaps ?? [],
            bestResidualNonScalar: best?.residualNonScalarCompute ?? [],
        };
    }

    private buildDirectDecomposition(
        opType: string,
        rec: AnalyzedRecipe,
    ): DirectDecompositionDetail {
        const probe = rec.dynamicAuditsByOp?.[opType] ??
            (rec.dynamicAudit?.testedOp === opType ? rec.dynamicAudit : undefined);

        let emittedNodes: OpQuantityEntry[];
        let provenance: "dynamic-probe" | "static-declared";
        let probeInputSummary: string | undefined;

        if (probe && probe.succeeded && probe.emittedEntries.length > 0) {
            provenance = "dynamic-probe";
            probeInputSummary = probe.probeInputSummary;
            emittedNodes = probe.emittedEntries;
        } else {
            provenance = "static-declared";
            const INDEXING_FALLBACK_OPS = new Set(["Add", "Sub", "Mul", "Div", "Mod"]);

            emittedNodes = rec.declaredProducedOps.map((prodOp) => {
                let role = this.staticClassifyRole(prodOp);

                // In static loop-lowering fallback (e.g. LowerElementWise), arithmetic ops
                // other than the target op itself are loop index/stride math, not compute.
                if (
                    rec.phase === "loop-lowering" &&
                    role === "compute" &&
                    prodOp !== opType &&
                    INDEXING_FALLBACK_OPS.has(prodOp)
                ) {
                    role = "indexing";
                }

                const isLoopBodyScalar =
                    rec.phase === "loop-lowering" &&
                    (role === "compute" || role === "indexing");

                return {
                    opType: prodOp,
                    rank: isLoopBodyScalar
                        ? "0D"
                        : CONTROL_OPS.has(prodOp) || prodOp === "Size"
                            ? "0D"
                            : prodOp === "Shape"
                                ? "1D"
                                : "unknown",
                    role,
                    count: 1,
                    inLoop: isLoopBodyScalar,
                    estimated: true,
                };
            });
        }

        const totalNodes = emittedNodes.reduce((sum, e) => sum + e.count, 0);
        const scalarComputeCount = emittedNodes
            .filter((e) => e.role === "compute" && e.rank === "0D")
            .reduce((sum, e) => sum + e.count, 0);
        const nonScalarComputeCount = emittedNodes
            .filter((e) => e.role === "compute" && e.rank !== "0D")
            .reduce((sum, e) => sum + e.count, 0);

        return {
            recipeName: rec.name,
            phase: rec.phase,
            precision: rec.precision,
            priority: rec.priority,
            shapeSupport: rec.shapeSupport,
            tags: rec.tags,
            exposesControlFlow: rec.exposesControlFlow,
            exposesDataAccess: rec.exposesDataAccess,
            provenance,
            probeInputSummary,
            emittedNodes,
            totalNodes,
            scalarComputeCount,
            nonScalarComputeCount,
        };
    }

    private computeReachableTerminalSets(
        rootOp: string,
        profile: HardwareProfile,
        directRecipesByOp: Map<string, AnalyzedRecipe[]>,
    ): { sets: ReachableTerminalSet[]; truncated: boolean } {
        interface RawMultiset {
            entries: Map<string, OpQuantityEntry>;
            recipes: Set<string>;
            approx: boolean;
        }

        let truncated = false;
        const mergeInto = (target: Map<string, OpQuantityEntry>, entry: OpQuantityEntry, multiplier = 1) => {
            const key = `${entry.opType}|${entry.rank}|${entry.role}|${entry.inLoop}`;
            const prev = target.get(key);
            if (prev) {
                target.set(key, {
                    ...prev,
                    count: prev.count + entry.count * multiplier,
                    estimated: prev.estimated || entry.estimated,
                });
            } else {
                target.set(key, {
                    ...entry,
                    count: entry.count * multiplier,
                });
            }
        };

        const signatureOf = (m: RawMultiset): string => {
            const parts = Array.from(m.entries.values())
                .sort((a, b) =>
                    a.opType.localeCompare(b.opType) ||
                    a.rank.localeCompare(b.rank) ||
                    a.role.localeCompare(b.role) ||
                    Number(a.inLoop) - Number(b.inLoop),
                )
                .map((e) => `${e.opType}:${e.rank}:${e.role}:${e.inLoop ? 1 : 0}=${e.count}`);
            return `${m.approx ? "A" : "E"}|${parts.join(",")}`;
        };

        const expandNode = (
            opType: string,
            rank: TensorRankBucket,
            role: OpRole,
            inLoop: boolean,
            estimated: boolean,
            path: Set<string>,
            isRoot: boolean,
        ): RawMultiset[] => {
            const leafSingleton = (): RawMultiset[] => {
                const map = new Map<string, OpQuantityEntry>();
                mergeInto(map, { opType, rank, role, count: 1, inLoop, estimated });
                return [{ entries: map, recipes: new Set(), approx: false }];
            };

            // Control, memory, view, and indexing ops do not expand further
            if (!isRoot && role !== "compute") {
                return leafSingleton();
            }

            // Already a scalar (0D) native primitive -> terminal!
            if (!isRoot && rank === "0D" && profile.nativePrimitives.has(opType)) {
                return leafSingleton();
            }

            const stateKey = `${opType}@${rank}`;
            if (path.has(stateKey)) {
                return leafSingleton();
            }

            const allRecipes = directRecipesByOp.get(opType) ?? [];
            // Never loop-lower an op that is already scalar (0D)
            const applicableRecipes = allRecipes.filter(
                (r) => !(rank === "0D" && r.phase === "loop-lowering"),
            );

            if (applicableRecipes.length === 0) {
                return leafSingleton();
            }

            const nextPath = new Set(path);
            nextPath.add(stateKey);

            const resultsBySig = new Map<string, RawMultiset>();

            // If a non-root op is already native in this profile (e.g., Vector-DSP native op),
            // allow stopping here as one valid terminal option only when it's not a scalar-cgra non-0D op
            if (!isRoot && profile.nativePrimitives.has(opType) && profile.name !== "scalar-cgra") {
                for (const leaf of leafSingleton()) {
                    resultsBySig.set(signatureOf(leaf), leaf);
                }
            }

            for (const rec of applicableRecipes) {
                const direct = this.buildDirectDecomposition(opType, rec);
                let partials: RawMultiset[] = [
                    {
                        entries: new Map(),
                        recipes: new Set([rec.name]),
                        approx: rec.precision === "approximation",
                    },
                ];

                for (const child of direct.emittedNodes) {
                    const childInLoop = inLoop || child.inLoop;

                    // Rank propagation: canonicalizing a 0D scalar op always produces 0D scalar ops
                    const effectiveChildRank: TensorRankBucket =
                        rank === "0D" && rec.phase === "canonicalization"
                            ? "0D"
                            : child.rank;

                    // Self-reference inside a loop-lowering recipe is the scalarized leaf op
                    const childExpansions =
                        child.opType === opType && rec.phase === "loop-lowering"
                            ? [
                                  (() => {
                                      const m = new Map<string, OpQuantityEntry>();
                                      mergeInto(m, {
                                          ...child,
                                          rank: "0D",
                                          inLoop: true,
                                      });
                                      return { entries: m, recipes: new Set<string>(), approx: false };
                                  })(),
                              ]
                            : expandNode(
                                  child.opType,
                                  effectiveChildRank,
                                  child.role,
                                  childInLoop,
                                  child.estimated,
                                  nextPath,
                                  false,
                              );

                    const nextPartials = new Map<string, RawMultiset>();
                    for (const acc of partials) {
                        for (const exp of childExpansions) {
                            const combinedEntries = new Map(acc.entries);
                            for (const e of exp.entries.values()) {
                                mergeInto(
                                    combinedEntries,
                                    { ...e, inLoop: e.inLoop || childInLoop },
                                    child.count,
                                );
                            }
                            const combined: RawMultiset = {
                                entries: combinedEntries,
                                recipes: new Set([...acc.recipes, ...exp.recipes]),
                                approx: acc.approx || exp.approx,
                            };
                            const sig = signatureOf(combined);
                            if (!nextPartials.has(sig)) {
                                if (nextPartials.size >= MAX_TERMINAL_SETS) {
                                    truncated = true;
                                    continue;
                                }
                                nextPartials.set(sig, combined);
                            }
                        }
                    }
                    partials = Array.from(nextPartials.values());
                }

                for (const p of partials) {
                    const sig = signatureOf(p);
                    if (!resultsBySig.has(sig)) {
                        if (resultsBySig.size >= MAX_TERMINAL_SETS) {
                            truncated = true;
                            continue;
                        }
                        resultsBySig.set(sig, p);
                    }
                }
            }

            return Array.from(resultsBySig.values());
        };

        const rawSets = expandNode(rootOp, "ND", "compute", false, false, new Set(), true);

        const sets: ReachableTerminalSet[] = rawSets.map((raw, idx) => {
            const ops = Array.from(raw.entries.values()).sort((a, b) =>
                a.role.localeCompare(b.role) ||
                a.opType.localeCompare(b.opType) ||
                a.rank.localeCompare(b.rank),
            );

            const computeByRank: Record<TensorRankBucket, number> = {
                "0D": 0,
                "1D": 0,
                "2D": 0,
                "ND": 0,
                unknown: 0,
            };

            let totalOps = 0;
            let indexingScalarCount = 0;
            let controlMemoryViewCount = 0;
            let hasEstimatedCounts = false;
            const blockingSet = new Set<string>();
            const residualNonScalarCompute: OpQuantityEntry[] = [];

            for (const entry of ops) {
                totalOps += entry.count;
                if (entry.estimated) hasEstimatedCounts = true;

                if (entry.role === "compute") {
                    computeByRank[entry.rank] += entry.count;
                    if (entry.rank !== "0D") {
                        residualNonScalarCompute.push(entry);
                    }
                } else if (entry.role === "indexing") {
                    indexingScalarCount += entry.count;
                } else {
                    controlMemoryViewCount += entry.count;
                }

                if (!profile.nativePrimitives.has(entry.opType)) {
                    blockingSet.add(entry.opType);
                }
            }

            const scalarComputeCount = computeByRank["0D"];
            const nonScalarComputeCount =
                computeByRank["1D"] +
                computeByRank["2D"] +
                computeByRank["ND"] +
                computeByRank.unknown;
            const totalCompute = scalarComputeCount + nonScalarComputeCount;
            const scalarComputeRatio =
                totalCompute === 0 ? 1 : scalarComputeCount / totalCompute;

            const blockingGaps = Array.from(blockingSet).sort();
            const reachesFullScalar =
                nonScalarComputeCount === 0 && blockingGaps.length === 0;

            return {
                id: `Set ${idx + 1}`,
                recipeChain: Array.from(raw.recipes).sort(),
                precision: raw.approx ? "approximation" : "exact",
                ops,
                totalOps,
                computeByRank: computeByRank as RankBreakdown,
                scalarComputeCount,
                nonScalarComputeCount,
                indexingScalarCount,
                controlMemoryViewCount,
                scalarComputeRatio,
                reachesFullScalar,
                blockingGaps,
                residualNonScalarCompute,
                hasEstimatedCounts,
            };
        });

        return { sets, truncated };
    }

    private staticClassifyRole(opType: string): OpRole {
        if (CONTROL_OPS.has(opType)) return "control";
        if (MEMORY_OPS.has(opType)) return "memory";
        if (VIEW_OPS.has(opType)) return "view";
        return "compute";
    }
}