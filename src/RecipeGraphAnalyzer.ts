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
    AbstractionLevelSummary,
    AnalyzedRecipe,
    ApproximationClass,
    ApproximationRootEntry,
    DialectHubEntry,
    DirectDecompositionDetail,
    DynamicHierarchyReport,
    HardwareProfile,
    LevelTransitionEdge,
    LibraryAnalysisReport,
    OpCoverageStatus,
    OperatorAnalysis,
    OperatorHierarchyMetrics,
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

const CONDITIONAL_FASTPATH_TAGS = new Set([
    "exact-power-of-two",
    "constant-divisor",
    "small-int",
    "conv-1x1",
    "noop",
]);
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
                if (b.priority !== a.priority) return b.priority - a.priority;
                if (a.phase !== b.phase) {
                    return a.phase === "canonicalization" ? -1 : 1;
                }
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

                const isConditionalRecipe = (recName: string): boolean => {
                    const rec = recipes.find((r) => r.name === recName);
                    return rec?.tags.some((t) => CONDITIONAL_FASTPATH_TAGS.has(t)) ?? false;
                };

                const generalExactSet = atlasDetail.terminalSets.find(
                    (s) =>
                        s.reachesFullScalar &&
                        s.precision === "exact" &&
                        !s.recipeChain.some((rn) => isConditionalRecipe(rn)),
                );
                const fastPathOnlyExact =
                    status === "fully-lowerable-exact" &&
                    generalExactSet === undefined &&
                    atlasDetail.terminalSets.some((s) => s.reachesFullScalar);

                operators.set(op, {
                    ...analysis,
                    bestShapeSupport,
                    status,
                    fastPathOnlyExact,
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

        // 6. Compute fully dynamic, recipe-driven abstraction hierarchy
        const hierarchy = this.computeDynamicHierarchy(profile, operators, directRecipesByOp);
        const approximationHotspots = this.computeApproximationHotspots(recipes, operators);

        return {
            profile,
            recipes,
            operators,
            cycles,
            prioritizedGaps,
            unimplementedSummary,
            hierarchy,
            approximationHotspots,
        };
    }

    private computeApproximationHotspots(
        recipes: AnalyzedRecipe[],
        operators: Map<string, OperatorAnalysis>,
    ): ApproximationRootEntry[] {
        const recipeByName = new Map<string, AnalyzedRecipe>();
        for (const r of recipes) recipeByName.set(r.name, r);

        const classifyRoot = (rootOp: string): ApproximationClass => {
            if (rootOp === "Div" || rootOp === "Reciprocal") return "rational-div";
            if (rootOp === "Sqrt") return "algebraic-sqrt";
            return "transcendental";
        };

        const rootStats = new Map<
            string,
            {
                rootRecipes: Set<string>;
                dependents: Set<string>;
                soleBlockers: Set<string>;
            }
        >();

        const isConditionalRecipe = (recName: string): boolean => {
            const rec = recipeByName.get(recName);
            return rec?.tags.some((t) => CONDITIONAL_FASTPATH_TAGS.has(t)) ?? false;
        };

        for (const op of operators.values()) {
            if (!op.atlasDetail) continue;
            if (op.status !== "fully-lowerable-approx" && !op.fastPathOnlyExact) continue;
            // Trace general-input fallback path (ignoring conditional constant fast-paths like exact power-of-2 Div)
            const fullSet =
                op.atlasDetail.terminalSets.find(
                    (s) =>
                        s.reachesFullScalar &&
                        !s.recipeChain.some((rn) => isConditionalRecipe(rn)),
                ) ??
                op.atlasDetail.terminalSets.find((s) => s.reachesFullScalar) ??
                op.atlasDetail.terminalSets[0];
            if (!fullSet) continue;

            const approxRoots = new Map<string, string[]>();
            for (const recName of fullSet.recipeChain) {
                const rec = recipeByName.get(recName);
                if (!rec || rec.precision !== "approximation") continue;
                const rootOp = rec.targetOps[0] ?? String(rec.rawTarget);
                const list = approxRoots.get(rootOp) ?? [];
                list.push(recName);
                approxRoots.set(rootOp, list);
            }

            if (approxRoots.size === 0) continue;
            const isSoleBlocker = approxRoots.size === 1;

            for (const [rootOp, rootRecNames] of approxRoots.entries()) {
                let entry = rootStats.get(rootOp);
                if (!entry) {
                    entry = {
                        rootRecipes: new Set<string>(),
                        dependents: new Set<string>(),
                        soleBlockers: new Set<string>(),
                    };
                    rootStats.set(rootOp, entry);
                }
                rootRecNames.forEach((rn) => entry!.rootRecipes.add(rn));
                entry.dependents.add(op.opType);
                if (isSoleBlocker) entry.soleBlockers.add(op.opType);
            }
        }

        return Array.from(rootStats.entries())
            .map(([rootOp, s]) => ({
                rootOp,
                rootRecipes: Array.from(s.rootRecipes).sort(),
                approxClass: classifyRoot(rootOp),
                transitiveDependentCount: s.dependents.size,
                soleBlockerOps: Array.from(s.soleBlockers).sort(),
                sampleDependentOps: Array.from(s.dependents).sort().slice(0, 12),
            }))
            .sort((a, b) => {
                if (b.soleBlockerOps.length !== a.soleBlockerOps.length) {
                    return b.soleBlockerOps.length - a.soleBlockerOps.length;
                }
                return b.transitiveDependentCount - a.transitiveDependentCount;
            });
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
            } else if (best.blockingGaps.length === 0) {
                scalarReadiness =
                    best.scalarComputeRatio > 0 ? "partial-scalar-native" : "native-vector-ready";
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

        const countBlockingGaps = (m: RawMultiset): number => {
            const gaps = new Set<string>();
            for (const e of m.entries.values()) {
                if (!profile.nativePrimitives.has(e.opType)) {
                    gaps.add(e.opType);
                }
            }
            return gaps.size;
        };

        const totalOpCount = (m: RawMultiset): number => {
            let sum = 0;
            for (const e of m.entries.values()) sum += e.count;
            return sum;
        };

        const pruneToBest = (map: Map<string, RawMultiset>): RawMultiset[] => {
            const all = Array.from(map.values()).sort(
                (a, b) =>
                    countBlockingGaps(a) - countBlockingGaps(b) ||
                    Number(a.approx) - Number(b.approx) ||
                    totalOpCount(a) - totalOpCount(b),
            );
            if (all.length > MAX_TERMINAL_SETS) {
                truncated = true;
                return all.slice(0, MAX_TERMINAL_SETS);
            }
            return all;
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

                    // Rank propagation: canonicalizing a 0D scalar op preserves 0D only for
                    // unknown/static-declared child ranks; never clobber concrete non-scalar
                    // ranks observed by dynamic probes (e.g., loss/reduction ops with 0D output).
                    const effectiveChildRank: TensorRankBucket =
                        rank === "0D" &&
                        rec.phase === "canonicalization" &&
                        child.rank === "unknown"
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
                                nextPartials.set(sig, combined);
                            }
                        }
                    }
                    partials = pruneToBest(nextPartials);
                }

                for (const p of partials) {
                    const sig = signatureOf(p);
                    if (!resultsBySig.has(sig)) {
                        resultsBySig.set(sig, p);
                    }
                }
            }

            return pruneToBest(resultsBySig);
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

    private getEffectiveProducedOps(opType: string, rec: AnalyzedRecipe): string[] {
        const probe =
            rec.dynamicAuditsByOp?.[opType] ??
            (rec.dynamicAudit?.testedOp === opType ? rec.dynamicAudit : undefined);
        if (probe && probe.succeeded && probe.actualProducedOps.length > 0) {
            return probe.actualProducedOps;
        }
        return rec.declaredProducedOps;
    }

    private computeDynamicHierarchy(
        profile: HardwareProfile,
        operators: Map<string, OperatorAnalysis>,
        directRecipesByOp: Map<string, AnalyzedRecipe[]>,
    ): DynamicHierarchyReport {
        // 1. Scope to operators actively participating in the recipe library or native profile
        const activeOps = new Set<string>();
        for (const [op, info] of operators.entries()) {
            if (
                info.isNativePrimitive ||
                info.directRecipes.length > 0 ||
                info.producedByRecipes.length > 0
            ) {
                activeOps.add(op);
            }
        }

        // 2. Build primary-path directed graph (using highest-priority recipe per op)
        const primaryRecipeByOp = new Map<string, AnalyzedRecipe>();
        const primaryEdges = new Map<string, string[]>();

        for (const op of activeOps) {
            if (profile.nativePrimitives.has(op)) {
                primaryEdges.set(op, []);
                continue;
            }
            const recs = directRecipesByOp.get(op) ?? [];
            if (recs.length === 0) {
                primaryEdges.set(op, []);
                continue;
            }
            const primary = recs[0];
            primaryRecipeByOp.set(op, primary);
            const deps = Array.from(
                new Set(
                    this.getEffectiveProducedOps(op, primary).filter(
                        (v) => v !== op && activeOps.has(v),
                    ),
                ),
            );
            primaryEdges.set(op, deps);
        }

        // 3. Tarjan's SCC condensation on the primary-path graph (guarantees a DAG)
        let indexCounter = 0;
        const indices = new Map<string, number>();
        const lowlink = new Map<string, number>();
        const onStack = new Set<string>();
        const stack: string[] = [];
        const sccIdByOp = new Map<string, number>();
        const sccs: string[][] = [];

        const strongConnect = (v: string) => {
            indices.set(v, indexCounter);
            lowlink.set(v, indexCounter);
            indexCounter++;
            stack.push(v);
            onStack.add(v);

            for (const w of primaryEdges.get(v) ?? []) {
                if (!indices.has(w)) {
                    strongConnect(w);
                    lowlink.set(v, Math.min(lowlink.get(v)!, lowlink.get(w)!));
                } else if (onStack.has(w)) {
                    lowlink.set(v, Math.min(lowlink.get(v)!, indices.get(w)!));
                }
            }

            if (lowlink.get(v) === indices.get(v)) {
                const sccId = sccs.length;
                const members: string[] = [];
                while (true) {
                    const w = stack.pop()!;
                    onStack.delete(w);
                    sccIdByOp.set(w, sccId);
                    members.push(w);
                    if (w === v) break;
                }
                sccs.push(members.sort());
            }
        };

        for (const op of Array.from(activeOps).sort()) {
            if (!indices.has(op)) strongConnect(op);
        }

        const sccCycles = sccs.filter((members) => members.length > 1);

        // 4. Bottom-up level calculation on the condensed SCC DAG
        const sccLevelMemo = new Map<number, number>();
        const unresolvedGaps: string[] = [];

        const computeSccLevel = (sccId: number): number => {
            if (sccLevelMemo.has(sccId)) return sccLevelMemo.get(sccId)!;
            const members = sccs[sccId];

            if (members.length === 1) {
                const op = members[0];
                if (profile.nativePrimitives.has(op)) {
                    sccLevelMemo.set(sccId, 0);
                    return 0;
                }
                if (!primaryRecipeByOp.has(op)) {
                    sccLevelMemo.set(sccId, -1);
                    return -1;
                }
            }

            let maxChildLevel = 0;
            for (const op of members) {
                for (const dep of primaryEdges.get(op) ?? []) {
                    const depScc = sccIdByOp.get(dep)!;
                    if (depScc === sccId) continue;
                    const depLevel = computeSccLevel(depScc);
                    if (depLevel > maxChildLevel) {
                        maxChildLevel = depLevel;
                    }
                }
            }

            const level = maxChildLevel + 1;
            sccLevelMemo.set(sccId, level);
            return level;
        };

        const levelByOp = new Map<string, number>();
        let maxLevel = 0;

        for (const op of Array.from(activeOps).sort()) {
            const lvl = computeSccLevel(sccIdByOp.get(op)!);
            levelByOp.set(op, lvl);
            if (lvl === -1) {
                unresolvedGaps.push(op);
            } else if (lvl > maxLevel) {
                maxLevel = lvl;
            }
        }

        // 5. Compute fan-in, fan-out, and min/max level spread across alternative recipes
        const reversePrimary = new Map<string, Set<string>>();
        for (const op of activeOps) reversePrimary.set(op, new Set());
        for (const [u, deps] of primaryEdges.entries()) {
            for (const v of deps) {
                reversePrimary.get(v)?.add(u);
            }
        }

        const getTransitiveFanIn = (target: string): number => {
            const visited = new Set<string>();
            const queue = Array.from(reversePrimary.get(target) ?? []);
            while (queue.length > 0) {
                const curr = queue.shift()!;
                if (visited.has(curr) || curr === target) continue;
                visited.add(curr);
                for (const parent of reversePrimary.get(curr) ?? []) {
                    if (!visited.has(parent)) queue.push(parent);
                }
            }
            return visited.size;
        };

        const metricsByOp = new Map<string, OperatorHierarchyMetrics>();

        for (const op of activeOps) {
            const level = levelByOp.get(op)!;
            const sccId = sccIdByOp.get(op)!;
            const primary = primaryRecipeByOp.get(op);
            const directFanIn = reversePrimary.get(op)?.size ?? 0;
            const transitiveFanIn = getTransitiveFanIn(op);
            const directFanOut = (primaryEdges.get(op) ?? []).length;
            const crossesIntoLoop =
                primary !== undefined &&
                (primary.phase === "loop-lowering" || primary.exposesControlFlow);

            let minLevel = level;
            let maxRecLevel = level;

            if (level > 0) {
                const recs = directRecipesByOp.get(op) ?? [];
                const recLevels: number[] = [];
                for (const rec of recs) {
                    const deps = this.getEffectiveProducedOps(op, rec).filter(
                        (v) => v !== op && activeOps.has(v) && sccIdByOp.get(v) !== sccId,
                    );
                    let maxDep = 0;
                    for (const dep of deps) {
                        const dl = levelByOp.get(dep) ?? 0;
                        if (dl > maxDep) maxDep = dl;
                    }
                    recLevels.push(maxDep + 1);
                }
                if (recLevels.length > 0) {
                    minLevel = Math.min(...recLevels);
                    maxRecLevel = Math.max(...recLevels);
                }
            }

            const metrics: OperatorHierarchyMetrics = {
                level,
                minLevel,
                maxLevel: maxRecLevel,
                primaryRecipe: primary?.name,
                sccId,
                directFanIn,
                transitiveFanIn,
                directFanOut,
                crossesIntoLoop,
            };
            metricsByOp.set(op, metrics);

            const existing = operators.get(op);
            if (existing) {
                operators.set(op, { ...existing, hierarchyMetrics: metrics });
            }
        }

        // 6. Aggregate emergent strata summaries (top-down: maxLevel -> 0)
        const levels: AbstractionLevelSummary[] = [];
        const loopBoundaryLevels: number[] = [];

        for (let k = maxLevel; k >= 0; k--) {
            const opsAtLevel = Array.from(activeOps)
                .filter((op) => levelByOp.get(op) === k)
                .sort();
            if (opsAtLevel.length === 0 && k > 0) continue;

            const catCounts = new Map<string, number>();
            let canonicalizationCount = 0;
            let loopLoweringCount = 0;
            let controlFlowBoundaryCount = 0;
            let approximationCount = 0;
            let fanInSum = 0;

            for (const op of opsAtLevel) {
                const cat = this.formatCategory(this.allSchemaOps.get(op)?.category) ?? "Other";
                catCounts.set(cat, (catCounts.get(cat) ?? 0) + 1);

                const primary = primaryRecipeByOp.get(op);
                if (primary) {
                    if (primary.phase === "canonicalization") canonicalizationCount++;
                    else loopLoweringCount++;
                    if (primary.precision === "approximation") approximationCount++;
                }

                const m = metricsByOp.get(op)!;
                if (m.crossesIntoLoop) controlFlowBoundaryCount++;
                fanInSum += m.transitiveFanIn;
            }

            const dominantCategories = Array.from(catCounts.entries())
                .map(([category, count]) => ({
                    category,
                    count,
                    ratio: opsAtLevel.length > 0 ? count / opsAtLevel.length : 0,
                }))
                .sort((a, b) => b.count - a.count || a.category.localeCompare(b.category));

            const avgFanIn = opsAtLevel.length > 0 ? fanInSum / opsAtLevel.length : 0;

            if (k > 0 && controlFlowBoundaryCount > 0) {
                loopBoundaryLevels.push(k);
            }

            let signature: string;
            if (k === 0) {
                signature = `Target Native Sinks (${profile.name})`;
            } else {
                const topCats = dominantCategories
                    .slice(0, 2)
                    .map((c) => c.category)
                    .join("/");
                const totalWithRec = canonicalizationCount + loopLoweringCount;
                const canonPct =
                    totalWithRec > 0 ? Math.round((canonicalizationCount / totalWithRec) * 100) : 0;
                const loopPct = 100 - canonPct;
                const phasePart =
                    canonPct === 100
                        ? "100% Canon"
                        : loopPct === 100
                          ? "100% Loop"
                          : `${canonPct}% Canon / ${loopPct}% Loop`;
                const cfPart =
                    controlFlowBoundaryCount > 0
                        ? ` · ${controlFlowBoundaryCount} Loop-Boundary`
                        : "";
                const approxPart =
                    approximationCount > 0 ? ` · ${approximationCount} Approx` : "";
                signature = `${topCats} · ${phasePart}${cfPart}${approxPart}`;
            }

            levels.push({
                level: k,
                operators: opsAtLevel,
                dominantCategories,
                canonicalizationCount,
                loopLoweringCount,
                controlFlowBoundaryCount,
                approximationCount,
                avgFanIn,
                signature,
            });
        }

        // 7. Identify Convergence Hubs (intermediate ops 0 < level < maxLevel)
        const convergenceHubs: DialectHubEntry[] = Array.from(activeOps)
            .filter((op) => {
                const lvl = levelByOp.get(op)!;
                const m = metricsByOp.get(op)!;
                return lvl > 0 && lvl < maxLevel && m.transitiveFanIn > 0;
            })
            .map((op) => {
                const m = metricsByOp.get(op)!;
                return {
                    opType: op,
                    level: m.level,
                    category: this.formatCategory(this.allSchemaOps.get(op)?.category),
                    directFanIn: m.directFanIn,
                    transitiveFanIn: m.transitiveFanIn,
                    directFanOut: m.directFanOut,
                    primaryPhase: primaryRecipeByOp.get(op)?.phase,
                };
            })
            .sort(
                (a, b) =>
                    b.transitiveFanIn - a.transitiveFanIn ||
                    b.directFanIn - a.directFanIn ||
                    a.opType.localeCompare(b.opType),
            )
            .slice(0, 15);

        // 8. Compute Inter-Level Transition Matrix
        const transMap = new Map<string, { fromLevel: number; toLevel: number; ops: Set<string>; count: number }>();
        let totalTransitions = 0;
        let progressiveTransitions = 0;

        for (const [u, deps] of primaryEdges.entries()) {
            const fromLevel = levelByOp.get(u)!;
            if (fromLevel <= 0) continue;
            for (const v of deps) {
                const toLevel = levelByOp.get(v)!;
                if (toLevel < 0 || toLevel >= fromLevel) continue;
                const key = `${fromLevel}->${toLevel}`;
                const entry = transMap.get(key) ?? {
                    fromLevel,
                    toLevel,
                    ops: new Set<string>(),
                    count: 0,
                };
                entry.count++;
                entry.ops.add(u);
                transMap.set(key, entry);

                totalTransitions++;
                if (fromLevel - toLevel === 1) progressiveTransitions++;
            }
        }

        const transitions: LevelTransitionEdge[] = Array.from(transMap.values())
            .map((e) => ({
                fromLevel: e.fromLevel,
                toLevel: e.toLevel,
                count: e.count,
                isProgressive: e.fromLevel - e.toLevel === 1,
                sampleOps: Array.from(e.ops).sort().slice(0, 5),
            }))
            .sort((a, b) => b.fromLevel - a.fromLevel || b.toLevel - a.toLevel);

        const progressiveRatio =
            totalTransitions > 0 ? progressiveTransitions / totalTransitions : 1;

        return {
            maxLevel,
            levels,
            unresolvedGaps: unresolvedGaps.sort(),
            convergenceHubs,
            transitions,
            progressiveRatio,
            loopBoundaryLevels: loopBoundaryLevels.sort((a, b) => b - a),
            sccCycles,
        };
    }
}