import {
    initializeSchemaRegistry,
    OpRegistry,
    OpCategory,
    Recipes,
    getRecipeMetadata,
    type BaseRecipe,
} from "@specs-feup/onnx-flow";
import type {
    AnalyzedRecipe,
    HardwareProfile,
    LibraryAnalysisReport,
    OperatorAnalysis,
    OpCoverageStatus,
    UnimplementedSummary,
} from "./types.js";

const OPSET_VERSION = 19;

export class RecipeGraphAnalyzer {
    private categoryToOps = new Map<OpCategory, string[]>();
    private allSchemaOps = new Map<string, OpCategory | undefined>();

    constructor() {
        initializeSchemaRegistry();
        this.indexSchemas();
    }

    /**
     * Extracts all registered ONNX operators from OpRegistry at opset 19.
     */
    private indexSchemas(): void {
        const registry = OpRegistry.getInstance() as unknown as {
            schemas?: Map<string, Map<number, { opType: string; category?: OpCategory }>>;
        };

        // Fallback list of standard ONNX ops in case OpRegistry internal map is private
        const knownOps = new Set<string>([
            ...Object.keys(Recipes.canonicalization),
            ...Object.keys(Recipes.loopLowering),
            "Add", "Sub", "Mul", "Div", "Mod", "Pow", "Abs", "Neg", "Sign", "Exp", "Log", "Sqrt",
            "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh",
            "Erf", "Ceil", "Floor", "Round", "Clip", "Relu", "LeakyRelu", "Elu", "Celu", "Selu",
            "Sigmoid", "HardSigmoid", "Softplus", "Softsign", "Mish", "Softmax",
            "And", "Or", "Xor", "Not", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
            "Max", "Min", "Where", "IsNan", "IsInf",
            "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
            "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
            "MatMul", "Gemm", "Conv", "AveragePool", "MaxPool",
            "BatchNormalization", "LayerNormalization", "InstanceNormalization", "GroupNormalization",
            "QuantizeLinear", "DequantizeLinear",
            "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Concat", "Split", "Slice", "Pad", "Expand",
            "Gather", "ScatterElements", "Range", "Shape", "Size", "Cast", "Identity",
            "Constant", "ConstantOfShape", "Loop", "If", "Scan",
        ]);

        if (registry.schemas instanceof Map) {
            for (const opName of registry.schemas.keys()) {
                knownOps.add(opName);
            }
        }

        const pubRegistry = OpRegistry.getInstance();
        for (const opName of knownOps) {
            const schema = pubRegistry.get(opName, OPSET_VERSION);
            const cat = schema?.category;
            this.allSchemaOps.set(opName, cat);
            if (cat !== undefined) {
                const list = this.categoryToOps.get(cat) ?? [];
                list.push(opName);
                this.categoryToOps.set(cat, list);
            }
        }
    }

    /**
     * Instantiates and normalizes all recipes from ONNX-Flow's catalog.
     */
    public loadRecipes(): AnalyzedRecipe[] {
        const instances: { recipe: BaseRecipe; explicitPhase: "canonicalization" | "loop-lowering" }[] = [];

        for (const RecipeClass of Object.values(Recipes.canonicalization)) {
            instances.push({ recipe: new RecipeClass(), explicitPhase: "canonicalization" });
        }
        for (const RecipeClass of Object.values(Recipes.loopLowering)) {
            instances.push({ recipe: new RecipeClass(), explicitPhase: "loop-lowering" });
        }

        return instances.map(({ recipe, explicitPhase }) => {
            const meta = getRecipeMetadata(recipe);
            const rawTarget = meta.targetOp;

            // Expand OpCategory (e.g. ElementWise, Reduction) into concrete ONNX op names
            let targetOps: string[];
            if (
                rawTarget === OpCategory.ElementWise ||
                rawTarget === OpCategory.Reduction ||
                rawTarget === OpCategory.ControlFlow ||
                rawTarget === OpCategory.DataMovement
            ) {
                targetOps = this.categoryToOps.get(rawTarget as OpCategory) ?? [String(rawTarget)];
            } else {
                targetOps = [String(rawTarget)];
            }

            return {
                name: meta.name,
                rawTarget,
                targetOps,
                phase: recipe.phase ?? explicitPhase,
                precision: meta.precision,
                exposesControlFlow: meta.exposesControlFlow,
                exposesDataAccess: meta.exposesDataAccess,
                declaredProducedOps: meta.producedOps,
            };
        });
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

                        // Self-reference inside loop lowering (e.g. ElementWise Add lowered into Loop + scalar Add)
                        if (nextOp === currentOp && rec.phase === "loop-lowering") {
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
                category: this. formatCategory(this.allSchemaOps.get(op)),
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

    /**
     * Returns all standard ONNX operator names known to OpRegistry that currently
     * have zero direct decomposition recipes.
     */
    public getUnimplementedSummary(profile: HardwareProfile, recipes: AnalyzedRecipe[]): UnimplementedSummary {
        const coveredOps = new Set<string>();
        for (const r of recipes) {
            r.targetOps.forEach((op) => coveredOps.add(op));
        }

        const registry = OpRegistry.getInstance() as unknown as {
            schemas?: Map<string, unknown>;
        };

        const schemaKeys = registry.schemas instanceof Map ? Array.from(registry.schemas.keys()) : [];
        const byCategory: Record<string, string[]> = {};
        let unimplementedCount = 0;

        const pubRegistry = OpRegistry.getInstance();
        for (const opName of schemaKeys) {
            if (!coveredOps.has(opName) && !profile.nativePrimitives.has(opName)) {
                unimplementedCount++;
                const schema = pubRegistry.get(opName, OPSET_VERSION);
                const catName = this.formatCategory(schema?.category) ?? "Uncategorized";
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
}