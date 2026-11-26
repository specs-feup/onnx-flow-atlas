import * as OnnxFlow from "@specs-feup/onnx-flow";
import {
    OnnxGraph,
    OnnxGraphTransformer,
    OperationNode,
    GraphBuilder,
    TensorNode,
    ConstantNode,
    RegionArgumentNode,
    DataType,
    Recipes,
    IndexingMode,
    Schedule,
    inferShapes,
    makeTensorProto,
    topologicalSortOperationNodes,
    type DecompositionRecipe,
    type LoopLoweringRecipe,
    type LoopCarryContext,
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    type AttributeMap,
} from "@specs-feup/onnx-flow";
import type {
    AnalyzedRecipe,
    DynamicProbeResult,
    FusedLoweringMetrics,
    FusedModeVariant,
    OpQuantityEntry,
    OpRole,
    TensorRankBucket,
} from "./types.js";

const CONTROL_OPS = new Set(["Loop", "If", "Scan"]);
const MEMORY_OPS = new Set([
    "Gather", "ScatterElements", "ScatterND", "GatherND", "GatherElements",
]);
const VIEW_OPS = new Set([
    "Reshape", "Squeeze", "Unsqueeze", "Identity",
    "Constant", "ConstantOfShape", "Shape", "Size",
]);
const INDEXING_ARITH_OPS = new Set([
    "Add", "Sub", "Mul", "Div", "Mod", "Floor", "Cast", "Range",
    "Equal", "Less", "Greater", "And", "Where",
]);
const INDEXING_BOOL_OPS = new Set(["Equal", "Less", "Greater", "And", "Or", "Not"]);
const REDUCTION_OPS = new Set(["ArgMax", "ArgMin"]);

type GraphNode = ReturnType<OperationNode.Class["outgoers"]["targets"]["toArray"]>[number];

const DEFAULT_FUSED_BENCHMARK_OPS = [
    "SoftmaxCrossEntropyLoss",
    "NegativeLogLikelihoodLoss",
    "LogSoftmax",
    "Softmax",
    "LayerNormalization",
    "BatchNormalization",
    "GroupNormalization",
    "InstanceNormalization",
    "Gelu",
    "Mish",
    "Hardmax",
    "HardSwish",
    "Det",
    "GRU",
    "LSTM",
];

export class DynamicProbeVerifier {
    public verifyCanonicalizationRecipes(analyzedRecipes: AnalyzedRecipe[]): void {
        this.verifyAllRecipes(analyzedRecipes);
    }

    public measureFusedBenchmarks(
        opTypes: string[] = DEFAULT_FUSED_BENCHMARK_OPS,
    ): FusedLoweringMetrics[] {
        const computeMaxLoopDepth = (g: OnnxGraph.Class, depth: number = 0): number => {
            let maxD = depth;
            for (const op of g.getOperationNodes()) {
                for (const sub of op.regions) {
                    const subDepth = op.type === "Loop" ? depth + 1 : depth;
                    maxD = Math.max(maxD, computeMaxLoopDepth(sub, subDepth));
                }
            }
            return maxD;
        };

        const collectPipelineCounts = (
            opType: string,
            extraOpts: Record<string, unknown> = {},
        ) => {
            const { graph } = this.buildSyntheticSingleOpGraph(opType, undefined, 1);
            const transformed = new OnnxGraphTransformer({
                canonicalize: true,
                loopLowering: true,
                fuse: true,
                coalesce: true,
                recurse: true,
                optimize: true,
                decomposeForCgra: false,
                indexingMode: IndexingMode.Pointer,
                ...extraOpts,
            }).apply(graph, false);

            const bucketMap = new Map<string, OpQuantityEntry>();
            const actualSet = new Set<string>();
            this.collectEmittedOpsRecursively(transformed, 0, true, actualSet, bucketMap);

            let loopCount = 0;
            let scalarComputeCount = 0;
            let nonScalarComputeCount = 0;
            let indexingScalarCount = 0;
            let controlMemoryViewCount = 0;
            let totalOps = 0;

            for (const entry of bucketMap.values()) {
                totalOps += entry.count;
                if (entry.opType === "Loop") loopCount += entry.count;
                if (entry.role === "compute") {
                    if (entry.rank === "0D") scalarComputeCount += entry.count;
                    else nonScalarComputeCount += entry.count;
                } else if (entry.role === "indexing") {
                    indexingScalarCount += entry.count;
                } else {
                    controlMemoryViewCount += entry.count;
                }
            }

            const barrierSet = new Set<string>();
            for (const topOp of transformed.getOperationNodes()) {
                if (topOp.type === "Loop") {
                    const srcTypes = topOp.getMetadata("sourceOpTypes");
                    if (Array.isArray(srcTypes)) {
                        for (const t of srcTypes) {
                            const s = String(t);
                            if (
                                s === "MatMul" ||
                                s === "Conv" ||
                                s === "MaxPool" ||
                                s === "AveragePool" ||
                                s === "Slice" ||
                                s === "Concat" ||
                                s === "Pad" ||
                                s === "TopK" ||
                                s === "ArgMax" ||
                                s === "ArgMin" ||
                                s === "Softmax" ||
                                s === "LayerNormalization" ||
                                s.startsWith("Reduce")
                            ) {
                                barrierSet.add(s);
                            }
                        }
                    }
                }
            }

            return {
                loopCount,
                maxLoopDepth: computeMaxLoopDepth(transformed, 0),
                scalarComputeCount,
                nonScalarComputeCount,
                indexingScalarCount,
                controlMemoryViewCount,
                infraPerComputeRatio:
                    (indexingScalarCount + controlMemoryViewCount) /
                    Math.max(1, scalarComputeCount),
                fusionBarrierOps: Array.from(barrierSet).sort(),
                totalOps,
            };
        };

        return opTypes.map((opType) => {
            try {
                const base = collectPipelineCounts(opType);
                const variants: FusedModeVariant[] = [];

                if (base.loopCount > 0) {
                    const scanRes = collectPipelineCounts(opType, { useScanOutputs: true });
                    variants.push({
                        label: "scan_outputs",
                        loopCount: scanRes.loopCount,
                        controlMemoryViewCount: scanRes.controlMemoryViewCount,
                        totalOps: scanRes.totalOps,
                    });
                }

                if (opType === "Softmax" || opType === "LayerNormalization" || opType === "LogSoftmax" || opType === "RMSNormalization") {
                    const loopRecipeName =
                        opType === "Softmax"
                            ? "LowerSoftmaxLoop"
                            : opType === "LogSoftmax"
                              ? "LowerLogSoftmaxLoop"
                              : opType === "RMSNormalization"
                                ? "LowerRMSNormLoop"
                                : "LowerLayerNormLoop";
                    const schedRes = collectPipelineCounts(opType, {
                        schedule: new Schedule().useRecipe(opType, loopRecipeName),
                    });
                    variants.push({
                        label: loopRecipeName,
                        loopCount: schedRes.loopCount,
                        controlMemoryViewCount: schedRes.controlMemoryViewCount,
                        totalOps: schedRes.totalOps,
                    });
                }

                return {
                    opType,
                    succeeded: true,
                    ...base,
                    variants,
                };
            } catch (err) {
                return {
                    opType,
                    succeeded: false,
                    loopCount: 0,
                    maxLoopDepth: 0,
                    scalarComputeCount: 0,
                    nonScalarComputeCount: 0,
                    indexingScalarCount: 0,
                    controlMemoryViewCount: 0,
                    infraPerComputeRatio: 0,
                    fusionBarrierOps: [],
                    totalOps: 0,
                    error: err instanceof Error ? err.message : String(err),
                };
            }
        });
    }

    public verifyAllRecipes(analyzedRecipes: AnalyzedRecipe[]): void {
        const canonMap = new Map<string, DecompositionRecipe>();
        const loopMap = new Map<string, LoopLoweringRecipe>();

        const canonRegistry: Record<string, new () => DecompositionRecipe> =
            Recipes.canonicalization;
        const loopRegistry: Record<string, new () => LoopLoweringRecipe> =
            Recipes.loopLowering;

        for (const R of Object.values(canonRegistry)) {
            const inst = new R();
            canonMap.set(inst.name, inst);
        }
        for (const R of Object.values(loopRegistry)) {
            const inst = new R();
            loopMap.set(inst.name, inst);
        }

        for (const analyzed of analyzedRecipes) {
            const isLoopLowering = analyzed.phase === "loop-lowering";
            const canonRecipe = canonMap.get(analyzed.name);
            const loopRecipe = loopMap.get(analyzed.name);
            if (!canonRecipe && !loopRecipe) continue;

            const auditsByOp: Record<string, DynamicProbeResult> = {};

            for (const testOp of analyzed.targetOps) {
                auditsByOp[testOp] = isLoopLowering && loopRecipe
                    ? this.probeLoopLoweringOp(loopRecipe, analyzed, testOp)
                    : this.probeCanonicalizationOp(canonRecipe!, analyzed, testOp);
            }

            analyzed.dynamicAuditsByOp = auditsByOp;
            if (analyzed.targetOps.length > 0) {
                analyzed.dynamicAudit = auditsByOp[analyzed.targetOps[0]];
            }
        }
    }

    private probeCanonicalizationOp(
        recipe: DecompositionRecipe,
        analyzed: AnalyzedRecipe,
        testOp: string,
    ): DynamicProbeResult {
        try {
            const { graph, opNode, probeInputSummary } = this.buildSyntheticSingleOpGraph(testOp, recipe.name);
            if (!recipe.match(opNode)) {
                return this.makeFailedResult(
                    testOp,
                    probeInputSummary,
                    analyzed.declaredProducedOps,
                    `recipe.match() returned false on standard synthetic ${testOp} node`,
                );
            }

            const builder = new GraphBuilder(graph, `probe_${recipe.name}_${testOp}`);
            recipe.apply(opNode, builder);
            if (graph.hasNode(opNode.id)) {
                builder.removeNode(opNode);
            }
            try {
                inferShapes(graph);
            } catch {
                // Non-fatal: continue with whatever shapes are already present
            }

            const bucketMap = new Map<string, OpQuantityEntry>();
            const actualSet = new Set<string>();
            this.collectEmittedOpsRecursively(graph, 0, false, actualSet, bucketMap);

            return this.finalizeProbeResult(
                recipe.name,
                testOp,
                probeInputSummary,
                analyzed.declaredProducedOps,
                actualSet,
                bucketMap,
            );
        } catch (err) {
            return this.makeFailedResult(
                testOp,
                "N/A",
                analyzed.declaredProducedOps,
                err instanceof Error ? err.message : String(err),
            );
        }
    }

    private probeLoopLoweringOp(
        recipe: LoopLoweringRecipe,
        analyzed: AnalyzedRecipe,
        testOp: string,
    ): DynamicProbeResult {
        try {
            const { graph, opNode, outShape, outDtype, probeInputSummary } =
                this.buildSyntheticSingleOpGraph(testOp, recipe.name);

            if (!recipe.match(opNode)) {
                return this.makeFailedResult(
                    testOp,
                    probeInputSummary,
                    analyzed.declaredProducedOps,
                    `recipe.match() returned false on standard synthetic ${testOp} node`,
                );
            }

            const outerBuilder = new GraphBuilder(graph, `probe_outer_${recipe.name}_${testOp}`);

            if (recipe.lowerStandalone) {
                recipe.lowerStandalone(opNode, outerBuilder, IndexingMode.Pointer);
                if (graph.hasNode(opNode.id)) {
                    outerBuilder.removeNode(opNode);
                }
                try {
                    inferShapes(graph);
                } catch {
                    // Non-fatal
                }
                const bucketMap = new Map<string, OpQuantityEntry>();
                const actualSet = new Set<string>();
                this.collectEmittedOpsRecursively(graph, 0, true, actualSet, bucketMap);
                return this.finalizeProbeResult(
                    recipe.name,
                    testOp,
                    `${probeInputSummary} [Nested/Standalone]`,
                    analyzed.declaredProducedOps,
                    actualSet,
                    bucketMap,
                );
            }

            const body = OnnxGraph.create();
            const innerBuilder = new GraphBuilder(body, `probe_body_${recipe.name}_${testOp}`);

            const valueMap = new Map<string, ValueNode>();
            const iter = innerBuilder.createTensor("loop_iter", DataType.INT64, [], "input") as ConcreteValueNode;
            const axes = innerBuilder.createConstant(
                "loop_axes",
                makeTensorProto(DataType.INT64, [1], [0]),
            ) as ConcreteValueNode;

            let carryShape: KnownShape = [];
            let targetShapeNode: ValueNode | undefined;
            if (recipe.getLoopBounds) {
                const bounds = recipe.getLoopBounds(opNode, outShape);
                if (Array.isArray(bounds.carryShape)) {
                    carryShape = bounds.carryShape;
                }
                if (bounds.targetShape && !Array.isArray(bounds.targetShape)) {
                    targetShapeNode = bounds.targetShape;
                }
            }

            const carryNode = innerBuilder.createTensor(
                "loop_carry",
                outDtype,
                carryShape,
                "input",
            ) as ConcreteValueNode;

            const loopCtx: LoopCarryContext = {
                indexingMode: IndexingMode.Pointer,
                outerBuilder,
                innerBuilder,
                addCarry: (tag, _initialVal, elemTy, shape) =>
                    innerBuilder.createTensor(`carry_${tag}`, elemTy, shape, "input") as TensorNode.Class,
                setNextState: () => {
                    // No-op in single-op probe harness
                },
            };

            recipe.apply(
                opNode,
                body,
                valueMap,
                iter,
                axes,
                outShape,
                carryNode,
                targetShapeNode,
                loopCtx,
            );

            if (recipe.postProcess) {
                const fakeLoopOut = outerBuilder.createTensor(
                    "loop_out",
                    outDtype,
                    outShape,
                    "intermediate",
                );
                recipe.postProcess(opNode, outerBuilder, fakeLoopOut);
            }

            if (graph.hasNode(opNode.id)) {
                outerBuilder.removeNode(opNode);
            }

            try {
                inferShapes(graph);
            } catch {
                // Non-fatal
            }
            try {
                inferShapes(body);
            } catch {
                // Non-fatal
            }

            const bucketMap = new Map<string, OpQuantityEntry>();
            const actualSet = new Set<string>();

            actualSet.add("Loop");
            bucketMap.set("Loop|0D|control|false", {
                opType: "Loop",
                rank: "0D",
                role: "control",
                count: 1,
                inLoop: false,
                estimated: false,
            });

            this.collectEmittedOpsRecursively(graph, 0, true, actualSet, bucketMap);
            this.collectEmittedOpsRecursively(body, 1, true, actualSet, bucketMap);

            return this.finalizeProbeResult(
                recipe.name,
                testOp,
                `${probeInputSummary} [AGU/Pointer]`,
                analyzed.declaredProducedOps,
                actualSet,
                bucketMap,
            );
        } catch (err) {
            return this.makeFailedResult(
                testOp,
                "N/A",
                analyzed.declaredProducedOps,
                err instanceof Error ? err.message : String(err),
            );
        }
    }

    private finalizeProbeResult(
        recipeName: string,
        testedOp: string,
        probeInputSummary: string,
        declaredProducedOps: string[],
        actualSet: Set<string>,
        bucketMap: Map<string, OpQuantityEntry>,
    ): DynamicProbeResult {
        const actualProducedOps = Array.from(actualSet).sort();
        const emittedEntries = Array.from(bucketMap.values()).sort(
            (a, b) =>
                a.opType.localeCompare(b.opType) ||
                a.rank.localeCompare(b.rank) ||
                Number(a.inLoop) - Number(b.inLoop),
        );

        const declaredSet = new Set(declaredProducedOps);
        const undeclaredOps = actualProducedOps.filter(
            (op) => !declaredSet.has(op) && !(recipeName === "LowerElementWise" && op === testedOp),
        );
        const unobservedOps = declaredProducedOps.filter((op) => !actualSet.has(op));

        return {
            testedOp,
            probeInputSummary,
            succeeded: true,
            actualProducedOps,
            emittedEntries,
            undeclaredOps,
            unobservedOps,
        };
    }

    private makeFailedResult(
        testedOp: string,
        probeInputSummary: string,
        declaredProducedOps: string[],
        error: string,
    ): DynamicProbeResult {
        return {
            testedOp,
            probeInputSummary,
            succeeded: false,
            actualProducedOps: [],
            emittedEntries: [],
            undeclaredOps: [],
            unobservedOps: [...declaredProducedOps],
            error,
        };
    }

    private buildSyntheticSingleOpGraph(
        opType: string,
        recipeName?: string,
        rnnSeqLen: number = 3,
    ) {
        const graph = OnnxGraph.create();
        const builder = new GraphBuilder(graph, "probe");

        const inputSummaries: string[] = [];
        const makeInput = (id: string, dtype: DataType, shape: number[]): ValueNode => {
            inputSummaries.push(`${id}:[${shape.join(",")}]`);
            return builder.createTensor(id, dtype, shape, "input");
        };

        let inputs: ValueNode[] = [];
        let attrs: AttributeMap = {};
        let outputNames: string[] = ["Y"];
        let outShape: number[] = [4, 4];
        let outDtype: DataType = DataType.FLOAT;
        let regions: OnnxGraph.Class[] | undefined;

        switch (opType) {
            case "Gemm":
                inputs = [
                    makeInput("A", DataType.FLOAT, [4, 8]),
                    makeInput("B", DataType.FLOAT, [8, 6]),
                    makeInput("C", DataType.FLOAT, [6]),
                ];
                attrs = { alpha: 1.0, beta: 1.0, transA: 0, transB: 0 };
                outShape = [4, 6];
                break;
            case "MatMul":
                inputs = [
                    makeInput("A", DataType.FLOAT, [4, 8]),
                    makeInput("B", DataType.FLOAT, [8, 6]),
                ];
                outShape = [4, 6];
                break;
            case "MatMulInteger":
                inputs = [
                    makeInput("A", DataType.UINT8, [4, 8]),
                    makeInput("B", DataType.UINT8, [8, 6]),
                    builder.createConstant("a_zp", makeTensorProto(DataType.UINT8, [], [0])),
                    builder.createConstant("b_zp", makeTensorProto(DataType.UINT8, [], [0])),
                ];
                outShape = [4, 6];
                outDtype = DataType.INT32;
                break;
            case "QLinearMatMul":
                inputs = [
                    makeInput("A", DataType.UINT8, [4, 8]),
                    builder.createConstant("a_scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("a_zp", makeTensorProto(DataType.UINT8, [], [0])),
                    makeInput("B", DataType.UINT8, [8, 6]),
                    builder.createConstant("b_scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("b_zp", makeTensorProto(DataType.UINT8, [], [0])),
                    builder.createConstant("y_scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("y_zp", makeTensorProto(DataType.UINT8, [], [0])),
                ];
                outShape = [4, 6];
                outDtype = DataType.UINT8;
                break;
            case "Conv":
                if (recipeName === "LowerConv1x1ToMatMul") {
                    inputs = [
                        makeInput("X", DataType.FLOAT, [1, 3, 8, 8]),
                        makeInput("W", DataType.FLOAT, [4, 3, 1, 1]),
                        makeInput("B", DataType.FLOAT, [4]),
                    ];
                    attrs = { kernel_shape: [1, 1], strides: [1, 1], pads: [0, 0, 0, 0] };
                    outShape = [1, 4, 8, 8];
                } else {
                    inputs = [
                        makeInput("X", DataType.FLOAT, [1, 3, 8, 8]),
                        makeInput("W", DataType.FLOAT, [4, 3, 3, 3]),
                        makeInput("B", DataType.FLOAT, [4]),
                    ];
                    attrs = { kernel_shape: [3, 3], strides: [1, 1], pads: [0, 0, 0, 0] };
                    outShape = [1, 4, 6, 6];
                }
                break;
            case "ConvInteger":
                inputs = [
                    makeInput("X", DataType.UINT8, [1, 3, 8, 8]),
                    makeInput("W", DataType.UINT8, [4, 3, 3, 3]),
                    builder.createConstant("x_zp", makeTensorProto(DataType.UINT8, [], [0])),
                    builder.createConstant("w_zp", makeTensorProto(DataType.UINT8, [], [0])),
                ];
                attrs = { strides: [1, 1], pads: [0, 0, 0, 0] };
                outShape = [1, 4, 6, 6];
                outDtype = DataType.INT32;
                break;
            case "QLinearConv":
                inputs = [
                    makeInput("X", DataType.UINT8, [1, 3, 8, 8]),
                    builder.createConstant("x_scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("x_zp", makeTensorProto(DataType.UINT8, [], [0])),
                    makeInput("W", DataType.UINT8, [4, 3, 3, 3]),
                    builder.createConstant("w_scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("w_zp", makeTensorProto(DataType.UINT8, [], [0])),
                    builder.createConstant("y_scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("y_zp", makeTensorProto(DataType.UINT8, [], [0])),
                ];
                attrs = { strides: [1, 1], pads: [0, 0, 0, 0] };
                outShape = [1, 4, 6, 6];
                outDtype = DataType.UINT8;
                break;
            case "ConvTranspose":
                inputs = [
                    makeInput("X", DataType.FLOAT, [1, 3, 4, 4]),
                    makeInput("W", DataType.FLOAT, [3, 4, 3, 3]),
                ];
                attrs = { strides: [1, 1], pads: [0, 0, 0, 0] };
                outShape = [1, 4, 6, 6];
                break;
            case "Range":
                inputs = [
                    builder.createConstant("start", makeTensorProto(DataType.INT64, [], [0])),
                    builder.createConstant("limit", makeTensorProto(DataType.INT64, [], [8])),
                    builder.createConstant("delta", makeTensorProto(DataType.INT64, [], [1])),
                ];
                inputSummaries.push("start:0, limit:8, delta:1");
                outShape = [8];
                outDtype = DataType.INT64;
                break;
            case "Transpose":
                inputs = [makeInput("X", DataType.FLOAT, [4, 8])];
                attrs = { perm: [1, 0] };
                outShape = [8, 4];
                break;
            case "Slice":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 8]),
                    builder.createConstant("starts", makeTensorProto(DataType.INT64, [2], [0, 0])),
                    builder.createConstant("ends", makeTensorProto(DataType.INT64, [2], [2, 4])),
                    builder.createConstant("axes", makeTensorProto(DataType.INT64, [2], [0, 1])),
                    builder.createConstant("steps", makeTensorProto(DataType.INT64, [2], [1, 1])),
                ];
                outShape = [2, 4];
                break;
            case "Pad":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4]),
                    builder.createConstant("pads", makeTensorProto(DataType.INT64, [4], [1, 1, 1, 1])),
                    builder.createConstant("val", makeTensorProto(DataType.FLOAT, [], [0])),
                ];
                attrs = { mode: "constant" };
                outShape = [4, 6];
                break;
            case "Tile":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4]),
                    builder.createConstant("repeats", makeTensorProto(DataType.INT64, [2], [2, 3])),
                ];
                outShape = [4, 12];
                break;
            case "Expand":
                inputs = [
                    makeInput("X", DataType.FLOAT, [1, 4]),
                    builder.createConstant("shape", makeTensorProto(DataType.INT64, [2], [3, 4])),
                ];
                outShape = [3, 4];
                break;
            case "Clip":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 4]),
                    builder.createConstant("min", makeTensorProto(DataType.FLOAT, [], [0])),
                    builder.createConstant("max", makeTensorProto(DataType.FLOAT, [], [6])),
                ];
                break;
            case "CastLike":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 4]),
                    makeInput("Target", DataType.INT32, [4, 4]),
                ];
                outDtype = DataType.INT32;
                break;
            case "PRelu":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 4]),
                    makeInput("Slope", DataType.FLOAT, [4]),
                ];
                break;
            case "QuantizeLinear":
                if (recipeName === "LowerQuantizeLinearExtended") {
                    inputs = [
                        makeInput("X", DataType.FLOAT, [2, 6]),
                        builder.createConstant("scale", makeTensorProto(DataType.FLOAT, [2, 3], [0.5, 1, 2, 0.5, 1, 2])),
                        builder.createConstant("zp", makeTensorProto(DataType.INT8, [2, 3], [0, 0, 0, 0, 0, 0])),
                    ];
                    attrs = { axis: 1, block_size: 2 };
                    outShape = [2, 6];
                    outDtype = DataType.INT8;
                } else {
                    inputs = [
                        makeInput("X", DataType.FLOAT, [4, 4]),
                        builder.createConstant("scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                        builder.createConstant("zp", makeTensorProto(DataType.UINT8, [], [0])),
                    ];
                    outDtype = DataType.UINT8;
                }
                break;
            case "DequantizeLinear":
                if (recipeName === "LowerDequantizeLinearBlocked") {
                    inputs = [
                        makeInput("X", DataType.INT8, [2, 6]),
                        builder.createConstant("scale", makeTensorProto(DataType.FLOAT, [2, 3], [0.5, 1, 2, 0.5, 1, 2])),
                        builder.createConstant("zp", makeTensorProto(DataType.INT8, [2, 3], [0, 0, 0, 0, 0, 0])),
                    ];
                    attrs = { axis: 1, block_size: 2 };
                    outShape = [2, 6];
                    outDtype = DataType.FLOAT;
                } else {
                    inputs = [
                        makeInput("X", DataType.UINT8, [4, 4]),
                        builder.createConstant("scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                        builder.createConstant("zp", makeTensorProto(DataType.UINT8, [], [0])),
                    ];
                }
                break;
            case "DynamicQuantizeLinear":
                inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
                outputNames = ["y", "y_scale", "y_zero_point"];
                outDtype = DataType.UINT8;
                break;
            case "LpNormalization":
                inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
                attrs =
                    recipeName === "LowerLpNormalizationGeneral"
                        ? { axis: -1, p: 3 }
                        : { axis: -1, p: 2 };
                outShape = [4, 4];
                break;
            case "Det":
                if (recipeName === "LowerDetDynamic") {
                    inputs = [makeInput("X", DataType.FLOAT, [2, 4, 4])];
                    outShape = [2];
                } else {
                    inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
                    outShape = [];
                }
                break;
            case "BatchNormalization":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4, 8, 8]),
                    makeInput("scale", DataType.FLOAT, [4]),
                    makeInput("B", DataType.FLOAT, [4]),
                    makeInput("mean", DataType.FLOAT, [4]),
                    makeInput("var", DataType.FLOAT, [4]),
                ];
                outShape = [2, 4, 8, 8];
                break;
            case "LayerNormalization":
            case "InstanceNormalization":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4, 8, 8]),
                    makeInput("scale", DataType.FLOAT, [4]),
                    makeInput("B", DataType.FLOAT, [4]),
                ];
                attrs = { axis: -1, epsilon: 1e-5 };
                outShape = [2, 4, 8, 8];
                break;
            case "GroupNormalization":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4, 8, 8]),
                    makeInput("scale", DataType.FLOAT, [4]),
                    makeInput("B", DataType.FLOAT, [4]),
                ];
                attrs = { num_groups: 2, epsilon: 1e-5 };
                outShape = [2, 4, 8, 8];
                break;
            case "AveragePool":
            case "MaxPool":
                inputs = [makeInput("X", DataType.FLOAT, [1, 4, 8, 8])];
                attrs = { kernel_shape: [3, 3], strides: [2, 2], pads: [0, 0, 0, 0] };
                outShape = [1, 4, 3, 3];
                if (recipeName === "LowerMaxPoolWithIndices") {
                    outputNames = ["Y", "Indices"];
                }
                break;
            case "LpPool":
                inputs = [makeInput("X", DataType.FLOAT, [1, 4, 8, 8])];
                attrs = { kernel_shape: [2, 2], strides: [2, 2], p: 2 };
                outShape = [1, 4, 4, 4];
                break;
            case "GlobalAveragePool":
            case "GlobalLpPool":
            case "GlobalMaxPool":
                inputs = [makeInput("X", DataType.FLOAT, [1, 4, 8, 8])];
                outShape = [1, 4, 1, 1];
                break;
            case "Softmax":
                inputs = [makeInput("X", DataType.FLOAT, [2, 8])];
                attrs = { axis: -1 };
                outShape = [2, 8];
                break;
            case "NegativeLogLikelihoodLoss":
            case "SoftmaxCrossEntropyLoss":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 5]),
                    makeInput("Target", DataType.INT64, [4]),
                ];
                attrs = { reduction: "mean" };
                outShape = [];
                break;
            case "OneHot":
                inputs = [
                    makeInput("Indices", DataType.INT64, [4]),
                    builder.createConstant("depth", makeTensorProto(DataType.INT64, [], [4])),
                    builder.createConstant("values", makeTensorProto(DataType.FLOAT, [2], [0, 1])),
                ];
                attrs = { axis: -1 };
                outShape = [4, 4];
                break;
            case "TopK":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 8]),
                    builder.createConstant("K", makeTensorProto(DataType.INT64, [1], [3])),
                ];
                attrs = { axis: 1, largest: 1, sorted: 1 };
                outputNames = ["Values", "Indices"];
                outShape = [4, 3];
                break;
            case "RNN":
                inputs = [
                    makeInput("X", DataType.FLOAT, [rnnSeqLen, 2, 4]),
                    makeInput("W", DataType.FLOAT, [1, 5, 4]),
                    makeInput("R", DataType.FLOAT, [1, 5, 5]),
                ];
                attrs = { hidden_size: 5 };
                outputNames = ["Y", "Y_h"];
                outShape = [rnnSeqLen, 1, 2, 5];
                break;
            case "GRU":
                inputs = [
                    makeInput("X", DataType.FLOAT, [rnnSeqLen, 2, 4]),
                    makeInput("W", DataType.FLOAT, [1, 15, 4]),
                    makeInput("R", DataType.FLOAT, [1, 15, 5]),
                ];
                attrs = { hidden_size: 5 };
                outputNames = ["Y", "Y_h"];
                outShape = [rnnSeqLen, 1, 2, 5];
                break;
            case "LSTM":
                inputs = [
                    makeInput("X", DataType.FLOAT, [rnnSeqLen, 2, 4]),
                    makeInput("W", DataType.FLOAT, [1, 20, 4]),
                    makeInput("R", DataType.FLOAT, [1, 20, 5]),
                ];
                attrs = { hidden_size: 5 };
                outputNames = ["Y", "Y_h", "Y_c"];
                outShape = [rnnSeqLen, 1, 2, 5];
                break;
            case "Scan": {
                inputs = [
                    makeInput("State0", DataType.FLOAT, [2, 4]),
                    makeInput("Seq", DataType.FLOAT, [3, 2, 4]),
                ];
                attrs = { num_scan_inputs: 1 };
                outputNames = ["StateOut", "ScanOut"];
                outShape = [2, 4];
                const body = OnnxGraph.create();
                const bb = new GraphBuilder(body, "scan_body");
                const stIn = bb.createTensor("s_in", DataType.FLOAT, [2, 4], "input");
                const xIn = bb.createTensor("x_in", DataType.FLOAT, [2, 4], "input");
                const [stOut] = bb.createOp("Add", [stIn, xIn], {}, [{ type: DataType.FLOAT, shape: [2, 4] }]);
                const [yOut] = bb.createOp("Identity", [stOut], {}, [{ type: DataType.FLOAT, shape: [2, 4] }]);
                (stOut as TensorNode.Class).setType("output");
                (yOut as TensorNode.Class).setType("output");
                regions = [body];
                break;
            }
            case "Einsum":
                inputs = [
                    makeInput("A", DataType.FLOAT, [4, 8]),
                    makeInput("B", DataType.FLOAT, [8, 6]),
                ];
                attrs = { equation: "ij,jk->ik" };
                outShape = [4, 6];
                break;
            case "CumSum":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 8]),
                    builder.createConstant("axis", makeTensorProto(DataType.INT64, [], [1])),
                ];
                outShape = [4, 8];
                break;
            case "GatherElements":
                inputs = [
                    makeInput("Data", DataType.FLOAT, [4, 4]),
                    makeInput("Indices", DataType.INT64, [2, 4]),
                ];
                attrs = { axis: 0 };
                outShape = [2, 4];
                break;
            case "GatherND":
                inputs = [
                    makeInput("Data", DataType.FLOAT, [4, 4]),
                    makeInput("Indices", DataType.INT64, [2, 1]),
                ];
                attrs = { batch_dims: 0 };
                outShape = [2, 4];
                break;
            case "Scatter":
            case "ScatterND":
                inputs = [
                    makeInput("Data", DataType.FLOAT, [4, 4]),
                    makeInput("Indices", DataType.INT64, opType === "ScatterND" ? [2, 1] : [2, 4]),
                    makeInput("Updates", DataType.FLOAT, [2, 4]),
                ];
                if (opType === "Scatter") attrs = { axis: 0 };
                outShape = [4, 4];
                break;
            case "Resize":
                inputs = [
                    makeInput("X", DataType.FLOAT, [1, 2, 4, 4]),
                    builder.createConstant("roi", makeTensorProto(DataType.FLOAT, [0], [])),
                    builder.createConstant("scales", makeTensorProto(DataType.FLOAT, [4], [1, 1, 2, 2])),
                ];
                attrs =
                    recipeName === "LowerResizeCubic"
                        ? { mode: "cubic", cubic_coeff_a: -0.75 }
                        : { mode: "linear" };
                outShape = [1, 2, 8, 8];
                break;
            case "ReverseSequence":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 2, 3]),
                    makeInput("SeqLens", DataType.INT64, [2]),
                ];
                attrs = { batch_axis: 1, time_axis: 0 };
                outShape = [4, 2, 3];
                break;
            case "ReduceSum":
                if (recipeName === "LowerReduceSumNoop") {
                    inputs = [
                        makeInput("X", DataType.FLOAT, [4, 8]),
                        builder.createConstant("axes", makeTensorProto(DataType.INT64, [0], [])),
                    ];
                    attrs = { keepdims: 1, noop_with_empty_axes: 1 };
                    outShape = [4, 8];
                    break;
                }
            // fall through
            case "ReduceMean":
            case "ReduceMax":
            case "ReduceMin":
            case "ReduceProd":
            case "ReduceL1":
            case "ReduceL2":
            case "ReduceLogSum":
            case "ReduceLogSumExp":
            case "ReduceSumSquare":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 8]),
                    builder.createConstant("axes", makeTensorProto(DataType.INT64, [1], [1])),
                ];
                attrs = { keepdims: 1 };
                outShape = [4, 1];
                break;
            case "ArgMax":
            case "ArgMin":
                inputs = [makeInput("X", DataType.FLOAT, [4, 8])];
                attrs = { axis: 1, keepdims: 1 };
                outShape = [4, 1];
                outDtype = DataType.INT64;
                break;
            case "Where":
                inputs = [
                    makeInput("Cond", DataType.BOOL, [4, 4]),
                    makeInput("X", DataType.FLOAT, [4, 4]),
                    makeInput("Y", DataType.FLOAT, [4, 4]),
                ];
                break;
            case "Add":
            case "Sub":
            case "Mul":
            case "Div":
            case "Mod":
            case "Pow":
            case "Min":
            case "Max":
            case "And":
            case "Or":
            case "Xor":
            case "Equal":
            case "Greater":
            case "GreaterOrEqual":
            case "Less":
            case "LessOrEqual": {
                const isBool = opType === "And" || opType === "Or" || opType === "Xor";
                const dt = isBool ? DataType.BOOL : DataType.FLOAT;
                if (
                    opType === "Div" &&
                    (recipeName === "LowerDivByConstant" || recipeName === "LowerDivByExactPowerOfTwo")
                ) {
                    inputs = [
                        makeInput("A", DataType.FLOAT, [4, 4]),
                        builder.createConstant("B_const", makeTensorProto(DataType.FLOAT, [], [2.0])),
                    ];
                } else if (opType === "Pow" && recipeName === "LowerPowSmallInt") {
                    inputs = [
                        makeInput("A", DataType.FLOAT, [4, 4]),
                        builder.createConstant("B_exp", makeTensorProto(DataType.FLOAT, [], [2.0])),
                    ];
                } else {
                    inputs = [
                        makeInput("A", dt, [4, 4]),
                        makeInput("B", dt, [4, 4]),
                    ];
                }
                if (recipeName === "LowerVariadicMin" || recipeName === "LowerVariadicMax") {
                    inputs.push(makeInput("C", dt, [4, 4]));
                }
                if (
                    isBool ||
                    opType === "Equal" ||
                    opType === "Greater" ||
                    opType === "GreaterOrEqual" ||
                    opType === "Less" ||
                    opType === "LessOrEqual"
                ) {
                    outDtype = DataType.BOOL;
                }
                break;
            }
            case "Concat":
                inputs = [
                    makeInput("A", DataType.FLOAT, [2, 4]),
                    makeInput("B", DataType.FLOAT, [2, 4]),
                ];
                attrs = { axis: 0 };
                outShape = [4, 4];
                break;
            case "Mean":
            case "Sum":
                inputs = [
                    makeInput("A", DataType.FLOAT, [4, 4]),
                    makeInput("B", DataType.FLOAT, [4, 4]),
                ];
                outShape = [4, 4];
                break;
            case "Split":
                inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
                attrs = { axis: 0 };
                outputNames = ["Y0", "Y1"];
                outShape = [2, 4];
                break;
            case "Round":
                inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
                break;
            case "RMSNormalization":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 1, 2, 4]),
                    makeInput("Scale", DataType.FLOAT, [4]),
                ];
                attrs = { axis: -1, epsilon: 1e-5 };
                outShape = [2, 1, 2, 4];
                break;
            case "RotaryEmbedding":
                inputs = [
                    makeInput("X", DataType.FLOAT, [1, 2, 4, 4]),
                    makeInput("cos_cache", DataType.FLOAT, [4, 2]),
                    makeInput("sin_cache", DataType.FLOAT, [4, 2]),
                ];
                attrs = { interleaved: 0 };
                outShape = [1, 2, 4, 4];
                break;
            case "Attention":
                inputs = [
                    makeInput("Q", DataType.FLOAT, [1, 2, 4, 4]),
                    makeInput("K", DataType.FLOAT, [1, 2, 4, 4]),
                    makeInput("V", DataType.FLOAT, [1, 2, 4, 4]),
                ];
                attrs = { q_num_heads: 2, kv_num_heads: 2 };
                outShape = [1, 2, 4, 4];
                break;
            case "SpaceToDepth":
                inputs = [makeInput("X", DataType.FLOAT, [1, 4, 4, 4])];
                attrs = { blocksize: 2 };
                outShape = [1, 16, 2, 2];
                break;
            case "TensorScatter":
                inputs = [
                    makeInput("past_cache", DataType.FLOAT, [1, 2, 4, 4]),
                    makeInput("update", DataType.FLOAT, [1, 2, 2, 4]),
                    makeInput("write_indices", DataType.INT64, [1]),
                ];
                attrs = { axis: -2, mode: "circular" };
                outShape = [1, 2, 4, 4];
                break;
            default:
                inputs = [
                    makeInput("X", opType === "Not" ? DataType.BOOL : DataType.FLOAT, [4, 4]),
                ];
                if (opType === "Not" || opType === "IsNaN" || opType === "IsInf") {
                    outDtype = DataType.BOOL;
                }
                break;
        }

        if (regions !== undefined) {
            const scanOp = builder
                .addNode("target_op", new OperationNode.Builder(opType, inputs, attrs, regions));
            for (const inp of inputs) {
                builder.addEdge(inp, scanOp, inp.literalType, inp.shape);
            }
            for (const name of outputNames) {
                const outT = builder.createTensor(name, outDtype, outShape, "output");
                builder.addEdge(scanOp, outT, outDtype, outShape);
            }
        } else {
            builder.createOpWithExact("target_op", opType, inputs, outputNames, attrs);
        }

        try {
            inferShapes(graph);
        } catch {
            // Fallback to static outShape if schema inference throws on synthetic node
        }

        const outputs = graph.getNodeById("target_op")!.outgoers.targets.toArray();
        for (const out of outputs) {
            if (out.is(TensorNode)) {
                out.as(TensorNode).setType("output");
            }
        }
        if (outputs.length > 0 && outputs[0].is(TensorNode)) {
            const inferred = this.extractTensorMeta(outputs[0].as(TensorNode)).shape;
            if (inferred && inferred.length > 0 && !inferred.includes(-1)) {
                outShape = inferred;
            }
        }

        const opNode = graph.getOperationNodes().first()!;
        return {
            graph,
            opNode,
            outShape,
            outDtype,
            probeInputSummary: inputSummaries.join(", "),
        };
    }

    private collectEmittedOpsRecursively(
        graph: OnnxGraph.Class,
        loopDepth: number,
        isLoopLowering: boolean,
        outOps: Set<string>,
        buckets: Map<string, OpQuantityEntry>,
        taintedIndexingValues: Set<string> = new Set(),
    ): void {
        let orderedOps: OperationNode.Class[];
        try {
            orderedOps = topologicalSortOperationNodes(graph);
        } catch {
            orderedOps = Array.from(graph.getOperationNodes());
        }

        for (const op of orderedOps) {
            outOps.add(op.type);

            const { rank, isIndexingOp, outputIds } = this.inspectOpNode(
                op,
                loopDepth,
                isLoopLowering,
                taintedIndexingValues,
            );
            const role = this.classifyRole(op.type, loopDepth, isIndexingOp);
            if (role === "indexing") {
                for (const id of outputIds) {
                    taintedIndexingValues.add(id);
                }
            }

            const inLoop = loopDepth > 0;
            const key = `${op.type}|${rank}|${role}|${inLoop}`;

            const existing = buckets.get(key);
            if (existing) {
                buckets.set(key, { ...existing, count: existing.count + 1 });
            } else {
                buckets.set(key, {
                    opType: op.type,
                    rank,
                    role,
                    count: 1,
                    inLoop,
                    estimated: false,
                });
            }

            for (const region of op.regions) {
                this.collectEmittedOpsRecursively(
                    region,
                    loopDepth + 1,
                    isLoopLowering,
                    outOps,
                    buckets,
                    new Set(taintedIndexingValues),
                );
            }
        }
    }

    private inspectOpNode(
        op: OperationNode.Class,
        loopDepth: number,
        isLoopLowering: boolean,
        taintedIndexingValues: ReadonlySet<string>,
    ): { rank: TensorRankBucket; isIndexingOp: boolean; outputIds: string[] } {
        const outputs = op.outgoers.targets.toArray();
        const inputs = op.incomers.sources.toArray();
        const outputIds = outputs.map((n) => n.id);

        // Polish #2: Reduction ops shrink rank, so inspect inputs before outputs
        const isReduction = op.type.startsWith("Reduce") || REDUCTION_OPS.has(op.type);
        const shapeCandidates = isReduction ? [...inputs, ...outputs] : [...outputs, ...inputs];

        let shape: number[] | undefined;
        for (const candidate of shapeCandidates) {
            const info = this.extractTensorMeta(candidate);
            if (info.shape !== undefined) {
                shape = info.shape;
                break;
            }
        }

        const isIntType = (dt?: DataType): boolean =>
            dt === DataType.INT64 || dt === DataType.INT32;

        const outDtypes = outputs.map((n) => this.extractTensorMeta(n).dtype);
        const hasIntOutput = outDtypes.some(isIntType);

        // Polish #3: Detect AGU wrap booleans (Equal/And on INT64 counters or tainted wrap signals)
        let isIndexingBool = false;
        if (loopDepth > 0 && INDEXING_BOOL_OPS.has(op.type) && inputs.length > 0) {
            const nonConstInputs = inputs.filter((n) => !n.is(ConstantNode));
            if (nonConstInputs.length > 0) {
                isIndexingBool = nonConstInputs.every(
                    (n) =>
                        taintedIndexingValues.has(n.id) ||
                        isIntType(this.extractTensorMeta(n).dtype),
                );
            }
        }

        const isIndexingOp =
            loopDepth > 0 &&
            INDEXING_ARITH_OPS.has(op.type) &&
            (hasIntOutput || isIndexingBool);

        if (shape !== undefined) {
            return {
                rank: this.shapeToBucket(shape, loopDepth),
                isIndexingOp,
                outputIds,
            };
        }

        if (op.type === "Size" || op.type === "Loop" || op.type === "If") {
            return { rank: "0D", isIndexingOp, outputIds };
        }
        if (op.type === "Shape") {
            return { rank: "1D", isIndexingOp: true, outputIds };
        }
        if (loopDepth > 0 && isLoopLowering) {
            return { rank: "0D", isIndexingOp, outputIds };
        }
        return { rank: "unknown", isIndexingOp, outputIds };
    }

    private extractTensorMeta(candidate: GraphNode): { shape?: number[]; dtype?: DataType } {
        let val: ValueNode | undefined;
        if (candidate.is(TensorNode)) val = candidate.as(TensorNode);
        else if (candidate.is(ConstantNode)) val = candidate.as(ConstantNode);
        else if (candidate.is(RegionArgumentNode)) val = candidate.as(RegionArgumentNode);

        if (!val) return {};

        const rawShape = val.shape;
        const dtype = val.literalType;
        if (Array.isArray(rawShape)) {
            return {
                shape: rawShape.map((d) => (typeof d === "number" ? d : -1)),
                dtype,
            };
        }
        return { dtype };
    }

    private shapeToBucket(shape: number[], loopDepth: number): TensorRankBucket {
        if (shape.length === 0) return "0D";
        if (shape.length === 1 && shape[0] === 1 && loopDepth > 0) return "0D";
        if (shape.length === 1) return "1D";
        if (shape.length === 2) return "2D";
        return "ND";
    }

    private classifyRole(opType: string, loopDepth: number, isIndexingOp: boolean): OpRole {
        if (CONTROL_OPS.has(opType)) return "control";
        if (MEMORY_OPS.has(opType)) return "memory";
        if (VIEW_OPS.has(opType)) return "view";
        if (loopDepth > 0 && isIndexingOp) return "indexing";
        return "compute";
    }
}