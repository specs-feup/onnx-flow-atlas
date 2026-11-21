import * as OnnxFlow from "@specs-feup/onnx-flow";
import {
    OnnxGraph,
    OperationNode,
    GraphBuilder,
    TensorNode,
    ConstantNode,
    RegionArgumentNode,
    DataType,
    Recipes,
    IndexingMode,
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
    "Constant", "ConstantOfShape", "Shape", "Size", "Transpose",
]);
const INDEXING_ARITH_OPS = new Set([
    "Add", "Sub", "Mul", "Div", "Mod", "Floor", "Cast", "Range",
    "Equal", "Less", "Greater", "And", "Where",
]);
const INDEXING_BOOL_OPS = new Set(["Equal", "Less", "Greater", "And", "Or", "Not"]);
const REDUCTION_OPS = new Set(["ArgMax", "ArgMin"]);

type GraphNode = ReturnType<OperationNode.Class["outgoers"]["targets"]["toArray"]>[number];

export class DynamicProbeVerifier {
    public verifyCanonicalizationRecipes(analyzedRecipes: AnalyzedRecipe[]): void {
        this.verifyAllRecipes(analyzedRecipes);
    }

    public verifyAllRecipes(analyzedRecipes: AnalyzedRecipe[]): void {
        const canonMap = new Map<string, DecompositionRecipe>();
        const loopMap = new Map<string, LoopLoweringRecipe>();

        for (const R of Object.values(Recipes.canonicalization)) {
            const inst = new R();
            canonMap.set(inst.name, inst);
        }
        for (const R of Object.values(Recipes.loopLowering)) {
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
            const { graph, opNode, probeInputSummary } = this.buildSyntheticSingleOpGraph(testOp);
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
                this.buildSyntheticSingleOpGraph(testOp);

            if (!recipe.match(opNode)) {
                return this.makeFailedResult(
                    testOp,
                    probeInputSummary,
                    analyzed.declaredProducedOps,
                    `recipe.match() returned false on standard synthetic ${testOp} node`,
                );
            }

            const outerBuilder = new GraphBuilder(graph, `probe_outer_${recipe.name}_${testOp}`);
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
        const undeclaredOps = actualProducedOps.filter((op) => !declaredSet.has(op));
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

    private buildSyntheticSingleOpGraph(opType: string) {
        const graph = OnnxGraph.create();
        const builder = new GraphBuilder(graph, "probe");

        const inputSummaries: string[] = [];
        const makeInput = (id: string, dtype: DataType, shape: number[]): ValueNode => {
            inputSummaries.push(`${id}:[${shape.join(",")}]`);
            return builder.createTensor(id, dtype, shape, "input");
        };

        let inputs: ValueNode[] = [];
        let attrs: AttributeMap = {};
        let outShape: number[] = [4, 4];
        let outDtype: DataType = DataType.FLOAT;

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
            case "Conv":
                inputs = [
                    makeInput("X", DataType.FLOAT, [1, 3, 8, 8]),
                    makeInput("W", DataType.FLOAT, [4, 3, 3, 3]),
                    makeInput("B", DataType.FLOAT, [4]),
                ];
                attrs = { kernel_shape: [3, 3], strides: [1, 1], pads: [0, 0, 0, 0] };
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
            case "QuantizeLinear":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 4]),
                    builder.createConstant("scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("zp", makeTensorProto(DataType.UINT8, [], [0])),
                ];
                outDtype = DataType.UINT8;
                break;
            case "DequantizeLinear":
                inputs = [
                    makeInput("X", DataType.UINT8, [4, 4]),
                    builder.createConstant("scale", makeTensorProto(DataType.FLOAT, [], [0.1])),
                    builder.createConstant("zp", makeTensorProto(DataType.UINT8, [], [0])),
                ];
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
                break;
            case "Softmax":
                inputs = [makeInput("X", DataType.FLOAT, [2, 8])];
                attrs = { axis: -1 };
                outShape = [2, 8];
                break;
            case "ReduceSum":
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
                inputs = [
                    makeInput("A", dt, [4, 4]),
                    makeInput("B", dt, [4, 4]),
                ];
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
            case "Round":
                inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
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

        builder.createOpWithExact("target_op", opType, inputs, ["Y"], attrs);
        try {
            inferShapes(graph);
        } catch {
            // Fallback to static outShape if schema inference throws on synthetic node
        }

        const outputs = graph.getNodeById("target_op")!.outgoers.targets.toArray();
        if (outputs.length > 0 && outputs[0].is(TensorNode)) {
            const outTensor = outputs[0].as(TensorNode);
            outTensor.setType("output");
            const inferred = this.extractTensorMeta(outTensor).shape;
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