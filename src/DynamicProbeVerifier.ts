import {
    OnnxGraph,
    GraphBuilder,
    TensorNode,
    DataType,
    Recipes,
    makeTensorProto,
    type DecompositionRecipe,
    type ValueNode,
    type AttributeMap,
} from "@specs-feup/onnx-flow";
import type { AnalyzedRecipe } from "./types.js";

export class DynamicProbeVerifier {
    public verifyCanonicalizationRecipes(analyzedRecipes: AnalyzedRecipe[]): void {
        const recipeMap = new Map<string, DecompositionRecipe>();
        for (const R of Object.values(Recipes.canonicalization)) {
            const inst = new R();
            recipeMap.set(inst.name, inst);
        }

        for (const analyzed of analyzedRecipes) {
            const recipe = recipeMap.get(analyzed.name);
            if (!recipe) continue;

            const testOp = analyzed.targetOps[0];
            try {
                const { graph, opNode } = this.buildSyntheticSingleOpGraph(testOp);
                if (!recipe.match(opNode)) {
                    analyzed.dynamicAudit = {
                        testedOp: testOp,
                        succeeded: false,
                        actualProducedOps: [],
                        undeclaredOps: [],
                        unobservedOps: [...analyzed.declaredProducedOps],
                        error: `recipe.match() returned false on standard synthetic ${testOp} node`,
                    };
                    continue;
                }

                const builder = new GraphBuilder(graph, `probe_${recipe.name}`);
                recipe.apply(opNode, builder);
                if (graph.hasNode(opNode.id)) {
                    builder.removeNode(opNode);
                }

                const actualSet = new Set<string>();
                this.collectEmittedOpsRecursively(graph, actualSet);
                const actualProducedOps = Array.from(actualSet).sort();

                const declaredSet = new Set(analyzed.declaredProducedOps);
                const undeclaredOps = actualProducedOps.filter((op) => !declaredSet.has(op));
                const unobservedOps = analyzed.declaredProducedOps.filter(
                    (op) => !actualSet.has(op),
                );

                analyzed.dynamicAudit = {
                    testedOp: testOp,
                    succeeded: true,
                    actualProducedOps,
                    undeclaredOps,
                    unobservedOps,
                };
            } catch (err) {
                analyzed.dynamicAudit = {
                    testedOp: testOp,
                    succeeded: false,
                    actualProducedOps: [],
                    undeclaredOps: [],
                    unobservedOps: [...analyzed.declaredProducedOps],
                    error: err instanceof Error ? err.message : String(err),
                };
            }
        }
    }

    private buildSyntheticSingleOpGraph(opType: string) {
        const graph = OnnxGraph.create();
        const builder = new GraphBuilder(graph, "probe");

        const makeInput = (id: string, dtype: DataType, shape: number[]): ValueNode =>
            builder.createTensor(id, dtype, shape, "input");

        let inputs: ValueNode[] = [];
        let attrs: AttributeMap = {};

        switch (opType) {
            case "Gemm":
                inputs = [
                    makeInput("A", DataType.FLOAT, [4, 8]),
                    makeInput("B", DataType.FLOAT, [8, 6]),
                    makeInput("C", DataType.FLOAT, [6]),
                ];
                attrs = { alpha: 1.0, beta: 1.0, transA: 0, transB: 0 };
                break;
            case "Slice":
                inputs = [
                    makeInput("X", DataType.FLOAT, [4, 8]),
                    builder.createConstant("starts", makeTensorProto(DataType.INT64, [2], [0, 0])),
                    builder.createConstant("ends", makeTensorProto(DataType.INT64, [2], [2, 4])),
                    builder.createConstant("axes", makeTensorProto(DataType.INT64, [2], [0, 1])),
                    builder.createConstant("steps", makeTensorProto(DataType.INT64, [2], [1, 1])),
                ];
                break;
            case "Pad":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4]),
                    builder.createConstant("pads", makeTensorProto(DataType.INT64, [4], [1, 1, 1, 1])),
                    builder.createConstant("val", makeTensorProto(DataType.FLOAT, [], [0])),
                ];
                attrs = { mode: "constant" };
                break;
            case "Expand":
                inputs = [
                    makeInput("X", DataType.FLOAT, [1, 4]),
                    builder.createConstant("shape", makeTensorProto(DataType.INT64, [2], [3, 4])),
                ];
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
                break;
            case "LayerNormalization":
            case "InstanceNormalization":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4, 8, 8]),
                    makeInput("scale", DataType.FLOAT, [4]),
                    makeInput("B", DataType.FLOAT, [4]),
                ];
                attrs = { axis: -1, epsilon: 1e-5 };
                break;
            case "GroupNormalization":
                inputs = [
                    makeInput("X", DataType.FLOAT, [2, 4, 8, 8]),
                    makeInput("scale", DataType.FLOAT, [4]),
                    makeInput("B", DataType.FLOAT, [4]),
                ];
                attrs = { num_groups: 2, epsilon: 1e-5 };
                break;
            case "AveragePool":
                inputs = [makeInput("X", DataType.FLOAT, [1, 4, 8, 8])];
                attrs = { kernel_shape: [3, 3], strides: [2, 2], pads: [0, 0, 0, 0] };
                break;
            case "Round":
                inputs = [makeInput("X", DataType.FLOAT, [4, 4])];
                break;
            case "Sub":
            case "Div":
            case "Mod":
            case "Pow":
            case "Min":
            case "Max":
            case "Or":
            case "GreaterOrEqual":
            case "Less":
            case "LessOrEqual":
                inputs = [
                    makeInput("A", opType === "Or" ? DataType.BOOL : DataType.FLOAT, [4, 4]),
                    makeInput("B", opType === "Or" ? DataType.BOOL : DataType.FLOAT, [4, 4]),
                ];
                break;
            case "Concat":
                inputs = [
                    makeInput("A", DataType.FLOAT, [2, 4]),
                    makeInput("B", DataType.FLOAT, [2, 4]),
                ];
                attrs = { axis: 0 };
                break;
            default:
                inputs = [
                    makeInput("X", opType === "Not" ? DataType.BOOL : DataType.FLOAT, [4, 4]),
                ];
                break;
        }

        builder.createOpWithExact("target_op", opType, inputs, ["Y"], attrs);
        const outputs = graph.getNodeById("target_op")!.outgoers.targets.toArray();
        if (outputs.length > 0 && outputs[0].is(TensorNode)) {
            outputs[0].as(TensorNode).setType("output");
        }

        const opNode = graph.getOperationNodes().first()!;
        return { graph, opNode };
    }

    private collectEmittedOpsRecursively(graph: OnnxGraph.Class, out: Set<string>): void {
        for (const op of graph.getOperationNodes()) {
            out.add(op.type);
            for (const region of op.regions) {
                this.collectEmittedOpsRecursively(region, out);
            }
        }
    }
}