import type { LibraryAnalysisReport } from "../types.js";

export interface DotExportOptions {
    /** Optional root operator to trace (generates a focused subgraph for just that op) */
    rootOp?: string;
    /** IncludeLoopLowering recipes in the visual graph */
    includeLoopLowering?: boolean;
}

export class DotFormatter {
    public static toDot(report: LibraryAnalysisReport, options: DotExportOptions = {}): string {
        const includeLoops = options.includeLoopLowering ?? true;
        const lines: string[] = [
            "digraph OnnxFlowRecipeAtlas {",
            "  rankdir=LR;",
            "  compound=true;",
            "  splines=true;",
            "  nodesep=0.35;",
            "  ranksep=0.65;",
            '  fontname="Helvetica";',
            '  node [fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=9];',
            "",
            "  // Legend",
            "  subgraph cluster_legend {",
            '    label="Legend (Target: ' + report.profile.name + ')";',
            "    fontsize=11;",
            '    style="rounded,dashed";',
            '    color="#a0aec0";',
            '    leg_prim [label="Native Primitive", shape=box, style="rounded,filled", fillcolor="#c6f6d5", color="#22543d"];',
            '    leg_exact [label="Lowerable Op (Exact)", shape=box, style="rounded,filled", fillcolor="#ebf8ff", color="#2b6cb0"];',
            '    leg_approx [label="Lowerable Op (Approx)", shape=box, style="rounded,filled", fillcolor="#fefcbf", color="#975a16"];',
            '    leg_gap [label="Unhandled Gap Op", shape=box, style="rounded,filled", fillcolor="#fed7d7", color="#9b2c2c", penwidth=2];',
            '    leg_can [label="Canonicalization Recipe", shape=cds, style="filled", fillcolor="#e2e8f0", color="#2b6cb0"];',
            '    leg_loop [label="Loop Lowering Recipe", shape=cds, style="filled", fillcolor="#feebc8", color="#c05621"];',
            "  }",
            "",
        ];

        // Determine which operators and recipes are in scope
        const activeOps = new Set<string>();
        const activeRecipes = new Set<string>();

        if (options.rootOp) {
            const queue = [options.rootOp];
            while (queue.length > 0) {
                const curr = queue.shift()!;
                if (activeOps.has(curr)) continue;
                activeOps.add(curr);

                const opInfo = report.operators.get(curr);
                if (!opInfo) continue;

                for (const rec of opInfo.directRecipes) {
                    if (!includeLoops && rec.phase === "loop-lowering") continue;
                    activeRecipes.add(rec.name);
                    for (const prod of rec.declaredProducedOps) {
                        if (!activeOps.has(prod)) queue.push(prod);
                    }
                }
            }
        } else {
            for (const rec of report.recipes) {
                if (!includeLoops && rec.phase === "loop-lowering") continue;
                activeRecipes.add(rec.name);
                rec.targetOps.forEach((o) => activeOps.add(o));
                rec.declaredProducedOps.forEach((o) => activeOps.add(o));
            }
        }

        // 1. Emit Operator Nodes
        for (const opName of Array.from(activeOps).sort()) {
            const info = report.operators.get(opName);
            let fill = "#ffffff";
            let stroke = "#4a5568";
            let penwidth = 1.2;
            let badge = "";

            if (info) {
                switch (info.status) {
                    case "native-primitive":
                        fill = "#c6f6d5";
                        stroke = "#22543d";
                        badge = "\\n[Native]";
                        break;
                    case "fully-lowerable-exact":
                        fill = "#ebf8ff";
                        stroke = "#2b6cb0";
                        break;
                    case "fully-lowerable-approx":
                        fill = "#fefcbf";
                        stroke = "#975a16";
                        badge = "\\n[Approx Path]";
                        break;
                    case "partial-or-blocked":
                        fill = "#fed7e2";
                        stroke = "#97266d";
                        badge = `\\n[Blocked by ${info.blockingLeafGaps.length} gap(s)]`;
                        break;
                    case "unhandled-gap":
                        fill = "#fed7d7";
                        stroke = "#9b2c2c";
                        penwidth = 2.2;
                        badge = `\\n[GAP: blocks ${info.transitiveDependentCount} op(s)]`;
                        break;
                }
            }

            lines.push(
                `  "op_${opName}" [label="${opName}${badge}", shape=box, style="rounded,filled", fillcolor="${fill}", color="${stroke}", penwidth=${penwidth}];`,
            );
        }

        lines.push("");

        // 2. Emit Recipe Nodes & Edges
        for (const rec of report.recipes) {
            if (!activeRecipes.has(rec.name)) continue;

            const isCanon = rec.phase === "canonicalization";
            const isApprox = rec.precision === "approximation";
            const edgeColor = isCanon ? "#2b6cb0" : "#dd6b20";
            const fillColor = isCanon ? "#ebf8ff" : "#feebc8";
            const edgeStyle = isApprox ? "dashed" : "solid";
            const approxLabel = isApprox ? " (≈)" : "";

            const recId = `rec_${rec.name}`;
            lines.push(
                `  "${recId}" [label="${rec.name}${approxLabel}\\n[${rec.phase} | p=${rec.priority} | ${rec.shapeSupport}]", shape=cds, style="filled${isApprox ? ",dashed" : ""}", fillcolor="${fillColor}", color="${edgeColor}", fontcolor="${edgeColor}"];`,
            );

            for (const targetOp of rec.targetOps) {
                if (!activeOps.has(targetOp)) continue;
                lines.push(
                    `  "op_${targetOp}" -> "${recId}" [color="${edgeColor}", style="${edgeStyle}"];`,
                );
            }

            for (const prodOp of rec.declaredProducedOps) {
                if (!activeOps.has(prodOp)) continue;
                lines.push(
                    `  "${recId}" -> "op_${prodOp}" [color="${edgeColor}", style="${edgeStyle}"];`,
                );
            }
        }

        lines.push("}");
        return lines.join("\n");
    }
}