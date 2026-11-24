import type { LibraryAnalysisReport } from "../types.js";

export interface DotExportOptions {
    /** Optional root operator to trace (generates a focused subgraph for just that op) */
    rootOp?: string;
    /** IncludeLoopLowering recipes in the visual graph */
    includeLoopLowering?: boolean;
    /** Render a top-down layered graph grouped by emergent abstraction level (L_max -> L0) */
    groupByLevel?: boolean;
}

export class DotFormatter {
    public static toDot(report: LibraryAnalysisReport, options: DotExportOptions = {}): string {
        if (options.groupByLevel) {
            return this.toHierarchyDot(report);
        }
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

    private static toHierarchyDot(report: LibraryAnalysisReport): string {
        const h = report.hierarchy;
        const lines: string[] = [
            "digraph OnnxFlowAbstractionHierarchy {",
            "  rankdir=TB;",
            "  compound=true;",
            "  splines=true;",
            "  nodesep=0.25;",
            "  ranksep=0.75;",
            '  fontname="Helvetica";',
            '  node [fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=8];',
            "",
        ];

        const hubSet = new Set(h.convergenceHubs.slice(0, 8).map((x) => x.opType));

        // 1. Emit one cluster per emergent stratum (top-down: L_max -> L0)
        for (const stratum of h.levels) {
            const isLeaf = stratum.level === 0;
            const clusterFill = isLeaf ? "#f0fff4" : "#f7fafc";
            const clusterStroke = isLeaf ? "#22543d" : "#4a5568";

            lines.push(`  subgraph cluster_L${stratum.level} {`);
            lines.push(
                `    label="Level ${stratum.level} (${stratum.operators.length} ops) — ${stratum.signature}";`,
            );
            lines.push(`    style="rounded,filled";`);
            lines.push(`    fillcolor="${clusterFill}";`);
            lines.push(`    color="${clusterStroke}";`);
            lines.push(`    fontsize=11;`);

            for (const opName of stratum.operators) {
                const info = report.operators.get(opName);
                const m = info?.hierarchyMetrics;
                const isHub = hubSet.has(opName);
                const fill = isLeaf
                    ? "#c6f6d5"
                    : isHub
                      ? "#fefcbf"
                      : m?.crossesIntoLoop
                        ? "#feebc8"
                        : "#ebf8ff";
                const stroke = isLeaf
                    ? "#22543d"
                    : isHub
                      ? "#975a16"
                      : m?.crossesIntoLoop
                        ? "#c05621"
                        : "#2b6cb0";
                const penwidth = isHub ? 2.2 : 1.2;
                const hubTag = isHub ? `\\n[Hub: ${m?.transitiveFanIn}↑]` : "";

                lines.push(
                    `    "op_${opName}" [label="${opName}${hubTag}", shape=box, style="rounded,filled", fillcolor="${fill}", color="${stroke}", penwidth=${penwidth}];`,
                );
            }
            lines.push("  }", "");
        }

        // 2. Emit primary-path lowering edges between operators
        const activeOps = new Set(h.levels.flatMap((s) => s.operators));
        for (const opName of activeOps) {
            const info = report.operators.get(opName);
            const m = info?.hierarchyMetrics;
            if (!m || m.level <= 0 || !m.primaryRecipe) continue;

            const rec = info?.directRecipes.find((r) => r.name === m.primaryRecipe);
            if (!rec) continue;

            const probe =
                rec.dynamicAuditsByOp?.[opName] ??
                (rec.dynamicAudit?.testedOp === opName ? rec.dynamicAudit : undefined);
            const produced =
                probe && probe.succeeded && probe.actualProducedOps.length > 0
                    ? probe.actualProducedOps
                    : rec.declaredProducedOps;

            const color = rec.phase === "canonicalization" ? "#2b6cb0" : "#dd6b20";
            const style = rec.precision === "approximation" ? "dashed" : "solid";

            for (const target of new Set(produced)) {
                if (target === opName || !activeOps.has(target)) continue;
                lines.push(
                    `  "op_${opName}" -> "op_${target}" [color="${color}", style="${style}"];`,
                );
            }
        }

        lines.push("}");
        return lines.join("\n");
    }
}