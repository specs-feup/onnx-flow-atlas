import type { LibraryAnalysisReport } from "../types.js";

export class MarkdownReporter {
    public static generateReport(report: LibraryAnalysisReport): string {
        const ops = Array.from(report.operators.values());
        const nativeCount = ops.filter((o) => o.status === "native-primitive").length;
        const exactCount = ops.filter((o) => o.status === "fully-lowerable-exact").length;
        const approxCount = ops.filter((o) => o.status === "fully-lowerable-approx").length;
        const blockedCount = ops.filter((o) => o.status === "partial-or-blocked").length;
        const gapCount = ops.filter((o) => o.status === "unhandled-gap").length;

        const lines: string[] = [
            `# ONNX-Flow Decomposition Library Analysis`,
            ``,
            `**Target Hardware Profile:** \`${report.profile.name}\` — *${report.profile.description}*`,
            ``,
            `## 1. Executive Coverage Summary`,
            `| Metric | Count |`,
            `| :--- | :---: |`,
            `| **Total Analyzed Operators** | ${ops.length} |`,
            `| **Total Registered Recipes** | ${report.recipes.length} |`,
            `| **Native Target Primitives** | ${nativeCount} |`,
            `| **Fully Lowerable (Exact)** | ${exactCount} |`,
            `| **Fully Lowerable (Approximation)** | ${approxCount} |`,
            `| **Partially Lowered (Blocked by Downstream Gaps)** | ${blockedCount} |`,
            `| **Unhandled Gap Operators (No Recipe & Not Native)** | ${gapCount} |`,
            `| **Algebraic Rewrite Cycles Detected** | ${report.cycles.length} |`,
            ``,
        ];

        // Section 2: Prioritized Roadmap (What to decompose next!)
        lines.push(`## 2. Priority Roadmap: What to Decompose Next`);
        lines.push(
            `The table below ranks **Unhandled Gap Operators** by how many higher-level operators transitively emit them during decomposition. Implementing a recipe for the top operators here will immediately unblock the largest number of existing recipes.`,
            ``,
        );

        const activeGaps = report.prioritizedGaps.filter(
            (g) => g.transitiveDependentCount > 0 || g.producedByRecipes.length > 0,
        );

        if (activeGaps.length === 0) {
            lines.push(`> All operators emitted by existing recipes are fully covered by \`${report.profile.name}\`!`);
        } else {
            lines.push(
                `| Priority | Missing Operator | Category | Transitive Ops Blocked | Directly Emitted By Recipes |`,
                `| :---: | :--- | :--- | :---: | :--- |`,
            );
            activeGaps.forEach((gap, idx) => {
                lines.push(
                    `| **#${idx + 1}** | \`${gap.opType}\` | ${gap.category ?? "—"} | **${gap.transitiveDependentCount}** | ${gap.producedByRecipes.map((r) => `\`${r}\``).join(", ") || "—"} |`,
                );
            });
        }
        lines.push(``);

        // Section 3: Operator Lowering Matrix
        lines.push(
            `## 3. Complete Operator Lowering Matrix`,
            `| Operator | Status | Direct Recipes (Phase / Precision) | Max Depth | Blocking Leaf Gaps (if any) |`,
            `| :--- | :--- | :--- | :---: | :--- |`,
        );

        for (const op of ops.sort((a, b) => a.opType.localeCompare(b.opType))) {
            if (op.directRecipes.length === 0 && op.producedByRecipes.length === 0 && !op.isNativePrimitive) {
                continue; // Skip unreferenced schema ops to keep table focused
            }
            const recipeBadges =
                op.directRecipes
                    .map(
                        (r) =>
                            `\`${r.name}\` (${r.phase === "canonicalization" ? "Canon" : "Loop"}, ${r.precision === "approximation" ? "≈Approx" : "Exact"})`,
                    )
                    .join("<br>") || "—";

            const gaps = op.blockingLeafGaps.map((g) => `\`${g}\``).join(", ") || "None";

            lines.push(
                `| \`${op.opType}\` | **${op.status}** | ${recipeBadges} | ${op.maxLoweringDepth} | ${gaps} |`,
            );
        }
        lines.push(``);

        // Section 4: Dynamic Audit Results (if run)
        const audited = report.recipes.filter((r) => r.dynamicAudit !== undefined);
        if (audited.length > 0) {
            lines.push(
                `## 4. Dynamic Synthetic Graph Audit (\`producedOps\` Verification)`,
                `| Recipe | Tested Op | Execution | Undeclared Ops (Emitted but missing in \`producedOps\`) | Unobserved Ops (In \`producedOps\` but not emitted) |`,
                `| :--- | :--- | :---: | :--- | :--- |`,
            );
            for (const rec of audited) {
                const a = rec.dynamicAudit!;
                const statusStr = a.succeeded ? "PASS" : `FAIL (${a.error})`;
                const undeclared = a.undeclaredOps.map((o) => `\`${o}\``).join(", ") || "None";
                const unobserved = a.unobservedOps.map((o) => `\`${o}\``).join(", ") || "None";
                lines.push(
                    `| \`${rec.name}\` | \`${a.testedOp}\` | ${statusStr} | ${undeclared} | ${unobserved} |`,
                );
            }
            lines.push(``);
        }

        // Section 5: Unimplemented Standard ONNX Operators (Categorized with Ratio)
        const summary = report.unimplementedSummary;
        if (summary.unimplementedCount > 0) {
            const ratioPercent = ((summary.unimplementedCount / summary.totalStandardOps) * 100).toFixed(1);
            lines.push(
                `## 5. Unimplemented Standard ONNX Operators`,
                `**${summary.unimplementedCount} out of ${summary.totalStandardOps}** standard ONNX specification operators (**${ratioPercent}%**) currently have **zero decomposition recipes** and are not marked as native primitives in \`${report.profile.name}\`.`,
                ``,
            );

            for (const [category, ops] of Object.entries(summary.byCategory).sort()) {
                lines.push(`### Category: ${category} (${ops.length})`);
                const opList = ops.map((op) => `\`${op}\``).join(", ");
                lines.push(`> ${opList}`, ``);
            }
        }

        return lines.join("\n");
    }
}