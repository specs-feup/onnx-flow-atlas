import type {
    LibraryAnalysisReport,
    OpQuantityEntry,
    PerOperatorAtlasEntry,
    ScalarReadinessStatus,
} from "../types.js";

export class MarkdownReporter {
    public static generateReport(report: LibraryAnalysisReport): string {
        const ops = Array.from(report.operators.values());
        const nativeCount = ops.filter((o) => o.status === "native-primitive").length;
        const exactCount = ops.filter((o) => o.status === "fully-lowerable-exact").length;
        const fastPathOnlyExactCount = ops.filter(
            (o) => o.status === "fully-lowerable-exact" && o.fastPathOnlyExact === true,
        ).length;
        const generalExactCount = exactCount - fastPathOnlyExactCount;
        const approxCount = ops.filter((o) => o.status === "fully-lowerable-approx").length;
        const blockedCount = ops.filter((o) => o.status === "partial-or-blocked").length;
        const gapCount = ops.filter((o) => o.status === "unhandled-gap").length;

        const decomposableOps = ops
            .filter((o) => o.atlasDetail !== undefined)
            .sort((a, b) => a.opType.localeCompare(b.opType));

        const fullScalarExact = decomposableOps.filter(
            (o) => o.atlasDetail!.scalarReadiness === "full-scalar-exact",
        ).length;
        const fullScalarApprox = decomposableOps.filter(
            (o) => o.atlasDetail!.scalarReadiness === "full-scalar-approx",
        ).length;

        const dynamicShapeCount = decomposableOps.filter(
            (o) => o.bestShapeSupport === "dynamic",
        ).length;
        const dynamicBatchCount = decomposableOps.filter(
            (o) => o.bestShapeSupport === "dynamic-batch",
        ).length;
        const staticOnlyCount = decomposableOps.filter(
            (o) => o.bestShapeSupport === "static",
        ).length;

        // Section 1: Executive Summary
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
            `| **Fully Lowerable (Exact)** | ${exactCount} (${generalExactCount} general, ${fastPathOnlyExactCount} fast-path-only) |`,
            `| **Fully Lowerable (Approximation)** | ${approxCount} |`,
            `| **Partially Lowered (Blocked by Downstream Gaps)** | ${blockedCount} |`,
            `| **Unhandled Gap Operators (No Recipe & Not Native)** | ${gapCount} |`,
            `| **Decomposable Ops Reaching 100% Scalar (\`0D\`) Compute** | ${fullScalarExact + fullScalarApprox} / ${decomposableOps.length} (${fullScalarExact} exact, ${fullScalarApprox} approx) |`,
            `| **Shape Generality (Decomposable Ops)** | ${dynamicShapeCount} dynamic, ${dynamicBatchCount} dynamic-batch, ${staticOnlyCount} static-only |`,
            `| **Algebraic Rewrite Cycles Detected** | ${report.cycles.length} |`,
            `| **Emergent Abstraction Levels** | ${report.hierarchy.maxLevel + 1} strata (Level 0..${report.hierarchy.maxLevel}, ${(report.hierarchy.progressiveRatio * 100).toFixed(0)}% progressive transitions) |`,
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

        // Section 3: Reverse Index — Scalar Lowering Readiness Coverage
        lines.push(
            `## 3. Reverse Index: Scalar (\`0D\`) Lowering Readiness`,
            `Shows whether each decomposable operator reaches a **100% Scalar (\`0D\`) native** terminal set, or which non-native scalar / tensor ops still block it.`,
            ``,
            `| Operator | Scalar Readiness | Best \`0D\` Compute Ratio | Terminal Sets | Blocking Ops / Residual Non-Scalar (Best Set) | Breakdown |`,
            `| :--- | :--- | :---: | :---: | :--- | :---: |`,
        );

        for (const op of decomposableOps) {
            const d = op.atlasDetail!;
            const badge = this.formatReadinessBadge(d.scalarReadiness);
            const pct = `${(d.bestScalarComputeRatio * 100).toFixed(0)}%`;

            const blockers: string[] = [];
            if (d.bestBlockingGaps.length > 0) {
                blockers.push(d.bestBlockingGaps.map((g) => `\`${g}\``).join(", "));
            }
            if (d.bestResidualNonScalar.length > 0) {
                blockers.push(
                    `Non-0D: ${d.bestResidualNonScalar.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
            const blockerCell = blockers.join(" · ") || "None";
            const anchor = `#op-${op.opType.toLowerCase()}`;

            lines.push(
                `| \`${op.opType}\` | ${badge} | **${pct}** | ${d.terminalSets.length}${d.truncatedTerminalSets ? "+" : ""} | ${blockerCell} | [Details](${anchor}) |`,
            );
        }
        lines.push(``);

        // Section 4: Emergent Abstraction Hierarchy (Dynamic MLIR-Style Stratification)
        const h = report.hierarchy;
        lines.push(
            `## 4. Emergent Abstraction Hierarchy (Dynamic Recipe Stratification)`,
            `Levels are computed live via bottom-up DAG stratification over primary-path recipe dependencies (\`Level 0\` = native primitives in \`${report.profile.name}\`, up to \`Level ${h.maxLevel}\`). No operator-to-level mapping is hardcoded.`,
            ``,
            `- **Emergent Depth:** ${h.maxLevel + 1} levels (\`L${h.maxLevel}\` → \`L0\`)`,
            `- **Progressive Lowering Ratio (\`Lk → Lk-1\`):** **${(h.progressiveRatio * 100).toFixed(1)}%**`,
            `- **Loop / Control-Flow Boundary Levels:** ${h.loopBoundaryLevels.map((l) => `\`L${l}\``).join(", ") || "None"}`,
            `- **Unresolved Sink Gaps (outside ladder):** ${h.unresolvedGaps.map((g) => `\`${g}\``).join(", ") || "None"}`,
            ``,
            `### 4.1 Emergent Strata Overview`,
            `| Level | Ops | Auto-Derived Signature | Canon / Loop | Loop Boundary Ops | Avg Transitive Fan-In | Operators |`,
            `| :---: | :---: | :--- | :---: | :---: | :---: | :--- |`,
        );

        for (const stratum of h.levels) {
            const phaseCell =
                stratum.level === 0
                    ? "Native"
                    : `${stratum.canonicalizationCount} / ${stratum.loopLoweringCount}`;
            const opPreview =
                stratum.operators.length <= 12
                    ? stratum.operators.map((o) => `\`${o}\``).join(", ")
                    : `${stratum.operators.slice(0, 10).map((o) => `\`${o}\``).join(", ")} *(+${stratum.operators.length - 10} more)*`;
            lines.push(
                `| **L${stratum.level}** | ${stratum.operators.length} | ${stratum.signature} | ${phaseCell} | ${stratum.controlFlowBoundaryCount} | ${stratum.avgFanIn.toFixed(1)} | ${opPreview} |`,
            );
        }
        lines.push(``);

        if (h.convergenceHubs.length > 0) {
            lines.push(
                `### 4.2 Convergence Hubs (Emergent Intermediate Dialect Ops)`,
                `Intermediate operators (\`0 < level < L${h.maxLevel}\`) ranked by how many higher-level operators transitively lower through them:`,
                ``,
                `| Rank | Operator | Level | Category | Primary Phase | Direct Fan-In | Transitive Fan-In | Fan-Out |`,
                `| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |`,
            );
            h.convergenceHubs.forEach((hub, idx) => {
                lines.push(
                    `| **#${idx + 1}** | \`${hub.opType}\` | **L${hub.level}** | ${hub.category ?? "—"} | ${hub.primaryPhase === "canonicalization" ? "Canon" : "Loop"} | ${hub.directFanIn} | **${hub.transitiveFanIn}** | ${hub.directFanOut} |`,
                );
            });
            lines.push(``);
        }

        if (h.transitions.length > 0) {
            lines.push(
                `### 4.3 Inter-Level Transition Matrix`,
                `| Transition | Style | Edge Count | Sample Source Ops |`,
                `| :---: | :---: | :---: | :--- |`,
            );
            for (const tr of h.transitions) {
                const style = tr.isProgressive
                    ? "Progressive (`-1`)"
                    : `Skip (\`-${tr.fromLevel - tr.toLevel}\`)`;
                lines.push(
                    `| \`L${tr.fromLevel} → L${tr.toLevel}\` | ${style} | **${tr.count}** | ${tr.sampleOps.map((o) => `\`${o}\``).join(", ")} |`,
                );
            }
            lines.push(``);
        }

        // Section 5: Complete Operator Lowering Matrix
        lines.push(
            `## 5. Complete Operator Lowering Matrix`,
            `| Operator | Status | Shape Support | Direct Recipes (Phase / Precision / Priority) | Max Depth | Blocking Leaf Gaps (if any) |`,
            `| :--- | :--- | :---: | :--- | :---: | :--- |`,
        );

        for (const op of ops.sort((a, b) => a.opType.localeCompare(b.opType))) {
            if (op.directRecipes.length === 0 && op.producedByRecipes.length === 0 && !op.isNativePrimitive) {
                continue;
            }
            const recipeBadges =
                op.directRecipes
                    .map(
                        (r) =>
                            `\`${r.name}\` (${r.phase === "canonicalization" ? "Canon" : "Loop"}, ${r.precision === "approximation" ? "≈Approx" : "Exact"}, p=${r.priority}, ${r.shapeSupport})`,
                    )
                    .join("<br>") || "—";

            const shapeCol = op.bestShapeSupport ?? "—";
            const gaps = op.blockingLeafGaps.map((g) => `\`${g}\``).join(", ") || "None";

            lines.push(
                `| \`${op.opType}\` | **${op.status}** | ${shapeCol} | ${recipeBadges} | ${op.maxLoweringDepth} | ${gaps} |`,
            );
        }
        lines.push(``);

        // Section 6: Dynamic Audit Results (if run)
        const audited = report.recipes.filter((r) => r.dynamicAudit !== undefined);
        if (audited.length > 0) {
            lines.push(
                `## 6. Dynamic Synthetic Graph Audit (\`producedOps\` Verification)`,
                `| Recipe | Phase | Tested Op | Execution | Undeclared Ops (Emitted, missing in \`producedOps\`) | Unobserved Ops (Declared, not emitted) |`,
                `| :--- | :---: | :--- | :---: | :--- | :--- |`,
            );
            for (const rec of audited) {
                const a = rec.dynamicAudit!;
                const statusStr = a.succeeded ? "PASS" : `FAIL (${a.error})`;
                const undeclared = a.undeclaredOps.map((o) => `\`${o}\``).join(", ") || "None";
                const unobserved = a.unobservedOps.map((o) => `\`${o}\``).join(", ") || "None";
                lines.push(
                    `| \`${rec.name}\` | ${rec.phase === "canonicalization" ? "Canon" : "Loop"} | \`${a.testedOp}\` | ${statusStr} | ${undeclared} | ${unobserved} |`,
                );
            }
            lines.push(``);
        }

        // Section 6.1 & 6.2: End-to-End Fused Lowering Benchmarks & Hotspots (if run)
        if (report.fusedBenchmarks && report.fusedBenchmarks.length > 0) {
            const sortedBenchmarks = [...report.fusedBenchmarks].sort((a, b) => {
                if (a.succeeded !== b.succeeded) return a.succeeded ? 1 : -1;
                if (b.loopCount !== a.loopCount) return b.loopCount - a.loopCount;
                if (b.totalOps !== a.totalOps) return b.totalOps - a.totalOps;
                return a.opType.localeCompare(b.opType);
            });

            const passedCount = sortedBenchmarks.filter((b) => b.succeeded).length;
            lines.push(
                `### 6.1 End-to-End Fused Lowering Benchmarks (\`Canonicalization + Fused LoopLowering\`)`,
                `Measures actual node counts across **${sortedBenchmarks.length}** decomposable operators (**${passedCount}** succeeded) under profile \`${report.profile.name}\` (\`fuse: true, coalesce: true, optimize: true, indexingMode: Pointer, recurse: ${report.profile.name === "scalar-cgra" ? "true" : "false"}\`), compared against actual \`fuse: false\` loop counts. Sorted descending by fused \`Loop\` count and total fused ops.`,
                ``,
                `| Operator | Status | Fused \`Loop\`s | Max Depth | Unfused \`Loop\`s (\`fuse=false\`) | Fused \`0D\` Compute | Fused Indexing | Fused Control/Mem/View | Infra / 0D | Barriers | Non-Native Leaks | Mode / Schedule Deltas | Total Fused Ops |`,
                `| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :---: |`,
            );

            for (const bm of sortedBenchmarks) {
                if (!bm.succeeded) {
                    lines.push(
                        `| \`${bm.opType}\` | FAIL (${bm.error}) | — | — | — | — | — | — | — | — | — | — | — |`,
                    );
                } else {
                    const unfusedDisplay =
                        bm.unfusedLoopCount !== undefined ? `${bm.unfusedLoopCount}` : "—";
                    const barriersStr =
                        bm.fusionBarrierOps.length > 0
                            ? bm.fusionBarrierOps.map((b) => `\`${b}\``).join(", ")
                            : "None";
                    const leaksStr =
                        bm.unloweredNonNativeOps.length > 0
                            ? bm.unloweredNonNativeOps.map((o) => `\`${o}\``).join(", ")
                            : "None";
                    const variantsStr =
                        bm.variants && bm.variants.length > 0
                            ? bm.variants
                                  .map((v) => {
                                      const dl = v.loopCount - bm.loopCount;
                                      const dt = v.totalOps - bm.totalOps;
                                      const dlStr = dl === 0 ? "0" : dl > 0 ? `+${dl}` : `${dl}`;
                                      const dtStr = dt === 0 ? "0" : dt > 0 ? `+${dt}` : `${dt}`;
                                      return `\`${v.label}\`: ${v.loopCount}L (${dlStr}), ${v.controlMemoryViewCount} infra, ${v.totalOps} ops (${dtStr})`;
                                  })
                                  .join("<br>")
                            : "—";
                    lines.push(
                        `| \`${bm.opType}\` | PASS | **${bm.loopCount}** | ${bm.maxLoopDepth} | ${unfusedDisplay} | ${bm.scalarComputeCount} | ${bm.indexingScalarCount} | **${bm.controlMemoryViewCount}** | ${bm.infraPerComputeRatio.toFixed(2)} | ${barriersStr} | ${leaksStr} | ${variantsStr} | ${bm.totalOps} |`,
                    );
                }
            }
            lines.push(``);

            const okBenchmarks = sortedBenchmarks.filter((b) => b.succeeded);
            if (okBenchmarks.length > 0) {
                const topLoops = [...okBenchmarks]
                    .sort((a, b) => b.loopCount - a.loopCount || b.totalOps - a.totalOps)
                    .slice(0, 15);
                const topTotalOps = [...okBenchmarks]
                    .sort((a, b) => b.totalOps - a.totalOps || b.loopCount - a.loopCount)
                    .slice(0, 15);
                const topInfraRatio = [...okBenchmarks]
                    .filter((b) => b.totalOps > 1 || b.loopCount > 0)
                    .sort(
                        (a, b) =>
                            b.infraPerComputeRatio - a.infraPerComputeRatio ||
                            b.controlMemoryViewCount - a.controlMemoryViewCount,
                    )
                    .slice(0, 15);

                lines.push(
                    `### 6.2 Performance Hotspot Leaderboards (Fused Pipeline)`,
                    `Top outliers in the fused CGRA lowering pipeline highlighting remaining loop-count, node-bloat, and indexing/control overhead pain points.`,
                    ``,
                    `#### Top 15 by Fused \`Loop\` Count`,
                    `| Rank | Operator | Fused \`Loop\`s | Max Depth | Total Fused Ops | Fusion Barriers |`,
                    `| :---: | :--- | :---: | :---: | :---: | :--- |`,
                );
                topLoops.forEach((bm, idx) => {
                    const barriersStr =
                        bm.fusionBarrierOps.length > 0
                            ? bm.fusionBarrierOps.map((b) => `\`${b}\``).join(", ")
                            : "None";
                    lines.push(
                        `| ${idx + 1} | \`${bm.opType}\` | **${bm.loopCount}** | ${bm.maxLoopDepth} | ${bm.totalOps} | ${barriersStr} |`,
                    );
                });
                lines.push(
                    ``,
                    `#### Top 15 by Total Fused Ops`,
                    `| Rank | Operator | Total Fused Ops | Fused \`Loop\`s | \`0D\` Compute | Indexing | Control/Mem/View |`,
                    `| :---: | :--- | :---: | :---: | :---: | :---: | :---: |`,
                );
                topTotalOps.forEach((bm, idx) => {
                    lines.push(
                        `| ${idx + 1} | \`${bm.opType}\` | **${bm.totalOps}** | ${bm.loopCount} | ${bm.scalarComputeCount} | ${bm.indexingScalarCount} | ${bm.controlMemoryViewCount} |`,
                    );
                });
                lines.push(
                    ``,
                    `#### Top 15 by Infra-per-Compute Ratio (\`(Indexing + Control/Mem/View) / max(1, 0D Compute)\`)`,
                    `| Rank | Operator | Infra / 0D | \`0D\` Compute | Indexing | Control/Mem/View | Total Fused Ops |`,
                    `| :---: | :--- | :---: | :---: | :---: | :---: | :---: |`,
                );
                topInfraRatio.forEach((bm, idx) => {
                    lines.push(
                        `| ${idx + 1} | \`${bm.opType}\` | **${bm.infraPerComputeRatio.toFixed(2)}** | ${bm.scalarComputeCount} | ${bm.indexingScalarCount} | ${bm.controlMemoryViewCount} | ${bm.totalOps} |`,
                    );
                });
                lines.push(``);
            }
        }

        // Section 6.2.1: Multi-Op, Dynamic & Edge-Case Scenario Probes
        if (report.scenarioBenchmarks && report.scenarioBenchmarks.length > 0) {
            lines.push(
                `### 6.2.1 Representative Multi-Op, Dynamic & Edge-Case Scenario Probes`,
                `Tests realistic dynamic shapes, fallback branches, and multi-consumer fork patterns missed by isolated single-op static fixtures.`,
                ``,
                `| Scenario | Status | Fused \`Loop\`s | Unfused \`Loop\`s | \`0D\` Compute | Infra Ops | Non-Native Leaks | Mode / Schedule Deltas | Total Ops |`,
                `| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |`,
            );
            for (const bm of report.scenarioBenchmarks) {
                if (!bm.succeeded) {
                    lines.push(`| \`${bm.opType}\` | FAIL (${bm.error}) | — | — | — | — | — | — | — |`);
                } else {
                    const leaksStr =
                        bm.unloweredNonNativeOps.length > 0
                            ? bm.unloweredNonNativeOps.map((o) => `\`${o}\``).join(", ")
                            : "None";
                    const variantsStr =
                        bm.variants && bm.variants.length > 0
                            ? bm.variants
                                  .map((v) => {
                                      const dl = v.loopCount - bm.loopCount;
                                      const dt = v.totalOps - bm.totalOps;
                                      const dlStr = dl === 0 ? "0" : dl > 0 ? `+${dl}` : `${dl}`;
                                      const dtStr = dt === 0 ? "0" : dt > 0 ? `+${dt}` : `${dt}`;
                                      return `\`${v.label}\`: ${v.loopCount}L (${dlStr}), ${v.totalOps} ops (${dtStr})`;
                                  })
                                  .join("<br>")
                            : "—";
                    lines.push(
                        `| \`${bm.opType}\` | PASS | **${bm.loopCount}** | ${bm.unfusedLoopCount ?? "—"} | ${bm.scalarComputeCount} | ${bm.indexingScalarCount + bm.controlMemoryViewCount} | ${leaksStr} | ${variantsStr} | ${bm.totalOps} |`,
                    );
                }
            }
            lines.push(``);
        }

        // Section 6.3: Approximation Root-Cause Roadmap
        if (report.approximationHotspots && report.approximationHotspots.length > 0) {
            lines.push(
                `### 6.3 Approximation Root-Cause Roadmap`,
                `Traces each \`fully-lowerable-approx\` operator back to the non-native leaf primitive(s) making it approximate, identifying single-blocker leverage (operators that immediately become **Exact** if that single root is supported exactly).`,
                ``,
                `| Root Primitive | Approx Class | Approx Recipes | Transitive Dependents | Sole-Blocker Flips to Exact | Sole-Blocker Operators (Sample) | Transitive Sample |`,
                `| :--- | :--- | :--- | :---: | :---: | :--- | :--- |`,
            );
            for (const entry of report.approximationHotspots) {
                const recipesStr =
                    entry.rootRecipes.map((r) => `\`${r}\``).join(", ") || "—";
                const soleSample =
                    entry.soleBlockerOps.length > 0
                        ? entry.soleBlockerOps.slice(0, 6).map((o) => `\`${o}\``).join(", ") +
                          (entry.soleBlockerOps.length > 6
                              ? ` (+${entry.soleBlockerOps.length - 6} more)`
                              : "")
                        : "—";
                const depSample =
                    entry.sampleDependentOps.length > 0
                        ? entry.sampleDependentOps.slice(0, 5).map((o) => `\`${o}\``).join(", ")
                        : "—";
                lines.push(
                    `| \`${entry.rootOp}\` | \`${entry.approxClass}\` | ${recipesStr} | ${entry.transitiveDependentCount} | **${entry.soleBlockerOps.length}** | ${soleSample} | ${depSample} |`,
                );
            }
            lines.push(``);
        }

        // Section 7: Per-Operator Decomposition & Reachable Terminal Sets
        lines.push(
            `## 7. Per-Operator Decomposition & Reachable Terminal Sets`,
            `Detailed text breakdown for all **${decomposableOps.length}** operators with at least one decomposition recipe. Rank tags: \`0D\` (Scalar), \`1D\` (Vector), \`2D\` (Matrix), \`ND\` (3D+ Tensor), \`?\` (Static/Unknown). \`(in Loop)\` marks ops emitted inside loop regions; \`~\` marks static lower-bound counts when no dynamic probe succeeded.`,
            ``,
        );

        for (const op of decomposableOps) {
            lines.push(...this.renderOperatorDetail(op.atlasDetail!));
        }

        // Section 8: Unimplemented Standard ONNX Operators
        const summary = report.unimplementedSummary;
        if (summary.unimplementedCount > 0) {
            const ratioPercent = ((summary.unimplementedCount / summary.totalStandardOps) * 100).toFixed(1);
            lines.push(
                `## 8. Unimplemented Standard ONNX Operators`,
                `**${summary.unimplementedCount} out of ${summary.totalStandardOps}** standard ONNX specification operators (**${ratioPercent}%**) currently have **zero decomposition recipes** and are not marked as native primitives in \`${report.profile.name}\`.`,
                ``,
            );

            for (const [category, opNames] of Object.entries(summary.byCategory).sort()) {
                lines.push(`### Category: ${category} (${opNames.length})`);
                lines.push(`> ${opNames.map((o) => `\`${o}\``).join(", ")}`, ``);
            }
        }

        return lines.join("\n");
    }

    /**
     * Renders a standalone Markdown breakdown for a single operator (usable both in the main report and per-op files).
     */
    public static renderOperatorDetail(detail: PerOperatorAtlasEntry): string[] {
        const lines: string[] = [
            `<a id="op-${detail.opType.toLowerCase()}"></a>`,
            `### \`${detail.opType}\`${detail.category ? ` *(${detail.category})*` : ""} — ${this.formatReadinessBadge(detail.scalarReadiness)}`,
            ``,
            `**Direct Decompositions (${detail.directDecompositions.length})**`,
        ];

        detail.directDecompositions.forEach((d, idx) => {
            const phaseLabel = d.phase === "canonicalization" ? "Canonicalization" : "Loop Lowering";
            const precLabel = d.precision === "approximation" ? "≈Approx" : "Exact";
            const tagsPart =
                d.tags.length > 0 ? ` · Tags: ${d.tags.map((t) => `\`${t}\``).join(", ")}` : "";
            const provLabel =
                d.provenance === "dynamic-probe"
                    ? `Dynamic Probe (${d.probeInputSummary})`
                    : `Static Declared`;

            lines.push(
                `${idx + 1}. **\`${d.recipeName}\`** — *${phaseLabel}, ${precLabel}, p=${d.priority}, ${d.shapeSupport}*${tagsPart} · Source: ${provLabel}`,
            );

            const computeNodes = d.emittedNodes.filter((e) => e.role === "compute");
            const indexingNodes = d.emittedNodes.filter((e) => e.role === "indexing");
            const infraNodes = d.emittedNodes.filter(
                (e) => e.role === "control" || e.role === "memory" || e.role === "view",
            );

            if (computeNodes.length > 0) {
                lines.push(
                    `   - **Compute (${d.scalarComputeCount} scalar \`0D\`, ${d.nonScalarComputeCount} non-scalar):** ${computeNodes.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
            if (indexingNodes.length > 0) {
                lines.push(
                    `   - **Indexing / Address Gen:** ${indexingNodes.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
            if (infraNodes.length > 0) {
                lines.push(
                    `   - **Control / Memory / View:** ${infraNodes.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
        });

        lines.push(
            ``,
            `**Reachable Terminal Operator Sets (${detail.terminalSets.length}${detail.truncatedTerminalSets ? "+ truncated" : ""})**`,
        );

        for (const set of detail.terminalSets) {
            const scalarPct = (set.scalarComputeRatio * 100).toFixed(0);
            const statusTag = set.reachesFullScalar
                ? "100% Scalar Ready"
                : set.blockingGaps.length > 0
                    ? `Blocked by: ${set.blockingGaps.map((g) => `\`${g}\``).join(", ")}`
                    : `Residual Non-Scalar Compute`;

            const r = set.computeByRank;
            const rankSummary = `0D: ${r["0D"]} | 1D: ${r["1D"]} | 2D: ${r["2D"]} | ND: ${r["ND"]}${r.unknown > 0 ? ` | ?: ${r.unknown}` : ""}`;

            lines.push(
                `- **${set.id}** (${set.precision === "approximation" ? "≈Approx" : "Exact"} · **${statusTag}** · Scalar Compute: **${scalarPct}%** [${rankSummary}])`,
            );
            lines.push(
                `  - *Recipes:* ${set.recipeChain.map((rc) => `\`${rc}\``).join(" → ")}`,
            );

            const scalarCompute = set.ops.filter((e) => e.role === "compute" && e.rank === "0D");
            const tensorCompute = set.ops.filter((e) => e.role === "compute" && e.rank !== "0D");
            const indexing = set.ops.filter((e) => e.role === "indexing");
            const infra = set.ops.filter(
                (e) => e.role === "control" || e.role === "memory" || e.role === "view",
            );

            if (scalarCompute.length > 0) {
                lines.push(
                    `  - **Scalar Compute (\`0D\`, ${set.scalarComputeCount}):** ${scalarCompute.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
            if (tensorCompute.length > 0) {
                lines.push(
                    `  - **Non-Scalar Compute (${set.nonScalarComputeCount}):** ${tensorCompute.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
            if (indexing.length > 0) {
                lines.push(
                    `  - **Indexing Scalars (${set.indexingScalarCount}):** ${indexing.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
            if (infra.length > 0) {
                lines.push(
                    `  - **Control / Memory / View (${set.controlMemoryViewCount}):** ${infra.map((e) => this.formatOpChip(e)).join(", ")}`,
                );
            }
        }

        lines.push(``, `---`, ``);
        return lines;
    }

    private static formatOpChip(e: OpQuantityEntry): string {
        const rankLabel = e.rank === "unknown" ? "?" : e.rank;
        const prefix = e.estimated ? "~" : "";
        const loopTag = e.inLoop && e.role !== "control" ? " *(in Loop)*" : "";
        return `\`${e.opType}[${rankLabel}] ×${prefix}${e.count}\`${loopTag}`;
    }

    private static formatReadinessBadge(status: ScalarReadinessStatus): string {
        switch (status) {
            case "full-scalar-exact":
                return "**100% Scalar (Exact)**";
            case "full-scalar-approx":
                return "**100% Scalar (≈Approx)**";
            case "scalar-blocked-non-native":
                return "0D Scalar (Non-Native Gaps)";
            case "partial-scalar-native":
                return "Partial Scalar (Native Vector)";
            case "native-vector-ready":
                return "Native Vector (Ready)";
            case "partial-scalar":
                return "Partial Scalar";
            case "non-scalar-or-blocked":
                return "Non-Scalar / Blocked";
        }
    }
}