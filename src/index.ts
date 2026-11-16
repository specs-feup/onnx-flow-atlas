#!/usr/bin/env node
import fs from "fs";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { graphviz } from "node-graphviz";
import { BUILTIN_PROFILES } from "./types.js";
import { RecipeGraphAnalyzer } from "./RecipeGraphAnalyzer.js";
import { DynamicProbeVerifier } from "./DynamicProbeVerifier.js";
import { DotFormatter } from "./formatters/DotFormatter.js";
import { MarkdownReporter } from "./formatters/MarkdownReporter.js";

export async function runCli(rawArgs: string[] = hideBin(process.argv)): Promise<void> {
    const argv = await yargs(rawArgs)
        .usage("Usage: onnx-flow-analyze [options]")
        .option("profile", {
            alias: "p",
            describe: "Target hardware primitive profile",
            choices: Object.keys(BUILTIN_PROFILES),
            default: "scalar-cgra",
        })
        .option("op", {
            describe: "Focus DOT graph on a single root operator's transitive lowering tree (e.g., Softmax, LayerNormalization)",
            type: "string",
        })
        .option("noLoopLowering", {
            describe: "Exclude loop-lowering recipes from the DOT visualization (show only algebraic canonicalization)",
            type: "boolean",
            default: false,
        })
        .option("verifyDynamic", {
            alias: "vd",
            describe: "Run synthetic 1-op graphs through recipes to verify declared producedOps",
            type: "boolean",
            default: true,
        })
        .option("report", {
            alias: "r",
            describe: "Output path for the Markdown analysis report (e.g., analysis.md)",
            type: "string",
        })
        .option("dot", {
            alias: "d",
            describe: "Output path for the Graphviz DOT file (e.g., recipes.dot)",
            type: "string",
        })
        .option("svg", {
            alias: "s",
            describe: "Render the Graphviz DOT graph directly to an SVG file (e.g., recipes.svg)",
            type: "string",
        })
        .help("help")
        .alias("help", "h")
        .parseAsync();

    const profile = BUILTIN_PROFILES[argv.profile];
    const analyzer = new RecipeGraphAnalyzer();
    const recipes = analyzer.loadRecipes();

    if (argv.verifyDynamic) {
        const verifier = new DynamicProbeVerifier();
        verifier.verifyCanonicalizationRecipes(recipes);
    }

    const report = analyzer.analyze(profile, recipes);
    const mdContent = MarkdownReporter.generateReport(report);

    if (argv.report) {
        fs.writeFileSync(argv.report, mdContent, "utf8");
        console.log(`Markdown analysis report written to ${argv.report}`);
    } else {
        console.log(mdContent);
    }

    if (argv.dot || argv.svg) {
        const dotContent = DotFormatter.toDot(report, {
            rootOp: argv.op,
            includeLoopLowering: !argv.noLoopLowering,
        });

        if (argv.dot) {
            fs.writeFileSync(argv.dot, dotContent, "utf8");
            console.log(`DOT graph written to ${argv.dot}`);
        }

        if (argv.svg) {
            const svgContent = await graphviz.layout(dotContent, "svg", "dot");
            fs.writeFileSync(argv.svg, svgContent, "utf8");
            console.log(`SVG diagram rendered to ${argv.svg}`);
        }
    }
}

runCli().catch((err) => {
    console.error(err);
    process.exit(1);
});