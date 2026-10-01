// 端到端演示：正确校验、定义域失败、图版本修改比较、重启后读取报告。
// 运行: npm run demo

import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Store } from "../src/db/store.js";
import { Service } from "../src/service.js";
import { GraphDef, GraphError } from "../src/core/types.js";

function hr(title: string): void {
  console.log("\n" + "=".repeat(64));
  console.log(title);
  console.log("=".repeat(64));
}

const shared: GraphDef = {
  name: "demo_poly",
  inputs: ["x", "y"],
  constants: { two: 2, c: 1 },
  nodes: {
    x2: { op: "pow", a: "x", b: "two" },
    xy: { op: "mul", a: "x", b: "y" },
    s: { op: "add", a: "x2", b: "xy" },
    f: { op: "add", a: "s", b: "c" },
  },
  outputs: ["f"],
};

const changed: GraphDef = {
  ...shared,
  constants: { two: 2, c: 10 },
};

const domainBad: GraphDef = {
  name: "demo_log",
  inputs: ["p"],
  constants: {},
  nodes: { neg: { op: "sub", a: "p", b: "p" }, ln: { op: "log", a: "neg" } },
  outputs: ["ln"],
};

function main(): void {
  const dir = mkdtempSync(join(tmpdir(), "autodiff-demo-"));
  const dbPath = join(dir, "demo.db");
  console.log(`数据目录: ${dir}`);

  const store = new Store(dbPath);
  const svc = new Service(store);

  hr("1) 导入图并做中心有限差分校验（应全部通过）");
  const imp = svc.importGraph(shared);
  console.log(`已保存为版本 ${imp.version}（graphId=${imp.graphId}）`);
  const inputs = { x: 2, y: 3 };
  const check = svc.check(shared.name, inputs, { baseStep: 1e-5 }, imp.version);
  for (const e of check.entries) {
    console.log(
      `  ∂f/∂${e.input}: 解析=${e.analytic} 数值=${e.numeric?.toFixed(8)} ` +
        `absErr=${e.absError?.toExponential(2)} relErr=${e.relError?.toExponential(2)} ` +
        `ok=${e.ok}`,
    );
  }
  console.log(`总体 allOk=${check.allOk}`);

  hr("2) 定义域失败：log(0)（必须明确诊断，禁止 NaN/Infinity）");
  svc.importGraph(domainBad);
  try {
    svc.gradient("demo_log", { p: 5 });
  } catch (err) {
    if (err instanceof GraphError) {
      console.log(`  捕获错误码 ${err.code}: ${err.message}`);
    } else {
      throw err;
    }
  }

  hr("3) 修改图（常量 c: 1 -> 10）创建新版本并比较结果");
  const v2 = svc.importGraph(changed);
  console.log(`修改后版本=${v2.version}（旧版本 ${imp.version} 保留）`);
  const r1 = svc.gradient("demo_poly", inputs, 1, false);
  const r2 = svc.gradient("demo_poly", inputs, 2, false);
  console.log(`  v1: f(2,3)=${r1.outputs.f}（c=1）`);
  console.log(`  v2: f(2,3)=${r2.outputs.f}（c=10），梯度不变 ∂f/∂x=${r2.jacobian.f!.x}`);

  store.close();

  hr("4) 模拟重启：重新打开数据库读取历史报告");
  const reopened = new Store(dbPath);
  const graphs = reopened.listGraphs();
  console.log("  图列表:", graphs);
  const versions = reopened
    .listVersions("demo_poly")
    .map((r) => `v${r.version}(id=${r.id})`);
  console.log("  demo_poly 版本:", versions.join(", "));
  const reports = reopened.listReports(imp.graphId);
  console.log(`  版本1持久化报告数: ${reports.length}`);
  for (const rep of reports) {
    const payload = JSON.parse(rep.payload);
    const summary =
      rep.kind === "check"
        ? `allOk=${payload.allOk}`
        : rep.kind === "gradient"
          ? `f=${payload.outputs.f}`
          : JSON.stringify(payload.outputs);
    console.log(`    report#${rep.id} [${rep.kind}] @ ${rep.created_at}: ${summary}`);
  }
  reopened.close();
  console.log("\n演示完成。");
}

main();
