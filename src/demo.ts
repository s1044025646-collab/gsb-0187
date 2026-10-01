import path from "node:path";
import { Store } from "./store.js";
import { Service } from "./service.js";
import { examples } from "./examples.js";

const DATA_DIR = process.env.DATA_DIR ?? path.join(process.cwd(), "data");
const service = new Service(new Store(path.join(DATA_DIR, "autodiff.db")));
const show = (title: string, x: unknown) => {
  console.log(`\n=== ${title} ===`);
  console.log(JSON.stringify(x, null, 2));
};

// 1. 导入内置算例
for (const def of Object.values(examples)) service.importGraph(def);
console.log("已导入内置算例:", Object.keys(examples).join(", "));

// 2. 一次正确的有限差分校验（mimo 的 f1）
show("正确校验: mimo / f1 @ (0.5, 2, 0.3)",
  service.check("mimo", { x: 0.5, y: 2, z: 0.3 }, "f1"));

// 3. 定义域失败：log(-1)
show("定义域失败: domain_error @ x=-1",
  service.evaluate("domain_error", { x: -1, y: 2, z: 3 }));
show("不可导点: domain_error 梯度 @ z=0 (abs 不可导)",
  service.gradient("domain_error", { x: 1, y: 2, z: 0 }));

// 4. 图版本修改：nonlinear 增加一个输出后形成 v2，对比两版本结果
const v1 = service.store.getGraph("nonlinear", 1)!;
show("nonlinear v1 求值 @ (1, 0.5)", service.evaluate("nonlinear", { x: 1, y: 0.5 }, 1));
const modified = structuredClone(v1.def);
modified.nodes.push({ name: "g2", op: "mul", args: ["g", "g"] });
modified.outputs = { ...modified.outputs, g_sq: "g2" };
service.importGraph(modified);
show("nonlinear v2 求值 @ (1, 0.5)（新增输出 g_sq）",
  service.evaluate("nonlinear", { x: 1, y: 0.5 }, 2));

// 5. 历史报告（模拟重启后读取：重新打开数据库）
const svc2 = new Service(new Store(path.join(DATA_DIR, "autodiff.db")));
const row = svc2.store.getGraph("nonlinear", 1)!;
show("重启后读取 nonlinear v1 的完整历史报告", svc2.store.listResults(row.id));
console.log("\n演示完成。数据库文件:", path.join(DATA_DIR, "autodiff.db"));
