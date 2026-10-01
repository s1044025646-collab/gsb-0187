import path from "node:path";
import { readFileSync } from "node:fs";
import { Store } from "./store.js";
import { Service } from "./service.js";
import { examples } from "./examples.js";

const DATA_DIR = process.env.DATA_DIR ?? path.join(process.cwd(), "data");
const service = new Service(new Store(path.join(DATA_DIR, "autodiff.db")));

function parseFlags(args: string[]): { pos: string[]; flags: Record<string, string> } {
  const pos: string[] = [];
  const flags: Record<string, string> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      flags[args[i].slice(2)] = args[i + 1] ?? "";
      i++;
    } else pos.push(args[i]);
  }
  return { pos, flags };
}

const out = (x: unknown) => console.log(JSON.stringify(x, null, 2));
const readJson = (file: string) => JSON.parse(readFileSync(file, "utf8"));

function toCsv(rows: any[]): string {
  const cols = ["id", "graph_id", "input_set_id", "kind", "created_at", "payload", "diagnostics"];
  const esc = (v: unknown) => `"${String(typeof v === "object" ? JSON.stringify(v) : v).replace(/"/g, '""')}"`;
  const lines = [cols.join(",")];
  for (const r of rows) lines.push(cols.map((c) => esc(r[c])).join(","));
  return lines.join("\n");
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const { pos, flags } = parseFlags(rest);
  const point = flags.point ? JSON.parse(flags.point) : undefined;
  const version = flags.version !== undefined ? Number(flags.version) : undefined;

  switch (cmd) {
    case "import":
      return out(service.importGraph(readJson(pos[0])));
    case "validate":
      return out(service.validate(readJson(pos[0])));
    case "examples": {
      const imported = Object.values(examples).map((d) => service.importGraph(d));
      return out(imported.map((r) => ({ name: r.name, version: r.version })));
    }
    case "list":
      return out(service.store.listGraphs().map((r) => ({ id: r.id, name: r.name, version: r.version, createdAt: r.createdAt })));
    case "eval":
      return out(service.evaluate(pos[0], point, version));
    case "gradient":
      return out(service.gradient(pos[0], point, version, flags.weights ? JSON.parse(flags.weights) : undefined));
    case "jacobian":
      return out(service.jacobian(pos[0], point, version));
    case "directional":
      return out(service.directional(pos[0], point, JSON.parse(flags.direction), version));
    case "check":
      return out(service.check(pos[0], point, flags.output, version,
        flags.step ? Number(flags.step) : undefined,
        flags.steps ? JSON.parse(flags.steps) : undefined));
    case "results": {
      const row = service.store.getGraph(pos[0], version);
      if (!row) throw new Error("图不存在");
      return out(service.store.listResults(row.id));
    }
    case "export": {
      const row = service.store.getGraph(pos[0], version);
      if (!row) throw new Error("图不存在");
      const rows = service.store.listResults(row.id);
      if (flags.format === "csv") return console.log(toCsv(rows));
      return out(rows);
    }
    default:
      console.log(`用法: node dist/src/cli.js <命令> [参数]
  import <file.json>                 导入图（内容变化自动创建新版本）
  validate <file.json>               校验图定义
  examples                           导入全部内置算例
  list                               列出所有图版本
  eval <name> --point '{"x":2}'      求值
  gradient <name> --point ... [--weights '{"f1":1}']
  jacobian <name> --point ...
  directional <name> --point ... --direction '{"x":1}'
  check <name> --output f --point ... [--step 1e-5] [--steps '{"x":1e-4}']
  results <name> [--version N]       查看历史结果
  export <name> --format json|csv    导出结果
环境变量: PORT（服务端口号，默认 8077）、DATA_DIR（数据目录，默认 ./data）`);
  }
}

main().catch((e) => { console.error("错误:", e.message ?? e); process.exit(1); });
