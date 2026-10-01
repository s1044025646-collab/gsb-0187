#!/usr/bin/env node
// 命令行接口：导入 / 验证 / 求值 / 梯度 / Jacobian / 方向导数 / 校验 / 导出。

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadConfig } from "./config.js";
import { Store } from "./db/store.js";
import { Service } from "./service.js";
import { GraphDef, GraphError, ValueMap } from "./core/types.js";
import {
  checkToCsv,
  gradientToCsv,
  valuesToCsv,
} from "./export/csv.js";
import { startServer } from "./api/server.js";

interface ParsedArgs {
  command: string;
  positional: string[];
  flags: Record<string, string>;
}

function parseArgs(argv: string[]): ParsedArgs {
  const [command = "", ...rest] = argv;
  const positional: string[] = [];
  const flags: Record<string, string> = {};
  for (let i = 0; i < rest.length; i++) {
    const token = rest[i]!;
    if (token.startsWith("--")) {
      const key = token.slice(2);
      const next = rest[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = "true";
      }
    } else {
      positional.push(token);
    }
  }
  return { command, positional, flags };
}

function readJsonFile<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(path), "utf-8")) as T;
}

function parseInlineJson(pathOrJson: string): ValueMap {
  if (pathOrJson.startsWith("{") || pathOrJson.startsWith("[")) {
    return JSON.parse(pathOrJson) as ValueMap;
  }
  return readJsonFile<ValueMap>(pathOrJson);
}

function emit(payload: unknown, format: string | undefined, out?: string): void {
  let text: string;
  if (format === "csv") {
    const obj = payload as Record<string, unknown>;
    if (obj.jacobian) text = gradientToCsv(JSON.parse(JSON.stringify(obj)));
    else if (obj.entries) text = checkToCsv(JSON.parse(JSON.stringify(obj)));
    else if (obj.outputs) text = valuesToCsv(obj.outputs as ValueMap);
    else throw new Error("该结果不支持 CSV 导出");
  } else {
    text = JSON.stringify(payload, null, 2);
  }
  if (out) writeFileSync(resolve(out), text, "utf-8");
  else process.stdout.write(text + "\n");
}

const HELP = `
计算图自动微分 CLI

用法: autodiff <命令> [参数]

命令:
  import  --file <graph.json>               导入图（相同定义复用版本，修改创建新版本）
  list                                      列出所有图与最新版本
  versions --name <图名>                    列出某图全部版本
  validate --name <图名> [--version N]      校验图（引用 / 环 / 唯一性）
  eval --name <图名> --inputs <f|'{...}'> [--version N]
  gradient --name <图名> --inputs <f|'{...}'> [--version N]
  directional --name <图名> --inputs <...> --direction '{...}' [--version N]
  vjp --name <图名> --inputs <...> --weights '{...}' [--version N]
  check --name <图名> --inputs <...> [--step 1e-5] [--per-input-step '{...}'] [--version N]
  report --id <报告ID> [--format csv]       读取持久化报告（重启后仍可用）

通用选项:
  --format csv           以 CSV 输出（gradient/check/eval/report）
  --out <file>           将输出写入文件
  --data-dir <dir>       覆盖数据目录（默认 data/）
  --port <N>             仅 serve 命令使用

  serve                   启动 HTTP API（端口可用 --port 或 PORT 环境变量）
`;

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const config = loadConfig();
  if (args.flags["data-dir"]) config.dataDir = resolve(args.flags["data-dir"]);
  const dbPath =
    args.flags["data-dir"] !== undefined
      ? resolve(args.flags["data-dir"], "autodiff.db")
      : config.dbPath;

  if (args.command === "serve") {
    const store = new Store(dbPath);
    const port = Number(args.flags.port ?? process.env.PORT ?? 3000);
    void startServer(new Service(store), port);
    return;
  }

  const store = new Store(dbPath);
  const service = new Service(store);

  try {
    switch (args.command) {
      case "import": {
        const def = readJsonFile<GraphDef>(required(args.flags.file, "--file"));
        console.log(JSON.stringify(service.importGraph(def), null, 2));
        break;
      }
      case "list":
        console.log(JSON.stringify(service.getStore().listGraphs(), null, 2));
        break;
      case "versions": {
        const name = required(args.flags.name, "--name");
        const rows = service
          .getStore()
          .listVersions(name)
          .map((r) => ({ id: r.id, version: r.version, createdAt: r.created_at }));
        console.log(JSON.stringify(rows, null, 2));
        break;
      }
      case "validate": {
        const name = required(args.flags.name, "--name");
        console.log(
          JSON.stringify(
            service.validateOnly(name, numFlag(args.flags.version)),
            null,
            2,
          ),
        );
        break;
      }
      case "eval":
      case "gradient":
      case "directional":
      case "vjp":
      case "check": {
        const name = required(args.flags.name, "--name");
        const inputs = parseInlineJson(required(args.flags.inputs, "--inputs"));
        const version = numFlag(args.flags.version);
        let payload: unknown;
        if (args.command === "eval") {
          payload = service.eval_(name, inputs, version);
        } else if (args.command === "gradient") {
          payload = service.gradient(name, inputs, version);
        } else if (args.command === "directional") {
          payload = service.directional_(
            name,
            inputs,
            parseInlineJson(required(args.flags.direction, "--direction")),
            version,
          );
        } else if (args.command === "vjp") {
          payload = service.vjp(
            name,
            inputs,
            parseInlineJson(required(args.flags.weights, "--weights")),
            version,
          );
        } else {
          payload = service.check(
            name,
            inputs,
            {
              baseStep: args.flags.step ? Number(args.flags.step) : undefined,
              perInputStep: args.flags["per-input-step"]
                ? JSON.parse(args.flags["per-input-step"])
                : undefined,
            },
            version,
          );
        }
        emit(payload, args.flags.format, args.flags.out);
        if (args.flags.format === undefined && !args.flags.out) {
          // emit already printed
        }
        break;
      }
      case "report": {
        const id = Number(required(args.flags.id, "--id"));
        const row = service.getStore().getReport(id);
        if (!row) throw new Error(`报告不存在: ${id}`);
        const payload = {
          id: row.id,
          graphId: row.graph_id,
          kind: row.kind,
          createdAt: row.created_at,
          payload: JSON.parse(row.payload),
        };
        if (args.flags.format === "csv") emit(payload.payload, "csv", args.flags.out);
        else emit(payload, undefined, args.flags.out);
        break;
      }
      case "help":
      case "--help":
      case "-h":
      case "":
        process.stdout.write(HELP);
        break;
      default:
        console.error(`未知命令: ${args.command}\n${HELP}`);
        process.exitCode = 2;
    }
  } catch (err) {
    if (err instanceof GraphError) {
      console.error(
        JSON.stringify(
          { error: { code: err.code, message: err.message, detail: err.detail ?? null } },
          null,
          2,
        ),
      );
    } else {
      console.error((err as Error).message);
    }
    process.exitCode = 1;
  } finally {
    store.close();
  }
}

function required(value: string | undefined, name: string): string {
  if (value === undefined) throw new Error(`缺少必需参数 ${name}`);
  return value;
}

function numFlag(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const n = Number(value);
  if (!Number.isInteger(n)) throw new Error(`版本号必须是整数: ${value}`);
  return n;
}

main();
