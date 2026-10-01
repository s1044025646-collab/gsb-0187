import {
  BINARY_OPS,
  ErrorCode,
  GraphDef,
  GraphError,
  UNARY_OPS,
} from "./types.js";

export interface ValidatedGraph {
  def: GraphDef;
  constants: Record<string, number>;
  // 拓扑序：所有节点（不含常量/输入）的求值顺序。
  order: string[];
  // 某个名字是否为输入/常量。
  inputSet: Set<string>;
  constSet: Set<string>;
}

function fail(
  code: ErrorCode,
  msg: string,
  detail?: Record<string, unknown>,
): never {
  throw new GraphError(code, msg, detail);
}

// 校验图并返回拓扑序。输入集与节点集（含常量名）全局唯一。
export function validateGraph(def: GraphDef): ValidatedGraph {
  if (!def || typeof def !== "object") {
    fail("GRAPH_INVALID", "图定义必须是 JSON 对象");
  }
  if (!def.name || typeof def.name !== "string") {
    fail("GRAPH_INVALID", "图必须包含字符串类型的 name");
  }
  if (!Array.isArray(def.inputs)) {
    fail("GRAPH_INVALID", "inputs 必须是字符串数组");
  }
  if (
    !def.nodes ||
    typeof def.nodes !== "object" ||
    Array.isArray(def.nodes)
  ) {
    fail("GRAPH_INVALID", "nodes 必须是对象（名字 -> 节点）");
  }
  if (!Array.isArray(def.outputs) || def.outputs.length === 0) {
    fail("GRAPH_INVALID", "outputs 必须是非空字符串数组");
  }

  const constants = def.constants ?? {};
  const inputSet = new Set<string>();
  const constSet = new Set<string>();
  const nodeNames = new Set<string>();

  for (const name of def.inputs) {
    if (typeof name !== "string" || name.length === 0) {
      fail("GRAPH_INVALID", "输入名必须是非空字符串", { name: String(name) });
    }
    if (inputSet.has(name)) {
      fail("DUPLICATE_NAME", `输入名重复: ${name}`, { name });
    }
    inputSet.add(name);
  }

  for (const [name, value] of Object.entries(constants)) {
    if (inputSet.has(name)) {
      fail("DUPLICATE_NAME", `常量名与输入冲突: ${name}`, { name });
    }
    if (typeof value !== "number" || !Number.isFinite(value)) {
      fail("GRAPH_INVALID", `常量必须是有限数: ${name}`, { name, value });
    }
    constSet.add(name);
  }

  for (const [name, node] of Object.entries(def.nodes)) {
    if (inputSet.has(name) || constSet.has(name)) {
      fail("DUPLICATE_NAME", `节点名与输入/常量冲突: ${name}`, { name });
    }
    if (nodeNames.has(name)) {
      fail("DUPLICATE_NAME", `节点名重复: ${name}`, { name });
    }
    nodeNames.add(name);

    if (!node || typeof node !== "object" || typeof node.op !== "string") {
      fail("GRAPH_INVALID", `节点缺少 op: ${name}`, { name });
    }
    const op = node.op;
    if (BINARY_OPS.has(op)) {
      if (typeof node.a !== "string" || typeof node.b !== "string") {
        fail(
          "GRAPH_INVALID",
          `二元算子 ${op} 需要字符串操作数 a 与 b: ${name}`,
          { name, op },
        );
      }
    } else if (UNARY_OPS.has(op)) {
      if (typeof node.a !== "string") {
        fail("GRAPH_INVALID", `一元算子 ${op} 需要操作数 a: ${name}`, {
          name,
          op,
        });
      }
    } else {
      fail("GRAPH_INVALID", `未知算子: ${String(op)}`, { name, op });
    }
  }

  const known = (n: string) =>
    inputSet.has(n) || constSet.has(n) || nodeNames.has(n);

  for (const [name, node] of Object.entries(def.nodes)) {
    if (!known(node.a)) {
      fail("UNKNOWN_REFERENCE", `节点 ${name} 的操作数 a 未定义: ${node.a}`, {
        node: name,
        operand: node.a,
      });
    }
    if (typeof node.b === "string" && !known(node.b)) {
      fail("UNKNOWN_REFERENCE", `节点 ${name} 的操作数 b 未定义: ${node.b}`, {
        node: name,
        operand: node.b,
      });
    }
  }

  for (const out of def.outputs) {
    if (!known(out)) {
      fail("UNKNOWN_REFERENCE", `输出引用未定义: ${out}`, {
        output: out,
      });
    }
  }

  const order = topoSort(def, nodeNames);

  return { def, constants, order, inputSet, constSet };
}

// Kahn 拓扑排序；存在环时给出明确诊断。
function topoSort(def: GraphDef, nodeNames: Set<string>): string[] {
  const indeg = new Map<string, number>();
  const deps = new Map<string, string[]>();
  for (const name of nodeNames) indeg.set(name, 0);

  for (const [name, node] of Object.entries(def.nodes)) {
    const list: string[] = [];
    if (nodeNames.has(node.a)) list.push(node.a);
    if (typeof node.b === "string" && nodeNames.has(node.b)) list.push(node.b);
    deps.set(name, list);
    indeg.set(name, list.length);
  }

  const ready: string[] = [];
  for (const [name, d] of indeg) if (d === 0) ready.push(name);
  ready.sort();

  const order: string[] = [];
  const dependents = new Map<string, string[]>();
  for (const [name, list] of deps) {
    for (const dep of list) {
      const arr = dependents.get(dep) ?? [];
      arr.push(name);
      dependents.set(dep, arr);
    }
  }

  while (ready.length > 0) {
    const cur = ready.shift()!;
    order.push(cur);
    for (const child of dependents.get(cur) ?? []) {
      const d = (indeg.get(child) ?? 0) - 1;
      indeg.set(child, d);
      if (d === 0) ready.push(child);
    }
  }

  if (order.length !== nodeNames.size) {
    const stuck = [...indeg.entries()]
      .filter(([, d]) => d > 0)
      .map(([n]) => n)
      .sort();
    throw new GraphError(
      "CYCLE_DETECTED",
      `检测到依赖环，涉及节点: ${stuck.join(", ")}`,
      { nodes: stuck },
    );
  }
  return order;
}
