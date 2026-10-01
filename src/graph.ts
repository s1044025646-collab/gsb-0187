import { GraphDef, NodeDef, OpName } from "./types.js";

const OPS: Record<OpName, number> = {
  add: 2, sub: 2, mul: 2, div: 2,
  pow: 1, exp: 1, log: 1, sin: 1, cos: 1, abs: 1,
};

export interface CompiledGraph {
  def: GraphDef;
  inputs: string[];
  constants: Record<string, number>;
  nodes: NodeDef[]; // 拓扑序
  outputs: Record<string, string>;
  /** 每个名字被哪些节点引用（反向传播用） */
  consumers: Map<string, number[]>;
}

export function compileGraph(def: GraphDef): CompiledGraph {
  if (!def || typeof def !== "object") throw new Error("图定义必须是 JSON 对象");
  const inputs = def.inputs ?? [];
  const constants = def.constants ?? {};
  const nodes = def.nodes ?? [];
  const outputs = def.outputs ?? {};
  if (!Array.isArray(inputs) || inputs.length === 0) throw new Error("inputs 必须是非空数组");
  if (!Array.isArray(nodes)) throw new Error("nodes 必须是数组");
  if (typeof outputs !== "object" || Object.keys(outputs).length === 0)
    throw new Error("outputs 必须是非空对象");

  const names = new Set<string>();
  for (const n of inputs) {
    if (names.has(n)) throw new Error(`名称重复: "${n}"（输入/常量/节点名必须全局唯一）`);
    names.add(n);
  }
  for (const k of Object.keys(constants)) {
    if (names.has(k)) throw new Error(`名称重复: "${k}"`);
    names.add(k);
    if (typeof constants[k] !== "number" || !Number.isFinite(constants[k]))
      throw new Error(`常量 "${k}" 必须是有限数值`);
  }
  for (const nd of nodes) {
    if (names.has(nd.name)) throw new Error(`名称重复: "${nd.name}"`);
    names.add(nd.name);
    if (!(nd.op in OPS)) throw new Error(`节点 "${nd.name}" 的算子未知: ${nd.op}`);
    const arity = OPS[nd.op];
    if (!Array.isArray(nd.args) || nd.args.length !== arity)
      throw new Error(`节点 "${nd.name}" 算子 ${nd.op} 需要 ${arity} 个参数`);
    if (nd.op === "pow" && !Number.isInteger(nd.exponent))
      throw new Error(`节点 "${nd.name}" 的 pow 指数必须是整数`);
  }
  const defined = new Set([...inputs, ...Object.keys(constants)]);
  for (const nd of nodes) {
    for (const a of nd.args) {
      if (!defined.has(a))
        throw new Error(`节点 "${nd.name}" 引用了未定义或后定义的名字: "${a}"（存在环或悬空引用）`);
    }
    defined.add(nd.name);
  }
  for (const [outName, ref] of Object.entries(outputs)) {
    if (!defined.has(ref)) throw new Error(`输出 "${outName}" 引用了未定义的名字: "${ref}"`);
  }
  detectCycle(nodes);

  const consumers = new Map<string, number[]>();
  nodes.forEach((nd, i) => {
    for (const a of nd.args) {
      if (!consumers.has(a)) consumers.set(a, []);
      consumers.get(a)!.push(i);
    }
  });
  return { def, inputs, constants, nodes, outputs, consumers };
}

function detectCycle(nodes: NodeDef[]): void {
  const index = new Map(nodes.map((n, i) => [n.name, i]));
  const state = new Array(nodes.length).fill(0);
  const visit = (i: number): void => {
    if (state[i] === 1) throw new Error(`检测到环，涉及节点 "${nodes[i].name}"`);
    if (state[i] === 2) return;
    state[i] = 1;
    for (const a of nodes[i].args) {
      const j = index.get(a);
      if (j !== undefined) visit(j);
    }
    state[i] = 2;
  };
  for (let i = 0; i < nodes.length; i++) visit(i);
}
