import { CompiledGraph } from "./graph.js";
import { Diagnostic, EvalError } from "./types.js";

const EXP_ARG_MAX = 709.0; // exp(709) 有限，exp(710) 溢出为 Infinity

/** 前向求值：返回所有名字 -> 数值。遇定义域错误抛 EvalError。 */
export function forward(
  g: CompiledGraph,
  point: Record<string, number>
): Map<string, number> {
  const values = new Map<string, number>();
  for (const name of g.inputs) {
    const v = point[name];
    if (typeof v !== "number" || !Number.isFinite(v))
      throw new EvalError({ code: "FD_EVAL_FAILED", message: `输入 "${name}" 缺失或非有限数值` });
    values.set(name, v);
  }
  for (const [k, v] of Object.entries(g.constants)) values.set(k, v);
  for (const nd of g.nodes) {
    const a = nd.args.map((n) => values.get(n)!);
    values.set(nd.name, applyOp(nd.op, a, nd.exponent, nd.name));
  }
  return values;
}

function applyOp(op: string, a: number[], exponent: number | undefined, node: string): number {
  switch (op) {
    case "add": return a[0] + a[1];
    case "sub": return a[0] - a[1];
    case "mul": return a[0] * a[1];
    case "div":
      if (a[1] === 0)
        throw new EvalError({ code: "DIV_ZERO", message: `节点 "${node}" 除数为零`, node });
      return a[0] / a[1];
    case "pow": return Math.pow(a[0], exponent!);
    case "exp":
      if (a[0] > EXP_ARG_MAX)
        throw new EvalError({ code: "EXP_OVERFLOW", message: `节点 "${node}" exp 参数 ${a[0]} 溢出`, node });
      return Math.exp(a[0]);
    case "log":
      if (a[0] <= 0)
        throw new EvalError({ code: "LOG_NONPOSITIVE", message: `节点 "${node}" log 输入非正: ${a[0]}`, node });
      return Math.log(a[0]);
    case "sin": return Math.sin(a[0]);
    case "cos": return Math.cos(a[0]);
    case "abs": return Math.abs(a[0]);
    default: throw new Error(`未知算子 ${op}`);
  }
}

/** 算子对第 i 个参数的偏导（在参数值 a 处）。不可导点抛 EvalError。 */
function partial(op: string, i: number, a: number[], exponent: number | undefined, node: string): number {
  switch (op) {
    case "add": return 1;
    case "sub": return i === 0 ? 1 : -1;
    case "mul": return i === 0 ? a[1] : a[0];
    case "div": return i === 0 ? 1 / a[1] : -a[0] / (a[1] * a[1]);
    case "pow": return exponent! * Math.pow(a[0], exponent! - 1);
    case "exp": return Math.exp(a[0]);
    case "log": return 1 / a[0];
    case "sin": return Math.cos(a[0]);
    case "cos": return -Math.sin(a[0]);
    case "abs":
      if (a[0] === 0)
        throw new EvalError({
          code: "ABS_NONDIFERENTIABLE",
          message: `节点 "${node}" abs 在 0 处不可导（策略：报错，不返回次梯度）`,
          node,
        });
      return a[0] > 0 ? 1 : -1;
    default: throw new Error(`未知算子 ${op}`);
  }
}

export function outputValues(
  g: CompiledGraph,
  values: Map<string, number>
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, ref] of Object.entries(g.outputs)) out[name] = values.get(ref)!;
  return out;
}

/** 前向模式：给定方向 direction（输入 -> 数），返回每个输出的方向导数。 */
export function directionalDerivative(
  g: CompiledGraph,
  point: Record<string, number>,
  direction: Record<string, number>
): { values: Record<string, number>; tangents: Record<string, number> } {
  const values = new Map<string, number>();
  const dots = new Map<string, number>();
  for (const name of g.inputs) {
    values.set(name, point[name]);
    dots.set(name, direction[name] ?? 0);
  }
  for (const [k, v] of Object.entries(g.constants)) { values.set(k, v); dots.set(k, 0); }
  for (const nd of g.nodes) {
    const a = nd.args.map((n) => values.get(n)!);
    values.set(nd.name, applyOp(nd.op, a, nd.exponent, nd.name));
    let d = 0;
    nd.args.forEach((arg, i) => {
      d += partial(nd.op, i, a, nd.exponent, nd.name) * dots.get(arg)!;
    });
    dots.set(nd.name, d);
  }
  const tangents: Record<string, number> = {};
  for (const [name, ref] of Object.entries(g.outputs)) tangents[name] = dots.get(ref)!;
  return { values: outputValues(g, values), tangents };
}

/**
 * 反向模式：返回所有输出关于所有输入的梯度（Jacobian 行）。
 * 对每个输出单独做一次反向传播；多条入边贡献累加（adj[x] += ...）。
 * 每次调用使用全新的 adjoint 表，不残留上次结果。
 */
export function reverseGradient(
  g: CompiledGraph,
  point: Record<string, number>,
  weights?: Record<string, number>
): { values: Record<string, number>; gradient: Record<string, number> } {
  const values = forward(g, point);
  // 累加器：名字 -> 伴随值
  const adj = new Map<string, number>();
  const bump = (name: string, delta: number) => adj.set(name, (adj.get(name) ?? 0) + delta);
  for (const [outName, ref] of Object.entries(g.outputs)) {
    bump(ref, weights?.[outName] ?? 1);
  }
  for (let i = g.nodes.length - 1; i >= 0; i--) {
    const nd = g.nodes[i];
    const bar = adj.get(nd.name);
    if (bar === undefined || bar === 0) continue;
    const a = nd.args.map((n) => values.get(n)!);
    nd.args.forEach((arg, j) => {
      bump(arg, bar * partial(nd.op, j, a, nd.exponent, nd.name));
    });
  }
  const gradient: Record<string, number> = {};
  for (const name of g.inputs) gradient[name] = adj.get(name) ?? 0;
  return { values: outputValues(g, values), gradient };
}

/** 多输出 Jacobian：rows[输出名][输入名]。 */
export function jacobian(
  g: CompiledGraph,
  point: Record<string, number>
): { values: Record<string, number>; jacobian: Record<string, Record<string, number>> } {
  const values = forward(g, point);
  const rows: Record<string, Record<string, number>> = {};
  for (const outName of Object.keys(g.outputs)) {
    const w: Record<string, number> = {};
    for (const o of Object.keys(g.outputs)) w[o] = o === outName ? 1 : 0;
    // 复用同一次前向数值，仅重做反向传播
    rows[outName] = reverseWithValues(g, values, w);
  }
  return { values: outputValues(g, values), jacobian: rows };
}

function reverseWithValues(
  g: CompiledGraph,
  values: Map<string, number>,
  weights: Record<string, number>
): Record<string, number> {
  const adj = new Map<string, number>();
  const bump = (name: string, delta: number) => adj.set(name, (adj.get(name) ?? 0) + delta);
  for (const [outName, ref] of Object.entries(g.outputs)) {
    const w = weights[outName] ?? 0;
    if (w !== 0) bump(ref, w);
  }
  for (let i = g.nodes.length - 1; i >= 0; i--) {
    const nd = g.nodes[i];
    const bar = adj.get(nd.name);
    if (bar === undefined || bar === 0) continue;
    const a = nd.args.map((n) => values.get(n)!);
    nd.args.forEach((arg, j) => {
      bump(arg, bar * partial(nd.op, j, a, nd.exponent, nd.name));
    });
  }
  const gradient: Record<string, number> = {};
  for (const name of g.inputs) gradient[name] = adj.get(name) ?? 0;
  return gradient;
}

/** 向量-Jacobian 乘积：weights 为每个输出的权重，返回关于输入的加权和梯度。 */
export function vjp(
  g: CompiledGraph,
  point: Record<string, number>,
  weights: Record<string, number>
): { values: Record<string, number>; vjp: Record<string, number> } {
  const r = reverseGradient(g, point, weights);
  return { values: r.values, vjp: r.gradient };
}

export function toDiagnostics(e: unknown): Diagnostic[] {
  if (e instanceof EvalError) return [e.diag];
  return [{ code: "FD_EVAL_FAILED", message: String(e) }];
}
