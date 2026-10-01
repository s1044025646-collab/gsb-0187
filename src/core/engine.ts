// 自动微分引擎：前向求值、前向方向导数、反向梯度 / Jacobian / 向量乘积。
// 不使用 eval，不依赖任何自动微分库。

import {
  DirectionalResult,
  EvalResult,
  GraphError,
  GradientReport,
  ValueMap,
} from "./types.js";
import { ValidatedGraph } from "./validate.js";

export const EXP_LIMIT = 709; // exp(709.78) 接近双精度上限。

function assertFinite(v: number, node: string): void {
  if (!Number.isFinite(v)) {
    throw new GraphError(
      "NUMERIC_OVERFLOW",
      `节点 ${node} 产生非有限结果（溢出或下溢导致的非法值）`,
      { node, value: v },
    );
  }
}

// 检查输入向量是否齐全且为有限数。
export function checkInputs(
  vg: ValidatedGraph,
  inputs: ValueMap,
): void {
  for (const name of vg.def.inputs) {
    if (!(name in inputs)) {
      throw new GraphError("MISSING_INPUT", `缺少输入值: ${name}`, {
        input: name,
      });
    }
    const v = inputs[name];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new GraphError("NON_FINITE_INPUT", `输入必须是有限数: ${name}`, {
        input: name,
        value: v,
      });
    }
  }
}

// 前向求值，返回所有节点数值（按拓扑序）。
export function evaluate(
  vg: ValidatedGraph,
  inputs: ValueMap,
): EvalResult {
  checkInputs(vg, inputs);
  const values: ValueMap = {};

  for (const name of vg.def.inputs) values[name] = inputs[name]!;
  for (const [name, value] of Object.entries(vg.constants)) values[name] = value;

  for (const name of vg.order) {
    const node = vg.def.nodes[name]!;
    const x = values[node.a]!;
    const y =
      typeof node.b === "string" ? values[node.b]! : 0;
    values[name] = computeOp(node.op, x, y, name, vg);
    assertFinite(values[name]!, name);
  }

  const outputs: ValueMap = {};
  for (const out of vg.def.outputs) outputs[out] = values[out]!;
  return { outputs, values, order: vg.order.slice() };
}

function computeOp(
  op: string,
  x: number,
  y: number,
  node: string,
  vg: ValidatedGraph,
): number {
  switch (op) {
    case "add":
      return x + y;
    case "sub":
      return x - y;
    case "mul":
      return x * y;
    case "div":
      if (y === 0) {
        throw new GraphError("DIV_BY_ZERO", `节点 ${node} 除以零`, {
          node,
          numerator: x,
        });
      }
      return x / y;
    case "pow": {
      // 仅支持整数幂；指数来自命名常量或整数节点值，此处按数值校验。
      if (!Number.isInteger(y)) {
        throw new GraphError(
          "POW_DOMAIN",
          `节点 ${node} 的幂 ${y} 不是整数（仅支持整数幂）`,
          { node, exponent: y },
        );
      }
      if (x === 0 && y <= 0) {
        throw new GraphError("POW_DOMAIN", `节点 ${node} 为 0 的非正整数幂`, {
          node,
          base: x,
          exponent: y,
        });
      }
      return Math.pow(x, y);
    }
    case "exp":
      if (x > EXP_LIMIT) {
        throw new GraphError(
          "EXP_OVERFLOW",
          `节点 ${node} 的 exp 输入 ${x} 超过安全上限 ${EXP_LIMIT}`,
          { node, arg: x },
        );
      }
      return Math.exp(x);
    case "log":
      if (x <= 0) {
        throw new GraphError("LOG_DOMAIN", `节点 ${node} 的 log 输入非正: ${x}`, {
          node,
          arg: x,
        });
      }
      return Math.log(x);
    case "sin":
      return Math.sin(x);
    case "cos":
      return Math.cos(x);
    case "abs":
      return Math.abs(x);
    default:
      throw new GraphError("GRAPH_INVALID", `未知算子: ${op}`, { op, node });
  }
}

// 前向模式：沿给定方向 direction 同时传播取值与方向导数（切线）。
export function directional(
  vg: ValidatedGraph,
  inputs: ValueMap,
  direction: ValueMap,
): DirectionalResult {
  checkInputs(vg, inputs);
  const values: ValueMap = {};
  const tangent: ValueMap = {};

  for (const name of vg.def.inputs) {
    values[name] = inputs[name]!;
    const d = direction[name];
    tangent[name] = typeof d === "number" && Number.isFinite(d) ? d : 0;
  }
  for (const [name, value] of Object.entries(vg.constants)) {
    values[name] = value;
    tangent[name] = 0;
  }

  for (const name of vg.order) {
    const node = vg.def.nodes[name]!;
    const x = values[node.a]!;
    const dx = tangent[node.a]!;
    const y = typeof node.b === "string" ? values[node.b]! : 0;
    const dy = typeof node.b === "string" ? tangent[node.b]! : 0;

    values[name] = computeOp(node.op, x, y, name, vg);
    assertFinite(values[name]!, name);
    tangent[name] = localTangent(node.op, x, y, dx, dy, name);
    assertFinite(tangent[name]!, name);
  }

  const outputs: ValueMap = {};
  const dirOut: ValueMap = {};
  for (const out of vg.def.outputs) {
    outputs[out] = values[out]!;
    dirOut[out] = tangent[out]!;
  }
  return {
    mode: "forward",
    direction: { ...direction },
    outputs,
    directional: dirOut,
  };
}

function localTangent(
  op: string,
  x: number,
  y: number,
  dx: number,
  dy: number,
  node: string,
): number {
  switch (op) {
    case "add":
      return dx + dy;
    case "sub":
      return dx - dy;
    case "mul":
      return dx * y + x * dy;
    case "div":
      if (y === 0) {
        throw new GraphError("DIV_BY_ZERO", `节点 ${node} 除以零`, { node });
      }
      return (dx * y - x * dy) / (y * y);
    case "pow":
      // d(x^n) = n x^(n-1) dx；指数必须是整数常量，故 dy 仅允许 0。
      if (dy !== 0) {
        throw new GraphError(
          "POW_DOMAIN",
          `节点 ${node} 的指数不允许是变量（仅支持整数常量幂）`,
          { node },
        );
      }
      if (x === 0 && y - 1 < 0) {
        throw new GraphError("POW_DOMAIN", `节点 ${node} 幂导数在 0 处无定义`, {
          node,
        });
      }
      return y * Math.pow(x, y - 1) * dx;
    case "exp":
      return Math.exp(x) * dx;
    case "log":
      if (x <= 0) {
        throw new GraphError("LOG_DOMAIN", `节点 ${node} 的 log 输入非正`, {
          node,
          arg: x,
        });
      }
      return dx / x;
    case "sin":
      return Math.cos(x) * dx;
    case "cos":
      return -Math.sin(x) * dx;
    case "abs":
      if (x === 0) {
        // 明确约定：abs 在 0 处不可导。任何经过该点的非零切线都无定义。
        throw new GraphError(
          "ABS_KINK",
          `节点 ${node} 的 abs 在输入 0 处不可导（分段绝对值的尖点策略：拒绝而非返回任意值）`,
          { node },
        );
      }
      return (x > 0 ? 1 : -1) * dx;
    default:
      throw new GraphError("GRAPH_INVALID", `未知算子: ${op}`, { op, node });
  }
}

// 反向模式：计算多个输出相对每个输入的完整 Jacobian。
export function reverseJacobian(
  vg: ValidatedGraph,
  inputs: ValueMap,
): GradientReport {
  const { outputs, values } = evaluate(vg, inputs);
  const inputNames = vg.def.inputs;
  const jac: Record<string, ValueMap> = {};

  for (const target of vg.def.outputs) {
    const grad: ValueMap = {};
    for (const name of Object.keys(values)) grad[name] = 0;
    grad[target] = 1;

    // 逆拓扑遍历；同一节点多条入边必须累加，不能覆盖。
    for (let i = vg.order.length - 1; i >= 0; i--) {
      const name = vg.order[i]!;
      const node = vg.def.nodes[name]!;
      const adj = grad[name]!;
      if (adj === 0) continue;
      const x = values[node.a]!;
      const y = typeof node.b === "string" ? values[node.b]! : 0;
      const [ga, gb] = localAdjoint(node.op, x, y, adj, name);
      grad[node.a] = grad[node.a]! + ga;
      if (typeof node.b === "string") {
        grad[node.b] = grad[node.b]! + gb;
      }
    }

    const row: ValueMap = {};
    for (const name of inputNames) row[name] = grad[name]!;
    jac[target] = row;
  }

  return { mode: "reverse", outputs, jacobian: jac };
}

function localAdjoint(
  op: string,
  x: number,
  y: number,
  adj: number,
  node: string,
): [number, number] {
  switch (op) {
    case "add":
      return [adj, adj];
    case "sub":
      return [adj, -adj];
    case "mul":
      return [adj * y, adj * x];
    case "div":
      if (y === 0) {
        throw new GraphError("DIV_BY_ZERO", `节点 ${node} 除以零`, { node });
      }
      return [adj / y, (-adj * x) / (y * y)];
    case "pow":
      if (x === 0 && y - 1 < 0) {
        throw new GraphError("POW_DOMAIN", `节点 ${node} 幂导数在 0 处无定义`, {
          node,
        });
      }
      return [adj * y * Math.pow(x, y - 1), 0];
    case "exp":
      return [adj * Math.exp(x), 0];
    case "log":
      if (x <= 0) {
        throw new GraphError("LOG_DOMAIN", `节点 ${node} 的 log 输入非正`, {
          node,
          arg: x,
        });
      }
      return [adj / x, 0];
    case "sin":
      return [adj * Math.cos(x), 0];
    case "cos":
      return [adj * -Math.sin(x), 0];
    case "abs":
      if (x === 0) {
        throw new GraphError(
          "ABS_KINK",
          `节点 ${node} 的 abs 在输入 0 处不可导（尖点策略：拒绝传播，不返回任意梯度）`,
          { node, adjoint: adj },
        );
      }
      return [adj * (x > 0 ? 1 : -1), 0];
    default:
      throw new GraphError("GRAPH_INVALID", `未知算子: ${op}`, { op, node });
  }
}

// 给定输出权重的向量乘积：J^T w（一次反向传播，无需全 Jacobian）。
export function vectorJacobianProduct(
  vg: ValidatedGraph,
  inputs: ValueMap,
  weights: ValueMap,
): { weighted: ValueMap; outputs: ValueMap } {
  const { outputs, values } = evaluate(vg, inputs);
  const grad: ValueMap = {};
  for (const name of Object.keys(values)) grad[name] = 0;

  for (const target of vg.def.outputs) {
    const w = weights[target] ?? 0;
    grad[target] = grad[target]! + w;
  }

  for (let i = vg.order.length - 1; i >= 0; i--) {
    const name = vg.order[i]!;
    const node = vg.def.nodes[name]!;
    const adj = grad[name]!;
    if (adj === 0) continue;
    const x = values[node.a]!;
    const y = typeof node.b === "string" ? values[node.b]! : 0;
    const [ga, gb] = localAdjoint(node.op, x, y, adj, name);
    grad[node.a] = grad[node.a]! + ga;
    if (typeof node.b === "string") {
      grad[node.b] = grad[node.b]! + gb;
    }
  }

  const weighted: ValueMap = {};
  for (const name of vg.def.inputs) weighted[name] = grad[name]!;
  return { weighted, outputs };
}
