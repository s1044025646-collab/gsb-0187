// 计算图核心数据结构定义。

export type OpType =
  | "add"
  | "sub"
  | "mul"
  | "div"
  | "pow"
  | "exp"
  | "log"
  | "sin"
  | "cos"
  | "abs";

export const BINARY_OPS: ReadonlySet<OpType> = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "pow",
]);

export const UNARY_OPS: ReadonlySet<OpType> = new Set([
  "exp",
  "log",
  "sin",
  "cos",
  "abs",
]);

export interface GraphNode {
  op: OpType;
  // 二元算子使用两个操作数；一元算子只使用 a。
  a: string;
  b?: string;
}

export interface GraphDef {
  name: string;
  inputs: string[];
  constants?: Record<string, number>;
  nodes: Record<string, GraphNode>;
  outputs: string[];
  // 可选：给输出起便于阅读的名字，缺省即输出引用名。
  outputNames?: Record<string, string>;
}

// 诊断错误码。
export type ErrorCode =
  | "GRAPH_INVALID"
  | "CYCLE_DETECTED"
  | "UNKNOWN_REFERENCE"
  | "DUPLICATE_NAME"
  | "MISSING_INPUT"
  | "NON_FINITE_INPUT"
  | "DIV_BY_ZERO"
  | "LOG_DOMAIN"
  | "EXP_OVERFLOW"
  | "ABS_KINK"
  | "POW_DOMAIN"
  | "NUMERIC_OVERFLOW";

export class GraphError extends Error {
  constructor(
    readonly code: ErrorCode,
    message: string,
    readonly detail?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "GraphError";
  }
}

// 一次求值的数值结果。
export type ValueMap = Record<string, number>;

export interface EvalResult {
  outputs: ValueMap;
  // 所有中间节点（含常量）的数值，用于缓存与调试。
  values: ValueMap;
  order: string[];
}

export interface GradientReport {
  mode: "reverse";
  outputs: ValueMap;
  // output -> input -> derivative
  jacobian: Record<string, ValueMap>;
}

export interface DirectionalResult {
  mode: "forward";
  direction: ValueMap;
  outputs: ValueMap;
  directional: ValueMap;
}

export interface CheckEntry {
  input: string;
  output: string;
  analytic: number;
  numeric: number | null;
  absError: number | null;
  relError: number | null;
  h: number;
  ok: boolean;
  reason?: string;
}

export interface CheckReport {
  baseStep: number;
  perInputStep: Record<string, number>;
  entries: CheckEntry[];
  allOk: boolean;
}
