export type OpName =
  | "add" | "sub" | "mul" | "div"
  | "pow" | "exp" | "log" | "sin" | "cos" | "abs";

export interface NodeDef {
  name: string;
  op: OpName;
  /** 引用的输入/常量/节点名称；pow 为单参数（底数） */
  args: string[];
  /** 仅 pow 使用：整数指数 */
  exponent?: number;
}

export interface GraphDef {
  name?: string;
  inputs: string[];
  constants?: Record<string, number>;
  nodes: NodeDef[];
  /** 输出名 -> 节点/输入/常量名 */
  outputs: Record<string, string>;
}

export type DiagCode =
  | "DIV_ZERO"
  | "LOG_NONPOSITIVE"
  | "EXP_OVERFLOW"
  | "ABS_NONDIFERENTIABLE"
  | "FD_EVAL_FAILED";

export interface Diagnostic {
  code: DiagCode;
  message: string;
  node?: string;
}

export class EvalError extends Error {
  constructor(public diag: Diagnostic) {
    super(diag.message);
    this.name = "EvalError";
  }
}
