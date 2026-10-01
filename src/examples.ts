import { GraphDef } from "./types.js";

/** 共享子图的多项式：f = (x^2 + x)^2 + (x^2 + x)，子表达式 u = x^2+x 被复用 */
export const polyShared: GraphDef = {
  name: "poly_shared",
  inputs: ["x"],
  nodes: [
    { name: "x2", op: "pow", args: ["x"], exponent: 2 },
    { name: "u", op: "add", args: ["x2", "x"] },
    { name: "u2", op: "pow", args: ["u"], exponent: 2 },
    { name: "f", op: "add", args: ["u2", "u"] },
  ],
  outputs: { f: "f" },
};

/** 多输入多输出：f1 = x*y + sin(z), f2 = exp(x) / (y + 1) */
export const mimo: GraphDef = {
  name: "mimo",
  inputs: ["x", "y", "z"],
  constants: { one: 1 },
  nodes: [
    { name: "xy", op: "mul", args: ["x", "y"] },
    { name: "sz", op: "sin", args: ["z"] },
    { name: "f1", op: "add", args: ["xy", "sz"] },
    { name: "ex", op: "exp", args: ["x"] },
    { name: "yp1", op: "add", args: ["y", "one"] },
    { name: "f2", op: "div", args: ["ex", "yp1"] },
  ],
  outputs: { f1: "f1", f2: "f2" },
};

/** 非线性组合：g = log(x^2 + 1) * cos(y) + |y| */
export const nonlinear: GraphDef = {
  name: "nonlinear",
  inputs: ["x", "y"],
  constants: { one: 1 },
  nodes: [
    { name: "x2", op: "pow", args: ["x"], exponent: 2 },
    { name: "x2p1", op: "add", args: ["x2", "one"] },
    { name: "lx", op: "log", args: ["x2p1"] },
    { name: "cy", op: "cos", args: ["y"] },
    { name: "prod", op: "mul", args: ["lx", "cy"] },
    { name: "ay", op: "abs", args: ["y"] },
    { name: "g", op: "add", args: ["prod", "ay"] },
  ],
  outputs: { g: "g" },
};

/** 定义域错误算例：h = log(x) + 1 / y + abs(z) */
export const domainError: GraphDef = {
  name: "domain_error",
  inputs: ["x", "y", "z"],
  constants: { one: 1 },
  nodes: [
    { name: "lx", op: "log", args: ["x"] },
    { name: "ry", op: "div", args: ["one", "y"] },
    { name: "s", op: "add", args: ["lx", "ry"] },
    { name: "az", op: "abs", args: ["z"] },
    { name: "h", op: "add", args: ["s", "az"] },
  ],
  outputs: { h: "h" },
};

export const examples: Record<string, GraphDef> = {
  poly_shared: polyShared,
  mimo,
  nonlinear,
  domain_error: domainError,
};
