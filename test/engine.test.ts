import { test } from "node:test";
import assert from "node:assert/strict";
import {
  directional,
  evaluate,
  reverseJacobian,
  vectorJacobianProduct,
} from "../src/core/engine.js";
import { validateGraph } from "../src/core/validate.js";
import { finiteCheck } from "../src/core/finiteDiff.js";
import { GraphDef, GraphError } from "../src/core/types.js";

function vg(def: GraphDef) {
  return validateGraph(def);
}

// f = x*y + x^2 + 3 ; g = (x*y + x^2)*y
const poly: GraphDef = {
  name: "poly",
  inputs: ["x", "y"],
  constants: { two: 2, three: 3 },
  nodes: {
    x2: { op: "pow", a: "x", b: "two" },
    xy: { op: "mul", a: "x", b: "y" },
    shared: { op: "add", a: "x2", b: "xy" },
    f: { op: "add", a: "shared", b: "three" },
    g: { op: "mul", a: "shared", b: "y" },
  },
  outputs: ["f", "g"],
};

test("手算多项式取值与导数", () => {
  const g = vg(poly);
  const inputs = { x: 2, y: 3 };
  const ev = evaluate(g, inputs);
  // shared = 4 + 6 = 10; f = 13; g = 30
  assert.equal(ev.outputs.f, 13);
  assert.equal(ev.outputs.g, 30);

  const { jacobian } = reverseJacobian(g, inputs);
  // df/dx = 2x + y = 7 ; df/dy = x = 2
  assert.ok(Math.abs(jacobian.f!.x! - 7) < 1e-12);
  assert.ok(Math.abs(jacobian.f!.y! - 2) < 1e-12);
  // dg/dx = (2x+y)*y = 21 ; dg/dy = (x^2+xy) + x*y = 10 + 6 = 16
  assert.ok(Math.abs(jacobian.g!.x! - 21) < 1e-12);
  assert.ok(Math.abs(jacobian.g!.y! - 16) < 1e-12);
});

test("链式法则: h = exp(log(x)) 在 x=2 处梯度为 1", () => {
  const def: GraphDef = {
    name: "chain",
    inputs: ["x"],
    constants: {},
    nodes: {
      l: { op: "log", a: "x" },
      e: { op: "exp", a: "l" },
    },
    outputs: ["e"],
  };
  const { jacobian } = reverseJacobian(vg(def), { x: 2 });
  assert.ok(Math.abs(jacobian.e!.x! - 1) < 1e-12);
});

test("共享节点被两条出边引用时梯度累加（不能覆盖）", () => {
  const g = vg(poly);
  const { jacobian } = reverseJacobian(g, { x: 1, y: 4 });
  // g = (x^2 + x*y)*y; dg/dy = shared + x*y = 5 + 4 = 9
  assert.ok(Math.abs(jacobian.g!.y! - 9) < 1e-12);

  // 反证：若把节点对同一输入的两条贡献“覆盖”而非累加，会得到错误结果。
  // 对 dg/dy，shared 贡献 5，额外的 *y 贡献 4；覆盖只会保留其中之一。
  const wrongOverwrite = 5; // 只保留来自 shared 的那条
  assert.notEqual(jacobian.g!.y, wrongOverwrite);
  const wrongOverwrite2 = 4; // 只保留乘法局部项
  assert.notEqual(jacobian.g!.y, wrongOverwrite2);
});

test("显式构造覆盖式反向传播，确认其结果错误而引擎正确", () => {
  // 用一个被重复引用两次的节点：out = s + s，s = x*y。
  // dout/dx 必须是 2y；覆盖实现只会得到 y。
  const def: GraphDef = {
    name: "dup",
    inputs: ["x", "y"],
    constants: {},
    nodes: {
      s: { op: "mul", a: "x", b: "y" },
      out: { op: "add", a: "s", b: "s" },
    },
    outputs: ["out"],
  };
  const g = vg(def);
  const correct = reverseJacobian(g, { x: 3, y: 5 }).jacobian.out!.x!;
  assert.equal(correct, 10); // 2*y

  // 错误实现：对同一被引用节点用赋值而非 +=。
  const values = evaluate(g, { x: 3, y: 5 }).values;
  const adj: Record<string, number> = { out: 1, s: 0, x: 0, y: 0 };
  // out = s+s，正确应 adj[s] += 1 两次。
  adj.s = 1; // 覆盖：第二条 +1 丢失
  adj.x += adj.s * values.y!; // 得到 5 而非 10
  assert.notEqual(adj.x, correct);
  assert.equal(adj.x, 5);
});

test("常量输出相对输入的梯度为零", () => {
  const def: GraphDef = {
    name: "constout",
    inputs: ["x"],
    constants: { k: 7 },
    nodes: {
      out: { op: "add", a: "k", b: "k" },
    },
    outputs: ["out"],
  };
  const { jacobian, outputs } = reverseJacobian(vg(def), { x: 100 });
  assert.equal(outputs.out, 14);
  assert.equal(jacobian.out!.x, 0);
});

test("前向方向导数与反向 VJP 的内积一致性", () => {
  const g = vg(poly);
  const inputs = { x: 1.5, y: -2.2 };
  const direction = { x: 0.7, y: -0.4 };
  const weights = { f: 1.3, g: -0.6 };

  const fwd = directional(g, inputs, direction);
  const vjp = vectorJacobianProduct(g, inputs, weights);

  // w^T (J d) 应等于 (J^T w)^T d
  const lhs =
    weights.f * fwd.directional.f! + weights.g * fwd.directional.g!;
  const rhs =
    vjp.weighted.x! * direction.x + vjp.weighted.y! * direction.y;
  assert.ok(Math.abs(lhs - rhs) < 1e-10, `lhs=${lhs} rhs=${rhs}`);
});

test("批量更换输入后梯度不串值", () => {
  const g = vg(poly);
  const batches = [
    { inputs: { x: 1, y: 1 }, fx: 3, fy: 1 },
    { inputs: { x: 2, y: 3 }, fx: 7, fy: 2 },
    { inputs: { x: -1, y: 4 }, fx: 2, fy: -1 },
  ];
  for (const b of batches) {
    const { jacobian } = reverseJacobian(g, b.inputs);
    const expectedFx = 2 * b.inputs.x + b.inputs.y;
    assert.ok(Math.abs(jacobian.f!.x! - expectedFx) < 1e-12);
    assert.ok(Math.abs(jacobian.f!.y! - b.fy) < 1e-12);
  }
  // 再算第一组，确认没有残留第三组的中间值/梯度。
  const again = reverseJacobian(g, batches[0]!.inputs).jacobian.f!;
  assert.equal(again.x, 3);
  assert.equal(again.y, 1);
});

test("有限差分校验通过且独立于反向结果", () => {
  const g = vg(poly);
  const rep = finiteCheck(g, { x: 2, y: 3 }, { baseStep: 1e-4 });
  assert.equal(rep.allOk, true);
  for (const e of rep.entries) {
    assert.equal(e.ok, true, e.reason);
    assert.ok(Math.abs((e.numeric ?? NaN) - e.analytic) < 1e-5);
  }
});

test("尺度差异输入使用每输入步长", () => {
  const def: GraphDef = {
    name: "scale",
    inputs: ["big", "small"],
    constants: {},
    nodes: {
      p: { op: "mul", a: "big", b: "small" },
    },
    outputs: ["p"],
  };
  const rep = finiteCheck(
    vg(def),
    { big: 1e6, small: 1e-6 },
    { baseStep: 1e-5, perInputStep: { small: 1e-8 } },
  );
  assert.equal(rep.allOk, true);
  assert.equal(rep.perInputStep.small, 1e-8);
});

test("除零返回明确诊断而非 Infinity", () => {
  const def: GraphDef = {
    name: "dz",
    inputs: ["a"],
    constants: { zero: 0 },
    nodes: { q: { op: "div", a: "a", b: "zero" } },
    outputs: ["q"],
  };
  assert.throws(
    () => evaluate(vg(def), { a: 1 }),
    (e: unknown) => e instanceof GraphError && e.code === "DIV_BY_ZERO",
  );
});

test("log 非正与 exp 溢出诊断", () => {
  const logDef: GraphDef = {
    name: "lg",
    inputs: ["x"],
    constants: {},
    nodes: { l: { op: "log", a: "x" } },
    outputs: ["l"],
  };
  assert.throws(
    () => evaluate(vg(logDef), { x: 0 }),
    (e: unknown) => e instanceof GraphError && e.code === "LOG_DOMAIN",
  );

  const expDef: GraphDef = {
    name: "ex",
    inputs: ["x"],
    constants: {},
    nodes: { e: { op: "exp", a: "x" } },
    outputs: ["e"],
  };
  assert.throws(
    () => evaluate(vg(expDef), { x: 1e4 }),
    (e: unknown) => e instanceof GraphError && e.code === "EXP_OVERFLOW",
  );
});

test("abs 在零尖点拒绝返回任意梯度", () => {
  const def: GraphDef = {
    name: "abs",
    inputs: ["t"],
    constants: {},
    nodes: {
      z: { op: "sub", a: "t", b: "t" },
      az: { op: "abs", a: "z" },
    },
    outputs: ["az"],
  };
  const g = vg(def);
  // 前向取值本身可以（abs(0)=0）。
  assert.equal(evaluate(g, { t: 5 }).outputs.az, 0);
  assert.throws(
    () => reverseJacobian(g, { t: 5 }),
    (e: unknown) => e instanceof GraphError && e.code === "ABS_KINK",
  );
});

test("abs 在非零处梯度为符号", () => {
  const def: GraphDef = {
    name: "abs2",
    inputs: ["t"],
    constants: {},
    nodes: { a: { op: "abs", a: "t" } },
    outputs: ["a"],
  };
  const g = vg(def);
  assert.equal(reverseJacobian(g, { t: 3 }).jacobian.a!.t, 1);
  assert.equal(reverseJacobian(g, { t: -3 }).jacobian.a!.t, -1);
});

test("未知引用与依赖环被拒绝", () => {
  const badRef: GraphDef = {
    name: "br",
    inputs: ["x"],
    constants: {},
    nodes: { n: { op: "add", a: "x", b: "ghost" } },
    outputs: ["n"],
  };
  assert.throws(
    () => validateGraph(badRef),
    (e: unknown) => e instanceof GraphError && e.code === "UNKNOWN_REFERENCE",
  );

  const cycle: GraphDef = {
    name: "cy",
    inputs: ["x"],
    constants: {},
    nodes: {
      a: { op: "add", a: "x", b: "b" },
      b: { op: "add", a: "a", b: "x" },
    },
    outputs: ["a"],
  };
  assert.throws(
    () => validateGraph(cycle),
    (e: unknown) => e instanceof GraphError && e.code === "CYCLE_DETECTED",
  );
});

test("有限差分在定义域失败时给出原因而非冒充通过", () => {
  const def: GraphDef = {
    name: "fdlog",
    inputs: ["x"],
    constants: {},
    nodes: { l: { op: "log", a: "x" } },
    outputs: ["l"],
  };
  // x-h < 0，差分会触碰非法域。
  const rep = finiteCheck(vg(def), { x: 1e-6 }, { baseStep: 1e-3 });
  assert.equal(rep.allOk, false);
  const failed = rep.entries.find((e) => !e.ok);
  assert.ok(failed?.reason?.includes("LOG_DOMAIN"));
});
