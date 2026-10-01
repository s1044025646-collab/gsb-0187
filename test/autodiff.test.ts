import { test } from "node:test";
import assert from "node:assert/strict";
import { compileGraph } from "../src/graph.js";
import { forward, outputValues, reverseGradient, jacobian, directionalDerivative } from "../src/evaluate.js";
import { finiteDifferenceCheck } from "../src/fdcheck.js";
import { examples, polyShared, mimo, domainError } from "../src/examples.js";
import { GraphDef } from "../src/types.js";

const close = (a: number, b: number, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);

test("手算导数: f = x^2 + 3x, f'(2) = 7", () => {
  const g = compileGraph({
    inputs: ["x"],
    constants: { three: 3 },
    nodes: [
      { name: "x2", op: "pow", args: ["x"], exponent: 2 },
      { name: "t", op: "mul", args: ["three", "x"] },
      { name: "f", op: "add", args: ["x2", "t"] },
    ],
    outputs: { f: "f" },
  });
  const r = reverseGradient(g, { x: 2 });
  close(r.values.f, 10);
  close(r.gradient.x, 7);
});

test("链式法则: f = sin(exp(x)), f'(0) = cos(1)*1", () => {
  const g = compileGraph({
    inputs: ["x"],
    nodes: [
      { name: "e", op: "exp", args: ["x"] },
      { name: "s", op: "sin", args: ["e"] },
    ],
    outputs: { f: "s" },
  });
  close(reverseGradient(g, { x: 0 }).gradient.x, Math.cos(1));
});

test("共享节点重复引用: u 被两条边消费，贡献必须累加（覆盖会得到错误值）", () => {
  // f = u*u，u = x+1。df/dx = 2u*1 = 2(x+1)。x=2 => 6。
  // 若反向传播覆盖而非累加，u 的伴随只会保留一条边的贡献，得到 3。
  const g = compileGraph({
    inputs: ["x"],
    constants: { one: 1 },
    nodes: [
      { name: "u", op: "add", args: ["x", "one"] },
      { name: "f", op: "mul", args: ["u", "u"] },
    ],
    outputs: { f: "f" },
  });
  const r = reverseGradient(g, { x: 2 });
  close(r.values.f, 9);
  close(r.gradient.x, 6);
});

test("共享子图多项式: f = u^2+u, u = x^2+x, x=1 => f=6, f'=15", () => {
  const g = compileGraph(polyShared);
  const r = reverseGradient(g, { x: 1 });
  close(r.values.f, 6);
  close(r.gradient.x, 15); // u=2, df/du=2u+1=5, du/dx=2x+1=3 => 15
});

test("常量输出的零梯度", () => {
  const g = compileGraph({
    inputs: ["x"],
    constants: { c: 5 },
    nodes: [{ name: "n", op: "mul", args: ["c", "c"] }],
    outputs: { f: "n" },
  });
  close(reverseGradient(g, { x: 99 }).gradient.x, 0);
});

test("前向与反向的内积一致性: <J v, w> = vjp(w)·v", () => {
  const g = compileGraph(mimo);
  const point = { x: 0.5, y: 2, z: 0.3 };
  const v = { x: 1.5, y: -0.7, z: 2.0 };
  const w = { f1: 0.8, f2: -1.2 };
  const dir = directionalDerivative(g, point, v);
  const lhs = dir.tangents.f1 * w.f1 + dir.tangents.f2 * w.f2;
  const rev = reverseGradient(g, point, w).gradient;
  const rhs = rev.x * v.x + rev.y * v.y + rev.z * v.z;
  close(lhs, rhs, 1e-9);
});

test("多输出 Jacobian 与逐输出梯度一致", () => {
  const g = compileGraph(mimo);
  const point = { x: 0.5, y: 2, z: 0.3 };
  const J = jacobian(g, point).jacobian;
  close(J.f1.x, 2); close(J.f1.y, 0.5); close(J.f1.z, Math.cos(0.3));
  close(J.f2.x, Math.exp(0.5) / 3);
  close(J.f2.y, -Math.exp(0.5) / 9);
  close(J.f2.z, 0);
});

test("批量更换输入后梯度不串值", () => {
  const g = compileGraph(polyShared);
  const r1 = reverseGradient(g, { x: 1 });
  const r2 = reverseGradient(g, { x: 3 });
  const r3 = reverseGradient(g, { x: 1 });
  close(r1.gradient.x, 15);
  close(r2.gradient.x, (2 * 12 + 1) * 7); // u=12, df/du=25, du/dx=7 => 175
  close(r3.gradient.x, r1.gradient.x);
});

test("中心有限差分校验通过", () => {
  const g = compileGraph(examples.nonlinear);
  const report = finiteDifferenceCheck(g, { x: 1.3, y: 0.7 }, "g");
  assert.ok(report.ok);
  for (const r of Object.values(report.results)) {
    assert.ok(r.relError! < 1e-5, `relError=${r.relError}`);
  }
});

test("有限差分支持按输入尺度设置步长", () => {
  const g = compileGraph({
    inputs: ["big", "small"],
    nodes: [
      { name: "b2", op: "pow", args: ["big"], exponent: 2 },
      { name: "s2", op: "pow", args: ["small"], exponent: 2 },
      { name: "f", op: "add", args: ["b2", "s2"] },
    ],
    outputs: { f: "f" },
  });
  const report = finiteDifferenceCheck(g, { big: 1e4, small: 1e-3 }, "f", 1e-5, { small: 1e-7 });
  assert.ok(report.ok);
  assert.equal(report.results.small.step, 1e-7);
  close(report.results.big.numeric!, 2e4, 1e-4);
});

test("定义域诊断: 除零 / log 非正 / exp 溢出 / abs 零点", () => {
  const g = compileGraph(domainError);
  assert.throws(() => forward(g, { x: -1, y: 2, z: 3 }), /log 输入非正/);
  assert.throws(() => forward(g, { x: 1, y: 0, z: 3 }), /除数为零/);
  assert.throws(() => reverseGradient(g, { x: 1, y: 2, z: 0 }), /abs 在 0 处不可导/);
  const gExp = compileGraph({
    inputs: ["x"],
    nodes: [{ name: "e", op: "exp", args: ["x"] }],
    outputs: { f: "e" },
  });
  assert.throws(() => forward(gExp, { x: 1000 }), /溢出/);
});

test("图校验: 重名 / 悬空引用 / 非法指数", () => {
  assert.throws(() => compileGraph({ inputs: ["x"], nodes: [{ name: "x", op: "exp", args: ["x"] }], outputs: { f: "x" } }), /重复/);
  assert.throws(() => compileGraph({ inputs: ["x"], nodes: [{ name: "n", op: "exp", args: ["nope"] }], outputs: { f: "n" } }), /未定义/);
  assert.throws(() => compileGraph({ inputs: ["x"], nodes: [{ name: "n", op: "pow", args: ["x"], exponent: 1.5 }], outputs: { f: "n" } }), /整数/);
});

test("除法与减法梯度手算: f = x/y - y, (x,y)=(6,2) => df/dx=0.5, df/dy=-2.5", () => {
  const g = compileGraph({
    inputs: ["x", "y"],
    nodes: [
      { name: "q", op: "div", args: ["x", "y"] },
      { name: "f", op: "sub", args: ["q", "y"] },
    ],
    outputs: { f: "f" },
  });
  const r = reverseGradient(g, { x: 6, y: 2 });
  close(r.gradient.x, 0.5);
  close(r.gradient.y, -1.5 - 1);
});

test("方向导数: f=x*y, 点(2,3), 方向(1,1) => 5", () => {
  const g = compileGraph({
    inputs: ["x", "y"],
    nodes: [{ name: "f", op: "mul", args: ["x", "y"] }],
    outputs: { f: "f" },
  });
  close(directionalDerivative(g, { x: 2, y: 3 }, { x: 1, y: 1 }).tangents.f, 5);
});

test("所有内置算例可编译", () => {
  for (const def of Object.values(examples)) compileGraph(def as GraphDef);
});
