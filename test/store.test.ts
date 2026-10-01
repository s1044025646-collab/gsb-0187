import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Store } from "../src/db/store.js";
import { Service } from "../src/service.js";
import { GraphDef } from "../src/core/types.js";

function fresh() {
  const dir = mkdtempSync(join(tmpdir(), "autodiff-"));
  const store = new Store(join(dir, "t.db"));
  return { dir, store, service: new Service(store) };
}

const v1: GraphDef = {
  name: "g",
  inputs: ["x"],
  constants: { k: 1 },
  nodes: { f: { op: "add", a: "x", b: "k" } },
  outputs: ["f"],
};

const v2: GraphDef = {
  name: "g",
  inputs: ["x"],
  constants: { k: 10 },
  nodes: { f: { op: "add", a: "x", b: "k" } },
  outputs: ["f"],
};

test("修改图创建新版本并保留旧结果，重启后可读取", () => {
  const { dir, store, service } = fresh();
  const a = service.importGraph(v1);
  assert.equal(a.version, 1);
  const r1 = service.gradient("g", { x: 2 }, 1);
  assert.equal(r1.jacobian.f!.x, 1);
  assert.equal(r1.outputs.f, 3);

  const b = service.importGraph(v2);
  assert.equal(b.version, 2);
  const r2 = service.gradient("g", { x: 2 }, 2);
  assert.equal(r2.outputs.f, 12);

  // 重新导入相同定义复用版本，不产生新版本。
  const again = service.importGraph(v1);
  assert.equal(again.version, 1);
  assert.equal(again.reused, true);

  // 模拟重启：用同一数据库路径新建 Store。
  store.close();
  const reopened = new Store(join(dir, "t.db"));
  const svc2 = new Service(reopened);
  const reports = reopened.listReports(a.graphId);
  assert.ok(reports.length >= 1);
  const old = svc2.gradient("g", { x: 2 }, 1, false);
  assert.equal(old.outputs.f, 3);
  const newest = svc2.gradient("g", { x: 2 }, 2, false);
  assert.equal(newest.outputs.f, 12);
  reopened.close();
});
