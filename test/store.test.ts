import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Store } from "../src/store.js";
import { Service } from "../src/service.js";
import { polyShared } from "../src/examples.js";

test("图版本与结果持久化", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "autodiff-"));
  try {
    const dbPath = path.join(dir, "t.db");
    let svc = new Service(new Store(dbPath));
    const v1 = svc.importGraph(polyShared);
    assert.equal(v1.version, 1);
    svc.evaluate("poly_shared", { x: 1 });
    // 修改图 => 新版本
    const modified = structuredClone(polyShared);
    modified.nodes.push({ name: "f2", op: "mul", args: ["f", "f"] });
    modified.outputs = { ...modified.outputs, f_sq: "f2" };
    const v2 = svc.importGraph(modified);
    assert.equal(v2.version, 2);
    svc.evaluate("poly_shared", { x: 1 }, 2);
    // 重复导入同内容 => 不新增版本
    assert.equal(svc.importGraph(polyShared).version, 1);
    // 模拟重启
    svc.store.close();
    svc = new Service(new Store(dbPath));
    const r1 = svc.store.listResults(svc.store.getGraph("poly_shared", 1)!.id);
    const r2 = svc.store.listResults(svc.store.getGraph("poly_shared", 2)!.id);
    assert.equal(r1.length, 1);
    assert.equal(r2.length, 1);
    assert.equal(r1[0].payload.values.f, 6);
    assert.equal(r2[0].payload.values.f_sq, 36);
    svc.store.close();
  } finally {
    // Windows 上 SQLite 句柄释放可能有延迟，清理失败不影响测试结果
    try { rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }); } catch { /* ignore */ }
  }
});
