# 计算图自动微分与梯度校验后端

基于 TypeScript + Node.js + SQLite（`node:sqlite`，无原生编译依赖）的本地后端，仅提供 HTTP API 与 CLI，无前端、无 Docker、无外部在线服务。

## 功能

- JSON 描述计算图：有名输入、常量、算子节点、多输出
- 算子：`add` `sub` `mul` `div` `pow`（整数幂）`exp` `log` `sin` `cos` `abs`
- 图校验：名称全局唯一、引用必须已定义、环检测
- 前向模式方向导数、反向模式输入梯度（多入边贡献**累加**）、多输出 Jacobian、加权 VJP
- 中心有限差分校验：逐输入返回解析梯度、数值近似、绝对/相对误差、失败原因；支持全局步长与按输入步长（默认 `h = step * max(1, |x|)` 适配输入尺度）
- SQLite 持久化：图版本（内容变化自动升版本，旧版本与旧结果保留）、输入集、求值结果、梯度/校验报告；重启后可完整读取
- 明确诊断：`DIV_ZERO`、`LOG_NONPOSITIVE`、`EXP_OVERFLOW`、`ABS_NONDIFERENTIABLE`

## 不可导点策略

`abs` 在 0 处**直接报错**（`ABS_NONDIFERENTIABLE`），不返回次梯度、不返回 0、不返回 NaN/Infinity。求值（仅前向数值）在定义域错误时同样报错，绝不把 NaN/Infinity 当作正常结果。

## 环境要求

- Windows + Node.js ≥ 22.5（推荐 23+；开发环境为 Node 25，使用内置 `node:sqlite`，启动时有一条 ExperimentalWarning，属正常）
- 无需 Python / 编译工具链 / Docker

## 快速开始

```powershell
npm install
npm run build     # 编译 TypeScript 到 dist/
npm test          # 构建并运行全部测试
npm start         # 启动 HTTP 服务（默认端口 8077）
npm run demo      # 运行演示（正确校验 / 定义域失败 / 版本修改对比 / 重启读报告）
```

配置项（环境变量）：

- `PORT`：HTTP 端口，默认 `8077`
- `DATA_DIR`：数据目录（SQLite 文件所在），默认项目内 `./data`

## 图定义格式

```json
{
  "name": "poly_shared",
  "inputs": ["x"],
  "constants": { "one": 1 },
  "nodes": [
    { "name": "x2", "op": "pow", "args": ["x"], "exponent": 2 },
    { "name": "u",  "op": "add", "args": ["x2", "x"] },
    { "name": "u2", "op": "pow", "args": ["u"], "exponent": 2 },
    { "name": "f",  "op": "add", "args": ["u2", "u"] }
  ],
  "outputs": { "f": "f" }
}
```

节点按定义顺序即拓扑序；引用未定义/后定义的名字会被拒绝。输入、常量、节点名全局唯一。

## CLI

```powershell
node dist/src/cli.js examples                                  # 导入全部内置算例
node dist/src/cli.js import graph.json                         # 导入图（内容变化自动新版本）
node dist/src/cli.js validate graph.json                       # 仅校验
node dist/src/cli.js list                                      # 列出所有图版本
node dist/src/cli.js eval mimo --point '{"x":0.5,"y":2,"z":0.3}'
node dist/src/cli.js gradient mimo --point '{"x":0.5,"y":2,"z":0.3}' --weights '{"f1":1,"f2":0}'
node dist/src/cli.js jacobian mimo --point '{"x":0.5,"y":2,"z":0.3}'
node dist/src/cli.js directional mimo --point '{"x":0.5,"y":2,"z":0.3}' --direction '{"x":1,"y":1,"z":1}'
node dist/src/cli.js check poly_shared --output f --point '{"x":1}' --step 1e-5
node dist/src/cli.js check poly_shared --output f --point '{"x":10000}' --steps '{"x":1e-3}'
node dist/src/cli.js results nonlinear --version 1             # 历史报告
node dist/src/cli.js export nonlinear --format csv             # JSON/CSV 导出
```

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/graphs` | 列出所有图版本 |
| GET | `/examples` | 查看内置算例定义 |
| POST | `/graphs/import` | 导入图（body 为图定义 JSON） |
| POST | `/validate` | 校验图定义 |
| GET | `/graphs/:name?version=N` | 读取图定义 |
| GET | `/graphs/:name/results?version=N` | 读取该版本全部历史结果 |
| POST | `/graphs/:name/evaluate` | 求值，body: `{"point": {...}, "version": 1}` |
| POST | `/graphs/:name/gradient` | 反向梯度，可加 `"weights": {"f1": 1}` |
| POST | `/graphs/:name/jacobian` | 多输出 Jacobian |
| POST | `/graphs/:name/directional` | 方向导数，body 含 `"direction": {...}` |
| POST | `/graphs/:name/check` | 有限差分校验，body: `{"point":{...},"output":"f1","step":1e-5,"steps":{"x":1e-4}}` |

所有计算结果自动写入 SQLite（`data/autodiff.db`），含输入集与诊断信息。

## 内置算例

- `poly_shared`：共享子图多项式 `f = u² + u, u = x² + x`
- `mimo`：多输入多输出 `f1 = x·y + sin(z)`，`f2 = exp(x)/(y+1)`
- `nonlinear`：非线性组合 `g = log(x²+1)·cos(y) + |y|`
- `domain_error`：定义域错误算例 `h = log(x) + 1/y + |z|`

## 测试

`npm test` 覆盖：手算导数、链式法则、共享节点重复引用（能识别"两条梯度贡献被覆盖"的实现错误）、常量输出零梯度、前向/反向内积一致性（`<Jv, w> = vjp(w)·v`）、批量更换输入不串值、有限差分与按尺度步长、四类定义域诊断、图校验错误、SQLite 版本与持久化。

## 项目结构

```
src/
  types.ts     类型与诊断
  graph.ts     图编译/校验/环检测
  evaluate.ts  前向求值、方向导数、反向梯度、Jacobian、VJP
  fdcheck.ts   中心有限差分校验
  store.ts     SQLite 持久化（图版本/输入集/结果）
  service.ts   业务编排（计算 + 落库）
  server.ts    HTTP API
  cli.ts       命令行
  demo.ts      演示脚本
  examples.ts  内置算例
test/          node:test 测试
```

## 限制

- 仅标量计算图（无张量/广播）；`pow` 仅支持整数指数
- 有限差分是独立验证手段，精度受浮点舍入限制（典型相对误差 1e-6 ~ 1e-10），不用于替代反向模式
- `node:sqlite` 在 Node 25 仍标记为 experimental（API 可能微调），但功能稳定可用
- 单进程服务，无并发写锁之外的多实例部署设计
