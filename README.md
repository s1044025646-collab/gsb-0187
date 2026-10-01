# 标量计算图自动微分与梯度校验后端

一个只提供 **HTTP API + CLI** 的本地后端，用 TypeScript、Node.js 与内置
SQLite（`node:sqlite`）实现。针对小规模**标量**计算图做数值求值、
前向模式方向导数、反向模式输入梯度、多输出 Jacobian、向量–Jacobian 乘积，
并用中心有限差分对解析梯度做独立校验。

不做编程语言解析、不含神经网络训练、不使用 GPU，**不调用任何自动微分库**，
核心微分算法全部手写；服务端也从不使用 `eval` 执行用户文本。

## 支持的算子

| 算子 | 含义 | 分类 |
| --- | --- | --- |
| `add` / `sub` / `mul` / `div` | 加 / 减 / 乘 / 除 | 二元（需要 `a`,`b`） |
| `pow` | 整数幂 `a^b`，`b` 为整数（建议用命名常量） | 二元 |
| `exp` / `log` / `sin` / `cos` / `abs` | 指数 / 自然对数 / 正弦 / 余弦 / 绝对值 | 一元（只需 `a`） |

### 不可导点与数值安全约定

- `div`：分母为 0 → 返回 `DIV_BY_ZERO`，不产生 `Infinity`。
- `log`：输入 `<= 0` → 返回 `LOG_DOMAIN`。
- `exp`：参数超过安全上限 709 → 返回 `EXP_OVERFLOW`，不产生 `Infinity`。
- `pow`：仅支持整数幂；`0` 的非正整数幂、非整数指数 → `POW_DOMAIN`。
- `abs`：分段绝对值。前向**求值**允许 `abs(0)=0`；但在 0 点（尖点）
  **拒绝求导**，返回 `ABS_KINK`。策略是“显式失败”，绝不把 `NaN`、
  `Infinity` 或任意 0 当作正常导数返回。非零处取符号 `±1`。
- 任何节点若计算出非有限值，都会得到 `NUMERIC_OVERFLOW` 诊断。

## 图的 JSON 描述

```json
{
  "name": "poly_shared",
  "inputs": ["x", "y"],
  "constants": { "two": 2, "three": 3 },
  "nodes": {
    "x2": { "op": "pow", "a": "x", "b": "two" },
    "xy": { "op": "mul", "a": "x", "b": "y" },
    "shared": { "op": "add", "a": "x2", "b": "xy" },
    "f": { "op": "add", "a": "shared", "b": "three" },
    "g": { "op": "mul", "a": "shared", "b": "y" }
  },
  "outputs": ["f", "g"]
}
```

规则：

- 输入名、常量名、节点名全局唯一；重复 → `DUPLICATE_NAME`。
- 节点只能引用已定义的输入 / 常量 / 节点；悬空引用 → `UNKNOWN_REFERENCE`。
- 依赖图必须无环；环 → `CYCLE_DETECTED`（Kahn 拓扑排序并列出卡住的节点）。
- 公共子表达式天然复用：节点只按拓扑序计算一次，值在整张图内共享；
  反向传播对同一节点的多条入边使用**累加**（`+=`），不会覆盖。
- 结构缓存（校验后的拓扑序）与某次输入的数值（节点取值 / 伴随值）严格分离，
  更换输入时从零重建，梯度不会跨请求残留。

示例图见 [`examples/`](examples/)。

## 环境要求

- Windows 本地运行；Node.js **>= 22.5**（内置 `node:sqlite`，开发环境为 Node 25）。
- 无需 Docker、WSL、付费服务或在线 API；运行时零第三方依赖。

## 构建、测试、启动、演示

```powershell
npm install        # 仅安装 TypeScript / tsx / 类型（开发依赖）
npm run build      # 编译到 dist/
npm test           # 运行 node:test 测试套件（16 项）
npm run demo       # 端到端演示（正确校验 / 定义域失败 / 版本比较 / 重启读取）

# HTTP 服务（默认端口 3000，可用 PORT 或 --port 覆盖）
npm start
# 或
node --experimental-sqlite dist/cli.js serve --port 8080
```

数据目录默认是项目内的 `data/`（SQLite 文件 `data/autodiff.db`），
可用环境变量 `DATA_DIR` 或 CLI 的 `--data-dir` 覆盖。

## CLI 用法

```powershell
node --experimental-sqlite dist/cli.js import --file examples/polynomial.json
node --experimental-sqlite dist/cli.js list
node --experimental-sqlite dist/cli.js validate --name poly_shared
node --experimental-sqlite dist/cli.js eval     --name poly_shared --inputs examples/inputs_poly.json
node --experimental-sqlite dist/cli.js gradient --name poly_shared --inputs examples/inputs_poly.json --format csv
node --experimental-sqlite dist/cli.js directional --name poly_shared --inputs examples/inputs_poly.json --direction '{"x":1,"y":0}'
node --experimental-sqlite dist/cli.js vjp --name poly_shared --inputs examples/inputs_poly.json --weights '{"f":1,"g":1}'
node --experimental-sqlite dist/cli.js check --name poly_shared --inputs examples/inputs_poly.json --step 1e-5
node --experimental-sqlite dist/cli.js report --id 1            # 重启后读取历史报告
node --experimental-sqlite dist/cli.js report --id 1 --format csv
```

`--inputs` / `--direction` / `--weights` 既可以是 JSON 文件路径，也可以是
内联 JSON 字符串。`--version N` 指定历史图版本；缺省使用最新版本。

## HTTP API

所有请求 / 响应为 JSON。业务 / 定义域错误返回 HTTP 422，结构为
`{ "error": { "code", "message", "detail" } }`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET`  | `/health` | 健康检查 |
| `GET`  | `/graphs` | 图与最新版本列表 |
| `GET`  | `/graphs/:name/versions` | 某图的全部版本 |
| `POST` | `/graphs` | 导入图（定义相同复用版本；内容变化创建新版本） |
| `POST` | `/graphs/:name/validate` | 校验引用 / 环 / 唯一性 |
| `POST` | `/graphs/:name/eval` | 前向求值 |
| `POST` | `/graphs/:name/gradient` | 反向模式 Jacobian |
| `POST` | `/graphs/:name/directional` | 前向方向导数（body 含 `direction`） |
| `POST` | `/graphs/:name/vjp` | 向量–Jacobian 乘积（body 含 `weights`） |
| `POST` | `/graphs/:name/check` | 中心有限差分校验（body 含 `options`） |
| `GET`  | `/reports/:id` | 读取持久化报告（支持 `?format=csv`） |

请求体可带 `version` 选择历史版本。`eval` / `gradient` / `check` /
`reports` 支持 `?format=csv` 导出 CSV。

`check` 请求示例：

```json
{
  "inputs": { "x": 1000000, "y": 0.000001 },
  "options": {
    "baseStep": 1e-5,
    "perInputStep": { "y": 1e-8 },
    "absTol": 1e-6,
    "relTol": 1e-4
  }
}
```

### 有限差分（独立验证）

校验使用中心差分 `(f(x+h) - f(x-h)) / (2h)`，与反向模式完全独立：
解析梯度来自反向传播，数值梯度来自重新求值，二者不共享代码路径。
每个输入–输出对返回：解析值、数值近似、绝对误差、相对误差、所用步长、
是否通过以及失败原因（例如差分点触碰 `LOG_DOMAIN`）。

步长按输入尺度自动缩放：`h_i = max(baseStep, baseStep·|x_i|)`，
也可用 `perInputStep` 单独指定（缓解输入尺度差异）。
有限差分结果**只作为验证**展示，绝不会被标记或冒充为反向模式梯度。

## 持久化与版本

SQLite 三张核心表：

- `graph_versions`：同名图每次定义变化生成新版本（相同定义复用）。
- `input_sets`：每次计算的输入 / 方向 / 权重集合。
- `reports`：求值、梯度、方向导数、VJP、校验报告（JSON 文本）。

修改图只新增版本，**旧版本与其报告永久保留**；进程重启后可用
`report --id` 或 `/reports/:id` 读取完整报告，并可对旧版本重新计算。

## 测试覆盖

- 可手算的多项式导数与链式法则（`exp(log x)=x`）。
- 共享节点被多条边引用时梯度累加，并**显式构造“覆盖式”错误实现**
  证明其得到错误值（5 而正确为 10），从而可识别覆盖类 bug。
- 常量输出相对输入为零梯度。
- 前向方向导数与反向 VJP 的内积一致性：`wᵀ(Jd) = (Jᵀw)ᵀd`。
- 批量更换输入后梯度不串值，重复计算无残留。
- 除零 / log 非正 / exp 溢出 / abs 零尖点均给出明确错误码。
- 有限差分通过、尺度差异步长、以及差分触碰定义域时报告失败原因。
- SQLite 版本化与“重启”后历史结果可读。

## 设计与限制

- 面向小规模标量图：不做向量化 / 矩阵算子 / GPU。
- `pow` 仅支持整数常量式指数（覆盖多项式等需求），不支持可变底数-指数联合求导。
- 数值精度受 IEEE-754 双精度与有限差分截断 / 舍入误差限制，容差可在 `options` 配置。
- `node:sqlite` 在当前 Node 版本会打印一条实验性警告（功能稳定可用），
  用 `--experimental-sqlite` 显式启用。
