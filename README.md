# 剧场扬声器共享 EQ 试算 API

剧场扬声器只能加载**一组**均衡参数。本服务用 FastAPI + SciPy 试算一枚 **32 抽头因果 FIR**，让它同时服务于 2～4 个座位：

- 输入各座位**等长**脉冲响应（采样率相同、已按直达声起点对齐、每条 ≤ 256 点）、目标延时（0～31 样点）、增益上限（≥ 1）；
- 约束 FIR 系数的 L1 范数（绝对值之和）不超过增益上限；
- 以**全长卷积** `h_k * g` 与延时单位脉冲的**均方误差（MSE）**评估每个座位；
- 与同延时直通（单位脉冲 FIR）相比，**每个座位的误差都不得增加**，并在此前提下**最小化最差座位误差**；
- **仅当最差误差确实降低时**才返回系数和各座位前后误差；`no_improvement`（无改善）与 `failed`（求解失败）分别报告。

## 数学模型

设座位 `k` 的脉冲响应为 `h_k`（长 `L`），FIR 为 `g`（32 抽头），全长卷积长度 `N = L + 31`，目标为 `d[n] = δ[n - n0]`（`n0` 为目标延时）：

```
minimize   max_k  (1/N) · ‖ h_k * g − d ‖²
subject to Σ|g_i| ≤ gain_limit
           MSE_k(g) ≤ MSE_k([1,0,…,0])   ∀k      （各座位不得变差）
```

这是凸优化问题（凸二次函数的最大值 + 凸约束），以 epigraph 形式用
`scipy.optimize.minimize(method="SLSQP")` 求解（解析梯度）。直通解
`[1,0,…,0]` 的 L1 范数为 1 ≤ gain_limit，因此约束集恒非空。返回前会对
候选解重新精确核算：L1 超限则按比例缩回边界，各座位误差用全长卷积重算，
并再次校验"不变差"与"最差误差严格下降"两个条件。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/eq/optimize` | 试算共享 FIR（三种结果状态见下） |
| GET  | `/v1/eq/example`  | 返回一个可直接 POST 的示例请求体 |
| GET  | `/health`         | 健康检查 |
| GET  | `/docs`           | Swagger UI 交互式文档 |

### 请求体

```json
{
  "impulse_responses": [[1.0, 0.5], [1.0, -0.5]],
  "target_delay": 1,
  "gain_limit": 2.0,
  "sample_rate": 48000
}
```

- `impulse_responses`：2～4 条等长脉冲响应，每条 1～256 点，数值须有限；
- `target_delay`：整数，0～31；
- `gain_limit`：≥ 1；
- `sample_rate`：可选，仅作元数据（各条脉冲响应采样率本就相同）。

### 响应（`status` 判别字段，HTTP 均为 200；参数非法为 422）

- `improved`：返回 `coefficients`（32 个系数）、`l1_norm`、各座位
  `error_before`/`error_after`、`worst_error_before`/`worst_error_after` 等；
- `no_improvement`：不返回系数，只给出各座位直通基线误差与说明；
- `failed`：求解器异常/未收敛，返回 `detail` 说明。

## 运行

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务监听 http://127.0.0.1:8000 ，文档在 http://127.0.0.1:8000/docs
```

### 本地开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

### 演示

```bash
./examples/demo.sh                 # 依次演示 improved / no_improvement / 422
curl -X POST http://127.0.0.1:8000/v1/eq/optimize \
  -H 'Content-Type: application/json' \
  -d @examples/request_improved.json
```

### 测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
app/
  main.py      # FastAPI 应用、路由、422 异常处理
  schemas.py   # 请求/响应模型（Pydantic 校验）
  solver.py    # 凸优化求解（SciPy SLSQP，epigraph 形式）
tests/test_api.py
examples/      # 可直接 POST 的示例请求与 demo 脚本
Dockerfile
docker-compose.yml
requirements.txt / requirements-dev.txt
```

## 设计说明与限制

- MSE 在全长卷积（`L + 31` 点）上取平均；目标为单位幅度延时脉冲。
- "改善"判定阈值：`worst_after < worst_before − 1e-9·max(1, worst_before)`，
  避免把数值噪声当作改善。
- 若求解器未收敛但最终迭代点经核验可行且严格更优，仍会返回 `improved`
  （`solver_success=false` 并在 `detail` 中注明）；无法得到可信结果时返回 `failed`。
- 单线程同步计算：问题规模很小（33 个变量），单次请求通常在几十毫秒内。
- `sample_rate` 不参与计算，仅作元数据；如需按频率加权或限制相位，需要扩展模型。
