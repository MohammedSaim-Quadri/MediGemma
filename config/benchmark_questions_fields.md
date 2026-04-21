# benchmark_questions.yaml 字段用途说明

本文档说明 `config/benchmark_questions.yaml` 中各字段的用途，区分哪些字段会作为大模型输入、哪些仅用于内部流程。

## 作为大模型输入的字段（2 个）

| 字段 | 位置 | 说明 |
|------|------|------|
| `uncertainty_instruction` | 顶层 | 不确定性处理指令，作为前缀拼接到每个问题前面 |
| `questions[].question` | 每个问题条目 | 具体的临床提问文本 |

### 拼接逻辑

在 `scripts/run_benchmark.py:load_questions_from_yaml()` 中实现：

```python
# scripts/run_benchmark.py L111-123
def load_questions_from_yaml(yaml_path: str) -> list[str]:
    uncertainty = data.get("uncertainty_instruction", "").strip()
    for q in data["questions"]:
        text = q["question"].strip()
        if uncertainty:
            text = f"{uncertainty}\n\n{text}"
        questions.append(text)
```

即每个发送给模型的 prompt 实际内容为：

```
{uncertainty_instruction}

{question}
```

## 不作为大模型输入的字段

| 字段 | 用途 |
|------|------|
| `version` | YAML 文件版本号 |
| `total_questions` | 问题总数（文档性质） |
| `questions[].id` | 问题标识（Q1-Q9），用于结果记录和评估报告 |
| `questions[].module` | 模块名称（如 "Wound Identification & Classification"），写入输出 JSONL |
| `questions[].report_section` | 报告章节标题，Markdown 报告生成时使用 |
| `questions[].expected_fields` | 期望的结构化输出 schema，用于解析模型回答和评估对比 |
| `questions[].evaluation_focus` | 评估重点，供 LLM-as-Judge 评分时参考（见 `config/eval_rubric.md`） |
| `json_output_schema` | 输出 JSONL 文件的结构定义 |

## 数据流示意

```
benchmark_questions.yaml
        │
        ├─→ uncertainty_instruction ──┐
        │                             ├─→ 拼接为 prompt 文本 ──→ 大模型推理
        ├─→ questions[].question ─────┘
        │
        ├─→ questions[].id / module ──→ 写入输出 JSONL 记录
        │
        ├─→ questions[].expected_fields ──→ 结果解析 & 自动化评估
        │
        ├─→ questions[].evaluation_focus ──→ LLM-as-Judge 评分参考
        │
        └─→ json_output_schema ──→ 输出格式验证
```
