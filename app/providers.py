import json
import httpx
from app.config import settings


class MockProvider:
    """Deterministic agent for validating benchmark orchestration, not model quality."""

    def next_action(self, task, history, model):
        task_id = task["id"]
        if "mock_answer" in task:
            for requirement in task.get("required_tool_calls", []):
                completed = any(
                    item.get("type") == "tool_result"
                    and item.get("tool") == requirement["tool"]
                    and all(item.get("result", {}).get(k) == v for k, v in requirement["arguments"].items())
                    for item in history
                )
                if not completed:
                    return {"type": "tool_call", "tool": requirement["tool"], "arguments": requirement["arguments"]}
            return {"type": "final", "answer": task["mock_answer"]}

        static_answers = {
            "math_001": "15",
            "logic_001": "10",
            "math_002": "25",
            "logic_002": "Cara",
            "planning_001": "implementation, testing, deployment",
            "planning_002": "10:10",
        }
        if task_id in static_answers:
            return {"type": "final", "answer": static_answers[task_id]}

        tool_results = [
            item for item in history
            if item.get("type") == "tool_result"
            and item.get("tool") == "product_lookup"
        ]
        tool_errors = [
            item for item in history
            if item.get("type") == "tool_error"
            and item.get("tool") == "product_lookup"
        ]
        looked_up = {
            item["result"]["sku"]
            for item in tool_results
        }

        if task_id == "multi_001":
            if "SKU-BLUE-02" not in looked_up:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-BLUE-02"},
                }
            return {"type": "final", "answer": "REORDER"}

        if task_id == "multi_002":
            if "SKU-RED-01" not in looked_up:
                return {
                    "type": "tool_call",
                    "tool": "product_lookup",
                    "arguments": {"sku": "SKU-RED-01"},
                }
            return {"type": "final", "answer": "STOCK_OK"}

        if task_id == "multi_003":
            for sku in ("SKU-BLUE-02", "SKU-RED-01"):
                if sku not in looked_up:
                    return {
                        "type": "tool_call",
                        "tool": "product_lookup",
                        "arguments": {"sku": sku},
                    }
            stock_by_sku = {
                item["result"]["sku"]: item["result"]["stock"]
                for item in tool_results
            }
            winner = max(stock_by_sku, key=stock_by_sku.get)
            return {"type": "final", "answer": winner}

        if task.get("category") == "failure_recovery":
            initial = task["initial_lookup"]
            recovery = task["required_recovery"]
            initial_sku = initial["arguments"]["sku"]
            recovery_sku = recovery["arguments"]["sku"]
            if not tool_errors and not looked_up:
                return {
                    "type": "tool_call",
                    "tool": initial["tool"],
                    "arguments": initial["arguments"],
                }
            if recovery_sku not in looked_up:
                return {
                    "type": "tool_call",
                    "tool": recovery["tool"],
                    "arguments": recovery["arguments"],
                }
            recovered = next(
                item["result"] for item in reversed(tool_results)
                if item["result"]["sku"] == recovery_sku
            )
            if task_id == "recovery_001":
                answer = "AVAILABLE" if recovered["stock"] > 0 else "OUT_OF_STOCK"
            elif task_id == "recovery_002":
                answer = "REORDER" if recovered["stock"] < 10 else "STOCK_OK"
            else:
                answer = "STOCK_OK" if recovered["stock"] >= 10 else "REORDER"
            return {"type": "final", "answer": answer}

        return {"type": "final", "answer": ""}


class OllamaProvider:
    def next_action(self, task, history, model):
        allowed_tools = (
            "product_lookup(sku: string) — returns product name and stock count."
            if "product_lookup" in task.get("tools", [])
            else "No tools are available for this task."
        )
        tool_observations = [
            item for item in history
            if item.get("type") in {"tool_result", "tool_error"}
        ]
        recovery_guidance = ""
        if any(item.get("type") == "tool_error" for item in history):
            recovery_guidance = """
FAILURE-RECOVERY RULES:
- A tool_error means the requested information was NOT retrieved. It does not tell you the product's stock or availability.
- Never infer a stock value, product status, or final answer from a failed lookup.
- Read the task instructions again and perform the specified recovery action using a different, valid argument.
- If the task names a fallback lookup, you MUST make that tool_call before giving a final answer.
- After the fallback tool_result arrives, base your final answer only on the returned fields and the task's condition.
- Once the required fallback result is present, do not make any more tool calls. Return the final answer immediately using that result.
- Do not return a final answer while a required recovery lookup is still outstanding.
"""
        decision_guidance = ""
        if task.get("tools"):
            decision_guidance = """
TOOL-RESULT DECISION RULES:
- Do not treat a successful lookup as automatically meaning STOCK_OK.
- Extract the exact numeric stock value from the latest successful lookup relevant to the task.
- Apply the comparison in the task literally: "below N" means value < N; "above N" means value > N; "at least N" means value >= N.
- Select the branch whose condition is true. For example, a stock value of 4 is below 10, not at least 10.
- Do not copy an answer pattern from another task. Re-evaluate the current task's condition every time.
"""
        reasoning_guidance = ""
        if not task.get("tools"):
            reasoning_guidance = """
REASONING AND VERIFICATION RULES:
- Solve the task carefully before producing the final answer. Do not guess or rush.
- For count problems with multiple groups, add every group's quantity first to get the total; then subtract only the items explicitly removed. Do not subtract removed items from just one group unless the question asks for that group's count.
- For fraction changes, first calculate the changed amount from the original total (total divided by the denominator for a unit fraction); subtract borrowed/removed items, then apply returns or additions in the order stated.
- For elapsed-time problems, convert the start time to minutes after midnight, add every duration including breaks, then convert the final total back to HH:MM. Do not omit a break or round a duration.
- For elapsed-time problems, maintain a running total after EACH duration; check that the difference between final and start time equals the sum of all durations.
- For percentage discounts, calculate the discount amount first (price × percent ÷ 100), then subtract it from the original price. Do not return the discount amount as the sale price.
- For proportional ratios, calculate the amount per base unit first, then multiply by the target unit count.
- For relational logic, translate each comparison into an ordering before selecting the answer. For statements using "all" and "some", do not assume that a subset overlaps with a separate group unless the premises establish that overlap.
- Before finalizing arithmetic, write down the numeric expression internally, evaluate it left to right, and independently check the result. Never output the working unless asked.
- If the answer is a time, verify by adding the total elapsed minutes back to the start time.
- Return only the exact requested answer format.
- Return only the exact requested answer format. Do not include workings unless asked.
"""
        recovery_requirement = task.get("required_recovery")
        recovery_requirement_text = (
            json.dumps(recovery_requirement)
            if recovery_requirement else "None"
        )
        required_tool_calls = task.get("required_tool_calls", [])
        completed_tool_calls = [
            item for item in history if item.get("type") == "tool_result"
        ]
        pending_tool_calls = [
            requirement for requirement in required_tool_calls
            if not any(
                item.get("tool") == requirement["tool"]
                and all(
                    item.get("result", {}).get(key) == value
                    for key, value in requirement["arguments"].items()
                )
                for item in completed_tool_calls
            )
        ]
        required_tool_calls_text = (
            json.dumps(pending_tool_calls) if pending_tool_calls else "None"
        )
        retry_guidance = ""
        if any(item.get("type") == "answer_retry_feedback" for item in history):
            retry_guidance = """
ANSWER SELF-CORRECTION:
- Your previous answer was rejected by the benchmark's answer check.
- Do not repeat it automatically. Re-read the task and observations from the beginning.
- For arithmetic, recompute each operation independently and check the result.
- For time calculations, add all durations (including breaks) in minutes before converting to HH:MM.
- For inventory decisions, identify the exact returned stock value and evaluate the stated inequality.
- For recovery tasks, use the successful fallback result, not the failed lookup.
- Return one corrected final answer in the required format.
"""
        forced_final_guidance = ""
        if any(
            item.get("type") == "tool_call_blocked_observations_complete"
            for item in history
        ):
            forced_final_guidance = """
FINAL ANSWER REQUIRED:
- The engine has blocked an unnecessary tool call because all required observations are already available.
- Do not call any tool again. Your next action MUST be a final action.
- Read the successful tool_result observations above and apply the condition in the original task.
- For recovery tasks, use the successful fallback lookup result, not the failed initial lookup.
- For comparison tasks, compare the returned stock values and return the requested SKU.
"""
        multi_tool_guidance = ""
        if required_tool_calls:
            multi_tool_guidance = """
MULTI-STEP TOOL REQUIREMENTS:
- Complete every listed required tool call; one successful lookup is not enough.
- After each result, make the next outstanding lookup.
- Compare actual returned values only after all required observations are available.
- Do not guess missing values or return a final answer early.
"""
        comparison_guidance = ""
        if task.get("id") == "multi_003":
            comparison_guidance = """
NUMERIC COMPARISON TASK:
- Identify the exact comparison requested by the task (more, less, highest, lowest, etc.).
- Read the numeric stock field from EVERY required successful product_lookup result.
- Compare the returned integers directly, then select the SKU that satisfies the requested comparison.
- Do not infer stock from SKU names, lookup order, product names, or prior examples.
- Do not select the first or last SKU automatically.
"""
        prompt = f"""
You are a reliable agent completing one benchmark task.
TASK: {task["prompt"]}
REQUIRED RECOVERY ACTION AFTER A TOOL ERROR: {recovery_requirement_text}
REQUIRED TOOL CALLS STILL OUTSTANDING: {required_tool_calls_text}
ALLOWED TOOLS: {allowed_tools}
TOOL OBSERVATIONS (authoritative; use these instead of repeating calls): {json.dumps(tool_observations)}
FULL ACTION HISTORY: {json.dumps(history)}
{reasoning_guidance}
{decision_guidance}
{recovery_guidance}
{multi_tool_guidance}
{comparison_guidance}
{retry_guidance}
{forced_final_guidance}

STRICT ACTION RULES:
- If no tools are available, return a final action. Never call any tool.
- Only call a tool explicitly listed in ALLOWED TOOLS. Never invent tools.
- A tool_result already contains the answer from that tool. Read its result fields and reason from them.
- If a tool observation answers the task, return a final action now unless REQUIRED TOOL CALLS STILL OUTSTANDING lists more calls.
- If required tool calls are outstanding, execute the next listed call before returning a final action.
- If an observation has "replayed": true, it is the cached result of an earlier call. Treat it exactly like a normal tool_result or tool_error, and do not retry it.
- Never repeat a tool call with identical arguments, even if the previous call was suppressed.
- If a tool returns an error, follow the task's recovery instruction using different arguments. Do not retry the same failed arguments.
- When the observations contain all information needed, return a final answer immediately.
- Return exactly one of these schemas:
{{"type":"tool_call","tool":"product_lookup","arguments":{{"sku":"SKU-BLUE-02"}}}}
{{"type":"final","answer":"your concise answer"}}
Do not invent tool results or claim a tool was called unless you return a tool_call action.
"""
        response = httpx.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            if not task.get("tools"):
                return {"type": "final", "answer": raw}
            raise ValueError(f"Model did not return valid JSON: {raw[:500]}")

        if not isinstance(action, dict):
            raise ValueError(f"Model action must be a JSON object: {raw[:500]}")

        action_type = action.get("type") or action.get("action")
        if action_type in {"final_answer", "answer", "respond"}:
            action_type = "final"
        if action_type in {"tool", "call_tool", "function_call"}:
            action_type = "tool_call"

        if action_type == "final":
            answer = action.get("answer", action.get("content", action.get("final_answer", "")))
            answer = str(answer)
            return {"type": "final", "answer": answer}

        if action_type == "tool_call":
            tool_name = action.get("tool", action.get("tool_name", action.get("name")))
            arguments = action.get("arguments", action.get("tool_input", action.get("parameters", {})))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    raise ValueError(f"Tool arguments were not valid JSON: {arguments[:300]}")
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool arguments must be a JSON object: {raw[:500]}")
            return {"type": "tool_call", "tool": tool_name, "arguments": arguments}

        nested = action.get("tool_call")
        if isinstance(nested, dict):
            return {
                "type": "tool_call",
                "tool": nested.get("tool", nested.get("name")),
                "arguments": nested.get("arguments", nested.get("parameters", {})),
            }

        raise ValueError(f"Model returned an unsupported action schema: {raw[:500]}")


def groq_chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_completion_tokens: int = 2048,
    response_schema: dict | None = None,
) -> str:
    """Call Groq's OpenAI-compatible Chat Completions API and return message content."""
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it to your local .env file "
            "or to the deployment environment variables."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "agentbench_response",
                "strict": True,
                "schema": response_schema,
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}
    # These reasoning controls are documented for GPT-OSS and Qwen3.8 models.
    if model.startswith("openai/gpt-oss") or model.startswith("qwen/qwen3.8"):
        payload["reasoning_effort"] = "medium"

    try:
        response = httpx.post(
            f"{settings.groq_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=max(settings.request_timeout_seconds, 90),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        try:
            error_data = exc.response.json()
            detail = (
                error_data.get("error", {}).get("message")
                if isinstance(error_data, dict) else None
            )
        except (ValueError, TypeError):
            detail = None
        detail = str(detail or "No error details returned by Groq.")
        if status_code == 401:
            message = "Groq rejected GROQ_API_KEY (401). Check that the key is valid."
        elif status_code == 403:
            message = (
                f"Groq denied access to model '{model}' (403). Check model access, "
                "project permissions, and account limits."
            )
        elif status_code == 404:
            message = f"Groq model or endpoint was not found (404): {model}."
        elif status_code == 429:
            message = (
                "Groq rate limit or quota reached (429). Check your Groq usage "
                "and plan limits, then retry."
            )
        else:
            message = f"Groq API request failed ({status_code})."
        raise ValueError(f"{message} {detail}") from exc
    except httpx.TimeoutException as exc:
        raise ValueError(
            "Groq request timed out. Increase REQUEST_TIMEOUT_SECONDS or retry."
        ) from exc
    except httpx.RequestError as exc:
        raise ValueError(f"Could not reach Groq API: {exc}") from exc

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("Groq returned an unexpected Chat Completions response.") from exc

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty assistant message.")
    return content.strip()


class GroqProvider:
    """Hosted Groq agent provider. No local model runtime is required."""

    def next_action(self, task, history, model):
        allowed_tools = task.get("tools", [])
        required_calls = task.get("required_tool_calls", [])
        completed_results = [
            item for item in history if item.get("type") == "tool_result"
        ]
        pending_calls = [
            requirement for requirement in required_calls
            if not any(
                item.get("tool") == requirement["tool"]
                and all(
                    item.get("result", {}).get(key) == value
                    for key, value in requirement["arguments"].items()
                )
                for item in completed_results
            )
        ]
        recovery = task.get("required_recovery")
        recovery_completed = bool(
            recovery and any(
                item.get("type") == "tool_result"
                and item.get("tool") == recovery["tool"]
                and all(
                    item.get("result", {}).get(key) == value
                    for key, value in recovery["arguments"].items()
                )
                for item in completed_results
            )
        )
        previous_retry = next(
            (item for item in reversed(history)
             if item.get("type") == "answer_retry_feedback"),
            None,
        )

        system_prompt = (
            "You are an accurate tool-using benchmark agent. Follow the task literally. "
            "Use only supplied observations; never invent tool results. "
            "Return one JSON object and no markdown. For a final answer, use "
            "{\"type\":\"final\",\"answer\":\"exact answer\","
            "\"evidence\":\"one-sentence verification summary\"}. "
            "For a tool call, use "
            "{\"type\":\"tool_call\",\"tool\":\"product_lookup\","
            "\"arguments\":{\"sku\":\"SKU-RED-01\"}}. "
            "Evidence is a concise audit note, not hidden chain-of-thought."
        )

        prompt = f"""
BENCHMARK TASK
{task["prompt"]}

ALLOWED TOOLS
{json.dumps(allowed_tools)}

REQUIRED TOOL CALLS STILL OUTSTANDING
{json.dumps(pending_calls) if pending_calls else "None"}

REQUIRED RECOVERY ACTION
{json.dumps(recovery) if recovery and not recovery_completed else "None"}

SUCCESSFUL TOOL OBSERVATIONS
{json.dumps(completed_results, ensure_ascii=False)}

FULL ACTION HISTORY
{json.dumps(history, ensure_ascii=False)}

DECISION RULES
- If REQUIRED TOOL CALLS STILL OUTSTANDING is not None, call the next outstanding tool now.
- If a lookup failed, its error contains no product stock. Perform the specified recovery lookup using its different SKU.
- After a successful fallback lookup, apply the condition in the original task to that returned stock value. Do not carry over the answer from another recovery task.
- For stock rules, apply inequalities literally: below N is < N; above N is > N; at least N is >= N; greater than N is > N; N or less is <= N.
- For arithmetic, write the numeric expression out before evaluating it. For a fraction change, calculate the fraction of the original total, subtract it, then apply returns. For a discount, subtract the discount amount from the original price. For a ratio, calculate one unit's amount before scaling.
- For time questions, add every duration and break in minutes to the start time, then convert once to HH:MM. Check the total elapsed minutes.
- For ordering questions, respect every stated dependency. For "all" and "some" logic, do not assume two groups overlap unless the premises prove it.
- The final answer must follow the exact format requested in the task. Put any brief calculation or observation check in the evidence field, not in the answer.

{(
  "RETRY REQUIRED: Your previous answer was rejected. "
  f"Previous answer: {previous_retry.get('candidate_answer', '')}. "
  "Do not repeat it. Re-evaluate the task and observations, and give a corrected answer."
) if previous_retry else ""}

NEXT ACTION
Return one tool_call if a required observation is missing. Otherwise return one final answer.
"""

        can_call_product_lookup = "product_lookup" in allowed_tools
        required_action_fields = [
            "type", "answer", "evidence", "tool", "arguments"
        ]
        nullable_arguments_schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        }
        response_schema = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["final", "tool_call"] if can_call_product_lookup
                    else ["final"],
                },
                "answer": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "evidence": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "tool": {
                    "anyOf": [
                        {"type": "string", "enum": ["product_lookup"]},
                        {"type": "null"},
                    ] if can_call_product_lookup else [{"type": "null"}]
                },
                "arguments": (
                    nullable_arguments_schema
                    if can_call_product_lookup else {"type": "null"}
                ),
            },
            "required": required_action_fields,
            "additionalProperties": False,
        }
        raw = groq_chat_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_completion_tokens=2048,
            response_schema=(
                response_schema
                if model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b",
                             "qwen/qwen3.8-27b"}
                else None
            ),
        )
        try:
            action = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Groq did not return valid JSON: {raw[:500]}") from exc

        if not isinstance(action, dict):
            raise ValueError("Groq action must be a JSON object.")

        action_type = action.get("type") or action.get("action")
        if action_type in {"final_answer", "answer", "respond"}:
            action_type = "final"
        if action_type in {"tool", "call_tool", "function_call"}:
            action_type = "tool_call"
        if not action_type and "answer" in action:
            action_type = "final"

        if action_type == "final":
            return {
                "type": "final",
                "answer": str(action.get("answer") or action.get("content") or ""),
                "evidence": str(action.get("evidence") or action.get("reasoning") or ""),
            }

        if action_type == "tool_call":
            tool_name = action.get("tool", action.get("tool_name", action.get("name")))
            arguments = action.get(
                "arguments", action.get("tool_input", action.get("parameters", {}))
            )
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("Groq returned invalid JSON tool arguments.") from exc
            if not isinstance(arguments, dict):
                raise ValueError("Groq tool-call arguments must be a JSON object.")
            return {
                "type": "tool_call",
                "tool": tool_name,
                "arguments": arguments,
            }

        # Convert an empty object into an empty final so the engine's bounded
        # answer-retry mechanism can recover rather than recording a parser crash.
        if not action:
            return {
                "type": "final",
                "answer": "",
                "evidence": "The model returned an empty action object.",
            }
        raise ValueError(f"Groq returned an unsupported action schema: {raw[:500]}")


def get_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "groq":
        return GroqProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unsupported provider: {name}")
