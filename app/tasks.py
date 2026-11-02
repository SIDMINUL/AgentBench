TASKS = [
    {
        "id": "math_001",
        "category": "reasoning",
        "prompt": "A shop has 12 notebooks, sells 5, then receives 8. How many now? Reply with the number only.",
        "expected": "15",
        "tools": [],
        "max_steps": 3,
    },
    {
        "id": "logic_001",
        "category": "reasoning",
        "prompt": "A box has 9 red and 4 blue pens. Three red pens are removed. How many pens remain? Reply with the number only.",
        "expected": "10",
        "tools": [],
        "max_steps": 3,
    },
    {
        "id": "multi_001",
        "category": "tool_use",
        "prompt": "Use product_lookup to find the stock for SKU-BLUE-02. If stock is below 10, answer REORDER; otherwise answer STOCK_OK.",
        "expected": "REORDER",
        "tools": ["product_lookup"],
        "max_steps": 4,
    },
]
