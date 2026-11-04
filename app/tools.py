from typing import Any

INVENTORY = {
    "SKU-RED-01": {"sku": "SKU-RED-01", "name": "Red Notebook", "stock": 27},
    "SKU-BLUE-02": {"sku": "SKU-BLUE-02", "name": "Blue Notebook", "stock": 4},
}


def product_lookup(sku: str) -> dict[str, Any]:
    if sku not in INVENTORY:
        raise ValueError(f"Unknown SKU: {sku}")
    return INVENTORY[sku]


TOOL_REGISTRY = {"product_lookup": product_lookup}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool not allowed: {name}")
    return TOOL_REGISTRY[name](**arguments)
