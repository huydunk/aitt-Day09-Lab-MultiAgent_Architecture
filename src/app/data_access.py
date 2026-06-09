from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ShoppingDataStore:
    """Mock-data store with fast index-based lookups."""

    def __init__(self, json_path: Path) -> None:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        self.metadata: dict[str, Any] = raw["metadata"]

        self.customers: dict[str, Any] = {c["customer_id"]: c for c in raw["customers"]}

        self.orders: dict[str, Any] = {o["order_id"]: o for o in raw["orders"]}

        self.orders_by_customer: dict[str, list[dict[str, Any]]] = {}
        for order in raw["orders"]:
            self.orders_by_customer.setdefault(order["customer_id"], []).append(order)

        self.vouchers_by_customer: dict[str, list[dict[str, Any]]] = {}
        for voucher in raw["vouchers"]:
            self.vouchers_by_customer.setdefault(voucher["customer_id"], []).append(voucher)

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        customer = self.customers.get(customer_id)
        if customer is None:
            return {"status": "not_found", "customer_id": customer_id}
        return {"status": "ok", "customer": customer}

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        orders = self.orders_by_customer.get(customer_id, [])
        if not orders:
            return {"status": "not_found", "customer_id": customer_id}
        sorted_orders = sorted(orders, key=lambda o: o.get("created_at", ""), reverse=True)
        return {"status": "ok", "customer_id": customer_id, "orders": sorted_orders[:limit]}

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            return {"status": "not_found", "order_id": order_id}
        return {"status": "ok", "order": order}

    def get_vouchers_by_customer_id(
        self,
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        vouchers = self.vouchers_by_customer.get(customer_id, [])
        if not vouchers:
            return {"status": "not_found", "customer_id": customer_id}
        if only_active:
            vouchers = [v for v in vouchers if v.get("status") == "active"]
        return {"status": "ok", "customer_id": customer_id, "vouchers": vouchers}


def build_data_tools(store: ShoppingDataStore) -> list:
    from langchain_core.tools import tool

    @tool
    def get_customer_by_id(customer_id: str) -> dict:
        """Look up a customer's profile (name, tier, loyalty points, account status)
        by their customer ID (e.g. 'C001'). Use this when the user asks about
        a specific customer's information."""
        return store.get_customer_by_id(customer_id)

    @tool
    def get_orders_by_customer_id(customer_id: str) -> dict:
        """Get the most recent orders placed by a customer, given their customer ID
        (e.g. 'C001'). Use this when the user asks for a customer's order history
        or wants to know what orders they have."""      
        return store.get_orders_by_customer_id(customer_id)

    @tool
    def get_order_detail_by_order_id(order_id: str) -> dict:
        """Get full details of a single order by its order ID (e.g. '1971').
        Returns status, items, shipping, payment, return eligibility and more.
        Use this when the user asks about a specific order."""
        return store.get_order_detail_by_order_id(order_id)

    @tool
    def get_vouchers_by_customer_id(customer_id: str, only_active: bool = False) -> dict:
        """Get vouchers belonging to a customer by their customer ID (e.g. 'C001').
        Set only_active=True to return only vouchers that are currently usable.
        Use this when the user asks about coupons, discounts, or vouchers."""
        return store.get_vouchers_by_customer_id(customer_id, only_active=only_active)

    return [
        get_customer_by_id,
        get_orders_by_customer_id,
        get_order_detail_by_order_id,
        get_vouchers_by_customer_id,
    ]
