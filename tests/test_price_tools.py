import ast
import builtins
import datetime
import io
import math
import os
import socket
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import CodeType, FunctionType
from unittest.mock import Mock, patch


def deny_network(event, arguments):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.bind"}:
        raise AssertionError("Outbound application network disabled")


sys.addaudithook(deny_network)


class ReturnPercentageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = {}
        for source in sorted(Path(__file__).resolve().parents[1].glob("[012]_*.py")):
            tree = ast.parse(source.read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
                "get_return_percentage", "convert_date_to_timestamp",
            }]
            namespace = {"__builtins__": vars(builtins), "datetime": datetime, "math": math, "logger": Mock()}
            compiled = compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec")
            for function in functions:
                if function.decorator_list or function.args.defaults or function.args.kw_defaults:
                    raise ValueError("Helper loading requires undecorated functions without defaults")
                code = next(constant for constant in compiled.co_consts
                            if isinstance(constant, CodeType) and constant.co_name == function.name)
                helper = FunctionType(code, namespace)
                helper.__annotations__ = {
                    argument.arg: getattr(builtins, argument.annotation.id)
                    for argument in function.args.args if argument.annotation is not None
                }
                if function.returns is not None:
                    helper.__annotations__["return"] = getattr(builtins, function.returns.id)
                namespace[function.name] = helper
            cls.helpers[source.name] = namespace

    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))

    def calculate(self, source, start_price, end_price):
        calculate = self.helpers[source]["get_return_percentage"]
        if source.startswith("2_"):
            return calculate(f"{start_price},{end_price}")
        return calculate(start_price, end_price)

    def assert_invalid_return(self, start_price, end_price, message):
        for source in self.helpers:
            with self.subTest(source=source, start_price=start_price, end_price=end_price):
                if source.startswith("0_"):
                    with self.assertRaisesRegex(ValueError, message):
                        self.calculate(source, start_price, end_price)
                elif source.startswith("1_"):
                    self.assertIsNone(self.calculate(source, start_price, end_price))
                else:
                    self.assertIn(message, self.calculate(source, start_price, end_price))

    def test_loads_all_helpers_without_importing_entry_points(self):
        self.assertEqual(len(self.helpers), 3)
        for namespace in self.helpers.values():
            self.assertTrue(callable(namespace["get_return_percentage"]))
            self.assertTrue(callable(namespace["convert_date_to_timestamp"]))

    def test_network_is_denied(self):
        with self.assertRaisesRegex(AssertionError, "Outbound application network disabled"):
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)

    def test_valid_returns(self):
        for source in self.helpers:
            for start_price, end_price, expected in [(100, 120, 20), (100, 75, -25), (100, 100, 0), (100, 0, -100)]:
                with self.subTest(source=source, start_price=start_price, end_price=end_price):
                    result = self.calculate(source, start_price, end_price)
                    if source.startswith("1_"):
                        result = result["return_percentage"]
                    elif source.startswith("2_"):
                        self.assertEqual(result, f"{expected:.2f}%")
                        result = float(result.rstrip("%"))
                    self.assertAlmostEqual(result, expected)

    def test_rejects_invalid_starting_prices(self):
        for price in [0, -1, float("nan"), float("inf"), -float("inf")]:
            self.assert_invalid_return(price, 100, "Starting price")

    def test_rejects_invalid_ending_prices(self):
        for price in [-1, float("nan"), float("inf"), -float("inf")]:
            self.assert_invalid_return(100, price, "Ending price")

    def test_rejects_overflow(self):
        self.assert_invalid_return(1e-300, 1e300, "Return percentage must be finite")

    def test_dates_are_utc_and_include_the_end_day(self):
        previous_timezone = os.environ.get("TZ")
        try:
            for timezone in ["UTC", "America/New_York", "Asia/Kolkata"]:
                with patch.dict(os.environ, {"TZ": timezone}):
                    time.tzset()
                    for source, namespace in self.helpers.items():
                        with self.subTest(source=source, timezone=timezone):
                            convert = namespace["convert_date_to_timestamp"]
                            if source.startswith("2_"):
                                actual = tuple(map(int, convert("2024-03-10,2024-03-10").split(",")))
                            else:
                                actual = convert("2024-03-10", "2024-03-10")
                                if source.startswith("1_"):
                                    actual = (actual["start_date_timestamp"], actual["end_date_timestamp"])
                            self.assertEqual(actual, (1710028800, 1710115200))
        finally:
            if previous_timezone is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous_timezone
            time.tzset()

    def test_rejects_reversed_and_invalid_dates(self):
        for start_date, end_date in [("2024-02-02", "2024-02-01"), ("2024-02-30", "2024-03-01")]:
            for source, namespace in self.helpers.items():
                with self.subTest(source=source, start_date=start_date, end_date=end_date):
                    convert = namespace["convert_date_to_timestamp"]
                    if source.startswith("0_"):
                        with self.assertRaises(ValueError):
                            convert(start_date, end_date)
                    elif source.startswith("1_"):
                        self.assertIsNone(convert(start_date, end_date))
                    else:
                        self.assertTrue(convert(f"{start_date},{end_date}").startswith("Error converting dates:"))
