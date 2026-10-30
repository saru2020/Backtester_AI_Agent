import ast
import builtins
import io
import os
import socket
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import CodeType, FunctionType
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, quote, urlsplit


def deny_network(event, arguments):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.bind"}:
        raise AssertionError("Outbound application network disabled")


sys.addaudithook(deny_network)


class MarketRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import requests

        cls.requests = requests
        cls.helpers = {}
        for source in sorted(Path(__file__).resolve().parents[1].glob("[012]_*.py")):
            tree = ast.parse(source.read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
                "get_symbol", "get_ticker_historical_prices",
            }]
            namespace = {"__builtins__": vars(builtins), "requests": requests, "os": os, "logger": Mock(), "quote": quote}
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
        self.enterContext(patch.dict(os.environ, {
            "YAHOO_FINANCE_SEARCH_API": "https://market-data.invalid/search?q=",
            "YAHOO_FINANCE_QUERY_API": "https://market-data.invalid/chart/",
        }, clear=True))

    def invoke(self, source, function):
        helper = self.helpers[source][function]
        if function == "get_symbol":
            return helper("Example & Sons")
        if source.startswith("2_"):
            return helper("^INDEX/A&B,1,2")
        return helper("^INDEX/A&B", 1, 2)

    def assert_failed_request(self, source, function):
        if source.startswith("0_") and function == "get_symbol":
            with self.assertRaises(self.requests.RequestException):
                self.invoke(source, function)
        else:
            result = self.invoke(source, function)
            if source.startswith("0_"):
                self.assertEqual(result, (None, None))
            elif source.startswith("1_"):
                self.assertIsNone(result)
            else:
                self.assertIn("Error", result)

    def test_network_is_denied(self):
        with self.assertRaisesRegex(AssertionError, "Outbound application network disabled"):
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)

    def test_search_encodes_company_names_and_has_timeouts(self):
        self.assertEqual(len(self.helpers), 3)
        for source in self.helpers:
            with self.subTest(source=source):
                response = Mock()
                response.json.return_value = {"quotes": [{"symbol": "FAKE"}]}
                with patch.object(self.requests, "get", return_value=response) as request:
                    result = self.invoke(source, "get_symbol")
                self.assertEqual(result, "FAKE" if source.startswith("0_") else {"symbol": "FAKE"})
                self.assertEqual(parse_qs(urlsplit(request.call_args.args[0]).query), {
                    "q": ["Example & Sons"], "lang": ["en-US"], "region": ["US"], "quotesCount": ["6"],
                })
                self.assertEqual(request.call_args.kwargs["timeout"], (5, 30))
                response.raise_for_status.assert_called_once_with()

    def test_history_encodes_symbols_and_has_timeouts(self):
        for source in self.helpers:
            with self.subTest(source=source):
                response = Mock()
                response.json.return_value = {"chart": {"result": [{"indicators": {"quote": [{"close": [100, 120]}]}}]}}
                with patch.object(self.requests, "get", return_value=response) as request:
                    result = self.invoke(source, "get_ticker_historical_prices")
                if source.startswith("0_"):
                    self.assertEqual(result, (100, 120))
                elif source.startswith("1_"):
                    self.assertEqual(result, {"start_date_timestamp_price": 100, "end_date_timestamp_price": 120})
                else:
                    self.assertEqual(result, "100,120")
                url = urlsplit(request.call_args.args[0])
                self.assertEqual(url.path, "/chart/%5EINDEX%2FA%26B")
                self.assertEqual(parse_qs(url.query), {"period1": ["1"], "period2": ["2"], "interval": ["1d"]})
                self.assertEqual(request.call_args.kwargs["timeout"], (5, 30))
                response.raise_for_status.assert_called_once_with()

    def test_http_errors_are_checked_before_reading_json(self):
        for source in self.helpers:
            for function in ["get_symbol", "get_ticker_historical_prices"]:
                with self.subTest(source=source, function=function):
                    response = Mock()
                    response.raise_for_status.side_effect = self.requests.HTTPError("Synthetic failure")
                    with patch.object(self.requests, "get", return_value=response):
                        self.assert_failed_request(source, function)
                    response.json.assert_not_called()

    def test_timeout_errors_keep_existing_failure_contracts(self):
        for source in self.helpers:
            for function in ["get_symbol", "get_ticker_historical_prices"]:
                with (
                    self.subTest(source=source, function=function),
                    patch.object(self.requests, "get", side_effect=self.requests.Timeout("Synthetic timeout")),
                ):
                    self.assert_failed_request(source, function)

    def test_empty_search_and_history_results(self):
        for source in self.helpers:
            for function in ["get_symbol", "get_ticker_historical_prices"]:
                with self.subTest(source=source, function=function):
                    response = Mock()
                    response.json.return_value = {"quotes": [], "chart": {"result": []}}
                    with patch.object(self.requests, "get", return_value=response):
                        result = self.invoke(source, function)
                    if source.startswith("2_"):
                        self.assertTrue(result.startswith("No "))
                    elif source.startswith("0_") and function == "get_ticker_historical_prices":
                        self.assertEqual(result, (None, None))
                    else:
                        self.assertIsNone(result)
