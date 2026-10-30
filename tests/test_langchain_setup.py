from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import Mock, patch


def deny_network(event, arguments):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.bind"}:
        raise AssertionError("Outbound application network disabled")


sys.addaudithook(deny_network)


class LangChainSetupTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.dict(os.environ, {
            "LLM_API_KEY": "offline-test-key",
            "YAHOO_FINANCE_SEARCH_API": "https://market-data.invalid/search?q=",
            "YAHOO_FINANCE_QUERY_API": "https://market-data.invalid/chart/",
            "PYTHON_DOTENV_DISABLED": "1",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
        }, clear=True))
        self.dotenv = self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch("logging.basicConfig"))
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        self.model = FakeListChatModel(responses=[self.action("Final Answer", "20.00%")])
        self.model_factory = self.enterContext(patch("langchain_openai.ChatOpenAI", return_value=self.model))
        self.market_request = self.enterContext(patch("requests.get", side_effect=AssertionError("Unexpected market request")))
        source = Path(__file__).resolve().parents[1] / "2_backtester_agent_langchain.py"
        specification = importlib.util.spec_from_file_location("langchain_under_test", source)
        self.module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = self.module
        self.addCleanup(sys.modules.pop, specification.name, None)
        specification.loader.exec_module(self.module)
        self.module.logger = Mock()

    @staticmethod
    def action(name, arguments):
        return "```json\n" + json.dumps({"action": name, "action_input": arguments}) + "\n```"

    def test_current_packages_construct_legacy_executor(self):
        self.assertEqual(self.module.agent.max_iterations, 5)
        self.assertEqual([tool.name for tool in self.module.tools], [
            "get_symbol", "convert_date_to_timestamp", "get_ticker_historical_prices", "get_return_percentage",
        ])
        for tool in self.module.tools:
            self.assertEqual(len(tool.args_schema.model_fields), 1)
        self.market_request.assert_not_called()

    def test_fake_model_response(self):
        self.assertEqual(self.module.process_query("Synthetic query"), "20.00%")
        self.market_request.assert_not_called()

    def test_synthetic_tool_workflow(self):
        self.model.responses = [
            self.action("get_symbol", {"company_name": "Example"}),
            self.action("convert_date_to_timestamp", {"date_str": "2024-01-01,2024-01-02"}),
            self.action("get_ticker_historical_prices", {"input_str": "FAKE,1704067200,1704240000"}),
            self.action("get_return_percentage", {"prices_str": "100,120"}),
            self.action("Final Answer", "20.00%"),
        ]
        responses = []
        for data in [
            {"quotes": [{"symbol": "FAKE"}]},
            {"chart": {"result": [{"indicators": {"quote": [{"close": [100, 120]}]}}]}},
        ]:
            response = Mock()
            response.json.return_value = data
            responses.append(response)
        self.market_request.side_effect = responses
        self.assertEqual(self.module.process_query("Synthetic query"), "20.00%")
        self.assertEqual(self.market_request.call_count, 2)
        self.assertIn("Example", self.market_request.call_args_list[0].args[0])
        self.assertIn("FAKE", self.market_request.call_args_list[1].args[0])
        for response in responses:
            response.raise_for_status.assert_called_once_with()

    def test_query_errors_are_reported(self):
        executor = Mock()
        executor.invoke.side_effect = RuntimeError("Synthetic failure")
        with patch.object(self.module, "agent", executor):
            self.assertEqual(self.module.process_query("Synthetic query"), "Error processing query: Synthetic failure")

    def test_network_is_denied_and_configuration_is_synthetic(self):
        with self.assertRaisesRegex(AssertionError, "Outbound application network disabled"):
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        self.dotenv.assert_called_once_with()
        self.assertEqual(self.model_factory.call_args.kwargs["api_key"], "offline-test-key")
        self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")