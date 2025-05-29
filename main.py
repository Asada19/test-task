import logging
import os
from datetime import datetime, timezone
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    """Состояние приложения."""

    messages: Annotated[list, add_messages]


class TimeTools:
    """Класс для работы с инструментами времени."""

    @staticmethod
    def get_current_time() -> dict:
        """
        Use this tool ONLY when the user explicitly asks for the current time,
        time-related information, or what time it is.

        Examples of when to use:
        - "What time is it?"
        - "Tell me the current time"
        - "What's the time now?"

        DO NOT use for general questions about capabilities or other topics.
        """
        try:
            local_time = datetime.now().isoformat()
            utc_time = datetime.now(timezone.utc).isoformat()
            logger.debug(f"Получено время: local={local_time}, utc={utc_time}")
            return {"local": local_time, "utc": utc_time}
        except Exception as e:
            logger.error(f"Ошибка получения времени: {str(e)}")
            return {"error": f"Ошибка получения времени: {str(e)}"}


class ChatBotGraph:
    """Класс для управления графом чат-бота."""

    def __init__(self, api_key: str):
        """Инициализация графа чат-бота."""
        self.api_key = api_key
        self.time_tools = TimeTools()
        self.llm = self._init_llm()
        self.llm_with_tools = self._bind_tools()
        self.graph = self._build_graph()
        logger.info("Граф чат-бота успешно инициализирован")

    def _init_llm(self):
        """Инициализация модели Claude."""
        return init_chat_model(
            model="claude-3-5-sonnet-20241022",
            model_provider="anthropic",
            temperature=0.7,
            api_key=self.api_key,
        )

    def _bind_tools(self):
        """Привязка инструментов к модели."""
        return self.llm.bind_tools([self.time_tools.get_current_time])

    def _build_graph(self):
        """Построение графа состояний."""
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", self._tool_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot", self._route_tool_calls, {"tools": "tools", END: END}
        )
        return graph_builder.compile()

    def _chatbot_node(self, state: State) -> dict:
        """Узел чат-бота для обработки сообщений."""
        try:
            llm_response = self.llm_with_tools.invoke(state["messages"])
            return {"messages": [llm_response]}
        except Exception as e:
            logger.error(f"Ошибка в узле чат-бота: {str(e)}")
            error_message = HumanMessage(content=f"Ошибка: {str(e)}")
            return {"messages": [error_message]}

    def _tool_node(self, state: State) -> dict:
        """Узел инструментов для выполнения вызовов инструментов."""
        ai_message = state["messages"][-1]
        tool_calls = getattr(ai_message, "tool_calls", [])
        tool_results = []

        for tool_call in tool_calls:
            if tool_call["name"] == "get_current_time":
                logger.debug(f"Выполняется вызов инструмента: {tool_call['name']}")
                result = self.time_tools.get_current_time()
                tool_results.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

        return {"messages": tool_results}

    def _route_tool_calls(self, state: State) -> str:
        """Маршрутизация вызовов инструментов."""
        ai_message = state["messages"][-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    def invoke(self, messages: list) -> dict:
        """Вызов графа с сообщениями."""
        return self.graph.invoke({"messages": messages})


class ChatBot:
    """Основной класс чат-бота."""

    def __init__(self):
        """Инициализация чат-бота."""
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.system_prompt = SystemMessage(
            content="You are a helpful AI assistant that responds in Russian language. "
            "You can answer questions and use available tools to provide accurate information. "
            "IMPORTANT: Only use tools when the user specifically requests that functionality. "
            "For example, only use the time tool when the user asks about the current time. "
            "Do NOT use tools for general questions about your capabilities or other topics. "
            "When using tools, provide the final answer to the user without sharing your thought process or reasoning. "
            "Be concise, helpful, and maintain a friendly conversational tone. "
            "Always respond in Russian, even if the user asks in another language."
        )
        self.graph = None

    def _validate_setup(self) -> bool:
        """Проверка настройки окружения."""
        if not self.api_key:
            logger.error("Не найден ANTHROPIC_API_KEY в переменных окружения")
            print("Ошибка: Не найден ANTHROPIC_API_KEY в переменных окружения")
            return False
        return True

    def _init_graph(self):
        """Инициализация графа."""
        try:
            self.graph = ChatBotGraph(self.api_key)
        except Exception as e:
            logger.error(f"Ошибка инициализации графа: {str(e)}")
            print(f"Ошибка инициализации: {str(e)}")
            return False
        return True

    def _get_user_input(self) -> str:
        """Получение ввода от пользователя."""
        return input("Вы: ").strip()

    def _is_exit_command(self, user_input: str) -> bool:
        """Проверка команды выхода."""
        return user_input.lower() in ["quit", "exit", "bye", "выход"]

    def _process_message(self, user_input: str) -> str:
        """Обработка сообщения пользователя."""
        try:
            logger.debug(f"Обработка сообщения пользователя: {user_input}")
            result = self.graph.invoke(
                [self.system_prompt, HumanMessage(content=user_input)]
            )
            response = result["messages"][-1].content
            logger.debug(f"Получен ответ от ассистента: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {str(e)}")
            return f"Произошла ошибка: {str(e)}"

    def _print_welcome(self):
        """Вывод приветственного сообщения."""
        welcome_msg = "Введите ваш вопрос (для выхода: quit/exit/bye)"
        logger.info("Чат-бот запущен")
        print(welcome_msg)
        print("-" * 50)

    def run(self):
        """Запуск чат-бота."""
        if not self._validate_setup():
            return

        if not self._init_graph():
            return

        self._print_welcome()

        while True:
            try:
                user_input = self._get_user_input()

                if not user_input:
                    continue

                if self._is_exit_command(user_input):
                    logger.info("Пользователь завершил сессию")
                    print("До свидания!")
                    break

                response = self._process_message(user_input)
                print(f"Ассистент: {response}")

            except KeyboardInterrupt:
                logger.info("Сессия прервана пользователем")
                print("\nДо свидания!")
                break
            except Exception as e:
                logger.error(f"Неожиданная ошибка в основном цикле: {str(e)}")
                print(f"Произошла ошибка: {str(e)}")


def main():
    chatbot = ChatBot()
    chatbot.run()


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    try:
        _chatbot_graph = ChatBotGraph(ANTHROPIC_API_KEY)
        graph = _chatbot_graph.graph
    except Exception as e:
        logger.warning(f"Не удалось создать граф для langgraph dev: {e}")
        graph = None
else:
    logger.warning("ANTHROPIC_API_KEY не найден, граф не создан")
    graph = None

if __name__ == "__main__":
    main()
