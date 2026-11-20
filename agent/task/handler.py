from fastapi.openapi.models import Components

from agent.domain.messages import BotMessage
from agent.domain.state import DialogueState
from agent.task.command.models import Command
from agent.task.flow.flows import FlowsList
from task.command.processor import CommandProcessor


class TaskHandler:

    def __init__(self,
                 flows: FlowsList,
                 command_processor: CommandProcessor):
        self.flows = flows
        self.command_processor = command_processor

    async def handle(self, commands: list[Command], state: DialogueState) -> list[BotMessage]:

        # TODO 推进流程的执行

        return [BotMessage(text="AI客服的回答：任务执行完成.....")]