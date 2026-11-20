from pathlib import Path

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agent.service.dialogue_service import DialogueService
from agent.repository.dialogue_state_repository import DialogueStateRepository
from agent.engine.dialogue_engine import DialogueEngine
from agent.utils import database
from clarify.responder import ClarifyResponder
from knowledge.handler import KnowledgeHandler
from knowledge.intents import KNOWLEDGE_INTENTS
from plan.turn_planner import TurnPlanner
from plan.turn_validator import TurnPlanValidator
from task.command.processor import CommandProcessor
from task.flow.loader import FlowLoader
from task.handler import TaskHandler


async def get_session():
    """
    实际流程：
        进入 async with → 打开 session
        yield session → 把 session 交给repository函数使用
        路由函数执行完毕 → 回到 yield 之后
        退出 async with → 自动关闭 session ✅
    :return:
    """
    async with database.async_session() as session:  # 异步方式获取session  获取session要网络传输（耗时的）
        yield session

async def get_dialogue_state_repository(session: AsyncSession = Depends(get_session)):
    """
    创建 DialogueStateRepository 实例

    依赖链执行顺序：
       1. get_session() 打开 session 并 yield
       2. 这里拿到 session，创建 Repository
       3. Repository 被Service使用，Service被路由函数使用
       4. 路由函数返回后，get_session() 继续执行，自动关闭 session
    :param session:
    :return:
    """
    return DialogueStateRepository(session=session)

async def get_engine():
    base_path = Path(__file__).parents[3]
    user_flow_path = base_path / "flow_config" / "user_flows.yml"
    system_flow_path = base_path / "flow_config" / "system_flows.yml"

    loader = FlowLoader()
    flow_list = loader.load_many([user_flow_path, system_flow_path])

    return DialogueEngine(turn_planner = TurnPlanner(),
                          task_handler = TaskHandler(flows=flow_list, command_processor=CommandProcessor()),
                          knowledge_handler=KnowledgeHandler(knowledge_intends=KNOWLEDGE_INTENTS),
                          turn_plan_validator=TurnPlanValidator(),
                          clarify_responder=ClarifyResponder())

async def get_dialogue_service(
        dialogue_state_repository: DialogueStateRepository = Depends(get_dialogue_state_repository),
        dialogue_engine: DialogueEngine = Depends(get_engine)
) -> DialogueService:
    return DialogueService(dialogue_state_repository=dialogue_state_repository, dialogue_engine=dialogue_engine)
