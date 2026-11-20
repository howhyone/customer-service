import time

from agent.domain.messages import UserMessage, ProcessResult, MessageType, BotMessage
from agent.domain.state import DialogueState
from agent.plan.models import TurnPlan, ClarifyReason
from agent.plan.turn_planner import TurnPlanner
from agent.plan.turn_validator import TurnPlanValidator
from agent.task.handler import TaskHandler
from agent.knowledge.handler import KnowledgeHandler
from clarify.responder import ClarifyResponder
from task.command.models import SetSlotsCommand, Command
from task.flow.flows import FlowsList
from task.flow.steps import CollectFlowStep


class DialogueEngine:

    def __init__(
            self,
            turn_planner: TurnPlanner,
            task_handler: TaskHandler,
            knowledge_handler: KnowledgeHandler,
            # chitchat_handler: ChitchatHandler,
            clarify_responder: ClarifyResponder,
            turn_plan_validator: TurnPlanValidator
    ) -> None:
        self.turn_planner = turn_planner
        self.task_handler = task_handler
        self.knowledge_handler = knowledge_handler
        # self.chitchat_handler = chitchat_handler
        self.clarify_responder = clarify_responder
        self.turn_plan_validator = turn_plan_validator

    async def process_message(self, dialogue_state: DialogueState, user_message: UserMessage) -> ProcessResult:

        # 根据user_message的内容，调用LLM进行路由，判断需要执行哪条轨道（业务流程、RAG检索、闲聊），然后执行某一条轨道
        # 1 准备会话（如果没有则创建、如果有则获取）
        self._prepare_session(dialogue_state)

        # 2 开启本轮会话（创建Turn对象）
        self._begin_turn(dialogue_state, user_message)

        # 3 判断消息的类型
        if user_message.type is MessageType.TEXT:
            # 3.1 if 处理文本消息，获取AI客服的回复
            messages = await self._handle_text_message(dialogue_state)
        else:
            # 3.2 else 处理对象消息
            dialogue_state.set_focused_object(user_message.object)
            messages = await self._handle_object_message(user_message, dialogue_state)

        # 4 提交本轮记录
        # 4.1 将AI客服的回复写入到pending_turn
        dialogue_state.pending_turn.bot_messages.extend(messages)
        # 4.2 将本轮的pending_turn提交到session（将Turn存储在Session的Turn列表中）
        dialogue_state.commit_pending_turn()

        # 5 组装结果并返回
        return ProcessResult(
            sender_id=dialogue_state.sender_id,
            message_id=user_message.message_id,
            messages=messages
        )
    def _prepare_session(self, dialogue_state: DialogueState) -> None:
        """
        获取或初始化session会话
        :param dialogue_state:
        :return:
        """
        # 1. 获取当前会话
        current_session = dialogue_state.current_session()

        # 2. 如果没有会话，则创建一个会话
        if current_session is None:
            dialogue_state.start_session()
            return

        # 3. 判断会话是否过期(1小时)
        now = time.time()
        if now - current_session.last_activity_at > 60 * 60 * 1:
            # 关闭会话
            dialogue_state.close_current_session()
            # 重置dialogue_state的运行状态
            dialogue_state.reset_runtime_state_for_new_session()
            # 创建新会话
            dialogue_state.start_session()
        else:
            # 更新会话（会话续期）
            current_session.last_activity_at = now

    def _begin_turn(self, dialogue_state, user_message):
        """
        开始一个Turn
        :param dialogue_state:
        :param user_message:
        :return:
        """
        dialogue_state.begin_turn(user_message)

    async def _handle_text_message(self, dialogue_state: DialogueState):

        # 使用TurnPlanner理解用户意图（LLM）,并生成本轮对话的计划
        turn_plan: TurnPlan = await self.turn_planner.predict(dialogue_state, self.task_handler.flows)

        # 使用TurnPlanValidator处理是否存在模型理解幻觉 TODO

        # 根据意图识别的结果执行对应的业务逻辑或者对无效意图做澄清处理
        if turn_plan.task is not None:
            return await self.task_handler.handle(
                commands = turn_plan.task.commands,
                state = dialogue_state
            )
        elif turn_plan.knowledge is not None:
            # 2. 防幻觉校验
            validated = self.turn_plan_validator.validate(turn_plan, self.task_handler.flows, dialogue_state,
                                                          self.knowledge_handler.knowledge_intends)
            if not validated.valid and validated.reason is not None:
                return await self.clarify_responder.respond(dialogue_state, validated.reason)
            pass
        else:
            # ChitchatHandler TODO
            pass

    async def _handle_object_message(
            self,
            message: UserMessage,
            state: DialogueState,
    ) -> list[BotMessage]:
        """
        处理对象消息
        :param message:
        :param state:
        :return:
        """

        # 1. 将对象解析成command(SetSlotsCommand)
        commands = self._resolve_object_commands(
            message=message,
            state=state,
            flows=self.task_handler.flows,
        )

        # 2. 判断command是否已经存在(流程的步骤刚好需要你点击的卡片)
        if commands:
            # 场景 A:对象能填当前槽 → 转成命令,推进流程
            return await self.task_handler.handle(commands=commands, state=state)

        # 有流程，没有匹配到 slot
        # 如果用户正在一个流程中,不打断,让流程继续。
        if state.active_task is not None:
            # 场景 C:在流程里但槽不匹配 → 不打断,空命令让流程继续
            return await self.task_handler.handle(commands=[], state=state)

        # 没有流程：
        # 场景 B:没有流程 → 澄清
        return await self.clarify_responder.respond(
            state=state,
            reason=ClarifyReason.OBJECT_REQUIRES_INTENT,
        )

    def _resolve_object_commands(
            self,
            message: UserMessage,
            state: DialogueState,
            flows: FlowsList
    ) -> list[Command]:
        """
        将对象消息转换成command
        :param message:
        :param state:
        :param flows:
        :return:
        """

        # 1. 获取聚焦对象
        focused_object = message.object
        if focused_object is None:
            return []

        # 2. 获取聚焦对象的类型
        object_type = focused_object.type

        # 3.1. 判读对象的类型 order
        if object_type == "order":
            if self._flow_has_unfilled_collect_slot(state, flows, "order_number"):
                return [SetSlotsCommand(command="set_slots", slots={"order_number": focused_object.id})]

            return []

        # 3.2. 判读对象的类型 product
        if object_type == "product":
            if self._flow_has_unfilled_collect_slot(state, flows, "product_id"):
                return [SetSlotsCommand(command="set_slots", slots={"product_id": focused_object.id})]

            return []

        return []

    def _flow_has_unfilled_collect_slot(
            self,
            state: DialogueState,
            flows: FlowsList,
            slot_name: str
    ) -> bool:
        """
        判断流程中是否有未填的收集槽位
        :param state:
        :param flows:
        :param slot_name:
        :return:
        """

        # 1. 获取活跃任务
        active_task = state.active_task

        # 2. 是否存在活跃任务
        if active_task is None:
            # 不存在
            return False

        # 3. 根据活跃任务的id获取流程
        flow_id = active_task.flow_id
        flow = flows.get_flow_by_id(flow_id)
        if flow is None:
            # 不存在
            return False

        # 4. 判断该流程中的当前槽位是否已经填过
        if active_task.slots.get(slot_name):
            return False

        # 5. 遍历流程的步骤,查找是否存在收集这个槽位的步骤
        for step in flow.steps:
            if isinstance(step, CollectFlowStep) and step.slot_name == slot_name:
                return True

        return False