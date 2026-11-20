import json
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from agent.domain.state import DialogueState
from agent.utils.llm import llm
# from agent.knowledge.intents import KnowledgeIntent
from agent.plan.models import TurnPlan
from agent.prompts.history_builder import HistoryBuilder
from agent.prompts.loader import load_prompt
from agent.task.flow.flows import FlowsList
from knowledge.intents import KnowledgeIntent


class TurnPlanner:
    """
    意图分析器
    作用：根据自然语言 调用LLM 分析轨道类型
    """

    async def predict(self, dialogue_state, flow_list, intents: Dict[str, KnowledgeIntent]) -> TurnPlan:

        # 1. 构建提示词模板中的占位的内容（七部分）
        inputs_prompt = self._build_prompt_inputs(dialogue_state, flow_list, intents)
        print(inputs_prompt)


        # 2. 调用模确定用户意图，生成TurnPlan对象
        turn_plan: TurnPlan = await self._predict_from_inputs(inputs_prompt)

        return turn_plan

    def _build_prompt_inputs(self, dialogue_state: DialogueState, flow_list,  intents: Dict[str, KnowledgeIntent]):

        # 1. 用户问题
        user_msg = HistoryBuilder.render_user_message(dialogue_state.pending_turn.user_message)
        # 2. 对话历史
        current_conversation = HistoryBuilder.build(dialogue_state.current_session().turns[-10:])
        # 3. 当前活跃任务
        active_task_json = json.dumps(
            dialogue_state.active_task.model_dump(mode="json"), ensure_ascii=False
        ) if dialogue_state.active_task is not None else None

        # 4. 挂起任务列表
        interrupted_tasks_json = json.dumps(
            [paused_task.model_dump(mode="json") for paused_task in dialogue_state.paused_tasks],
            ensure_ascii= False
        )

        # 5. 聚焦对象
        focused_object_json = json.dumps(
            dialogue_state.focused_object.model_dump(mode="json"), ensure_ascii=False
        ) if dialogue_state.focused_object is not None else None

        # 6. 流程清单
        # available_flows_json = json.dumps(
        #     [flow.model_dump(mode="json").items() for flow in flow_list.flows], ensure_ascii=False
        # )

        flows_dict = {
            "flows":[{k: v for k, v in flow.model_dump(mode="json").items() if k != "steps"} for flow in flow_list.flows]
        }
        available_flows_json = json.dumps(flows_dict, ensure_ascii= False)

        # 7. 知识意图清单
        knowledge_intents_json = json.dumps(
            [{"id": intent.id, "description": intent.description} for intent in intents.values()], ensure_ascii=False)

        return {
            "user_message": user_msg,
            "current_conversation": current_conversation,
            "active_task_json": active_task_json,
            "interrupted_tasks_json": interrupted_tasks_json,
            "focused_object_json": focused_object_json,
            "available_flows_json": available_flows_json,
            "knowledge_intents_json": knowledge_intents_json
        }

    async def _predict_from_inputs(self, inputs_prompt: dict[str, Any])->TurnPlan:

        prompt_template_str = load_prompt("turn_plan")
        prompt_template = PromptTemplate.from_template(template=prompt_template_str, template_format="jinja2")

        # 构建langcian的链式调用
        chain = prompt_template | llm | JsonOutputParser()
        llm_response_dict: Dict[str, Any] = await chain.ainvoke(inputs_prompt)

        return TurnPlan.from_dict(llm_response_dict)


if __name__ == "__main__":
    # 使用退款消息调用真实模型，打印计划并检查是否启动退款流程。
    import asyncio
    from pathlib import Path

    from agent.domain.messages import MessageType, UserMessage
    from agent.task.command.models import StartFlowCommand
    from agent.task.flow.loader import FlowLoader

    async def main():
        root = Path(__file__).resolve().parents[2]
        flow_list = FlowLoader().load(root / "flow_config" / "user_flows.yml")

        state = DialogueState(sender_id="test_user")
        state.start_session()
        state.begin_turn(UserMessage(
            sender_id="test_user",
            message_id="test_message",
            type=MessageType.TEXT,
            text="我想申请退款",
        ))

        plan = await TurnPlanner().predict(state, flow_list)
        print(plan.model_dump_json(indent=2))

        assert plan.task is not None, "预期识别为业务任务"
        assert any(
            isinstance(command, StartFlowCommand)
            and command.flow == "refund_request"
            for command in plan.task.commands
        ), "预期启动退款流程"

    asyncio.run(main())
