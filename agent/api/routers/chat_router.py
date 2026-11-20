import uuid
from fastapi import APIRouter, Depends

from agent.api.routers.dependencies import get_dialogue_service
from agent.domain.schemas import ChatRequest, ChatResponse, ChatBotMessage, ChatObject, HistoryResponse, HistoryMessage
from agent.domain.messages import ProcessResult, UserMessage, MessageType, FocusedObject
from agent.service.dialogue_service import DialogueService

router = APIRouter()

# 定义路由接口

@router.post("/api/chat")
async def chat_endpoint(
        chat_request: ChatRequest,
        dialogue_service: DialogueService = Depends(get_dialogue_service)
) -> ChatResponse:
    # 1. 处理输入接口模型
    user_message = _build_user_message(chat_request)
    # 2. 业务处理
    process_result: ProcessResult = await dialogue_service.process_message(user_message)
    # 3. 处理输出接口模型
    return _build_chat_response(process_result)

def _build_user_message(chat_request: ChatRequest) -> UserMessage:
    """
    将请求数据模型转换为领域数据模型 供业务使用
    :param chat_request:
    :return:
    """
    return UserMessage(
        sender_id=chat_request.sender_id,
        message_id=chat_request.message_id if chat_request.message_id else str(uuid.uuid4()),
        type=MessageType.TEXT if chat_request.text else MessageType.OBJECT,
        text=chat_request.text,
        object=FocusedObject(
            id=chat_request.object.id,
            type=chat_request.object.type,
            title=chat_request.object.title,
            attributes=chat_request.object.attributes,
        ) if chat_request.object else None
    )

def _build_chat_response(process_result: ProcessResult) -> ChatResponse:
    return ChatResponse(
        sender_id=process_result.sender_id,
        message_id=process_result.message_id,
        messages=[
            ChatBotMessage(
                text=bot_msg.text,
                object=ChatObject(
                    type=bot_msg.object.type,
                    id=bot_msg.object.id,
                    title=bot_msg.object.title,
                    attributes=bot_msg.object.attributes
                ) if bot_msg.object else None
            )
            for bot_msg in process_result.messages
        ]
    )

@router.get('/api/chat/history')
async def history(sender_id: str) -> HistoryResponse:
    return HistoryResponse(
        sender_id=sender_id,
        messages=[
            HistoryMessage(role='user', text='你好'),
            HistoryMessage(role='bot', text='我不好'),
        ],
    )


if __name__ == '__main__':
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agent.domain.messages import BotMessage

    # 替换业务服务，仅测试路由和模型转换，不访问数据库或大模型。
    service = AsyncMock(spec=DialogueService)
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_dialogue_service] = lambda: service

    with TestClient(test_app) as client:
        for payload in [
            {'sender_id': 'test-user', 'message_id': 'test-message', 'text': '你好'},
            {'sender_id': 'test-user', 'object': {
                'id': 'order-1', 'type': 'order', 'title': '测试订单',
                'attributes': {'status': '待发货'},
            }},
        ]:
            focused_object = FocusedObject(**payload['object']) if 'object' in payload else None
            service.process_message.return_value = ProcessResult(
                sender_id=payload['sender_id'], message_id='reply-1',
                messages=[BotMessage(text='收到', object=focused_object)],
            )
            response = client.post('/api/chat', json=payload)
            assert response.status_code == 200, response.text
            assert response.json() == {
                'sender_id': 'test-user', 'message_id': 'reply-1',
                'messages': [{'text': '收到', 'object': payload.get('object')}],
            }
            service.process_message.assert_awaited_once()
            message = service.process_message.call_args.args[0]
            assert message.sender_id == payload['sender_id']
            assert message.text == payload.get('text')
            assert message.object == focused_object
            assert message.type == (MessageType.TEXT if 'text' in payload else MessageType.OBJECT)
            if 'message_id' in payload:
                assert message.message_id == payload['message_id']
            else:
                assert str(uuid.UUID(message.message_id)) == message.message_id
            service.process_message.reset_mock()

    print('测试通过：文本消息、对象消息、消息 ID 和响应转换')
