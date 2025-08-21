from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from PIL import Image
import io

# 导入统一日志系统和分析模块日志装饰器
from tradingagents.utils.logging_init import get_logger
from tradingagents.utils.tool_logging import log_analyst_module
logger = get_logger("analysts.image_analysis")

def create_image_analyst(llm, toolkit):
    @log_analyst_module("image_analysis")
    def image_analyst_node(state):
        # 获取上传的图片
        uploaded_image = state.get("uploaded_image")
        if not uploaded_image:
            return {
                "messages": ["未上传图片，请上传图片后再进行分析。"],
                "analysis_report": "未上传图片，无法进行分析。"
            }

        # 将上传的图片转换为PIL Image对象
        image = Image.open(io.BytesIO(uploaded_image))

        # 定义系统消息
        system_message = (
            """您是一位专业的图像分析师，负责分析上传的图片内容。

您的主要职责包括：
1. 识别图片中的主要对象和场景
2. 分析图片的色彩和构图特点
3. 评估图片的视觉效果和可能的情感表达
4. 提供图片的详细描述和分析报告

分析要点：
- 图片中的主要对象和场景描述
- 色彩和构图的特点
- 图片可能传达的情感或信息
- 图片的视觉效果评估

请撰写详细的中文分析报告，并在报告末尾附上Markdown表格总结关键发现。"""
        )

        # 定义提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位有用的AI助手，与其他助手协作。"
                    " 使用提供的工具来推进回答问题。"
                    " 如果您无法完全回答，没关系；具有不同工具的其他助手"
                    " 将从您停下的地方继续帮助。执行您能做的以取得进展。"
                    " 如果您或任何其他助手有最终分析结果：**分析完成**，"
                    " 请在您的回应前加上最终分析结果：**分析完成**，以便团队知道停止。"
                    " 您可以访问以下工具：{tool_names}。\n{system_message}。"
                    " 请用中文撰写所有分析内容。"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # 安全地获取工具名称，处理函数和工具对象
        tool_names = []
        for tool in toolkit:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            elif hasattr(tool, '__name__'):
                tool_names.append(tool.__name__)
            else:
                tool_names.append(str(tool))

        prompt = prompt.partial(tool_names=", ".join(tool_names))
        prompt = prompt.partial(system_message=system_message)

        # 将图片传递给LLM进行分析
        chain = prompt | llm.bind_tools(toolkit)
        result = chain.invoke({"messages": [{"role": "user", "content": "请分析上传的图片。"}]})

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "analysis_report": report,
        }

    return image_analyst_node